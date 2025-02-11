from pathlib import Path
from random import random
from typing import Tuple, List, Type

import numpy as np
import pandas as pd
from pyod.models.base import BaseDetector
from pyod.models.lof import LOF
from pyod.models.lunar import LUNAR
from torch import Tensor

import main
from models.Generator import GeneratorSigmoidSTE, GeneratorUpperSoftmax, GeneratorSpectralNorm
from modules.od_module import VMMD_od

from text.Embedding.gpt2 import GPT2
from text.dataset.SimpleDataset import SimpleDataset
from text.dataset.ag_news import AGNews
from text.dataset.emotions import EmotionDataset
from text.outlier_detection.odm import OutlierDetectionModel
from text.outlier_detection.pyod_odm import EmbeddingBaseDetector
from text.visualizer.collective_visualizer import CollectiveVisualizer
from vmmd import VMMD


class VGAN_ODM(OutlierDetectionModel):

    def __init__(self, dataset, model, train_size, test_size, inlier_label=None, base_detector: Type[BaseDetector] = None, pre_embed=False, use_cached=False):
        self.space = "Embedding" if pre_embed else "Tokenized"
        self.model = model
        self.vgan = VMMD_od(penalty_weight=0.1, generator=GeneratorSigmoidSTE,
                            lr=1e-5)
        self.number_of_subspaces = 50
        self.base_detector: Type[BaseDetector] = base_detector
        if base_detector is None:
            self.base_detector = LUNAR

        self.detectors: List[BaseDetector] = []
        self.init_dataset = self.use_embedding if pre_embed else self.use_tokenized
        self.pre_embed = pre_embed
        super().__init__(dataset, model, train_size, test_size, inlier_label, use_cached=use_cached)

    def _get_detector(self) -> BaseDetector:
        if not self.pre_embed:
            return EmbeddingBaseDetector(self.model, lambda: self.base_detector)
        return self.base_detector()

    def train(self):
        self.init_dataset()
        train = self.x_train.to(self.device)


        epochs = int(10 ** 6.7 / len(train) + 200)
        self.vgan.epochs = epochs
        print(f"training vmmd for {epochs} epochs.")

        #with self.ui.display():
        for epoch in self.vgan.yield_fit(train, yield_epochs=200):
            #self.ui.update(f"Fitting VGAN, current epoch {epoch}")
            if epoch != 0:
                print(f"({epoch}, {self.vgan.train_history[self.vgan.generator_loss_key][-1]})")


        with self.ui.display("Generating subspaces"):
            subspaces = self.vgan.generate_subspaces(self.number_of_subspaces)
            unique_subspaces, proba = np.unique(
                np.array(subspaces.to('cpu')), axis=0, return_counts=True)
            self.subspaces = Tensor(unique_subspaces).to(self.device)
            self.proba = proba / proba.sum()

        # If one subspace has probabilities over 0.5, it decides the inlier label
        # To improve runtime we thus can remove the other subspaces
        #main_subspace_exits = (self.proba > 0.5).sum() > 0
        #if main_subspace_exits:
            #self.subspaces = self.subspaces[self.proba > 0.5]
            #self.proba = self.proba[self.proba > 0.5]
            #print("Main subspace found, removing other subspaces")

        # Project the dataset into each of the generated subspaces
        #projected_datasets = [self.project_dataset(train, subspace) for subspace in self.subspaces]
        # Fit a detector on each of the projected datasets. To avoid memory issues, we fit the detectors one by one and project the dataset one by one
        with self.ui.display():
            for subspace in self.subspaces:
                self.ui.update(f"Fitting detector {len(self.detectors)} / {len(self.subspaces)}")
                detector = self._get_detector()
                detector.fit(self.project_dataset(train, subspace).cpu().numpy())
                self.detectors.append(detector)

    def predict(self):
        test = self.x_test.to(self.device)
        #projected_datasets = [self.project_dataset(test, subspace) for subspace in self.subspaces]
        # Predict on each of the projected test datasets
        predictions = []
        predictions_probas = []
        with self.ui.display():
            for detector, subspace in zip(self.detectors, self.subspaces):
                self.ui.update(f"Predicting with detector {len(predictions)} / {len(self.subspaces)}")
                projected = self.project_dataset(test, subspace).cpu().numpy()
                prediction = detector.predict(projected)
                prediction_proba = detector.predict_proba(projected)[:,1]
                predictions.append(prediction)
                predictions_probas.append(prediction_proba)
            predictions_tensor = Tensor(predictions).T
            predictions_probas_tensor = Tensor(predictions_probas).T
            # Combine the predictions of each detector weighted by the probability of the subspace
            predictions_aggregated = (predictions_tensor * self.proba).sum(dim=1)
            predictions_probas_aggregated = (predictions_probas_tensor * self.proba).sum(dim=1)
            max_proba = predictions_probas_aggregated.max()
            min_proba = predictions_probas_aggregated.min()
            normalized_predictions_probas = (predictions_probas_aggregated - min_proba) / (max_proba - min_proba)
            predictions_rounded = normalized_predictions_probas.round().int()
            #self.predictions = predictions_aggregated.round().int().tolist()
            self.predictions = predictions_rounded.tolist()
        #del self.detectors
        #del self.subspaces

    def _get_name(self):
        return f"VGAN + {self.base_detector.__name__} + {self.space[0]}"

    def _get_predictions(self) -> List[int]:
        return self.predictions

    def get_space(self):
        return self.space

    def evaluate(self, output_path: Path = None, print_results = False) -> pd.DataFrame:
        output_path = output_path / self._get_name()
        visualizer = CollectiveVisualizer(tokenized_data=self.x_test, tokenizer = self.model, vmmd_model=self.vgan, export_path=str(output_path), text_visualization=not self.pre_embed)
        visualizer.visualize(samples=30, epoch=self.vgan.epochs)
        self.vgan.model_snapshot(path_to_directory=output_path)
        return super().evaluate(output_path=output_path)

    def project_dataset(self, dataset: Tensor, subspace: Tensor) -> Tensor:
        """
        :param dataset: Dataset to project. A tensor of shape (n_samples, n_features)
        :param subspace: Subspace to project on. A tensor of shape (n_features)

        :return: Projected dataset. A tensor of shape (n_samples, n_features)
        """
        #alt_approach = dataset @ subspace
        # expand subspace to match dataset shape
        subspace_expanded = subspace.expand(dataset.shape[0], -1)
        projected = dataset * subspace_expanded
        # Only pass the features that were selected, instead of leaving them empty
        subspace_mask = subspace != 0

        #If no feature is selected, select at least one at random.
        #TODO: Does this make sense?
        if not subspace_mask.any():
            print("Warning: Projection to zero space.")
            random_index = int(random() * len(subspace_mask))
            subspace_mask[random_index] = True
        trimmed = projected[:, subspace_mask]
        return trimmed

if __name__ == '__main__':
    #vmmd = VGAN_ODM(SimpleDataset(["This is an example"], 300), GPT2(), 200, 100)
    vmmd = VGAN_ODM(AGNews(), GPT2(), -1, -1)
    vmmd.train()
    vmmd.predict()
    vmmd.evaluate()