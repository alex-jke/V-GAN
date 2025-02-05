from pathlib import Path
from typing import Tuple, List, Type

import numpy as np
import pandas as pd
from pyod.models.base import BaseDetector
from pyod.models.lof import LOF
from pyod.models.lunar import LUNAR
from torch import Tensor

import main
from models.Generator import GeneratorSigmoidSTE
from modules.od_module import VMMD_od

from text.Embedding.gpt2 import GPT2
from text.dataset.SimpleDataset import SimpleDataset
from text.dataset.emotions import EmotionDataset
from text.outlier_detection.odm import OutlierDetectionModel
from text.outlier_detection.pyod_odm import EmbeddingBaseDetector
from vmmd import VMMD


class VGAN_ODM(OutlierDetectionModel):

    def __init__(self, dataset, model, train_size, test_size, inlier_label=None, base_detector: Type[BaseDetector] = None, use_embedding=False):
        self.space = "Embedding" if use_embedding else "Tokenized"
        self.model = model
        self.vgan = VMMD_od(epochs=2000, penalty_weight=0.5, generator=GeneratorSigmoidSTE)
        self.number_of_subspaces = 100
        self.base_detector: Type[BaseDetector] = base_detector
        if base_detector is None:
            self.base_detector = LUNAR

        self.detectors: List[BaseDetector] = []
        self.init_dataset = self.use_embedding if use_embedding else self.use_tokenized
        super().__init__(dataset, model, train_size, test_size, inlier_label)

    def _get_detector(self) -> BaseDetector:
        if not self.use_embedding:
            return EmbeddingBaseDetector(self.model, lambda: self.base_detector)
        return self.base_detector()

    def train(self):
        self.init_dataset()
        train = self.x_train.to(self.device)
        self.vgan.fit(train)
        subspaces = self.vgan.generate_subspaces(self.number_of_subspaces)
        unique_subspaces, proba = np.unique(
            np.array(subspaces.to('cpu')), axis=0, return_counts=True)
        self.subspaces = Tensor(unique_subspaces).to(self.device)
        self.proba = proba / proba.sum()

        # If one subspace has probabilities over 0.5, it decides the inlier label
        # To improve runtime we thus can remove the other subspaces
        main_subspace_exits = (self.proba > 0.5).sum() > 0
        if main_subspace_exits:
            self.subspaces = self.subspaces[self.proba > 0.5]
            self.proba = self.proba[self.proba > 0.5]
            #print("Main subspace found, removing other subspaces")

        # Project the dataset into each of the generated subspaces
        #projected_datasets = [self.project_dataset(train, subspace) for subspace in self.subspaces]
        # Fit a detector on each of the projected datasets. To avoid memory issues, we fit the detectors one by one and project the dataset one by one
        self.detectors = [self._get_detector().fit(self.project_dataset(train, subspace).cpu().numpy()) for subspace in self.subspaces]

    def predict(self):
        test = self.x_test.to(self.device)
        #projected_datasets = [self.project_dataset(test, subspace) for subspace in self.subspaces]
        # Predict on each of the projected test datasets
        predictions =Tensor([detector.predict(self.project_dataset(test, subspace).cpu().numpy())
                             for detector, subspace in zip(self.detectors, self.subspaces)])
        # Combine the predictions of each detector weighted by the probability of the subspace
        predictions_aggregated = (predictions.T * self.proba).sum(dim=1)
        self.predictions = predictions_aggregated.round().int().tolist()
        del self.detectors
        del self.subspaces

    def _get_name(self):
        return f"VGAN + {self.base_detector.__name__} + {self.space[0]}"

    def _get_predictions(self) -> List[int]:
        return self.predictions

    def get_space(self):
        return self.space

    def evaluate(self, output_path: Path = None) -> pd.DataFrame:
        output_path = output_path / self._get_name()
        main.visualize(tokenized_data=self.x_test, tokenizer = self.model, model=self.vgan, path=str(output_path), text_visualization=not self.use_embedding())
        self.vgan.model_snapshot(path_to_directory=output_path, )
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
        return projected

if __name__ == '__main__':
    #vmmd = VGAN_ODM(SimpleDataset(["This is an example"], 300), GPT2(), 200, 100)
    vmmd = VGAN_ODM(EmotionDataset(), GPT2(), 200, 100)
    vmmd.train()
    vmmd.predict()
    vmmd.evaluate()