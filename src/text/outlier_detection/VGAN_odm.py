from pathlib import Path
from random import random
from typing import Tuple, List, Type

import numpy as np
import pandas as pd
from pyod.models.base import BaseDetector
from pyod.models.lof import LOF
from pyod.models.lunar import LUNAR
from sel_suod.models.base import sel_SUOD
from torch import Tensor

import torch
from models.Generator import GeneratorSigmoidSTE, GeneratorUpperSoftmax, GeneratorSpectralNorm
from modules.od_module import VMMD_od

from text.Embedding.gpt2 import GPT2
from text.dataset.SimpleDataset import SimpleDataset
from text.dataset.ag_news import AGNews
from text.dataset.emotions import EmotionDataset
from text.outlier_detection.ensemle_odm import EnsembleODM
from text.outlier_detection.odm import OutlierDetectionModel
from text.outlier_detection.pyod_odm import EmbeddingBaseDetector
from text.visualizer.collective_visualizer import CollectiveVisualizer
from vmmd import VMMD


class VGAN_ODM(EnsembleODM):

    def __init__(self, dataset, model, train_size, test_size, inlier_label=None, base_detector: Type[BaseDetector] = None, pre_embed=False, use_cached=False,
                 subspace_distance_lambda= 1.0, output_path: Path | None = None, classifier_delta = 1.0):
        self.space = "Embedding" if pre_embed else "Tokenized"
        self.model = model
        self.seed = 777
        self.output_path = output_path
        self.vgan = VMMD_od(penalty_weight=0.1, generator=GeneratorSigmoidSTE,
                            lr=1e-5, epochs=5000, seed=self.seed, path_to_directory=output_path)
        self.number_of_subspaces = 50
        self.base_detector: Type[BaseDetector] = base_detector
        if base_detector is None:
            self.base_detector = LUNAR

        self.detectors: List[BaseDetector] = []
        self.init_dataset = self.use_embedding if pre_embed else self.use_tokenized
        self.pre_embed = pre_embed
        self.ensemble_model = None
        self.subspace_distance_lambda = subspace_distance_lambda
        self.classifier_delta = classifier_delta
        self.loaded_model = False
        super().__init__(dataset, model, train_size, test_size, inlier_label, use_cached=use_cached)

    def _get_detector(self) -> BaseDetector:
        if not self.pre_embed:
            return EmbeddingBaseDetector(self.model.get_embedding_fun(batch_first=True), lambda: self.base_detector)
        return self.base_detector()

    def train_ensemble(self):
        self.vgan.approx_subspace_dist(add_leftover_features=False, subspace_count=50)
        if self.classifier_delta == 0:
            return
        self.ensemble_model = sel_SUOD(base_estimators=[self._get_detector()], subspaces=self.vgan.subspaces,
                                       n_jobs=6, bps_flag=False, approx_flag_global=False, verbose=True)

        self.ensemble_model.fit(self.x_train.cpu())

    def train(self):
        self.init_dataset()
        if self.seed is not None and self.output_path is not None:
            generator_path = self.output_path / "models" / "generator_0.pt"
            if generator_path.exists():
                self.vgan.load_models(generator_path, ndims=self.x_train.shape[1])
                self.loaded_model = True
                self.train_ensemble()
                return

        train = self.x_train.to(self.device)

        epochs = int(10 ** 6.7 / len(train) + 400) * 2
        #epochs = epochs if self.pre_embed else epochs * 2
        self.vgan.epochs = epochs
        print(f"training vmmd for {self.vgan.epochs} epochs.")

        if train.shape[0] != len(self.y_train): #TODO embedding dataset somehow seems to be trained with too many samples.
            raise RuntimeError(f"The training data and label should have the same length.")

        #with self.ui.display():
        for epoch in self.vgan.yield_fit(train, yield_epochs=200):
            #self.ui.update(f"Fitting VGAN, current epoch {epoch}")
            if epoch != 0:
                print(f"({epoch}, {self.vgan.train_history[self.vgan.generator_loss_key][-1]})")

        self.train_ensemble()

    def get_ensemble_decision_function(self):
        if self.ensemble_model is None:
            raise RuntimeError(f"Ensemble model not initialized. Please call train() first.")
        test = self.x_test.to(self.device)
        decision_function_scores_ens = self.ensemble_model.decision_function(
            test.cpu())
        agg_dec_fun = self.aggregator_funct(
            decision_function_scores_ens, weights=self.vgan.proba, type="avg")
        return agg_dec_fun

    def predict(self):
        agg_dec_fun = torch.zeros_like(self.y_test).cpu().numpy()
        if self.classifier_delta != 0:
            agg_dec_fun = self.get_ensemble_decision_function()
            if not self.pre_embed or self.subspace_distance_lambda == 0.0:
                self.predictions = agg_dec_fun
                return

        subspace_min_distance = []
        subspaces = Tensor(self.vgan.subspaces).to(self.device)
        max_dist = self.x_test.shape[1]**0.5
        for point in self.x_test:
            min_distance = max_dist# Should not be more due to normalization.
            for subspace in subspaces:
                sub_dist = (point - subspace * point).norm()
                min_distance = min(min_distance, sub_dist)
            subspace_min_distance.append(min_distance)
        dist_tensor = Tensor(subspace_min_distance) / max_dist * self.subspace_distance_lambda

        self.predictions = self.classifier_delta * agg_dec_fun + dist_tensor.cpu().numpy()

    def _get_name(self):
        return f"VGAN + {self.base_detector.__name__} + {self.space[0]} + (λ{self.subspace_distance_lambda}, ∂{self.classifier_delta})"

    def _get_predictions(self) -> List[float]:
        return self.predictions

    def get_space(self):
        return self.space

    def evaluate(self, output_path: Path = None, print_results = False) ->( pd.DataFrame, pd.DataFrame):

        if output_path is not None and not self.loaded_model:
            output_path = output_path / "VGAN"
            visualizer = CollectiveVisualizer(tokenized_data=self.x_test, tokenizer = self.model, vmmd_model=self.vgan, export_path=str(output_path), text_visualization=not self.pre_embed)
            visualizer.visualize(samples=30, epoch=self.vgan.epochs)
            self.vgan.model_snapshot(path_to_directory=output_path)
        return super().evaluate(output_path=output_path)

class DistanceVGAN_ODM(VGAN_ODM):
    def __init__(self, dataset, model, train_size, test_size, inlier_label=None, pre_embed=False, use_cached=False,
                 output_path=None):
        super().__init__(dataset, model, train_size, test_size, inlier_label, pre_embed=pre_embed,
                         use_cached=use_cached, classifier_delta = 0.0, output_path=output_path)

    def _get_name(self):
        return f"VGAN + only distance + {self.space[0]}"

class EnsembleVGAN_ODM(VGAN_ODM):
    def __init__(self, dataset, model, train_size, test_size, inlier_label=None, pre_embed=False,
                 use_cached=False, base_detector=None, output_path = None):
        super().__init__(dataset, model, train_size, test_size, inlier_label, pre_embed=pre_embed,
                         use_cached=use_cached, subspace_distance_lambda = 0.0, base_detector=base_detector,
                         output_path=output_path)

    def _get_name(self):
        return f"VGAN + {self.base_detector.__name__} + {self.space[0]}"


if __name__ == '__main__':
    #vmmd = VGAN_ODM(SimpleDataset(["This is an example"], 300), GPT2(), 200, 100)
    vmmd = VGAN_ODM(AGNews(), GPT2(), -1, -1)
    vmmd.train()
    vmmd.predict()
    vmmd.evaluate()