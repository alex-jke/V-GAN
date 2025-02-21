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
from modules.od_module import VMMD_od, VGAN_od

from text.Embedding.gpt2 import GPT2
from text.dataset.SimpleDataset import SimpleDataset
from text.dataset.ag_news import AGNews
from text.dataset.emotions import EmotionDataset
from text.outlier_detection.ensemle_odm import EnsembleODM
from text.outlier_detection.odm import OutlierDetectionModel
from text.outlier_detection.pyod_odm import EmbeddingBaseDetector
from text.outlier_detection.space.space import Space
from text.outlier_detection.space.token_space import TokenSpace
from text.visualizer.collective_visualizer import CollectiveVisualizer
VGAN = "VGAN"
VMMD = "VMMD"

class VGAN_ODM(EnsembleODM):

    def __init__(self, dataset, space: Space, base_detector: Type[BaseDetector] = None, use_cached=False,
                 subspace_distance_lambda= 1.0, output_path: Path | None = None, classifier_delta = 1.0, model_type = VGAN):
        self.model_type = model_type
        if model_type != VGAN and model_type != VMMD:
            raise ValueError("VGAN_ODM only supports 'VGAN' and 'VMMD' models, passed: " + model_type)
        self.model = space.model
        self.seed = 777
        self.number_of_subspaces = 50
        self.base_detector: Type[BaseDetector] = base_detector
        if base_detector is None:
            self.base_detector = LUNAR

        self.base_output_path = output_path
        self.output_path = None
        self.detectors: List[BaseDetector] = []
        self.ensemble_model = None
        self.subspace_distance_lambda = subspace_distance_lambda
        self.classifier_delta = classifier_delta
        self.loaded_model = False
        super().__init__(dataset=dataset, space=space, use_cached=use_cached)

    def _get_detector(self) -> BaseDetector:
        if isinstance(self.space, TokenSpace): #TODO: find a better solution
            return EmbeddingBaseDetector(self.model.get_embedding_fun(batch_first=True), lambda: self.base_detector)
        return self.base_detector()

    def train_ensemble(self):
        self.vgan.approx_subspace_dist(add_leftover_features=False, subspace_count=50)
        if self.classifier_delta == 0:
            return
        self.ensemble_model = sel_SUOD(base_estimators=[self._get_detector()], subspaces=self.vgan.subspaces,
                                       n_jobs=4, bps_flag=False, approx_flag_global=False, verbose=True)

        self.ensemble_model.fit(self.x_train.cpu())

    def init_model(self):
        self.output_path = self.base_output_path / self.model_type / self.get_space()
        if self.model_type == "VMMD":
            self.vgan = VMMD_od(penalty_weight=0.1, generator=GeneratorSigmoidSTE,
                                lr=1e-5, epochs=10_000, seed=self.seed, path_to_directory=self.output_path,
                                weight_decay=0.0)
        elif self.model_type == "VGAN":
            self.vgan = VGAN_od(penalty=0.1, generator=GeneratorSigmoidSTE,
                                lr_G=1e-5, lr_D=1e-4, epochs=10_000, seed=self.seed, path_to_directory=self.output_path,
                                weight_decay=0.0)

    def _train(self):
        self.init_model()
        if self.seed is not None and self.output_path is not None:
            generator_path = self.output_path / "models" / "generator_0.pt"
            if generator_path.exists():
                self.vgan.load_models(generator_path, ndims=self.x_train.shape[1])
                self.loaded_model = True
                self.train_ensemble()
                return

        train = self.x_train.to(self.device)
        samples = self.x_train.shape[0]
        epochs = int(10 ** 6.7 / samples + 400) * 4
        # TODO: find a better way to set the epochs.
        if self.model_type == VGAN:
            epochs = int(10 ** 7.5 / samples + 7000) * 2
        if self.get_space() == "Tokenized":
            epochs = int(epochs * 1.5)
        batch_size = min(2500, samples)
        #epochs = epochs if self.pre_embed else epochs * 2
        self.vgan.epochs = epochs
        self.vgan.batch_size = batch_size
        print(f"training vmmd for {self.vgan.epochs} epochs.")

        if train.shape[0] != len(self.y_train):
            # TODO: this is causing errors, when not caching the data. Also when cached for the first time.
            raise RuntimeError(f"The training data and label should have the same length.")

        for epoch in self.vgan.yield_fit(train, yield_epochs=200):
            if epoch != 0:
                print(f"({epoch}, {self.vgan.train_history[self.vgan.generator_loss_key][-1]})")
        self.visualize_results()
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

    def _predict(self):
        agg_dec_fun = torch.zeros_like(self.y_test).cpu().numpy()
        if self.classifier_delta != 0:
            agg_dec_fun = self.get_ensemble_decision_function()
            if self.subspace_distance_lambda == 0.0:
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

    def visualize_results(self):
        if self.output_path is not None and not self.loaded_model:
            self.output_path.mkdir(parents=True, exist_ok=True)
            visualizer = CollectiveVisualizer(tokenized_data=self.x_test, tokenizer=self.model, vmmd_model=self.vgan,
                                              export_path=str(self.output_path), text_visualization=isinstance(self.space, TokenSpace))
            visualizer.visualize(samples=30, epoch=self.vgan.epochs)

    def _get_name(self):
        return f"{self.model_type} + {self.base_detector.__name__} + {self.get_space()[0]} + (λ{self.subspace_distance_lambda}, ∂{self.classifier_delta})"

    def _get_predictions(self) -> List[float]:
        return self.predictions.tolist()

    def evaluate(self, output_path: Path = None, print_results = False) ->( pd.DataFrame, pd.DataFrame):
        return super().evaluate(output_path=output_path)

class DistanceVGAN_ODM(VGAN_ODM):
    def __init__(self, dataset, space: Space, use_cached=False,
                 output_path=None, model_type: str = VGAN):
        super().__init__(dataset,
                         use_cached=use_cached, classifier_delta = 0.0, output_path=output_path, model_type=model_type, space=space)

    def _get_name(self):
        return f"{self.model_type} + only distance + {self.get_space()[0]}"

class EnsembleVGAN_ODM(VGAN_ODM):
    def __init__(self, dataset, space: Space,
                 use_cached=False, base_detector=None, output_path = None, model_type: str = VGAN):
        super().__init__(dataset,
                         use_cached=use_cached, subspace_distance_lambda = 0.0, base_detector=base_detector,
                         output_path=output_path, model_type=model_type, space=space)

    def _get_name(self):
        return f"{self.model_type} + {self.base_detector.__name__} + {self.get_space()[0]}"
