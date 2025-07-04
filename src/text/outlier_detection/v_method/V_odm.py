from pathlib import Path
from typing import List, Type, Callable, Optional

import numpy as np
import pandas as pd
from numpy import ndarray
from pyod.models.base import BaseDetector
from pyod.models.lunar import LUNAR
from sel_suod.models.base import sel_SUOD
from torch import Tensor

import torch

from text.outlier_detection.ensemle_odm import EnsembleODM
from text.outlier_detection.pyod_odm import TransformBaseDetector
from text.outlier_detection.space.space import Space
from text.outlier_detection.space.token_space import TokenSpace
from text.outlier_detection.space_type import SpaceType
from text.outlier_detection.v_method.numerical_v_adapter import NumericalVOdmAdapter
from text.outlier_detection.v_method.vmmd_adapter import VMMDAdapter
from text.visualizer.od import od_subspace_visualizer


class V_ODM(EnsembleODM):

    def __init__(self, dataset, space: Space, base_detector: Type[BaseDetector] = None, use_cached=False,
                 subspace_distance_lambda= 1.0, output_path: Path | None = None, classifier_delta = 1.0,
                 odm_model: Optional[NumericalVOdmAdapter] = None, **params):

        # The number of subspaces to sample from the random operator. Currently set to 50, as the runtime rises rapidly with more subspaces.
        # Since the subspaces are sorted by probability this means the 50 most likely subspaces are sampled
        self.num_subspaces = 200

        if odm_model is None:
            odm_model = VMMDAdapter()
        self.odm_model: NumericalVOdmAdapter = odm_model
        self.model = space.model
        if base_detector is None:
            base_detector = LUNAR
        self.base_detector: Type[BaseDetector] = base_detector

        self.base_output_path = output_path
        self.detectors: List[BaseDetector] = []
        self.ensemble_model = None
        self.subspace_distance_lambda = subspace_distance_lambda
        self.classifier_delta = classifier_delta
        super().__init__(dataset=dataset, space=space, use_cached=use_cached, base_method=base_detector, **params)

    def get_space_type(self) -> SpaceType:
        return SpaceType.VGAN

    def _get_transformation_function(self) -> Callable[[ndarray], ndarray]:
        """
        Returns a function that can be used to transform the input data to the space used by the model.
        If the space is a token space, the base detector is expected to work on the embeddings generated from the projected tokens.
        """

        if isinstance(self.space, TokenSpace): #TODO: find a better solution
            return TransformBaseDetector.configure_transform_fun(
                transformation= self.model.get_embedding_fun(batch_first=True),
                normalize = True,
                standardize = True
            )

        return lambda x: x

    def _get_detector(self) -> BaseDetector:
        if isinstance(self.space, TokenSpace): # TODO: better solution needed
            transform_fun = self._get_transformation_function()
            return TransformBaseDetector(transform_fun=transform_fun, base_detector= lambda: self.base_detector)
        return self.base_detector()


    def _train_ensemble(self):
        """
        If the classifier delta is 0, the ensemble model is not trained, as its results will not be used in the final prediction
        """
        if self.classifier_delta == 0:
            return
        subspaces = self.odm_model.get_subspaces(num_subspaces=self.num_subspaces).astype(bool)
        self.ensemble_model = sel_SUOD(base_estimators=[self._get_detector()], subspaces=subspaces,
                                       n_jobs=1, bps_flag=False, approx_flag_global=False, verbose=True)

        self.ensemble_model.fit(self.x_train.cpu())

    def _train(self):
        # The model is initialized here, as preparing the data (such as embedding) might take a not insignificant amount of time.
        # To allow for a fair comparison
        self.odm_model.init_model(self.data, self.base_output_path, self.space)
        self.odm_model.train()
        self._train_ensemble()

    @staticmethod
    def _calculate_distances(transform: Callable[[np.ndarray], np.ndarray], subspaces: Tensor,
                             x_test: Tensor) -> Tensor:
        """
        Calculates for each datapoint its distance to the closest subspace. The distance is measured in the transformed space.
        :param transform: The transformation function to apply to the data to measure the distance.
            For example, if the space is a token space, the data should be transformed into the embedding space, before measuring the distance.
        :param subspaces: A two-dimensional tensor containing the subspaces to measure the distance to. Shape: (num_subspaces, num_features)
        :param x_test: The data to measure the distance for. As a two-dimensional tensor. Shape: (num_samples, num_features)
        :return: A tensor of the distances for each data point of shape (num_samples,)
        """

        subspace_dist = Tensor([])

        # Should not be more due to normalization.
        max_dist = x_test.shape[1] ** 0.5

        test_data = x_test.cpu().numpy()
        transformed_points = Tensor(transform(test_data)).cpu()
        for subspace in subspaces:
            try:
                projected: np.ndarray = (subspace * x_test).cpu().numpy()
            except RuntimeError as e:
                print(e)
                raise e
            transformed_projected = Tensor(transform(projected)).cpu()
            sub_distances = Tensor(transformed_points - transformed_projected).norm(dim=1)
            sub_distances = sub_distances.unsqueeze(1)
            subspace_dist = torch.cat((subspace_dist, sub_distances), dim=1)

        # In case only one subspace was found.
        agg_dist = subspace_dist.min(dim=1).values if len(subspace_dist.shape) == 2 else subspace_dist
        print(f"Aggregated over {subspaces.shape} datapoints.")
        dist_tensor = agg_dist / max_dist
        return dist_tensor

    def _get_ensemble_decision_function(self):
        if self.ensemble_model is None:
            raise RuntimeError(f"Ensemble model not initialized. Please call train() first.")
        test = self.x_test.to(self.device)
        decision_function_scores_ens = self.ensemble_model.decision_function(
            test.cpu())
        agg_dec_fun = self.aggregator_funct(
            decision_function_scores_ens, weights=self.odm_model.get_probabilities(), type="avg")
        return agg_dec_fun

    def _get_distances(self) -> Tensor:
        subspaces = Tensor(self.odm_model.get_subspaces()).to(self.device)
        transform = self._get_transformation_function()
        x_test = self.x_test
        subspace_distance_lambda = self.subspace_distance_lambda
        return self._calculate_distances(transform, subspaces, x_test) * subspace_distance_lambda


    def _predict(self):
        agg_dec_fun = torch.zeros_like(self.y_test).cpu().numpy()
        dist_tensor = torch.zeros_like(self.y_test)
        # No need to calculate the ensemble decision function if the classifier delta is 0.
        if self.classifier_delta != 0.0:
            agg_dec_fun = self._get_ensemble_decision_function()

        # No need to calculate the subspace distance if the lambda is 0.
        if self.subspace_distance_lambda != 0.0:
            dist_tensor = self._get_distances()

        self.predictions = self.classifier_delta * agg_dec_fun + dist_tensor.cpu().numpy()

        od_subspace_visualizer.plot_subspaces(self.odm_model.get_subspaces(), self.odm_model.get_probabilities(),
                                              self.base_output_path /  "subspaces_used" / f"{self._get_name()}")

    def _get_name(self):
        return f"{self.odm_model.get_name()} + {self.base_detector.__name__} + {self.get_space()[0]} + (λ{self.subspace_distance_lambda}, ∂{self.classifier_delta})"

    def _get_predictions(self) -> List[float]:
        return self.predictions.tolist()

    def evaluate(self, output_path: Path = None) ->( pd.DataFrame, pd.DataFrame):
        return super().evaluate(output_path=output_path)
