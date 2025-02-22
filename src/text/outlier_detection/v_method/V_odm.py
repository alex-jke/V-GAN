from pathlib import Path
from typing import List, Type

import pandas as pd
from pyod.models.base import BaseDetector
from pyod.models.lunar import LUNAR
from sel_suod.models.base import sel_SUOD
from torch import Tensor

import torch

from text.outlier_detection.ensemle_odm import EnsembleODM
from text.outlier_detection.pyod_odm import TransformBaseDetector
from text.outlier_detection.space.space import Space
from text.outlier_detection.space.token_space import TokenSpace
from text.outlier_detection.v_method.base_v_adapter import BaseVOdmAdapter
from text.outlier_detection.v_method.vmmd_adapter import VMMDAdapter

class V_ODM(EnsembleODM):

    def __init__(self, dataset, space: Space, base_detector: Type[BaseDetector] = None, use_cached=False,
                 subspace_distance_lambda= 1.0, output_path: Path | None = None, classifier_delta = 1.0,
                 odm_model: BaseVOdmAdapter = VMMDAdapter()):

        # The number of subspaces to sample from the random operator. Currently set to 50, as the runtime rises rapidly with more subspaces.
        # This way, subspaces with an occurrence of less than 2% are expected to not contribute much. Since sampling is random, they might still do,
        # but the probability is low.
        self.num_subspaces = 50

        self.odm_model = odm_model
        self.model = space.model
        self.base_detector: Type[BaseDetector] = base_detector
        if base_detector is None:
            self.base_detector = LUNAR

        self.base_output_path = output_path
        self.detectors: List[BaseDetector] = []
        self.ensemble_model = None
        self.subspace_distance_lambda = subspace_distance_lambda
        self.classifier_delta = classifier_delta
        super().__init__(dataset=dataset, space=space, use_cached=use_cached)

    def _get_transformation_function(self):
        """
        Returns a function that can be used to transform the input data to the space used by the model.
        If the space is a token space, the base detector is expected to work on the embeddings generated from the projected tokens.
        """
        if isinstance(self.space, TokenSpace): #TODO: find a better solution
            return self.model.get_embedding_fun(batch_first=True)
        return lambda x: x

    def _get_detector(self) -> BaseDetector:
        transform_fun = self._get_transformation_function()
        return TransformBaseDetector(transform_fun=transform_fun, base_detector= lambda: self.base_detector)


    def _train_ensemble(self):
        """
        If the classifier delta is 0, the ensemble model is not trained, as its results will not be used in the final prediction
        """
        if self.classifier_delta == 0:
            return
        self.ensemble_model = sel_SUOD(base_estimators=[self._get_detector()], subspaces=self.odm_model.get_subspaces(),
                                       n_jobs=4, bps_flag=False, approx_flag_global=False, verbose=True)

        self.ensemble_model.fit(self.x_train.cpu())

    def _train(self):
        # The model is initialized here, as preparing the data (such as embedding) might take a not insignificant amount of time.
        # To allow for a fair comparison
        self.odm_model.init_model(self.data, self.base_output_path, self.space)
        self.odm_model.train(num_subspaces=self.num_subspaces)
        self._train_ensemble()

    def _get_ensemble_decision_function(self):
        if self.ensemble_model is None:
            raise RuntimeError(f"Ensemble model not initialized. Please call train() first.")
        test = self.x_test.to(self.device)
        decision_function_scores_ens = self.ensemble_model.decision_function(
            test.cpu())
        agg_dec_fun = self.aggregator_funct(
            decision_function_scores_ens, weights=self.odm_model.get_subspace_probabilities(), type="avg")
        return agg_dec_fun

    def _get_distances(self):

        # Use python list to avoid memory issues. Torch tensor operation
        # would likely involve unsquezing and expanding the tensor.
        # Doing this for all subspaces and all points would likely cause
        # memory issues. Also, currently this approach still seems fast
        # enough. Though, it should be kept in mind.
        subspace_min_distance = []
        subspaces = Tensor(self.odm_model.get_subspaces()).to(self.device)

        transform = self._get_transformation_function()

        # Should not be more due to normalization.
        max_dist = self.x_test.shape[1] ** 0.5

        # For each point, calculate the distance to the closest subspace.
        # The distance is calculated in the transformed space. Which,
        # currently is the embedding space. That is, either the point is in the token space and transformed into the embedding space,
        # using the embedding function, or the point is already in the embedding space. Then, the identity function is applied.
        for point in self.x_test:
            min_distance = max_dist
            transformed_point = transform(point)

            for subspace in subspaces:
                transformed_projected_point = transform(subspace * point)
                sub_dist = (transformed_point - transformed_projected_point).norm()
                min_distance = min(min_distance, sub_dist)

            subspace_min_distance.append(min_distance)

        dist_tensor = Tensor(subspace_min_distance) / max_dist * self.subspace_distance_lambda
        return dist_tensor

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

    def _get_name(self):
        return f"{self.odm_model.get_name()} + {self.base_detector.__name__} + {self.get_space()[0]} + (λ{self.subspace_distance_lambda}, ∂{self.classifier_delta})"

    def _get_predictions(self) -> List[float]:
        return self.predictions.tolist()

    def evaluate(self, output_path: Path = None, print_results = False) ->( pd.DataFrame, pd.DataFrame):
        return super().evaluate(output_path=output_path)
