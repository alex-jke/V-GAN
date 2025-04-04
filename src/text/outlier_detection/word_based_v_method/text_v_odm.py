from typing import List, Type, Optional

import torch
from numpy import ndarray
from pyod.models.base import BaseDetector
from pyod.models.lunar import LUNAR as pyod_LUNAR
from torch import Tensor

from text.UI import cli
from text.UI.cli import ConsoleUserInterface
from text.dataset.dataset import Dataset, AggregatableDataset
from text.outlier_detection.ensemle_odm import EnsembleODM
from text.outlier_detection.space.prepared_data import PreparedData
from text.outlier_detection.space.space import Space
from text.outlier_detection.space.word_space import WordSpace
from text.outlier_detection.word_based_v_method.text_v_adapter import TextVMMDAdapter


class TextVOdm(EnsembleODM):

    def __init__(self, dataset: AggregatableDataset, space: WordSpace, base_method: Type[BaseDetector] = pyod_LUNAR, inlier_label: int | None = None, use_cached: bool = False,
                 output_path: str | None = None, subspace_distance_lambda=1.0, classifier_delta=1.0):
        if not isinstance(space, WordSpace):
            raise ValueError("TextVOdm only works with WordSpace.")
        if not isinstance(dataset, AggregatableDataset):
            raise ValueError("TextVOdm only works with AggregatableDataset.")
        super().__init__(dataset=dataset, space=space, base_method=base_method, inlier_label=inlier_label, use_cached=use_cached)
        self.subspace_distance_lambda = subspace_distance_lambda
        self.classifier_delta = classifier_delta
        self.output_path = output_path
        output_path = output_path + "/vmmd_text" if output_path is not None else None
        self.v_method = TextVMMDAdapter(dataset, self.space, output_path=output_path)
        self.amount_subspaces = 5
        self.detectors: List[BaseDetector] = []
        self.embedded_data: List[PreparedData] = []
        self.prepared_data: Optional[PreparedData] = None
        self.space = space
        self.dataset = dataset

    def _train_ensemble(self):
        subspaces = self.v_method.get_subspaces(self.amount_subspaces)
        ui = cli.get()
        with ui.display():
            for i in range(self.amount_subspaces):
                ui.update(f"Fitting for subspace {i + 1}/{self.amount_subspaces}")
                subspace = subspaces[i]
                with torch.no_grad():
                    prepared_data = self.space.transform_dataset(dataset=self.dataset, use_cached=False, inlier_label=self.inlier_label, masks=subspace)
                self.embedded_data.append(prepared_data)
                x_data = prepared_data.x_train
                detector = self.base_method()
                detector.fit(x_data.cpu().numpy())
                self.detectors.append(detector)

    def _train(self):
        model = self.space.model
        self.v_method.train(emdedding_model=model)
        self._train_ensemble()

    @staticmethod
    def _calculate_distances(x_test: Tensor, embedded_data: List[PreparedData], space) -> Tensor:
        # Get the unprojected data samples and ensure its shape is correct
        assert len(x_test.shape) == 2

        # Get the projections and find for each point the minimum distance to a projected version.
        x_projections = torch.stack([data.x_test for data in embedded_data], dim=2)
        dif = x_test.unsqueeze(dim=2) - x_projections
        distances = dif.norm(p=2, dim=1)
        min_distances = distances.min(dim=1).values
        assert len(min_distances.shape) == 1 and min_distances.shape[0] == space.test_size, (f"The shape of the min_distances tensor is not correct. "
                                                                                          f"Expected: {space.test_size}, got {min_distances.shape}")

        return min_distances

    def _predict_ensemble(self) -> Tensor:
        ensemble_scores = torch.zeros_like(self.y_test).float().cpu()
        for detector, data, proba in zip(self.detectors, self.embedded_data, self.v_method.get_probabilities()):
            # Predict the scores for each detector
            score: ndarray = detector.decision_function(data.x_test.cpu().numpy())
            ensemble_scores += Tensor(score).cpu() * proba
        ensemble_scores /= len(self.detectors)
        return ensemble_scores

    def _predict(self):
        subspace_dists = torch.zeros_like(self.y_test).cpu()
        if self.subspace_distance_lambda != 0:
            subspace_dists = self._calculate_distances(self.x_test, self.embedded_data, self.space).cpu()

        ensemble_scores = torch.zeros_like(self.y_test)
        if self.classifier_delta != 0:
            ensemble_scores = self._predict_ensemble()
        # Combine the two scores
        combined_scores = self.classifier_delta * ensemble_scores + self.subspace_distance_lambda * subspace_dists
        assert len(combined_scores.shape) == 1 and combined_scores.shape == self.y_test.shape
        self.predictions = combined_scores

    def _get_name(self) -> str:
        return "TextV + " + self.base_method.__name__ + " + self.space.name "+ f"(λ{self.subspace_distance_lambda}, ∂{self.classifier_delta})"

    def _get_predictions(self) -> List[float]:
        return self.predictions.tolist()