from abc import ABC

import numpy as np
import random

from text.outlier_detection.odm import OutlierDetectionModel


class EnsembleODM(OutlierDetectionModel, ABC):

    def aggregator_funct(self, decision_function: np.array, type: str = "avg", weights: np.ndarray = None) -> np.ndarray:
        assert type in ["avg", "exact"], f"{type} aggregation not found"

        if type == "avg":
            return np.average(decision_function, axis=1, weights=weights)

        if type == "exact":
            weights = weights / weights.sum()
            random_indexes = random.choices(
                range(decision_function.shape[1]), k=decision_function.shape[0])
            aggregated_scores = [weights[random_indexes[i]]
                                 * (decision_function[i])[random_indexes[i]] for i in range(decision_function.shape[0])]
            return aggregated_scores