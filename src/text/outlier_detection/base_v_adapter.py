from abc import ABC
from typing import Tuple

from numpy import ndarray

from modules.od_module import ODModule


class BaseVAdapter(ABC):
    @staticmethod
    def _get_top_subspaces(model: ODModule, num_subspaces: int) -> Tuple[ndarray[float], ndarray[float]]:
        """
        Returns the num_subspaces most probable subspaces, with their probabilities.
        """
        model.approx_subspace_dist(add_leftover_features=False, subspace_count=1000)
        subspaces = model.subspaces
        proba = model.proba

        amount = min(num_subspaces, len(proba))
        idx = proba.argsort()[-amount:][::-1]
        top_subspaces = subspaces[idx]
        top_proba = proba[idx]

        # Ignore Subspaces contributing less than 0.2% to the model if the rest of the subspaces make up at least 80%
        threshold = 0.002
        if top_proba[top_proba > threshold].sum() > 0.8:
            top_subspaces = top_subspaces[top_proba > threshold]
            top_proba = top_proba[top_proba > threshold]

        return top_subspaces, top_proba