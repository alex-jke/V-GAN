from abc import ABC
from pathlib import Path
from typing import Tuple

from numpy import ndarray

from modules.od_module import ODModule, VMMD_od, VGAN_od
from modules.text.vmmd_text import VMMDTextLightning


class BaseVAdapter(ABC):
    """
    Base class for all V_ODM adapters. This class is the base of all v method based adapters.
    """
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

    def _load_model(self, base_path: Path, features: int, model: VMMD_od | VGAN_od | VMMDTextLightning) -> None:
        """
        Loads the model from the base_path.
        """
        if base_path is None:
            return
        generator_path = base_path / "models" / "generator_0.pt"
        if generator_path.exists():
            model.load_models(generator_path, ndims=features)
            print(f"Loaded model {model.__class__.__name__} from {generator_path}")
            self.loaded_model = True