from pathlib import Path
from typing import Type, Optional

from numpy import ndarray

from models.Generator import GeneratorSigmoidSTE, Generator_big
from modules.text.vmmd_text import VMMDTextLightning
from modules.text.vmmd_text_lightning import VMMDTextLightningBase
from text.Embedding.huggingmodel import HuggingModel
from text.Embedding.llama import LLama1B
from text.dataset.dataset import AggregatableDataset
from text.outlier_detection.base_v_adapter import BaseVAdapter
from text.outlier_detection.space.space import Space
from text.vmmd_lightning_text_experiments import VMMDLightningTextExperiment


class TextVMMDAdapter(BaseVAdapter):
    """
    A class to adapt the VMMD model for text data.
    This class is designed to be used with the subclasses of VMMDTextLightningBase.
    It handles the training and visualization of the model.
    """
    def __init__(self,
                 dataset: AggregatableDataset,
                 space: Space,
                 output_path: Optional[Path] = None,):
        self.dataset = dataset
        self.space = space
        self.output_path = output_path
        self.model: VMMDTextLightningBase | None = None
        self.subspaces: Optional[ndarray] = None
        self.proba: Optional[ndarray] = None

    def train(self, emdedding_model: HuggingModel):
        model_run = VMMDLightningTextExperiment(
            emb_model=emdedding_model,
            vmmd_model=VMMDTextLightning,
            generator=GeneratorSigmoidSTE,
            version="v1",
            dataset=self.dataset,
            samples=self.space.train_size,
            yield_epochs=10,
            lr=1e-2,
            weight_decay=0.04,
            penalty_weight=1.,
            batch_size=10,
            epochs=1,
            export_path=self.output_path,
            export=self.output_path is not None,
        )
        self.model = model_run.run()

    def get_subspaces(self, num_subspaces: int = 50) -> ndarray[float]:
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        if self.subspaces is None:
            self.subspaces, self.proba = self._get_top_subspaces(self.model, num_subspaces)

        return self.subspaces

    def get_probabilities(self, num_subspaces: int = 50) -> ndarray[float]:
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        if self.proba is None:
            self.subspaces, self.proba = self._get_top_subspaces(self.model, num_subspaces)

        return self.proba
