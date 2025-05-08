from pathlib import Path
from typing import Type, Optional

import numpy as np
from numpy import ndarray

import text.UI.cli
from models.Generator import GeneratorSigmoidSTE, Generator_big, GeneratorSoftmaxSTE, GeneratorUpperSoftmax
from modules.text.vmmd_text import VMMDTextLightning
from modules.text.vmmd_text_lightning import VMMDTextLightningBase
from text.Embedding.LLM.huggingmodel import HuggingModel
from text.Embedding.LLM.llama import LLama1B
from text.Embedding.unification_strategy import StrategyInstance, UnificationStrategy
from text.UI import cli
from text.dataset.dataset import AggregatableDataset
from text.dataset_converter.dataset_preparer import DatasetPreparer
from text.outlier_detection.base_v_adapter import BaseVAdapter
from text.outlier_detection.space.space import Space
from text.vmmd_lightning_text_experiments import VMMDLightningTextExperiment

ui = cli.get()

class TextVMMDAdapter(BaseVAdapter):
    """
    A class to adapt the VMMD model for text data.
    This class is designed to be used with the subclasses of VMMDTextLightningBase.
    It handles the training and visualization of the model.
    """

    def get_name(self):
        return "TextVMMD + " + self.generator.__name__

    def __init__(self,
                 dataset: AggregatableDataset,
                 space: Space,
                 inlier_label,
                 output_path: Optional[Path] = None,
                 aggregation_strategy: StrategyInstance = UnificationStrategy.TRANSFORMER.create(),
                 use_mmd: bool = False,
                 generator: Type[Generator_big]  = GeneratorSigmoidSTE,
                 epochs = 25,
                 batch_size = 10):
        self.dataset = dataset
        self.space = space
        agg_str = aggregation_strategy.key()
        loss_str = ("mmd" if use_mmd else "lse")
        self.output_path = output_path / "VMMD_Text"  / agg_str / generator.__name__ / loss_str if output_path is not None else None
        self.model: VMMDTextLightningBase | None = None
        self.subspaces: Optional[ndarray] = None
        self.proba: Optional[ndarray] = None
        self.use_mmd = use_mmd
        self.strategy = aggregation_strategy
        self.emdedding_model: HuggingModel = space.model
        self.inlier_label = inlier_label
        self.generator = generator
        self.epochs = epochs
        self.batch_size = batch_size

    def _get_params(self):
        NotImplementedError("Change params based on use_mmd and transformer_aggregation")

    def train(self):
        self._get_params()
        epochs = self.epochs
        sequence_length = DatasetPreparer.get_max_sentence_length(np.array(self.dataset.get_training_data()[0].tolist()))
        max_sequence_length = self.space.model.max_word_length() - self.dataset.prompt.prompt_length
        if sequence_length > max_sequence_length:
            print(f"Trimming sequence length (words) from {sequence_length} to {max_sequence_length}.")
            sequence_length = max_sequence_length
        model_run = VMMDLightningTextExperiment(
            emb_model=self.emdedding_model,
            vmmd_model=VMMDTextLightning,
            generator=self.generator,
            version="",
            dataset=self.dataset,
            samples=self.space.train_size,
            yield_epochs=10,
            lr=1e-3,
            weight_decay=0.0,
            penalty_weight=0.01,
            batch_size=self.batch_size,
            epochs=epochs,
            export_path=self.output_path,
            export=self.output_path is not None,
            aggregation_strategy=self.strategy,
            use_mmd=self.use_mmd,
            labels = [self.inlier_label],
            sequence_length=sequence_length
        )
        with ui.display(f"Training"):
            self.model = model_run.run()
        #model_run.visualize(epoch=epochs, model=self.model, sentences=self.model)

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
