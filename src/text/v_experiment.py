import os
from abc import abstractmethod
from argparse import ArgumentError
from datetime import datetime
from pathlib import Path
from tokenize import Ignore

import torch
from torch import Tensor

# === Imports from your modules ===
from models.Generator import GeneratorSigmoid, FakeGenerator, GeneratorUpperSoftmax, GeneratorSigmoidSTE, \
    GeneratorSpectralNorm, GeneratorSigmoidSTEMBD
from modules.od_module import VMMD_od, VGAN_od
from text.Embedding.bert import Bert
from text.Embedding.gpt2 import GPT2
from text.Embedding.deepseek import DeepSeek1B, DeepSeek7B
from text.Embedding.gpt2ExtraSubspace import GPT2ExtraSubspaces
from text.Embedding.huggingmodel import HuggingModel
from text.Embedding.tokenizer import Tokenizer
from text.dataset.SimpleDataset import SimpleDataset
from text.dataset.ag_news import AGNews
from text.dataset.dataset import Dataset
from text.dataset.emotions import EmotionDataset
from text.dataset.imdb import IMBdDataset
from text.dataset.nlp_adbench import NLP_ADBench
from text.dataset.synthetic_dataset import SyntheticDataset
from text.dataset.wikipedia_slim import WikipediaPeopleDataset
from src.text.dataset_converter.dataset_embedder import DatasetEmbedder
from text.dataset_converter.dataset_processor import DatasetProcessor
from text.dataset_converter.dataset_tokenizer import DatasetTokenizer
from text.visualizer.average_alpha_visualizer import AverageAlphaVisualizer
from text.visualizer.collective_visualizer import CollectiveVisualizer
from text.visualizer.timer import Timer
from vgan import VGAN

from vmmd import VMMD, model_eval


# === Utility: Device selection ===
def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda:0')
    elif torch.backends.mps.is_available():
        return torch.device('mps:0')
    else:
        return torch.device('cpu')


DEVICE = get_device()


# === Experiment class ===
class VExperiment:
    """
    Encapsulates one experiment (data processing, model training/evaluation, visualization).
    """

    def __init__(self,
                 dataset: Dataset, model: HuggingModel, generator_class=GeneratorSigmoidSTE,
                 sequence_length: int = None, epochs: int = 2000,
                 batch_size: int = 500, samples: int = 2000, penalty_weight: float = 0.5, lr: float = 0.007,
                 momentum: float = 0.99, weight_decay: float = 0.04,
                 version: str = "0.0", yield_epochs: int = 200, train: bool = False, pre_embed: bool = False,
                 use_embedding: bool = False,
                 gradient_clipping=False):
        self.dataset = dataset
        self.model = model
        self.generator_class = generator_class
        self.sequence_length: int | None = sequence_length
        self.epochs = epochs
        self.batch_size = batch_size
        self.samples = samples
        self.penalty_weight = penalty_weight
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.version = version
        self.yield_epochs = yield_epochs
        self.train = train
        self.pre_embed = pre_embed
        self.device = DEVICE
        self.gradient_clipping = gradient_clipping
        if use_embedding and pre_embed:
            raise ValueError("Cannot pre-embed and use embedding")
        self.embedding_fun = model.get_embedding_fun(batch_first=True) if use_embedding else lambda x: x
        self.export_path = self._build_export_path()
        self.v_model: VGAN_od | VMMD_od | None = None

    def _build_export_path(self) -> str:
        embedding_str = "embedding" if self.pre_embed else "token"
        sl_str = self.sequence_length if self.sequence_length is not None else "(longest)"
        base_dir = os.path.join(
            os.getcwd(),
            'experiments',
            self._get_name(),
            f"{self.version}",
            embedding_str,
            self.generator_class.__name__,
            self.model._model_name,
            f"{self.dataset.name}_sl{sl_str}_vmmd_{self.model.model_name}_e{self.epochs}_pw{self.penalty_weight}_s{self.samples}"
        )
        if self.train:
            base_dir += "_" + datetime.now().strftime("%Y%m%d-%H%M%S")
        return base_dir

    def prepare_data(self):
        processor = DatasetProcessor(
            dataset=self.dataset,
            model=self.model,
            sequence_length=self.sequence_length,
            samples=self.samples,
            pre_embed=self.pre_embed
        )
        self.tokenized_data, self.normalized_data = processor.process()
        self.sequence_length = self.tokenized_data.shape[1]
        self.export_path = self._build_export_path()

    def visualize(self, epoch: int):
        visualizer = CollectiveVisualizer(tokenized_data=self.tokenized_data,
                                          tokenizer=self.model,
                                          vmmd_model=self.v_model,
                                          export_path=self.export_path,
                                          text_visualization=not self.pre_embed)
        visualizer.visualize(epoch=epoch, samples=30)

    @abstractmethod
    def _get_model(self) -> VMMD_od | VGAN_od:
        pass

    @abstractmethod
    def _get_name(self) -> str:
        pass

    def run(self):
        self.prepare_data()

        # Create VMMD model instance
        self.v_model = self._get_model()

        evals = []
        timer = Timer(amount_epochs=self.epochs, export_path=self.export_path)
        model_path = Path(self.export_path) / "models" / "generator_0.pt"

        # If a saved experiment exists, load it.
        if os.path.exists(self.export_path) and model_path.exists():
            self.v_model.load_models(path_to_generator=model_path, ndims=self.sequence_length)
        else:
            for epoch in self.v_model.yield_fit(X=self.normalized_data, yield_epochs=self.yield_epochs,
                                                embedding=self.embedding_fun):
                timer.measure(epoch=epoch)
                timer.pause()

                self.visualize(epoch=epoch)

                eval_result = model_eval(self.v_model, self.normalized_data.cpu())
                evals.append(eval_result)

                timer.resume()

        # Final visualization and myopic check
        self.visualize(epoch=self.epochs)

        p_value_df = self.v_model.check_if_myopic(x_data=self.tokenized_data.cpu().numpy(), count=1000)
        print(f"{self.dataset.name}\n{p_value_df}\n")
        return self.v_model, evals
