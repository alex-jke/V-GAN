import os
from datetime import datetime
from pathlib import Path
from tokenize import Ignore

import torch
from torch import Tensor

# === Imports from your modules ===
from models.Generator import GeneratorSigmoid, FakeGenerator, GeneratorUpperSoftmax, GeneratorSigmoidSTE, \
    GeneratorSpectralNorm
from modules.od_module import VMMD_od
from text.Embedding.bert import Bert
from text.Embedding.gpt2 import GPT2
from text.Embedding.deepseek import DeepSeek1B
from text.Embedding.gpt2ExtraSubspace import GPT2ExtraSubspaces
from text.Embedding.huggingmodel import HuggingModel
from text.Embedding.tokenizer import Tokenizer
from text.dataset.SimpleDataset import SimpleDataset
from text.dataset.ag_news import AGNews
from text.dataset.dataset import Dataset
from text.dataset.emotions import EmotionDataset
from text.dataset.imdb import IMBdDataset
from text.dataset.synthetic_dataset import SyntheticDataset
from text.dataset.wikipedia_slim import WikipediaPeopleDataset
from src.text.dataset_converter.dataset_embedder import DatasetEmbedder
from text.dataset_converter.dataset_processor import DatasetProcessor
from text.dataset_converter.dataset_tokenizer import DatasetTokenizer
from text.visualizer.average_alpha_visualizer import AverageAlphaVisualizer
from text.visualizer.collective_visualizer import CollectiveVisualizer
from text.visualizer.timer import Timer

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
class Experiment:
    """
    Encapsulates one experiment (data processing, model training/evaluation, visualization).
    """
    def __init__(self,
                 dataset: Dataset, model: HuggingModel,  generator_class = GeneratorSigmoidSTE,  sequence_length: int = 50, epochs: int = 2000,
                 batch_size: int = 500,  samples: int = 2000,  penalty_weight: float = 0.5,  lr: float = 0.007,  momentum: float = 0.99, weight_decay: float = 0.04,
                 version: str ="0.0", yield_epochs: int = 200, train: bool = False, use_embedding: bool = False):
        self.dataset = dataset
        self.model = model
        self.generator_class = generator_class
        self.sequence_length = sequence_length
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
        self.use_embedding = use_embedding
        self.device = DEVICE

        self.export_path = self._build_export_path()
        self.vmmd = None

    def _build_export_path(self) -> str:
        embedding_str = "embedding" if self.use_embedding else "token"
        base_dir = os.path.join(
            os.getcwd(),
            'experiments',
            f"{self.version}",
            embedding_str,
            self.generator_class.__name__,
            self.model._model_name,
            f"{self.dataset.name}_sl{self.sequence_length}_vmmd_{self.model.model_name}_e{self.epochs}_pw{self.penalty_weight}_s{self.samples}"
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
            use_embedding=self.use_embedding
        )
        self.tokenized_data, self.normalized_data = processor.process()

    def visualize(self, epoch: int):
        visualizer = CollectiveVisualizer(tokenized_data=self.tokenized_data,
                                tokenizer=self.model,
                                vmmd_model=self.vmmd,
                                export_path=self.export_path,
                                text_visualization=not self.use_embedding)
        visualizer.visualize(epoch=epoch, samples=30)

    def run(self):
        self.prepare_data()

        # Create VMMD model instance
        self.vmmd = VMMD_od(path_to_directory=self.export_path, epochs=self.epochs,  batch_size=self.batch_size,  lr=self.lr,
            momentum=self.momentum, weight_decay=self.weight_decay,  seed=None, penalty_weight=self.penalty_weight,
            generator=self.generator_class, print_updates=True
        )

        evals = []
        timer = Timer(amount_epochs=self.epochs, export_path=self.export_path)
        model_path = Path(self.export_path) / "models" / "generator_0.pt"

        # If a saved experiment exists, load it.
        if os.path.exists(self.export_path) and model_path.exists():
            self.vmmd.load_models(path_to_generator=model_path, ndims=self.sequence_length)
            # (Optionally) perform evaluation here.
        else:
            # Training loop (using a generator interface from VMMD_od)
            for epoch in self.vmmd.yield_fit(X=self.normalized_data, yield_epochs=self.yield_epochs):
                timer.measure(epoch=epoch)
                timer.pause()

                self.visualize(epoch=epoch)

                eval_result = model_eval(self.vmmd, self.normalized_data.cpu())
                evals.append(eval_result)

                timer.resume()

        # Final visualization and myopic check
        self.visualize(epoch=self.epochs)

        p_value_df = self.vmmd.check_if_myopic(x_data=self.tokenized_data.cpu().numpy(), count=1000)
        print(f"{self.dataset.name}\n{p_value_df}\n")
        return self.vmmd, evals

# === Example: Running a fake experiment ===
def run_fake_experiment():
    version = '0.41_fake_no_normalization'
    fake_subspaces = [
        ([1, 1, 0, 0, 0, 0], 0.33),
        ([0, 0, 1, 1, 0, 0], 0.33),
        ([0, 0, 0, 0, 1, 1], 0.34)
    ]
    samples = ["an example", "two words", "another one", "a fourth"]
    amount_samples = 2000
    batch_size = amount_samples // 4

    experiment = Experiment(
        dataset=SimpleDataset(samples=samples, amount_samples=int(amount_samples / 0.8 + 1)),
        model=GPT2ExtraSubspaces(3),
        generator_class=FakeGenerator(fake_subspaces),
        sequence_length=6,
        epochs=15,
        batch_size=batch_size,
        samples=amount_samples,
        penalty_weight=0,
        lr=0.1,
        momentum=0.9,
        weight_decay=0.005,
        version=version,
        yield_epochs=10,
        train=False,
        use_embedding=False
    )
    experiment.run()

def run_everything():
    # List of models and generators to run experiments with.
    models = [DeepSeek1B(), GPT2(), Bert()]
    generators = [GeneratorSigmoidSTE, GeneratorUpperSoftmax, GeneratorSigmoid]
    version = '0.458'
    penalty = 0.5
    weight_decay = 0.04
    lr = 0.007
    momentum = 0.99
    use_embedding_options = [False]

    # Define experiment configurations for different datasets.
    experiments_configs = [
        {
            "dataset": EmotionDataset(),
            "epochs": 2000,
            "batch_size": 500,
            "samples": 20000,
            "penalty_weight": penalty,
            "sequence_length": 50,
            "lr": lr / 3,
            "momentum": momentum,
            "weight_decay": weight_decay,
            "yield_epochs": 100,
            "train": False,
        },
        {
            "dataset": WikipediaPeopleDataset(),
            "epochs": 5000,
            "batch_size": 500,
            "samples": 10000,
            "penalty_weight": penalty,
            "sequence_length": 1000,
            "lr": lr,
            "momentum": momentum,
            "weight_decay": weight_decay,
            "yield_epochs": 100,
            "train": False,
        },
        {
            "dataset": AGNews(),
            "epochs": 8000,
            "batch_size": 256,
            "samples": 1024,
            "penalty_weight": penalty,
            "sequence_length": 50,
            "lr": lr,
            "momentum": momentum,
            "weight_decay": weight_decay,
            "yield_epochs": 400,
            "train": False,
        },
        {
            "dataset": IMBdDataset(),
            "epochs": 6000,
            "batch_size": 500,
            "samples": 2500,
            "penalty_weight": penalty,
            "sequence_length": 300,
            "lr": lr,
            "momentum": momentum,
            "weight_decay": weight_decay,
            "yield_epochs": 200,
            "train": False,
        },
        {
            "dataset": SimpleDataset(samples=["an example", "two words", "another one"], amount_samples=30000),
            "epochs": 4000,
            "batch_size": 500,
            "samples": 10000,
            "penalty_weight": penalty,
            "sequence_length": 6,
            "lr": lr,
            "momentum": momentum,
            "weight_decay": weight_decay,
            "yield_epochs": 100,
            "train": False,
        }
    ]

    # Loop over configurations, models, and generators.
    for use_embedding in use_embedding_options:
        for generator in generators:
            for model in models:
                for config in experiments_configs:
                    experiment = Experiment(
                        model=model,
                        version=version,
                        use_embedding=use_embedding,
                        generator_class=generator,
                        **config
                    )
                    experiment.run()

# === Main entry point experiments ===
if __name__ == '__main__':
    #generators = [GeneratorUpperSoftmax, GeneratorSigmoidSTE, GeneratorSigmoid, GeneratorSpectralNorm]
    generators = [GeneratorSpectralNorm]
    for generator in generators:
        experiment = Experiment(EmotionDataset(), GPT2(), version="0.459_adam", generator_class=generator, epochs=200, use_embedding=True,
                                samples=200_000, lr=1e-5, yield_epochs=20)
        experiment.run()