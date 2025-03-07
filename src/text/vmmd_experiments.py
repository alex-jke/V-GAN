import os
from abc import abstractmethod
from argparse import ArgumentError
from datetime import datetime
from pathlib import Path
from tokenize import Ignore
from typing import Type

import torch
from torch import Tensor

# === Imports from your modules ===
from models.Generator import GeneratorSigmoid, FakeGenerator, GeneratorUpperSoftmax, GeneratorSigmoidSTE, \
    GeneratorSpectralNorm, GeneratorSigmoidSTEMBD, Generator_big
from modules.od_module import VMMD_od
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
from text.od_experiment import Experiment
from text.v_experiment import VExperiment
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
class VMMDExperiment(VExperiment):
    """
    Encapsulates one experiment for VMMD (data processing, model training/evaluation, visualization).
    """

    def _get_model(self):
        return VMMD_od(path_to_directory=self.export_path, epochs=self.epochs,  batch_size=self.batch_size,  lr=self.lr,
            momentum=self.momentum, weight_decay=self.weight_decay,  seed=self.seed, penalty_weight=self.penalty_weight,
            generator=self.generator_class, print_updates=True, gradient_clipping= self.gradient_clipping
        )

    def _get_name(self) -> str:
        return "VMMD"

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

    experiment = VMMDExperiment(
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
        pre_embed=False
    )
    experiment.run()

def run_everything():
    # List of models and generators to run experiments with.
    models = [DeepSeek1B(), GPT2(), Bert()]
    generators = [GeneratorSigmoidSTE]#, GeneratorUpperSoftmax, GeneratorSigmoid]
    version = '0.46_adam'
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
                        pre_embed=use_embedding,
                        generator_class=generator,
                        **config
                    )
                    experiment.run()

def run_all_datasets():
    datasets = [AGNews(), EmotionDataset(), IMBdDataset(), WikipediaPeopleDataset()] + NLP_ADBench.get_all_datasets()
    models = [GPT2(), Bert(), DeepSeek1B()]
    for pre_embed in [False]:#, True]:
        for model in models:
            for dataset in datasets:
                filtered_data_len = (dataset.get_training_data()[1] == dataset.get_possible_labels()[0]) * 1
                train_length = filtered_data_len.sum()
                #train_length = len(dataset.get_training_data()[1])
                epochs = int(10 ** 6.7 / train_length + 400) * 2
                lr = 1e-5
                epochs = epochs if pre_embed else epochs * 2
                yield_epochs = epochs // 20
                experiment = VMMDExperiment(dataset=dataset, model=model, epochs=epochs, lr=lr, pre_embed=pre_embed,
                                        samples=10_000, version="0.47", yield_epochs=yield_epochs,
                                        penalty_weight=0.1, generator_class=GeneratorSigmoidSTE, gradient_clipping = False,
                                        weight_decay=0.0)
                experiment.run()

def test_embedding():
    dataset = EmotionDataset()
    model = GPT2()
    generator = GeneratorSigmoidSTE
    version = '0.468_embedding_grid+manual_one_hot'
    for lr in range(16,1, -3):
        lr *= 1e-4
        for penalty in range(0,6):
            penalty *= 0.2
            experiment = VMMDExperiment(
                dataset=dataset,
                model=model,
                version=version,
                pre_embed=False,
                generator_class=generator,
                epochs=100,
                batch_size=500,
                samples=2000,
                penalty_weight=penalty,
                lr=lr,
                yield_epochs=5,
                train=True,
                use_embedding=True,
                weight_decay=0.0,
                seed=777
            )
            experiment.run()

# === Main entry point experiments ===
if __name__ == '__main__':
    #generators = [GeneratorUpperSoftmax, GeneratorSigmoidSTE, GeneratorSigmoid, GeneratorSpectralNorm]
    """generators = [GeneratorSigmoidSTE]
    for generator in generators:
        experiment = Experiment(EmotionDataset(), GPT2(), version="0.459_adam+inner_embedding", generator_class=generator, epochs=200, pre_embed=False, use_embedding=True,
                                samples=50_000, lr=5e-5, yield_epochs=5, train=True)
        experiment.run()"""
    #run_all_datasets()
    test_embedding()