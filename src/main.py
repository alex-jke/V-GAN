import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from torch import Tensor

from models.Generator import GeneratorSigmoid, FakeGenerator, GeneratorUpperSoftmax
from modules.od_module import VMMD_od
from text.Embedding.bert import Bert
from text.Embedding.gpt2 import GPT2
from text.Embedding.deepseek import DeepSeek1B
from text.Embedding.gpt2ExtraSubspace import GPT2ExtraSubspaces
from text.Embedding.huggingmodel import HuggingModel
from text.Embedding.tokenizer import Tokenizer
from text.UI.cli import ConsoleUserInterface
from text.dataset.SimpleDataset import SimpleDataset
from text.dataset.ag_news import AGNews
from text.dataset.dataset import Dataset
from text.dataset.emotions import EmotionDataset
from text.dataset.imdb import IMBdDataset
from text.dataset.synthetic_dataset import SyntheticDataset
from text.dataset.wikipedia_slim import WikipediaPeopleDataset
from text.tokenizer.dataset_tokenizer import DatasetTokenizer
from text.visualizer.alpha_visualizer import AlphaVisualizer
from text.visualizer.average_alpha_visualizer import AverageAlphaVisualizer
from text.visualizer.random_alpha_visualizer import RandomAlphaVisualizer
from text.visualizer.subspace_visualizer import SubspaceVisualizer
from text.visualizer.value_visualizer import ValueVisualizer
from text.visualizer.timer import Timer

from vgan import VGAN
from vmmd import VMMD, model_eval

SUBSPACE_PROBABILITY_COLUMN = 'probability'
SUBSPACE_COLUMN = 'subspace'
device = torch.device('cuda:0' if torch.cuda.is_available(
) else 'mps:0' if torch.backends.mps.is_available() else 'cpu')
subspaces = []


def visualize(tokenized_data: Tensor, tokenizer: Tokenizer, model: VMMD, path: str, epoch: int = -1):
    params = {"model": model, "tokenized_data": tokenized_data, "tokenizer": tokenizer, "path": path}

    average_alpha_visualizer = AverageAlphaVisualizer(**params)
    average_alpha_visualizer.visualize(samples=5, epoch=epoch)

    random_alpha_visualizer = RandomAlphaVisualizer(**params)
    random_alpha_visualizer.visualize(samples=5, epoch=epoch)

    value_visualizer = ValueVisualizer(**params)
    value_visualizer.visualize(samples=0, epoch=epoch) #todo: maybe plot sample subspaces with probability as transparency
    value_visualizer.visualize(samples=100, epoch=epoch)

    subspace_visualizer = SubspaceVisualizer(**params)
    subspace_visualizer.visualize(samples=1, epoch=epoch)

def pipeline(dataset: Dataset, model: HuggingModel, sequence_length: int, epochs: int, batch_size: int, samples: int,
             train: bool = False, lr: float = 0.001, momentum=0.99, weight_decay=0.04, version: str = '0',
             penalty_weight: float = 0.0, generator= None, yield_epochs = 100, use_embedding= False):
    sequence_length = min(sequence_length, model.max_token_length())

    dataset_tokenizer = DatasetTokenizer(tokenizer=model, dataset=dataset, max_samples=samples, min_samples=samples)

    # Tensor is of the shape (max_rows, max_length / sequence_length + 1, sequence_length)
    data: Tensor = dataset_tokenizer.get_tokenized_training_data()
    #first_part = data[:, 0, :]  # Convert to (max_rows, sequence_length) by taking first sequence_length tokens.
    first_part = data[:, :sequence_length]
    first_part_cpu = first_part.cpu()
    first_part = first_part.to(device)
    if device.type == 'cuda':
        first_part = first_part.float()
    first_part_normalized = first_part #torch.nn.functional.normalize(first_part, p=2, dim=1)

    embedding = model.get_embedding_fun() if use_embedding else lambda x: x

    path = os.path.join(os.getcwd(), 'text', 'experiments', f"{version}",
                        f"{dataset.name}_{sequence_length}_vmmd_{model.model_name}"
                        f"_{epochs}_{penalty_weight}")
    export_path = path
    if train:
        export_path = export_path + "_" + datetime.now().strftime("%Y%m%d-%H%M%S")

    vmmd = VMMD_od(path_to_directory=export_path, epochs=epochs, batch_size=batch_size, lr=lr, momentum=momentum,
                   weight_decay=weight_decay, seed=None, penalty_weight=penalty_weight, generator=generator)

    evals = []
    subspaces.append(dataset.name)
    if os.path.exists(export_path):
        vmmd.load_models(path_to_generator=Path(path) / "models" / "generator_0.pt", ndims=sequence_length)
        epochs = -1
        #evals.append(model_eval(vmmd, first_part_cpu))
    else:
        timer = Timer(amount_epochs=epochs, export_path=export_path)
        for epoch in vmmd.fit(X=first_part_normalized, yield_epochs= yield_epochs, embedding = embedding):
            timer.measure(epoch=epoch)
            timer.pause()

            visualize(tokenized_data=first_part, tokenizer=model, model=vmmd, path=export_path, epoch=epoch)
            eval = model_eval(vmmd, first_part_normalized.cpu())
            evals.append(eval)
            subspaces.append(eval)

            timer.resume()

    p_value_df = vmmd.check_if_myopic(x_data=first_part.cpu().numpy(), count=1000)
    print(dataset.name, "\n", p_value_df, "\n")

    #for eval in evals:
        #print(eval)

    visualize(tokenized_data=first_part, tokenizer=model, model=vmmd, path=export_path, epoch=epochs)
    return vmmd


def all_fake():
    version = '0.41_fake_no_normalization'
    fake_subspaces = [([1,1, 0,0,0,0], 0.33),
                      ([0,0, 1,1,0,0], 0.33),
                      ([0,0, 0,0,1,1], 0.34)]
    samples = ["an example", "two words", "another one", "a fourth"]


    amount_samples = 2000
    batch_size = amount_samples // 4
    simple_params = {"model": GPT2ExtraSubspaces(3),
                     "epochs": 15, "batch_size": batch_size, "samples": amount_samples,
                     "penalty_weight": 0,
                     "sequence_length": 6,
                     "dataset": SimpleDataset(samples=samples, amount_samples=int(amount_samples / 0.8 + 1)),
                     "generator": FakeGenerator(fake_subspaces),
                     "lr": 0.1, "momentum": 0.9,
                     "weight_decay": 0.005, "version": version, "train": False, "yield_epochs": 10}
    vmmd = pipeline(**simple_params)
    #average_loss = vmmd.train_history['generator_loss'].mean()
    #p_value = vmmd.check_if_myopic(vmmd.X_data.cpu().numpy(), count=1000).iloc[0, 0]



if __name__ == '__main__':
    version = '0.42_embedding'
    generator = GeneratorSigmoid
    model = GPT2()
    penalty = 0
    wiki_params = {"model": model, "epochs": 1000, "batch_size": 500, "samples": 8_0, "penalty_weight": penalty,
                   "sequence_length": 1000, "dataset": WikipediaPeopleDataset(), "lr": 0.25, "momentum": 0.9,
                   "weight_decay": 0.005, "version": version, "train": False} #contains 34.000 datapoints

    ag_news_params = {"model": model, "epochs": 600, "batch_size": 64, "samples": 1024, "penalty_weight": penalty,
                      "sequence_length": 50, "dataset": AGNews(), "lr": 0.5, "momentum": 0.9, "weight_decay": 0.005,
                      "version": version, "train": False, "use_embedding": True, "yield_epochs": 10} # contains 4000 datapoints

    imdb_params = {"model": model, "epochs": 1000, "batch_size": 500, "samples": 2000, "penalty_weight": penalty,
                   "sequence_length": 300, "dataset": IMBdDataset(), "lr": 0.5, "momentum": 0.9, "weight_decay": 0.005,
                   "version": version, "train": False}

    emotions_params = {"model": model, "epochs": 500, "batch_size": 100, "samples": 2000, "penalty_weight": penalty,
                       "sequence_length": 50, "dataset": EmotionDataset(), "lr": 0.05, "momentum": 0.9,
                       "weight_decay": 0.005, "version": version, "train": True, "use_embedding": True,
                       "yield_epochs": 20} #contains 96.000 datapoints

    simple_params = {"model": GPT2ExtraSubspaces(3), "epochs": 4000, "batch_size": 500, "samples": 2000, "penalty_weight": penalty,
                       "sequence_length": 6 ,
                        "dataset": SimpleDataset(samples=["an example", "two words", "another one"], amount_samples=3000),
                        "generator": generator,
                        "lr": 0.01, "momentum": 0.9,
                       "weight_decay": 0.005, "version": version, "train": False}

    #pipeline(**ag_news_params)
    pipeline(**emotions_params)
    #pipeline(**imdb_params)
    #pipeline(**wiki_params)
    #pipeline(**simple_params)
    #all_fake()

    #print(model.detokenize([151646]))

    #print(subspaces)
    # dataset = EmotionDataset()
