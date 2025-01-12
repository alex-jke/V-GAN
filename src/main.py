import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from torch import Tensor

from modules.od_module import VMMD_od
from text.Embedding.bert import Bert
from text.Embedding.gpt2 import GPT2
from text.Embedding.huggingmodel import HuggingModel
from text.Embedding.tokenizer import Tokenizer
from text.dataset.ag_news import AGNews
from text.dataset.dataset import Dataset
from text.dataset.emotions import EmotionDataset
from text.dataset.imdb import IMBdDataset
from text.dataset.wikipedia_slim import WikipediaPeopleDataset
from text.tokenizer.dataset_tokenizer import DatasetTokenizer
from text.visualizer.alpha_visualizer import AlphaVisualizer
from text.visualizer.average_alpha_visualizer import AverageAlphaVisualizer
from text.visualizer.random_alpha_visualizer import RandomAlphaVisualizer
from text.visualizer.value_visualizer import ValueVisualizer

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
    average_alpha_visualizer.visualize(samples=30, epoch=epoch)

    random_alpha_visualizer = RandomAlphaVisualizer(**params)
    random_alpha_visualizer.visualize(samples=30, epoch=epoch)

    value_visualizer = ValueVisualizer(**params)
    value_visualizer.visualize(samples=0, epoch=epoch) #todo: maybe plot sample subspaces with probability as transparency
    value_visualizer.visualize(samples=100, epoch=epoch)


def pipeline(dataset: Dataset, model: HuggingModel, sequence_length: int, epochs: int, batch_size: int, samples: int,
             train: bool = False, lr: float = 0.001, momentum=0.99, weight_decay=0.04, version: str = '0',
             penalty_weight: float = 0.0):
    sequence_length = min(sequence_length, model.max_token_length())

    dataset_tokenizer = DatasetTokenizer(tokenizer=model, dataset=dataset, sequence_length=sequence_length)

    # Tensor is of the shape (max_rows, max_length / sequence_length + 1, sequence_length)
    data: Tensor = dataset_tokenizer.get_tokenized_training_data(max_rows=samples)
    first_part = data[:, 0, :]  # Convert to (max_rows, sequence_length) by taking first sequence_length tokens.
    first_part_cpu = first_part.cpu()
    first_part = first_part.to(device)
    if device == 'cuda:0':
        first_part = first_part.float()


    embedding = model.get_embedding_fun()

    export_path = None
    path = os.path.join(os.getcwd(), 'text', 'experiments', f"{version}",
                        f"{dataset.name}_{sequence_length}_vmmd_{model.model_name}"
                        f"_{epochs}_{penalty_weight}")
    export_path = path
    if train:
        export_path = export_path + "_" + datetime.now().strftime("%Y%m%d-%H%M%S")

    vmmd = VMMD_od(path_to_directory=export_path, epochs=epochs, batch_size=batch_size, lr=lr, momentum=momentum,
                   weight_decay=weight_decay, seed=None, penalty_weight=penalty_weight)

    evals = []
    subspaces.append(dataset.name)
    if os.path.exists(export_path):
        vmmd.load_models(path_to_generator=Path(path) / "models" / "generator_0.pt", ndims=sequence_length)
        epochs = -1
        #evals.append(model_eval(vmmd, first_part_cpu))
    else:
        for epoch in vmmd.fit(X=first_part):

            visualize(tokenized_data=first_part, tokenizer=model, model=vmmd, path=export_path, epoch=epoch)
            eval = model_eval(vmmd, first_part_cpu)
            evals.append(eval)
            subspaces.append(eval)

         # , embedding=embedding)
    
    # subspaces: pd.DataFrame = model_eval(model=vmmd, X_data=first_part.cpu()).sort_values(by='probability',
    # ascending=False)
    # subspace df has columns: 'subspace', 'probability',
    # print("Subspace with highest probability:", subspaces.iloc[0])
    # print("Subspace with lowest probability:", subspaces.iloc[-1])

    p_value_df = vmmd.check_if_myopic(x_data=first_part.cpu().numpy(), count=1000)
    print(dataset.name, "\n", p_value_df, "\n")

    for eval in evals:
        print(eval)

    visualize(tokenized_data=first_part, tokenizer=model, model=vmmd, path=export_path, epoch=epochs)


if __name__ == '__main__':
    version = '0.39_sigmoid'
    model = GPT2()
    penalty = 1
    wiki_params = {"model": model, "epochs": 1000, "batch_size": 500, "samples": 5_000, "penalty_weight": penalty,
                   "sequence_length": 1000, "dataset": WikipediaPeopleDataset(), "lr": 0.25, "momentum": 0.9,
                   "weight_decay": 0.005, "version": version, "train": False} #contains 34.000 datapoints

    ag_news_params = {"model": model, "epochs": 1000, "batch_size": 200, "samples": 1000, "penalty_weight": penalty,
                      "sequence_length": 50, "dataset": AGNews(), "lr": 0.5, "momentum": 0.9, "weight_decay": 0.005,
                      "version": version, "train": False}

    imdb_params = {"model": model, "epochs": 1000, "batch_size": 500, "samples": 2000, "penalty_weight": penalty,
                   "sequence_length": 300, "dataset": IMBdDataset(), "lr": 0.5, "momentum": 0.9, "weight_decay": 0.005,
                   "version": version, "train": False}

    emotions_params = {"model": model, "epochs": 1000, "batch_size": 500, "samples": 2000, "penalty_weight": penalty,
                       "sequence_length": 50, "dataset": EmotionDataset(), "lr": 0.5, "momentum": 0.9,
                       "weight_decay": 0.005, "version": version, "train": False} #contains 96.000 datapoints

    pipeline(**ag_news_params)
    pipeline(**emotions_params)
    pipeline(**imdb_params)
    pipeline(**wiki_params)

    print(subspaces)

    # dataset = EmotionDataset()
