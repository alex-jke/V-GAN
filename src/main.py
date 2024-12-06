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
from text.tokenizer.dataset_tokenizer import DatasetTokenizer
from text.visualizer.alpha_visualizer import AlphaVisualizer

from vgan import VGAN
from vmmd import VMMD, model_eval

SUBSPACE_PROBABILITY_COLUMN = 'probability'
SUBSPACE_COLUMN = 'subspace'
VERSION = '0.221'#'0.23'
device = torch.device('cuda:0' if torch.cuda.is_available(
        ) else 'mps:0' if torch.backends.mps.is_available() else 'cpu')

def visualize(tokenized_data: Tensor, tokenizer: Tokenizer, model: VMMD):
    visualizer = AlphaVisualizer(model=model, tokenized_data=tokenized_data, tokenizer=tokenizer)
    visualizer.visualize_html(samples=30)


def pipeline(dataset: Dataset, model: HuggingModel, sequence_length: int, epochs: int, batch_size: int, samples: int,
             train: bool = False, lr: float = 0.001, momentum=0.99, weight_decay=0.04):
    sequence_length = min(sequence_length, model.max_token_length())

    dataset_tokenizer = DatasetTokenizer(tokenizer=model, dataset=dataset, sequence_length=sequence_length)

    # Tensor is of the shape (max_rows, max_length / sequence_length + 1, sequence_length)
    data: Tensor = dataset_tokenizer.get_tokenized_training_data(max_rows=samples)
    first_part = data[:, 0, :]  # Convert to (max_rows, sequence_length) by taking first sequence_length tokens.
    first_part = first_part.to(device)
    first_part = first_part.float()
    #first_part = first_part.cuda().float() # todo: necessary for cuda, if statement wont work.

    embedding = model.get_embedding_fun()

    export_path = None
    path = os.path.join(os.getcwd(), 'text','experiments', f"{dataset.name}_{sequence_length}_vmmd_{model.model_name}"
                                                    f"_{epochs}_{VERSION}")
    export_path = path
    if train:
        export_path = export_path + "_" + datetime.now().strftime("%Y%m%d-%H%M%S")

    vmmd = VMMD_od(path_to_directory=export_path, epochs=epochs, batch_size=batch_size, lr=lr, momentum=momentum,
                weight_decay=weight_decay, seed=None)


    if os.path.exists(export_path):
        vmmd.load_models(path_to_generator=Path(path) / "models" / "generator_0.pt", ndims=sequence_length)
    else:
        vmmd.fit(X=first_part)#, embedding=embedding)

    #subspaces: pd.DataFrame = model_eval(model=vmmd, X_data=first_part.cpu()).sort_values(by='probability',
                                                                                          #ascending=False)
    # subspace df has columns: 'subspace', 'probability',
    #print("Subspace with highest probability:", subspaces.iloc[0])
    #print("Subspace with lowest probability:", subspaces.iloc[-1])

    p_value_df = vmmd.check_if_myopic(x_data=first_part.cpu().numpy(), count=1000)
    print(p_value_df)

    visualize(tokenized_data=first_part, tokenizer=model, model=vmmd)


if __name__ == '__main__':

    pipeline(
        dataset=IMBdDataset(),
        model=GPT2(),
        sequence_length=300,
        epochs=2000,
        batch_size=500,
        samples=10000,
        train=False,
        lr=0.05,
        momentum=0.9,
        weight_decay=0.01
    )



