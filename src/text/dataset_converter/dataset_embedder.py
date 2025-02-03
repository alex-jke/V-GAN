import os
from pathlib import Path
from typing import Callable

import pandas as pd
import torch
from torch import Tensor

from src.text.Embedding.huggingmodel import HuggingModel
from src.text.UI.cli import ConsoleUserInterface

from src.text.dataset.dataset import Dataset


class DatasetEmbedder:
    def __init__(self, dataset: Dataset, model=HuggingModel):
        self.embedding_function: Callable[[Tensor], Tensor] = model.get_embedding_fun()
        self.ui = ConsoleUserInterface()
        self.dir_path = Path(os.getcwd()) / 'text' / 'resources' / dataset.name / "embedding"
        self.path = self.dir_path / f"{model._model_name}.csv"
        self.device = torch.device('cuda:0' if torch.cuda.is_available(
        ) else 'mps:0' if torch.backends.mps.is_available() else 'cpu')

    def embed(self, tokenized_dataset: Tensor) -> Tensor:
        """
        Given a tokenized dataset, embeds it.
        :param tokenized_dataset: The tokenized dataset to embed. A tensor of shape (num_samples, max_sample_length).
        :return: The embedded dataset. A tensor of shape (num_samples, embedding_size).
        """
        self.ui.update(data="Creating embedded dataset...")
        embedded = self._get_embedded_tensor(tokenized_dataset)

        return embedded

    def _get_embedded_tensor(self, tokenized_dataset: Tensor) -> Tensor:
        embedded_dataframe: pd.DataFrame = self._get_embedded_dataframe(tokenized_dataset)
        tensor: Tensor = Tensor(embedded_dataframe.to_numpy()).to(self.device)
        return tensor


    def _get_embedded_dataframe(self, tokenized_dataset: Tensor) -> pd.DataFrame:
        dataset = pd.DataFrame()
        start_index = 0
        if self.path.exists():
            dataset = pd.read_csv(self.path)
            start_index = len(dataset)
        else:
            os.makedirs(self.dir_path, exist_ok=True)

        if start_index >= len(tokenized_dataset):
            return dataset

        step_size = 100
        for i in range(start_index, len(tokenized_dataset), step_size):
            self.ui.update(data=f"Creating embedded dataset... {i}/{len(tokenized_dataset)}")
            end_index = i + step_size if i + step_size < len(tokenized_dataset) else len(tokenized_dataset)
            remainder_dataset = tokenized_dataset[i:end_index]
            embedded = self.embedding_function(remainder_dataset)
            embedded_tensor = embedded.permute(1, 0)
            embedded_dataset = pd.DataFrame(embedded_tensor.cpu().numpy())
            embedded_dataset.to_csv(self.path, index=False, mode='a', header=False)

        df = pd.read_csv(self.path)
        return df

