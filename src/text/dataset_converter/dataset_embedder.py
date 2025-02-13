import os
from pathlib import Path
from typing import Callable

import pandas as pd
import torch
from pandas import Series
from torch import Tensor

from src.text.Embedding.huggingmodel import HuggingModel
from src.text.UI.cli import ConsoleUserInterface

from src.text.dataset.dataset import Dataset
from text.UI import cli
from text.dataset_converter.dataset_tokenizer import DatasetTokenizer


class DatasetEmbedder:
    def __init__(self, dataset: Dataset, model: HuggingModel):
        self.embedding_function: Callable[[Tensor], Tensor] = model.get_embedding_fun(batch_first=True)
        self.ui = cli.get()
        self.dir_path = Path(os.path.dirname(__file__)) / '..' / 'resources' / dataset.name / "embedding" / f"{model._model_name}"
        self.dataset = dataset
        self.model = model
        self.desired_labels = None
        self.labels: Series | None = None
        self.device = torch.device('cuda:0' if torch.cuda.is_available(
        ) else 'mps:0' if torch.backends.mps.is_available() else 'cpu')

    def embed(self, train: bool, samples: int, labels: list | None = None) -> (Tensor, Tensor):
        """
        Given a tokenized dataset, embeds it.
        :param train : Whether to embed the training or testing dataset.
        :param samples: The number of samples that should be returned
        :param labels: the list of labels to filter for. If left empty, all labels are returned.
        :return: Pair of:
                        - The embedded dataset. A tensor of shape (num_samples, embedding_size).
                        - The labels of the dataset. A tensor of shape (num_samples).
        """
        self.desired_labels = labels
        tokenizer = DatasetTokenizer(tokenizer=self.model, dataset=self.dataset, max_samples=samples)
        path = self.dir_path / f"{'train' if train else 'test'}.csv"
        tokenized_data, labels = tokenizer.get_tokenized_training_data(class_labels=labels) if train else tokenizer.get_tokenized_testing_data()
        embedded = self._get_embedded_tensor(tokenized_data, path, samples)
        return embedded, labels

    def _get_embedded_tensor(self, tokenized_dataset: Tensor, path: Path, samples: int) -> Tensor:
        embedded_dataframe: pd.DataFrame = self._get_embedded_dataframe(tokenized_dataset, path, samples)

        #labels = self.labels.reset_index(drop=True)
        #mask = labels.isin(self.desired_labels) if self.desired_labels is not None else labels.apply(lambda row: True)
        #trimmed_mask = mask[:len(embedded_dataframe)]
        #filtered_df = embedded_dataframe[trimmed_mask]
        trimmed_df = embedded_dataframe.iloc[:samples]

        tensor: Tensor = Tensor(trimmed_df.to_numpy()).to(self.device)
        return tensor


    def _get_embedded_dataframe(self, tokenized_dataset: Tensor, path: Path, samples: int) -> pd.DataFrame:
        dataset = pd.DataFrame()
        start_index = 0
        if path.exists():
            dataset = pd.read_csv(path, header=None)
            start_index = len(dataset)
        else:
            os.makedirs(self.dir_path, exist_ok=True)

        if start_index > len(tokenized_dataset):
            return dataset

        #loaded_labels = self.labels.reset_index(drop=True)[:start_index]
        #mask = loaded_labels.isin(self.desired_labels) if self.desired_labels is not None else loaded_labels.apply(lambda row: True)
        #amount_inliers = len(loaded_labels[mask])

        #if amount_inliers >= samples:
            #return dataset

        step_size = 100
        with self.ui.display():
            for i in range(start_index, len(tokenized_dataset), step_size):
                self.ui.update(data=f"Creating embedded dataset... {i}/{len(tokenized_dataset)}")
                end_index = i + step_size if i + step_size < len(tokenized_dataset) else len(tokenized_dataset)
                remainder_dataset = tokenized_dataset[i:end_index]
                embedded = self.embedding_function(remainder_dataset)
                #embedded_tensor = embedded.permute(1, 0)
                embedded_dataset = pd.DataFrame(embedded.cpu().numpy())
                embedded_dataset.to_csv(path, index=False, mode='a', header=False)
                #loaded_labels = self.labels.reset_index(drop=True)[:end_index]
                #mask = loaded_labels.isin(self.desired_labels) if self.desired_labels is not None else loaded_labels.apply(lambda row: True)
                #amount_inliers = len(loaded_labels[mask])
                #if amount_inliers >= samples:
                    #break

        df = pd.read_csv(path, header=None)
        return df

