import ast
import json
import os
import sys
from itertools import takewhile
from pathlib import Path
from typing import List

import pandas as pd
import torch
from IPython.core.guarded_eval import list_non_mutating_methods
from numpy import ndarray
from pandas import Series
from torch import Tensor, tensor

from ..Embedding.tokenizer import Tokenizer
from ..UI import cli
from ..UI.cli import ConsoleUserInterface
from ..dataset.dataset import Dataset


class DatasetTokenizer:
    def __init__(self, tokenizer: Tokenizer, dataset: Dataset, max_samples: int = -1):
        self.tokenizer = tokenizer
        self.tokenizer_name = tokenizer.__class__.__name__
        self.dataset_train_name = "train"
        self.dataset_test_name = "test"
        self.file_extension = ".csv"
        self.resource_path = Path(os.path.dirname(__file__)) / '..' / 'resources'
        self.path = self.resource_path / dataset.name / "tokenized"
        self.base_file_name = f"{self.tokenizer_name}_{dataset.name}"
        self.max_samples = max_samples
        self.dataset_path: Path | None = None
        #self.sequence_length = sequence_length
        #self.base_file_name = f"{self.tokenizer_name}_{self.sequence_length}_{dataset.name}"
        self.dataset = dataset
        self.padding_token = self.tokenizer.padding_token
        self.device = torch.device('cuda' if torch.cuda.is_available(
        ) else 'mps:0' if torch.backends.mps.is_available() else 'cpu')
        self.class_label = None
        self.ui = cli.get()

    def get_tokenized_training_data(self, class_labels: List[str] = None) -> (Tensor, Tensor):
        """
        Get the tokenized training data. If the tokenized data does not exist, it will be created. This way, the
        tokenized data is only created once.
        :param class_label: The class label to return. If None, the first class label is returned.
        :return: A pair of:    - The tokenized training data as a tensor. The tensor is of shape (max_rows, max_sample_length).
                                        - The labels of the dataset. The tensor is of shape (max_rows).
        """

        self.dataset_path = self.path / f"{self.base_file_name}_{self.dataset_train_name}_s{self.max_samples}_l{class_labels}_{self.file_extension}"
        return self._get_tensor(self.dataset_train_name, class_labels=class_labels)

    def get_tokenized_testing_data(self, class_labels: List[str] = None) -> (Tensor, Tensor):
        """
        Get the tokenized testing data. If the tokenized data does not exist, it will be created. This way, the
        tokenized data is only created once.        :param class_label: The class label to return. If None, the first class label is returned.
        :return: A pair of:    - The tokenized testing data as a tensor. The tensor is of shape (max_rows, max_sample_length).
                                        - The labels of the dataset. The tensor is of shape (max_rows).

        """
        self.dataset_path = self.path / f"{self.base_file_name}_{self.dataset_test_name}__s{self.max_samples}_l{class_labels}_{self.file_extension}"
        return self._get_tensor(self.dataset_test_name, class_labels=class_labels)

    def _get_tensor(self, dataset_name, class_labels: List[str] | None) -> (Tensor, Tensor):
        """
        Get the tokenized data as a tensor.
        :param dataset_name: The name of the dataset to get the tokenized data for.
        :param class_labels: The class labels to filter for. If None, all class labels are returned.
        :return: A pair of:     - The tokenized data as a tensor. The tensor is of shape (max_rows, max_sample_length).
                                        - The labels of the dataset. The tensor is of shape (max_rows).
        """
        class_labels: list = self.dataset.get_possible_labels() if class_labels is None else class_labels
        dataset = self._get_dataset(dataset_name, class_labels)
        max_rows = self.max_samples if self.max_samples > 0 else len(dataset)
        max_rows = min(max_rows, len(dataset))
        #data = dataset[self.dataset.x_label_name].apply(lambda x: ast.literal_eval(x)).tolist()
        #data = [ast.literal_eval(x) for x in dataset[self.dataset.x_label_name]]
        #data = [json.loads(x) for x in dataset[self.dataset.x_label_name]]
        data = []
        with self.ui.display():
            for x in dataset[self.dataset.x_label_name]:
                self.ui.update(f"reading tokenized dataset {len(data)} / {max_rows}")
                try:
                    json_list = json.loads(x)
                    data.append(json_list)
                except json.JSONDecodeError:
                    print(f"Warning: Skipping invalid JSON in row: {x}")  # Debugging
                    #json_list = []

        max_length = max([len(token_list) for token_list in data])

        for i in range(len(data)):
            data[i] = data[i] + [self.padding_token] * (max_length - len(data[i]))

        data_tensor = torch.tensor(data, dtype=torch.int).to(self.device)

        trimmed_tensor = self._trim_tensor(data_tensor)

        return trimmed_tensor, torch.tensor(dataset[self.dataset.y_label_name][:max_rows].tolist()).to(self.device)


    def _trim_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Finds the longest sequence of non-padding tokens across rows and
        trims the tensor to that length.
        """
        padding_token = self.tokenizer._padding_token
        # Create a boolean mask: True where token is not padding.
        mask = tensor != padding_token
        # Flip the mask along the sequence dimension (assumed dim=1).
        mask_rev = torch.flip(mask, dims=[1])

        # For each row, check if there is any non-padding token.
        non_padding_exists = mask_rev.any(dim=1)
        L = tensor.size(1)
        # For each row, find the index of the first non-padding token in the reversed row.
        # (Note: If all tokens are padding, argmax returns 0; we fix that below.)
        first_non_padding = mask_rev.float().argmax(dim=1)
        # For rows that are all padding, set the index to L (so that effective length becomes 0).
        first_non_padding = torch.where(
            non_padding_exists,
            first_non_padding,
            torch.full_like(first_non_padding, L)
        )
        # Effective length per row is: total length - number of trailing padding tokens.
        effective_lengths = L - first_non_padding
        # Determine the maximum effective (non-padding) length.
        max_sequence_length = effective_lengths.max().item()

        trimmed = tensor[:, :max_sequence_length]
        return trimmed

    def _get_dataset(self, dataset_name: str, class_labels: list) -> pd.DataFrame:
        if not os.path.exists(self.dataset_path):
            self._create_tokenized_dataset(dataset_name, class_labels)
        df = pd.read_csv(self.dataset_path)
        samples = self.max_samples if self.max_samples > 0 else len(self.dataset.get_testing_data()[0])
        if df[self.dataset.y_label_name].isin(class_labels).sum() < samples:
            self._create_tokenized_dataset(dataset_name, class_labels)
        df = pd.read_csv(self.dataset_path)
        y_label = self.dataset.y_label_name
        filtered_df = df[df[y_label].isin(class_labels)].reset_index(drop=True)
        filtered_df = filtered_df.iloc[:self.max_samples] if self.max_samples > 0 else filtered_df
        #if self.min_samples is not None and len(filtered_df) < self.min_samples:
            #self.max_samples = self.max_samples +(self.min_samples - len(filtered_df)) * (self.max_samples / len(filtered_df))
            #return self._get_dataset(dataset_name, class_label) # This could lead to an infinite loop
        return filtered_df


    def _tokenize(self, x: Series, y: Series, dataset_type: str, class_labels: list) -> pd.DataFrame:
        #length = len(x) if self.max_samples < 0 else self.max_samples
        counter = Counter()
        path = self.path / f"{self.base_file_name}_temp_{dataset_type}.csv"
        max_samples = self.max_samples if self.max_samples > 0 else len(x)

        # x = x[:int(length / 50000)]
        # length = len(x)
        def tokenize(x):
            # if counter.counter % one_percent == 0:
            self.ui.update(f"\r{counter.counter} tokenized")
            counter.increase_counter()
            tokenized: Tensor = self.tokenizer.tokenize(x)
            # print(tokenized)
            return tokenized.tolist()

        # Save the tokenized data to a csv file at path every 100 samples to avoid data loss on crash. If the file
        # already exists, for the given amount of samples, the data is loaded from the file.
        start_row = 0
        amount_inliers = 0
        tokenized_x = pd.DataFrame()
        if os.path.exists(path):
            tokenized_df = pd.read_csv(path)
            tokenized_x = tokenized_df[self.dataset.x_label_name]
            start_row = len(tokenized_x)
            counter = Counter(start_row)

            amount_inliers = tokenized_df[self.dataset.y_label_name].isin(class_labels).sum()

        length = len(x)
        with self.ui.display():
            for i in range(start_row, length, 100):
                self.ui.update(f"tokenizing {i} / {length}")
                end_index = i + 100 if i + 100 < length else length
                newly_tokenized = pd.Series(x[i:end_index]).apply(tokenize)
                new_labels = y[i:end_index]
                tokenized_df = pd.DataFrame({self.dataset.x_label_name: newly_tokenized, self.dataset.y_label_name: new_labels})

                amount_inliers += tokenized_df[self.dataset.y_label_name].isin(class_labels).sum()

                if i == 0:
                    if not os.path.exists(path):
                        os.makedirs(self.path, exist_ok=True)
                    tokenized_df.to_csv(path, index=False)
                else:
                    tokenized_df.to_csv(path, mode='a', header=False, index=False)
                if amount_inliers >= max_samples:
                    break

        fully_tokenized = pd.read_csv(path)
        #return pd.DataFrame({self.dataset.x_label_name: fully_tokenized})
        return fully_tokenized

    def _create_tokenized_dataset(self, dataset_name: str, class_labels: list):
        if dataset_name == self.dataset_train_name:
            x, y = self.dataset.get_training_data()
        elif dataset_name == self.dataset_test_name:
            x, y = self.dataset.get_testing_data()
        else:
            raise ValueError(f"Unknown dataset name: {dataset_name}")

        with self.ui.display():
            tokenized_x = self._tokenize(x, y, dataset_name, class_labels)

        # As a series of length 1 is passed to the apply function, the tokenized data for that row is the first element
        # Furthermore, the tokenized data is a string of the form "[1, 2, 3, 4, 5]" which is split by the comma
        max_token_length = tokenized_x.apply(lambda token_list: len(token_list.iloc[0].split(",")), axis=1).max()

        def vec_transform(x):
            tokens_amount = len(x.iloc[0].split(","))
            if tokens_amount > max_token_length:
                transformed = x.iloc[0][:-1] + ", " + str([self.padding_token] * (max_token_length - tokens_amount))[1:]
            else:
                transformed = x.iloc[0]
            return transformed


        tokenized_padded = tokenized_x.apply(lambda row: vec_transform(row), axis=1).reset_index(drop=True)
        y_trimmed = y[:len(tokenized_padded)].reset_index(drop=True)
        #tokenized_df = pd.DataFrame({self.dataset.x_label_name: tokenized_padded, self.dataset.y_label_name: y_trimmed})
        tokenized_df = pd.DataFrame({self.dataset.x_label_name: tokenized_padded})
        tokenized_df[self.dataset.y_label_name] = y_trimmed
        if not os.path.exists(self.resource_path):
            os.mkdir(self.resource_path)
        if not os.path.exists(self.path):
            os.mkdir(self.path)
        tokenized_df.to_csv(self.dataset_path, index=False)

class Counter:
    def __init__(self, start_value: int = 0):
        self.counter = start_value

    def increase_counter(self):
        self.counter = self.counter + 1
