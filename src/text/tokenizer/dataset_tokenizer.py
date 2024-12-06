import ast
import os
from pathlib import Path
from typing import List

import pandas as pd
import torch
from torch import Tensor, tensor

from ..Embedding.tokenizer import Tokenizer
from ..dataset.dataset import Dataset


class DatasetTokenizer:
    def __init__(self, tokenizer: Tokenizer, dataset: Dataset, sequence_length=300):
        self.tokenizer = tokenizer
        self.tokenizer_name = tokenizer.__class__.__name__
        self.dataset_train_name = "train"
        self.dataset_test_name = "test"
        self.file_extension = ".csv"
        self.resource_path = Path(os.getcwd()) / 'text' / 'resources'
        self.path = Path(os.getcwd()) / 'text' / 'resources' / dataset.name
        self.sequence_length = sequence_length
        self.base_file_name = f"{self.tokenizer_name}_{self.sequence_length}_{dataset.name}"
        self.dataset = dataset
        self.device = torch.device('cuda:0' if torch.cuda.is_available(
        ) else 'mps:0' if torch.backends.mps.is_available() else 'cpu')

    def get_tokenized_training_data(self, max_rows:int = -1, class_label: str = None) -> Tensor:
        """
        Get the tokenized training data. If the tokenized data does not exist, it will be created. This way, the
        tokenized data is only created once.
        :param max_rows: The maximum number of rows to return. If -1, all rows are returned.
        :param class_label: The class label to return. If None, the first class label is returned.
        :return: A three-dimensional tensor of the tokenized training data. The Tensor is of shape
        (max_rows, sequence_length, max_length / sequence_length + 1).
        """
        if class_label is None:
            class_label = self.dataset.get_possible_labels()[0]

        return self._get_tensor(self.dataset_train_name, max_rows, class_label=class_label)

    def get_tokenized_testing_data(self, max_rows:int = -1, class_label: str = None) -> Tensor:
        """
        Get the tokenized testing data. If the tokenized data does not exist, it will be created. This way, the
        tokenized data is only created once.
        :param max_rows: The maximum number of rows to return. If -1, all rows are returned.
        :param class_label: The class label to return. If None, the first class label is returned.
        :return: A three-dimensional tensor of the tokenized testing data. The Tensor is of shape
        (max_rows, max_length / sequence_length + 1, sequence_length).
        """

        if class_label is None:
            class_label = self.dataset.get_possible_labels()[0]

        return self._get_tensor(self.dataset_test_name, max_rows, class_label=class_label)

    def _get_tensor(self, dataset_name, max_rows, class_label: str) -> Tensor:
        dataset = self._get_dataset(dataset_name, class_label)
        max_rows = max_rows if max_rows > 0 else len(dataset)
        data: List[List[List[int]]] = [ast.literal_eval(dataset[self.dataset.x_label_name][i]) for i in range(max_rows)]

        max_lists = max([len(list_of_lists) for list_of_lists in data])

        for i in range(len(data)):
            data[i] = data[i] + [[self.tokenizer.padding_token] * self.sequence_length] * (max_lists - len(data[i]))

        data_tensor = torch.tensor(data, dtype=torch.int).to(self.device)

        return data_tensor

    def _trim_tensor(self, tensor: Tensor) -> Tensor:
        """
        Finds the longest sequence of non-padding tokens and trims the tensor to that length.
        """
        max_sequence_length = 0
        for i in range(tensor.shape[0]):
            sequence_length = 0
            for j in range(tensor.shape[2]):
                if tensor[i, 0, j] != self.tokenizer.padding_token:
                    sequence_length += 1
            max_sequence_length = max(max_sequence_length, sequence_length)
        print(f"Trimming tensor to length {max_sequence_length}")
        return tensor[:, :max_sequence_length]

    def _get_dataset(self, dataset_name: str, class_label: str) -> pd.DataFrame:
        file_path = self.path / f"{self.base_file_name}_{dataset_name}{self.file_extension}"
        if not os.path.exists(file_path):
            self._create_tokenized_dataset(dataset_name)
        y_label = self.dataset.y_label_name
        df = pd.read_csv(file_path)
        filtered_df = df[df[y_label] == class_label].reset_index(drop=True)
        return filtered_df

    def _create_tokenized_dataset(self, dataset_name: str):

        if dataset_name == self.dataset_train_name:
            x, y = self.dataset.get_training_data()
        elif dataset_name == self.dataset_test_name:
            x, y = self.dataset.get_testing_data()
        else:
            raise ValueError(f"Unknown dataset name: {dataset_name}")

        tokenized_x = pd.Series(x).apply(lambda x: self.tokenizer.tokenize(x))
        padding_token = self.tokenizer.padding_token
        def vec_transform(x):
            return [x + [padding_token] * (self.sequence_length - len(x))] \
                if len(x) < self.sequence_length else [x[:self.sequence_length]] + vec_transform(
                x[self.sequence_length:])


        tokenized_padded = tokenized_x.apply(vec_transform)
        tokenized_df = pd.DataFrame({self.dataset.x_label_name: tokenized_padded, self.dataset.y_label_name: y})
        if not os.path.exists(self.resource_path):
            os.mkdir(self.resource_path)
        if not os.path.exists(self.path):
            os.mkdir(self.path)
        tokenized_df.to_csv(self.path / f"{self.base_file_name}_{dataset_name}{self.file_extension}", index=False)
