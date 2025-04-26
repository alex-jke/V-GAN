from abc import ABC
from typing import Tuple, Callable, Optional, Dict

import numpy
import pandas as pd
import torch
from numpy import ndarray
from torch import Tensor

from text.UI import cli
from text.dataset.dataset import Dataset
from text.outlier_detection.space.prepared_data import PreparedData
from text.outlier_detection.space.space import Space

ui = cli.get()
class TokenSpace(Space):
    """
    This class represents the token space that the outlier detection models should operate in.
    """

    def __init__(self, **params):
        self.cache: Dict[str, PreparedData] = {}
        super().__init__(**params)

    @property
    def name(self):
        return "Token-space"

    def get_tokenized(self, dataset: Dataset, inlier_label) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        _x_train, y_train = self._get_tokenized_with_labels(train=True, dataset=dataset, inlier_label=[inlier_label])
        _x_test, y_test = self._get_tokenized_with_labels(train=False, dataset=dataset,
                                                          inlier_label=dataset.get_possible_labels())

        train_length = _x_train.shape[1]
        test_length = _x_test.shape[1]

        pad_length = max(train_length, test_length)

        x_train = torch.nn.functional.pad(_x_train, (0, pad_length - train_length),
                                          value=self.model.padding_token).int()
        x_test = torch.nn.functional.pad(_x_test, (0, pad_length - test_length),
                                         value=self.model.padding_token).int()

        return x_train, y_train, x_test, y_test

    def embed_tokenized(self, tokenized) -> Tensor:
        embeddings = []
        assert len(tokenized.shape) == 2, f"Tokenized data should be 2D (batch, sequence length), but got {tokenized.shape}"
        with ui.display():
            for i in range(tokenized.shape[0]):
                ui.update(f"Embedding {i}/{tokenized.shape[0]}")
                sample = tokenized[i]
                embedded_tokens = self.model.fully_embed_tokenized(sample)
                embedding = embedded_tokens.mean(dim=0)
                embeddings.append(embedding)
        return torch.stack(embeddings)

    def transform_dataset(self, dataset: Dataset, use_cached: bool, inlier_label, mask: Optional[ndarray]) -> PreparedData:

        token_x_train, y_train, token_x_test, y_test = self.get_tokenized(dataset, inlier_label)

        id = dataset.name + str(use_cached) + str(inlier_label)
        if mask is None and use_cached and id in self.cache:
            return self.cache[id]

        if mask is not None and len(mask.shape) != 1:
            raise NotImplementedError("TokenSpace does not support masks with more than 1 dimension. Please use a mask of shape (sequence_length).")

        if mask is not None:
            token_x_train = token_x_train[:, :mask.shape[0]]
            token_x_test = token_x_test[:, :mask.shape[0]]

            assert mask.dtype == numpy.int64, f"Mask should be of type int, but got {mask.dtype}"

            bool_mask = (mask == 1)

            with ui.display():
                ui.update("train set")
                token_x_train = token_x_train[:,bool_mask]
                ui.update("test set")
                token_x_test = token_x_test[:,bool_mask]

        x_train = self.embed_tokenized(token_x_train)
        x_test = self.embed_tokenized(token_x_test)

        prepared_data = PreparedData(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, space=self.name, inlier_labels=[inlier_label])

        if mask is None and use_cached:
            print("Cached")
            self.cache[id] = prepared_data

        return prepared_data

    def _get_tokenized_with_labels(self, train: bool, dataset: Dataset, inlier_label: list) -> Tuple[Tensor, Tensor]:
        """
        Tokenizes data and returns both tokenized data and corresponding labels.
        Handles both training and test cases.
        """

        get_data = dataset.get_training_data if train else dataset.get_testing_data
        filtered_data, filtered_labels = self._process_data(get_data, inlier_label=inlier_label, amount=self.train_size if train else self.test_size)

        filtered_labels_tensor = Tensor(filtered_labels.tolist()).int().to(self.model.device)
        tokenized_data = self._filter_and_tokenize(filtered_data, self.train_size if train else self.test_size)
        if len(tokenized_data) != len(filtered_labels_tensor):
            raise ValueError(f"The amount of labels and the amount of tokenized samples do not match. labels:  {len(filtered_labels_tensor)} != {len(tokenized_data)} (samples)")
        return tokenized_data, filtered_labels_tensor

    def _process_data(self, get_data: Callable[[], Tuple[pd.Series, pd.Series]], inlier_label: list, amount: int) -> Tuple[pd.Series, pd.Series]:
        """Filters training data to samples matching the inlier label used for training. Thus, the Dataset
        now only contains inliers."""
        data, labels = get_data()
        filtered_data = data[labels.isin(inlier_label)] # Todo: add the discarded outliers to the test set
        selected_label = labels[labels.isin(inlier_label)]
        return filtered_data[:amount], selected_label[: amount]

    def _filter_and_tokenize(self, data: pd.Series, size: int) -> Tensor:
        """Truncates data to specified size and tokenizes it."""
        length = min(size, len(data)) if size > 0 else len(data)
        filtered_data = data[:length]
        tokenized = self.model.tokenize_batch(filtered_data.tolist())
        return tokenized