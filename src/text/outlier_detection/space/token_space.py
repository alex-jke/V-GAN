from abc import ABC
from typing import Tuple, Callable

import pandas as pd
import torch
from torch import Tensor

from text.dataset.dataset import Dataset
from text.outlier_detection.space.prepared_data import PreparedData
from text.outlier_detection.space.space import Space


class TokenSpace(Space):
    """
    This class represents the token space that the outlier detection models should operate in.
    """

    @property
    def name(self):
        return "Token-space"

    def transform_dataset(self, dataset: Dataset, use_cached: bool, inlier_label) -> PreparedData:
        _x_train, y_train = self._get_tokenized_with_labels(train=True, dataset=dataset, inlier_label=inlier_label)
        _x_test, y_test = self._get_tokenized_with_labels(train=False, dataset=dataset, inlier_label=inlier_label)

        train_length = _x_train.shape[1]
        test_length = _x_test.shape[1]

        pad_length = max(train_length, test_length)

        x_train = torch.nn.functional.pad(_x_train, (0, pad_length - train_length),
                                                value=self.model.padding_token).int()
        x_test = torch.nn.functional.pad(_x_test, (0, pad_length - test_length),
                                               value=self.model.padding_token).int()

        return PreparedData(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, space=self.name)

    def _get_tokenized_with_labels(self, train: bool, dataset: Dataset, inlier_label) -> Tuple[Tensor, Tensor]:
        """
        Tokenizes data and returns both tokenized data and corresponding labels.
        Handles both training and test cases.
        """

        get_data = dataset.get_training_data if train else dataset.get_testing_data
        filtered_data, filtered_labels = self._process_data(get_data, inlier_label=inlier_label)

        filtered_labels_tensor = Tensor(filtered_labels.tolist()).int().to(self.model.device)
        tokenized_data = self._filter_and_tokenize(filtered_data, self.train_size if train else self.test_size)
        if len(tokenized_data) != len(filtered_labels_tensor):
            raise ValueError(f"The amount of labels and the amount of tokenized samples do not match. labels:  {len(filtered_labels_tensor)} != {len(tokenized_data)} (samples)")
        return tokenized_data, filtered_labels_tensor

    def _process_data(self, get_data: Callable[[], Tuple[pd.Series, pd.Series]], inlier_label) -> Tuple[pd.Series, pd.Series]:
        """Filters training data to samples matching the inlier label used for training. Thus, the Dataset
        now only contains inliers."""
        data, labels = get_data()
        filtered_data = data[labels == inlier_label] # Todo: add the discarded outliers to the test set
        selected_label = labels[labels == inlier_label]
        return filtered_data, selected_label

    def _filter_and_tokenize(self, data: pd.Series, size: int) -> Tensor:
        """Truncates data to specified size and tokenizes it."""
        length = min(size, len(data)) if size > 0 else len(data)
        filtered_data = data[:length + 1]
        return self.model.tokenize_batch(filtered_data.tolist())