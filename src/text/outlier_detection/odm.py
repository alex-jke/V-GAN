from abc import ABC, abstractmethod
from typing import Tuple, List, Callable

import pandas as pd
from torch import Tensor

from text.Embedding.huggingmodel import HuggingModel
from text.dataset.dataset import Dataset


from sklearn.metrics import roc_auc_score
from typing import List, Tuple
from abc import ABC, abstractmethod

not_initizalied_error_msg = "The train data has not been set. Have you called use_embedding or use_tokenized?"

class OutlierDetectionModel(ABC):
    """
    Abstract class for outlier detection models. Specifically for one-class classification.
    """
    def __init__(self, dataset: Dataset, model: HuggingModel, train_size: int, test_size: int, inlier_label: int | None = None ):
        self.dataset = dataset
        self.model = model
        self.train_size = train_size
        self.test_size = test_size
        self.name = self._get_name()
        if inlier_label is None:
            self.inlier_label = self.dataset.get_possible_labels()[0]
        self._x_test = self._y_test = self._x_train = self._y_train = None

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def predict(self):
        pass

    @abstractmethod
    def _get_name(self):
        pass

    @abstractmethod
    def _get_predictions_expected(self) -> Tuple[List[int], List[int]]:
        pass

    @property
    def x_train(self) -> Tensor:
        if self._x_train is None:
            raise ValueError(not_initizalied_error_msg)
        return self._x_train

    @property
    def y_train(self) -> Tensor:
        if self._y_train is None:
            raise ValueError(not_initizalied_error_msg)
        return self._y_train

    @property
    def x_test(self) -> Tensor:
        if self._x_test is None:
            raise ValueError(not_initizalied_error_msg)
        return self._x_test

    @property
    def y_test(self) -> Tensor:
        if self._y_test is None:
            raise ValueError(not_initizalied_error_msg)
        return self._y_test


    def evaluate(self):
        # Get predicted and actual labels
        predicted_inlier, actual_inlier = self._get_predictions_expected()

        # Calculate accuracy
        correct_predictions = [1 if x == y else 0 for x, y in zip(predicted_inlier, actual_inlier)]
        accuracy = sum(correct_predictions) / len(correct_predictions)

        # Calculate true positives, false positives, and false negatives
        true_positives = sum([1 if x == 1 and y == 1 else 0 for x, y in zip(predicted_inlier, actual_inlier)])
        false_positives = sum([1 if x == 1 and y == 0 else 0 for x, y in zip(predicted_inlier, actual_inlier)])
        false_negatives = sum([1 if x == 0 and y == 1 else 0 for x, y in zip(predicted_inlier, actual_inlier)])

        # Calculate recall and precision
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0

        # Calculate AUC
        auc = roc_auc_score(actual_inlier, predicted_inlier)

        # Calculate percentage of inliers and outliers
        percentage_inlier = sum(actual_inlier) / len(actual_inlier) * 100
        percentage_outlier = 100 - percentage_inlier

        # Return evaluation metrics
        return (f"Method: {self.name}\n"
                f"\taccuracy: {accuracy * 100:.2f}%\n"
                f"\trecall: {recall * 100:.2f}%\n"
                f"\tprecision: {precision * 100:.2f}%\n"
                f"\tauc: {auc:.4f}"
                f"\tpercentage inlier: {percentage_inlier:.2f}%\n"
                f"\tpercentage outlier: {percentage_outlier:.2f}%\n"
                f"\ttrue positives: {true_positives}\n"
                f"\tfalse positives: {false_positives}\n"
                f"\tfalse negatives: {false_negatives}\n"
                f"\tamount of samples: {len(actual_inlier)}"
                )

    def use_embedding(self) -> None:
        """Processes and embeds training and testing data.
        This method is used when the classification model should use the embeddings."""
        embedding_func = self.model.get_embedding_fun()

        # Get tokenized data and corresponding labels
        tokenized_train, y_train = self._get_tokenized_with_labels(train=True)
        tokenized_test, y_test = self._get_tokenized_with_labels(train=False)

        # Assign labels
        self._y_train = y_train
        self._y_test = y_test

        # Generate embeddings and reshape
        self._x_train = self._generate_embeddings(tokenized_train, embedding_func)
        self._x_test = self._generate_embeddings(tokenized_test, embedding_func)

    def use_tokenized(self) -> None:
        """Sets the train and test data for the classification model to use the tokenized data."""
        self._x_train, self._y_train = self._get_tokenized_with_labels(train=True)
        self._x_test, self._y_test = self._get_tokenized_with_labels(train=False)

    def _get_tokenized_with_labels(self, train: bool) -> Tuple[Tensor, pd.Series]:
        """
        Tokenizes data and returns both tokenized data and corresponding labels.
        Handles both training and test cases.
        """
        if train:
            data, labels = self.dataset.get_training_data()
            filtered_data, filtered_labels = self._process_training_data(data, labels)
        else:
            data, labels = self.dataset.get_testing_data()
            filtered_data, filtered_labels = self._process_testing_data(data, labels)

        tokenized_data = self._filter_and_tokenize(filtered_data, self.train_size if train else self.test_size)
        return tokenized_data, filtered_labels

    def _process_training_data(self, data: pd.Series, labels: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """Filters training data to samples matching the inlier label used for training. Thus, the Dataset
        now only contains inliers."""
        filtered_data = data[labels == self.inlier_label]
        selected_label = pd.Series([self.inlier_label] * len(filtered_data))
        return filtered_data, selected_label

    def _process_testing_data(self, data: pd.Series, labels: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """Filters testing data to the specified test size."""
        filtered_labels = labels[:self.test_size]
        return data, filtered_labels

    def _filter_and_tokenize(self, data: pd.Series, size: int) -> Tensor:
        """Filters data to specified size and tokenizes it."""
        filtered_data = data[:size]
        return self.model.tokenize_batch(filtered_data.tolist())

    def _generate_embeddings(self, tokenized_data: Tensor, embedding_func: Callable[[Tensor], Tensor]) -> Tensor:
        """Generates embeddings and adjusts tensor dimensions."""
        return embedding_func(tokenized_data).permute(1, 0)

