from abc import ABC, abstractmethod
from pathlib import Path
from time import time
from typing import Tuple, List, Callable

import pandas as pd
import torch.nn.functional
from pandas import Series
from torch import Tensor

from text.Embedding.huggingmodel import HuggingModel
from text.UI import cli
from text.UI.cli import ConsoleUserInterface
from text.dataset.dataset import Dataset


from sklearn.metrics import roc_auc_score
from typing import List, Tuple
from abc import ABC, abstractmethod

from text.dataset_converter.dataset_embedder import DatasetEmbedder
from text.dataset_converter.dataset_tokenizer import DatasetTokenizer

not_initizalied_error_msg = "The train data has not been set. Have you called use_embedding or use_tokenized?"

class OutlierDetectionModel(ABC):
    """
    Abstract class for outlier detection models. Specifically for one-class classification.
    """
    def __init__(self, dataset: Dataset, model: HuggingModel, train_size: int, test_size: int, inlier_label: int | None = None, use_cached: bool = False):
        self.dataset = dataset
        self.model = model
        self.train_size = train_size
        self.test_size = test_size
        self.use_cached = use_cached
        self.name = self._get_name()
        self.inlier_label = inlier_label
        if inlier_label is None:
            self.inlier_label = self.dataset.get_possible_labels()[0]
        self._x_test = self._y_test = self._x_train = self._y_train = None
        self.device = self.model.device
        self.ui = cli.get()

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
    def _get_predictions(self) -> List[float]:
        pass

    @abstractmethod
    def get_space(self):
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

    def start_timer(self):
        self.start_time = time()

    def stop_timer(self):
        self.time_elapsed = time() - self.start_time

    def evaluate(self, output_path: Path = None, print_results = False) -> pd.DataFrame:
        """
        Evaluate the performance of a predictive model against a labeled test dataset.

        This method computes various evaluation metrics for the model including accuracy,
        recall, precision, AUC (Area Under the Curve), and other relevant statistics such
        as percentages of inliers and outliers as well as confusion matrix components.
        Results and metrics are stored as DataFrames, and performance details are printed
        to the console.

        Parameters:
            output_path (Path): The path where evaluation results can potentially
                be saved. This is not used in this implementation. It is included
                to allow subclasses to save extra results to a file.
            print_results (bool): Whether to print the evaluation results to the console.

        Returns:
            pd.DataFrame: A DataFrame summarizing the evaluation metrics including
                model accuracy, precision, recall, AUC, inliers and outliers percentages,
                confusion matrix values, and other associated data.

        Raises:
            ValueError: If the predicted and actual labels are not of the same length.
        """
        # Get predicted and actual labels
        predicted_inlier = self._get_predictions()
        actual_inlier = [0 if x == self.inlier_label else 1 for x in self.y_test]

        # Calculate accuracy
        correct_predictions = [1 if x == y else 0 for x, y in zip(predicted_inlier, actual_inlier)]
        accuracy = sum(correct_predictions) / len(correct_predictions)

        # Calculate true positives, false positives, and false negatives
        true_positives = sum([1 if x == 1 and y == 1 else 0 for x, y in zip(predicted_inlier, actual_inlier)])
        false_positives = sum([1 if x == 1 and y == 0 else 0 for x, y in zip(predicted_inlier, actual_inlier)])
        false_negatives = sum([1 if x == 0 and y == 1 else 0 for x, y in zip(predicted_inlier, actual_inlier)])
        true_negatives = sum([1 if x == 0 and y == 0 else 0 for x, y in zip(predicted_inlier, actual_inlier)])

        # Calculate recall and precision
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0

        # Calculate AUC
        common_len = min(len(predicted_inlier), len(actual_inlier))
        if common_len < len(predicted_inlier) or common_len < len(actual_inlier): #todo: check if this is causing problems
            print(f"Warning: Predicted ({len(predicted_inlier)}) and actual labels ({len(actual_inlier)}) have different lengths. Trimming to common length: {common_len}.")
        predicted_inlier_trimmed = predicted_inlier[:common_len]
        actual_inlier_trimmed = actual_inlier[:common_len]
        auc = roc_auc_score(actual_inlier_trimmed, predicted_inlier_trimmed)

        # Calculate percentage of inliers and outliers
        percentage_inlier = sum(actual_inlier) / len(actual_inlier) * 100
        percentage_outlier = 100 - percentage_inlier

        self.results = pd.DataFrame({
            "actual": actual_inlier_trimmed,
            "predicted": predicted_inlier_trimmed
        })

        metrics = pd.DataFrame({
            "method": [self.name],
            "space": [self.get_space()],
            "accuracy": [accuracy],
            "recall": [recall],
            "precision": [precision],
            "auc": [auc],
            "time_taken": [self.time_elapsed],
            "percentage_inlier": [percentage_inlier],
            "percentage_outlier": [percentage_outlier],
            "true_positives": [true_positives],
            "false_positives": [false_positives],
            "false_negatives": [false_negatives],
            "true_negatives": [true_negatives],
            "total_test_samples": [len(actual_inlier)],
            "total_train_samples": [len(self.x_train)]
        })

        # Return evaluation metrics
        if print_results:
            print(f"Method: {self.name}\n"
                    f"{'='*40}\n"
                    f"  Space:              {self.get_space()}\n"
                    f"{'-'*40}\n"
                    f"  Accuracy:           {accuracy * 100:>7.2f}%\n"
                    f"  Recall:             {recall * 100:>7.2f}%\n"
                    f"  Precision:          {precision * 100:>7.2f}%\n"
                    f"  AUC:                {auc:>7.4f}\n"
                    f"{'-'*40}\n"
                    f"  Percentage Inlier:  {percentage_inlier:>7.2f}%\n"
                    f"  Percentage Outlier: {percentage_outlier:>7.2f}%\n"
                    f"{'-'*40}\n"
                    f"  True Positives:     {true_positives:>7}\n"
                    f"  False Positives:    {false_positives:>7}\n"
                    f"  False Negatives:    {false_negatives:>7}\n"
                    f"  True Negatives:     {true_negatives:>7}\n"
                    f"{'-'*40}\n"
                    f"  Total Samples:      {len(actual_inlier):>7}\n"
                    f"  Time Taken:         {self.time_elapsed:>7.2f} seconds\n"
                    f"{'='*40}")

        return metrics

    def use_embedding(self) -> None:
        """Processes and embeds training and testing data.
        This method is used when the classification model should use the embeddings."""

        embedding_func = self.model.get_embedding_fun(batch_first=True)
        if self.use_cached:
            self.use_embedding_cached()
            return

        # Get tokenized data and corresponding labels
        tokenized_train, y_train = self._get_tokenized_with_labels(train=True)
        tokenized_test, y_test = self._get_tokenized_with_labels(train=False)

        # Assign labels
        self._y_train = y_train
        self._y_test = y_test

        # Generate embeddings and reshape
        #self._x_train = self._generate_embeddings(tokenized_train, embedding_func)
        #self._x_test = self._generate_embeddings(tokenized_test, embedding_func)
        _x_train = embedding_func(tokenized_train)
        _x_test = embedding_func(tokenized_test)
        self._x_train = torch.nn.functional.normalize(_x_train, p=2, dim=1)
        self._x_test = torch.nn.functional.normalize(_x_test, p=2, dim=1)

    def use_tokenized(self) -> None:
        """Sets the train and test data for the classification model to use the tokenized data."""
        _x_train, self._y_train = self._get_tokenized_with_labels(train=True)
        _x_test, self._y_test = self._get_tokenized_with_labels(train=False)

        train_length = _x_train.shape[1]
        test_length = _x_test.shape[1]

        pad_length = max(train_length, test_length)

        self._x_train = torch.nn.functional.pad(_x_train, (0, pad_length - train_length), value=self.model.padding_token)
        self._x_test = torch.nn.functional.pad(_x_test, (0, pad_length - test_length), value=self.model.padding_token)

    def _get_tokenized_with_labels(self, train: bool) -> Tuple[Tensor, Tensor]:
        """
        Tokenizes data and returns both tokenized data and corresponding labels.
        Handles both training and test cases.
        """
        if self.use_cached:
            return self._get_tokenized_with_labels_cached(train)

        if train:
            data, labels = self.dataset.get_training_data()
            filtered_data, filtered_labels = self._process_training_data(data, labels)
        else:
            data, labels = self.dataset.get_testing_data()
            filtered_data, filtered_labels = self._process_testing_data(data, labels)

        filtered_labels_tensor = Tensor(filtered_labels.tolist()).int().to(self.model.device)
        tokenized_data = self._filter_and_tokenize(filtered_data, self.train_size if train else self.test_size)
        return tokenized_data, filtered_labels_tensor

    def _process_training_data(self, data: pd.Series, labels: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """Filters training data to samples matching the inlier label used for training. Thus, the Dataset
        now only contains inliers."""
        filtered_data = data[labels == self.inlier_label] # Todo: add the discarded outliers to the test set
        selected_label = pd.Series([self.inlier_label] * self.train_size)
        return filtered_data, selected_label

    def _process_testing_data(self, data: pd.Series, labels: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """Filters testing data to the specified test size."""
        filtered_labels = labels[:self.test_size]
        return data, filtered_labels

    def _filter_and_tokenize(self, data: pd.Series, size: int) -> Tensor:
        """Truncates data to specified size and tokenizes it."""
        length = min(size, len(data))
        filtered_data = data[:length]
        return self.model.tokenize_batch(filtered_data.tolist())

    def _get_tokenized_with_labels_cached(self, train: bool) -> Tuple[Tensor, Tensor]:
        dataset_tokenizer = DatasetTokenizer(self.model, self.dataset, max_samples=self.train_size if train else self.test_size)
        if train:
            tokenized, labels = dataset_tokenizer.get_tokenized_training_data(class_labels=[self.inlier_label])
            length = min(self.train_size, len(tokenized))
            return tokenized[:length], Tensor([self.inlier_label] * length).int().to(self.model.device)
        else:
            tokenized, labels = dataset_tokenizer.get_tokenized_testing_data()
            length = min(self.test_size, len(tokenized))
            return tokenized[:length], labels[:length]

    def use_embedding_cached(self):

        dataset_embedder = DatasetEmbedder(self.dataset, self.model)

        self._x_train, self._y_train = dataset_embedder.embed(train=True, samples=self.train_size, labels=[self.inlier_label])
        #mask = _y_train == self.inlier_label
        #_x_train_trimmed = _x_train[mask]
        #length = min(self.train_size, len(_x_train_trimmed))
        #self._x_train = _x_train_trimmed[:length]
        #self._y_train = _y_train[mask][:length]

        self. _x_test,self. _y_test = dataset_embedder.embed(train=False, samples=self.test_size)
        #length = min(self.test_size, len(_x_test))
        #self._x_test = _x_test[:length]
        #self._y_test = _y_test[:length]




