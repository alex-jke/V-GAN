from abc import ABC, abstractmethod
from pathlib import Path
from time import time
from typing import Tuple, List, Callable

import pandas as pd
import numpy as np
import torch.nn.functional
from pandas import Series
from sklearn.preprocessing import StandardScaler
from torch import Tensor

from text.Embedding.huggingmodel import HuggingModel
from text.UI import cli
from text.UI.cli import ConsoleUserInterface
from text.dataset.dataset import Dataset


from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score, f1_score
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
        self.method_column = "method"

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

    def evaluate(self, output_path: Path = None )-> (pd.DataFrame, pd.DataFrame):
        """
        Evaluate the performance of a predictive model against a labeled test dataset.

        This method computes various evaluation metrics for the model including
         ROC AUC (Area Under the Curve), PRAUC, F1 and other relevant statistics such
        as percentages of inliers and outliers as well as confusion matrix components.
        Results and metrics are stored as DataFrames, and performance details can be printed
        to the console.

        Parameters:
            output_path (Path): The path where evaluation results can potentially
                be saved. This is not used in this implementation. It is included
                to allow subclasses to save extra results to a file.

        Returns:
            A pair of:
                pd.DataFrame: A DataFrame summarizing the evaluation metrics including
                    model accuracy, precision, recall, AUC, inliers and outliers percentages,
                    confusion matrix values, and other associated data.
                pd.DataFrame: A DataFrame containing common parameters used in the evaluation
                    such as the dataset name, model name, inlier label, and other relevant
                    information.
        """
        # Get predicted and actual labels
        decision_function_scores = self._get_predictions()
        y_test = [0 if x == self.inlier_label else 1 for x in self.y_test]

        # Calculate AUC
        common_len = min(len(decision_function_scores), len(y_test))
        if common_len < len(decision_function_scores) or common_len < len(y_test): #todo: check if this is causing problems
            print(f"Warning: Predicted ({len(decision_function_scores)}) and actual labels ({len(y_test)}) have different lengths. Trimming to common length: {common_len}.")
            decision_function_scores = decision_function_scores[:common_len]
            y_test = y_test[:common_len]
        try:
            auc = roc_auc_score(y_test, decision_function_scores)
        except ValueError as e:
            print(e)
            raise e
        prauc = average_precision_score(y_test, decision_function_scores)
        f1 = f1_score(y_test, (decision_function_scores > np.quantile(decision_function_scores, .80)) * 1)


        # Calculate percentage of inliers and outliers
        percentage_inlier = sum(y_test) / len(y_test) * 100
        percentage_outlier = 100 - percentage_inlier

        self.results = pd.DataFrame({
            "actual": y_test,
            "predicted": decision_function_scores,
        })

        self.metrics = pd.DataFrame({
            self.method_column: [self.name],
            "space": [self.get_space()],
            "auc": [auc],
            "prauc": [prauc],
            "f1": [f1],
            "time_taken": [self.time_elapsed],
        })

        self.common_parameters = pd.DataFrame({
             "percentage_inlier": [percentage_inlier],
             "percentage_outlier": [percentage_outlier],
            "total_test_samples": [len(y_test)],
            "total_train_samples": [len(self.x_train)],
            "inlier_label": [self.inlier_label],
            "outlier_labels": str([label for label in self.dataset.get_possible_labels() if label != self.inlier_label]),
            "model": [self.model.model_name],
            "dataset": [self.dataset.name],
        })

        return self.metrics, self.common_parameters

    def use_embedding(self) -> None:
        """Processes and embeds training and testing data.
        This method is used when the classification model should use the embeddings."""

        if self.use_cached:
            self.use_embedding_cached()
            return

        # Get tokenized data and corresponding labels
        tokenized_train, y_train = self._get_tokenized_with_labels(train=True)
        tokenized_test, y_test = self._get_tokenized_with_labels(train=False)

        # Assign labels
        self._y_train = y_train
        self._y_test = y_test

        self._x_train = self.__prepare_embedding(tokenized_train)
        self._x_test = self.__prepare_embedding(tokenized_test)

    def __prepare_embedding(self, tokenized: Tensor) -> Tensor:
        embedding_func = self.model.get_embedding_fun(batch_first=True)
        embedded = embedding_func(tokenized)
        means = embedded.mean(1, keepdim=True)
        stds = embedded.std(1, keepdim=True)
        standardized = (embedded - means) / stds
        normalized = torch.nn.functional.normalize(standardized, p=2, dim=1)
        return normalized

    def use_tokenized(self) -> None:
        """Sets the train and test data for the classification model to use the tokenized data."""
        _x_train, self._y_train = self._get_tokenized_with_labels(train=True)
        _x_test, self._y_test = self._get_tokenized_with_labels(train=False)

        train_length = _x_train.shape[1]
        test_length = _x_test.shape[1]

        pad_length = max(train_length, test_length)

        self._x_train = torch.nn.functional.pad(_x_train, (0, pad_length - train_length), value=self.model.padding_token).int()
        self._x_test = torch.nn.functional.pad(_x_test, (0, pad_length - test_length), value=self.model.padding_token).int()

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

        self._x_test,self._y_test = dataset_embedder.embed(train=False, samples=self.test_size)
        #length = min(self.test_size, len(_x_test))
        #self._x_test = _x_test[:length]
        #self._y_test = _y_test[:length]




