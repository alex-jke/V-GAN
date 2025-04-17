from abc import ABC, abstractmethod
from pathlib import Path
from time import time
from typing import Tuple, List, Callable, Type

import pandas as pd
import numpy as np
import torch.nn.functional
from pandas import Series
from pyod.models.base import BaseDetector
from sklearn.preprocessing import StandardScaler
from torch import Tensor

from text.Embedding.LLM.huggingmodel import HuggingModel
from text.UI import cli
from text.UI.cli import ConsoleUserInterface
from text.dataset.dataset import Dataset


from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score, f1_score
from typing import List, Tuple
from abc import ABC, abstractmethod

from text.outlier_detection.space.embedding_space import EmbeddingSpace
from text.outlier_detection.space.prepared_data import PreparedData
from text.outlier_detection.space.space import Space
from text.outlier_detection.space.token_space import TokenSpace

not_initizalied_error_msg = "The train data has not been set. Have you called use_embedding or use_tokenized?"

class OutlierDetectionModel(ABC):
    """
    Abstract class for outlier detection models. Specifically for one-class classification.
    """
    def __init__(self, dataset: Dataset, space: Space, base_method: Type[BaseDetector], inlier_label: int | None = None, use_cached: bool = False):
        self.dataset = dataset
        self.use_cached = use_cached
        self.inlier_label = inlier_label
        if inlier_label is None:
            self.inlier_label = self.dataset.get_possible_labels()[0]
        self.ui = cli.get()
        self.method_column = "method"
        self.data: PreparedData | None = None
        self.space: Space = space
        self.device = self.space.model.device
        self.base_method: Type[BaseDetector] = base_method

    @abstractmethod
    def _train(self):
        pass

    def train(self):
        self._start_timer()
        self.data = self.space.transform_dataset(self.dataset, self.use_cached, self.inlier_label)
        assert len(self.data.y_train.unique()) == 1 and int(self.data.y_train.unique()) == self.inlier_label, \
            f"Training data contains other data, than just the inlier data. Expected {self.inlier_label}, got {self.data.y_train.unique()}"
        self._train()

    def predict(self):
        self._predict()
        self._stop_timer()

    @abstractmethod
    def _predict(self):
        pass

    @abstractmethod
    def _get_name(self) -> str:
        pass

    @abstractmethod
    def _get_predictions(self) -> List[float]:
        pass

    def get_space(self) -> str:
        return self.space.name

    @property
    def x_train(self) -> Tensor:
        if self.data is None:
            raise ValueError(not_initizalied_error_msg)
        return self.data.x_train

    @property
    def y_train(self) -> Tensor:
        if self.data is None:
            raise ValueError(not_initizalied_error_msg)
        return self.data.y_train

    @property
    def x_test(self) -> Tensor:
        if self.data is None:
            raise ValueError(not_initizalied_error_msg)
        return self.data.x_test

    @property
    def y_test(self) -> Tensor:
        if self.data is None:
            raise ValueError(not_initizalied_error_msg)
        return self.data.y_test

    def _start_timer(self):
        self.start_time = time()

    def _stop_timer(self):
        self.time_elapsed = time() - self.start_time

    def evaluate(self, output_path: Path = None)-> (pd.DataFrame, pd.DataFrame):
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
            auc = roc_auc_score(y_true=y_test, y_score=decision_function_scores)
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
            self.method_column: [self._get_name()],
            "space": [self.get_space()],
            "auc": [auc],
            "prauc": [prauc],
            "f1": [f1],
            "time_taken": [self.time_elapsed],
            "base": [self.base_method.__name__]
        })

        self.common_parameters = pd.DataFrame({
             "percentage_inlier": [percentage_inlier],
             "percentage_outlier": [percentage_outlier],
            "total_test_samples": [len(y_test)],
            "total_train_samples": [len(self.x_train)],
            "inlier_label": [self.inlier_label],
            "outlier_labels": str([label for label in self.dataset.get_possible_labels() if label != self.inlier_label]),
            "model": [self.space.model.model_name],
            "dataset": [self.dataset.name],
        })

        return self.metrics, self.common_parameters



