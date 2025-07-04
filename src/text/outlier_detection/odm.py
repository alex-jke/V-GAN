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
from text.outlier_detection.space_type import SpaceType

not_initizalied_error_msg = "The train data has not been set. Have you called use_embedding or use_tokenized?"

METHOD_COL = "method"
SPACE_COL = "space"
SPACE_TYPE_COL = "space_type"
AUC_COL = "auc"
PRAUC_COL = "prauc"
F1_COL = "f1"
TIME_TAKEN_COL = "time_taken"
BASE_COL = "base"

PERCENTAGE_INLIER_COL = "percentage_inlier"
PERCENTAGE_OUTLIER_COL = "percentage_outlier"
TOTAL_TEST_SAMPLES_COL = "total_test_samples"
TOTAL_TRAIN_SAMPLES_COL = "total_train_samples"
INLIER_LABEL_COL = "inlier_label"
OUTLIER_LABEL_COL = "outlier_labels"
MODEL_COL = "model"
DATASET_COL = "dataset"


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
        self.method_column = METHOD_COL
        self._data: PreparedData | None = None
        self.space: Space = space
        self.device = self.space.model.device
        self.base_method: Type[BaseDetector] = base_method

    @abstractmethod
    def get_space_type(self) -> SpaceType:
        pass

    @abstractmethod
    def _train(self):
        pass

    @property
    def data(self) -> PreparedData:
        if self._data is None:
            self._data = self.space.transform_dataset(self.dataset, self.use_cached, self.inlier_label, None)
            assert len(self.data.y_train.unique()) == 1 and int(self._data.y_train.unique()) == self.inlier_label, \
                f"Training data contains other data, than just the inlier data. Expected {self.inlier_label}, got {self._data.y_train.unique()}"
        return self._data

    def train(self):
        self._start_timer()
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
        percentage_outlier = sum(y_test) / len(y_test) * 100
        percentage_inlier = 100 - percentage_outlier

        self.results = pd.DataFrame({
            "actual": y_test,
            "predicted": decision_function_scores,
        })

        self.metrics = pd.DataFrame({
            self.method_column: [self._get_name()],
            SPACE_COL: [self.get_space()],
            AUC_COL: [auc],
            PRAUC_COL: [prauc],
            F1_COL: [f1],
            TIME_TAKEN_COL: [self.time_elapsed],
            BASE_COL: [self.base_method.__name__],
            #SPACE_TYPE_COL: [self.get_space_type()]
        })

        self.common_parameters = pd.DataFrame({
            PERCENTAGE_INLIER_COL : [percentage_inlier],
            PERCENTAGE_OUTLIER_COL : [percentage_outlier],
            TOTAL_TEST_SAMPLES_COL : [len(y_test)],
            TOTAL_TRAIN_SAMPLES_COL : [len(self.x_train)],
            INLIER_LABEL_COL : [self.inlier_label],
            OUTLIER_LABEL_COL : str([label for label in self.dataset.get_possible_labels() if label != self.inlier_label]),
            MODEL_COL : [self.space.model.model_name],
            DATASET_COL : [self.dataset.name],
        })

        return self.metrics, self.common_parameters



