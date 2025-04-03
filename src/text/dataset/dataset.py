import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple, List

import numpy as np
from pandas import Series
from sklearn.model_selection import train_test_split

from text.dataset.aggregatable import Aggregatable


class Dataset(ABC):

    def __init__(self):
        self.imported = False
        self.x_train = self.y_train = self.x_test = self.y_test = None
        self.resources_path = Path(os.path.dirname(__file__)) / '../resources'
        self.dir_path = self.resources_path / self.name

    def get_training_data(self, filter_labels: list | None = None) -> Tuple[Series, Series]:
        """
        Returns the training data of the dataset.
        :param filter_labels: The labels to filter for. If left empty, all labels are returned.
        :return: A tuple of:
                    - The training data of the dataset.
                    - The labels of the training data.
        """
        if not self.imported:
            self._import_data()
            self.imported = True
        if filter_labels is None:
            return self.x_train, self.y_train
        return self._get_filtered_data(self.x_train, self.y_train, filter_labels)

    def get_testing_data(self, filter_labels: list | None) -> Tuple[Series, Series]:
        """
        Returns the testing data of the dataset.
        :param filter_labels: The labels to filter for. If left empty, all labels are returned.
        :return: A tuple of:
                    - The testing data of the dataset.
                    - The labels of the testing data.
        """
        if not self.imported:
            self._import_data()
            self.imported = True
        if filter_labels is None:
            return self.x_test, self.y_test
        return self._get_filtered_data(self.x_test, self.y_test, filter_labels)

    def _get_filtered_data(self, data: Series, labels: Series, filter_labels: list) -> Tuple[Series, Series]:
        for label in filter_labels:
            if label not in self.get_possible_labels():
                raise ValueError(f"Label {label} is not a possible label. Possible labels are {self.get_possible_labels()}")
        idx = labels.isin(filter_labels)
        return data[idx].reset_index(drop=True), labels[idx].reset_index(drop=True)


    def get_one_class_testing_data(self, class_label: str) -> np.ndarray:
        if class_label not in self.get_possible_labels():
            raise ValueError(f"Class label {class_label} is not a possible label.")
        if not self.imported:
            self._import_data()
            self.imported = True
        return self.x_test[self.y_test == class_label]

    def split(self, data):
        x = data[self.x_label_name]
        y = data[self.y_label_name]

        # Split the data into training and test sets
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        self.imported = True

    @property
    def average_length(self):
        if not self.imported:
            self._import_data()
        return self.x_train.apply(lambda x: len(x)).mean()

    @abstractmethod
    def get_possible_labels(self) -> list:
        pass

    @abstractmethod
    def _import_data(self):
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def x_label_name(self) -> str:
        """
        Returns the name of the x-axis label.
        :return: A string representing the x-axis label.
        """
        pass

    @property
    @abstractmethod
    def y_label_name(self):
        """
        Returns the name of the y-axis label.
        :return: A string representing the y-axis label.
        """
        pass

class AggregatableDataset(Dataset, Aggregatable, ABC):
    """
    A dataset that can be aggregated.
    """
