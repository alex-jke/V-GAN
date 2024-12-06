from abc import ABC, abstractmethod
from typing import Tuple, List

import numpy as np
from sklearn.model_selection import train_test_split


class Dataset(ABC):

    def __init__(self):
        self.imported = False
        self.x_train = self.y_train = self.x_test = self.y_test = None

    def get_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        if not self.imported:
            self._import_data()
        return self.x_train, self.y_train

    def get_testing_data(self) -> Tuple[np.ndarray, np.ndarray]:
        if not self.imported:
            self._import_data()
        return self.x_test, self.y_test

    def get_one_class_testing_data(self, class_label: str) -> np.ndarray:
        if class_label not in self.get_possible_labels():
            raise ValueError(f"Class label {class_label} is not a possible label.")
        if not self.imported:
            self._import_data()
        return self.x_test[self.y_test == class_label]

    def split(self, data):
        x = data[self.x_label_name]
        y = data[self.y_label_name]

        # Split the data into training and test sets
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        self.imported = True

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