from abc import ABC, abstractmethod
from typing import Tuple, List

import numpy as np


class Dataset(ABC):

    def __init__(self):
        self.imported = False
        self.x_train = self.y_train = self.x_test = self.y_test = None

    @abstractmethod
    def get_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        pass

    @abstractmethod
    def get_testing_data(self) -> Tuple[np.ndarray, np.ndarray]:
        pass

    def get_one_class_testing_data(self, class_label: str) -> np.ndarray:
        if class_label not in self.get_possible_labels():
            raise ValueError(f"Class label {class_label} is not a possible label.")
        if not self.imported:
            self._import_data()
        return self.x_test[self.y_test == class_label]

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