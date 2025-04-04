from typing import Tuple

import numpy as np
from numpy import ndarray
from pandas import Series
from torch import Tensor

from text.dataset.dataset import Dataset


class DatasetPreparer:
    """
    A class that prepares a dataset to be used by vmmd text.
    """
    def __init__(self, dataset: Dataset, max_samples: int= -1):
        self.dataset = dataset
        self.max_samples = max_samples

    def get_training_data(self, labels: list = None) -> ndarray[str]:
        """
        Returns the training data of the dataset as a ndarray.
        :param labels: The labels to filter for. If left empty, the first label of the dataset is used.
        :return: The training data of the dataset.
        """
        return self.get_training_data_with_labels(labels)[0]

    def get_training_data_with_labels(self, training_labels: list = None) -> Tuple[ndarray[str], ndarray[str]]:
        """
        Returns the training data of the dataset as an .
        :param training_labels: The labels to filter for. If left empty, the first label of the dataset is used.
        :return: A tuple of:
                    - The filtered training data of the dataset as a ndarray of strings.
                    - The labels of the training data as a ndarray of strings.
        """
        if training_labels is None:
            training_labels = [self.dataset.get_possible_labels()[0]]
        data, labels = self.dataset.get_training_data(training_labels)

        if self.max_samples > 0:
            data = data[:self.max_samples]
            labels = labels[:self.max_samples]

        data = data.apply(lambda sentence: self.clean(sentence))#self.split_and_clean(sentence))

        # Convert the pandas Series to a numpy array
        data_np: ndarray[str] = data.to_numpy(dtype=str)
        labels_np: ndarray[str] = labels.to_numpy(dtype=str)
        return data_np, labels_np

    @staticmethod
    def clean(sentence: str) -> str:
        """
        Removes any empty strings.
        Also removes any special characters.
        :param sentence: The sentence to clean.
        :return: The cleaned sentence.
        """
        cleaned = (sentence
                    .replace("\\", "")
                    .replace("(", "")
                    .replace(")", "")
                    .replace("[", "")
                    .replace("]", "")
                    .replace("{", "")
                    .replace("}", "")
                    .replace(".", "")
                    .replace(",", "")
                    .replace(":", "")
                    .replace(";", "")
                    .replace("!", "")
                    .replace("?", "")
                    .replace("-", " "))
        #return np.array(cleaned.split(" "))
        return cleaned


