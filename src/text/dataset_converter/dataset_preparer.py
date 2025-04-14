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
        return self.get_data_with_labels(labels, train=True)[0]

    def get_data_with_labels(self, labels: list = None, train: bool = True) -> Tuple[ndarray[str], ndarray[str]]:
        """
        Returns the training data of the dataset as an .
        :param labels: The labels to filter for. If left empty, the first label of the dataset is used for training data
            and all labels are used for testing data.
        :return: A tuple of:
                    - The filtered training data of the dataset as a ndarray of strings.
                    - The labels of the training data as a ndarray of strings.
        """
        if labels is None:
            labels = [self.dataset.get_possible_labels()[0]] if train else self.dataset.get_possible_labels()
        if train:
            data, labels = self.dataset.get_training_data(labels)
        else:
            data, labels = self.dataset.get_testing_data(labels)

        if self.max_samples > 0:
            data = data[:self.max_samples]
            labels = labels[:self.max_samples]

        #data = data.apply(lambda sentence: self.clean(sentence))#self.split_and_clean(sentence))

        # Convert the pandas Series to a numpy array
        data_np: ndarray[str] = data.to_numpy(dtype=str)
        labels_np: ndarray[str] = labels.to_numpy(dtype=str)
        return data_np, labels_np

    @staticmethod
    def get_average_sentence_length(x_data: ndarray[str], seperator: str = " ") -> int:
        """
        Calculate the average sentence length in the dataset.
        :param x_data: A numpy array of sentences.
        :return: The average sentence length, as an integer.
        """
        sequence_length = int(np.mean([len(x.split(seperator)) for x in x_data]))
        return sequence_length

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


