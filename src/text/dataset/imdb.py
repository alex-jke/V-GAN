from typing import Tuple, List

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

from .dataset import Dataset


class IMBdDataset(Dataset):

    def get_possible_labels(self) -> list:
        return ["positive", "negative"]

    def get_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        if not self.imported:
            self._import_data()
        return self.x_train, self.y_train

    def get_testing_data(self) -> Tuple[np.ndarray, np.ndarray]:
        if not self.imported:
            self._import_data()
        return self.x_test, self.y_test

    def _import_data(self):
        data = pd.read_csv("hf://datasets/scikit-learn/imdb/IMDB Dataset.csv")
        x = data[self.x_label_name]
        y = data[self.y_label_name]

        # Split the data into training and test sets
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        self.imported = True

    @property
    def name(self) -> str:
        return "IMDb"

    @property
    def x_label_name(self) -> str:
        return "review"

    @property
    def y_label_name(self):
        return "sentiment"
