import csv
import os
from typing import Tuple, List

import numpy as np
import pandas as pd

from .dataset import Dataset

X = "Description"
Y = "Class Index"


class AGNews(Dataset):
    """
    AG News Dataset class. It reads the train and test files from the resource folder and stores them as pandas dataframes.
    Consists of class ids 1-4 where 1-World, 2-Sports, 3-Business, 4-Sci/Tech
    """

    def get_possible_labels(self) -> list:
        return [1, 2, 3, 4]
        #return ['World', 'Sports', 'Business', 'Sci/Tech']

    def get_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        if not self.imported:
            self._import_data()
        return self.x_train, self.y_train

    def get_testing_data(self) -> Tuple[np.ndarray, np.ndarray]:
        if not self.imported:
            self._import_data()
        return self.x_test, self.y_test

    @property
    def name(self) -> str:
        return "AG News"

    def _import_data(self):
        self.ag_news_dir = os.path.join(os.path.dirname(__file__), '../resources/ag_news')
        self.train = self._read_file('train.csv')
        self.test = self._read_file('test.csv')

        self.x_train, self.y_train = self.train[X], self.train[Y]
        self.x_test, self.y_test = self.test[X], self.test[Y]
        self.imported = True

    def _read_file(self, file_name) -> pd.DataFrame:
        file_path = os.path.join(self.ag_news_dir, file_name)
        data = pd.read_csv(file_path)
        return data

    @property
    def x_label_name(self) -> str:
        return X

    @property
    def y_label_name(self):
        return Y
