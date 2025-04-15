import csv
import os
from typing import Tuple, List

import numpy as np
import pandas as pd

from .dataset import Dataset, AggregatableDataset
from .prompt import Prompt

X = "Description"
Y = "Class Index"


class AGNews(AggregatableDataset):
    """
    AG News Dataset class. It reads the train and test files from the resource folder and stores them as pandas dataframes.
    Consists of class ids 1-4 where 1-World, 2-Sports, 3-Business, 4-Sci/Tech
    """
    prompt = Prompt(  sample_prefix="Description :",
                                    label_prefix="News type :",
                                    samples=["EU foreign ministers meet in wake of deadly Russian attack on Sumy as Zelenskyy issues plea for Trump to visit Ukraine – Europe live.",
                                        "Stocks rally as electronics get a tariff break."],
                                    labels=["World", "Business"])

    def __init__(self):
        super().__init__(self.prompt)

    def get_possible_labels(self) -> list:
        return [1, 2, 3, 4]
        #return ['World', 'Sports', 'Business', 'Sci/Tech']

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
