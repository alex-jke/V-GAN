import csv
import os
from typing import Tuple, List

import numpy as np
import pandas as pd

from .dataset import Dataset, AggregatableDataset

X = "Description"
Y = "Class Index"


class AGNews(AggregatableDataset):
    """
    AG News Dataset class. It reads the train and test files from the resource folder and stores them as pandas dataframes.
    Consists of class ids 1-4 where 1-World, 2-Sports, 3-Business, 4-Sci/Tech
    """

    def prefix(self) -> List[str]:
        return (#"Description: TOKYO, April 14 (Reuters) - Japanese Prime Minister Shigeru Ishiba said on Monday his country does not plan to make big concessions and won't rush to reach a deal in upcoming tariff negotiations with U.S. President Donald Trump's administration."
                #"News type: World"
                "Description : EU foreign ministers meet in wake of deadly Russian attack on Sumy as Zelenskyy issues plea for Trump to visit Ukraine – Europe live."
                "News type : World\n"
                "Description : Stocks rally as electronics get a tariff break."
                "News type : Business\n"
                "Description :").split(" ")

    def suffix(self) -> List[str]:
        return ". News type :".split(" ")

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
