from typing import Tuple, List

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

from .dataset import Dataset


class IMBdDataset(Dataset):

    def get_possible_labels(self) -> list:
        #return ["positive", "negative"]
        return [1, 0]

    def _import_data(self):
        data = pd.read_csv("hf://datasets/scikit-learn/imdb/IMDB Dataset.csv")
        data[self.y_label_name] = np.where(data["sentiment"] == "positive", 1, 0)
        self.split(data)

    @property
    def name(self) -> str:
        return "IMDb"

    @property
    def x_label_name(self) -> str:
        return "review"

    @property
    def y_label_name(self):
        return "sentiment"
