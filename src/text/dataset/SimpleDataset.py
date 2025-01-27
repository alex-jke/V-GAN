from typing import List

import pandas as pd

from text.dataset.dataset import Dataset


class SimpleDataset(Dataset):

    def __init__(self, samples: List[str], amount_samples: int):
        self.samples = samples
        self.amount_samples = amount_samples

        super().__init__()
    def get_possible_labels(self) -> list:
        return [0]

    def _import_data(self):
        data = []
        for i in range(self.amount_samples):
            data.append(self.samples[i % len(self.samples)])

        df = pd.DataFrame(data, columns=[self.x_label_name])
        df[self.y_label_name] = 0
        self.split(df)

    @property
    def name(self) -> str:
        return "Simple Dataset"

    @property
    def x_label_name(self) -> str:
        return "x"

    @property
    def y_label_name(self):
        return "y"