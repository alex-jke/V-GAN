from typing import List

import kagglehub
import pandas as pd

from text.dataset.aggregatable import Aggregatable
from text.dataset.dataset import Dataset, AggregatableDataset


class EmotionDataset(AggregatableDataset):

    def prefix(self) -> List[str]:

        return ["text", ":", "I", "am", "happy", "because", "it", "is", "sunny", ".", "feeling", ":", "not",
                   "sadness", "\n",
                   "text", ":", "I", "feel", "sad", "because", "I", "have", "no", "friends", ".", "feeling", ":",
                   "sadness""\n",
                   "text", ":"]

    def suffix(self) -> List[str]:
        return [".", "feeling", ":"]

    def get_possible_labels(self) -> list:
        # Six categories: sadness (0), joy (1), love (2), anger (3), fear (4), and surprise (5).
        return [0, 1, 2, 3, 4, 5]

    def _import_data(self):
        # Download latest version
        if (self.dir_path / "text.csv").exists():
            data = pd.read_csv(self.dir_path / "text.csv")
        else:
            path = kagglehub.dataset_download("nelgiriyewithana/emotions")
            data = pd.read_csv(path + "/text.csv")
            if not self.dir_path.exists():
                self.dir_path.mkdir(parents=True)
            data.to_csv(self.dir_path / "text.csv", index=False)

        self.split(data)

    @property
    def name(self) -> str:
        return "Emotions"

    @property
    def x_label_name(self) -> str:
        return "text"

    @property
    def y_label_name(self):
        return "label"