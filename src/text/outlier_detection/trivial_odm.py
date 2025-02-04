from random import random
from typing import List

from text.outlier_detection.odm import OutlierDetectionModel


class TrivialODM(OutlierDetectionModel):

    def __init__(self, dataset, model, train_size, test_size, inlier_label=None, guess_inlier_rate= 0.5):
        self.guess_rate = guess_inlier_rate
        super().__init__(dataset, model, train_size, test_size, inlier_label)

    def train(self):
        self.use_tokenized()

    def predict(self):
        self.predictions = [1 if random() <= self.guess_rate else 0 for i in range(len(self.x_test))]

    def _get_name(self):
        return f"Trivial + {self.guess_rate}"

    def _get_predictions(self) -> List[int]:
        return self.predictions

    def get_space(self):
        return "Tokenized"