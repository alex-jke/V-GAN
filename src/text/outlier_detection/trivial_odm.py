from random import random
from typing import List

from text.outlier_detection.odm import OutlierDetectionModel
from text.outlier_detection.space_type import SpaceType


class TrivialODM(OutlierDetectionModel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _train(self):
        pass

    def _predict(self):
        self.predictions = [random() for _ in range(len(self.x_test))]

    def _get_name(self):
        return f"Trivial"

    def _get_predictions(self) -> List[float]:
        return self.predictions

    def get_space_type(self) -> SpaceType:
        return SpaceType.RANDOM_GUESS