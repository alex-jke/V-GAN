from abc import ABC, abstractmethod

from text.Embedding.huggingmodel import HuggingModel
from text.dataset.dataset import Dataset


class OutlierDetectionModel(ABC):

    def __init__(self, dataset: Dataset, model: HuggingModel, train_size: int, test_size: int):
        self.dataset = dataset
        self.model = model
        self.train_size = train_size
        self.test_size = test_size
        self.name = self._get_name()

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def predict(self):
        pass

    @abstractmethod
    def _get_name(self):
        pass