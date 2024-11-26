from abc import ABC, abstractmethod

from dataset.dataset import Dataset


class Classifier(ABC):

    @abstractmethod
    def train(self) -> None:
        pass

    @abstractmethod
    def predict(self):
        pass
