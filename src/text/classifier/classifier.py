from abc import ABC, abstractmethod



class Classifier(ABC):

    @abstractmethod
    def train(self) -> None:
        pass

    @abstractmethod
    def predict(self):
        pass
