from abc import ABC, abstractmethod
from typing import List

import numpy as np


class Embedding(ABC):

    @abstractmethod
    def embed(self, data: str) -> np.ndarray:
        pass

    @abstractmethod
    def embed_words(self, words: List[str]) -> List[np.ndarray]:
        pass

    @abstractmethod
    def decode(self, data: np.ndarray) -> str:
        pass