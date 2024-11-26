from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class Vocabulary(ABC):

    @abstractmethod
    def get_word(self, vec: np.ndarray) -> str | None:
        pass

    @abstractmethod
    def get_vec(self, word: str) -> np.ndarray:
        pass

    @abstractmethod
    def containsWord(self, word: str) -> bool:
        pass

    @abstractmethod
    def containsVec(self, vec: np.ndarray) -> bool:
        pass

    @abstractmethod
    def get_size(self) -> int:
        pass

    @abstractmethod
    def get_words(self) -> list[str]:
        pass

    @abstractmethod
    def get_vecs(self) -> list[np.ndarray]:
        pass
