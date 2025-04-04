from abc import ABC, abstractmethod
from typing import List

from torch import Tensor


class Tokenizer(ABC):

    @abstractmethod
    def tokenize(self, data: str) -> Tensor:
        pass

    @abstractmethod
    def detokenize(self, tokens: List[int]) -> str:
        pass

    @property
    @abstractmethod
    def _padding_token(self) -> int:
        pass