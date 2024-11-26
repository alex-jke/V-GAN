from abc import ABC, abstractmethod
from typing import List


class Tokenizer(ABC):

    @abstractmethod
    def tokenize(self, data: str) -> List[int]:
        pass

    @abstractmethod
    def detokenize(self, tokens: List[int]) -> str:
        pass

    @property
    @abstractmethod
    def padding_token(self) -> int:
        pass