from enum import Enum
from typing import Dict, Optional, Union


class StrategyInstance():
    def __init__(self, strategy: "UnificationStrategy", param: Optional[int]):
        self.param = param
        self.strategy = strategy

    def key(self) -> str:
        return self.strategy.key

    def equals(self, other: Union['StrategyInstance', "UnificationStrategy"]) -> bool:
        if isinstance(other, UnificationStrategy):
            return self.key() == other.key
        return self.key() == other.key()

    def __repr__(self) -> str:
        if self.param is not None:
            return f"{self.strategy.name}({self.param})"
        return self.strategy.name


class UnificationStrategy(Enum):
    """
    The unification strategy for embeddings of sentences, which can be of different lengths.
    """
    #Returns the first padding length vectors in a tensor. The embeddings are padded, if shorter and trimmed if longer.
    PADDING = ("pad", True)

    # Returns the mean of all embeddings
    MEAN = ("avg", False)

    # Lets the transformer perform the aggregation by taking the embedding of a significant token.
    TRANSFORMER = ("NPTE", False)

    def __init__(self, key: str, requires_param: bool):
        self.key: str = key
        self.requires_param: bool = requires_param

    def create(self, param: Optional[int] = None) -> StrategyInstance:
        if self.requires_param and param is None:
            raise ValueError(f"{self.name} requires a parameter.")
        return StrategyInstance(self, param)
