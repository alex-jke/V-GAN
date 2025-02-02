from typing import Callable

from torch import Tensor

from text.UI.cli import ConsoleUserInterface


class DatasetEmbedder:
    def __init__(self, embedding_function: Callable[[Tensor], Tensor]):
        self.embedding_function = embedding_function
        self.ui = ConsoleUserInterface()

    def embed(self, tokenized_dataset: Tensor) -> Tensor:
        self.ui.update(data="Creating embedded dataset...")
        embedded = self.embedding_function(tokenized_dataset)
        return embedded
