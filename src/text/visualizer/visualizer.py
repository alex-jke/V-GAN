import os
from abc import ABC, abstractmethod
from pathlib import Path

from torch import Tensor

from text.Embedding.tokenizer import Tokenizer


class Visualizer(ABC):
    """
    Abstract class for visualizing data.
    """

    def __init__(self, model, tokenized_data: Tensor, tokenizer: Tokenizer, path: str):
        super().__init__()
        self.num_subspaces = 500
        # Tensor is of the shape (num_subspaces, sequence_length)
        self.model = model
        self.subspaces: Tensor = self.get_subspaces(self.num_subspaces)
        self.avg_subspace: Tensor = self.subspaces.sum(dim=0) / self.num_subspaces

        self.tokenized_data = tokenized_data
        self.tokenizer: Tokenizer = tokenizer

        self.output_dir = Path(path) / "visualizations"

    def get_subspaces(self, samples: int = 1):
        return self.model.generate_subspaces(samples)

    @abstractmethod
    def visualize(self, samples: int = 1):
        """
        Visualizes the data.
        :param text: The text to visualize.
        """
        raise NotImplementedError