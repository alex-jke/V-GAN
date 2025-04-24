from pathlib import Path
from typing import Optional, Type

from numpy import ndarray

from models.Generator import GeneratorUpperSoftmax, Generator_big
from text.outlier_detection.space.prepared_data import PreparedData
from text.dataset.dataset import Dataset
from text.outlier_detection.base_v_adapter import BaseVAdapter
from text.outlier_detection.space.space import Space
from text.outlier_detection.space.token_space import TokenSpace
from text.outlier_detection.v_method.vmmd_adapter import VMMDAdapter


class TokenVAdapter(BaseVAdapter):

    def __init__(self, dataset: Dataset, space: TokenSpace, inlier_label, output_path: Optional[Path]=None, generator: Type[Generator_big] = GeneratorUpperSoftmax):
        self.token_data = PreparedData(*space.get_tokenized(dataset, inlier_label=inlier_label), space=space.name, inlier_labels=[inlier_label])
        self.output_path = output_path
        self.adapter = VMMDAdapter(generator=generator)
        self.generator = generator
        self.space = space

    def train(self):
        self.adapter.init_model(self.token_data, self.output_path, self.space)
        self.adapter.train()

    def get_subspaces(self, num_subspaces: int = 50) -> ndarray[float]:
        return self.adapter.get_subspaces(num_subspaces)

    def get_probabilities(self, num_subspaces: int = 50) -> ndarray[float]:
        return self.adapter.get_probabilities(num_subspaces)

    def get_name(self) -> str:
        return "TokenVMMD + " + self.generator.__name__