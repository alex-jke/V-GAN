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
        self.dataset = dataset
        self.space = space
        self.inlier_label = inlier_label
        self.output_path = output_path
        self.adapter = VMMDAdapter(generator=generator, export_generator=True)
        self.generator = generator
        self.space = space

    def train(self):
        token_data = PreparedData(*self.space.get_tokenized(self.dataset, inlier_label=self.inlier_label), space=self.space.name,
                                       inlier_labels=[self.inlier_label])
        self.adapter.init_model(token_data, self.output_path, self.space)
        self.adapter.train()

    def get_subspaces(self, num_subspaces: int = 50) -> ndarray[float]:
        return self.adapter.get_subspaces(num_subspaces)

    def get_probabilities(self, num_subspaces: int = 50) -> ndarray[float]:
        return self.adapter.get_probabilities(num_subspaces)

    def get_name(self) -> str:
        return "TokenVMMD + " + self.generator.__name__