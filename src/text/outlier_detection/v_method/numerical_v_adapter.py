from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple, Optional

import numpy
from numpy import ndarray

from modules.od_module import VGAN_od, VMMD_od, ODModule
from modules.text.vmmd_text_lightning import VMMDTextLightningBase
from text.outlier_detection.base_v_adapter import BaseVAdapter
from text.outlier_detection.space.prepared_data import PreparedData
from text.outlier_detection.space.space import Space
from text.outlier_detection.space.token_space import TokenSpace
from text.visualizer.collective_visualizer import CollectiveVisualizer
from text.visualizer.od import od_subspace_visualizer


class NumericalVOdmAdapter(BaseVAdapter):
    """
    Base class for all V_ODM adapters, that work with numerical data. This class is used to create the vmmd or vgan model
    used for outlier detection. The model is created in the init_model method.
    """
    def __init__(self):
        self.model: Optional[ODModule] = None
        self.loaded_model = False
        self.data: PreparedData | None= None
        self.space: Space | None = None
        self.output_path: Path | None = None
        self.initialized = False
        self.subspaces: ndarray | None = None
        self.proba: ndarray | None = None

    @staticmethod
    def _remove_zero_subspaces(subspaces: ndarray, proba: ndarray) -> Tuple[ndarray, ndarray]:
        """
        Removes the zero subspaces from the list of subspaces and their probabilities.
        :param subspaces: The subspaces to remove the zero subspaces from.
        :param proba: The probabilities of the subspaces.
        :return: The subspaces and probabilities without the zero subspaces.
        """
        non_zero_subspaces = subspaces.sum(axis=1) != 0
        return subspaces[non_zero_subspaces], proba[non_zero_subspaces]

    def _init_subspaces(self, num_subspaces: int):
        # Select the num_subspaces most probable subspaces
        subspaces, proba = self._get_top_subspaces(self.model, num_subspaces)
        # Remove the subspace with only zeros
        subspaces, proba = self._remove_zero_subspaces(subspaces, proba)

        if proba.shape[0] == 0:
            raise ValueError("Only the zero subspace was passed.")

        self.subspaces = subspaces
        self.proba = proba

    def init_model(self, data: PreparedData, base_path: Path, space: Space):
        """
        Initializes the model used for outlier detection. If an already trained model is found in the base_path
        the model is loaded from the file. If no model is found a new model is created and trained.
        """
        if self.subspaces is not None:
            raise RuntimeError("Init model was called on an already used instance of an adapter. This could lead to "
                               "unexpected behavior, as some attributes are stored within the class.")
        self.data = data
        self.space = space
        if base_path is not None:
            self.output_path = base_path / self.get_name() / self.space.name
        self.model = self._init_model(data, space)
        print(f"Attempting to load model for space {self.space.name}.")
        self._load_model(self.output_path, data.x_train.shape[1], self.model)
        self.initialized = True

    def train(self, print_epochs: int = 300):
        """
        If the model could be loaded, it skips the training, otherwise it trains a model.
        :param print_epochs: The number of epochs between each print.
        :param num_subspaces: The number of subspaces to sample from the random operator.
        """
        self.__assert_initialized()
        if not self.loaded_model:
            self._train(print_epochs)

    def _train(self, print_epochs: int):
        """
        Trains the model.
        """
        print(f"Training {self.get_name()} model for {self.model.epochs} epochs.")
        for epoch in self.model.yield_fit(self.data.x_train, yield_epochs=print_epochs):
            loss = self.model.train_history[self.model.generator_loss_key][-1] if epoch > 0 else float("nan")
            print(f"({epoch}, {loss})")
        self.visualize_results()

    @abstractmethod
    def _init_model(self, data: PreparedData, space: Space) -> VMMD_od | VGAN_od | VMMDTextLightningBase:
        """
        Private method that should be implemented by the subclass. This method should initialize the model used for
        outlier detection.
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """
        Returns the name of the model.
        """
        pass

    def get_subspaces(self, num_subspaces=50) -> ndarray:
        """
        Returns subspace_count operator samples from the random operator. This currently being axis parallel subspaces.
        """
        self.__assert_initialized()

        if self.subspaces is None:
            self._init_subspaces(num_subspaces)

        return self.subspaces

    def get_subspace_probabilities(self, num_subspaces=50):
        """
        Returns the probabilities of the subspaces. This is used to determine the importance of the subspaces.
        """
        self.__assert_initialized()
        if self.proba is None:
            self._init_subspaces(num_subspaces)

        return self.proba

    def visualize_results(self):
        """
        Visualizes the results of the v-method. This can help to understand the results of the model.
        """
        self.__assert_initialized()
        if self.output_path is not None and not self.loaded_model:
            self.output_path.mkdir(parents=True, exist_ok=True)
            visualizer = CollectiveVisualizer(tokenized_data=self.data.x_test, tokenizer=self.space.model, vmmd_model=self.model,
                                              export_path=str(self.output_path),
                                              text_visualization=isinstance(self.space, TokenSpace))
            visualizer.visualize(samples=30, epoch=self.model.epochs)

    def __assert_initialized(self):
        if not self.initialized:
            raise ValueError("The model has not been initialized. Have you called init_model?")

