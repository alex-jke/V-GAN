from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple

from numpy import ndarray

from modules.od_module import VGAN_od, VMMD_od
from modules.text.vmmd_text_lightning import VMMDTextLightningBase
from text.outlier_detection.space.prepared_data import PreparedData
from text.outlier_detection.space.space import Space
from text.outlier_detection.space.token_space import TokenSpace
from text.visualizer.collective_visualizer import CollectiveVisualizer
from text.visualizer.od import od_subspace_visualizer


class BaseVOdmAdapter(ABC):
    """
    Base class for all V_ODM adapters. This class is used to create the vmmd or vgan model
    used for outlier detection. The model is created in the init_model method.
    """
    def __init__(self):
        self.model: VMMD_od | VGAN_od | VMMDTextLightningBase | None = None
        self.loaded_model = False
        self.data: PreparedData | None= None
        self.space: Space | None = None
        self.output_path: Path | None = None
        self.initialized = False
        self.subspaces: ndarray | None = None
        self.proba: ndarray | None = None

    def _init_subspaces(self, num_subspaces: int):
        self.model.approx_subspace_dist(add_leftover_features=False, subspace_count=1000)
        subspaces = self.model.subspaces
        proba = self.model.proba

        # Select the num_subspaces most probable subspaces
        self.subspaces, self.proba = self._get_top_subspaces(num_subspaces, proba, subspaces)

    def init_model(self, data: PreparedData, base_path: Path, space: Space):
        """
        Initializes the model used for outlier detection. If an already trained model is found in the base_path
        the model is loaded from the file. If no model is found a new model is created and trained.
        """
        self.data = data
        self.space = space
        if base_path is not None:
            self.output_path = base_path / self.get_name() / self.space.name
        self.model = self._init_model(data, space)
        features = space.get_n_dims(self.data.x_train)
        self._load_model(self.output_path, features, self.model)
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

    def _load_model(self, base_path: Path, features: int, model: VMMD_od | VGAN_od):
        """
        Loads the model from the base_path.
        """
        if base_path is None:
            return
        generator_path = base_path / "models" / "generator_0.pt"
        if generator_path.exists():
            model.load_models(generator_path, ndims=features)
            self.loaded_model = True

    def get_subspaces(self, num_subspaces=50) -> ndarray:
        """
        Returns subspace_count operator samples from the random operator. This currently being axis parallel subspaces.
        """
        self.__assert_initialized()

        if self.subspaces is None:
            self._init_subspaces(num_subspaces)

        return self.subspaces

    @staticmethod
    def _get_top_subspaces(num_subspaces: int, proba: ndarray, subspaces: ndarray) -> Tuple[ndarray, ndarray]:
        """
        Returns the num_subspaces most probable subspaces, with their probabilities.
        """
        amount = min(num_subspaces, len(proba))
        idx = proba.argsort()[-amount:][::-1]
        top_subspaces = subspaces[idx]
        top_proba = proba[idx]

        # Ignore Subspaces contributing less than 0.2% to the model if the rest of the subspaces make up at least 80%
        threshold = 0.002
        if top_proba[top_proba > threshold].sum() > 0.8:
            top_subspaces = top_subspaces[top_proba > threshold]
            top_proba = top_proba[top_proba > threshold]

        return top_subspaces, top_proba

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

