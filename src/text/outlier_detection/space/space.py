from abc import ABC, abstractmethod
from typing import Optional

from numpy import ndarray
from torch import Tensor

from text.Embedding.LLM.huggingmodel import HuggingModel
from text.dataset.dataset import Dataset
from text.outlier_detection.space.prepared_data import PreparedData


class Space(ABC):
    """
    This class represents the space that the outlier detection models
    should operate in.
    """
    def __init__(self, model: HuggingModel, train_size: int = -1, test_size: int = -1):
        """
        Initializes the Space object.
        :param model: The model to use to generate the space.
        :param train_size: The size of the training set.
        :param test_size: The size of the test set.
        """
        self.model = model
        self.train_size = train_size
        self.test_size = test_size

    @abstractmethod
    def transform_dataset(self, dataset: Dataset, use_cached: bool, inlier_label, mask: Optional[ndarray[float]]) -> PreparedData:
        """
        Prepares the data for the outlier detection model.
        :param dataset: The dataset to prepare the data from.
        :param use_cached: Whether to use cached data.
            If True, the data will be loaded from cached files if it exists or
            cached files will be created.
            If False, the data will be created from scratch and not cached.
        :param inlier_label: The label of the inliers.
        :param mask: The mask to apply to the data. That is, the projection
        into a subspace.
        :return: The PreparedData object that contains the training and
         testing data, projected in this space and then transformed into
         the embedding space.
        testing data.
        """
        pass


    @property
    @abstractmethod
    def name(self):
        """
        Returns the name of the space.
        """
        pass
