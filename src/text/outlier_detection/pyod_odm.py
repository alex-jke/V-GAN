from abc import abstractmethod, ABC
from typing import Tuple, List, Type, Callable

from numpy import ndarray
from pyod.models.base import BaseDetector
from pyod.models.lof import LOF as pyod_LOF
from pyod.models.lunar import LUNAR as pyod_LUNAR
from pyod.models.ecod import ECOD as pyod_ECOD
from pyod.models.feature_bagging import FeatureBagging as pyod_FeatureBagging
from sklearn.preprocessing import StandardScaler
from torch import Tensor
import torch

from text.Embedding.huggingmodel import HuggingModel
from text.dataset.dataset import Dataset
from text.outlier_detection.odm import OutlierDetectionModel


class PyODM(OutlierDetectionModel, ABC):
    def __init__(self, dataset: Dataset, model: HuggingModel, train_size: int, test_size: int, pre_embed = True, use_cached = False):
        #self.space = "Embedding" if pre_embed else "Tokenized"
        self.initializing_fun = self.use_embedding if pre_embed else self.use_tokenized
        super().__init__(dataset=dataset, model=model, train_size=train_size, test_size=test_size, use_cached=use_cached)
        self.od_model = self._get_model()

    def train(self):
        self.initializing_fun()
        self.od_model.fit(self.x_train.cpu().numpy(), None)

    def predict(self):
        test = self.x_test
        decision_function = self.od_model.decision_function(test.cpu().numpy())
        self.decision_function = decision_function

    def _get_predictions(self) -> List[int]:
        return self.decision_function

    @abstractmethod
    def _get_model(self):
        pass


#TODO: set the contamination parameter, as there is no contamination.
class LOF(PyODM):
    def _get_model(self):
        return pyod_LOF()

    def _get_name(self):
        return f"LOF"

class LUNAR(PyODM):
    def _get_model(self):
        return pyod_LUNAR()

    def _get_name(self):
        return f"LUNAR"

class ECOD(PyODM):
    def _get_model(self):
        return pyod_ECOD()

    def _get_name(self):
        return f"ECOD"

class FeatureBagging(PyODM):

    def __init__(self, dataset: Dataset, model: HuggingModel, base_detector:  Type[BaseDetector], train_size: int, test_size: int, pre_embed = True, use_cached = False):
        """
        :param dataset: The dataset to use.
        :param model: The model to use.
        :param base_estimator: The base estimator to use. Either 'LOF', 'LUNAR', or 'ECOD'.
        :param train_size: The number of training samples.
        :param test_size: The number of test samples.
        :param use_embedding: Whether to use the embedding or the tokenized data.
        """
        if base_detector is None:
            base_detector = pyod_LUNAR

        self.base_name = base_detector.__name__
        self.base_estimator: BaseDetector = base_detector()

        if not pre_embed:
            self.base_estimator = EmbeddingBaseDetector(model.get_embedding_fun(batch_first=True), lambda: base_detector)

        self.__model = pyod_FeatureBagging(base_estimator=self.base_estimator)
        super().__init__(dataset, model, train_size, test_size, pre_embed, use_cached=use_cached)

    def _get_model(self):
        return self.__model

    def _get_name(self):
        return f"FeatureBagging + {self.base_name} + {self.get_space()}"

class EmbeddingBaseDetector(BaseDetector):

    def __init__(self, embedding_fun: Callable[[Tensor], Tensor], base_detector: Callable[[],Type[BaseDetector]]):
        # self.model = model #saving the model will cause CUDA out of memory as model is large
        self.base_detector: Callable[[], Type[BaseDetector]] = base_detector # This is required because, when the base detectors are duplicated in the feature bagging,
        # instead for each param to check if its of type BaseDetector it checks if it has the get_params method.
        # Then an error it caused was it trying to call get_params on the base_detector class, that self was not passed.
        #self.embedding_fun = self.model.get_embedding_fun(batch_first=True)
        self.embedding_fun = embedding_fun
        self._classes = 2
        super().__init__()

    def fit(self, X: ndarray, y=None):
        self._estimator = self.base_detector()(contamination = self.contamination)
        embedded = self._embed(X)
        self._estimator.fit(embedded, y)
        self.decision_scores_ = self._estimator.decision_scores_
        self.threshold_ = self._estimator.threshold_
        self.labels_ = self._estimator.labels_

    def decision_function(self, X):
        embedded = self._embed(X)
        return self._estimator.decision_function(embedded)

    def _embed(self, X: ndarray) -> ndarray:
        x_tensor = Tensor(X)
        embedded: Tensor = self.embedding_fun(x_tensor)
        embedded = embedded
        means = embedded.mean(1, keepdim=True)
        stds = embedded.std(1, keepdim=True)
        standardized = (embedded - means) / stds
        normalized = torch.nn.functional.normalize(standardized, p=2, dim=1)
        return normalized.cpu().numpy()

