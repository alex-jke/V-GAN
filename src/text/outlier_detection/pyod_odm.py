from abc import abstractmethod, ABC
from typing import Tuple, List, Type, Callable

from numpy import ndarray
from prompt_toolkit.layout.processors import Transformation
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
from text.outlier_detection.space.space import Space
from text.outlier_detection.space.token_space import TokenSpace


class PyODM(OutlierDetectionModel, ABC):
    def __init__(self, dataset: Dataset, space: Space, base_detector: Type[BaseDetector], use_cached = False):
        #self.space = "Embedding" if pre_embed else "Tokenized"
        super().__init__(dataset=dataset, space=space, use_cached=use_cached, base_method=base_detector)
        self.od_model = self._get_model()

    def _train(self):
        self.od_model.fit(self.x_train.cpu().numpy(), None)

    def _predict(self):
        test = self.x_test
        decision_function = self.od_model.decision_function(test.cpu().numpy())
        self.decision_function = decision_function

    def _get_predictions(self) -> List[int]:
        return self.decision_function

    @abstractmethod
    def _get_model(self) -> BaseDetector:
        pass

class BasePyODM(PyODM, ABC):
    def __init__(self, dataset: Dataset, space: Space, use_cached = False):
        super().__init__(dataset, space, self._get_model().__class__, use_cached)

#TODO: set the contamination parameter, as there is no contamination.
class LOF(BasePyODM):
    def _get_model(self):
        return pyod_LOF()

    def _get_name(self):
        return f"LOF" + f" + {self.get_space()}"

class LUNAR(BasePyODM):
    def _get_model(self):
        return pyod_LUNAR()

    def _get_name(self):
        return f"LUNAR" + f" + {self.get_space()}"

class ECOD(BasePyODM):
    def _get_model(self):
        return pyod_ECOD()

    def _get_name(self):
        return f"ECOD" + f" + {self.get_space()}"

class FeatureBagging(PyODM):

    def __init__(self, dataset: Dataset, space: Space, base_detector:  Type[BaseDetector], use_cached = False):
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

        if isinstance(space, TokenSpace):
            transformation = TransformBaseDetector.configure_transform_fun(
                transformation= space.model.get_embedding_fun(batch_first=True),
                normalize=True,
                standardize=True
            )
            self.base_estimator = TransformBaseDetector(transformation, lambda: base_detector)

        self.__model = pyod_FeatureBagging(base_estimator=self.base_estimator)
        super().__init__(dataset=dataset, space=space, use_cached=use_cached, base_detector=base_detector)

    def _get_model(self):
        return self.__model

    def _get_name(self):
        return f"FeatureBagging + {self.base_name} + {self.get_space()}"

class TransformBaseDetector(BaseDetector):
    """
    A wrapper for a base detector that transforms the input data before passing it to the base detector.
    An example of this is to give the base detector tokenized data, while the model is trained on embeddings.
    """

    def __init__(self, transform_fun: Callable[[ndarray], ndarray],
                 base_detector: Callable[[],Type[BaseDetector]]):
        # self.model = model #saving the model will cause CUDA out of memory as model is large
        self.base_detector: Callable[[], Type[BaseDetector]] = base_detector # This is required because, when the base detectors are duplicated in the feature bagging,
        # instead for each param to check if its of type BaseDetector it checks if it has the get_params method.
        # Then an error it caused was it trying to call get_params on the base_detector class, that self was not passed.
        #self.embedding_fun = self.model.get_embedding_fun(batch_first=True)
        self.transform_fun = transform_fun
        self._classes = 2
        super().__init__()

    def fit(self, X: ndarray, y=None):
        self._estimator = self.base_detector()(contamination = self.contamination)
        embedded = self.transform_fun(X)
        self._estimator.fit(embedded, y)
        self.decision_scores_ = self._estimator.decision_scores_
        self.threshold_ = self._estimator.threshold_
        self.labels_ = self._estimator.labels_

    def decision_function(self, X):
        embedded = self.transform_fun(X)
        return self._estimator.decision_function(embedded)

    @staticmethod
    def configure_transform_fun(transformation: Callable[[Tensor], Tensor],
                                normalize: bool, standardize: bool) -> Callable[[ndarray], ndarray]:
        """
        Configure the transformation to apply to the data.
        :param transformation: The base transformation to apply.
        :param normalize: Whether to normalize the data after the transformation
        :param standardize: Whether to standardize the data before applying the transformation.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else
                              ("mps" if torch.backends.mps.is_available() else "cpu"))
        def transform(x: ndarray) -> ndarray:
            """
            Transforms the data, given the base transformation to apply.
            """
            x_tensor = Tensor(x).to(device)
            embedded: Tensor = transformation(x_tensor)
            standardized = embedded
            if standardize:
                #todo: is using dimension 1 correct?
                #raise NotImplementedError("Check TODO.")
                means = embedded.mean(1, keepdim=True)
                stds = embedded.std(1, keepdim=True)
                standardized = (embedded - means) / stds
            normalized = standardized
            if normalize:
                normalized = torch.nn.functional.normalize(standardized, p=2, dim=1)
            return normalized.cpu().numpy()
        return transform

