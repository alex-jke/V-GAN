from typing import Tuple, List, Type

import numpy as np
from pyod.models.base import BaseDetector
from pyod.models.lof import LOF
from pyod.models.lunar import LUNAR
from torch import Tensor

from modules.od_module import VMMD_od

from text.Embedding.gpt2 import GPT2
from text.dataset.SimpleDataset import SimpleDataset
from text.dataset.emotions import EmotionDataset
from text.outlier_detection.odm import OutlierDetectionModel
from vmmd import VMMD


class VGAN_ODM(OutlierDetectionModel):

    def __init__(self, dataset, model, train_size, test_size, inlier_label=None, base_detector: Type[BaseDetector] = None, use_embedding=False):
        self.vgan = VMMD(epochs=400, weight=0.1, lr=0.5)
        self.number_of_subspaces = 500
        self.base_detector: Type[BaseDetector] = base_detector
        if base_detector is None:
            self.base_detector = LUNAR
        self.detectors: List[BaseDetector] = []
        self.init_dataset = self.use_embedding if use_embedding else self.use_tokenized
        super().__init__(dataset, model, train_size, test_size, inlier_label)

    def train(self):
        self.init_dataset()
        train = self.x_train.to(self.device)
        self.vgan.fit(train)
        subspaces = self.vgan.generate_subspaces(self.number_of_subspaces)
        unique_subspaces, proba = np.unique(
            np.array(subspaces.to('cpu')), axis=0, return_counts=True)
        self.subspaces = Tensor(unique_subspaces).to(self.device)
        self.proba = proba / proba.sum()

        # Project the dataset into each of the generated subspaces
        projected_datasets = [self.project_dataset(train, subspace) for subspace in self.subspaces]
        # Fit a detector on each of the projected datasets
        self.detectors = [self.base_detector().fit(projected_dataset.cpu().numpy()) for projected_dataset in projected_datasets]

    def predict(self):
        test = self.x_test.to(self.device)
        projected_datasets = [self.project_dataset(test, subspace) for subspace in self.subspaces]
        # Predict on each of the projected test datasets
        predictions =Tensor([detector.predict(projected_test.cpu().numpy()) for detector, projected_test in zip(self.detectors, projected_datasets)])
        # Combine the predictions of each detector weighted by the probability of the subspace
        predictions_aggregated = (predictions.T * self.proba).sum(dim=1)
        self.predictions = predictions_aggregated.round().int().tolist()

    def _get_name(self):
        return f"VGAN + {self.base_detector.__name__}"

    def _get_predictions(self) -> List[int]:
        return self.predictions

    def get_space(self):
        return "Tokenized"

    def project_dataset(self, dataset: Tensor, subspace: Tensor) -> Tensor:
        """
        :param dataset: Dataset to project. A tensor of shape (n_samples, n_features)
        :param subspace: Subspace to project on. A tensor of shape (n_features)

        :return: Projected dataset. A tensor of shape (n_samples, n_features)
        """
        #alt_approach = dataset @ subspace
        # expand subspace to match dataset shape
        subspace_expanded = subspace.expand(dataset.shape[0], -1)
        projected = dataset * subspace_expanded
        return projected

if __name__ == '__main__':
    #vmmd = VGAN_ODM(SimpleDataset(["This is an example"], 300), GPT2(), 200, 100)
    vmmd = VGAN_ODM(EmotionDataset(), GPT2(), 200, 100)
    vmmd.train()
    vmmd.predict()
    vmmd.evaluate()