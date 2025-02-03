from abc import ABC, abstractmethod
from typing import Tuple, List

from text.Embedding.huggingmodel import HuggingModel
from text.dataset.dataset import Dataset


from sklearn.metrics import roc_auc_score
from typing import List, Tuple
from abc import ABC, abstractmethod

class OutlierDetectionModel(ABC):

    def __init__(self, dataset: Dataset, model: HuggingModel, train_size: int, test_size: int):
        self.dataset = dataset
        self.model = model
        self.train_size = train_size
        self.test_size = test_size
        self.name = self._get_name()

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def predict(self):
        pass

    @abstractmethod
    def _get_name(self):
        pass

    @abstractmethod
    def _get_predictions_expected(self) -> Tuple[List[int], List[int]]:
        pass

    def evaluate(self):
        # Get predicted and actual labels
        predicted_inlier, actual_inlier = self._get_predictions_expected()

        # Calculate accuracy
        correct_predictions = [1 if x == y else 0 for x, y in zip(predicted_inlier, actual_inlier)]
        accuracy = sum(correct_predictions) / len(correct_predictions)

        # Calculate true positives, false positives, and false negatives
        true_positives = sum([1 if x == 1 and y == 1 else 0 for x, y in zip(predicted_inlier, actual_inlier)])
        false_positives = sum([1 if x == 1 and y == 0 else 0 for x, y in zip(predicted_inlier, actual_inlier)])
        false_negatives = sum([1 if x == 0 and y == 1 else 0 for x, y in zip(predicted_inlier, actual_inlier)])

        # Calculate recall and precision
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0

        # Calculate AUC
        auc = roc_auc_score(actual_inlier, predicted_inlier)

        # Calculate percentage of inliers and outliers
        percentage_inlier = sum(actual_inlier) / len(actual_inlier) * 100
        percentage_outlier = 100 - percentage_inlier

        # Return evaluation metrics
        return (f"Method: {self.name}\n"
                f"\taccuracy: {accuracy * 100:.2f}%\n"
                f"\trecall: {recall * 100:.2f}%\n"
                f"\tprecision: {precision * 100:.2f}%\n"
                f"\tauc: {auc:.4f}"
                f"\tpercentage inlier: {percentage_inlier:.2f}%\n"
                f"\tpercentage outlier: {percentage_outlier:.2f}%\n"
                f"\ttrue positives: {true_positives}\n"
                f"\tfalse positives: {false_positives}\n"
                f"\tfalse negatives: {false_negatives}\n"
                f"\tamount of samples: {len(actual_inlier)}"
                )
