from abc import ABC, abstractmethod
from typing import List, Tuple

import pandas as pd
from transformers.models.cvt.convert_cvt_original_pytorch_checkpoint_to_pytorch import embeddings

from text.Embedding.deepseek import DeepSeek1B
from text.Embedding.gpt2 import GPT2
from text.Embedding.huggingmodel import HuggingModel
from text.dataset.dataset import Dataset
from text.dataset.emotions import EmotionDataset
from text.dataset_converter.dataset_tokenizer import DatasetTokenizer
from text.outlier_detection.odm import OutlierDetectionModel
from sklearn.neighbors import LocalOutlierFactor


class LOF(OutlierDetectionModel, ABC):

    def __init__(self, dataset: Dataset, model: HuggingModel, train_size: int, test_size: int, label: int | None = None):
        self.predicted_inlier = self.actual_inlier = None
        self.lof = LocalOutlierFactor()
        super().__init__(dataset=dataset, model=model, train_size=train_size, test_size=test_size, inlier_label=label)

    def train(self):
        self.lof.fit(self.x_train.cpu())

    def predict(self):
        predictions = self.lof.fit_predict(self.x_test.cpu())
        self.predicted_inlier = [1 if x == 1 else 0 for x in predictions]
        self.actual_inlier = [1 if x == self.inlier_label else 0 for x in self.y_test]

    def _get_predictions_expected(self) -> Tuple[List[int], List[int]]:
        if self.predicted_inlier is None or self.actual_inlier is None:
            self.predict()
        return self.predicted_inlier, self.actual_inlier


class EmbeddingLOF(LOF):

    def __init__(self, dataset: Dataset, model: HuggingModel, train_size: int, test_size: int, label: int | None = None):
        super().__init__(dataset, model, train_size, test_size, label)
        self.use_embedding()

    def _get_name(self):
        return f"EmbeddingLOF_{self.model.model_name}_{self.dataset.name}"

if __name__ == '__main__':
    current_time = pd.Timestamp.now()
    lof = EmbeddingLOF(dataset=EmotionDataset(), model=DeepSeek1B(), train_size=100, test_size=10)
    lof.train()
    lof.predict()
    print(lof.evaluate())
    print("Time taken: ", pd.Timestamp.now() - current_time)
    #print(lof.y_test.tolist())
