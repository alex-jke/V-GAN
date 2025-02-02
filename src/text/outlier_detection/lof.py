from abc import ABC, abstractmethod
from typing import List

import pandas as pd
from transformers.models.cvt.convert_cvt_original_pytorch_checkpoint_to_pytorch import embeddings

from text.Embedding.gpt2 import GPT2
from text.Embedding.huggingmodel import HuggingModel
from text.dataset.dataset import Dataset
from text.dataset.emotions import EmotionDataset
from text.dataset_converter.dataset_tokenizer import DatasetTokenizer
from text.outlier_detection.odm import OutlierDetectionModel
from sklearn.neighbors import LocalOutlierFactor


class LOF(OutlierDetectionModel, ABC):

    def __init__(self, dataset: Dataset, model: HuggingModel, train_size: int, test_size: int):
        self.x_train = None
        self.x_test = None
        self.y_test: List[int] = None
        self.y_train: int = None
        self.lof = LocalOutlierFactor()
        super().__init__(dataset=dataset, model=model, train_size=train_size, test_size=test_size)
        self.prepareDataset()

    def train(self):
        self.lof.fit(self.x_train.cpu())

    def predict(self):
        predictions = self.lof.fit_predict(self.x_test.cpu())
        predicted_inlier = [1 if x == 1 else 0 for x in predictions]
        actual_inlier = [1 if x == self.y_train else 0 for x in self.y_test]
        correct_predictions = [1 if x == y else 0 for x, y in zip(predicted_inlier, actual_inlier)]
        return (f"Method: {self.name}\n"
                f"\tcorrect: {sum(correct_predictions) / len(correct_predictions) * 100}%\n"
                f"\tpercentage inlier: {sum(actual_inlier) / len(actual_inlier) * 100}%")

    @abstractmethod
    def prepareDataset(self):
        pass

class EmbeddingLOF(LOF):

    def prepareDataset(self):
        train_x, train_y = self.dataset.get_training_data()
        self.y_train = train_y[0]
        filtered_data_series: pd.Series = train_x[train_y == self.y_train][:self.train_size]
        filtered_data: List[str] = filtered_data_series.tolist()
        tokenized_train = self.model.tokenize_batch(filtered_data)

        test_x, test_y = self.dataset.get_testing_data()
        self.y_test = test_y[:self.test_size]
        filtered_data_series = test_x[:self.test_size]
        filtered_data = filtered_data_series.tolist()
        tokenized_test = self.model.tokenize_batch(filtered_data)

        embedding_fun = self.model.get_embedding_fun()

        self.x_train = embedding_fun(tokenized_train).permute(1,0)
        self.x_test = embedding_fun(tokenized_test).permute(1,0)
        (_,y_test) = self.dataset.get_testing_data()
        self.y_test = y_test[:self.test_size]

    def _get_name(self):
        return f"EmbeddingLOF_{self.model.model_name}_{self.dataset.name}"

if __name__ == '__main__':
    lof = EmbeddingLOF(dataset=EmotionDataset(), model=GPT2(), train_size=1000, test_size=100)
    lof.train()
    print(lof.predict())
    #print(lof.y_test.tolist())
