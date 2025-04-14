import unittest
from typing import Tuple

from text.Embedding.llama import LLama3B
from text.Embedding.unification_strategy import UnificationStrategy
from text.dataset.dataset import AggregatableDataset
from text.dataset.emotions import EmotionDataset
from text.outlier_detection.pyod_odm import LUNAR
from text.outlier_detection.space.embedding_space import EmbeddingSpace
from text.outlier_detection.space.token_space import TokenSpace
from text.outlier_detection.space.word_space import WordSpace


class LunarTest(unittest.TestCase):

    def run_comparison(self, dataset: AggregatableDataset):
        model = LLama3B()
        train_size = 1_000
        test_size = 1_000
        word_space = WordSpace(strategy=UnificationStrategy.TRANSFORMER, model=model, test_size=test_size,
                               train_size=train_size)
        embedding_space = EmbeddingSpace(model=model, train_size=train_size, test_size=test_size)

        shared_params = {
            "dataset": dataset,
            "use_cached": True,
        }

        word_lunar = LUNAR(
            space=word_space,
            **shared_params,
        )
        emb_lunar = LUNAR(
            space=embedding_space,
            **shared_params
        )
        word_lunar.train()
        word_lunar.predict()
        word_metrics = word_lunar.evaluate()[0]

        emb_lunar.train()
        emb_lunar.predict()
        emb_metrics = emb_lunar.evaluate()[0]

        print("Word LUNAR metrics: ", word_metrics)
        print("Embedding LUNAR metrics: ", emb_metrics)


    def test_emotions(self):
        dataset = EmotionDataset()
        self.run_comparison(dataset)

