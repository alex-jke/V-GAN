import unittest
from typing import List

import numpy as np
import torch

from text.Embedding.LLM.llama import LLama3B, LLama1B
from text.UI import cli
from text.dataset.aggregatable import Aggregatable
from text.dataset.dataset import AggregatableDataset

ui = cli.get()
class MockAggregatable(Aggregatable):

    def prefix(self) -> List[str]:
        return "sentence: word word word, type: artificial \n sentence: ".split(" ")

    def suffix(self) -> List[str]:
        return [",", "type:"]


class MemoryLeakTest(unittest.TestCase):

    def test_emb_model(self):
        """
        Test for memory leak in the full process.
        """
        model = LLama1B()
        sentence : str = " ".join(["word"] * 300)
        sentences: np.ndarray[str] = np.array([sentence for _ in range(15_000)])
        # This method causes no spiking increases in memory usage. Slowly increases.
        #model.embed_sentences(sentences, dataset=MockAggregatable(), verbose=True)

        # This method has some sort of memory leak. It suddenly increases steeply in memory usage.

        model._get_sentence_embeddings(sentences, dataset=MockAggregatable(), verbose=True)
        # -> Memory leak present here -> Now fixed.

    def test_sentence_by_sentence(self):
        """
        Test for memory leak if embedding sentence by sentence.
        """
        model = LLama1B()
        sentence: str = " ".join(["word"] * 300)
        sentences: np.ndarray[str] = np.array([sentence for _ in range(15_000)])
        with ui.display():
            for i in range(sentences.shape[0]):
                ui.update(f"{i} / {sentences.shape[0]}")
                s = sentences[i]
                model.embed_sentences([s], dataset=MockAggregatable(), verbose=True)
        # -> Memory leak is not present here, indicating, that something about storing the embeddings in the model is causing the memory leak.
        # What happens, if we save the embeddings from the model?

    def test_sentence_by_sentence_saving(self):
        """
        Test for memory leak if embedding sentence by sentence.
        """
        model = LLama1B()
        sentence: str = " ".join(["word"] * 300)
        sentences: np.ndarray[str] = np.array([sentence for _ in range(15_000)])
        embeddings = []
        with ui.display():
            for i in range(sentences.shape[0]):
                ui.update(f"{i} / {sentences.shape[0]}")
                s = sentences[i]
                embedding = model.embed_sentences([s], dataset=MockAggregatable(), verbose=True)
                embeddings.append(embedding)
        # -> Memory leak is not present here, indicating, that something about storing the embeddings in the model is causing the memory leak.