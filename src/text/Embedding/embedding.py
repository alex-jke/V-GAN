from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np
import torch
from torch import Tensor

from text.UI.cli import ConsoleUserInterface
from text.dataset.dataset import Dataset
ui = ConsoleUserInterface()

class Embedding(ABC):

    @abstractmethod
    def embed(self, data: str) -> np.ndarray:
        """
        Embeds a single string into a vector.
        :param data: The string to embed.
        :return: The embedding of the string, as a numpy array, representing one vector.
            The shape of the array should be (n_dim,).
        """
        pass

    @abstractmethod
    def embed_words(self, words: List[str]) -> np.ndarray:
        """
        Embeds a list of words into vectors.
        :param words: The list of words to embed.
        :return: The embeddings of the words, as a numpy array, representing a list of vectors.
            The shape of the array should be (n_words, n_dim).
        """
        pass

    def get_words(self, sentence: str, seperator: str = " ") -> List[str]:
        """
        Gets a list of words from the sentence. Seperator is used to separate words.
        :param sentence: The sentence to get words from.
        :param seperator: The separator to use to separate words.
        :return: The list of words.
        """
        return sentence.split(seperator)

    def embed_sentences(self, sentences: np.ndarray[str], padding_length: int, seperator: str = " ") -> Tensor:
        """
        Embeds a list of sentences into vectors. It splits the sentences into words using the seperator.
        It then embeds the words into vectors and pads them to the padding length.
        :param sentences: The list of sentences to embed as a numpy array of sentences as strings.
        :param padding_length: The length to pad the sentences to. The sentences will be padded with zero vectors.
        :param seperator: The seperator to split the sentences into words.
        :return: The embeddings of the sentences, as a tensor.
            The shape of the tensor should be (n_sentences, n_words, n_embedding_dim).
        """
        embeddings = []
        i = 0
        with ui.display():
            for sentence in sentences:
                ui.update(f"Embedding sentence: {i}/{len(sentences)}")
                i += 1
                words = self.get_words(sentence, seperator)
                embedded = Tensor(self.embed_words(words))
                if len(words) < padding_length:
                    embedded = torch.nn.functional.pad(embedded, (0, 0, 0, padding_length - len(words)))
                elif len(words) > padding_length:
                    embedded = embedded[:padding_length]
                embeddings.append(embedded)
            stacked =  torch.stack(embeddings)
        return stacked
