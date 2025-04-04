from abc import ABC, abstractmethod
from typing import List, Tuple, Optional

import numpy as np
import torch
from torch import Tensor

from text.UI.cli import ConsoleUserInterface
from text.dataset.aggregatable import Aggregatable
from text.dataset.dataset import Dataset
ui = ConsoleUserInterface()

class Embedding(ABC):

    def __init__(self):
        """
        Initializes the embedding model.
        """
        self.prefix: Optional[List[str]] = None
        self.suffix: Optional[List[str]] = None

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
    def embed_words(self, words: List[str], mask: Optional[Tensor], aggregate: bool) -> np.ndarray:
        """
        Embeds a list of words into vectors.
        :param words: The list of words to embed.
        :param mask: The mask to use. If None, all words are embedded.
        :param aggregate: If True, the embeddings are aggregated to a single vector.
            For static embeddings, this is the mean of the embeddings.
            For dynamic embeddings, such as transformers, this is the last hidden state of the
            last suffix token.
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

    def embed_sentences(self, sentences: np.ndarray[str], padding_length: Optional[int] = None, seperator: str = " ", masks: Optional[Tensor] = None,
                        aggregate: bool = True, dataset: Optional[Aggregatable] = None) -> Tensor:
        """
        Embeds a list of sentences into vectors. It splits the sentences into words using the seperator.
        It then embeds the words into vectors and pads them to the padding length.
        :param sentences: The list of sentences to embed as a numpy array of sentences as strings.
        :param padding_length: The length to pad the sentences to. The sentences will be padded with zero vectors.
            If -1 is given, the sentences will not be truncated. Which does mean that torch will give a stack error,
            if the sentences are not of the same length.
            Padding length only is relevant if aggregate is False.
        :param seperator: The seperator to split the sentences into words.
        :param masks: The masks to use. If None, all words are embedded.
            The mask tensor should be of shape: (n_sentences, padding_length).
            Alternatively, the mask can be a single mask, which is then used for all sentences.
        :param aggregate: If True, the embeddings are aggregated to a single tensor.
            This is the mean of the embeddings for static embeddings.
            For dynamic embeddings, such as transformers, this is the last hidden state of the
            last suffix token.
            If False, the embeddings are returned as a list of tensors.
        :param dataset: The dataset to use as an aggregatable dataset.
            It provides a prefix and suffix to the sentences to allow in-context learning.
            Only used if aggregate is True.
        :return: The embeddings of the sentences, as a tensor.
            The shape of the tensor should be (n_sentences, n_words, n_embedding_dim).
        """

        if aggregate and dataset is None:
            raise ValueError("If aggregate is True, dataset must be provided.")

        if aggregate:
            self.prefix = dataset.prefix()
            self.suffix = dataset.suffix()

        assert padding_length is not None or aggregate, "Padding length needs to be set if aggregate is not set to True"

        embeddings = []
        #with ui.display():
        for i in range(len(sentences)):
            #print("|", end="")
            #ui.update(f"Embedding sentence {i + 1}/{len(sentences)}")
            sentence = sentences[i]
            mask = None
            if masks is not None:
                mask = masks[i] if len(masks.shape) > 1 else masks
            words = self.get_words(sentence, seperator)
            if padding_length is not None and len(words) > padding_length:
                words = words[:padding_length]
            elif padding_length is not None and len(words) < padding_length and masks is not None:
                mask = mask[:len(words)]
            embedded = self.embed_words(words, mask, aggregate)
            if not aggregate:
                if embedded.shape[0] < padding_length:
                    embedded = torch.nn.functional.pad(embedded, (0, 0, 0, padding_length - embedded.shape[0]))
                #elif embedded.shape[0] > padding_length > 0:
                    #embedded = embedded[:padding_length]
                if embedded.shape[0] != padding_length:
                    raise ValueError(f"Expected shape of {padding_length}, but got {embedded.shape[0]}")
            embeddings.append(embedded)
        try:
            stacked =  torch.stack(embeddings)
        except RuntimeError as e:
            raise e
        return stacked
