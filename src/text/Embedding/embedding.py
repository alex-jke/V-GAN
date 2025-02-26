from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np
import torch
from torch import Tensor

from text.dataset.dataset import Dataset


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
            The shape of the array should be (n_words,
        """
        pass

    def embed_with_mask(self, sentences: np.ndarray[str], masks: Tensor, seperator: str = " ") -> Tuple[Tensor, Tensor]:
        """
        Embeds an array of sentences into a single vector for each sentence, using a mask for each.
        The mask will be used to mask out words in the string. If the string has more words than the mask,
        Only the first n words will be used, where n is the length of the mask.
        If the mask has more words than the string, the mask will be truncated.
        :param sentences: The sentences to embed, as a numpy ndarray of strings.
        :param masks: The masks to use. The masks should be a numpy array of shape (n_sentences, n_words).
        :param seperator: The seperator to use to separate the sentences into words.
        :return: A pair of The embeddings of the strings, as a numpy array, representing one vector for each sentence. The shape of the array should be (n_sentences, n_dim):
            - Without the mask applied
            - With the mask applied
        """
        if len(sentences) != masks.shape[0]:
            raise ValueError(f"The number of sentences and masks must match. Got {len(sentences)} sentences and {masks.shape[0]} masks.")
        embeddings = []
        masked_embeddings = []
        for i in range(len(sentences)):
            words: List[str] = sentences[i].split(seperator)
            sentence: np.ndarray[str] = np.array(words, dtype=str)
            embedding, masked_embedding = self.embed_sentence_with_mask(sentence, masks[i].cpu().numpy())
            embeddings.append(embedding)
            masked_embeddings.append(masked_embedding)
        return torch.stack(embeddings), torch.stack(masked_embeddings)


    def embed_sentence_with_mask(self, words: np.ndarray[str], mask: np.ndarray) -> Tuple[Tensor, Tensor]:
        """
        Embeds an array of words into a single vector, using a mask.
        The mask will be used to mask out words in the string. If the string has more words than the mask,
        Only the first n words will be used, where n is the length of the mask.
        If the mask has more words than the string, the mask will be truncated.
        :param words: The words to embed, as a numpy ndarray of strings.
        :param mask: The mask to use. The mask should be a numpy array of shape (n,).
        :return: A pair of The embedding of the string, as a numpy array, representing one vector. The shape of the array should be (n_dim,):
            - Without the mask applied
            - With the mask applied
        """
        length = min(len(words), mask.shape[0])
        masked_embeddings =[]
        embeddings = []
        for i in range(length):
            embedding = self.embed(words[i])
            embeddings.append(embedding)
            if mask[i] == 1:
                masked_embeddings.append(embedding)

        without_mask =  Tensor(np.mean(embeddings, axis=0))
        with_mask = Tensor(np.mean(masked_embeddings, axis=0))
        return without_mask, with_mask
