import gc
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional

import numpy as np
import torch
from torch import Tensor

from text.Embedding.unification_strategy import UnificationStrategy, StrategyInstance
from text.UI import cli
from text.UI.cli import ConsoleUserInterface
from text.dataset.aggregatable import Aggregatable
from text.dataset.dataset import Dataset


ui = cli.get()

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
    def embed_words(self, words: List[str], mask: Optional[Tensor], strategy: StrategyInstance) -> np.ndarray:
        """
        Embeds a list of words into vectors.
        :param words: The list of words to embed.
        :param mask: The mask to use. If None, all words are embedded.
        :param strategy: The unification strategy to use.
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

    def embed_sentences(self, sentences: np.ndarray[str], seperator: str = " ", masks: Optional[Tensor] = None,
                        strategy: StrategyInstance = UnificationStrategy.TRANSFORMER.create(), dataset: Optional[Aggregatable] = None,
                        verbose: bool = False) -> Tensor:
        """
        Embeds a list of sentences into vectors. It splits the sentences into words using the seperator.
        It then embeds the words into vectors and pads them to the padding length.
        :param sentences: The list of sentences to embed as a numpy array of sentences as strings.
        :param seperator: The seperator to split the sentences into words.
        :param masks: The masks to use. If None, all words are embedded.
            The mask tensor should be of shape: (n_sentences, padding_length).
            Alternatively, the mask can be a single mask, which is then used for all sentences.
        :param strategy: The unification strategy to use. As each sentence is of different length and a tensor is returned,
            a common format needs to be chosen. This can, for example, be setting a padding length, taking the mean over
            all embedding vectors or letting the LLM aggregate the vector by way of prompting.
        :param dataset: The dataset to use as an aggregatable dataset.
            It provides a prefix and suffix to the sentences to allow in-context learning.
            Only used if aggregate is True.
        :param verbose: A boolean, whether to display the progress to the console.
        :return: The embeddings of the sentences, as a tensor.
            The shape of the tensor should be (n_sentences, n_words, n_embedding_dim).
            Note, that if aggregated, the n_words dimension is 1.
        """

        if strategy.equals(UnificationStrategy.PADDING):
            return self._get_sentence_embeddings(sentences, seperator, masks, strategy, dataset, verbose)

        # This is done, as this avoids a memory leak, when embedding all of them at once. TODO: why is this happening?
        embeddings = []
        amount_sentences = len(sentences)
        with ui.display():
            for i in range(amount_sentences):
                if verbose:
                    ui.update(f"Embedding sentence {i + 1}/{amount_sentences}")
                sentence = sentences[i:i+1]
                mask = masks
                if masks is not None:
                    mask = masks[i:i+1] if len(masks.shape) > 1 else masks
                embedding = self._get_sentence_embeddings(sentence, seperator, mask, strategy, dataset, False)
                embeddings.append(embedding)
        stacked = torch.stack(embeddings)
        assert stacked.shape[1] == 1
        stacked = stacked.mean(dim=1) #dim one just contains 1 entry
        return stacked
        # TODO: Currently, the Embedding based approach means and normalizes the embeddings.
        # Would that make sense here? The main problem is that the stacked tensor might be 3D, if the padding strategy is used.
        # Since it is unclear, what the downstream task is, it is not clear, if this is the right approach.
        # If for example, the downstream task then means the embeddings, the vectors are not normalized.


    def _get_sentence_embeddings(self, sentences: np.ndarray[str], seperator: str = " ", masks: Optional[Tensor] = None,
                        strategy: StrategyInstance = UnificationStrategy.TRANSFORMER.create(), dataset: Optional[Aggregatable] = None,
                        verbose: bool = False) -> Tensor:

        transformer_agg = strategy.equals(UnificationStrategy.TRANSFORMER)
        padding_strategy = strategy.equals(UnificationStrategy.PADDING)
        padding_length = strategy.param if padding_strategy else None

        if transformer_agg and dataset is None:
            raise ValueError("If the aggregation strategy is Transformer, dataset must be provided.")

        if transformer_agg:
            self.prefix = dataset.prefix()
            self.suffix = dataset.suffix()

        embeddings = []
        with ui.display():
            for i in range(len(sentences)):
                #if verbose:
                    #ui.update(f"Embedding sentence {i + 1}/{sentences.shape[0]}")
                sentence = sentences[i:i+1]
                embedded = self.loop_body(sentence, i, seperator, masks, padding_length, padding_strategy, strategy)
                embeddings.append(embedded)
        stacked = torch.stack(embeddings)
        return stacked

    def loop_body(self, sentence: np.ndarray[str], i: int, seperator: str, masks: Optional[Tensor], padding_length: Optional[int], padding_strategy: bool, strategy: StrategyInstance):

        sentence = sentence[0]
        words = self.get_words(sentence, seperator)
        words = [word for word in words if
                 word != ""]  # Remove the empty strings, which can occur if two consecutive spaces are in the data. This causes NaN embeddings.

        words, mask = self._update_words_and_mask(words, masks, i, padding_length)

        embedded = self._get_sentence_embedding(words, mask, padding_strategy, strategy, padding_length)

        return embedded


    @staticmethod
    def _update_words_and_mask(words: List[str], masks: Optional[Tensor], i: int, padding_length: Optional[int]) -> Tuple[List[str], Optional[Tensor]]:
        mask = None
        if masks is not None:
            mask = masks[i] if len(masks.shape) > 1 else masks
            if len(words) > mask.shape[0]:
                words = words[ :mask.shape[0]]  #this now only applies if mask is set, as if everything after is masked
            elif len(words) < mask.shape[0]:
                mask = mask[:len(words)]
            assert len(words) == mask.shape[0], "Mask and Words should have same length, after being trimmed."

        if padding_length is not None and len(words) > padding_length:
            words = words[:padding_length]

        elif padding_length is not None and len(words) < padding_length and masks is not None:
            mask = mask[:len(words)]

        return words, mask

    def _get_sentence_embedding(self, words: List[str], mask: Optional[Tensor], padding_strategy: bool, strategy: StrategyInstance, padding_length: Optional[int]) -> Tensor:
        embedded = self.embed_words(words, mask, strategy)
        if padding_strategy:
            if embedded.shape[0] < padding_length:
                embedded = torch.nn.functional.pad(embedded, (0, 0, 0, padding_length - embedded.shape[0]))
            # elif embedded.shape[0] > padding_length > 0:
            # embedded = embedded[:padding_length]
            if embedded.shape[0] != padding_length:
                raise ValueError(f"Expected shape of {padding_length}, but got {embedded.shape[0]}")
        if torch.isnan(embedded).any():
            raise ValueError("Computed Embedding with NaN values.")
        return embedded

