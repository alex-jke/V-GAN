from abc import ABC, abstractmethod
from typing import List, Callable

import numpy as np
import torch
from numpy import ndarray
from torch import Tensor

from .embedding import Embedding
from .tokenizer import Tokenizer
from ..UI.cli import ConsoleUserInterface


class HuggingModel(Tokenizer, Embedding, ABC):

    @property
    @abstractmethod
    def _model_name(self):
        pass


    @property
    @abstractmethod
    def _tokenizer(self):
        pass

    @property
    @abstractmethod
    def _model(self):
        pass


    def __init__(self):
        self.device = torch.device('cuda:0' if torch.cuda.is_available(
        ) else 'mps:0' if torch.backends.mps.is_available() else 'cpu')
        print(f"Using device: {self.device}, cuda: {torch.cuda.is_available()}, mps: {torch.backends.mps.is_available()}")
        self.tokenizer = self._tokenizer
        self.model = self._model.to(self.device)
        self.model_name = self._model_name
        self.padding_token = self._padding_token

    def tokenize(self, data: str) -> List[int]:
        tokenized = self.tokenizer(data, return_tensors='pt')
        input_ids = tokenized['input_ids']
        input_list = input_ids.tolist()
        first_elem = input_list[0]
        return first_elem

    def tokenize_batch(self, data: List[str]) -> Tensor:
        tokenized_list = [Tensor(self.tokenize(d)) for d in data]
        max_length = max([len(t) for t in tokenized_list])
        padded_token_list = [torch.nn.functional.pad(t, (0, max_length - len(t)), value=self.padding_token) for t in tokenized_list]
        tensor = torch.stack(padded_token_list)
        return tensor.int()

    def detokenize(self, words: List[int]) -> str:
        return self.tokenizer.decode(words)

    def embed(self, data: str) -> np.ndarray:
        tokenized = self.tokenize(data)
        return self.embed_tokenized(tokenized)

    def embed_words(self, words: List[str]) -> List[np.ndarray]:
        return [self.embed(data=word) for word in words]

    def decode(self, embedding: np.ndarray) -> str:
        with torch.no_grad():
            output = self.decode2tokenized(embedding)
            decoded = self.detokenize([output])
        return decoded

    @abstractmethod
    def embed_tokenized(self, tokenized: List[int]) -> List[np.ndarray]:
        pass

    @abstractmethod
    def fully_embed_tokenized(self, tokenized: Tensor) -> Tensor:
        """
        This method expects a one-dimensional tensor of token indices and returns the corresponding embeddings.
        :param tokenized: 1D Tensor of token indices.
        :return: two-dimensional Tensor where each token index is an embedding. (embedding_size, num_tokens)
        """
        pass
    @abstractmethod
    def decode2tokenized(self, embedding: List[np.ndarray]) -> List[int]:
        pass

    @property
    def _padding_token(self) -> int:
        token = self.tokenizer.pad_token_id
        if token is None:
            token = self.tokenizer.eos_token_id
        if token is None:
            raise ValueError("No padding token found in tokenizer.")
        #return token
        return 0 #todo: might cause conflict with another symbol

    def aggregateEmbeddings(self, embeddings: Tensor):
        """
        The function defines how given a collection of embeddings it is aggregated into a single embedding.
        :param embeddings: A three-dimensional tensor of embeddings. The first dimension is the size of the embeddings,
        the second is the batch dimension and the third is the number of tokens
        :return: A two-dimensional Tensor, where the first dimension is the batch dimension and the second is the
        embedding dimension
        """
        aggregated = embeddings.mean(dim=-1)
        return aggregated

    def get_embedding_fun(self, batch_first = False) -> Callable[[Tensor], Tensor]:
        def embedding(data: Tensor) -> Tensor:
            """
            This method takes a tensor of tokenized datapoints and returns the embeddings.
            :param data: A two-dimensional tensor of tokenized datapoints. The first dimension is the number of datapoints and the
            second is the number of tokens.
            :return: A two-dimensional Tensor aggregated in accordance to the aggregate function.
            """
            embeddings = torch.tensor([], dtype=torch.int).to(self.device)
            ui = ConsoleUserInterface.get()
            with torch.no_grad(), ui.display():

                for (i, partial_review) in enumerate(data):
                    ui.update(f"Embedding {i+1}/{len(data)}")
                    partial_review: Tensor
                    embedded: Tensor = self.fully_embed_tokenized(partial_review.int()) #returns a (embedding_size, num_tokens) tensor
                     #add extra third dimension
                    unsqueezed = embedded.unsqueeze(1)
                    aggregated = self.aggregateEmbeddings(embeddings = unsqueezed)

                    try:
                        embeddings = torch.cat((embeddings, aggregated), dim=1)
                    except:
                        print(embeddings.shape, aggregated.shape)
                        raise
                #aggregated = self.aggregateEmbeddings(embeddings = embeddings)
            if batch_first:
                return embeddings.T
            ui.done()
            return embeddings
        return embedding

    def max_token_length(self) -> int:
        return self.tokenizer.model_max_length



