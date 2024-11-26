from abc import ABC, abstractmethod
from typing import List, Callable

import numpy as np
import torch
from torch import Tensor
from transformers import AutoTokenizer, AutoModel, BertForMaskedLM, GPT2Tokenizer, GPT2LMHeadModel, GPT2Model

from .embedding import Embedding
from .tokenizer import Tokenizer


class HuggingModel(Tokenizer, Embedding, ABC):

    def __init__(self, model_name, tokenizer, model):
        self.model_name = model_name
        self.tokenizer = tokenizer
        self.device = torch.device('cuda:0' if torch.cuda.is_available(
        ) else 'mps:0' if torch.backends.mps.is_available() else 'cpu')
        self.model = model.to(self.device)

    def tokenize(self, data: str) -> List[int]:
        embedded = self.tokenizer(data, return_tensors='pt')
        return embedded['input_ids'].tolist()[0]

    def detokenize(self, words: List[int]) -> str:
        return self.tokenizer.decode(words)

    def set_tokenizer(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer

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
    def padding_token(self) -> int:
        token = self.tokenizer.pad_token_id
        if token is None:
            token = self.tokenizer.eos_token_id
        if token is None:
            raise ValueError("No padding token found in tokenizer.")
        return token

    def get_embedding_fun(self) -> Callable[[Tensor], Tensor]:
        def embedding(data: Tensor) -> Tensor:
            """
            This method takes a tensor of tokenized reviews and returns the embeddings.
            :param data: A two-dimensional tensor of tokenized reviews. The first dimension is the number of reviews and the
            second is the number of tokens.
            :return: A three-dimensional tensor of embeddings. The first dimension is the size of the embeddings,
            the second is the number of reviews, and the third is the number of tokens.
            """
            embeddings = torch.tensor([], dtype=torch.int).to(self.device)
            with torch.no_grad():
                for (i, partial_review) in enumerate(data):
                    partial_review: Tensor
                    embedded: Tensor = self.fully_embed_tokenized(partial_review.int()) #returns a (embedding_size, num_tokens) tensor
                     #add extra third dimension
                    unsqueezed = embedded.unsqueeze(1)

                    try:
                        embeddings = torch.cat((embeddings, unsqueezed), dim=1)
                    except:
                        print(embeddings.shape, unsqueezed.shape)
                        raise
            return embeddings
        return embedding

    def max_token_length(self) -> int:
        return self.tokenizer.model_max_length

class GPT2(HuggingModel):
    def __init__(self):
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        model = GPT2LMHeadModel.from_pretrained("gpt2")
        super().__init__("gpt2", tokenizer, model)

    def decode2tokenized(self, embedding: np.ndarray) -> int:
        with torch.no_grad():
            outputs = self.model.generate(input_ids=embedding)
        return outputs
class LLama(HuggingModel):
    def __init__(self):
        model_name = 'llama3.2'
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        model = GPT2LMHeadModel.from_pretrained(model_name)
        super().__init__(model_name, tokenizer, model)

if __name__ == '__main__':
    bert = GPT2()
    text = "Hello, world! This is a test."
    print("Original:", text)

    tokenized = bert.tokenize(text)
    print("Tokenized:", tokenized)

    embedded = bert.embed_tokenized(tokenized)
    print("Embedded:", embedded)

    deemedbedded = bert.decode2tokenized(embedded)
    print("Decoded:", bert.decode(embedded))

    detokenized = bert.detokenize(tokenized)
    print("Detokenized:", detokenized)
