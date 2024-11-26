from cmath import sqrt
from typing import List, Dict, Tuple

import numpy as np
import torch
from torch import Tensor
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Model

from .huggingmodel import HuggingModel


class GPT2(HuggingModel):
    def __init__(self, max_length=300):

        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        model = GPT2Model.from_pretrained("gpt2")
        self.embedded_cache: Dict[int, Tensor] = {}
        super().__init__("gpt2", tokenizer, model)

    def decode2tokenized(self, embedding: List[np.ndarray]) -> List[int]:
        """
        This method takes a list of token embeddings and returns the closest token index for each.
        :param embedding: A list of embeddings.
        :return: A list of token indices closest to each embedding.
        """
        # Convert list of numpy embeddings to a single tensor and ensure dtype is float32
        embeddings = torch.tensor(np.stack(embedding), dtype=torch.float32)
        # Retrieve token embeddings for the vocabulary and ensure they are also float32
        token_embeddings = self.get_token_embeddings(dtype=embeddings.dtype)
        # Compute cosine similarity between embeddings and vocab embeddings
        embeddings = torch.nn.functional.normalize(embeddings, dim=1)
        token_embeddings = torch.nn.functional.normalize(token_embeddings, dim=1)

        similarities = torch.matmul(embeddings, token_embeddings.T)
        # Retrieve the index of the most similar token for each embedding
        indices = torch.argmax(similarities, dim=1).tolist()
        # Convert tensor to list of integers
        return indices

    def embed_tokenized(self, tokenized: List[int]) -> List[np.ndarray]:
        token_vecs = self.tokens_to_vecs(tokenized)
        # Convert list of token vectors to a tensor
        tokenized = torch.tensor(token_vecs)
        #tokenized = torch.tensor(tokenized).unsqueeze(0)
        with torch.no_grad():
            #outputs = self.model(tokenized)
            #embeddings = outputs[0]
            embedding_matrix = self.get_token_embeddings(dtype=tokenized.dtype)
            embeddings = torch.matmul(tokenized, embedding_matrix)
        # Convert tensor to list of numpy arrays
        embeddings = embeddings.tolist()
        embeddings = [np.array(embedding) for embedding in embeddings]
        return embeddings

    def fully_embed_tokenized(self, tokenized: Tensor) -> Tensor:
        """
        This method takes a list of token indices and returns the corresponding embeddings.
        The embeddings are taken from the last layer of the model.
        :param tokenized: A list of token indices.
        :return: A two-dimensional Tensor where each token index is an embedding. (embedding_size, num_tokens)
        """
        key = hash(tokenized)
        cached = self.embedded_cache.get(key)
        if cached is not None:
            return cached

        token_vec = torch.tensor(tokenized).to(self.device)
        with torch.no_grad():
            outputs = self.model(token_vec)
            embeddings = outputs.last_hidden_state.T

        self.embedded_cache[key] = embeddings
        return embeddings

    def tokens_to_vecs(self, token: List[int]) -> Tensor:
        """
        This method takes a list of token indices and returns the corresponding vectors.
        The vectors are the size of the vocabulary.
        :param token: the indices of where the vectors will be one
        :return: A two-dimensional Tensor where each token index is a one-hot vector.
        """
        tensor = torch.nn.functional.one_hot(torch.tensor(token), num_classes=self.tokenizer.vocab_size)
        return tensor
    def get_token_embeddings(self, dtype = torch.float32) -> Tensor:
        return self.model.transformer.wte.weight.data.to(dtype)



if __name__ == '__main__':
    gpt2 = GPT2()
    text = "Hello, world! This is a test."
    print("Original:", text)


    tokenized = gpt2.tokenize(text)
    print("Tokenized:", tokenized)

    embedded: List[np.array] = gpt2.embed_tokenized(tokenized)
    #only print the first 100 symbols
    print("Embedded:", str(embedded)[:100])

    #deemedbedded = gpt2.decode2tokenized(embedded)
    #print("Decoded embedding to tokens:", gpt2.decode2tokenized(embedded))

    #detokenized = gpt2.detokenize(deemedbedded)
    #print("Detokenized actual:", detokenized)

    detokenized = gpt2.detokenize(tokenized)
    print("Detokenized should:", detokenized)