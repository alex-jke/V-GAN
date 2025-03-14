from cmath import sqrt
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import GPT2Tokenizer, GPT2Model

from src.text.Embedding.huggingmodel import HuggingModel


class GPT2(HuggingModel):

    def __init__(self):
        super().__init__()
        self.embedded_cache: Dict[int, Tensor] = {}

    @property
    def _tokenizer(self):
        return GPT2Tokenizer.from_pretrained("gpt2")

    @property
    def _model_name(self):
        return "gpt2"

    @property
    def _model(self):
        gpt =  GPT2Model.from_pretrained("gpt2").to(self.device)
        #compiled_model = torch.compile(gpt)
        #return compiled_model
        return gpt

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

    def fully_embed_tokenized(self, tokenized: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """
        This method takes a list of token indices and returns the corresponding embeddings.
        The embeddings are taken from the last layer of the model.
        :param tokenized: A list of token indices.
        :param mask: A mask to apply to the tokenized input. If None, no mask is applied.
        :return: A two-dimensional Tensor where each token index is an embedding. (num_tokens, embedding_size)
        """
        #attention_mask = torch.not_equal(tokenized, self.padding_token)

        # Shorten tokenized sequence to only those tokens, that are not padding tokens
        #filtered_tokens = tokenized[attention_mask]
        #filtered_mask = mask[attention_mask] if mask is not None else torch.ones_like(filtered_tokens)

        # If no tokens remain, return a zero tensor with the gradients remaining.
        #if filtered_tokens.shape[0] == 0:
            #zero_tensor = torch.tensor([0]*768).to(self.device).to(dtype=torch.float32).unsqueeze(1)
            #grad_tensor = tokenized.median()
            #assert grad_tensor == 0
            #return zero_tensor + grad_tensor
        max_length = self.model.config.n_positions
        #length = min(filtered_tokens.shape[0], max_length)
        length = min(max_length, tokenized.size(0))
        #token_vec = filtered_tokens[:length]
        token_vec = tokenized[:length]
        #trimmed_mask = filtered_mask[:length]
        trimmed_mask = mask[:length] if mask is not None else torch.ones_like(token_vec)

        # One-hot encode the vector and pass the gradient along. This is only necessary, if the tokenized tensor is of float, which means, that the mask
        # was applied before
        one_hot = F.one_hot(token_vec.long(), self.model.wte.weight.shape[0]).float() + (token_vec - token_vec.detach()).unsqueeze(1)
        weight_mat = self.model.get_input_embeddings().weight
        inputs_embeds = one_hot @ weight_mat
        # Very important, that the batch dimension is added, otherwise the model will still embed, but no attention is applied
        batched_embeddings = inputs_embeds.unsqueeze(0)
        batched_mask = trimmed_mask.unsqueeze(0)
        if mask is not None:
            outputs = self.model(inputs_embeds=batched_embeddings, attention_mask=batched_mask)
        else:
            outputs = self.model(inputs_embeds=batched_embeddings)

        # Remove the batch dimension and return the embeddings
        embeddings = outputs.last_hidden_state[0]

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

    words = gpt2.embed_words(text.split(" "))
    sentence = gpt2.embed(text)
    print(words.equal(sentence))
    #tokenized = gpt2.tokenize(text)
    #print("Tokenized:", tokenized)

    #embedded: List[np.array] = gpt2.embed_tokenized(tokenized)
    #only print the first 100 symbols
    #print("Embedded:", str(embedded)[:100])

    #deemedbedded = gpt2.decode2tokenized(embedded)
    #print("Decoded embedding to tokens:", gpt2.decode2tokenized(embedded))

    #detokenized = gpt2.detokenize(deemedbedded)
    #print("Detokenized actual:", detokenized)

    #detokenized = gpt2.detokenize(tokenized)
    #print("Detokenized should:", detokenized)