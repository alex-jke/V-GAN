from abc import ABC, abstractmethod
from typing import List, Callable, Optional

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F

from .embedding import Embedding
from .tokenizer import Tokenizer
from .unification_strategy import UnificationStrategy, StrategyInstance
from ..UI import cli


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
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available(
        ) else 'mps' if torch.backends.mps.is_available() else 'cpu')
        print(f"Using device: {self.device}, cuda: {torch.cuda.is_available()}, mps: {torch.backends.mps.is_available()}")
        self.tokenizer = self._tokenizer
        self.model = self._model.to(self.device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        self.model_name = self._model_name
        self.padding_token = self._padding_token
        self.ui = cli.get()
        self.prefix_mask: Optional[Tensor] = None
        self.suffix_mask: Optional[Tensor] = None

    def tokenize(self, data: str) -> Tensor: #List[int]:
        tokenized = self.tokenizer(data, return_tensors='pt')
        input_ids = tokenized['input_ids']
        #input_list = input_ids.tolist()
        #first_elem = input_list[0]
        #return first_elem
        first_elem = input_ids[0]
        return first_elem

    def tokenize_batch(self, data: List[str]) -> Tensor:
        if len(data) == 0:
            raise ValueError("No data to tokenize.")
        with self.ui.display():
            tokenized_list = []
            for i, d in enumerate(data):
                self.ui.update(f"Tokenizing {i+1}/{len(data)}")
                tokenized_list.append(Tensor(self.tokenize(d)))
            #tokenized_list = [Tensor(self.tokenize(d)) for d in data]
        max_length = max([t.shape[0] for t in tokenized_list])
        padded_token_list = [torch.nn.functional.pad(t, (0, max_length - len(t)), value=self.padding_token) for t in tokenized_list]
        tensor = torch.stack(padded_token_list).int().to(self.device)
        return tensor

    def detokenize(self, words: List[int]) -> str:
        return self.tokenizer.decode(words)

    def embed(self, data: str, mask: Optional[Tensor] = None) -> Tensor:
        """
        Embeds a single string into a vector. The string is tokenized and then embedded.
        An optional mask can be used to mask out certain tokens.
        :param data: The string to embed.
        :param mask: A mask to use to mask out certain tokens. Important to note that the mask is applied after tokenization.
            This means that the mask should be of the same length as the tokenized data.
            As an example, if the word "example" is tokenized to "['ex', 'amp', 'le']", the mask should be of length 3.
        :return: The embedding of the string, as a tensor of shape (token_amount, embedding_dim).
        """
        tokenized = self.tokenize(data).to(self.device)
        return self.fully_embed_tokenized(tokenized, mask)

    def _convert_word_to_token_mask(self, tokenized: List[Tensor], mask: Tensor) -> Tensor:
        token_word_lengths = Tensor([tokens.shape[0] for tokens in tokenized]).int().to(self.device)
        assert token_word_lengths.shape[0] == mask.shape[0], f"The mask length and word lengths do not match. Mask: {mask.shape[0]}, Tokens: {token_word_lengths.shape[0]} "
        try:
            mask = mask.repeat_interleave(token_word_lengths, dim=0)
        except RuntimeError as e:
            raise e
        return mask

    def _aggregate_token_to_word_embedding(self, embedded: Tensor, tokenized: List[Tensor]) -> Tensor:
        """
        Aggregates the token embeddings to word embeddings. The aggregation is done by averaging the token embeddings.
        :param embedded: The embeddings of the tokens.
        :param tokenized: The tokenized data.
        :return: The embeddings of the words.
            As an example, if the word "example" is tokenized to "['ex', 'amp', 'le']" and the embeddings of the tokens are
            {'ex': [1, 2], 'amp': [3, 4], 'le': [5, 6]}, the word embedding would be [3, 4].
        """
        word_embeddings = Tensor().to(self.device)
        start_index = 0
        for i, tokens in enumerate(tokenized):
            n_embeddings = tokens.shape[0]
            end_index = start_index + n_embeddings
            assert end_index <= embedded.shape[0]
            if n_embeddings == 1:
                word_embedding = embedded[start_index].unsqueeze(0)
            else:
                word_embedding = embedded[start_index:end_index].mean(dim=0).unsqueeze(0)
                #word_embedding = embedded[end_index -1].unsqueeze(0)
            start_index = end_index
            word_embeddings = torch.concat((word_embeddings, word_embedding), dim=0)
        return word_embeddings

    def _embed_words_full(self, words: List[str], mask: Optional[Tensor] = None) -> Tensor:
        #tokenized = [self.tokenize(word) for word in words]
        tokenized = []
        previous_words = []
        previous_length = 0
        for word in words:
            # Only the first word keeps the begin of input token.
            previous_words.append(word)
            current_sequence = " ".join(previous_words)
            tokens = self.tokenize(current_sequence)
            if len(tokenized) == 0:
                tokenized.append(tokens)
            else:
                current = tokens[previous_length:]
                tokenized.append(current)
            previous_length = tokens.shape[0]
        #if mask is None:
            #mask = torch.ones(len(words)).to(self.device)
        #else:
            #pass
        expanded_mask = self._convert_word_to_token_mask(tokenized, mask) if mask is not None else None
        #embeddings = self.embed(sentence, expanded_mask)
        tokenized_tensor = torch.concat(tokenized, dim=0).to(self.device).float()
        embeddings = self.fully_embed_tokenized(tokenized_tensor, expanded_mask)
        aggregated = self._aggregate_token_to_word_embedding(embeddings, tokenized)

        # Mean the embeddings. Otherwise, setting an embedding to zero might change the meaning.
        normed = torch.nn.functional.normalize(aggregated, dim=1)
        meaned = normed.mean(dim=0)
        aggregated = normed - meaned

        # Apply the mask to the embeddings, so that the masked tokens are zeroed out
        masked = aggregated * mask.unsqueeze(1).expand_as(aggregated) if mask is not None else aggregated
        return masked

    def get_prefix_mask(self) -> Tensor:
        if self.prefix_mask is None:
            self.prefix_mask = torch.ones(len(self.prefix)).to(self.device)
        return self.prefix_mask

    def get_suffix_mask(self) -> Tensor:
        if self.suffix_mask is None:
            self.suffix_mask = torch.ones(len(self.suffix)).to(self.device)
        return self.suffix_mask

    def _embed_words_last(self, words: List[str], mask: Optional[Tensor] = None) -> Tensor:

        classification_added_words = self.prefix + words + self.suffix
        #added_mask = Tensor([1, 1, 1]).to(self.device) if mask is not None else None
        prefix_mask = self.get_prefix_mask()
        suffix_mask = self.get_suffix_mask()
        classification_added_mask = torch.concat((prefix_mask, mask, suffix_mask)) if mask is not None else None
        masked = self._embed_words_full(classification_added_words, classification_added_mask)
        last_entry = masked[-1]
        expanded = last_entry.unsqueeze(0)
        return expanded

    def embed_words(self, words: List[str], mask: Optional[Tensor] = None, strategy: StrategyInstance = UnificationStrategy.TRANSFORMER.create()) -> Tensor:
        if strategy.equals(UnificationStrategy.TRANSFORMER):
            return self._embed_words_last(words, mask)
        if strategy.equals(UnificationStrategy.MEAN):
            return self._embed_words_full(words, mask).mean(dim=0).unsqueeze(0)
        if strategy.equals(UnificationStrategy.PADDING):
            return self._embed_words_full(words, mask)
        raise NotImplementedError(f"The strategy {strategy} is not implemented for huggingmodels.")

    def embed_tokenized(self, tokenized: Tensor) -> Tensor:
        """
        This method takes a list of token indices and returns the
         corresponding embeddings for the first layer of the transformer.
         That is without any context.
        :param tokenized: A tensor of token indices of shape (num_tokens).
        :return: A two-dimensional Tensor where each token index is an embedding. (num_tokens, embedding_size)
        """
        max_length = self.tokenizer.model_max_length
        token_vec = tokenized[:max_length]
        input_embeds_mat = self.model.get_input_embeddings().weight.data
        one_hot = (F.one_hot(token_vec.long(), input_embeds_mat.shape[0]).float() + (
                    token_vec - token_vec.detach()).unsqueeze(1)).to(input_embeds_mat.dtype)
        inputs_embeds = one_hot @ input_embeds_mat
        return inputs_embeds

    @abstractmethod
    def fully_embed_tokenized(self, tokenized: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """
        This method expects a one-dimensional tensor of token indices and returns the corresponding embeddings.
        :param tokenized: 1D Tensor of token indices.
        :param mask: A mask to use to mask out certain tokens. Important to note that the mask is applied after tokenization.
            This means that the mask should be of the same length as the tokenized data.
            As an example, if the word "example" is tokenized to "['ex', 'amp', 'le']", the mask should be of length 3.
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
            ui = cli.get()
            #with torch.no_grad():#, ui.display():
            with ui.display():

                for (i, partial_review) in enumerate(data):
                    #ui.update(f"Embedding {i+1}/{len(data)}")
                    partial_review: Tensor
                    embedded: Tensor = self.fully_embed_tokenized(partial_review).T #returns a (embedding_size, num_tokens) tensor
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
                transposed = embeddings.T
                normed = torch.nn.functional.normalize(transposed, dim=1)
                meaned = normed - normed.mean(dim=0)
                return meaned
            return embeddings
        return embedding

    def max_token_length(self) -> int:
        return self.tokenizer.model_max_length

    def _get_4d_causal_mask(self, attention_mask: Tensor) -> Tensor:
        """
            Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
            `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

            Args:
                attention_mask (`torch.Tensor`):
                    A 2D attention mask of shape `(batch_size, key_value_length)`
            [Taken from the model's source code. and adjusted to be able to pass the gradient]
        """
        sequence_length = target_length = attention_mask.shape[-1]
        batch_size = attention_mask.shape[0]
        dtype = self.model.get_input_embeddings().weight.data.dtype
        device = self.device
        cache_position = torch.arange(sequence_length, device=device)
        causal_mask = torch.zeros((sequence_length, target_length), dtype=dtype, device=device)
        if sequence_length != 1:
            # Create an upper triangular mask (1 for positions to mask, 0 for unmasked)
            causal_mask = torch.triu(torch.ones_like(causal_mask), diagonal=1)
        # Apply the cache condition (1 for masked positions, 0 for allowed positions)
        cache_cond = (torch.arange(target_length, device=device, dtype=dtype) > cache_position.reshape(-1, 1))
        causal_mask = causal_mask * cache_cond.to(dtype)
        # Convert binary mask to additive mask: 1 becomes min_dtype and 0 stays 0
        min_dtype = torch.finfo(dtype).min
        causal_mask = causal_mask * (min_dtype - 0)
        # Expand to 4D
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1, -1)
        if attention_mask is not None:
            mask_length = attention_mask.shape[-1]
            causal_slice = causal_mask[:, :, :, :mask_length]
            # Use a differentiable combination:
            # When attention_mask is 1, use the causal value; when 0, use min_dtype.
            combined = causal_slice * attention_mask[:, None, None, :] + min_dtype * (
                        1 - attention_mask[:, None, None, :])
            # Replace the slice with the combined result
            causal_mask = torch.cat([combined, causal_mask[:, :, :, mask_length:]], dim=-1)

        return causal_mask.to(dtype)
