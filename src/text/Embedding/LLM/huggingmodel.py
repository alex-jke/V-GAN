import logging
from abc import ABC, abstractmethod
from typing import List, Callable, Optional

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F
from transformers import PreTrainedModel

from text.Embedding.embedding import Embedding
from text.Embedding.tokenizer import Tokenizer
from text.Embedding.unification_strategy import UnificationStrategy, StrategyInstance
from text.UI import cli

WORDS_TO_TOKENS_FACTOR = 1.5

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


    def __init__(self, max_token_length: int = 5120, debug: bool = True):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available(
        ) else 'mps' if torch.backends.mps.is_available() else 'cpu')
        #print(f"Using device: {self.device}, cuda: {torch.cuda.is_available()}, mps: {torch.backends.mps.is_available()}")
        self.__tokenizer = None
        self.__model: Optional[HuggingModel] = None
        #self.model = self._model
        #if torch.cuda.device_count() > 1:
            #print(f"Using {torch.cuda.device_count()} GPUs.")
            #model = nn.DataParallel(model)
        #self.model = model.to(self.device)
        self.model_name = self._model_name
        self._max_token_length = max_token_length
        self.__padding_token = None
        #self.padding_token = self._padding_token
        self.ui = cli.get()
        self.prefix_mask: Optional[Tensor] = None
        self.suffix_mask: Optional[Tensor] = None
        self._token_length_warning_given = False
        self.debug = debug

    @property
    def padding_token(self):
        if self.__padding_token is None:
            self.__padding_token = self._padding_token
        return self.__padding_token


    @property
    def model(self):
        if self.__model is None:
            self.__model = self._model
            #self.__model.eval()
            self.__model.train()
            self.model.gradient_checkpointing_enable()
            for param in self.__model.parameters():
                param.requires_grad = False
        return self.__model

    @property
    def tokenizer(self):
        if self.__tokenizer is None:
            self.__tokenizer = self._tokenizer
        return self.__tokenizer

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
        assert token_word_lengths.shape[0] == mask.shape[0], f"The word mask length and word lengths do not match. Mask: {mask.shape[0]}, Tokens: {token_word_lengths.shape[0]} "
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
        word_embeddings = Tensor().to(embedded.device)
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


    def _tokenize_words(self, words: List[str], max_token_length: Optional[int] = None) -> List[Tensor]:
        tokenized = []
        #previous_words = []
        previous_word: Optional[str] = None
        previous_length = 0
        prefix = "I" # Add a prefix to the tokenization of the words, as sometimes the token changes, if at the front of a sequence.
        prefix_tokens = self.tokenize(prefix)
        prefix_length = prefix_tokens.shape[0]
        current_length = 0
        for word in words:
            # Only the first word keeps the begin of input token.

            #previous_words.append(word)
            current_sequence = " ".join([prefix, word]) if previous_word is not None else word
            tokens = self.tokenize(current_sequence)
            if len(tokenized) != 0:
                tokens = tokens[prefix_length:]

            if max_token_length is not None and tokens.shape[0] + current_length > max_token_length:
                logging.warning(f"Tried tokenizing sequence longer than maximum sequence length of "
                                f"{self.max_token_length()}. The input will be trimmed.")
                break
            tokenized.append(tokens)
            previous_word = word
            current_length += tokens.shape[0]
        if self.debug:
            concat = torch.concat(tokenized, dim=0)
            assert concat.equal(self.tokenize(" ".join(words))[:concat.shape[0]])
        return tokenized


    def _embed_words_full(self, words: List[str], mask: Optional[Tensor] = None, suffix: Optional[List[str]] = None) -> Tensor:
        if mask is not None and mask.shape[0] > self.max_word_length():
            raise RuntimeError(f"A mask was passed, thats length is longer that the maximum amount: {mask.shape[0] > self.max_word_length()}")

        if suffix is not None:
            tokenized_suffix = self._tokenize_words(words=suffix)
            tokenized_suffix[0] = tokenized_suffix[0][1:] # Remove the beginning of sequence token.
            suffix_length = sum([tokens.shape[0] for tokens in tokenized_suffix])
            tokenized_sample = self._tokenize_words(words=words, max_token_length= self.max_token_length() - suffix_length)
            tokenized = tokenized_sample + tokenized_suffix
        else:
            tokenized = self._tokenize_words(words=words, max_token_length=self.max_token_length())

        expanded_mask = self._convert_word_to_token_mask(tokenized, mask) if mask is not None else None
        if expanded_mask is not None and expanded_mask.shape[0] > self.max_token_length():
            raise ValueError(f"Passed a mask that is longer, than the maximum token length. {self.max_token_length()}.")
        #embeddings = self.embed(sentence, expanded_mask)
        tokenized_tensor = torch.concat(tokenized, dim=0).to(self.device).float()
        if tokenized_tensor.shape[0] > self.max_token_length():
            raise RuntimeError("Tokenized tensor is longer than max token length. This should not occur and signifies,"
                               "that something is wrong with the truncation process before.")
        embeddings = self.fully_embed_tokenized(tokenized_tensor, expanded_mask)
        aggregated = self._aggregate_token_to_word_embedding(embeddings, tokenized)

        #aggregated = torch.nn.functional.normalize(aggregated, dim=1)

        # Apply the mask to the embeddings, so that the masked tokens are zeroed out
        masked = aggregated * mask.to(aggregated.device).unsqueeze(1).expand_as(aggregated) if mask is not None else aggregated
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
        masked = self._embed_words_full(self.prefix + words, classification_added_mask, suffix=self.suffix)
        last_entry = masked[-1]
        expanded = last_entry.unsqueeze(0)
        return expanded

    def embed_words(self, words: List[str], mask: Optional[Tensor] = None, strategy: StrategyInstance = UnificationStrategy.TRANSFORMER.create(), trim: bool = True) -> Tensor:
        if trim:
            words = words[:self.max_word_length()]
        if strategy.equals(UnificationStrategy.TRANSFORMER):
            return torch.nn.functional.normalize(self._embed_words_last(words, mask), dim=1)
        if strategy.equals(UnificationStrategy.MEAN):
            return torch.nn.functional.normalize(self._embed_words_full(words, mask).mean(dim=0).unsqueeze(0), dim=1)
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
                    token_vec - token_vec.detach()).unsqueeze(1)).to(input_embeds_mat.dtype).to(input_embeds_mat.device)
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

    def get_embedding_fun(self, batch_first = False, remove_padding = False) -> Callable[[Tensor], Tensor]:
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
            longest_sequence = data.shape[1]
            #print(f"Longest sequence is {longest_sequence} tokens.")
            with ui.display():

                for (i, partial_review) in enumerate(data):
                    ui.update(f"Embedding {i+1}/{len(data)}")
                    partial_review: Tensor

                    if remove_padding:
                        # Remove padding tokens
                        partial_review = partial_review[partial_review != self.padding_token]

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
                #meaned = transposed - transposed.mean(dim=0)
                normed = torch.nn.functional.normalize(transposed, dim=1)

                return normed
                #return transposed
            return embeddings
        return embedding

    def max_token_length(self) -> int:
        return min(self.tokenizer.model_max_length, self._max_token_length)

    def max_word_length(self) -> int:
        return int(self.max_token_length() / WORDS_TO_TOKENS_FACTOR)

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
        # Convert binary mask to additive mask: 0 becomes min_dtype and 1 becomes 0
        min_dtype = torch.finfo(dtype).min
        causal_mask = (1 - causal_mask) * min_dtype
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
