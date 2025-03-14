from typing import List, Optional

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from transformers import LlamaModel

from text.Embedding.huggingmodel import HuggingModel


class LLama(HuggingModel):
    @property
    def _model_name(self):
        return self.get_model_name()

    @property
    def _tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self._model_prefix + self.get_model_name(), trust_remote_code=True)
        return tokenizer

    @property
    def _model_prefix(self):
        return "meta-llama/"

    @property
    def _model(self):
        model = AutoModel.from_pretrained(self._model_prefix + self.get_model_name(), trust_remote_code=True, torch_dtype=torch.float16)
        return model

    def embed_tokenized(self, tokenized: Tensor) -> Tensor:
        max_length = self._tokenizer.model_max_length
        token_vec = tokenized[:max_length]
        input_embeds_mat = self.model.get_input_embeddings().weight.data
        one_hot = (F.one_hot(token_vec.long(), input_embeds_mat.shape[0]).float() + (
                    token_vec - token_vec.detach()).unsqueeze(1)).to(input_embeds_mat.dtype)
        inputs_embeds = one_hot @ input_embeds_mat
        return inputs_embeds

    def fully_embed_tokenized(self, tokenized: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        input_embeds = self.embed_tokenized(tokenized).unsqueeze(0)
        mask = mask.unsqueeze(0) if mask is not None else None
        if mask is not None:
            outputs = self.model(inputs_embeds=input_embeds, attention_mask=mask)
        else:
            outputs = self.model(inputs_embeds=input_embeds)
        embeddings = outputs[0]
        de_batched = embeddings[0]
        return de_batched

    def decode2tokenized(self, embedding: List[np.ndarray]) -> List[int]:
        raise NotImplementedError

    def embed_words(self, words: List[str], mask: Optional[Tensor] = None, aggregate: bool = True) -> Tensor:
        if not aggregate:
            return super().embed_words(words, mask)
        classification_added_words = words + ["I", "am", "feeling"]
        added_mask = Tensor([1, 1, 1]).to(self.device) if mask is not None else None
        classification_added_mask = torch.concat((mask, added_mask)) if mask is not None else None
        masked = super().embed_words(classification_added_words, classification_added_mask)
        last_entry = masked[-1]
        expanded = last_entry.unsqueeze(0)
        return expanded

    def get_model_name(self)->str:
        return "Llama-3.2-1B"