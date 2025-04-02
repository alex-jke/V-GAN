from abc import abstractmethod, ABC
from typing import List, Optional

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from transformers import LlamaModel

from text.Embedding.huggingmodel import HuggingModel


class LLama(HuggingModel, ABC):
    @property
    def _model_name(self):
        return self.get_model_name()

    @property
    def _tokenizer(self):
        print("Tokenizer loaded")
        tokenizer = AutoTokenizer.from_pretrained(self._model_prefix + self.get_model_name(), trust_remote_code=True)
        return tokenizer

    @property
    def _model_prefix(self):
        return "meta-llama/"

    @property
    def _model(self):
        print("Model loaded")
        # TODO: switching the model to run on bfloat16 causes infinite gradient norms.
        model = AutoModel.from_pretrained(self._model_prefix + self.get_model_name(), trust_remote_code=True, torch_dtype=torch.float16,
                                          attn_implementation="eager")
        return model

    def fully_embed_tokenized(self, tokenized: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        input_embeds = self.embed_tokenized(tokenized).unsqueeze(0)
        mask = mask.unsqueeze(0) if mask is not None else None
        if mask is not None:
            causal_mask = self._get_4d_causal_mask(mask)
            input_embeds = input_embeds.to(self.model.get_input_embeddings().weight.data.dtype)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_flash_sdp(False)
            #outputs = self.model(inputs_embeds=input_embeds, attention_mask=mask, output_attentions=True)
            outputs = self.model(inputs_embeds=input_embeds, attention_mask=causal_mask, use_cache=False)
            #if not torch.allclose(outputs[0], causal_output[0]):
                #expected = [o for o_list_list in (outputs[0].tolist()) for o_list in o_list_list for o in o_list]
                #actual = [co for co_list_list in causal_output[0].tolist() for co_list in co_list_list for co in co_list]
                #diffs = [(e, a) for e,a in zip(expected, actual) if abs(e - a) < 1e-6]
                #raise ValueError("Causal mask and normal mask do not produce the same output."
                #                 f"{diffs} differences") TODO: look into this appear to be the same:
        else:
            outputs = self.model(inputs_embeds=input_embeds)
        embeddings = outputs[0]
        de_batched = embeddings[0]
        return de_batched

    def decode2tokenized(self, embedding: List[np.ndarray]) -> List[int]:
        raise NotImplementedError

    @abstractmethod
    def get_model_name(self)->str:
        pass

class LLama1B(LLama):
    def get_model_name(self)->str:
        return "Llama-3.2-1B"

class LLama3B(LLama):
    def get_model_name(self)->str:
        return "Llama-3.2-3B"