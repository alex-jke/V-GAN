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

    def fully_embed_tokenized(self, tokenized: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        input_embeds = self.embed_tokenized(tokenized).unsqueeze(0)
        mask = mask.unsqueeze(0) if mask is not None else None
        if mask is not None:
            causal_mask = self._get_4d_causal_mask(mask)
            input_embeds = input_embeds.to(self.model.get_input_embeddings().weight.data.dtype)
            #normal_outputs = self.model(inputs_embeds=input_embeds, attention_mask=mask, output_attentions=True)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_flash_sdp(False)
            outputs = self.model(inputs_embeds=input_embeds, attention_mask=causal_mask)
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

    def decode2tokenized(self, embedding: List[np.ndarray]) -> List[int]:
        raise NotImplementedError

    def get_model_name(self)->str:
        return "Llama-3.2-1B"