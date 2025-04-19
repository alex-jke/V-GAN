from abc import ABC, abstractproperty, abstractmethod
from typing import Optional

import torch
from torch import Tensor
from transformers import AutoTokenizer, AutoModelForCausalLM

from text.Embedding.LLM.huggingmodel import HuggingModel


class CausalLLM(HuggingModel, ABC):
    """
    This class is a wrapper for causal LLM models.
    It uses the Hugging Face Transformers library to load and use the model.
    It also provides methods for embedding sentences and tokenized inputs.
    """


    @property
    @abstractmethod
    def _model_prefix(self):
        """
        Returns the prefix for the model name. For example, "meta-llama/" for Llama models.
        This should be overridden in subclasses to provide the correct prefix.
        """
        pass

    @property
    def _tokenizer(self):
        print("Tokenizer loaded")
        try:
            tokenizer = AutoTokenizer.from_pretrained(self._model_prefix + self._model_name,
                                                      trust_remote_code=True
                                                      )
        except OSError as ose:
            print(
                "An OSError was raised. This likely happened due to this LLM being a restricted model. Please verify, that"
                "you are logged into huggingface and have access to the LLM models used here. Check readme to see how to log in. The exception:")
            raise ose
        return tokenizer

    def get_dtype(self) -> Optional:
        return None

    @property
    def _model(self):
        dtype = self.get_dtype()
        if dtype is None:
            model = AutoModelForCausalLM.from_pretrained(
            self._model_prefix + self._model_name,
            trust_remote_code=True,
            device_map='cuda',
            attn_implementation="eager",
        )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                self._model_prefix + self._model_name,
                trust_remote_code=True,
                torch_dtype=self.get_dtype(),
                device_map='cuda',
                attn_implementation="eager",
            )
        print(f"Loaded {self._model_name} model.")
        return model

    def fully_embed_tokenized(self, tokenized: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        mask = mask.unsqueeze(0) if mask is not None else None
        if mask is not None:
            input_embeds = self.embed_tokenized(tokenized).unsqueeze(0)
            causal_mask = self._get_4d_causal_mask(mask)
            input_embeds = input_embeds.to(self.model.get_input_embeddings().weight.data.dtype)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_flash_sdp(False)
            outputs = self.model(inputs_embeds=input_embeds, attention_mask=causal_mask, use_cache=False)
        else:
            with torch.no_grad():
                outputs = self.model(input_ids=tokenized.int().unsqueeze(0), use_cache=False)
            outputs = outputs # Otherwise the line below complains about usage before assignment.
        embeddings = outputs[0]
        de_batched = embeddings[0]
        return de_batched

