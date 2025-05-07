import logging
import warnings
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

    def __init__(self, dtype = None, **params):
        self._dtype = dtype
        super().__init__(**params)


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
        return self._dtype

    @property
    def _model(self) -> AutoModelForCausalLM:
        dtype = self.get_dtype()
        if dtype is None:
            model = AutoModelForCausalLM.from_pretrained(
            self._model_prefix + self._model_name,
            trust_remote_code=True,
            #device_map='cuda',
            attn_implementation="eager",
        ).to(self.device)
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

    def output_to_debatched(self, outputs):
        embeddings = outputs.hidden_states
        if embeddings is None:
            raise RuntimeError(f"Model returned NoneType. This should not happen and means, that something is not"
                               f"behaving as expected.")
        last_layer = embeddings[-1]  # + (input_embeds - input_embeds.detach())# TODO: check if this is correct.
        de_batched = last_layer[0]
        return de_batched

    def fully_embed_tokenized(self, tokenized: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        mask = mask.unsqueeze(0) if mask is not None else None
        if mask is not None:
            if mask.shape[0] > self.max_token_length():
                raise ValueError(f"Mask of length {mask.shape[0]} is larger than max token input length of {self.max_token_length()}")

            input_embeds = self.embed_tokenized(tokenized)
            expanded_mask = mask.T.expand_as(input_embeds)
            input_embeds = input_embeds * expanded_mask
            input_embeds = input_embeds.unsqueeze(0)
            #with torch.no_grad():
            if True:
                mask_passed = mask.int().float() + (mask - mask.detach())
                causal_mask = self._get_4d_causal_mask(mask_passed)
                input_embeds = input_embeds.to(self.model.get_input_embeddings().weight.dtype)
                torch.backends.cuda.enable_mem_efficient_sdp(False)
                torch.backends.cuda.enable_flash_sdp(False)
                #self.model: AutoModelForCausalLM = self.model
                outputs = self.model(inputs_embeds=input_embeds,
                                     attention_mask=causal_mask,
                                     #attention_mask=mask.int().float(),
                                     use_cache=False,
                                     output_hidden_states=True
                                     )
            de_batched = self.output_to_debatched(outputs)
            assert de_batched.shape[0] == mask.shape[1], "Debatch was not successful."
            normalized = torch.nn.functional.normalize(de_batched)
            return normalized

            #with torch.no_grad():
                #outputs = self.model(input_ids=tokenized.int().unsqueeze(0), attention_mask=mask, output_hidden_states=True, use_cache=False)
        else:
            with torch.no_grad():
                if tokenized.shape[0] > self.max_token_length():
                    if not self._token_length_warning_given:
                        logging.warning(f"Input contains token sequences of length larger than the maximum length {self.max_token_length()}. "
                                      f"Found: {tokenized.shape[0]}. This token tensor and further token tensors of length"
                                      f"larger than {self.max_token_length()} will be trimmed. No further warnings will be given.")
                        self._token_length_warning_given = True
                    tokenized = tokenized[:self.max_token_length()]
                unsqueezed = tokenized.int().unsqueeze(0)
                outputs = self.model(input_ids=unsqueezed, use_cache=False, output_hidden_states=True)
            outputs = outputs # Otherwise the line below complains about usage before assignment.
        de_batched = self.output_to_debatched(outputs)
        #return de_batched
        normalized = torch.nn.functional.normalize(de_batched)
        return normalized

