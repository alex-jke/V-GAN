import logging
from abc import abstractmethod, ABC
from typing import List, Optional

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from transformers import LlamaModel

from text.Embedding.LLM.causal_llm import CausalLLM
from text.Embedding.LLM.huggingmodel import HuggingModel
from text.Embedding.unification_strategy import UnificationStrategy


class LLama(CausalLLM, ABC):
    """
    This class is a wrapper around the Llama model from Hugging Face's transformers library.
    """
    @property
    def _model_name(self):
        return self.get_model_name()

    @property
    def _tokenizer(self):
        print("Tokenizer loaded")
        try:
            tokenizer = AutoTokenizer.from_pretrained(self._model_prefix + self.get_model_name(), trust_remote_code=True)
        except OSError as ose:
            print("An OSError was raised. This likely happened due to LLama being a restricted model. Please verify, that"
                  "you are logged into huggingface and have access to the LLama models used here. Check readme to see how to log in. The exception:")
            raise ose
        return tokenizer

    @property
    def _model_prefix(self):
        return "meta-llama/"

    @property
    def _model(self):
        print("Model loaded")
        # TODO: switching the model to run on bfloat16 causes infinite gradient norms.
        model = AutoModel.from_pretrained(self._model_prefix + self.get_model_name(), trust_remote_code=True, torch_dtype=torch.float16,
                                          attn_implementation="eager",
                                          device_map="auto")
                                          #).to(self.device)
        return model

    def output_to_debatched(self, outputs):
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

class LLama3BInstruct(LLama):
    def get_model_name(self) ->str:
        return "Llama-3.2-3B-Instruct"

if __name__ == "__main__":
    # Test the LLama class
    llama = LLama1B()
    sentence = np.array(["Hello, world! This is a test."])
    embedded = llama.embed_sentences(sentence, strategy=UnificationStrategy.MEAN.create())
    print(embedded)