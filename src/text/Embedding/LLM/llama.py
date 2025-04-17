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
    def _model_prefix(self):
        return "meta-llama/"

    """@property
    def _model(self):
        print("Model loaded")
        # TODO: switching the model to run on bfloat16 causes infinite gradient norms.
        model = AutoModel.from_pretrained(self._model_prefix + self.get_model_name(), trust_remote_code=True, torch_dtype=torch.float16,
                                          attn_implementation="eager", device_map = 'auto')
        return model"""


class LLama1B(LLama):

    @property
    def _model_name(self)->str:
        return "Llama-3.2-1B"

class LLama3B(LLama):
    def _model_name(self)->str:
        return "Llama-3.2-3B"

class LLama3BInstruct(LLama):
    def _model_name(self) ->str:
        return "Llama-3.2-3B-Instruct"

if __name__ == "__main__":
    # Test the LLama class
    llama = LLama1B()
    sentence = np.array(["Hello, world! This is a test."])
    embedded = llama.embed_sentences(sentence, strategy=UnificationStrategy.MEAN.create())
    print(embedded)