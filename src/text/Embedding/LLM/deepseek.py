from abc import ABC
from typing import List, Dict, Callable, Optional

import numpy as np
import torch
from statsmodels.tools.tools import unsqueeze
from torch import Tensor, LongTensor
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2ForCausalLM

from text.Embedding.LLM.causal_llm import CausalLLM
from text.UI import cli
from text.UI.cli import ConsoleUserInterface
from .huggingmodel import HuggingModel


class DeepSeek(CausalLLM, ABC):

    @property
    def _model_prefix(self):
        return "deepseek-ai/"

class DeepSeek1B(DeepSeek):

    @property
    def _model_name(self) -> str:
        return "DeepSeek-R1-Distill-Qwen-1.5B"

class DeepSeek14B(DeepSeek):

    @property
    def _model_name(self) -> str:
        raise NotImplementedError("The model has not been implemented yet. It takes up a considerable amount of storage space.")
        return #"DeepSeek-R1-Distill-Qwen-14B"

class DeepSeek7B(DeepSeek1B):
    @property
    def _model_name(self) -> str:
        return "DeepSeek-R1-Distill-Qwen-7B"


if __name__ == '__main__':
    model = DeepSeek1B()
    tokenized = model.tokenize("Hello World!")
    print("tokenized:", tokenized)
    tokenized_tensor = torch.tensor(tokenized).unsqueeze(0).to(model.device)
    print("tokenized_tensor",tokenized_tensor)
    fully_embedded = model.fully_embed_tokenized(tokenized_tensor)
    print("fully embedded:", fully_embedded)