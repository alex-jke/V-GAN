from abc import ABC
from typing import Optional

import numpy as np
import torch
from torch import Tensor
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

from text.Embedding.LLM.causal_llm import CausalLLM
from text.Embedding.LLM.huggingmodel import HuggingModel
from text.Embedding.unification_strategy import UnificationStrategy


class OpenELM(CausalLLM, ABC):
    """
    This class is a wrapper for apples OpenELM models.
    It uses the Hugging Face Transformers library to load and use the model.
    """

    @property
    def _model_prefix(self):
        return "apple/"


class OpenELM1B(OpenELM):
    """
    This class is a wrapper for the OpenELM 1B model.
    It uses the Hugging Face Transformers library to load and use the model.
    """

    @property
    def _model_name(self):
        return "OpenELM-1_1B"

class OpenELM1BInstruct(OpenELM):
    """
    This class is a wrapper for the OpenELM 1B Instruct model.
    It uses the Hugging Face Transformers library to load and use the model.
    """

    @property
    def _model_name(self):
        return "OpenELM-1_1B-Instruct"


class OpenELM3B(OpenELM):
    """
    This class is a wrapper for the OpenELM 1B model.
    It uses the Hugging Face Transformers library to load and use the model.
    """

    @property
    def _model_name(self):
        return "OpenELM-3B"


class OpenELM3BInstruct(OpenELM):
    """
    This class is a wrapper for the OpenELM 1B Instruct model.
    It uses the Hugging Face Transformers library to load and use the model.
    """

    @property
    def _model_name(self):
        return "OpenELM-3B-Instruct"