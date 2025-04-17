from typing import Optional

import numpy as np
import torch
from torch import Tensor
from transformers import AutoTokenizer, AutoModel

from text.Embedding.LLM.causal_llm import CausalLLM
from text.Embedding.LLM.huggingmodel import HuggingModel
from text.Embedding.unification_strategy import UnificationStrategy


class Ministral8BInstruct(CausalLLM):
    """
    This class is a wrapper for the Ministral 8B Instruct model.
    It uses the Hugging Face Transformers library to load and use the model.
    """

    @property
    def _model_prefix(self):
        return "mistralai/"

    @property
    def _model_name(self):
        return "Ministral-8B-Instruct-2410"


if __name__ == "__main__":
    # Test the Gemma class
    ministral = Ministral8BInstruct()
    sentence = np.array(["Hello, world! This is a test."])
    embedded = ministral.embed_sentences(sentence, strategy=UnificationStrategy.MEAN.create())
    print("Embedded:", embedded)

