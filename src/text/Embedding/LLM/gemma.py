from abc import ABC
from typing import Optional

import numpy as np
import torch
from torch import Tensor
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

from text.Embedding.LLM.causal_llm import CausalLLM
from text.Embedding.LLM.huggingmodel import HuggingModel
from text.Embedding.unification_strategy import UnificationStrategy


class Gemma(CausalLLM, ABC):
    """
    This class is a wrapper for Gemma 3 models.
    It uses the Hugging Face Transformers library to load and use the model.
    """

    @property
    def _model_prefix(self):
        return "google/"

class Gemma4BIt(Gemma):
    """
    This class is a wrapper for the Gemma 3 4B Instruction tuned model.
    It uses the Hugging Face Transformers library to load and use the model.
    """
    _model_name = "gemma-3-4b-it"

class Gemma4BPt(Gemma):
    """
    This class is a wrapper for the Gemma 3 4B Pretrained model.
    It uses the Hugging Face Transformers library to load and use the model.
    """
    _model_name = "gemma-3-4b-pt"

class Gemma1BIt(Gemma):
    """
    This class is a wrapper for the Gemma 3 1B Instruction tuned model.
    It uses the Hugging Face Transformers library to load and use the model.
    """
    _model_name = "gemma-3-1b-it"

class Gemma1BPt(Gemma):
    """
    This class is a wrapper for the Gemma 3 1B Pretrained model.
    It uses the Hugging Face Transformers library to load and use the model.
    """
    _model_name = "gemma-3-1b-pt"

class Gemma12BIt(Gemma):
    """
    This class is a wrapper for the Gemma 3 12B Instruction tuned model.
    It uses the Hugging Face Transformers library to load and use the model.
    """
    _model_name = "gemma-3-12b-it"

class Gemma12BPt(Gemma):
    """
    This class is a wrapper for the Gemma 3 12B Pretrained model.
    It uses the Hugging Face Transformers library to load and use the model.
    """
    _model_name = "gemma-3-12b-pt"

if __name__ == "__main__":
    # Test the Gemma class
    gemma = Gemma1BIt()
    sentence = np.ndarray(["Hello, world! This is a test."])
    embedded = gemma.embed_sentences(sentence, strategy=UnificationStrategy.MEAN.create())
    print("Embedded:", embedded)