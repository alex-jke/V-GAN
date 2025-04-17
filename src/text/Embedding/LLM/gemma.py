from abc import ABC
from typing import Optional

import numpy as np
import torch
from torch import Tensor
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

from text.Embedding.LLM.huggingmodel import HuggingModel
from text.Embedding.unification_strategy import UnificationStrategy


class Gemma(HuggingModel, ABC):
    """
    This class is a wrapper for Gemma 3 models.
    It uses the Hugging Face Transformers library to load and use the model.
    """
    _google_prefix = "google/"

    @property
    def _tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self._google_prefix + self._model_name, trust_remote_code=True)
        print(f"Loaded {self._model_name} tokenizer.")
        return tokenizer


    @property
    def _model(self):

        model = AutoModelForCausalLM.from_pretrained(
            self._google_prefix + self._model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map='auto'
        )
        print(f"Loaded {self._model_name} model.")
        return model

    def fully_embed_tokenized(self, tokenized: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        mask = mask.unsqueeze(0) if mask is not None else None

        if mask is not None:
            raise NotImplementedError("Masking is not implemented for this model.")
        else:
            outputs = self.model(input_ids=tokenized.int().unsqueeze(0), use_cache=False)

        # Extract the last hidden state
        last_hidden_state = outputs.last_hidden_state
        # Remove the batch dimension
        de_batched = last_hidden_state[0]
        return de_batched

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