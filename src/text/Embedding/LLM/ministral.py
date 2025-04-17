from typing import Optional

import numpy as np
import torch
from torch import Tensor
from transformers import AutoTokenizer, AutoModel

from text.Embedding.LLM.huggingmodel import HuggingModel
from text.Embedding.unification_strategy import UnificationStrategy


class Ministral8BInstruct(HuggingModel):
    """
    This class is a wrapper for the Ministral 8B Instruct model.
    It uses the Hugging Face Transformers library to load and use the model.
    """
    _mistral_prefix = "mistralai/"

    @property
    def _model_name(self):
        return "Ministral-8B-Instruct-2410"

    @property
    def _tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self._mistral_prefix + self._model_name, trust_remote_code=True)
        print(f"Loaded {self._model_name} tokenizer.")
        return tokenizer


    @property
    def _model(self):

        model = AutoModel.from_pretrained(
            self._mistral_prefix + self._model_name,
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

if __name__ == "__main__":
    # Test the Gemma class
    ministral = Ministral8BInstruct()
    sentence = np.ndarray(["Hello, world! This is a test."])
    embedded = ministral.embed_sentences(sentence, strategy=UnificationStrategy.MEAN.create())
    print("Embedded:", embedded)

