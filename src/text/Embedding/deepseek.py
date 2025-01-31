from typing import List, Dict, Callable

import numpy as np
import torch
from torch import Tensor, LongTensor
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2ForCausalLM

from text.UI.cli import ConsoleUserInterface
from .huggingmodel import HuggingModel


class DeepSeek1B(HuggingModel):

    def __init__(self):
        super().__init__()
        self.embedded_cache: Dict[int, Tensor] = {}
        self.ui = ConsoleUserInterface()

    @property
    def tokenizer(self) -> AutoTokenizer:
        return AutoTokenizer.from_pretrained(f"deepseek-ai/{self.model_name}", trust_remote_code=True)

    @property
    def model_name(self) -> str:
        return "DeepSeek-R1-Distill-Qwen-1.5B"

    @property
    def model(self) -> Qwen2ForCausalLM:
        return AutoModelForCausalLM.from_pretrained(f"deepseek-ai/{self.model_name}", trust_remote_code=True).to(self.device)

    def embed_tokenized(self, tokenized: List[int]) -> List[np.ndarray]:
        pass

    def fully_embed_tokenized(self, tokenized: Tensor) -> Tensor:
        key = hash(tokenized)
        cached = self.embedded_cache.get(key)
        if cached is not None:
            return cached

        token_vec = torch.tensor(tokenized).to(self.device)

        with torch.no_grad():
            self.ui.update("embedding...")
            outputs = self.model.model(token_vec)
            self.ui.update("done | ")
            embeddings = outputs.last_hidden_state.T

        self.embedded_cache[key] = embeddings
        return embeddings

    def aggregateEmbeddings(self, embeddings: Tensor):
        return torch.mean(embeddings, dim=1)

    def get_embedding_fun(self) -> Callable[[Tensor], Tensor]:
        def embedding(tensor: Tensor) -> Tensor:
            embedded = self.fully_embed_tokenized(tensor)
            aggregated = self.aggregateEmbeddings(embedded)
            return aggregated
        return embedding


    def decode2tokenized(self, embedding: List[np.ndarray]) -> List[int]:
        raise NotImplemented


if __name__ == '__main__':
    model = DeepSeek1B()
    tokenized = model.tokenize("Hello World!")
    print("tokenized:", tokenized)
    tokenized_tensor = torch.tensor(tokenized).unsqueeze(0).to(model.device)
    print("tokenized_tensor",tokenized_tensor)
    fully_embedded = model.fully_embed_tokenized(tokenized_tensor)
    print("fully embedded:", fully_embedded)