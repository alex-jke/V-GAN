from abc import ABC
from typing import List, Dict, Callable

import numpy as np
import torch
from torch import Tensor, LongTensor
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2ForCausalLM

from text.UI import cli
from text.UI.cli import ConsoleUserInterface
from .huggingmodel import HuggingModel


class DeepSeek(HuggingModel, ABC):

    @property
    def _tokenizer(self) -> AutoTokenizer:
        return AutoTokenizer.from_pretrained(f"deepseek-ai/{self._model_name}", trust_remote_code=True)

    @property
    def _model(self) -> Qwen2ForCausalLM:
        _model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=f"deepseek-ai/{self._model_name}", trust_remote_code=True).to(self.device)
        return _model.half().to(self.device)

    def __init__(self):
        super().__init__()
        self.embedded_cache: Dict[int, Tensor] = {}
        self.ui = cli.get()

    def embed_tokenized(self, tokenized: List[int]) -> List[np.ndarray]:
        raise NotImplementedError

    def fully_embed_tokenized(self, tokenized: Tensor) -> List[Tensor]:
        token_vec = tokenized.clone().detach().int().to(self.device)

        # Remove final padding tokens, if all tensors are padded longer than the longest tensor, this is not necessary.
        largest_non_token_index = torch.max(torch.nonzero(token_vec)).item()
        trimmed_token_vec = token_vec[:largest_non_token_index + 1]
        #remainder = token_vec[largest_non_token_index:]
        attention_mask = torch.not_equal(trimmed_token_vec, self.padding_token)

        with torch.no_grad():
            outputs = self.model.model(trimmed_token_vec, attention_mask=attention_mask) # not surprisingly, this takes the majority of the time.
            embeddings: Tensor = outputs.last_hidden_state
        embeddings_list = [embeddings[i] for i in range(embeddings.shape[0])]
        attention_mask_list = [attention_mask[i] for i in range(attention_mask.shape[0])]

        # Cut out the embeddings of the padding tokens, as they might interfere with the aggregation.
        attention_mask_list = [attention_mask.unsqueeze(1).expand(embeddings.shape[1], embeddings.shape[2]) for attention_mask in attention_mask_list]
        embeddings_list = [embeddings[attention_mask] for embeddings, attention_mask in zip(embeddings_list, attention_mask_list)]
        embeddings_list = [embedding.view(-1, embeddings.shape[2])  for embedding in embeddings_list]
        return embeddings_list

    def aggregateEmbeddings(self, embeddings: List[Tensor]):
        aggregated = [torch.mean(emb, dim=0) for emb in embeddings]
        stacked = torch.stack(aggregated)
        return stacked

    def get_embedding_fun(self, chunk_size = 10, batch_first=False) -> Callable[[Tensor], Tensor]:

        def embedding(tensor: Tensor) -> Tensor:
            chunks = torch.split(tensor, chunk_size, dim=0)
            aggregated = Tensor().to(self.device)
            with torch.no_grad(), self.ui.display():
                for chunk in chunks:
                    fully_embedded: List[Tensor] = self.fully_embed_tokenized(chunk)
                    aggregated_chunk = self.aggregateEmbeddings(fully_embedded)
                    aggregated = torch.cat((aggregated, aggregated_chunk), dim=0)
                    self.ui.update(f"Embedded {aggregated.shape[1]}/{tensor.shape[0]}")
            if batch_first:
                return aggregated.T
            return aggregated
        return embedding


    def decode2tokenized(self, embedding: List[np.ndarray]) -> List[int]:
        raise NotImplemented

class DeepSeek1B(DeepSeek):

    @property
    def _model_name(self) -> str:
        return "DeepSeek-R1-Distill-Qwen-1.5B"

class DeepSeek14B(DeepSeek):

    @property
    def _model_name(self) -> str:
        return "DeepSeek-R1-Distill-Qwen-14B"


if __name__ == '__main__':
    model = DeepSeek1B()
    tokenized = model.tokenize("Hello World!")
    print("tokenized:", tokenized)
    tokenized_tensor = torch.tensor(tokenized).unsqueeze(0).to(model.device)
    print("tokenized_tensor",tokenized_tensor)
    fully_embedded = model.fully_embed_tokenized(tokenized_tensor)
    print("fully embedded:", fully_embedded)