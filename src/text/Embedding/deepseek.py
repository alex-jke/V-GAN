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
        if torch.backends.mps.is_available():
            _model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=f"deepseek-ai/{self._model_name}", trust_remote_code=True,
                torch_dtype=torch.float16 # MPS currently does not support bfloat16
            ).to(self.device)

        else:
            _model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=f"deepseek-ai/{self._model_name}", trust_remote_code=True,
                #torch_dtype=torch.bfloat16
            ).to(self.device)

        return _model

    def __init__(self):
        super().__init__()
        self.embedded_cache: Dict[int, Tensor] = {}
        self.ui = cli.get()

    def embed_tokenized(self, tokenized: List[int]) -> List[np.ndarray]:
        raise NotImplementedError

    def fully_embed_tokenized(self, tokenized: Tensor) -> List[Tensor]:
        """
        Embeds the tokenized input and returns the embeddings of the individual tokens.
        :param tokenized: The tokenized input. A tensor of shape (batch_size, sequence_length).
        :return: A list of tensors, each containing the embeddings of the individual tokens.
            The reason for returning a list of tensors is that the number of tokens in each sample might differ, as padding tokens are removed after embedding.
        """
        token_vec = tokenized#tokenized.clone().detach().int().to(self.device)

        # Remove all the padding tokens, that are shared by all samples, as they do not carry any information.
        # Also, for the remaining tokens, create an attention mask.
        raise RuntimeError("Adapt the embedding logic as in GPT2")
        init_attention_mask = torch.not_equal(token_vec, self.padding_token)
        attention_mask_all = init_attention_mask.any(dim=0)
        trimmed_token_vec = token_vec[:, attention_mask_all]
        attention_mask = torch.not_equal(trimmed_token_vec, self.padding_token)
        if not attention_mask.any():
            return [Tensor([0] * 1536).to(self.device)] * len(tokenized)

        with torch.no_grad():
            outputs = self.model.model(trimmed_token_vec, attention_mask) # not surprisingly, this takes the majority of the time.
            embeddings: Tensor = outputs.last_hidden_state#.to(dtype=torch.float32)
        embeddings_list = [embeddings[i] for i in range(embeddings.shape[0])]
        attention_mask_list = [attention_mask[i] for i in range(attention_mask.shape[0])]

        # Cut out the embeddings of the padding tokens, as they might interfere with the aggregation.
        attention_mask_list = [attention_mask.unsqueeze(1).expand(embeddings.shape[1], embeddings.shape[2]) for attention_mask in attention_mask_list]
        embeddings_list = [embeddings[attention_mask] for embeddings, attention_mask in zip(embeddings_list, attention_mask_list)]
        embeddings_list = [embedding.view(-1, embeddings.shape[2])  for embedding in embeddings_list]
        return embeddings_list

    def aggregateEmbeddings(self, embeddings: List[Tensor]):
        #print(f"Aggregated {embeddings[0].shape[0]} embeddings")
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
                    self.ui.update(f"Embedded {aggregated.shape[0]}/{tensor.shape[0]}")
            if batch_first:
                return aggregated
            return aggregated.T
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
        return #"DeepSeek-R1-Distill-Qwen-14B"

class DeepSeek7B(DeepSeek1B):
    @property
    def _model_name(self) -> str:
        return "DeepSeek-R1-Distill-Qwen-7B"

    def get_embedding_fun(self, chunk_size = 2, batch_first=False) -> Callable[[Tensor], Tensor]:
        return super().get_embedding_fun(chunk_size, batch_first)

if __name__ == '__main__':
    model = DeepSeek1B()
    tokenized = model.tokenize("Hello World!")
    print("tokenized:", tokenized)
    tokenized_tensor = torch.tensor(tokenized).unsqueeze(0).to(model.device)
    print("tokenized_tensor",tokenized_tensor)
    fully_embedded = model.fully_embed_tokenized(tokenized_tensor)
    print("fully embedded:", fully_embedded)