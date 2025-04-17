import os
from typing import List, Dict

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM

from .huggingmodel import HuggingModel


class Bert(HuggingModel):

    def __init__(self):
        super().__init__()
        self.embedded_cache: Dict[int, Tensor] = {}

    @property
    def _tokenizer(self):
        return AutoTokenizer.from_pretrained(self._model_name)

    @property
    def _model_name(self):
        return 'bert-base-cased'

    @property
    def _model(self):
        return AutoModel.from_pretrained(self._model_name,
                                         use_cache=False).to(self.device)

    def decode2tokenized(self, embeddings: List[np.ndarray]) -> List[int]:
        """
        This method takes a list of token embeddings and returns the closest token index for each.
        :param embeddings: List of individual token embeddings as numpy arrays.
        :return: List of token indices closest to each embedding.
        """
        # Convert list of numpy embeddings to a single tensor and ensure dtype is float32
        embeddings = torch.tensor(np.stack(embeddings), dtype=torch.float32).to(self.device)  # Shape: (num_tokens, embedding_dim)

        # Retrieve token embeddings for the vocabulary and ensure they are also float32
        token_embeddings: Tensor = self.get_token_embeddings().to(
            dtype=torch.float32)  # Shape: (vocab_size, embedding_dim)

        # Removes the CLS, SEP, and MASK tokens, as those were always chosen as the closest tokens
        tokens_to_remove = [103, 102, 101]
        for token in tokens_to_remove:
            token_embeddings = torch.cat((token_embeddings[:token], token_embeddings[token + 1:]), 0)

        # Normalize embeddings for cosine similarity
        embeddings = torch.nn.functional.normalize(embeddings, dim=1)
        token_embeddings = torch.nn.functional.normalize(token_embeddings, dim=1)

        # Compute cosine similarities for each token embedding against the vocabulary
        similarities = torch.matmul(embeddings, token_embeddings.T)  # Shape (num_tokens x vocab_size)

        # Find the index with the highest similarity for each token embedding
        closest_tokens = torch.argmax(similarities, dim=1).tolist()  # Convert result to a list of token IDs

        return closest_tokens


    def embed_tokenized(self, tokenized: List[int]) -> List[np.ndarray]:
        """
        This method takes a tokenized list of integers and returns the embeddings.
        :param tokenized: A list of indexes of the tokens.
        :return: A list of embeddings.
        """
        tokenized = torch.tensor(tokenized).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(tokenized)
            embeddings: Tensor = outputs.last_hidden_state

        list_embeddings = embeddings.tolist()[0]
        # Convert the embeddings to a list of numpy arrays
        return [np.array(embedding) for embedding in list_embeddings]

    def get_token_embeddings(self) -> Tensor:
        """
        This method returns the token embeddings.
        :return: A 28_996 x 768 tensor.
        """
        embeddings = self.model.embeddings.word_embeddings.weight.detach()
        return embeddings

    def get_token_embeddings_csv(self) -> pd.DataFrame:
        """
        This method returns the token embeddings. If the csv file with the embeddings is not found, it will
        create it.
        :return:
        """
        embeddings = self.model.embeddings.word_embeddings.weight.detach().numpy()
        csv_file = f"../resources/{self.model_name}_embeddings.csv"

        if os.path.exists(csv_file):
            return pd.read_csv(csv_file)

        file = open(csv_file, 'x')
        token_size = len(self.tokenizer)

    def aggregateEmbeddings(self, embeddings: Tensor):
        cls_token = embeddings[:,:,0]
        return cls_token


    def fully_embed_tokenized(self, tokenized: Tensor) -> Tensor:
        """
        This method takes a list of token indices and returns the corresponding embeddings.
        The embeddings are taken from the last layer of the model.
        :param tokenized: A list of token indices.
        :return: A two-dimensional Tensor where each token index is an embedding. (num_tokens, embedding_size)
        """
        raise RuntimeError("Adapt the embedding logic as in GPT2")
        maximum_length = 512
        #with torch.no_grad():
        #with self.ui.display():
        #tokenized = tokenized.clone().detach().to(self.device)
        attention_mask = torch.not_equal(tokenized, self.padding_token)
        token_vec = tokenized[attention_mask].unsqueeze(0).to(self.device) # todo the unsqueeze causes mps out of memory
        if len(token_vec.shape) > 1:
            token_vec = token_vec[:, :maximum_length] #todo: cutting off tokens for bert.
        else:
            token_vec = token_vec[:maximum_length]
        if (token_vec.shape[-1] == 0):
            return torch.zeros_like(Tensor([0] * 768)).to(self.device).unsqueeze(1)
        #with torch.no_grad():
        outputs = self.model(token_vec)
            # BERT returns a 768 x num_tokens x 1 tensor, so we need to remove the last dimension
        embeddings = outputs.last_hidden_state.T[:, :, 0].T

        #self.embedded_cache[key] = embeddings
        return embeddings


    def bert_encode(self, texts, tokenizer, model, max_length):
        input_ids = []
        attention_masks = []
        model.to(self.device)

        for text in texts:
            # encoded_dict = tokenizer.encode_plus(
            #     text,
            #     add_special_tokens=True,
            #     max_length=max_length,
            #     pad_to_max_length=True,
            #     return_attention_mask=True,
            #     return_tensors='pt',
            # )
            encoded_dict = tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=max_length,
                padding='max_length',  # Updated padding argument
                return_attention_mask=True,
                return_tensors='pt',
                truncation=True
            )

            input_ids.append(encoded_dict['input_ids'].to(self.device))
            attention_masks.append(encoded_dict['attention_mask'].to(self.device))

        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_masks)

        first_output = outputs[0]
        middle_first_output = first_output[:, 0, :]
        features = middle_first_output.cpu().numpy()
        return features

if __name__ == '__main__':
    bert = Bert()
    text = "Hello, world! This is a test."
    print("Original:", text)

    """embeddings = bert.get_token_embeddings()

    tokenized = bert.tokenize(text)
    tokenized_tensor = torch.tensor(tokenized).unsqueeze(0).to(bert.device)
    print("Tokenized:", tokenized)

    embedded: List[np.array] = bert.embed_tokenized(tokenized)
    #only print the first 100 symbols
    print("Embedded:", str(embedded)[:100])
    print("fully embedded:", bert.fully_embed_tokenized(tokenized_tensor))

    deemedbedded = bert.decode2tokenized(embedded)
    print("Decoded embedding to tokens:", bert.decode2tokenized(embedded))

    detokenized = bert.detokenize(deemedbedded)
    print("Detokenized actual:", detokenized)

    detokenized = bert.detokenize(tokenized)
    print("Detokenized should:", detokenized)"""

    features = bert.bert_encode([text], bert.tokenizer, bert.model, 512)
    print("Features:", features)
