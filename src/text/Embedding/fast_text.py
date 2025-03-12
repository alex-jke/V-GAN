import time
from typing import List, Optional

import numpy as np
import torch.cuda
from staticvectors import StaticVectors
from torch import Tensor

from text.Embedding.deepseek import DeepSeek1B
from text.Embedding.embedding import Embedding
from text.Embedding.huggingmodel import HuggingModel
from text.Embedding.tokenizer import Tokenizer
from text.dataset.ag_news import AGNews
from text.dataset.emotions import EmotionDataset
from text.dataset_converter.dataset_preparer import DatasetPreparer


class FastText(Embedding):

    def __init__(self, normalize: bool = True):
        super().__init__()
        self.model = StaticVectors("neuml/fasttext")
        self.normalize = normalize

    def embed(self, data: str) -> np.ndarray:
        try:
            return self.model.embeddings(data.split(" "), normalize=self.normalize).mean(0)
        except ValueError as e:
            raise e

    def embed_words(self, words: List[str], mask: Optional[Tensor]) -> np.ndarray:
        if mask is not None:
            raise NotImplementedError("Masking is not implemented for FastText.")
        return self.model.embeddings(words, normalize=self.normalize)


if __name__ == '__main__':
    #print(f"time taken for MiniLM embeddings") # 22.97/1000s
    #print(f"time taken for DeepSeek embedding {time.time() - start} seconds") # 149.21/1000s
    #print(f"time taken for DeepSeek tokenizations {time.time() - start} seconds") # 0.30/1000s
    #sentences = ["This is an example sentence", "Each sentence is converted"]
    #np_sentence = np.array([sentence.split(" ") for sentence in sentences])
    #np_sentences = [np.array(sentence.split(" ")) for sentence in sentences]
    #np_sentence = np.stack(np_sentences)
    #print(np_sentence)
    fast_text = FastText()
    #embed = fast_text.embed("This is an example sentence")
    #print(embed.shape)
    #embed_words = fast_text.embed_words(["This", "is", "an", "example", "sentence"])
    #print(embed_words.shape)
    #print("embed nothing:" , fast_text.embed(""))
    dataset = AGNews()
    preparer = DatasetPreparer(dataset)
    x = preparer.get_training_data()
    embedded = fast_text.embed_sentences(x, 50)
    print(embedded.shape)
    test_mask = Tensor([1,0]*25)

    masked = embedded * test_mask.unsqueeze(1)

    print(masked.shape)

    #print(f"time taken for FastText embedding {time.time() - start} seconds") #0.14/1000s
