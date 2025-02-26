from torch import Tensor

from text.Embedding.bert import Bert
from text.Embedding.deepseek import DeepSeek1B, DeepSeek7B
from text.Embedding.gpt2 import GPT2
from text.dataset.ag_news import AGNews
from text.dataset.emotions import EmotionDataset
from text.dataset.imdb import IMBdDataset
from text.dataset.nlp_adbench import NLP_ADBench
from text.dataset_converter.dataset_embedder import DatasetEmbedder
import random

if __name__ == '__main__':
    test_tensor = Tensor([[1,1], [2,2], [3,3]])
    test_mask = Tensor([1, 0, 1])
    masked = test_tensor * test_mask.unsqueeze(1)
    print(masked)