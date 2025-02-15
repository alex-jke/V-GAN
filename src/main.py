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
    models = [DeepSeek1B, GPT2, Bert, DeepSeek7B]
    datasets = [AGNews(), EmotionDataset(), IMBdDataset()] + NLP_ADBench.get_all_datasets()
    random.shuffle(datasets)
    for model in models:
        for dataset in datasets:
            embedder = DatasetEmbedder(dataset, model())
            labels = dataset.get_possible_labels()[:1]
            embedder.embed(train=True,samples=-1, labels=labels)
            embedder.embed(train=False,samples=-1)