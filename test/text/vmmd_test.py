import unittest

from models.Generator import GeneratorSigmoidSTE
from modules.od_module import VMMD_od
from text.Embedding.deepseek import DeepSeek1B
from text.dataset.ag_news import AGNews
from text.dataset.emotions import EmotionDataset
from text.dataset_converter.dataset_tokenizer import DatasetTokenizer
from vmmd import model_eval


class vmmdExperimentTest(unittest.TestCase):

    def test_ag_deepseek(self):
        deepseek = DeepSeek1B()
        ag_news = EmotionDataset()
        vmmd = VMMD_od(epochs=2000, penalty_weight=0.1, generator=GeneratorSigmoidSTE, lr=1e-5, print_updates=True)

        tokenizer = DatasetTokenizer(tokenizer=deepseek, dataset=ag_news, max_samples=20000)
        tokenized = tokenizer.get_tokenized_training_data(class_labels=ag_news.get_possible_labels()[:1])[0]

        embedding_fun = deepseek.get_embedding_fun(batch_first=True)
        embedded = embedding_fun(tokenized)

        for epoch in vmmd.yield_fit(embedded, yield_epochs=200):
            print(f"Epoch: {epoch}")
        model_eval(vmmd, embedded.cpu())