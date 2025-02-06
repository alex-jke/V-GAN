import os
import unittest

from text.Embedding.bert import Bert
from text.Embedding.deepseek import DeepSeek1B
from text.Embedding.gpt2 import GPT2
from text.dataset.ag_news import AGNews
from text.dataset.emotions import EmotionDataset
from text.dataset.imdb import IMBdDataset
from text.od_experiment import Experiment
from text.outlier_detection.VGAN_odm import VGAN_ODM


class ODExperimentTest(unittest.TestCase):

    def test_ag_gpt2(self):
        exp = Experiment(AGNews(), GPT2(), skip_error=False)
        exp.run()

    def test_emotions_gpt2_vgan_lunar_embedding(self):
        dataset = EmotionDataset()
        model = GPT2()
        train_size = 200
        test_size = 20
        vgan = VGAN_ODM(dataset, model, train_size, test_size, pre_embed =True)
        vgan.vgan.lr = 0.5
        vgan.vgan.epochs = 100
        exp = Experiment(dataset, model, skip_error=False,
                         train_size=train_size, test_size=test_size,
                         models=[vgan])
        exp.run()
    def test_emotions_gpt2_vgan_lunar_no_pre_embedding(self):
        dataset = EmotionDataset()
        model = GPT2()
        train_size: int = 2000
        test_size: int = 200
        vgan = VGAN_ODM(dataset, model, train_size, test_size, pre_embed =False)
        exp = Experiment(dataset, model, skip_error=False,
                         train_size=train_size, test_size=test_size,
                         models=[vgan])
        exp.run()

    def test_imdb_deepseek_vgam_lunar_no_pre_embedding(self):
        dataset = IMBdDataset()
        model = DeepSeek1B()
        train_size: int = 2000
        test_size: int = 200
        vgan = VGAN_ODM(dataset, model, train_size, test_size, pre_embed =False)
        exp = Experiment(dataset, model, skip_error=False,
                         train_size=train_size, test_size=test_size,
                         models=[vgan])
        exp.run()

    def test_all(self):
        train_size = 10
        test_size = 20
        datasets = [EmotionDataset(), IMBdDataset(), AGNews()]
        embedding_models = [GPT2(), Bert(), DeepSeek1B()]
        for dataset in datasets:
            for model in embedding_models:
                exp = Experiment(dataset, model, skip_error=False,
                                 train_size=train_size, test_size=test_size)
                for model in exp.models:
                    if isinstance(model, VGAN_ODM):
                        model.vgan.epochs = 500
                        model.vgan.lr = 0.5
                exp.run()

if __name__ == '__main__':
    unittest.main()
