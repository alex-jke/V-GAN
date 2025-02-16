import os
import unittest
from typing import List

from text.Embedding.bert import Bert
from text.Embedding.deepseek import DeepSeek1B
from text.Embedding.gpt2 import GPT2
from text.dataset.ag_news import AGNews
from text.dataset.dataset import Dataset
from text.dataset.emotions import EmotionDataset
from text.dataset.imdb import IMBdDataset
from text.dataset.nlp_adbench import NLP_ADBench
from text.od_experiment import Experiment
from text.outlier_detection.VGAN_odm import VGAN_ODM
from text.outlier_detection.pyod_odm import LUNAR, LOF, ECOD


class ODExperimentTest(unittest.TestCase):

    def test_ag_gpt2(self):
        exp = Experiment(AGNews(), GPT2(), skip_error=False)
        exp.run()

    def test_ag_deepseek_vgan(self):
        dataset = AGNews()
        emb_model = GPT2()
        model = VGAN_ODM(dataset, emb_model, 10_000, 10_000, use_cached=True, pre_embed=True)
        #model.vgan.epochs = 200
        exp = Experiment(dataset, emb_model, skip_error=False, models=[model], use_cached=True)
        exp.run()

    def test_emotion_gpt2(self):
        experiment = Experiment(dataset=EmotionDataset(), emb_model=GPT2(), train_size=-1, test_size=-1,
                                experiment_name=f"0.2_adam+large", run_cachable=True, use_cached=True, skip_error=False)
        experiment.run()

    def test_emotions_gpt2_vgan_lunar_embedding(self):
        dataset = EmotionDataset()
        model = GPT2()
        train_size = 200
        test_size = 20
        vgan = VGAN_ODM(dataset, model, train_size, test_size, pre_embed =True)
        vgan.vgan.lr = 1e-4
        vgan.vgan.epochs = 3000
        exp = Experiment(dataset, model, skip_error=False,
                         train_size=train_size, test_size=test_size,
                         models=[vgan])
        exp.run()
        print(exp.result_df)
    def test_emotions_gpt2_vgan_lunar_no_pre_embedding(self):
        dataset = EmotionDataset()
        model = GPT2()
        train_size: int = 20000
        test_size: int = 2000
        vgan = VGAN_ODM(dataset, model, train_size, test_size, pre_embed =True)
        exp = Experiment(dataset, model, skip_error=False,
                         train_size=train_size, test_size=test_size,
                         models=[vgan])
        exp.run()
        print(f"auc: {exp.result_df['auc']}")

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

    def test_emotion_gpt2_vgan_lunar_no_pre_embedding(self):
        dataset = AGNews()
        model = DeepSeek1B()
        train_size: int = 10000
        test_size: int = 1000
        vgan = VGAN_ODM(dataset, model, train_size, test_size, pre_embed=True)
        exp = Experiment(dataset, model, skip_error=False,
                         train_size=train_size, test_size=test_size,
                         models=[vgan], use_cached=True)
        exp.run()
        auc = float(exp.result_df["auc"])
        print(exp.result_df)
        print(auc)

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

    def test_full_datasets_lunar(self):
        datasets: List[Dataset] =[AGNews()] #NLP_ADBench.get_all_datasets()
        samples = 100_000
        test_samples = 10_000
        embedding_model = GPT2()
        for dataset in datasets:
            exp = Experiment(dataset, embedding_model, skip_error=False, experiment_name="lunar", models=[LUNAR(dataset, embedding_model, -1, -1)]
                             ,use_cached=True)
            exp.run()

    def test_tiny(self):
        dataset = AGNews()
        model = DeepSeek1B()
        train_size = 50
        test_size = 100
        models = [LUNAR(dataset, model, train_size, test_size, use_cached=True), LOF(dataset, model, train_size, test_size, use_cached=True), ECOD(dataset, model, train_size, test_size, use_cached=True)]
        exp = Experiment(dataset, model, skip_error=False, train_size=train_size, test_size=test_size, models=models, experiment_name="tiny")
        exp.run()

    def test_nlp_adbench_emotion_deepseek(self):
        dataset = NLP_ADBench.emotion()
        model = DeepSeek1B()
        train_size = 100_000
        test_size = 10_000
        exp = Experiment(dataset, model, skip_error=False,
                         experiment_name="emotion", train_size=train_size, test_size=test_size,
                         use_cached=True)
        exp.run()


if __name__ == '__main__':
    unittest.main()
