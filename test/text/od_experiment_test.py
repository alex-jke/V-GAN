import os
import unittest
from typing import List

from pyod.models.lof import LOF as pyodLOF
from pyod.models.lunar import LUNAR as pyodLUNAR

from text.Embedding.LLM.bert import Bert
from text.Embedding.LLM.deepseek import DeepSeek1B, DeepSeek7B
from text.Embedding.LLM.gpt2 import GPT2
from text.dataset.ag_news import AGNews
from text.dataset.dataset import Dataset
from text.dataset.emotions import EmotionDataset
from text.dataset.imdb import IMBdDataset
from text.dataset.nlp_adbench import NLP_ADBench
from text.od_experiment import Experiment
from text.outlier_detection.pyod_odm import LUNAR, LOF, ECOD, FeatureBagging
from text.outlier_detection.space.embedding_space import EmbeddingSpace
from text.outlier_detection.space.token_space import TokenSpace
from text.outlier_detection.v_method.V_odm import V_ODM
from text.outlier_detection.word_based_v_method.text_v_odm import TextVOdm
from text.outlier_detection.word_based_v_method.token_v_adapter import TokenVAdapter


class ODExperimentTest(unittest.TestCase):

    def test_ag_gpt2(self):
        exp = Experiment(AGNews(), GPT2(), skip_error=False, use_cached=True)
        exp.run()

    def test_ag_deepseek_vgan(self):
        dataset = AGNews()
        emb_model = GPT2()
        space = EmbeddingSpace(model=emb_model, train_size=10_000, test_size=10_000)
        model = V_ODM(dataset, space=space, use_cached=True)
        #model.vgan.epochs = 200
        exp = Experiment(dataset, emb_model, skip_error=False, models=[model], use_cached=True)
        exp.run()

    def test_emotion_gpt2(self):
        experiment = Experiment(dataset=AGNews(), emb_model=GPT2(), train_size=-1, test_size=-1,
                                experiment_name=f"test_refactor", run_cachable=True, use_cached=True, skip_error=False)
        experiment.run()

    def test_emotions_gpt2_vgan_lunar_embedding(self):
        dataset = EmotionDataset()
        model = GPT2()
        train_size = 200
        test_size = 20
        vgan = V_ODM(dataset, model, train_size, test_size, pre_embed =True)
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
        vgan = V_ODM(dataset, model, train_size, test_size, pre_embed =True)
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
        vgan = V_ODM(dataset, model, train_size, test_size, pre_embed =False)
        exp = Experiment(dataset, model, skip_error=False,
                         train_size=train_size, test_size=test_size,
                         models=[vgan])
        exp.run()

    def test_emotion_gpt2_vgan_lunar_no_pre_embedding(self):
        dataset = AGNews()
        model = DeepSeek1B()
        train_size: int = 10000
        test_size: int = 1000
        vgan = V_ODM(dataset, model, train_size, test_size, pre_embed=True)
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
                    if isinstance(model, V_ODM):
                        model.vgan.epochs = 500
                        model.vgan.lr = 0.5
                exp.run()

    def test_full_datasets_lunar(self):
        datasets: List[Dataset] =[AGNews()] #NLP_ADBench.get_all_datasets()
        samples = 10_000
        test_samples = 10_000
        embedding_model = GPT2()
        space = EmbeddingSpace(embedding_model, samples, test_samples)
        for dataset in datasets:
            exp = Experiment(dataset, embedding_model, skip_error=False, experiment_name="lunar", models=[LUNAR(dataset, space)]
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

    def test_nlp_adbench_deepseek(self):
        dataset = NLP_ADBench.bbc()
        model = DeepSeek7B()
        train_size = 10_000
        test_size = 10_000
        exp = Experiment(dataset, model, skip_error=False,
                         experiment_name="emotion", train_size=train_size, test_size=test_size,
                         use_cached=True)
        exp.run()

    def test_ranking(self):
        dataset = AGNews()
        model = DeepSeek1B()
        space = EmbeddingSpace(model, 100, 100)
        methods = [#LUNAR(dataset, space, use_cached=True),
                    LOF(dataset, space, use_cached=True),
                    LUNAR(dataset, space, use_cached=True),
                    FeatureBagging(dataset=dataset, space=space, base_detector=pyodLOF, use_cached=True),
                    FeatureBagging(dataset=dataset, space=space, base_detector=pyodLUNAR, use_cached=True)
            ]
        exp = Experiment(dataset, model,  models=methods, runs=5, experiment_name="ranking5")
        exp.run()

    def test_token_vmmd(self):
        dataset = NLP_ADBench.sms_spam()
        model = DeepSeek1B()
        space = TokenSpace(model=model, train_size=1_00, test_size=100)
        v_adapter= TokenVAdapter(dataset=dataset, space=space, inlier_label=0)
        v_method = TextVOdm(dataset=dataset, space=space, v_adapter=v_adapter, inlier_label=0)
        v_method.train()
        v_method.predict()
        result_df = v_method.evaluate()[0]
        print(result_df.columns.to_list())
        print(result_df.iloc[0].to_list())


if __name__ == '__main__':
    unittest.main()
