import os
import unittest
from itertools import takewhile
from pathlib import Path
from time import time
from pyod.models.lof import LOF
from pyod.models.lunar import LUNAR as pyod_LUNAR

import pandas as pd
import torch
from torch import Tensor

from text.Embedding.deepseek import DeepSeek1B, DeepSeek14B, DeepSeek7B
from text.Embedding.gpt2 import GPT2
from text.Embedding.llama import LLama3B
from text.Embedding.unification_strategy import UnificationStrategy
from text.dataset.ag_news import AGNews
from text.dataset.emotions import EmotionDataset
from text.outlier_detection.space.embedding_space import EmbeddingSpace
from text.outlier_detection.space.word_space import WordSpace
from text.outlier_detection.v_method.V_odm import V_ODM
from text.outlier_detection.pyod_odm import LUNAR
from text.outlier_detection.v_method.distance_v_odm import DistanceV_ODM
from text.outlier_detection.v_method.ensembe_v_odm import EnsembleV_ODM
from text.outlier_detection.v_method.vmmd_adapter import VMMDAdapter
from text.visualizer.result_visualizer.result_visualizer import ResultVisualizer
from vmmd import VMMD


class OutlierDetectionMethodTest(unittest.TestCase):

    def test_cache_token(self):
        first_time = time()
        model = DeepSeek1B()
        lunar = LUNAR(EmotionDataset(), model, 2000, 200, pre_embed=False, use_cached=True)
        lunar.train()
        lunar.predict()
        print("First time:", time() - first_time)
        second_time = time()
        lunar2 = LUNAR(EmotionDataset(), model, 2000, 200, pre_embed=False, use_cached=False)
        lunar2.train()
        lunar2.predict()
        print("Second time:", time() - second_time)
        self.assertEqual(lunar2.x_train.shape[0], lunar.x_train.shape[0])
        redundant_cache_padding = 1000
        redundant_non_cache_padding = 1000
        for i in range(lunar.x_train.shape[0]):
            cache_sample = lunar.x_train[i]
            non_cache_sample = lunar2.x_train[i]
            is_padding = lambda x: x == model.padding_token
            cache_padding = [x for x in takewhile(is_padding, reversed(cache_sample.tolist()))]
            non_cache_padding = [x for x in takewhile(is_padding, reversed(non_cache_sample.tolist()))]
            redundant_cache_padding =           min(len(cache_padding), redundant_cache_padding)
            redundant_non_cache_padding =   min(len(non_cache_padding), redundant_non_cache_padding)
        self.assertEqual(redundant_cache_padding, redundant_non_cache_padding, "Padding is not equal")
        self.assertEquals(lunar2.x_train.shape, lunar.x_train.shape)

        self.assertTrue(torch.equal(lunar.x_train, lunar2.x_train), "Cached and uncached x_train are not equal")

    def test_cache_embedding(self):
        first_time = time()
        lunar = LUNAR(EmotionDataset(), GPT2(), 20_000, 2000, pre_embed=True, use_cached=True)
        lunar.train()
        lunar.predict()
        lunar.evaluate()
        print("First time:", time() - first_time)
        second_time = time()
        lunar2 = LUNAR(EmotionDataset(), GPT2(), 1000, 200, pre_embed=True, use_cached=False)
        lunar2.train()
        lunar2.predict()
        print("Second time:", time() - second_time)
        self.assertEqual(lunar2.x_train.shape, lunar.x_train.shape)

        cached_x_train: Tensor = lunar.x_train
        uncached_x_train: Tensor = lunar2.x_train

        self.assertTrue(torch.allclose(cached_x_train, uncached_x_train, atol=1e-8), "Cached and uncached x_train are not equal")

    def test_deepseek7b(self):
        model = DeepSeek7B()
        sample = "This is an example."
        tokenized = model.tokenize_batch([sample])
        embedded = model.fully_embed_tokenized(tokenized)[0]
        #df = pd.DataFrame(embedded.cpu().numpy())
        print(embedded)

    def test_auc(self):
        dataset = EmotionDataset()
        model = DeepSeek7B()
        lunar = LUNAR(dataset = dataset, model = model, train_size = 1_000, test_size = 1000, pre_embed = True,
                      use_cached = False)
        lunar.start_timer()
        lunar.train()
        lunar.predict()
        lunar.stop_timer()
        lunar.evaluate()
        print(f"auc: {lunar.metrics['auc']}")

    def test_vgan_subspace_distance(self):
        dataset= AGNews()
        model = DeepSeek1B()
        path =  Path(os.path.dirname(__file__)) / ".."/ ".." / "results" / "classifier_test" / "vgan_subspace_distance.csv"
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
        step_size = 0.2
        #lambdas = [x *step_size for x in range(int(10/step_size),int(20/step_size))]
        deltas = [x * step_size for x in range(int(0/step_size),int(2/step_size))]
        lambdas = [1]
        for lambda_used in lambdas:
            for delta in deltas:
                vgan_dist = V_ODM(dataset, model, 10_000, 10_000, pre_embed=True, use_cached=True,
                                  subspace_distance_lambda=lambda_used, classifier_delta=delta,
                                  output_path=path.parent / "VGAN")
                vgan_dist.start_timer()
                vgan_dist.train()
                vgan_dist.predict()
                vgan_dist.stop_timer()
                result_dist = vgan_dist.evaluate(path.parent)[0]
                result_dist["distance_lambda_used"] = lambda_used
                result_dist["classifier_delta"] = delta
                if path.exists():
                    result_dist.to_csv(path, mode="a", header=False)
                else:
                    result_dist.to_csv(path)
        result_df = pd.read_csv(path)
        visualizer = ResultVisualizer(result=result_df, output_dir=path.parent)
        #x_column = "distance_lambda_used"
        x_column = "classifier_delta"
        y_columns = ["auc", "prauc", "f1"]
        for y_column in y_columns:
            visualizer.visualize(x_column=x_column, y_column=y_column, type=visualizer.line)

    def test_vmmd_only_distance(self):
        dataset = AGNews()
        model = DeepSeek1B()
        space = EmbeddingSpace(model=model, test_size=1_000)
        vmmd = VMMDAdapter(epochs= 300, dataset_specific_params=False, lr=10e-3)
        v_odm = DistanceV_ODM(dataset=dataset, space=space, odm_model=vmmd)
        v_odm.train()
        v_odm.predict()
        result_df, _ = v_odm.evaluate()
        print(result_df)

    def test_hybrid_lunar_vmmd(self):
        dataset = AGNews()
        model = DeepSeek1B()
        space = EmbeddingSpace(model=model, test_size=1_00, train_size=1_00)
        vmmd = VMMDAdapter(epochs=300, dataset_specific_params=False, lr=10e-3)
        v_odm = V_ODM(dataset=dataset, space=space, odm_model=vmmd)
        v_odm.train()
        v_odm.predict()
        result_df: pd.DataFrame = v_odm.evaluate()[0]
        print(result_df.columns.to_list())
        print(result_df.iloc[0].to_list())

    def test_lof_vmmd(self):
        dataset = AGNews()
        model = DeepSeek1B()
        space = EmbeddingSpace(model=model, test_size=1_00)
        vmmd = VMMDAdapter(epochs=300, dataset_specific_params=False, lr=10e-3)
        v_odm = EnsembleV_ODM(dataset=dataset, space=space, odm_model=vmmd, base_detector=LOF)
        v_odm.train()
        v_odm.predict()
        result_df = v_odm.evaluate()[0]
        print(result_df.columns.to_list())
        print(result_df.iloc[0].to_list())

    def test_lunar_vmmd_word(self):
        dataset = EmotionDataset()
        model = LLama3B()
        path = Path(os.path.dirname(__file__)) / ".." / ".." / "results" / "VMMD_Test_manual"
        #space = WordSpace(model=model, train_size=1_00, test_size=1_00, strategy=UnificationStrategy.TRANSFORMER)
        for space in [
            #EmbeddingSpace(model=model, train_size=1_00, test_size=1_00),
            WordSpace(model=model, train_size=1_00, test_size=1_00, strategy=UnificationStrategy.TRANSFORMER)
        ]:
            vmmd = VMMDAdapter(epochs=300, dataset_specific_params=False, lr=1e-3)
            v_odm = EnsembleV_ODM(dataset=dataset, space=space, odm_model=vmmd, base_detector=pyod_LUNAR, output_path=path)
            v_odm.train()
            #detector_numbers_train = [detector.detector_num for detector in v_odm.ensemble_model.base_estimators]
            v_odm.predict()
            #detector_numbers_predict = [detector.detector_num for detector in v_odm.ensemble_model.base_estimators]
            result_df = v_odm.evaluate()[0]
            print(result_df.columns.to_list())
            print(result_df.iloc[0].to_list())







