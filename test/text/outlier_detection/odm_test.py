import unittest
from itertools import takewhile
from time import time

import torch
from torch import Tensor

from text.Embedding.deepseek import DeepSeek1B
from text.Embedding.gpt2 import GPT2
from text.dataset.emotions import EmotionDataset
from text.outlier_detection.pyod_odm import LUNAR


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
        lunar = LUNAR(EmotionDataset(), GPT2(), 400, 200, pre_embed=True, use_cached=True)
        lunar.train()
        lunar.predict()
        print("First time:", time() - first_time)
        second_time = time()
        lunar2 = LUNAR(EmotionDataset(), GPT2(), 400, 200, pre_embed=True, use_cached=False)
        lunar2.train()
        lunar2.predict()
        print("Second time:", time() - second_time)
        self.assertEqual(lunar2.x_train.shape, lunar.x_train.shape)

        cached_x_train: Tensor = lunar.x_train
        uncached_x_train: Tensor = lunar2.x_train

        self.assertTrue(torch.allclose(cached_x_train, uncached_x_train, atol=1e-8), "Cached and uncached x_train are not equal")

