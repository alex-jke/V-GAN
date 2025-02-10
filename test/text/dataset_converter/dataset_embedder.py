import unittest
from time import time

import torch
from torch import Tensor

from text.Embedding.bert import Bert
from text.Embedding.deepseek import DeepSeek1B
from text.Embedding.gpt2 import GPT2
from text.dataset.emotions import EmotionDataset
from text.dataset.imdb import IMBdDataset
from src.text.dataset_converter.dataset_embedder import DatasetEmbedder
from text.dataset_converter.dataset_tokenizer import DatasetTokenizer


class DatasetEmbedderTest(unittest.TestCase):
    sample_text = "Hello world"
    def test_embedding_dim(self):
        models = [GPT2(), Bert(), DeepSeek1B()]
        dimensions = [768, 768, 1536]
        for i in range(len(models)):
            tokenized = models[i].tokenize(self.sample_text)
            tokenized_tensor = Tensor(tokenized).unsqueeze(0).int()
            embedding_fun = models[i].get_embedding_fun()
            embedded = embedding_fun(tokenized_tensor)
            self.assertEqual(embedded.shape[0], dimensions[i], f"Model {models[i].model_name} has wrong dimension: "
                                                               f"{embedded.shape}, expected: {dimensions[i]}, 1")

    def test_imdb(self):
        with torch.no_grad():
            model = GPT2()
            dataset = IMBdDataset()
            tokenizer = DatasetTokenizer(model, dataset, max_samples=100)
            tokenized_data = tokenizer.get_tokenized_training_data().to(model.device)
            embedding_fun = model.get_embedding_fun(batch_first=True)
            embedded = embedding_fun(tokenized_data)
            self.assertEqual(embedded.shape[0], 100, f"Model {model.model_name} has wrong dimension: "
                                                       f"{embedded.shape}, expected: 100, 768")

    def test_embedding_speed(self):
        start_time = time()
        model = DeepSeek1B()
        dataset = IMBdDataset()
        training_data, _ = dataset.get_training_data()
        amount = 12
        filtered_data = training_data[:amount].tolist()
        tokenized = model.tokenize_batch(filtered_data)
        embed_start_time = time()
        embed = model.get_embedding_fun(batch_first=True)(tokenized)
        end_time = time()
        print(f"Embedding {amount} samples took {end_time - start_time} seconds.")
        print(f"total time per embedding: {(end_time - start_time) / amount} seconds.")
        print(f"per embedding {(end_time - embed_start_time) / amount} seconds.")
        # 2.024, 1.799 for standard
        # 1.520, 1.487 for half precision
        # 1.465, 2.180, 1,392 for half precision and postfix tokens trimmed
        # 2.025 for compiled
        # 2.603, 1.817 for torch.inference_mode()
        #1.935, 1,46 trimmed, inference_mode, half

        # time per embedding just embedding:
        # 1.276 for standard
        # 0.905 trimmed + half precision
        # 0.930 for half precision
        # 1.290 trimmed

    def test_embeddings_equal(self):
        """
        Tests if the embeddings are equal for the same input. This is important for caching.
        """
        model = GPT2()
        text = "Hello world"
        tokenized = model.tokenize(text)
        tokenized_tensor = Tensor(tokenized).unsqueeze(0).int()
        embedding_fun = model.get_embedding_fun()
        embedded = embedding_fun(tokenized_tensor)
        embedded2 = embedding_fun(tokenized_tensor)
        self.assertTrue(torch.allclose(embedded, embedded2, atol=1e-8), "Embeddings are not equal")

    def test_cache_equal(self):
        """
        Tests if the embeddings are equal for the same input for non cached and cached embeddings. This is important for caching.
        """
        model = GPT2()
        dataset = EmotionDataset()
        amount = 2
        samples: str = dataset.get_training_data()[0].iloc[:amount].tolist()
        tokenized = model.tokenize_batch(samples)
        tokenized_tensor = Tensor(tokenized).int()
        embedding_fun = model.get_embedding_fun(batch_first=True)
        embedded = embedding_fun(tokenized_tensor)

        embedded_cached, _ = DatasetEmbedder(dataset=dataset, model= model).embed(train=True, samples= amount)

        self.assertEqual(embedded.shape, embedded_cached.shape, "Shapes are not equal")
        self.assertTrue(torch.allclose(embedded, embedded_cached, atol=1e-8), "Cached and uncached embeddings are not equal")

    def test_ui_equal(self):
        model = GPT2()
        dataset = EmotionDataset()
        amount = 20000
        embedder = DatasetEmbedder(dataset=dataset, model=model)
        embedded, labels = embedder.embed(train=True, samples=amount)
        self.assertEqual(embedder.ui, embedder.model.ui, "UIs are not equal")

if __name__ == '__main__':
    unittest.main()
