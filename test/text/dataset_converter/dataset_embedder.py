import unittest

import torch
from torch import Tensor

from text.Embedding.bert import Bert
from text.Embedding.deepseek import DeepSeek1B
from text.Embedding.gpt2 import GPT2
from text.dataset.imdb import IMBdDataset
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

if __name__ == '__main__':
    unittest.main()
