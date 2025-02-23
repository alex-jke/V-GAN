import unittest

import torch
from torch.functional import Tensor

from text.Embedding.deepseek import DeepSeek, DeepSeek1B


class DeepSeekTest(unittest.TestCase):
    device = "cuda" if torch.cuda.is_available() else (
        ("mps" if torch.backends.mps.is_available() else "cpu"))

    def test_embedding_padding(self):
        model = DeepSeek1B()
        tokenized = model.tokenize_batch(["Hello World"])
        print("number tokens:", tokenized.shape[1])
        embedding_fun = model.get_embedding_fun(batch_first=True)
        embedding = embedding_fun(tokenized)

        print("--------------------")
        # Create a tensor of shape (1, 1) with the padding token
        extra_padding_token = Tensor([model.padding_token]*600).unsqueeze(0).to(device=self.device)
        tokenized_with_padding = torch.cat((tokenized, extra_padding_token), dim=1)
        embedding_padded = embedding_fun(tokenized_with_padding)

        self.assertEqual(embedding.shape, embedding_padded.shape)
        self.assertTrue(torch.allclose(embedding, embedding_padded))
