import unittest

import torch
from torch import Tensor

from text.Embedding.llama import LLama

device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
class LlamaTest(unittest.TestCase):

    def test_causal_mask(self):
         model = LLama()
         sample = ["I", "am", "feeling"]
         mask = Tensor([0, 1, 1]).to(device)
         mask.requires_grad = True
         embedding = model.embed_words(sample, mask)
         self.assertIsNotNone(embedding.grad_fn)

    def test_gradient_mask(self):
        model = LLama()
        mask = Tensor([0., 1., 1.]).to(device)
        mask.requires_grad = True
        mask = mask.unsqueeze(0)
        mask = mask * 1
        causal_mask = model._get_4d_causal_mask(mask)
        self.assertIsNotNone(causal_mask.grad_fn)