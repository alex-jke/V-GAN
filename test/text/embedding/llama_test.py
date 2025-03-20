import unittest

import torch
from torch import Tensor

from text.Embedding.llama import LLama1B

device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
class LlamaTest(unittest.TestCase):

    def test_causal_mask(self):
         model = LLama1B()
         sample = ["I"]#["I", "am", "feeling"]
         mask = Tensor([1#, 1, 1
                        ]).to(device)
         mask.requires_grad = True
         embedding = model.embed_words(sample, mask)
         self.assertIsNotNone(embedding.grad_fn)

    def test_gradient_mask(self):
        model = LLama1B()
        mask = Tensor([0., 1., 1.]).to(device)
        mask.requires_grad = True
        mask = mask.unsqueeze(0)
        mask = mask * 1
        causal_mask = model._get_4d_causal_mask(mask)
        self.assertIsNotNone(causal_mask.grad_fn)

    def test_embed_words(self):
        model = LLama1B()
        sample = ["I", "feel", "sad"]
        mask = Tensor([1, 1, 1]).to(device)
        embedding = model.embed_words(sample, mask=mask)
        self.assertIsNotNone(embedding)