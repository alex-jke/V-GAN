import unittest
import random
from typing import Tuple, List

import torch
from torch import Tensor, nn

from text.Embedding.deepseek import DeepSeek1B
from text.Embedding.llama import LLama1B
from text.dataset.emotions import EmotionDataset

device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

class DeepSeekTest(unittest.TestCase):

    def test_causal_mask_gradient(self):
        model = DeepSeek1B()
        #sample = ["I"]  # ["I", "am", "feeling"]
        #mask = Tensor([1  # , 1, 1
        #               ]).to(device)
        generator = Generator()

        for sample, mask in generator(10):

            embedding = model.embed_words(sample, mask)
            self.assertIsNotNone(embedding.grad_fn)
            embedding.mean().backward()
            #self.assertIsNotNone(mask.grad)
            params = generator.parameters()
            self.assertIsNotNone(params)
            for param in params:
                grad = param.grad
                self.assertIsNotNone(grad)
                # TODO: Figure out, why the gradient is zero, even for LLama1B
                self.assertNotEqual(grad.norm().sum(), 0.0)

            embedding_no_mask = model.embed_words(sample)
            self.assertIsNone(embedding_no_mask.grad_fn)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.rand = nn.Parameter(torch.Tensor([0.8]))

    def get_samples_and_random_masks(self, amount: int) -> List[Tuple[List[str], Tensor]]:
        possible_words = ["I", "am", "feeling", "good", "bad", "terrible", "like", "doing", "nothing"]
        sample_masks = []
        for i in range(amount):
            sample = random.choices(possible_words, k=random.randint(1, len(possible_words)))
            #mask = torch.rand(len(sample)).round().to(device)
            mask = (torch.rand(len(sample)) * self.rand).round().to(device)
            sample_masks.append((sample, mask))
        return sample_masks

    def forward(self, amount: int):
        samples = self.get_samples_and_random_masks(amount)
        return samples