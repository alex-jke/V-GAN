import random
from typing import List

import numpy as np
from torch import Tensor

from .gpt2 import GPT2


class GPT2ExtraSubspaces(GPT2):

    def __init__(self, subspaces: int):
        self.subspaces = subspaces
        super().__init__()

    def tokenize(self, data: str) -> List[int]:
        tokenized = super().tokenize(data)
        subspace = random.Random().randint(0, self.subspaces - 1)
        return [0] * len(tokenized) * subspace + tokenized

