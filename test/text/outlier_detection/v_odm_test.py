import unittest

import numpy as np
import torch
from numpy import ndarray
from torch.functional import Tensor

from text.outlier_detection.v_method.V_odm import V_ODM


class VODMTest(unittest.TestCase):

    def test_calculate_distances(self):
        transform = lambda x: x
        subspaces =Tensor( [[1, 0], [0, 1]])
        x_test = Tensor([[1,0], [0, 1], [1, 1]])
        normalized = torch.nn.functional.normalize(x_test, p=2, dim=1)
        distances = V_ODM._calculate_distances(transform, subspaces, normalized)

        # The distance between the first two points is 0, as they are in the same subspace.
        # The distance of the last point is 0.707 to each subspace. Devided by the maximum distance of sqrt(2) this results in 0.5.
        self.assertListEqual( [0, 0, 0.5], distances.tolist())
