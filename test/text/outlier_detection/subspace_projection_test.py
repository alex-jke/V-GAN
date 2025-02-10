import unittest

from torch import Tensor


class SubspaceProjectionTest(unittest.TestCase):

    def test_projection_trimming(self):
        subspace = Tensor([1,0,0,1,0])
        dataset = Tensor([[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15]])
        subspace_expanded = subspace.expand(dataset.shape[0], -1)
        projected = dataset * subspace_expanded
        subspace_mask = subspace != 0
        trimmed = projected[:,subspace_mask]
        self.assertEqual(trimmed.shape[1], 2)
