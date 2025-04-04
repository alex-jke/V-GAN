import unittest
import torch
import numpy as np
from typing import List
from dataclasses import dataclass
from torch import Tensor

from text.outlier_detection.word_based_v_method.text_v_adapter import TextVMMDAdapter
from text.outlier_detection.word_based_v_method.text_v_odm import TextVOdm


# Mock class for PreparedData
@dataclass
class PreparedData:
    x_test: Tensor


# Mock class for space
class MockSpace:
    def __init__(self, test_size):
        self.test_size = test_size

# The method to be tested
_calculate_distances = TextVOdm._calculate_distances


class TestCalculateDistances(unittest.TestCase):

    def test_normal_case(self):
        """Test with normal inputs and verify correct distances are calculated."""
        # Create test data: 3 samples with 2 features each
        x_test = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

        # Create two different embeddings
        embedding1 = PreparedData(x_test=torch.tensor([[1.1, 2.1], [3.1, 4.1], [5.1, 6.1]]))
        embedding2 = PreparedData(x_test=torch.tensor([[0.9, 1.9], [2.9, 3.9], [4.9, 5.9]]))
        embedded_data = [embedding1, embedding2]

        # Create a mock space
        space = MockSpace(test_size=torch.Size([3]))

        # Calculate expected distances manually
        # For each point, find minimum distance to its projections
        expected_distances = torch.tensor([
            min(torch.norm(torch.tensor([1.0, 2.0]) - torch.tensor([1.1, 2.1])),
                torch.norm(torch.tensor([1.0, 2.0]) - torch.tensor([0.9, 1.9]))),
            min(torch.norm(torch.tensor([3.0, 4.0]) - torch.tensor([3.1, 4.1])),
                torch.norm(torch.tensor([3.0, 4.0]) - torch.tensor([2.9, 3.9]))),
            min(torch.norm(torch.tensor([5.0, 6.0]) - torch.tensor([5.1, 6.1])),
                torch.norm(torch.tensor([5.0, 6.0]) - torch.tensor([4.9, 5.9])))
        ])

        # Call the function
        distances = _calculate_distances(x_test, embedded_data, space)

        # Check results
        self.assertEqual(distances.shape, torch.Size([3]))
        torch.testing.assert_close(distances, expected_distances)

    def test_single_sample(self):
        """Test with a single sample."""
        x_test = torch.tensor([[1.0, 2.0]])
        embedding = PreparedData(x_test=torch.tensor([[1.5, 2.5]]))
        embedded_data = [embedding]
        space = MockSpace(test_size=torch.Size([1]))

        distances = _calculate_distances(x_test, embedded_data, space)

        expected = torch.tensor([0.7071])  # sqrt(0.5²+0.5²) ~ 0.7071
        self.assertEqual(distances.shape, torch.Size([1]))
        torch.testing.assert_close(distances, expected, rtol=1e-4, atol=1e-4)

    def test_multiple_embeddings(self):
        """Test with multiple embeddings."""
        x_test = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

        # Create three different embeddings
        embedding1 = PreparedData(x_test=torch.tensor([[1.0, 2.0], [3.0, 4.0]]))  # Same as original
        embedding2 = PreparedData(x_test=torch.tensor([[1.2, 2.2], [3.2, 4.2]]))  # Slight offset
        embedding3 = PreparedData(x_test=torch.tensor([[0.5, 1.5], [2.5, 3.5]]))  # Larger offset

        embedded_data = [embedding1, embedding2, embedding3]
        space = MockSpace(test_size=torch.Size([2]))

        distances = _calculate_distances(x_test, embedded_data, space)

        # For identical projections, distances should be 0
        expected = torch.tensor([0.0, 0.0])
        self.assertEqual(distances.shape, torch.Size([2]))
        torch.testing.assert_close(distances, expected)

    def test_higher_dimensional_data(self):
        """Test with higher-dimensional feature vectors."""
        # 2 samples with 5 features each
        x_test = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0],
                               [6.0, 7.0, 8.0, 9.0, 10.0]])

        embedding = PreparedData(x_test=torch.tensor([[1.5, 2.5, 3.5, 4.5, 5.5],
                                                      [6.5, 7.5, 8.5, 9.5, 10.5]]))
        embedded_data = [embedding]
        space = MockSpace(test_size=torch.Size([2]))

        distances = _calculate_distances(x_test, embedded_data, space)

        # Expected distances calculated manually
        expected = torch.tensor([
            torch.sqrt(torch.tensor(5 * 0.5 ** 2)),  # sqrt(5*0.5²) = sqrt(1.25) ≈ 1.118
            torch.sqrt(torch.tensor(5 * 0.5 ** 2))  # sqrt(5*0.5²) = sqrt(1.25) ≈ 1.118
        ])

        self.assertEqual(distances.shape, torch.Size([2]))
        torch.testing.assert_close(distances, expected)

    def test_empty_embeddings_list(self):
        """Test behavior with empty embeddings list - should raise an error."""
        x_test = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        embedded_data = []
        space = MockSpace(test_size=torch.Size([2]))

        with self.assertRaises(Exception):
            _calculate_distances(x_test, embedded_data, space)

    def test_invalid_input_shape(self):
        """Test with invalid x_test shape (not 2D)."""
        # 1D tensor
        x_test = torch.tensor([1.0, 2.0])
        embedding = PreparedData(x_test=torch.tensor([[1.0, 2.0]]))
        embedded_data = [embedding]
        space = MockSpace(test_size=torch.Size([2]))

        with self.assertRaises(AssertionError):
            _calculate_distances(x_test, embedded_data, space)

    def test_mismatched_shapes(self):
        """Test with mismatched shapes between x_test and embedded_data."""
        x_test = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

        # Different number of samples
        embedding = PreparedData(x_test=torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))
        embedded_data = [embedding]
        space = MockSpace(test_size=torch.Size([2]))

        with self.assertRaises(Exception):
            _calculate_distances(x_test, embedded_data, space)


if __name__ == '__main__':
    unittest.main()