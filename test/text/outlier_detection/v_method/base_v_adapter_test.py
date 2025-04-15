import unittest

import numpy as np

from text.outlier_detection.v_method.numerical_v_adapter import NumericalVOdmAdapter


class BaseVOdmAdapterTest(unittest.TestCase):

    def test_get_top_subspaces(self):

        proba = np.array([0.1, 0.3, 0.2, 0.9, 0.4])
        subspaces = np.array(['A', 'B', 'C', 'D', 'E'])

        # Expect Top-3: highest probabilities: 0.9, 0.4, 0.3
        expected_top_subspaces = np.array(['D', 'E', 'B'])
        expected_top_proba = np.array([0.9, 0.4, 0.3])

        result_subspaces, result_proba = NumericalVOdmAdapter._get_top_subspaces(3, proba, subspaces)

        # Check if the results are as expected.
        self.assertTrue(np.array_equal(expected_top_subspaces, result_subspaces))
        self.assertTrue(np.array_equal(expected_top_proba, result_proba))

    def test_get_top_subspaces_treshold(self):

        proba = np.array([0.1, 0.3, 0.2, 0.9, 0.4] + [0.001] * 100)
        subspaces = np.array(['A', 'B', 'C', 'D', 'E'] + [f'F{i}' for i in range(100)])

        # Expect Top-3: highest probabilities: 0.9, 0.4, 0.3
        expected_top_subspaces = np.array(['D', 'E', 'B', "C", "A"])
        expected_top_proba = np.array([0.9, 0.4, 0.3, 0.2, 0.1])

        result_subspaces, result_proba = NumericalVOdmAdapter._get_top_subspaces(50, proba, subspaces)

        # Check if the results are as expected.
        self.assertTrue(np.array_equal(expected_top_subspaces, result_subspaces))
        self.assertTrue(np.array_equal(expected_top_proba, result_proba))

class TestRemoveZeroSubspaces(unittest.TestCase):

    def test_remove_some_zero_subspaces(self):
        subspaces = np.array([[0, 0], [1, 2], [0, 0]])
        proba = np.array([0.1, 0.9, 0.2])
        exp_subspaces = np.array([[1, 2]])
        exp_proba = np.array([0.9])
        result_subspaces, result_proba = NumericalVOdmAdapter._remove_zero_subspaces(subspaces, proba)
        np.testing.assert_array_equal(result_subspaces, exp_subspaces)
        np.testing.assert_array_equal(result_proba, exp_proba)

    def test_no_zero_subspaces(self):
        subspaces = np.array([[1, 2], [3, 4]])
        proba = np.array([0.3, 0.7])
        result_subspaces, result_proba = NumericalVOdmAdapter._remove_zero_subspaces(subspaces, proba)
        np.testing.assert_array_equal(result_subspaces, subspaces)
        np.testing.assert_array_equal(result_proba, proba)

    def test_all_zero_subspaces(self):
        subspaces = np.array([[0, 0], [0, 0]])
        proba = np.array([0.5, 0.5])
        exp_subspaces = np.empty((0, subspaces.shape[1]))
        exp_proba = np.array([])
        result_subspaces, result_proba = NumericalVOdmAdapter._remove_zero_subspaces(subspaces, proba)
        np.testing.assert_array_equal(result_subspaces, exp_subspaces)
        np.testing.assert_array_equal(result_proba, exp_proba)

    def test_empty_inputs(self):
        subspaces = np.empty((0, 2))
        proba = np.array([])
        result_subspaces, result_proba = NumericalVOdmAdapter._remove_zero_subspaces(subspaces, proba)
        np.testing.assert_array_equal(result_subspaces, subspaces)
        np.testing.assert_array_equal(result_proba, proba)