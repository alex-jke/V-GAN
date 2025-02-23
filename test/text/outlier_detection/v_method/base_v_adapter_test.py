import unittest

import numpy as np

from text.outlier_detection.v_method.base_v_adapter import BaseVOdmAdapter


class BaseVOdmAdapterTest(unittest.TestCase):

    def test_get_top_subspaces(self):

        proba = np.array([0.1, 0.3, 0.2, 0.9, 0.4])
        subspaces = np.array(['A', 'B', 'C', 'D', 'E'])

        # Expect Top-3: highest probabilities: 0.9, 0.4, 0.3
        expected_top_subspaces = np.array(['D', 'E', 'B'])
        expected_top_proba = np.array([0.9, 0.4, 0.3])

        result_subspaces, result_proba = BaseVOdmAdapter._get_top_subspaces(3, proba, subspaces)

        # Check if the results are as expected.
        self.assertTrue(np.array_equal(expected_top_subspaces, result_subspaces))
        self.assertTrue(np.array_equal(expected_top_proba, result_proba))

    def test_get_top_subspaces_treshold(self):

        proba = np.array([0.1, 0.3, 0.2, 0.9, 0.4] + [0.001] * 100)
        subspaces = np.array(['A', 'B', 'C', 'D', 'E'] + [f'F{i}' for i in range(100)])

        # Expect Top-3: highest probabilities: 0.9, 0.4, 0.3
        expected_top_subspaces = np.array(['D', 'E', 'B', "C", "A"])
        expected_top_proba = np.array([0.9, 0.4, 0.3, 0.2, 0.1])

        result_subspaces, result_proba = BaseVOdmAdapter._get_top_subspaces(50, proba, subspaces)

        # Check if the results are as expected.
        self.assertTrue(np.array_equal(expected_top_subspaces, result_subspaces))
        self.assertTrue(np.array_equal(expected_top_proba, result_proba))