import unittest

from text.dataset.ag_news import AGNews
from text.dataset.nlp_adbench import NLP_ADBench
from text.dataset_converter.dataset_preparer import DatasetPreparer


class DatasetPreparerTest(unittest.TestCase):
    def test_get_data_shape(self):
        dataset = NLP_ADBench.agnews()
        preparer = DatasetPreparer(dataset)
        data = preparer.get_training_data()
        self.assertEqual(data.shape, (66098,))