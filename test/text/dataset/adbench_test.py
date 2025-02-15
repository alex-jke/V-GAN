import unittest

from text.dataset.nlp_adbench import NLP_ADBench


class ADBenchDatasetTest(unittest.TestCase):
    def test_contain_outlier(self):
        no_outliers = []
        datasets = NLP_ADBench.get_all_datasets()
        for dataset in datasets:
            _, y = dataset.get_testing_data()
            if not 1 in y:
                no_outliers.append(dataset.name)

        self.assertEqual(len(no_outliers), 0, f"Datasets without outliers: {no_outliers}")