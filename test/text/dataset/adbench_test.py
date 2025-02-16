import unittest

from text.dataset.nlp_adbench import NLP_ADBench


class ADBenchDatasetTest(unittest.TestCase):
    def test_contain_outlier(self):
        no_outliers = []
        no_truncated_outliers = []
        datasets = NLP_ADBench.get_all_datasets()
        for dataset in datasets:
            _, y = dataset.get_testing_data()
            if len(y.unique()) != 2:
                no_outliers.append(dataset.name)
            truncated = y[:1000]
            if len(truncated.unique()) != 2:
                no_truncated_outliers.append(dataset.name)

        self.assertEqual(len(no_outliers), 0, f"Datasets without outliers: {no_outliers}")
        self.assertEqual(len(no_truncated_outliers), 0, f"Datasets without outliers in first 1000 rows:"
                                                        f" {no_truncated_outliers}")