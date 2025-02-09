import unittest

from text.Embedding.gpt2 import GPT2
from text.dataset.emotions import EmotionDataset
from text.dataset.imdb import IMBdDataset
from text.dataset_converter.dataset_tokenizer import DatasetTokenizer


class DatasetTokenizerTest(unittest.TestCase):

    def test_correct_length(self):
        model = GPT2()
        dataset = EmotionDataset()
        samples = 2000
        filtered_labels = [dataset.get_possible_labels()[0]]
        dt = DatasetTokenizer(tokenizer=model, dataset=dataset, max_samples=samples)
        tokenized_data, actual_labels = dt.get_tokenized_training_data(class_labels=filtered_labels)
        dataset_path = dt.dataset_path
        #dataset_path.unlink()
        self.assertTrue(all([label in filtered_labels for label in actual_labels]), f"Filtered labels are not correct: {actual_labels}, should only contain {filtered_labels}")
        print(f"deleted {dataset_path}")

        self.assertEqual(tokenized_data.shape[0], samples)
        # Delete the created file
