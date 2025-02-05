import unittest

from text.Embedding.gpt2 import GPT2
from text.dataset.ag_news import AGNews
from text.od_experiment import perform_experiment


class ODExperimentTest(unittest.TestCase):

    def test_ag_gpt2(self):
        perform_experiment(AGNews(), GPT2())


if __name__ == '__main__':
    unittest.main()
