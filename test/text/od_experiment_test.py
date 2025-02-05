import unittest

from text.Embedding.gpt2 import GPT2
from text.dataset.ag_news import AGNews
from text.dataset.emotions import EmotionDataset
from text.od_experiment import Experiment
from text.outlier_detection.VGAN_odm import VGAN_ODM


class ODExperimentTest(unittest.TestCase):

    def test_ag_gpt2(self):
        exp = Experiment(AGNews(), GPT2(), skip_error=False)
        exp.run()

    def test_emotions_gpt2_vgan_lunar_embedding(self):
        dataset = EmotionDataset()
        model = GPT2()
        train_size = 200
        test_size = 20
        vgan = VGAN_ODM(dataset, model, train_size, test_size, pre_embed =True)
        vgan.vgan.lr = 0.5
        vgan.vgan.epochs = 100
        exp = Experiment(dataset, model, skip_error=False,
                         train_size=train_size, test_size=test_size,
                         models=[vgan])
        exp.run()
    def test_emotions_gpt2_vgan_lunar_no_pre_embedding(self):
        dataset = EmotionDataset()
        model = GPT2()
        train_size = 200
        test_size = 20
        vgan = VGAN_ODM(dataset, model, train_size, test_size, pre_embed =False)
        vgan.vgan.lr = 0.5
        vgan.vgan.epochs = 400
        exp = Experiment(dataset, model, skip_error=False,
                         train_size=train_size, test_size=test_size,
                         models=[vgan])
        exp.run()

if __name__ == '__main__':
    unittest.main()
