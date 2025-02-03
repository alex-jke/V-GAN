from text.Embedding.deepseek import DeepSeek1B
from text.Embedding.huggingmodel import HuggingModel
from text.dataset.dataset import Dataset
from text.dataset.emotions import EmotionDataset
from text.outlier_detection.odm import OutlierDetectionModel
from pyod.models.lunar import LUNAR



class Lunar(OutlierDetectionModel):

    def __init__(self, dataset: Dataset, model: HuggingModel, train_size: int, test_size: int):
        super().__init__(dataset, model, train_size, test_size)
        self.lunar = LUNAR()
        self.use_embedding()

    def train(self):
        self.lunar.fit(self.x_train.cpu(), self.y_train.cpu())

    def predict(self):
        pass

    def _get_name(self):
        pass

    def _get_predictions_expected(self):
        pass

if __name__ == '__main__':
    lunar = Lunar(dataset=EmotionDataset(), model=DeepSeek1B(), train_size=1000, test_size=100)