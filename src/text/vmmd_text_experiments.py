import os
from datetime import datetime

from text.Embedding.fast_text import FastText
from text.dataset.ag_news import AGNews
from text.dataset.dataset import Dataset
from text.dataset.emotions import EmotionDataset
from text.dataset_converter.dataset_preparer import DatasetPreparer
from text.v_experiment import VBaseExperiment
from text.visualizer.collective_visualizer import CollectiveVisualizer
from vmmd_text import VMMD_Text


class VMMDTextExperiment:

    def __init__(self, dataset: Dataset, version: str, samples: int = -1, sequence_length: int = -1, train: bool = False, epochs: int = 2000):
        self.dataset = dataset
        self.version = version
        self.samples = samples
        self.sequence_length = sequence_length
        self.train = train
        self.epochs = epochs
        self.export_path = self._build_export_path()

    def _get_name(self) -> str:
        return "VMMD_Text"

    def run(self):
        model = VMMD_Text(print_updates=True, path_to_directory=self.export_path, epochs=self.epochs)
        embedding_fun = FastText().embed_sentences
        preparer = DatasetPreparer(self.dataset)
        x_train = preparer.get_training_data()
        for epoch in model.yield_fit(x_train, embedding_fun, yield_epochs=self.epochs // 10):
            self.visualize(epoch, model)

    def visualize(self, epoch: int, model):
        visualizer = CollectiveVisualizer(tokenized_data=None, tokenizer=None, vmmd_model=model,
                                          export_path=self.export_path, text_visualization=False)
        visualizer.visualize(epoch=epoch, samples=30)

    def _build_export_path(self) -> str:
        sl_str = self.sequence_length if self.sequence_length > 0 else "(all)"
        base_dir = os.path.join(
            os.getcwd(),
            'experiments',
            "VMMD_Text",
            f"{self.version}",
            f"{self.dataset.name}_sl{sl_str}_s{self.samples}"
        )
        if self.train:
            base_dir += "_" + datetime.now().strftime("%Y%m%d-%H%M%S")
        return base_dir

if __name__ == '__main__':
    dataset = EmotionDataset()
    experiment = VMMDTextExperiment(dataset=dataset, version="0.1", train=True, epochs=20)
    experiment.run()

