import os
from datetime import datetime
from typing import List

from numpy import ndarray

from text.Embedding.fast_text import FastText
from text.dataset.ag_news import AGNews
from text.dataset.dataset import Dataset
from text.dataset.emotions import EmotionDataset
from text.dataset.imdb import IMBdDataset
from text.dataset.nlp_adbench import NLP_ADBench
from text.dataset_converter.dataset_preparer import DatasetPreparer
from text.v_experiment import VBaseExperiment
from text.visualizer.collective_visualizer import CollectiveVisualizer
from vmmd_text import VMMD_Text


class VMMDTextExperiment:

    def __init__(self, dataset: Dataset, version: str, samples: int = -1, sequence_length: int = -1, train: bool = False, epochs: int = 2000,
                 penalty_weight: float = 0.1):
        self.dataset = dataset
        self.version = version
        self.samples = samples
        self.sequence_length = sequence_length
        self.train = train
        self.epochs = epochs
        self.penalty_weight = penalty_weight
        self.export_path = self._build_export_path()
        self.emb_model = FastText()

    def _get_name(self) -> str:
        return "VMMD_Text"

    def run(self):
        model = VMMD_Text(print_updates=True, path_to_directory=self.export_path, epochs=self.epochs, weight=self.penalty_weight,
                          sequence_length=self.sequence_length)
        embedding_fun = self.emb_model.embed_sentences
        preparer = DatasetPreparer(self.dataset)
        x_train = preparer.get_training_data()
        for epoch in model.yield_fit(x_train, embedding_fun, yield_epochs=self.epochs // 10):
            self.visualize(epoch, model, x_train)

    def visualize(self, epoch: int, model, sentences: ndarray):
        samples = 30
        sentences: List[List[str]] = [self.emb_model.get_words(sentence) for sentence in sentences[:samples]]
        visualizer = CollectiveVisualizer(tokenized_data=sentences, tokenizer=None, vmmd_model=model,
                                          export_path=self.export_path, text_visualization=True)
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
    datasets = [EmotionDataset(), AGNews(), IMBdDataset()] + NLP_ADBench.get_all_datasets()
    for dataset in datasets:
        experiment = VMMDTextExperiment(dataset=dataset, version="0.1", train=True, epochs=2000, sequence_length=40, penalty_weight=0.1)
        experiment.run()

