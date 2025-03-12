import os
from datetime import datetime
from typing import List, Type

from numpy import ndarray
from transformers import GPT2Model

from VMMDBase import VMMDBase
from models.Generator import GeneratorSigmoidSTE, Generator_big, GeneratorSoftmaxSTE, GeneratorUpperSoftmax, \
    GeneratorSoftmax, GeneratorSoftmaxSTEMBD, Generator, GeneratorSigmoidSoftmaxSTE, GeneratorSigmoidSoftmaxSigmoid
from modules.text.vmmd_text import VmmdText
from modules.text.vmmd_text_preembed import VMMDTextPreEmbed
from text.Embedding.embedding import Embedding
from text.Embedding.fast_text import FastText
from text.Embedding.gpt2 import GPT2
from text.dataset.ag_news import AGNews
from text.dataset.dataset import Dataset
from text.dataset.emotions import EmotionDataset
from text.dataset_converter.dataset_preparer import DatasetPreparer
from text.v_experiment import VBaseExperiment
from text.visualizer.collective_visualizer import CollectiveVisualizer
from modules.text.vmmd_text_base import VMMDTextBase


class VMMDTextExperiment:

    def __init__(self, dataset: Dataset, version: str, samples: int = -1, sequence_length: int | None = None, train: bool = False, epochs: int = 2000,
                 penalty_weight: float = 0.1, batch_size: int = 2000, weight_decay = 0, generator: Generator_big = GeneratorSigmoidSTE,
                 lr: float = 10e-5, gradient_clipping: bool = False, emb_model: Embedding = FastText(normalize=True),
                 v_method: Type[VMMDTextBase] = VMMDTextPreEmbed):
        self.dataset = dataset
        self.version = version
        self.samples = samples
        self.sequence_length = sequence_length
        self.train = train
        self.epochs = epochs
        self.penalty_weight = penalty_weight
        self.generator = generator
        self.emb_model: Embedding = emb_model
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.lr = lr
        self.gradient_clipping = gradient_clipping
        self.v_method: Type[VMMDTextBase] = v_method
        self.export_path = self._build_export_path()

    def _get_name(self) -> str:
        return "VMMD_Text"

    def run(self):
        model = self.v_method(print_updates=True, path_to_directory=self.export_path, epochs=self.epochs, weight=self.penalty_weight,
                             sequence_length=self.sequence_length, batch_size=self.batch_size, weight_decay=self.weight_decay,
                             generator=self.generator, lr=self.lr, gradient_clipping=self.gradient_clipping)
        embedding_fun = self.emb_model.embed_sentences
        preparer = DatasetPreparer(self.dataset, max_samples=self.samples)
        x_train = preparer.get_training_data()
        for epoch in model.yield_fit(x_train, embedding_fun, yield_epochs=self.epochs // 20):
            self.visualize(epoch, model, x_train)
        self.visualize(self.epochs, model, x_train)

    def visualize(self, epoch: int, model, sentences: ndarray):
        samples = 30
        sentences: List[List[str]] = [self.emb_model.get_words(sentence) for sentence in sentences[:samples]]
        visualizer = CollectiveVisualizer(tokenized_data=sentences, tokenizer=None, vmmd_model=model,
                                          export_path=self.export_path, text_visualization=True)
        visualizer.visualize(epoch=epoch, samples=30)
        model._export(model.generator, export_params=False)

    def _build_export_path(self) -> str:
        sl_str = self.sequence_length if self.sequence_length is not None else "(avg)"
        base_dir = os.path.join(
            os.getcwd(),
            'experiments',
            "VMMD_Text",
            self.emb_model.__class__.__name__,
            self.generator.__name__,
            f"{self.version}",
            f"{self.dataset.name}_sl{sl_str}_s{self.samples}"
        )
        if self.train:
            base_dir += "_" + datetime.now().strftime("%Y%m%d-%H%M%S")
        return base_dir

if __name__ == '__main__':
    """
    params_sig = {"version":"0.139_sigmoid+16_latent", "train":False, "epochs":5_000, "penalty_weight":0.0, "samples":10_000, "weight_decay":0, "generator": GeneratorSigmoidSTE, "lr":5e-4, "gradient_clipping":False}
    params_soft = {"version":"0.139+larger_betas(adam)+sigmoid_act+no_batchnorm", "train":False, "epochs":4_000, "penalty_weight":0, "samples":10_000,
            "weight_decay":0, "generator": GeneratorSigmoidSoftmaxSTE, "batch_size": 1000, "lr":1e-4, "gradient_clipping":False}
    #for params in [params_sig, params_soft]:
    datasets = ([
                   EmotionDataset(), AGNews(),
                   IMBdDataset()
                ] +
                NLP_ADBench.get_all_datasets()[:1] +
                NLP_ADBench.get_all_datasets()[2:]
                + [WikipediaPeopleDataset(), ])
    for dataset in datasets:
        experiment = VMMDTextExperiment(dataset=dataset, **params_sig)
        experiment.run()"""

    params_sig = {"version":"0.1391_sigmoid+16_latent", "train":False, "epochs":10, "penalty_weight":0.0, "samples":100, "weight_decay":0, "generator": GeneratorSigmoidSTE, "lr":5e-3, "gradient_clipping":False,
                  "emb_model": GPT2(), "v_method": VmmdText
                  }
    VMMDTextExperiment(dataset=EmotionDataset(), **params_sig).run()
