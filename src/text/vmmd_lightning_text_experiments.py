from datetime import datetime
import os
from typing import List, Type, Optional

import numpy as np
import torch
from numpy import ndarray
from pytorch_lightning import Trainer, Callback
from pytorch_lightning.strategies import DeepSpeedStrategy
from torch.utils.data import DataLoader, TensorDataset

from models.Generator import GeneratorSpectralSigmoidSTE, GeneratorSigmoidSTE, GeneratorSoftmaxSTE, Generator_big
from modules.text.vmmd_text import VMMDTextLightning
from modules.text.vmmd_text_base import VMMDTextBase
from modules.text.vmmd_text_lightning import VMMDTextLightningBase
from text.Embedding.huggingmodel import HuggingModel
from text.Embedding.llama import LLama1B
from text.dataset.dataset import Dataset
from text.dataset.emotions import EmotionDataset
from text.dataset_converter.dataset_preparer import DatasetPreparer
from text.visualizer.collective_visualizer import CollectiveVisualizer
from text.visualizer.lightning.callback import VisualizationCallback


class VMMDLightningTextExperiment:
    """
    A class to run VMMD experiments with PyTorch Lightning on Text data.
    This class is designed to be used with the subclasses of VMMDTextLightningBase.
    It handles the training and visualization of the model.
    """
    def __init__(self,
                 emb_model: HuggingModel,
                 vmmd_model: Type[VMMDTextLightningBase],
                 generator: Type[Generator_big],
                 version: str,
                 dataset: Dataset,
                 samples: int,
                 yield_epochs: int,
                 lr: float,
                 weight_decay: float,
                 penalty_weight: float,
                 batch_size: int,
                 epochs: int,
                 sequence_length: Optional[int] = None,
                 transformer_aggregation: bool = True,
                 train_flag: bool = True):
        self.emb_model = emb_model
        self.generator = generator
        self.version = version
        self.dataset = dataset
        self.samples = samples
        self.yield_epochs = yield_epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.penalty_weight = penalty_weight
        self.batch_size = batch_size
        self.epochs = epochs
        self.sequence_length = sequence_length
        self.transformer_aggregation = transformer_aggregation
        self.train_flag = train_flag
        self.vmmd_model = vmmd_model
        self.seperator = " "
        self.export_path = self.build_export_path()

    def build_export_path(self) -> str:
        base_dir = os.path.join(
            os.getcwd(),
            'experiments',
            self.vmmd_model.__name__,
            self.emb_model.__class__.__name__,
            self.generator.__name__,
            f"{self.version}",
            f"agg_{'t' if self.transformer_aggregation else 'avg'}",
            f"{self.dataset.name}_sl(avg)_s{self.samples}"
        )
        if self.train_flag:
            base_dir += "_" + datetime.now().strftime("%Y%m%d-%H%M%S")
        return base_dir

    def visualize(self, epoch: int, model, sentences: ndarray):
        samples = 30
        tokenized_sentences: List[List[str]] = [
            self.emb_model.get_words(sentence) for sentence in sentences[:samples]
        ]
        visualizer = CollectiveVisualizer(
            tokenized_data=tokenized_sentences,
            tokenizer=None,
            vmmd_model=model,
            export_path=self.export_path,
            text_visualization=True
        )
        visualizer.visualize(epoch=epoch, samples=samples)
        model._export(model.generator, export_params=False)

    def _get_average_sentence_length(self, x_data: ndarray[str]) -> int:
        """
        Calculate the average sentence length in the dataset.
        :param x_data: A numpy array of sentences.
        :return: The average sentence length, as an integer.
        """
        sequence_length = int(np.mean([len(x.split(self.seperator)) for x in x_data]))
        return sequence_length

    def _prepare_data(self, model: VMMDTextLightningBase) -> ndarray[str]:
        """
        Prepare the data for training.
        :param model: The VMMDTextLightningBase model to use.
        :return: The training data, as a numpy array of sentences.
        """
        # Prepare training data using a DatasetPreparer.
        preparer = DatasetPreparer(self.dataset, max_samples=self.samples)
        _x_train = preparer.get_training_data()

        embedding_fun = lambda samples, padding_length, masks: emb_model.embed_sentences(samples, padding_length,
                                                                                         masks=masks, aggregate=True)
        # Get the average sentence length from the dataset.
        if self.sequence_length is None:
            self.sequence_length = self._get_average_sentence_length(_x_train)
        x_train = model.get_training_data(_x_train, embedding_fun, _x_train)
        return x_train

    def _prepare_vmmd_model(self):
        """
        Prepare the VMMD model for training.
        """
        return self.vmmd_model(
            emb_model=self.emb_model,
            sequence_length=self.sequence_length,
            lr=self.lr,
            weight_decay=self.weight_decay,
            weight=self.penalty_weight
        )


    def run(self):
        """
        Run the VMMD experiment, including training and visualization.
        The parameters for the experiment are set in the constructor.
        The training data is prepared using the DatasetPreparer class.
        The model is trained using PyTorch Lightning.
        The visualization is done using the CollectiveVisualizer class.
        :return: None
        """
        # Instantiate the model with the provided hyperparameters.
        model = self._prepare_vmmd_model()

        x_train = self._prepare_data(model)
        data_loader = DataLoader(x_train, batch_size=self.batch_size, drop_last=True, pin_memory=True,
            shuffle=True, num_workers=10, persistent_workers=True)

        # Set up the visualization callback.
        vis_cb = VisualizationCallback(
            emb_model=self.emb_model,
            export_path=self.export_path,
            dataset=self.dataset,
            yield_epochs=self.yield_epochs,
            samples=self.samples
        )

        trainer = Trainer(
            max_epochs=self.epochs,
            callbacks=[vis_cb],
            default_root_dir=self.export_path,
            log_every_n_steps=1, # Log every step, as the visualizer loads the csv file created by the logger.
            accelerator="gpu"
        )
        # Start training.
        trainer.fit(model, train_dataloaders=data_loader)


if __name__ == "__main__":

    emb_model = LLama1B()
    dataset = EmotionDataset()
    generator = GeneratorSoftmaxSTE
    version = "0.03_MixtureRQ+bn+grid"
    sampless = [300]
    yield_epochs = 1
    batch_size = 10
    penalty_weights = [0.0]
    lrs = [1e-2]
    epochss = [25]
    weight_decays = [1e-5]

    for samples in sampless:
        for weight_decay in weight_decays:
            for penalty_weight in penalty_weights:
                for epochs, lr in zip(epochss, lrs):
                    exp = VMMDLightningTextExperiment(
                        emb_model=emb_model,
                        vmmd_model=VMMDTextLightning,
                        generator=generator,
                        version=version,
                        dataset=dataset,
                        samples=samples,
                        yield_epochs=yield_epochs,
                        lr=lr,
                        weight_decay=weight_decay,
                        penalty_weight=penalty_weight,
                        batch_size=batch_size,
                        epochs=epochs
                    )
                    exp.run()



