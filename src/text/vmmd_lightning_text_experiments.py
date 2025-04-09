from datetime import datetime
import os
from pathlib import Path
from typing import List, Type, Optional, Callable

import numpy as np
import torch
from numpy import ndarray
from pytorch_lightning import Trainer, Callback
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from torch.utils.data import DataLoader

from models.Generator import GeneratorSpectralSigmoidSTE, GeneratorSigmoidSTE, GeneratorSoftmaxSTE, Generator_big
from modules.text.vmmd_text import VMMDTextLightning
from modules.text.vmmd_text_base import VMMDTextBase
from modules.text.vmmd_text_lightning import VMMDTextLightningBase
from text.Embedding.huggingmodel import HuggingModel
from text.Embedding.llama import LLama1B, LLama3B
from text.Embedding.unification_strategy import UnificationStrategy, StrategyInstance
from text.dataset.dataset import Dataset, AggregatableDataset
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
                 dataset: AggregatableDataset,
                 samples: int,
                 yield_epochs: int,
                 lr: float,
                 weight_decay: float,
                 penalty_weight: float,
                 batch_size: int,
                 epochs: int,
                 export_path: Optional[Path] = None,
                 export: Optional[bool] = True,
                 sequence_length: Optional[int] = None,
                 aggregation_strategy: StrategyInstance = UnificationStrategy.TRANSFORMER.create(),
                 train_flag: bool = True,
                 use_mmd: bool = False):
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
        self.strategy = aggregation_strategy
        self.train_flag = train_flag
        self.vmmd_model = vmmd_model
        self.seperator = " "
        self.export_path = export_path
        self.export = export
        self.loaded_model = False
        self.use_mmd = use_mmd
        if self.export_path is None and export:
            self.export_path = self.build_export_path()

    def build_export_path(self) -> Path:
        base_dir = Path(os.path.join(
            os.getcwd(),
            'experiments',
            self.vmmd_model.__name__,
            self.emb_model.__class__.__name__,
            self.generator.__name__,
            f"{self.version}",
            self.strategy.key(),
            f"{self.dataset.name}_sl{self.sequence_length}_s{self.samples}"
        ))
        if self.train_flag:
            base_dir = Path(str(base_dir) + ("_" + datetime.now().strftime("%Y%m%d-%H%M%S")))
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
            export_path=str(self.export_path),
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

    def _prepare_data(self) -> ndarray[str]:
        """
        Prepare the data for training.
        :param model: The VMMDTextLightningBase model to use.
        :return: The training data, as a numpy array of sentences.
        """
        # Prepare training data using a DatasetPreparer.
        preparer = DatasetPreparer(self.dataset, max_samples=self.samples)
        _x_train = preparer.get_training_data()

        # Get the average sentence length from the dataset.
        if self.sequence_length is None:
            self.sequence_length = self._get_average_sentence_length(_x_train)

        return _x_train

    def _prepare_vmmd_model(self, embedding_fun: Callable) -> VMMDTextLightningBase:
        """
        Prepare the VMMD model for training.
        """
        return self.vmmd_model(
            embedding=embedding_fun,
            sequence_length=self.sequence_length,
            lr=self.lr,
            weight_decay=self.weight_decay,
            weight=self.penalty_weight,
            generator=self.generator,
            seed=777,
            strategy=self.strategy,
            use_mmd=self.use_mmd,
            batch_size=self.batch_size, # This is not needed for the training itself, just for the params file.
            epochs=self.epochs, # This is also not needed for the training itself, just for the params file.
        )

    def _load_if_exists(self, model: VMMDTextLightningBase, embedding_fun) -> VMMDTextLightningBase:
        """
        Load the model if it exists, otherwise return the model.
        :param model: The VMMDTextLightningBase model to load.
        :return: The loaded model, or the original model if it does not exist.
        """
        if self.export_path is not None and self.export_path.exists():
            print(f"Loading model from {self.export_path}")
            ckpt_path_dir = self.export_path / "tensorboard_logs" / "version_0" / "checkpoints"
            ckpt_path_list = list(ckpt_path_dir.glob("*.ckpt"))
            ckpt_path = ckpt_path_list[-1] if len(ckpt_path_list) > 0 else None
            if ckpt_path is None:
                raise FileNotFoundError(f"Checkpoint not found in {ckpt_path_dir}.")
                #print("No checkpoint found. This is unexpected behavior. If the model path exists a ckpt file should also exist.")
                #return model
            model = VMMDTextLightning.load_from_checkpoint(checkpoint_path=ckpt_path)

            # The embedding function is not saved in the checkpoint, so it needs to be set it again.
            model.embedding = embedding_fun
            self.loaded_model = True
        return model

    def run(self)  -> VMMDTextLightningBase:
        """
        Run the VMMD experiment, including training and visualization.
        The parameters for the experiment are set in the constructor.
        The training data is prepared using the DatasetPreparer class.
        The model is trained using PyTorch Lightning.
        The visualization is done using the CollectiveVisualizer class.
        :return: The trained VMMD model.
        """
        # Instantiate the model with the provided hyperparameters.
        _x_train = self._prepare_data()

        #strategy = UnificationStrategy.TRANSFORMER.create() if self.transformer_aggregation else UnificationStrategy.MEAN.create()

        embedding_fun = lambda samples, padding_length, masks: self.emb_model.embed_sentences(sentences=samples,
                                                                                         masks=masks,
                                                                                         strategy = self.strategy,
                                                                                         dataset=self.dataset)
        model = self._prepare_vmmd_model(embedding_fun)

        # Load the model if it exists.
        model = self._load_if_exists(model, embedding_fun)
        if self.loaded_model:
            print("Model loaded successfully.")
            return model

        x_train = model.get_training_data(_x_train, embedding_fun, _x_train)

        data_loader = DataLoader(x_train, batch_size=self.batch_size, drop_last=True, pin_memory=True,
            shuffle=True, num_workers=10, persistent_workers=True)

        # Set up the visualization callback.
        vis_cb = VisualizationCallback(
            emb_model=self.emb_model,
            export_path=str(self.export_path),
            dataset=self.dataset,
            yield_epochs=self.yield_epochs,
            samples=30
        )

        tensorboard_logger = TensorBoardLogger(
            save_dir=self.export_path,
            name="tensorboard_logs"
        )
        print("TensorBoard Logs at: ", self.export_path  / "tensorboard_logs")

        csv_logger = CSVLogger(
            save_dir=self.export_path,
            name="lightning_logs",
            version=0
        )

        loggers = [tensorboard_logger, csv_logger] if self.export else []

        trainer = Trainer(
            max_epochs=self.epochs,
            callbacks=[vis_cb],
            default_root_dir=self.export_path,
            log_every_n_steps=1, # Log every step, as the visualizer loads the csv file created by the logger.
            accelerator="gpu",
            logger=loggers,
        )
        # Start training.
        trainer.fit(model, train_dataloaders=data_loader)
        return model


if __name__ == "__main__":

    emb_model = LLama3B()
    dataset = EmotionDataset()
    generator = GeneratorSigmoidSTE
    version = "0.056"
    sampless = [3000]
    yield_epochs = 1
    batch_size = 100
    penalty_weights = [0.0]
    lrs = [5e-3]
    epochss = [25]
    weight_decays = [0.0]
    for strategy in [UnificationStrategy.TRANSFORMER.create(), UnificationStrategy.MEAN.create()]:
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
                            epochs=epochs,
                            aggregation_strategy=strategy,
                            use_mmd = False,
                        )
                        exp.run()



#python -m tensorboard.main --logdir=
#/home/i40/jenkea/PycharmProjects/V-GAN/src/text/experiments/VMMDTextLightning/LLama3B/GeneratorSigmoidSTE/0.054_MSE/agg_avg/Emotions_slNone_s3000/_20250407-083216/tensorboard_logs