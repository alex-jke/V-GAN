from typing import Optional

from pytorch_lightning import Callback, Trainer

from modules.text.vmmd_text_base import VMMDTextBase
from text.Embedding.huggingmodel import HuggingModel
from text.dataset.dataset import Dataset
from text.dataset_converter.dataset_preparer import DatasetPreparer
from text.visualizer.collective_visualizer import CollectiveVisualizer


class VisualizationCallback(Callback):
    """
    Callback to visualize the VMMD Lightning model during training.
    """
    def __init__(self, emb_model: HuggingModel, export_path: str, dataset: Dataset, yield_epochs: Optional[int], samples=30):
        """
        The callback to visualize the VMMD Lightning model during training. It should be created before the training starts.
        It is then passed to the Trainer.
        :param emb_model: The embedding model to use as a HuggingModel.
        :param export_path: The path to export the visualizations to as a string.
        :param dataset: The dataset to use as a Dataset.
        :param yield_epochs: The number of epochs to wait before yielding the visualization.
            The callback will start the visualization pipeline every yield_epochs epochs.
            If None, the callback will be triggered 10 times during the training
            (every 10% of the training).
        :param samples: The number of samples to visualize.
        """
        self.emb_model = emb_model
        self.export_path = export_path
        self.dataset = dataset
        self.samples = samples
        self.yield_epochs = yield_epochs

    def on_train_epoch_end(self, trainer: Trainer, pl_module: VMMDTextBase):

        if not isinstance(pl_module, VMMDTextBase):
            raise ValueError("The model must be an instance of VMMDTextBase.")

        if self.yield_epochs is None:
            self.yield_epochs = max(1, trainer.max_epochs // 10)

        epoch = trainer.current_epoch
        if epoch % self.yield_epochs == 0 or epoch == trainer.max_epochs - 1:
            # Prepare data for visualization.
            preparer = DatasetPreparer(self.dataset, max_samples=30)
            x_train = preparer.get_training_data()
            sentences = [self.emb_model.get_words(sentence) for sentence in x_train[:self.samples]]
            visualizer = CollectiveVisualizer(
                tokenized_data=sentences,
                tokenizer=None,
                vmmd_model=pl_module,
                export_path=self.export_path,
                text_visualization=True
            )
            visualizer.visualize(epoch=epoch, samples=self.samples)
            pl_module._export(pl_module.generator, export_params=epoch == trainer.max_epochs - 1, export_path=self.export_path)