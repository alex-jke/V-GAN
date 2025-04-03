from datetime import datetime
import os
from typing import List

import torch
from numpy import ndarray
from pytorch_lightning import Trainer, Callback
from pytorch_lightning.strategies import DeepSpeedStrategy
from torch.utils.data import DataLoader, TensorDataset

from models.Generator import GeneratorSpectralSigmoidSTE, GeneratorSigmoidSTE, GeneratorSoftmaxSTE
from modules.text.vmmd_text import VMMDTextLightning
from text.Embedding.llama import LLama1B
from text.dataset.emotions import EmotionDataset
from text.dataset_converter.dataset_preparer import DatasetPreparer
from text.visualizer.collective_visualizer import CollectiveVisualizer


def _build_export_path() -> str:
    #sl_str = self.sequence_length if self.sequence_length is not None else "(avg)"
    transformer_aggregation = True
    train = True
    base_dir = os.path.join(
        os.getcwd(),
        'experiments',
        "VMMD_Text_Lightning",
        emb_model.__class__.__name__,
        generator.__name__,
        f"{version}",
        f"agg_" + "t" if transformer_aggregation else "avg",
        f"{dataset.name}_sl(avg)_s{samples}"
    )
    if train:
        base_dir += "_" + datetime.now().strftime("%Y%m%d-%H%M%S")
    return base_dir

def visualize(self, epoch: int, model, sentences: ndarray):
    samples = 30
    sentences: List[List[str]] = [self.emb_model.get_words(sentence) for sentence in sentences[:samples]]
    visualizer = CollectiveVisualizer(tokenized_data=sentences, tokenizer=None, vmmd_model=model,
                                      export_path=self.export_path, text_visualization=True)
    visualizer.visualize(epoch=epoch, samples=30)
    model._export(model.generator, export_params=False)

class VisualizationCallback(Callback):
    def __init__(self, emb_model, export_path, dataset, yield_epochs, samples=30):
        self.emb_model = emb_model
        self.export_path = export_path
        self.dataset = dataset
        self.yield_epochs = yield_epochs
        self.samples = samples

    def on_train_epoch_end(self, trainer, pl_module):
        #print("callback triggered")
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

def train():
    # Instantiate your model (pass required hyperparameters)
    model = VMMDTextLightning(emb_model=emb_model, sequence_length=17, lr=lr, weight_decay=weight_decay, weight=penalty_weight)

    # Create a DataLoader or a LightningDataModule for your training data.
    preparer = DatasetPreparer(dataset, max_samples=samples)
    _x_train = preparer.get_training_data()
    x_train = model.get_training_data(_x_train)
    data_loader = DataLoader(x_train, batch_size=batch_size)

    exp_path = _build_export_path()
    vis_cb = VisualizationCallback(
        emb_model=emb_model,
        export_path=exp_path,
        dataset=dataset,
        yield_epochs=yield_epochs,
        samples=30
    )

    trainer = Trainer(max_epochs=epochs, callbacks=[vis_cb],
                      default_root_dir=exp_path,
                      log_every_n_steps=1,
                      accelerator="gpu",
                      # strategy="deepspeed_stage_2_offload"
                      # strategy="ddp"
                      # strategy=DeepSpeedStrategy(offload_optimizer=True)
                      )
    # trainer = Trainer(accelerator="gpu", devices=4, strategy="deepspeed_stage_3_offload", precision=16)

    # Train the model
    trainer.fit(model, train_dataloaders=data_loader)

if __name__ == "__main__":

    emb_model = LLama1B()
    dataset = EmotionDataset()
    generator = GeneratorSoftmaxSTE
    version = "0.03_MixtureRQ+bn+grid"
    sampless = [3000, 5000, 10_000]
    yield_epochs = 1
    batch_size = 100
    penalty_weights = [0.0, .1, .5, 1.0]
    lrs = [1e-1, 1e-2, 1e-3]
    epochss = [10, 25, 75]
    weight_decays = [0.0, 1e-5, 0.1]

    embedding_fun = lambda samples, padding_length, masks: emb_model.embed_sentences(samples, padding_length, masks=masks, aggregate=True)
    for samples in sampless:
        for weight_decay in weight_decays:
            for penalty_weight in penalty_weights:
                for epochs, lr in zip(epochss, lrs):
                    train()



