import inspect
import os
import operator
from collections import defaultdict
from pathlib import Path
from typing import Optional, Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from lightning_fabric import seed_everything
from torch import Tensor
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from matplotlib import pyplot

from colors import VGAN_GREEN, COMPLIMENTARY
from models.Generator import Generator, Generator_big, GeneratorSigmoidSTE
from text.Embedding.huggingmodel import HuggingModel
from text.Embedding.llama import LLama1B


class VMMDLightningBase(pl.LightningModule):
    def __init__(self,
                 embedding: Optional[Callable[[np.ndarray[str], int, Optional[Tensor]], Tensor]] = None,
                 batch_size=500, epochs=500, lr=1e-4, momentum=0.99, seed=777,
                 weight_decay=1e-4, path_to_directory=None, weight=0, generator=None,
                 print_updates=False, gradient_clipping=False):
        super().__init__()
        self.save_hyperparameters()  # Save hyperparameters in Lightning
        self.train_history = defaultdict(list)
        self.generator_loss_key = "generator_loss"
        self.mmd_loss_key = "mmd_loss"
        self.gradient_key = "gradient"
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.momentum = momentum
        self.seed = seed if seed is not None else np.random.randint(10, 10000, 1)
        self.provided_generator = generator if generator is not None else GeneratorSigmoidSTE
        self.weight_decay = weight_decay
        self.path_to_directory = path_to_directory
        self.generator_optimizer = None
        self.embedding = embedding
        self.weight = weight
        self.print_updates = print_updates
        self.apply_gradient_clipping = gradient_clipping
        self._latent_size = None
        seed_everything(self.seed)
        torch.set_float32_matmul_precision('high')

    def _plot_gradients(self):
        gradients = self.train_history[self.gradient_key]
        plt.style.use('ggplot')
        x = np.linspace(1, len(gradients), len(gradients))
        fig, ax = plt.subplots()
        ax.plot(x, gradients, color=VGAN_GREEN, label="Gradient norm", linewidth=2)
        ax.legend(loc='best')
        plt.xlabel("Epoch")
        plt.ylabel(self.gradient_key)
        plt.savefig(Path(self.path_to_directory) / "gradients.png")
        plt.close()

    def _create_plot(self):
        train_df = pd.read_csv(Path(self.path_to_directory) / "lightning_logs" / "version_0" / "metrics.csv")
        self.train_history = train_df.groupby("epoch").mean().reset_index()
        self._plot_gradients()
        generator_y = self.train_history[self.generator_loss_key]
        mmd_y = self.train_history[self.mmd_loss_key]
        x = np.linspace(1, len(generator_y), len(generator_y))
        fig, ax1 = plt.subplots()
        ax1.plot(x, generator_y, color=VGAN_GREEN, label="Generator loss", linewidth=2)
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Generator Loss", color=VGAN_GREEN)
        ax1.tick_params(axis='y', labelcolor=VGAN_GREEN)
        ax2 = ax1.twinx()
        ax2.plot(x, mmd_y, color=COMPLIMENTARY, label="MMD loss", linewidth=2)
        ax2.set_ylabel("MMD Loss", color=COMPLIMENTARY)
        ax2.tick_params(axis='y', labelcolor=COMPLIMENTARY)
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
        fig.tight_layout()
        return plt, ax1

    def _plot_loss(self, path_to_directory):
        plot, _ = self._create_plot()
        plot.savefig(Path(path_to_directory) / "train_history.png", format="png", dpi=1200)
        plot.close()

    def get_params(self) -> dict:
        return {
            'batch size': self.batch_size,
            'epochs': self.epochs,
            'lr': self.lr,
            'momentum': self.momentum,
            'weight decay': self.weight_decay,
            'seed': self.seed,
            'generator optimizer': self.generator_optimizer,
            'penalty weight': self.weight,
            'generator': self.provided_generator.__name__ if inspect.isclass(self.provided_generator)
                         else str(self.provided_generator),
            'gradient_clipping': self.apply_gradient_clipping
        }

    def model_snapshot(self, path_to_directory=None):
        self._plot_loss(path_to_directory)

    def load_models(self, path_to_generator, ndims, device=None):
        if device is None:
            device = self.device()
        self.generator = self.get_the_networks(ndims, latent_size=max(int(ndims / 16), 1)).to(device)
        self.generator.load_state_dict(torch.load(path_to_generator, map_location=device))
        self.generator.eval()
        self.generator_optimizer = f'Loaded Model from {path_to_generator} with {ndims} dimensions'
        self._latent_size = max(int(ndims / 16), 1)

    def get_the_networks(self, ndims: int, latent_size: int, device=None):
        if device is None:
            device = self.device()
        self._latent_size = latent_size
        if inspect.isclass(self.provided_generator):
            generator = self.provided_generator(img_size=ndims, latent_size=latent_size).to(device)
        else:
            generator = self.provided_generator
        return generator

    def generate_subspaces(self, nsubs, round=True):
        if self._latent_size is None:
            raise RuntimeError('Latent size not set.')
        noise_tensor = torch.Tensor(nsubs, self._latent_size).to('cpu')
        torch.manual_seed(self.seed)
        noise_tensor.normal_()
        self.generator = self.generator.to(self.device())
        noise_tensor = noise_tensor.to(self.device())
        u = self.generator(noise_tensor)
        if round:
            u = (u >= 0.5).int()
        return u.detach()

    def _get_data_loader(self, data: np.array):
        num_workers = 0
        pin_memory = torch.cuda.is_available() or torch.backends.mps.is_available()
        return DataLoader(
            data, batch_size=self.batch_size, drop_last=True, pin_memory=pin_memory,
            shuffle=True, num_workers=num_workers, persistent_workers=False
        )

    def _get_noise_tensor(self, latent_size: int):
        if torch.cuda.is_available():
            return torch.FloatTensor(self.batch_size, latent_size).to("cuda")
        elif torch.backends.mps.is_available():
            return torch.FloatTensor(self.batch_size, latent_size).to(torch.device('mps'))
        else:
            return torch.FloatTensor(self.batch_size, latent_size)

    def _export(self, generator, export_params=True, export_path: Optional[str] = None):
        path = self.path_to_directory

        if path is None:
            path = export_path
            self.path_to_directory = path

        if path is not None:
            path_to_directory = Path(path)
            if not path_to_directory.exists():
                os.makedirs(path_to_directory)
            models_dir = path_to_directory / 'models'
            if not models_dir.exists():
                os.mkdir(models_dir)
            run_number = int(len(os.listdir(models_dir)))
            if export_params:
                torch.save(generator.state_dict(), models_dir / f'generator_{run_number}.pt')
            self.model_snapshot(path_to_directory)

    def device(self) -> torch.device:
        return torch.device('cuda' if torch.cuda.is_available()
                            else ("mps" if torch.backends.mps.is_available() else "cpu"))

    # The following methods must be implemented in subclasses:
    def training_step(self, batch, batch_idx):
        raise NotImplementedError("Define training_step in subclass.")

    def configure_optimizers(self):
        raise NotImplementedError("Define configure_optimizers in subclass.")