import inspect
from typing import Optional, List

import torch
from collections import defaultdict

from matplotlib import pyplot
from torch.utils.data import DataLoader

from colors import VGAN_GREEN, COMPLIMENTARY
from i_vmmd_base import IVMMDBase
from models.Generator import Generator, Generator_big

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import os
import operator

from models.Mmd_loss_constrained import MMDLossConstrained


class VMMDBase(IVMMDBase):
    '''
    Base class for V-MMD and its variants, providing common functionality for training and visualization.
    This class is abstract and should not be instantiated directly. It is the base class for the non-kernel learning
    V-MMD, the application of a GMMN to the problem of Subspace Generation. As a GMMN, no kernel learning is performed.

    '''

    def __init__(self, batch_size=500, epochs=500, lr=10e-5, momentum=0.99, seed=777, weight_decay=10e-5, path_to_directory=None,
                 weight=0, generator = None, print_updates=None, gradient_clipping=False, export_generator=True):
        self.storage = locals()
        self.train_history = defaultdict(list)
        self.generator_loss_key = "generator_loss"
        self.mmd_loss_key = "mmd_loss"
        self.gradient_key = "gradient"
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.momentum = momentum
        self.seed = seed
        if self.seed is None:
            self.seed = np.random.randint(10, 10000, 1)
        self.provided_generator = generator
        if generator is None:
            self.provided_generator = Generator_big
        self.weight_decay = weight_decay
        self.path_to_directory = path_to_directory
        self.generator_optimizer = None
        self.weight = weight
        self.latent_size_factor = 16
        self.export_generator = export_generator
  
        self.device = torch.device('cuda:0' if torch.cuda.is_available(
        ) else 'mps:0' if torch.backends.mps.is_available() else 'cpu')

        self.print_updates = print_updates
        if print_updates is None:
            self.print_updates = False
        self.apply_gradient_clipping = gradient_clipping
        self._latent_size: int | None = None
        self.cuda = torch.cuda.is_available()
        self.mps = torch.backends.mps.is_available()



    def _create_plot(self) -> pyplot:
        self._plot_gradients()
        train_history = self.train_history
        plt.style.use('ggplot')
        generator_y = train_history[self.generator_loss_key]
        mmd_y = train_history[self.mmd_loss_key]
        x = np.linspace(1, len(generator_y), len(generator_y))

        # Create figure and primary axis for generator loss
        fig, ax1 = plt.subplots()

        # Plot generator loss on primary y-axis
        color1 = VGAN_GREEN
        ax1.plot(x, generator_y, color=color1, label="Generator loss", linewidth=2)
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Generator Loss", color=color1)
        ax1.tick_params(axis='y', labelcolor=color1)

        # Create secondary y-axis for MMD loss
        ax2 = ax1.twinx()
        color2 = COMPLIMENTARY
        ax2.plot(x, mmd_y, color=color2, label="MMD loss", linewidth=2)
        ax2.set_ylabel("MMD Loss", color=color2)
        ax2.tick_params(axis='y', labelcolor=color2)

        # Create a combined legend that shows both lines
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

        # Adjust layout to prevent overlap of labels
        fig.tight_layout()

        return plt, ax1

    def get_params(self) -> dict:
        return {'batch size': self.batch_size, 'epochs': self.epochs, 'lr_g': self.lr,
                'momentum': self.momentum, 'weight decay': self.weight_decay,
                'batch_size': self.batch_size, 'seed': self.seed,
                'generator optimizer': self.generator_optimizer,
                'penalty weight': self.weight,
                'generator': self.generator.__class__.__name__,
                'gradient_clipping': self.apply_gradient_clipping}

    def model_snapshot(self, path_to_directory=None, run_number=0, show=False):
        ''' Creates an snapshot of the model

        Saves important information regarding the training of the model
        Args:
            - path_to_directory (Path): Specifies the path to directory (relative to the WD)
            - show (bool): Boolean specifying if a pop-up window should open to show the plot for previsualization.
        '''

        if path_to_directory == None:
            path_to_directory = self.path_to_directory
        path_to_directory = Path(path_to_directory)
        if operator.not_(path_to_directory.exists()):
            os.mkdir(path_to_directory)
        if operator.not_((path_to_directory/"train_history").exists()):
            os.mkdir(path_to_directory / "train_history")

        pd.DataFrame({self.generator_loss_key: self.train_history[self.generator_loss_key],
                      self.mmd_loss_key: self.train_history[self.mmd_loss_key]}).to_csv(
            path_to_directory/'train_history'/f'generator_loss_{run_number}.csv', header=False, index=False)
        if os.path.isfile(path_to_directory/'params.csv') != True:
            pd.DataFrame(self.get_params(), [0]).to_csv(
                path_to_directory / 'params.csv')
        else:
            params = pd.read_csv(path_to_directory / 'params.csv', index_col=0)
            params_new = pd.DataFrame(self.get_params(), [run_number])
            params = params.reindex(params.index.union(params_new.index))
            params.update(params_new)
            params.to_csv(
                path_to_directory / 'params.csv')
        self._plot_loss(path_to_directory, show)

    def load_models(self, path_to_generator, ndims, device: str = None):
        '''Loads models for prediction

        In case that the generator has already been trained, this method allows to load it (and optionally the discriminator) for generating subspaces
        Args:
            - path_to_generator: Path to the generator (has to be stored as a .keras model)
            - path_to_discriminator: Path to the discriminator (has to be stored as a .keras model) (Optional)
        '''
        if device is None:
            device = self.device
        #self.generator = Generator_big(img_size=ndims, latent_size=max(int(ndims/16), 1)).to(device)
        self.generator = self.get_the_networks(ndims, latent_size=max(int(ndims/self.latent_size_factor), 1)).to(device)
        self.generator.load_state_dict(torch.load(path_to_generator, map_location=device))
        self.generator.eval()  # This only works for dropout layers
        self._latent_size = max(int(ndims / self.latent_size_factor), 1)
        self.generator_optimizer = f'Loaded Model from {path_to_generator} with {ndims} dimensions in the full space and {self._latent_size} latent size.'
        print(self.generator_optimizer)

    def generate_subspaces(self, nsubs, round = True):
        # Need to load in cpu as mps Tensor module doesn't properly fix the seed
        if self._latent_size is None:
            raise RuntimeError('Latent size not set.')
        noise_tensor = torch.Tensor(nsubs, self._latent_size).to('cpu')
        if not self.seed == None:
            torch.manual_seed(self.seed)
        noise_tensor.normal_()
        u = self.generator(noise_tensor.to(self.device))
        #u = torch.greater_equal(u, 1/u.shape[1])
        if round:
            u = torch.greater_equal(u, 0.5) * 1
        u = u.detach()

        return u

    def _set_seed(self):
        torch.manual_seed(self.seed)
        if self.cuda:
            torch.cuda.manual_seed(self.seed)
        elif self.mps:
            torch.mps.manual_seed(self.seed)

    def _get_data_loader(self, data: np.array):
        num_workers = 0
        persistent_works = False
        if self.cuda:
            return DataLoader(
                data, batch_size=self.batch_size, drop_last=True, pin_memory=not self.cuda, shuffle=True,
                num_workers=num_workers, persistent_workers=persistent_works)
        else:  # Uses CUDA if Available, otherwise MPS or nothing
            return DataLoader(
                data, batch_size=self.batch_size, drop_last=True, pin_memory=self.mps, shuffle=True,
                num_workers=num_workers, persistent_workers=persistent_works)

    def _get_noise_tensor(self, latent_size: int):
        if self.cuda:  # Need to open this if statement as the Tensor function has to be called from diferent modules depending of the device
            # noise_tensor = torch.cuda.FloatTensor(self.batch_size, latent_size)#.to(self.device)#to(torch.device('cuda'))
            return torch.FloatTensor(self.batch_size, latent_size).to(self.device)
        elif self.mps:
            return torch.mps.Tensor(
                self.batch_size, latent_size).to(torch.device('mps'))
        else:
            return torch.Tensor(self.batch_size, latent_size)

    def _export(self, generator, export_params= True, export_path: Optional[str] = None):
        if self.path_to_directory is None:
            self.path_to_directory = export_path
        if not self.path_to_directory is None:
            path_to_directory = Path(self.path_to_directory)
            if operator.not_(path_to_directory.exists()):
                os.makedirs(path_to_directory)
            if operator.not_(Path(path_to_directory/'models').exists()):
                os.mkdir(path_to_directory / 'models')
            run_number = int(len(os.listdir(path_to_directory / 'models')))
            if export_params:
                torch.save(generator.state_dict(),
                           path_to_directory/'models'/f'generator_{run_number}.pt')
            self.model_snapshot(path_to_directory, run_number, show=True)

    def _log_epoch(self, generator_loss, mmd_loss, generator, gradient):
        if self.print_updates:
            print(f"Average loss in the epoch: {generator_loss}, mmd loss: {mmd_loss}, gradient norm: {gradient}")

        self.train_history[self.generator_loss_key].append(generator_loss)
        self.train_history[self.mmd_loss_key].append(mmd_loss)
        self.train_history[self.gradient_key].append(gradient)
        self.generator = generator
