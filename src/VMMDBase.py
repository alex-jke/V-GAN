import inspect

import torch
from collections import defaultdict

from matplotlib import pyplot
from torch.utils.data import DataLoader

from colors import VGAN_GREEN, COMPLIMENTARY
from models.Generator import Generator, Generator_big

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import os
import operator

class VMMDBase:
    '''
    V-MMD, a Subspace-Generative Moment Matching Network.

    Class for the method VMMD, the application of a GMMN to the problem of Subspace Generation. As a GMMN, no
    kernel learning is performed. The default values for the kernel are
    '''

    def __init__(self, batch_size=500, epochs=500, lr=10e-5, momentum=0.99, seed=777, weight_decay=10e-5, path_to_directory=None,
                 weight=0, generator = None, print_updates=None, gradient_clipping=False):
        self.storage = locals()
        self.train_history = defaultdict(list)
        self.generator_loss_key = "generator_loss"
        self.mmd_loss_key = "mmd_loss"
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
        train_history = self.train_history
        plt.style.use('ggplot')
        generator_y = train_history[self.generator_loss_key]
        mmd_y = train_history[self.mmd_loss_key]
        x = np.linspace(1, len(generator_y), len(generator_y))
        fig, ax = plt.subplots()
        ax.plot(x, generator_y, color=VGAN_GREEN,
                label="Generator loss", linewidth=2)
        ax.plot(x, mmd_y, color=COMPLIMENTARY,
                label="MMD loss", linewidth=2)

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        ax.legend(loc="upper right")

        return plt, ax

    def _plot_loss(self, path_to_directory, show=False):
        plot, _ = self._create_plot()
        plot.savefig(path_to_directory / "train_history.png",
                    format="pdf", dpi=1200)

        if show == True:
            print("The show option has been depricated due to lack of utility")

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
        self.generator = self.get_the_networks(ndims, latent_size=max(int(ndims/16), 1)).to(device)
        self.generator.load_state_dict(torch.load(path_to_generator, map_location=device))
        self.generator.eval()  # This only works for dropout layers
        self.generator_optimizer = f'Loaded Model from {path_to_generator} with {ndims} dimensions in the latent space'
        self._latent_size = max(int(ndims / 16), 1)

    def get_the_networks(self, ndims: int, latent_size: int, device: str = None) -> Generator_big:
        """Object function to obtain the networks' architecture

        Args:
            ndims (int): Number of dimensions of the full space
            latent_size (int): Number of dimensions of the latent space
            device (str, optional): CUDA device to mount the networks to. Defaults to None.

        Returns:
            generator: A generator model (child class from torch.nn.Module)
        """
        if device == None:
            device = self.device

        # Check if only the constructor or a whole generator was passed.
        self._latent_size = latent_size
        if inspect.isclass(self.provided_generator):
            generator = self.provided_generator(
                img_size=ndims, latent_size=latent_size).to(device)
        else:
            generator = self.provided_generator

        return generator

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

    def _export(self, generator):
        if not self.path_to_directory is None:
            path_to_directory = Path(self.path_to_directory)
            if operator.not_(path_to_directory.exists()):
                os.makedirs(path_to_directory)
            if operator.not_(Path(path_to_directory/'models').exists()):
                os.mkdir(path_to_directory / 'models')
            run_number = int(len(os.listdir(path_to_directory/'models')))
            torch.save(generator.state_dict(),
                       path_to_directory/'models'/f'generator_{run_number}.pt')
            self.model_snapshot(path_to_directory, run_number, show=True)

    def _log_epoch(self, generator_loss, mmd_loss, generator):
        if self.print_updates:
            print(f"Average loss in the epoch: {generator_loss}, mmd loss: {mmd_loss}")

        self.train_history[self.generator_loss_key].append(generator_loss)
        self.train_history[self.mmd_loss_key].append(mmd_loss)
        self.generator = generator

