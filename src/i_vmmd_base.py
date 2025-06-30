import inspect
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, List
import torch_two_sample as tts


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from colors import VGAN_GREEN
from models.Generator import Generator_big
from models.Mmd_loss_constrained import MMDLossConstrained


class IVMMDBase(ABC):
    """
    Base class for V-MMD and its variants, providing common functionality for training and visualization.
    This class is abstract and should not be instantiated directly.
    It does not assume any specific underlying model or training framework. As such, it can be used with various
    generator models and training setups, including PyTorch Lightning or custom training loops in PyTorch.
    It requires subclasses to implement the `_create_plot` method for generating specific plots.
    Attributes that need to be defined in subclasses include:
        provided_generator (Generator_big): The generator model to be used.
        path_to_directory (str): Directory path where results will be saved.
        train_history (dict): Dictionary containing training history data, maps from a string to a list.
        gradient_key (str): Key for accessing gradient data in `train_history`.
        device (str): Device to run the model on, e.g., 'cuda' or 'cpu'.
    """

    def _plot_gradients(self):
        """
        Plots the gradient norms from the training history and saves the plot as an image.
        This method uses the `gradient_key` to access the gradient norms from `train_history`.
        The plot is styled using 'ggplot' and saved in the specified directory of `path_to_directory.
        """
        gradients = self.train_history[self.gradient_key]
        plt.style.use('ggplot')
        x = np.linspace(1, len(gradients), len(gradients))
        fig, ax = plt.subplots()
        ax.plot(x, gradients, color=VGAN_GREEN,
                label="Gradient norm", linewidth=2)
        ax.legend(loc='best')
        plt.xlabel("Epoch")
        plt.ylabel(self.gradient_key)
        plt.savefig(Path(self.path_to_directory) / "gradients.png")
        plt.close()

    @abstractmethod
    def _create_plot(self):
        """
        Abstract method to create a plot for the training history.
        This method should be implemented in subclasses to define how the training history is visualized.
        """
        pass

    def _plot_loss(self, path_to_directory, x_data: Optional[np.ndarray[str]] = None):
        """
        Plots the training history and saves it as an image in the directory specified by `path_to_directory`.
        """
        plot, _ = self._create_plot()
        plot.savefig(Path(path_to_directory) / "train_history.png", format="png", dpi=1200)
        plot.close()

    def get_the_networks(self, ndims: int, latent_size: int, device: str = None) -> Generator_big:
        """Gets the generator model based on the provided dimensions and latent size.
        This method checks if the provided generator is a class or an instance. If it is a class, it initializes
        a new instance with the given dimensions and latent size. If it is an instance, it returns the instance as is.
        This method is used to create the generator model that will be trained in the VMMD framework.
        It is expected that the provided generator is a child class of `torch.nn.Module` and can be used for training.
        This method is typically called during the initialization of the VMMD model to set up the generator network.
        It ensures that the generator is compatible with the specified number of dimensions and latent size,
        and it mounts the generator to the specified device if provided.

        Args:
            ndims (int): Number of dimensions of the full space
            latent_size (int): Number of dimensions of the latent space.
            device (str, optional): Device to mount the networks to. Defaults to None.

        Returns:
            generator: A generator model (child class from torch.nn.Module)
        """
        if device is None:
            device = self.device

        # Check if only the constructor or a whole generator was passed.
        self._latent_size = latent_size
        if inspect.isclass(self.provided_generator):
            generator = self.provided_generator(
                img_size=ndims, latent_size=latent_size).to(device)
        else:
            generator = self.provided_generator

        return generator

    def _check_if_myopic(self, x_sample, ux_sample, u_subspaces, bandwidth: float | List[float] = 0.01, count=500):
        """
        Checks if the provided samples are myopic by calculating the MMD statistic.
        If the bandwidth is not provided, it uses the bandwidth from the MMDLossConstrained instance.
        The method returns a DataFrame with the p-values for the given bandwidths.
        Args:
            x_sample (Tensor): Sample data to be compared.
            ux_sample (Tensor): projected sample data.
            u_subspaces (Tensor): Subspace representations of the projections used to obtain ux_sample.
            bandwidth (float | List[float], optional): Bandwidth for the MMD calculation. Defaults to 0.01.
            count (int, optional): Number of samples to use for the MMD calculation. Defaults to 500. Should
            represent the number of samples in x_sample and ux_sample.
        """

        x_sample = x_sample.to(ux_sample.device)
        results = []

        if type(bandwidth) == float:
            bandwidth = [bandwidth]

        if not hasattr(self, 'bandwidth'):
            mmd_loss = MMDLossConstrained(self.weight)
            mmd_loss.forward(
                x_sample, ux_sample, u_subspaces * 1)
            self.bandwidth = mmd_loss.bandwidth

        bandwidth.sort()
        for bw in bandwidth:
            mmd = tts.MMDStatistic(count, count)

            _, distances = mmd(x_sample, ux_sample, alphas=[
                bw], ret_matrix=True)
            results.append(mmd.pval(distances))

        bw = self.bandwidth.item()
        mmd = tts.MMDStatistic(count, count)
        _, distances = mmd(x_sample, ux_sample, alphas=[
            bw], ret_matrix=True)
        results.append(mmd.pval(distances))

        bandwidth.append("recommended bandwidth")
        return pd.DataFrame([results], columns=bandwidth, index=["p-val"])