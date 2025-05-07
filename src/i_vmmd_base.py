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
    def _plot_gradients(self):
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
        pass

    def _plot_loss(self, path_to_directory, x_data: Optional[np.ndarray[str]] = None):
        plot, _ = self._create_plot()
        plot.savefig(Path(path_to_directory) / "train_history.png", format="png", dpi=1200)
        plot.close()

    def get_the_networks(self, ndims: int, latent_size: int, device: str = None) -> Generator_big:
        """Object function to obtain the networks' architecture

        Args:
            ndims (int): Number of dimensions of the full space
            latent_size (int): Number of dimensions of the latent space
            device (str, optional): CUDA device to mount the networks to. Defaults to None.

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