import warnings
from typing import Callable, Tuple, Iterable

import numpy as np
import pandas as pd
import torch_two_sample as tts
import torch
from matplotlib import pyplot as plt
from numpy import ndarray
from torch import Tensor
from torch.nn.functional import normalize
from tqdm import tqdm

from VMMDBase import VMMDBase
from models.Generator import GeneratorSigmoidSTE
from models.Mmd_loss_constrained import MMDLossConstrained
from models.Mmd_loss_constrained import RBF as RBFConstrained
from text.Embedding.fast_text import FastText
from text.dataset.ag_news import AGNews
from text.dataset.nlp_adbench import NLP_ADBench
from text.dataset_converter.dataset_preparer import DatasetPreparer


class VMMD_Text(VMMDBase):
    def __init__(self, pre_embed: bool = True ,sequence_length: int | None = None, seperator: str = " ", **kwargs):
        """
        Initializes the VMMD_Text model.
        :param sequence_length: The length of the sequences. If None, the average length of the sequences in the data will be used.
        :param seperator: The separator between the words in the sequences.
        param kwargs: Additional keyword arguments.
        """
        if 'generator' not in kwargs:
            kwargs['generator'] = GeneratorSigmoidSTE
        super().__init__(**kwargs)
        self.sequence_length = sequence_length
        self.seperator = ' '
        self.x_data: Tensor | None = None
        self.pre_embed = pre_embed
        #self.emb_fun = FastText(normalize=True).embed_sentences

    def fit(self, x_data: ndarray[str],
            embedding: Callable[[ndarray[str], int], Tensor]):
        """
        Trains the model.
        :param x_data: The data to train the model on. The data should be a one-dimensional numpy array, where each element is a sentence as a string.
            This is due, to sentences having different lengths.
        :param embedding: The embedding function to use. It is expected to be able to take in two parameters, the sentences as a numpy array of strings,
        and the length to pad to or trim to. The function should return the embeddings of the sentences, as a three-dimensional tensor of shape (n_sentences, n_words, n_dims).
        """
        for _ in self.yield_fit(x_data, embedding):
            pass

    def yield_fit(self, x_data_str: ndarray[str],
                  embedding: Callable[[ndarray[str], int], Tensor],
                  yield_epochs=None) -> Iterable[int]:
        """
        Trains the model.
        :param x_data_str: The data to train the model on. The data should be a one-dimensional numpy array, where each element is a sentence.
            This is due, to sentences having different lengths.
        :param embedding: The embedding function to use. It is expected to be able to take in two parameters, the sentences as a numpy array of strings,
        and the length to pad to or trim to. The function should return the embeddings of the sentences, as a three-dimensional tensor of shape (n_sentences, n_words, n_dims).
        :param yield_epochs: The number of epochs between each print.
        """
        self._set_seed()
        n_dims = self.sequence_length if self.sequence_length is not None else self._get_average_sequence_length(x_data_str)

        if self.pre_embed:
            self.x_data = embedding(x_data_str, n_dims)
            x_data = self.x_data.to(self.device)


        self._latent_size = latent_size = max(n_dims // 16, 1)
        samples = x_data_str.shape[0]
        self.batch_size = min(self.batch_size, samples)

        generator = self.get_the_networks(
            n_dims, latent_size, device=self.device)
        optimizer = torch.optim.Adam(generator.parameters(), lr=self.lr, weight_decay=self.weight_decay#, betas=(0.5, 0.9)
                                     )
        self.generator_optimizer = optimizer.__class__.__name__
        kernel = RBFConstrained()
        loss_function = MMDLossConstrained(weight=self.weight, kernel=kernel)

        #TODO: check if this is correct, having the data loader outside the loop. -> should be changed
        data_loader = self._get_data_loader(x_data)

        for epoch in range(self.epochs):
            if self.print_updates:
                print(f'\rEpoch {epoch} of {self.epochs}')
            generator_loss = 0
            mmd_loss = 0
            gradient = 0

            #data_loader = self._get_data_loader(x_data)
            batch_number = data_loader.__len__()

            noise_tensor = self._get_noise_tensor(latent_size)

            for batch in tqdm(data_loader, leave=False):
                noise_tensor.normal_()

                #OPTIMIZATION STEP#
                #batch = batch.to(self.device)
                optimizer.zero_grad()
                subspaces = generator(noise_tensor)

                embeddings = batch.mean(1)
                # Calculate the masked embeddings by multiplying the batch of shape (batch_size, sequence_length, n_dims) with the subspaces of shape (batch_size, sequence_length)
                masked_embeddings = (batch * subspaces.unsqueeze(2)).mean(1)

                masked_embeddings = masked_embeddings.to(self.device)
                embeddings = embeddings.to(self.device)
                batch_loss = loss_function(embeddings, masked_embeddings, subspaces)
                batch_mmd_loss = loss_function.mmd_loss
                self.bandwidth = loss_function.bandwidth
                batch_loss.backward()

                if self.apply_gradient_clipping:
                    grad_list = [param.grad.norm() for param in generator.parameters()]
                    grads = Tensor(grad_list)
                    trimmed = grads.greater_equal(torch.ones_like(grads)).int().sum()
                    if trimmed > 0:
                        print(f'trimmed {trimmed} gradients')
                    torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)

                gradient += Tensor([param.grad.norm() for param in generator.parameters()]).mean() / batch_number

                optimizer.step()
                generator_loss += float(batch_loss.to(
                    'cpu').detach().numpy()) / batch_number
                mmd_loss += float(batch_mmd_loss.to(
                    'cpu').detach().numpy()) / batch_number

            self._log_epoch(generator_loss, mmd_loss, generator, gradient)
            if yield_epochs is not None and epoch % yield_epochs == 0:
                yield epoch

        self.generator = generator
        self._export(generator)

    def check_if_myopic(self, count= 500, bandwidth: float | ndarray = 0.01):
        x_data: Tensor = self.x_data
        if count > x_data.shape[0]:
            warnings.warn(f"Selected 'count': {count} is greater than the number of samples {x_data.shape[0]} in the dataset. Setting count to {x_data.shape[0]}. This might lead to unexpected results.")
            count = x_data.shape[0]
        results = []

        #x_data = normalize(x_data, axis=0)
        indices = torch.randperm(x_data.size(0))[:count]
        #x_sample = torch.mps.Tensor(pd.DataFrame(x_data.mean(1)).sample(count).to_numpy()).to(self.device)
        x_sample = x_data[indices].mean(1).to(self.device)

        indices = torch.randperm(x_data.size(0))[:count]
        ux_x_sample = x_data[indices].to(self.device)
        u_subspaces = self.generate_subspaces(count).to(self.device)

        #ux_sample = u_subspaces * torch.mps.Tensor(x_sample).to(self.device) + torch.mean(x_sample, dim=0) * (1-u_subspaces)
        ux_sample = (ux_x_sample * u_subspaces.unsqueeze(2)).mean(1) + (ux_x_sample.mean(0) * (1 - u_subspaces.unsqueeze(2))).mean(1)
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

    def _plot_loss(self, path_to_directory, show=False):
        plot, ax = self._create_plot()
        p_values = self.check_if_myopic(count=1000)
        recomended_p_value = p_values["recommended bandwidth"].values[0]
        recommended_bandwidth = self.bandwidth.item()

        # add the p-value to the plot in the top right corner
        plt.text(0.5, 0.99, f'{"recommended bandwidth"}\n({recommended_bandwidth}): {recomended_p_value}',
                 ha='center', va='top',
                 transform=ax.transAxes, color='black', fontsize=8)

        plot.savefig(path_to_directory / "train_history.png",
                     format="png", dpi=1200) #todo: change back to pdf

    def _get_average_sequence_length(self, x_data: ndarray) -> int:
        """
        Returns the average length of the sequences in the data.
        """
        sequence_length = int(np.mean([len(x.split(self.seperator)) for x in x_data]))
        return sequence_length

