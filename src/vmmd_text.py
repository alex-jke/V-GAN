from typing import Callable, Tuple, Iterable

import numpy as np
import torch
from numpy import ndarray
from torch import Tensor
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
    def __init__(self, sequence_length: int | None = None, **kwargs):
        """
        Initializes the VMMD_Text model.
        :param sequence_length: The length of the sequences. If None, the average length of the sequences in the data will be used.
        :param kwargs: Additional keyword arguments.
        """
        if 'generator' not in kwargs:
            kwargs['generator'] = GeneratorSigmoidSTE
        super().__init__(**kwargs)
        self.sequence_length = sequence_length

    def fit(self, x_data: ndarray[str],
            embedding: Callable[[ndarray[str], int], Tensor]):
        """
        Trains the model.
        :param x_data: The data to train the model on. The data should be a one-dimensional numpy array, where each element is a sentence as a string.
            This is due, to sentences having different lengths.
        :param embedding: The embedding function to use. It is expected to be able to take in two parameters, the sentence as a string, and the subspace masks
            as a two-dimensional Tensor. The function should return the embeddings of the sentences with and without the masks applied.
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

        x_data = embedding(x_data_str, n_dims).to(self.device)

        self._latent_size = latent_size = max(n_dims // 16, 1)
        samples = x_data.shape[0]
        self.batch_size = min(self.batch_size, samples)

        generator = self.get_the_networks(
            n_dims, latent_size, device=self.device)
        optimizer = torch.optim.Adam(generator.parameters(), lr=self.lr, weight_decay=self.weight_decay,
                                     betas=(0.5, 0.9))
        self.generator_optimizer = optimizer.__class__.__name__
        kernel = RBFConstrained()
        loss_function = MMDLossConstrained(weight=self.weight, kernel=kernel)

        #TODO: check if this is correct, having the data loader outside the loop.
        data_loader = self._get_data_loader(x_data)

        for epoch in range(self.epochs):
            if self.print_updates:
                print(f'\rEpoch {epoch} of {self.epochs}')
            generator_loss = 0
            mmd_loss = 0

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

                optimizer.step()
                generator_loss += float(batch_loss.to(
                    'cpu').detach().numpy()) / batch_number
                mmd_loss += float(batch_mmd_loss.to(
                    'cpu').detach().numpy()) / batch_number

            self._log_epoch(generator_loss, mmd_loss, generator)
            if yield_epochs is not None and epoch % yield_epochs == 0:
                yield epoch

        self.generator = generator
        self._export(generator)

    def _get_average_sequence_length(self, x_data: ndarray) -> int:
        """
        Returns the average length of the sequences in the data.
        """
        return int(np.mean([len(x) for x in x_data]))

