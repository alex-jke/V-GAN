from abc import abstractmethod
from typing import Callable, Iterable, Optional, List

import numpy as np
import torch
from matplotlib import pyplot as plt
from numpy import ndarray
from torch import Tensor
from tqdm import tqdm

from VMMDBase import VMMDBase
from models.Generator import GeneratorSigmoidSTE
from models.Mmd_loss_constrained import MMDLossConstrained
from models.Mmd_loss_constrained import RBF as RBFConstrained


class VMMDTextBase(VMMDBase):
    def __init__(self, sequence_length: int | None = None, seperator: str = " ", **kwargs):
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
        self.embedding: Optional[Callable[[ndarray[str], int], Tensor]] = None
        self.n_dims: Optional[int] = None
        self.original_data: Optional[ndarray[str]] = None

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

    @abstractmethod
    def _get_training_data(self, x_data: ndarray[str], embedding: Callable[[ndarray[str], int], Tensor], n_dims: int) -> Tensor | ndarray[str]:
        """
        Prepares the training data for the VMMD_Text model. Whatever is returned here is used in the _convert_batch function.
        In the end, a one-dimensional tensor needs to be returned.
        :param x_data: The data to prepare.
        :param embedding: The embedding function, that can be used to embed the sentences.
        :n_dims: The number of dimensions the vectors should have. That is usually the number of words.
        """
        pass

    @abstractmethod
    def _convert_batch(self, batch: ndarray[str] | Tensor, embedding: Callable[[ndarray[str], int], Tensor], mask: Optional[Tensor]) -> Tensor:
        """
        This method converts a batch into a one-dimensional tensor of shape (n_dims).
        :param batch: The batch to convert.
        :param embedding: An embedding function that can be used to convert sentences into embeddings.
        :param mask: An optional mask tensor that is applied to words in the batch.
        :return: A one-dimensional tensor of shape (n_dims).
        """
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
        self.n_dims = n_dims = self.sequence_length if self.sequence_length is not None else self._get_average_sequence_length(x_data_str)
        self.embedding = embedding

        self.original_data = x_data_str
        x_data = self._get_training_data(x_data_str, embedding, n_dims)

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

        for epoch in range(self.epochs):
            data_loader = self._get_data_loader(x_data)
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

                #embeddings = batch.mean(1)
                embeddings = self._convert_batch(batch, embedding, None)
                masked_embeddings = self._convert_batch(batch, embedding, subspaces)

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

                gradients = [param.grad.norm() for param in generator.parameters()]
                gradient += Tensor(gradients).mean() / batch_number

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

    @abstractmethod
    def check_if_myopic(self, count=500, bandwidth: float | ndarray = 0.01):
        pass

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
        sequence_length = int(np.mean([len(self._sentence_to_words(x)) for x in x_data]))
        return sequence_length

    def _sentence_to_words(self, sentence: str) -> List[str]:
        """
        Returns a list of words from the sentence.
        """
        return sentence.split(self.seperator)

