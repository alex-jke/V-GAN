import warnings
from typing import Callable, Optional

import pandas as pd
import numpy as np
from numpy import ndarray
import torch
from torch import Tensor
import torch.nn.functional as F

from modules.text.vmmd_text_base import VMMDTextBase
from modules.text.vmmd_text_lightning import VMMDTextLightningBase


class VmmdText(VMMDTextBase):
    """
    An implementation of VMMDTextBase that embeds the data each epoch.
    """
    def _get_training_data(self, x_data: ndarray[str], embedding: Callable[[ndarray[str], int], Tensor], n_dims: int) -> Tensor | ndarray[str]:
        self._n_dims = n_dims
        return x_data

    def _embed(self, batch: ndarray[str] | Tensor, embedding: Callable[[ndarray[str], int, Optional[Tensor]], Tensor], masks: Optional[Tensor]) -> Tensor:
        """
        Converts a batch to an embedding tensor. If masks is None, all words are embedded.
        Otherwise, only the words, where the mask is one are embedded.
        :param batch: The batch to convert, as a numpy array of cleaned sentences.
        :param embedding: The embedding function to use.
        :param masks: The masks to use. If None, all words are embedded.
        :return: The embeddings of the batch, as a tensor of shape (n_samples, n_tokens, n_embedding_dim).
        """
        #if masks is None:
        return embedding(batch, self._n_dims, masks)

    def _convert_batch(self, batch: ndarray[str] | Tensor, embedding: Callable[[ndarray[str], int, Optional[Tensor]], Tensor],
                       mask: Optional[Tensor]) -> Tensor:
        embedded = self._embed(batch, embedding, mask)
        meaned = embedded.mean(1)
        normed = F.normalize(meaned, p=2, dim=1)
        return normed

    def check_if_myopic(self, count=500, bandwidth: float | ndarray = 0.01):
        x_data: ndarray[str] = self.original_data
        # training_data = self._get_training_data(self.original_data, self.embedding, self.n_dims)
        # x_data = self._convert_batch(training_data, self.embedding, self.n_dims)
        if count > x_data.shape[0]:
            warnings.warn(
                f"Selected 'count': {count} is greater than the number of samples {x_data.shape[0]} in the dataset. Setting count to {x_data.shape[0]}. This might lead to unexpected results.")
            count = x_data.shape[0]

        # x_data = normalize(x_data, axis=0)
        indices = torch.randperm(x_data.size)[:count]
        x_sample = self._convert_batch(x_data[indices], self.embedding, None).to(self.device)

        indices = torch.randperm(x_data.size)[:count]
        ux_x_sample = x_data[indices]#.to(self.device)
        u_subspaces = self.generate_subspaces(count).to(self.device)

        # ux_sample = u_subspaces * torch.mps.Tensor(x_sample).to(self.device) + torch.mean(x_sample, dim=0) * (1-u_subspaces)
        #ux_sample = (ux_x_sample * u_subspaces.unsqueeze(2)).mean(1) + (ux_x_sample.mean(0) * (1 - u_subspaces.unsqueeze(2))).mean(1)
        ux_subspaces = self._convert_batch(ux_x_sample, self.embedding, u_subspaces)
        ux_1_subspaces = F.normalize(self._embed(ux_x_sample, self.embedding, 1 - u_subspaces))
        ux_1_subspaces_mean0 = ux_1_subspaces.mean(0)
        ux_1_subspaces_mean01 = ux_1_subspaces_mean0.mean(0)
        ux_1_subspaces_mean01_exp = ux_1_subspaces_mean01.expand_as(ux_subspaces)
        #ux_1_subspaces = F.normalize(self._embed(ux_x_sample, self.embedding, 1 - u_subspaces)).mean(0).mean(0).unsqueeze(0).expand_as(ux_subspaces)
        ux_sample = ux_subspaces + ux_1_subspaces_mean01_exp

        return self._check_if_myopic(x_sample=x_sample, ux_sample=ux_sample, u_subspaces=u_subspaces,
                                     bandwidth=bandwidth, count=count)

class VMMDTextLightning(VMMDTextLightningBase):
    """
    An implementation of VMMDTextBase that embeds the data each epoch.
    """
    def get_training_data(self, x_data: ndarray[str], embedding, n_dims) -> Tensor | ndarray[str]:
        return x_data

    def _embed(self, batch: ndarray[str] | Tensor, embedding: Callable[[ndarray[str], int, Optional[Tensor]], Tensor], masks: Optional[Tensor]) -> Tensor:
        """
        Converts a batch to an embedding tensor. If masks is None, all words are embedded.
        Otherwise, only the words, where the mask is one are embedded.
        :param batch: The batch to convert, as a numpy array of cleaned sentences.
        :param embedding: The embedding function to use.
        :param masks: The masks to use. If None, all words are embedded.
        :return: The embeddings of the batch, as a tensor of shape (n_samples, n_tokens, n_embedding_dim).
        """
        #if masks is None:
        return embedding(batch, self.n_dims, masks)

    def _convert_batch(self, batch: ndarray[str] | Tensor, embedding: Callable[[ndarray[str], int, Optional[Tensor]], Tensor],
                       mask: Optional[Tensor]) -> Tensor:
        embedded = self._embed(batch, embedding, mask)
        meaned = embedded.mean(1)
        normed = F.normalize(meaned, p=2, dim=1)
        return normed

    def check_if_myopic(self, count=500, bandwidth: float | ndarray = 0.01):
        x_data: ndarray[str] = self.original_data
        # training_data = self._get_training_data(self.original_data, self.embedding, self.n_dims)
        # x_data = self._convert_batch(training_data, self.embedding, self.n_dims)
        if count > x_data.shape[0]:
            warnings.warn(
                f"Selected 'count': {count} is greater than the number of samples {x_data.shape[0]} in the dataset. Setting count to {x_data.shape[0]}. This might lead to unexpected results.")
            count = x_data.shape[0]

        # x_data = normalize(x_data, axis=0)
        indices = torch.randperm(x_data.size)[:count]
        x_sample = self._convert_batch(x_data[indices], self.embedding, None).to(self.device)

        indices = torch.randperm(x_data.size)[:count]
        ux_x_sample = x_data[indices]#.to(self.device)
        u_subspaces = self.generate_subspaces(count).to(self.device)

        # ux_sample = u_subspaces * torch.mps.Tensor(x_sample).to(self.device) + torch.mean(x_sample, dim=0) * (1-u_subspaces)
        #ux_sample = (ux_x_sample * u_subspaces.unsqueeze(2)).mean(1) + (ux_x_sample.mean(0) * (1 - u_subspaces.unsqueeze(2))).mean(1)
        ux_subspaces = self._convert_batch(ux_x_sample, self.embedding, u_subspaces)
        ux_1_subspaces = F.normalize(self._embed(ux_x_sample, self.embedding, 1 - u_subspaces))
        ux_1_subspaces_mean0 = ux_1_subspaces.mean(0)
        ux_1_subspaces_mean01 = ux_1_subspaces_mean0.mean(0)
        ux_1_subspaces_mean01_exp = ux_1_subspaces_mean01.expand_as(ux_subspaces)
        #ux_1_subspaces = F.normalize(self._embed(ux_x_sample, self.embedding, 1 - u_subspaces)).mean(0).mean(0).unsqueeze(0).expand_as(ux_subspaces)
        ux_sample = ux_subspaces + ux_1_subspaces_mean01_exp

        return self._check_if_myopic(x_sample=x_sample, ux_sample=ux_sample, u_subspaces=u_subspaces,
                                     bandwidth=bandwidth, count=count)