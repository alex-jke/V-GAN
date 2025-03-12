from typing import Callable, Optional
import warnings
import pandas as pd

from numpy import ndarray
from torch import Tensor
import torch

from models.Mmd_loss_constrained import MMDLossConstrained
from modules.text.vmmd_text_base import VMMDTextBase


class VMMDTextPreEmbed(VMMDTextBase):
    """
    An implementation of VMMDTextBase that pre-embeds the data.
    """

    def _get_training_data(self, x_data: ndarray[str], embedding: Callable[[ndarray[str], int, Optional[Tensor]], Tensor], n_dims: int) -> Tensor | ndarray[str]:
        self.x_data = embedding(x_data, n_dims, None)
        return self.x_data.to(self.device)

    def _convert_batch(self, batch: ndarray[str] | Tensor, embedding: Callable[[ndarray[str], int, Optional[Tensor]], Tensor], mask: Optional[Tensor]) -> Tensor:
        """
        Converts a batch to an embedding tensor.
        """
        if mask is None:
            return batch.mean(dim=1)
        # Calculate the masked embeddings by multiplying the batch of shape (batch_size, sequence_length, n_dims) with the subspaces of shape (batch_size, sequence_length)
        return (batch * mask.unsqueeze(2)).mean(1)

    def check_if_myopic(self, count= 500, bandwidth: float | ndarray = 0.01):
        x_data: Tensor = self.x_data
        #training_data = self._get_training_data(self.original_data, self.embedding, self.n_dims)
        #x_data = self._convert_batch(training_data, self.embedding, self.n_dims)
        if count > x_data.shape[0]:
            warnings.warn(f"Selected 'count': {count} is greater than the number of samples {x_data.shape[0]} in the dataset. Setting count to {x_data.shape[0]}. This might lead to unexpected results.")
            count = x_data.shape[0]

        #x_data = normalize(x_data, axis=0)
        indices = torch.randperm(x_data.size(0))[:count]
        #x_sample = torch.mps.Tensor(pd.DataFrame(x_data.mean(1)).sample(count).to_numpy()).to(self.device)
        x_sample = x_data[indices].mean(1).to(self.device)

        indices = torch.randperm(x_data.size(0))[:count]
        ux_x_sample = x_data[indices].to(self.device)
        u_subspaces = self.generate_subspaces(count).to(self.device)

        #ux_sample = u_subspaces * torch.mps.Tensor(x_sample).to(self.device) + torch.mean(x_sample, dim=0) * (1-u_subspaces)
        ux_sample = (ux_x_sample * u_subspaces.unsqueeze(2)).mean(1) + (ux_x_sample.mean(0) * (1 - u_subspaces.unsqueeze(2))).mean(1)

        return self._check_if_myopic(x_sample=x_sample, ux_sample=ux_sample, u_subspaces=u_subspaces, bandwidth=bandwidth)
