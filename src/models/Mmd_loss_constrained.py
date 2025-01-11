from abc import ABC, abstractmethod

import torch
from torch import nn

from models.norm import VectorNorm, FrobeniusNorm, MatrixNorm, L2Norm

SHAPE_LEN_VECTOR = 2
SHAPE_LEN_MATRIX = 3

class RBF(nn.Module):

    def __init__(self, n_kernels=5, mul_factor=2.0, bandwidth=None, embedding = lambda x: x,
                 vector_norm: VectorNorm = L2Norm(),
                 matrix_norm: MatrixNorm = FrobeniusNorm()):
        super().__init__()
        device = torch.device('cuda:0' if torch.cuda.is_available(
        ) else 'mps:0' if torch.backends.mps.is_available() else 'cpu')

        self.bandwidth_multipliers = mul_factor ** (
            torch.arange(n_kernels) - n_kernels // 2).to(device)
        self.bandwidth = bandwidth
        self.embedding = embedding
        self.vector_norm = vector_norm
        self.matrix_norm = matrix_norm

    def get_bandwidth(self, L2_distances):
        if self.bandwidth is None:
            n_samples = L2_distances.shape[0]
            self.bandwidth = L2_distances.data.sum() / (n_samples ** 2 - n_samples)
            return L2_distances.data.sum() / (n_samples ** 2 - n_samples)

        return self.bandwidth

    def forward(self, X: torch.Tensor):
        '''
        X: torch.Tensor
            The input tensor of shape (n_samples, feature_dim * 2) (X and Y are concatenated)
            Alternatively, a shape of (embedding_dim, n_samples, feature_dim)
        '''
        X_embedded = self.embedding(X)
        if len(X_embedded.shape) == SHAPE_LEN_VECTOR:
            norm = self.vector_norm.compute_distance_matrix
        elif len(X_embedded.shape) == SHAPE_LEN_MATRIX:
            norm = self.matrix_norm.compute_distance_matrix
        else:
            raise ValueError("Input Tensor has to either be of shape (BxN) or (ExBxN)")

        distances = norm(X_embedded)

        squared_distances = distances ** 2

        result = self._compute_rbf_kernel(squared_distances)

        return result

    def _compute_rbf_kernel(self, L2_distances):
        """
        This method also works for L2_distance tensors of three dimensions. Expected, however, is a two-dimensional
        tensor of size (2Bx2B)
        """
        # Get bandwidth value (single scalar) based on the input L2_distances
        base_bandwidth = self.get_bandwidth(L2_distances)  # scalar

        # Multiply by bandwidth multipliers to get multiple bandwidths
        # Shape: (n_kernels,)
        bandwidths = base_bandwidth * self.bandwidth_multipliers

        # Reshape bandwidths for broadcasting
        # Shape: (n_kernels, 1, 1, 1) to match (n_kernels, number_samples, feature_space_dimension, embedding_dimension)
        bandwidths_padded = bandwidths[:, None, None, None]

        # Reshape L2_distances for broadcasting
        # Shape: (1, number_samples, feature_space_dimension, embedding_dimension)
        L2_distances = L2_distances[None, ...]

        # Compute the RBF kernel for each bandwidth
        # Exponential part of RBF kernel: exp(- L2_distances / (2 * bandwidths**2))
        result_full = torch.exp(-L2_distances / (bandwidths_padded))
        result = result_full.sum(dim=0)

        # Sum over the embeddings dimension
        result_reduced = result.mean(dim=0) # check if this still is a norm. Try to turn it into a norm.

        # Result should be of shape (2*batch_size, 2*batch_size)
        return result_reduced


class MMDLossConstrained(nn.Module):
    '''
    Constrained loss by the number of features selected
    '''

    def __init__(self, weight, kernel=RBF(), subspace_amount_penalty = 3):
        super().__init__()
        self.kernel = kernel
        self.weight = weight
        self.device = torch.device('cuda:0' if torch.cuda.is_available(
        ) else 'mps:0' if torch.backends.mps.is_available() else 'cpu')
        self.subspace_penalty = subspace_amount_penalty

    def forward(self, X, Y, U: torch.Tensor):
        K = self.kernel(torch.vstack([X, Y]))
        self.bandwidth = self.kernel.bandwidth
        self.bandwidth_multipliers = self.kernel.bandwidth_multipliers
        X_size = X.shape[0]
        XX = K[:X_size, :X_size].mean()
        XY = K[:X_size, X_size:].mean()
        YY = K[X_size:, X_size:].mean()

        #ones = torch.ones(U.shape[1]).to(self.device)
        #topk = torch.topk(U, 10, 0).values.float().mean(dim=0)
        avg = U.float().mean(dim=0).sum() / U.shape[1]
        #avg_u = U.float().mean(dim=0)
        #mean = torch.mean(ones - topk)
        #penalty = self.weight * (mean)
        penalty = self.weight * (avg) #self.weight*(mean)

        return XX - 2 * XY + YY + penalty

