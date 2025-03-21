import math
from abc import ABC, abstractmethod

import torch
from torch import nn, Tensor

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
        n_samples = L2_distances.shape[0]
        self.bandwidth = L2_distances.data.sum() / (n_samples ** 2 - n_samples)
        return self.bandwidth

    """def get_bandwidth(self, L2_distances):
        # Flatten and compute median of non-zero distances
        flat_dist = L2_distances.flatten()
        non_zero = flat_dist[flat_dist > 0]
        self.bandwidth = torch.median(non_zero)
        return self.bandwidth"""

    def forward(self, X: torch.Tensor, Y: torch.Tensor):
        '''
        X: torch.Tensor
            The input tensor of shape (n_samples, feature_dim * 2) (X and Y are concatenated)
            Alternatively, a shape of (embedding_dim, n_samples, feature_dim)
        '''
        X_embedded = self.embedding(X)
        Y_embedded = self.embedding(Y)

        embedded = torch.vstack([X_embedded, Y_embedded])
        if len(embedded.shape) == SHAPE_LEN_VECTOR:
            norm = self.vector_norm.compute_distance_matrix
        elif len(embedded.shape) == SHAPE_LEN_MATRIX:
            norm = self.matrix_norm.compute_distance_matrix
        else:
            raise ValueError("Input Tensor has to either be of shape (BxN) or (ExBxN)")

        distances = norm(embedded)

        squared_distances = distances ** 2

        if torch.isnan(squared_distances).any():
            print("L2 distances are nan.")

        result = self._compute_rbf_kernel(squared_distances)

        if torch.isnan(result).any():
            print("result are nan.")
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

        if torch.isnan(result_reduced).any():
            print("kernel produced nan value")

        # Result should be of shape (2*batch_size, 2*batch_size)
        return result_reduced


class MMDLossConstrained(nn.Module):
    '''
    Constrained loss by the number of features selected
    '''

    def __init__(self, weight, kernel=RBF(), subspace_amount_penalty = 3, middle_penalty = None):
        super().__init__()
        self.kernel = kernel
        self.weight = weight
        self.device = torch.device('cuda:0' if torch.cuda.is_available(
        ) else 'mps:0' if torch.backends.mps.is_available() else 'cpu')
        self.subspace_penalty = subspace_amount_penalty
        self.middle_penalty = middle_penalty
        if self.middle_penalty is None:
            self.middle_penalty = 0.0

    def diversity_loss(self, M, tau=0.1):
        # M: shape (n, d) with entries in [0,1]
        diff = M.unsqueeze(1) - M.unsqueeze(0)  # (n, n, d)
        dist = torch.sum(diff ** 2, dim=-1)  # (n, n) squared distances
        sim = torch.exp(-dist / tau)  # similarity: high if rows are similar
        # Zero out self-similarity
        sim = sim - torch.diag_embed(sim.diagonal(0))
        return sim.mean()  # Lower loss => higher diversity

    def get_loss(self, X, Y, U, apply_penalty = True):
        #K = self.kernel(torch.vstack([X, Y]))
        K = self.kernel(X, Y)
        self.bandwidth = self.kernel.bandwidth
        self.bandwidth_multipliers = self.kernel.bandwidth_multipliers
        X_size = X.shape[0]
        XX = K[:X_size, :X_size].mean()
        XY = K[:X_size, X_size:].mean()
        YY = K[X_size:, X_size:].mean()

        if torch.isnan(XX).any() or torch.isnan(XY).any() or torch.isnan(YY).any():
            raise ValueError("XX or XY contain nan values.")
            XX = torch.where(torch.isnan(XX), torch.zeros_like(XX), XX)
            XY = torch.where(torch.isnan(XY), torch.zeros_like(XY), XY)
            YY = torch.where(torch.isnan(YY), torch.zeros_like(YY), YY)

        # ones = torch.ones(U.shape[1]).to(self.device)
        # topk = torch.topk(U, 10, 0).values.float().mean(dim=0)
        mean = U.float().mean(dim=0)
        avg = mean.sum() / U.shape[1]
        #zero = torch.zeros_like(mean)
        #feature_selection_penalty = (torch.less_equal(mean,zero)* 1.0).mean() if apply_penalty else 0
        # avg_u = U.float().mean(dim=0)
        # mean = torch.mean(ones - topk)
        # penalty = self.weight * (mean)
        #penalty = self.weight * (avg) if apply_penalty else 0 # self.weight*(mean)
        penalty = 0#self.weight * torch.exp(avg) if apply_penalty else 0  # self.weight*(mean)
        diversity_loss = self.diversity_loss(U.float()) * self.weight if apply_penalty else 0
        #u_sizes = U.float().sum(dim=1)
        #median = u_sizes.median()
        #penalty = self.weight * (median) if  apply_penalty else 0

        # middle penalty to punish the generator for generating subspaces with prob close to 0.5
        #middle_matrix = (-U.float() * (U.float() - 1))
        #middle_penalty = middle_matrix.mean(dim=0).sum() / U.shape[1] if apply_penalty else 0
        #middle_penalty *= self.middle_penalty

        mmd_loss = XX - 2 * XY + YY
        if math.isnan(mmd_loss):
            raise ValueError("mmd is nan.")
        return (mmd_loss +
                penalty +
                diversity_loss# + middle_penalty# + feature_selection_penalty
                , mmd_loss)

    def forward(self, X, Y, U: torch.Tensor, apply_penalty = True):
        full_loss, self.mmd_loss = self.get_loss(X, Y, U, apply_penalty)
        return full_loss

if __name__ == '__main__':
    mmd = MMDLossConstrained(1)
    test_tensor = Tensor([[1.,1.], [0., 1.], [1., 0.]])
    div_loss = mmd.diversity_loss(test_tensor)
    print(div_loss)
