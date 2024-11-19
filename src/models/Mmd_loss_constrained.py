import torch
from torch import nn


BATCH_DIM = 0
SHAPE_LEN_WITH_BATCH_DIM = 3

class RBF(nn.Module):

    def __init__(self, n_kernels=5, mul_factor=2.0, bandwidth=None, embedding = lambda x: x):
        super().__init__()
        device = torch.device('cuda:0' if torch.cuda.is_available(
        ) else 'mps:0' if torch.backends.mps.is_available() else 'cpu')

        self.bandwidth_multipliers = mul_factor ** (
            torch.arange(n_kernels) - n_kernels // 2).to(device)
        self.bandwidth = bandwidth
        self.embedding = embedding

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

        L2_distances = torch.cdist(X_embedded, X_embedded)
        L2_distances = L2_distances
        if len(L2_distances.shape) == SHAPE_LEN_WITH_BATCH_DIM:
            # Computes the mean over the batch dimension (dim: ExBxN -> PxN), this being a norm for matrices of ExN.
            L2_distances = L2_distances.mean(dim=BATCH_DIM)

        L2_squared = L2_distances ** 2

        result = self._compute_rbf_kernel(L2_squared)

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

    def __init__(self, weight, kernel=RBF()):
        super().__init__()
        self.kernel = kernel
        self.weight = weight
        self.device = torch.device('cuda:0' if torch.cuda.is_available(
        ) else 'mps:0' if torch.backends.mps.is_available() else 'cpu')

    def forward(self, X, Y, U):
        K = self.kernel(torch.vstack([X, Y]))
        self.bandwidth = self.kernel.bandwidth
        self.bandwidth_multipliers = self.kernel.bandwidth_multipliers
        X_size = X.shape[0]
        XX = K[:X_size, :X_size].mean()
        XY = K[:X_size, X_size:].mean()
        YY = K[X_size:, X_size:].mean()
        return XX - 2 * XY + YY + self.weight*(torch.mean(torch.ones(U.shape[1]).to(self.device) - torch.topk(U, 1, 0).values))
