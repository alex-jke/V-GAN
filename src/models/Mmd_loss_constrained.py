import math
from abc import ABC, abstractmethod

import torch
from torch import nn, Tensor

from models.norm import VectorNorm, FrobeniusNorm, MatrixNorm, L2Norm

SHAPE_LEN_VECTOR = 2
SHAPE_LEN_MATRIX = 3

class RationalQuadratic(nn.Module):
    def __init__(self, alpha=0.5):
        """
        Rational Quadratic Kernel.

        Args:
            alpha (float): Shape parameter. Controls the tail behavior of the kernel.
                           Higher values make the kernel behave more like the RBF kernel.
        """
        super().__init__()
        self.alpha = alpha

    def get_bandwidth(self, L2_distances):
        n_samples = L2_distances.shape[0]
        self.bandwidth = L2_distances.data.sum() / (n_samples ** 2 - n_samples)
        return L2_distances.data.sum() / (n_samples ** 2 - n_samples)

    def forward(self, X):
        """
        Compute the Rational Quadratic Kernel matrix.

        Args:
            X (torch.Tensor): Input tensor of shape (n_samples, n_features).

        Returns:
            torch.Tensor: Kernel matrix of shape (n_samples, n_samples).
        """
        self.bandwidth = self.get_bandwidth(X)
        L2_distances = torch.cdist(X, X) ** 2
        return (1 + L2_distances / (2 * self.alpha)) ** (-self.alpha)


class MixtureRQLinear(nn.Module):

    def get_bandwidth(self, L2_distances):
        n_samples = L2_distances.shape[0]
        self.bandwidth = L2_distances.data.sum() / (n_samples ** 2 - n_samples)
        return L2_distances.data.sum() / (n_samples ** 2 - n_samples)

    def __init__(self, alphas=None, linear_weight=1.0):
        """
        Mixture of Rational Quadratic Kernels with a Linear Kernel.

        Args:
            alphas (list of float): List of alpha values for the RQ kernels.
                                   Each alpha corresponds to a different RQ kernel.
            linear_weight (float): Weight for the linear kernel.
        """
        super().__init__()
        if alphas is None:
            alphas = [0.2, 0.5, 1.0, 2.0, 5]
        self.alphas = alphas
        self.linear_weight = linear_weight

    def forward(self, X: torch.Tensor, Y: torch.Tensor):
        """
        Compute the mixture of RQ kernels and linear kernel.

        Args:
            X (torch.Tensor): Input tensor of shape (n_samples, n_features).
            Y (torch.Tensor): Input tensor of shape (n_samples, n_features).

        Returns:
            torch.Tensor: Combined kernel matrix of shape (n_samples, n_samples).
        """
        stacked = torch.vstack([X, Y])
        L2_distances = torch.cdist(stacked, stacked) ** 2
        self.bandwidth = self.get_bandwidth(L2_distances)

        rq_kernels = []
        for alpha in self.alphas:
            rq_kernel = (1 + L2_distances / (2 * alpha)) ** (-alpha)
            rq_kernels.append(rq_kernel)

        rq_mixture = torch.sum(torch.stack(rq_kernels), dim=0)

        linear_kernel = torch.matmul(stacked, stacked.T)

        combined_kernel = rq_mixture + self.linear_weight * linear_kernel

        return combined_kernel

class EfficientRBF(nn.Module):
    def __init__(self,
                 n_kernels: int = 5,
                 mul_factor: float = 2.0,
                 embedding=lambda x: x):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available(
        ) else 'mps' if torch.backends.mps.is_available() else 'cpu')
        self.n_kernels = n_kernels
        self.embedding = embedding
        # μ_i = mul_factor ** (i - n_kernels//2) for i∈[0..K)
        bw_mult = mul_factor ** (torch.arange(n_kernels) - n_kernels//2)
        self.bandwidth_multipliers = bw_mult
        self.register_buffer("bw_mult", bw_mult.float())

    def forward(self, X: torch.Tensor, Y: torch.Tensor):
        """
        X, Y: (B, D)
        Returns three (B,B) blocks K_xx, K_xy, K_yy that match the original RBF.
        """
        X = self.embedding(X)
        Y = self.embedding(Y).to(X.device)
        B = X.size(0)

        # squared norms
        x_norm = (X*X).sum(1, keepdim=True)   # (B,1)
        y_norm = (Y*Y).sum(1, keepdim=True)   # (B,1)

        # pairwise squared distances
        D_xx = x_norm + x_norm.t() - 2*(X @ X.t())
        D_yy = y_norm + y_norm.t() - 2*(Y @ Y.t())
        D_xy = x_norm + y_norm.t() - 2*(X @ Y.t())

        # build the off-diag bandwidth pool exactly like the original
        mask = ~torch.eye(B, device=X.device, dtype=torch.bool)
        flat_xx = D_xx[mask]            # B^2 – B entries
        flat_yy = D_yy[mask]
        flat_xy = D_xy.reshape(-1)      # B^2 entries
        # include both X->Y and Y->X just once each
        all_offdiag = torch.cat([
            flat_xx,
            flat_yy,
            flat_xy,
            flat_xy  # duplicate for Y->X
        ], dim=0)

        # exact same denominator (4B^2 – 2B)
        base_bw = all_offdiag.sum() / all_offdiag.numel()
        self.bandwidth = base_bw
        if base_bw.device != self.bw_mult.device:
            base_bw = base_bw.to(self.bw_mult.device)
        bws = base_bw * self.bw_mult  # (K,)

        # accumulate sum over kernels K)
        K_xx = torch.zeros_like(D_xx)
        K_xy = torch.zeros_like(D_xy)
        K_yy = torch.zeros_like(D_yy)
        for σ2 in bws:
            inv = 1.0/σ2
            K_xx += torch.exp(-D_xx * inv)
            K_xy += torch.exp(-D_xy * inv)
            K_yy += torch.exp(-D_yy * inv)

        return K_xx, K_xy, K_yy

class RBF(nn.Module):

    def __init__(self, n_kernels=5, mul_factor=2.0, bandwidth=None, embedding = lambda x: x,
                 vector_norm: VectorNorm = L2Norm(),
                 matrix_norm: MatrixNorm = FrobeniusNorm()):
        super().__init__()
        device = torch.device('cuda' if torch.cuda.is_available(
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

    def __init__(self, weight, kernel=EfficientRBF(), subspace_amount_penalty = 3, middle_penalty = None):
        super().__init__()
        self.kernel = kernel
        self.weight = weight
        self.device = torch.device('cuda' if torch.cuda.is_available(
        ) else 'mps:0' if torch.backends.mps.is_available() else 'cpu')
        self.subspace_penalty = subspace_amount_penalty
        self.middle_penalty = middle_penalty
        if self.middle_penalty is None:
            self.middle_penalty = 0.0

    @staticmethod
    def diversity_loss(M, tau=0.1):
        # M: shape (n, d) with entries in [0,1]
        dist = torch.cdist(M, M) ** 2  # (n, n) squared distances
        sim = torch.exp(-dist / tau)  # similarity: high if rows are similar
        # Zero out self-similarity
        sim = sim - torch.diag_embed(sim.diagonal(0))
        return sim.mean()  # Lower loss => higher diversity

    def get_loss(self, X, Y, U, apply_penalty = True):
        #K = self.kernel(torch.vstack([X, Y]))
        #K = self.kernel(X, Y)

        #X_size = X.shape[0]
        #XX = K[:X_size, :X_size].mean()
        #XY = K[:X_size, X_size:].mean()
        #YY = K[X_size:, X_size:].mean()

        U = U.to(X.device)


        #kernel_eff = EfficientRBF()
        K_xx, K_xy, K_yy = self.kernel(X, Y)
        self.bandwidth = self.kernel.bandwidth
        self.bandwidth_multipliers = self.kernel.bandwidth_multipliers
        XX= K_xx.mean()
        XY = K_xy.mean()
        YY = K_yy.mean()

        #assert torch.allclose(XX, XX_eff) and torch.allclose(XY, XY_eff) and torch.allclose(YY, YY_eff)

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
        penalty = 0
        diversity_loss = 0
        if apply_penalty or self.weight != 0.0:
            penalty = self.weight * avg #torch.exp(avg)  # self.weight*(mean)
            #diversity_loss = self.diversity_loss(U.float())
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
                penalty
                 + diversity_loss# + middle_penalty# + feature_selection_penalty
                , mmd_loss)

    def forward(self, X, Y, U: torch.Tensor, apply_penalty = True):
        full_loss, self.mmd_loss = self.get_loss(X, Y, U, apply_penalty)
        return full_loss

if __name__ == '__main__':
    mmd = MMDLossConstrained(1)
    test_tensor = Tensor([[1.,1.], [0., 1.], [1., 0.]])
    div_loss = mmd.diversity_loss(test_tensor)
    print(div_loss)
