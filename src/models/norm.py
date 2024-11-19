from abc import ABC, abstractmethod

import torch


class Norm(ABC):
    """
    Abstract class for computing norms. It computes the distance matrix between the inputs.
    """

    @abstractmethod
    def compute_distance_matrix(self, X: torch.Tensor) -> torch.Tensor:
        """
        Computes the distance matrix between the inputs. The distance matrix is a square matrix where the element
        (i, j) is the distance between the ith and jth input. That is ||X[i] - X[j]||.
        """
        raise NotImplementedError


class VectorNorm(Norm, ABC):
    """
    A subclass of Norm that computes the vector norm of the input.
    """

    @abstractmethod
    def compute_distance_matrix(self, X: torch.Tensor) -> torch.Tensor:
        """
        Computes the distance matrix between the input vectors. The distance matrix is a square matrix where the element
        (i, j) is the distance between the ith and jth Vector. That is ||X[i] - X[j]||.
        """
        raise NotImplementedError


class MatrixNorm(Norm, ABC):
    """
    A subclass of Norm that computes the matrix norm of the input.
    """

    @abstractmethod
    def compute_distance_matrix(self, X: torch.Tensor) -> torch.Tensor:
        """
        Computes the distance matrix between the input matrices. The distance matrix is a square matrix where the element
        (i, j) is the distance between the ith and jth Matrix. That is ||X[i] - X[j]||.
        """
        raise NotImplementedError


class L2Norm(VectorNorm):
    """
    Computes the L2 norm of the input vectors.
    """

    def compute_distance_matrix(self, X: torch.Tensor) -> torch.Tensor:
        """
        Computes the L2 norm of the input vectors. The distance matrix is a square matrix where the element
        (i, j) is the L2 norm between the ith and jth Vector. That is ||X[i] - X[j]||.
        """
        return torch.cdist(X, X)


class FrobeniusNorm(MatrixNorm):
    """
    Computes the Frobenius norm of the input matrices.
    """

    def compute_distance_matrix(self, X: torch.Tensor) -> torch.Tensor:
        """
        Computes the Frobenius norm of the input matrices. The distance matrix is a square matrix where the element
        (i, j) is the Frobenius norm between the ith and jth Matrix. That is ||X[i] - X[j]||.
        """
        # change the tensor from ExBxN to BxExN
        return self.more_efficient(X)
        reordered = X.permute(1, 0, 2)
        difference_matrix = reordered.unsqueeze(0) - reordered.unsqueeze(1)

        norm_matrix = torch.linalg.matrix_norm(difference_matrix, dim=(2, 3))

        return norm_matrix

    def more_efficient(self, X: torch.Tensor) -> torch.Tensor:
        reordered = X.permute(1, 0, 2)
        flattened = torch.flatten(reordered, start_dim=1)

        norm_matrix = torch.cdist(flattened, flattened)

        return norm_matrix



class AverageNorm(MatrixNorm):
    """
    Computes the average norm of the input matrices. The average norm expects the matrix to be in the shape ExBxN, where
    E is the embedding dimension, B the batch dimension, and N the feature dimension.
    """

    def compute_distance_matrix(self, X: torch.Tensor) -> torch.Tensor:
        """
        Computes the average norm of the input matrices. The distance matrix is a square matrix where the element
        (i, j) is the average norm between the ith and jth Matrix. That is ||X[i] - X[j]||m.
        """
        dists = torch.cdist(X, X, p=2)
        return dists.mean(dim=0)

if __name__ == '__main__':
    # Create a 3x2x4 tensor, NxBxE, where N is the feature dimension, B the batch dimension, and E the embedding dimension
    example_matrix = torch.Tensor([
        [[1, 2, 3, 4], [5, 6, 7, 8]],
        [[9, 10, 11, 12], [13, 14, 15, 16]],
        [[17, 18, 19, 20], [21, 22, 23, 24]]])

    # The expected matrix should be of shape BxB, that is 2x2
    frobenius_matrix = FrobeniusNorm().compute_distance_matrix(example_matrix)
    frob_2 = FrobeniusNorm().more_efficient(example_matrix)
    print(frobenius_matrix)
    print(frob_2)