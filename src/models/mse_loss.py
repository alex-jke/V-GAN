import torch
from torch import nn

from models.Mmd_loss_constrained import MMDLossConstrained


class MSELoss(nn.Module):
    def __init__(self, weight=1.0):
        """
        Initializes the MSELoss module.
        :param weight: The weight for the loss function.
        """
        super(MSELoss, self).__init__()
        self.weight = weight
        self.mmd_loss = None

    def forward(self, X, Y, U):
        """
        Computes the MSE loss between two tensors X and Y.
        :param X: The first tensor.
        :param Y: The second tensor.
        :return: The MSE loss between X and Y.
        """
        # Summing to increase the numerical stability, as mean led to 0.
        mse = (X - Y).pow(2).mean(0).sum()
        self.mmd_loss = mse
        diversity = MMDLossConstrained.diversity_loss(U) * self.weight * 10
        regularization = U.mean() * self.weight
        return mse + diversity + regularization