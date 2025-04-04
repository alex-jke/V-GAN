import warnings
from abc import abstractmethod
from typing import Optional

import torch
from numpy import ndarray
from pytorch_lightning.utilities import grad_norm
from torch import Tensor
import pytorch_lightning as pl
from models.Generator import GeneratorSigmoidSTE
from models.Mmd_loss_constrained import MMDLossConstrained, RBF as RBFConstrained, MixtureRQLinear
from VMMDBase import VMMDBase
from models.mse_loss import MSELoss
from modules.od_module import ODModule
from modules.text.vmmd_lightning import VMMDLightningBase


class VMMDTextLightningBase(VMMDLightningBase, ODModule):
    def __init__(self, sequence_length: int, seperator: str = " ", **kwargs):
        # Assume necessary hyperparameters (e.g., lr, weight_decay, weight, epochs) are in kwargs.
        if 'generator' not in kwargs:
            kwargs['generator'] = GeneratorSigmoidSTE
        super().__init__(**kwargs)
        self.sequence_length = sequence_length
        self.seperator = seperator
        # In Lightning, we assume the sequence length is known beforehand.
        self.n_dims = sequence_length
        self._latent_size = max(self.n_dims // 4, 1)
        # Create generator network using base class method.
        self.generator = self.get_the_networks(self.n_dims, self._latent_size)
        # Create loss function.
        kernel = MixtureRQLinear()#RBFConstrained()
        #self.loss_function = MMDLossConstrained(weight=self.hparams.get("weight", 1.0), kernel=kernel)
        self.loss_function = MSELoss(weight=self.hparams.get("weight", 1.0))

    def forward(self, x: Tensor) -> Tensor:
        # Optionally define forward pass (e.g., for inference)
        return self.generator(x)

    def on_before_optimizer_step(self, optimizer):
        # Compute the 2-norm for each layer
        norms = grad_norm(self.generator, norm_type=2)
        avg_norm = 0
        for _, norm in norms.items():
            avg_norm += norm
        if len(norms) == 0:
            norm = Tensor([0])
            warnings.warn("Generator did not return any parameters.")
        else:
            norm = avg_norm / len(norms)
        self.log(self.gradient_key, norm, prog_bar=True)
        self.log_dict(norms)

    def on_after_backward(self):
        for name, param in self.named_parameters():
            self.logger.experiment.add_histogram(f"gradients/{name}", param.grad, self.global_step)
            self.logger.experiment.add_histogram(f"weights/{name}", param, self.global_step)

    def training_step(self, batch, batch_idx):
        # Assume batch is a numpy array of sentences or similar.
        # Prepare noise and subspaces.
        noise_tensor = self._get_noise_tensor(self._latent_size)
        noise_tensor.normal_()  # Sample noise
        subspaces = self.generator(noise_tensor)

        # Convert the batch to embeddings.
        embeddings = self._convert_batch(batch, self.embedding, None)
        masked_embeddings = self._convert_batch(batch, self.embedding, subspaces)

        # Compute loss.
        loss = self.loss_function(embeddings, masked_embeddings, subspaces)
        self.log(self.generator_loss_key, loss, prog_bar=True)
        mmd_loss = self.loss_function.mmd_loss
        self.log(self.mmd_loss_key, mmd_loss, prog_bar=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.generator.parameters(),
            lr=self.hparams.get("lr", 1e-3),
            weight_decay=self.hparams.get("weight_decay", 0)
        )
        return optimizer

    # The following methods remain abstract or utility; implement as needed:
    @abstractmethod
    def _convert_batch(self, batch, embedding, mask) -> Tensor:
        """
        Convert a batch to a tensor.
        Must be implemented.
        """
        raise NotImplementedError

    @abstractmethod
    def get_training_data(self, x_data, embedding, n_dims):
        """
        Prepare training data.
        Must be implemented.
        """
        raise NotImplementedError

    @abstractmethod
    def check_if_myopic(self, x_data: Optional[ndarray], count=500, bandwidth: float | Tensor = 0.01):
        """
        Custom evaluation method.
        Must be implemented.
        """
        raise NotImplementedError