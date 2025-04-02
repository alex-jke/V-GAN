import warnings

import deepspeed
import torch
from pytorch_lightning.utilities import grad_norm
from torch import Tensor
import pytorch_lightning as pl
from models.Generator import GeneratorSigmoidSTE
from models.Mmd_loss_constrained import MMDLossConstrained, RBF as RBFConstrained
from VMMDBase import VMMDBase
from modules.text.vmmd_lightning import VMMDLightningBase


class VMMDTextLightningBase(VMMDLightningBase):
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
        kernel = RBFConstrained()
        self.loss_function = MMDLossConstrained(weight=self.hparams.get("weight", 1.0), kernel=kernel)

    def forward(self, x: Tensor) -> Tensor:
        # Optionally define forward pass (e.g., for inference)
        return self.generator(x)

    def on_before_optimizer_step(self, optimizer):
        # Compute the 2-norm for each layer
        # If using mixed precision, the gradients are already unscaled here
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

    def training_step(self, batch, batch_idx):
        # Assume batch is a numpy array of sentences or similar.
        # Prepare noise and subspaces.
        noise_tensor = self._get_noise_tensor(self._latent_size)
        noise_tensor.normal_()  # Sample noise
        subspaces = self.generator(noise_tensor)

        # Convert the batch to embeddings.
        # 'self.embedding' should be set externally.
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
        """optimizer = deepspeed.ops.adam.DeepSpeedCPUAdam(
            self.generator.parameters(),
            lr=self.hparams.get("lr", 1e-3),
            weight_decay=self.hparams.get("weight_decay", 0)
        )
        return optimizer"""

    # The following methods remain abstract or utility; implement as needed:
    def _convert_batch(self, batch, embedding, mask) -> Tensor:
        """
        Convert a batch to a tensor.
        Must be implemented.
        """
        raise NotImplementedError

    def _get_training_data(self, x_data, embedding, n_dims):
        """
        Prepare training data.
        Must be implemented.
        """
        raise NotImplementedError

    def check_if_myopic(self, count=500, bandwidth: float | Tensor = 0.01):
        """
        Custom evaluation method.
        Must be implemented.
        """
        raise NotImplementedError