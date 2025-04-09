import random
from cmath import log
from typing import List, Tuple, Optional

import torch
from torch import nn, Tensor


# Regular function definition does not appear to work properly within a Sequential definition of a network in Pytorchs
class upper_softmax(nn.Module):
    def __init__(self):
        super().__init__()  # Dummy intialization as there is no parameter to learn

    # This function applies a softmax to the input tensor and then sets all values to one that are larger than 1/n.
    def forward(self, input):
        x = torch.nn.functional.softmax(input, 1)
        x_less = torch.less(x, 1/x.shape[1]) * x
        x_ge = torch.greater_equal(x, 1/x.shape[1])
        x_upper_softmax = x_less + x_ge
        #x = x_upper_softmax + (x - x.detach())
        #x = x_ge.detach() + (x_less - x_less.detach())
        #return x
        return x_upper_softmax


class upper_lower_softmax(nn.Module):
    def __init__(self):
        super().__init__()  # Dummy intialization as there is no parameter to learn

    def forward(self, x):
        x = torch.nn.functional.softmax(x, 1)
        selected = torch.greater_equal(x, 1/x.shape[1])
        x = x*selected + (~selected)*1e-08
        return x

class DyT(nn.Module):
    """
    A dynamic hyperbolic tangent function that can learn the scaling factor.
    See paper: https://arxiv.org/pdf/2503.10622
    Name: Transformers without Normalization
    """
    def __init__(self, size: int, init_alpha=0.5):
        super().__init__()
        self.tanh = nn.Tanh()
        self.alpha = nn.Parameter(torch.tensor(init_alpha))
        self.gamma = nn.Parameter(torch.ones(size))
        self.beta = nn.Parameter(torch.zeros(size))

    def forward(self, x):
        """
        Calculates the output of the dynamic hyperbolic tangent function.
        Uses the formula:  γ * tanh(α * x) + β
        :param x: The input tensor.
        :return: The output tensor.
        """
        tanh =  self.tanh(self.alpha * x)
        return self.gamma * tanh + self.beta


class Generator(nn.Module):
    def __init__(self, img_size, latent_size):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(img_size, img_size * 2),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(img_size * 2),
            nn.Linear(img_size * 2, img_size),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(img_size),
            nn.Linear(img_size, img_size // 2),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(img_size // 2),
            nn.Linear(img_size // 2, img_size),
            upper_softmax()
        )

    def forward(self, input):
        return self.main(input)

class ValueExtractor(nn.Module):
    def __init__(self):
        super(ValueExtractor, self).__init__()
        self.value: Optional[Tensor] = None

    def forward(self, input):
        self.value = input.detach().mean(dim=0)
        return input


class Generator_big(nn.Module):
    def __init__(self, latent_size, img_size, activation_function: nn.Module=upper_softmax()):
        rel_size = int(img_size/latent_size)
        self.latent_size = latent_size
        self.img_size = img_size
        amount_layers = 4
        self.increase = log(rel_size, amount_layers).real
        super(Generator_big, self).__init__()
        self.final_activation_function = activation_function
        self.avg_mask = ValueExtractor()

        layers = [self.get_layer(layer) for layer in range(1, amount_layers)]
        layers += [self.get_layer(amount_layers, last=True)]
        self.main = nn.Sequential(*layers)

    def get_layer(self, layer_num: int, last=False):
        input_size = max(round(pow(self.increase, layer_num - 1) * self.latent_size), 1)
        output_size = max(round(pow(self.increase, layer_num) * self.latent_size), 1)

        layer = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.LeakyReLU(0.2),
            #nn.Sigmoid(),
            nn.BatchNorm1d(output_size)
            #DyT(output_size)
        )
        last_layer = nn.Sequential(
            nn.Linear(input_size, self.img_size),
            self.avg_mask,
            self.final_activation_function
        )
        return last_layer if last else layer

    def forward(self, input):
        return self.main(input)

class GeneratorUpperSoftmax(Generator_big):
    def __init__(self, latent_size, img_size):
        super().__init__(latent_size, img_size, upper_softmax())


class BinaryStraightThrough(nn.Module):
    def __init__(self, threshold=0.5):
        super().__init__()
        self.threshold = threshold

    def forward(self, x):
        # Forward: Threshold to 0/1
        x_binary = (x > self.threshold).int()
        # Backward: Pass gradients through threshold
        x_binary = x_binary + (x - x.detach())
        return x_binary

class AnnealingSigmoid(nn.Module):
    def __init__(self, alpha: float = 1.0, beta: float = 0.5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, x):
        # Sigmoid function with annealing
        sigmoid = torch.sigmoid(self.alpha * x)
        # Update alpha and beta for annealing
        self.alpha += self.beta
        return sigmoid

class GeneratorSigmoid(Generator_big):
    def __init__(self, latent_size, img_size):
        super().__init__(latent_size, img_size, nn.Sigmoid())

class GeneratorSigmoidAnnealing(Generator_big):
    def __init__(self, latent_size, img_size):
        super().__init__(latent_size, img_size, activation_function=AnnealingSigmoid())

class GeneratorSigmoidSTE(GeneratorSigmoid):
    def __init__(self, latent_size, img_size):
        super().__init__(latent_size, img_size)
        self.binarize = BinaryStraightThrough()

    def forward(self, input):
        x = super().forward(input)
        return self.binarize(x)

class GeneratorSpectralSigmoidSTE(GeneratorSigmoidSTE):
    def get_layer(self, layer_num: int, last=False):
        input_size = max(round(pow(self.increase, layer_num - 1) * self.latent_size), 1)
        output_size = max(round(pow(self.increase, layer_num) * self.latent_size), 1)

        layer = nn.Sequential(
            nn.utils.spectral_norm(
                nn.Linear(input_size, output_size)
            ),
            DyT(output_size),
            #nn.BatchNorm1d(output_size),
            nn.LeakyReLU(0.2),
        )
        last_layer = nn.Sequential(
            nn.Linear(input_size, self.img_size),
            self.avg_mask,
            self.final_activation_function
        )
        return last_layer if last else layer

class SigmoidSoftmax(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Sigmoid(),
            nn.Softmax(dim=1)
        )
    def forward(self, input):
        return self.main(input)
class PowSigmoid(nn.Module):
    def __init__(self):
        super().__init__()
        self.exp = 1

    def forward(self, input):
        x = torch.nn.functional.sigmoid(input)
        return x ** self.exp

class GeneratorSoftmax(Generator_big):
    def __init__(self, latent_size, img_size):
        super().__init__(latent_size, img_size, nn.Softmax(dim=1))

class GeneratorSoftmaxSTE(GeneratorSoftmax):
    def __init__(self, latent_size, img_size):
        super().__init__(latent_size, img_size)
        self.binarize = BinaryStraightThrough(threshold=1/img_size)

    def forward(self, input):
        x = super().forward(input)
        return self.binarize(x)

class GeneratorSigmoidSoftmaxSTE(Generator_big):
    def __init__(self, latent_size, img_size):
        super().__init__(latent_size, img_size, activation_function=SigmoidSoftmax())
        self.binarize = BinaryStraightThrough(threshold=1 / img_size)

    def forward(self, input):
        x = super().forward(input)
        return self.binarize(x)

class GeneratorSigmoidSoftmaxSigmoid(Generator_big):
    def __init__(self, latent_size, img_size):
        self.exp = 1
        super().__init__(latent_size, img_size, activation_function = SigmoidSoftmax())

    def forward(self, input):
        x = super().forward(input) * 100
        sig = nn.functional.sigmoid(x - 100/self.img_size)
        powed = sig ** self.exp
        self.exp += 0.01
        return powed

# -----------------------------
# Mini-Batch Discrimination Module
# -----------------------------

class MiniBatchDiscrimination(nn.Module):
    """
    Implements mini-batch discrimination as described in:
    Salimans et al., "Improved Techniques for Training GANs" (2016).

    Given an input feature vector x of shape (batch_size, in_features),
    this module learns a tensor T (of shape (in_features, out_features, kernel_dim))
    and computes for each sample an additional feature vector that measures
    similarity to other samples in the mini-batch.
    """
    def __init__(self, in_features, out_features, kernel_dim):
        """
        Args:
            in_features (int): Number of input features.
            out_features (int): Number of discrimination features to produce.
            kernel_dim (int): Dimensionality of the kernels.
        """
        super(MiniBatchDiscrimination, self).__init__()
        self.out_features = out_features
        self.kernel_dim = kernel_dim
        # Initialize T with a random normal distribution.
        self.T = nn.Parameter(torch.randn(in_features, out_features, kernel_dim))

    def forward(self, x: Tensor) -> Tensor:
        """
        Computes for each sample an additional feature vector that measures similarity to other samples in the mini-batch.
        :param x: Tensor of shape (batch_size, in_features).
        :return: Tensor of shape (batch_size, in_features + out_features).
        """
        # x: (batch_size, in_features)
        batch_size = x.size(0)
        # Multiply input with T to get M of shape (batch_size, out_features, kernel_dim)
        M = x.matmul(self.T.view(x.size(1), -1))
        M = M.view(batch_size, self.out_features, self.kernel_dim)
        # Compute the L1 distance between each pair of examples along kernel_dim.
        # We expand M so that we can compute pairwise differences.
        M_i = M.unsqueeze(0)   # Shape: (1, batch_size, out_features, kernel_dim)
        M_j = M.unsqueeze(1)   # Shape: (batch_size, 1, out_features, kernel_dim)
        # Compute L1 distances and sum over the kernel dimension.
        abs_diff = torch.abs(M_i - M_j).sum(3)  # Shape: (batch_size, batch_size, out_features)
        # Apply a negative exponential to obtain similarity measures.
        c = torch.exp(-abs_diff)
        # For each sample, sum the similarities to all other samples (exclude self-similarity by subtracting 1).
        mbd_features = c.sum(1) - 1  # Shape: (batch_size, out_features)
        # Concatenate the original features with the mini-batch discrimination features.
        return torch.cat([x, mbd_features], dim=1)


# -----------------------------
# New Generator with Mini-Batch Discrimination
# -----------------------------

class GeneratorSTEMBD(nn.Module):
    def __init__(self, generator: Generator_big, mbd_out_features, mbd_kernel_dim, binarize: BinaryStraightThrough | None = None):
        """
            Args:
                mbd_out_features (int): Number of output features for mini-batch discrimination.
                mbd_kernel_dim (int): Dimensionality of the kernels for mini-batch discrimination.
                generator (Generator_big): Generator model.
                binarize (BinaryStraightThrough | None): Binarization model.
            """
        super(GeneratorSTEMBD, self).__init__()
        self.binarize = binarize

        # In our network (built in Generator_big), the final layer is the last element in self.main.
        # It is defined as: nn.Sequential(nn.Linear(in_features, img_size), final_activation_function)
        # We extract the input size of that linear layer.
        final_layer = generator.main[-1]
        in_features = final_layer[0].in_features

        # Create the mini-batch discrimination module.
        self.mbd = MiniBatchDiscrimination(in_features, mbd_out_features, mbd_kernel_dim)

        # Create a new final layer that accepts the concatenated features.
        self.new_final_layer = nn.Sequential(
            nn.Linear(in_features + mbd_out_features, generator.img_size),
            generator.final_activation_function
        )

        # Build a feature extractor by removing the original final layer from self.main.
        self.feature_extractor = nn.Sequential(*list(generator.main.children())[:-1])

    def forward(self, input):
        # Pass input through the feature extractor to get intermediate features.
        features = self.feature_extractor(input)
        # Apply mini-batch discrimination: this concatenates extra features to the original ones.
        features_with_mbd = self.mbd(features)
        # Pass the augmented features through the new final layer.
        out = self.new_final_layer(features_with_mbd)
        # Finally, apply the straight-through binarization.
        if self.binarize is not None:
            return self.binarize(out)
        return out

class GeneratorSigmoidSTEMBD(GeneratorSigmoid):
    """
    A subclass of GeneratorSTEBD that integrates a mini-batch discrimination
    module right before the final output layer. This aims to help reduce mode collapse
    by letting the generator “sense” the diversity in the mini-batch.
    """
    def __init__(self, latent_size, img_size, mbd_out_features=16, mbd_kernel_dim=8):
        super().__init__(latent_size, img_size)
        self.batch_discrimination = GeneratorSTEMBD(generator=self, mbd_out_features=mbd_out_features, mbd_kernel_dim=mbd_kernel_dim,
                         binarize=BinaryStraightThrough(threshold=0.5))
    def forward(self, input):
        return self.batch_discrimination(input)


class GeneratorSoftmaxSTEMBD(GeneratorSoftmax):

    def __init__(self, latent_size, img_size):
        super().__init__(latent_size, img_size)
        binarize = BinaryStraightThrough(threshold=1.0/img_size)
        self.batch_discrimination = GeneratorSTEMBD(generator=self, mbd_out_features=1, mbd_kernel_dim=1,
                         binarize=binarize)

    def forward(self, input):
        return self.batch_discrimination(input)


class FakeGenerator(nn.Module):
    def __init__(self, subspaces: List[Tuple[List[int], float]]):
        super(FakeGenerator, self).__init__()
        self.subspaces = []
        self.device = torch.device('cuda:0' if torch.cuda.is_available()
                                   else 'mps:0' if torch.backends.mps.is_available() else 'cpu')

        for subspace, proba in subspaces:
            for _ in range(int(proba * 100)):
                self.subspaces.append(torch.tensor(subspace, dtype=torch.float32, requires_grad=True))

        #shuffle the subspaces
        random.shuffle(self.subspaces)
        self.main = nn.Sequential(
            nn.Linear(1, 1)
        )

    def forward(self, input):
        batch_size = input.shape[0]
        # Use torch.stack to create a tensor with gradient tracking
        subspaces = torch.stack([self.subspaces[i % len(self.subspaces)] for i in range(batch_size)]).to(self.device)
        # shuffle the subspaces
        subspaces = subspaces[torch.randperm(subspaces.size()[0])]
        return subspaces


class GeneratorSoftmaxSTESpectralNorm(GeneratorSoftmaxSTE):

    def get_layer(self, layer_num: int, last=False):
        input_size = max(round(pow(self.increase, layer_num - 1) * self.latent_size), 1)
        output_size = max(round(pow(self.increase, layer_num) * self.latent_size), 1)

        layer = nn.Sequential(
            nn.utils.spectral_norm(
                nn.Linear(input_size, output_size)
            ),
            #nn.Sigmoid(),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(output_size)
        )
        last_layer = nn.Sequential(
            nn.Linear(input_size, self.img_size),
            self.avg_mask,
            self.final_activation_function
        )
        return last_layer if last else layer

