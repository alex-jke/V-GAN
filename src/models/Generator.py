import random
from cmath import log
from typing import List, Tuple

import torch
from torch import nn


# Regular function definition does not appear to work properly within a Sequential definition of a network in Pytorchs
class upper_softmax(nn.Module):
    def __init__(self):
        super().__init__()  # Dummy intialization as there is no parameter to learn

    # This function applies a softmax to the input tensor and then sets all values to one that are larger than 1/n.
    def forward(self, x):
        x = torch.nn.functional.softmax(x, 1)
        x_less = torch.less(x, 1/x.shape[1])*x
        x_ge = torch.greater_equal(x, 1/x.shape[1])
        x = x_less + x_ge
        return x


class upper_lower_softmax(nn.Module):
    def __init__(self):
        super().__init__()  # Dummy intialization as there is no parameter to learn

    def forward(self, x):
        x = torch.nn.functional.softmax(x, 1)
        selected = torch.greater_equal(x, 1/x.shape[1])
        x = x*selected + (~selected)*1e-08
        return x


class Generator(nn.Module):
    def __init__(self, latent_size):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(latent_size, latent_size * 2),
            nn.ReLU(),
            nn.BatchNorm1d(latent_size * 2),
            nn.Linear(latent_size * 2, latent_size),
            nn.ReLU(),
            nn.BatchNorm1d(latent_size),
            nn.Linear(latent_size, latent_size // 2),
            nn.ReLU(),
            nn.BatchNorm1d(latent_size // 2),
            nn.Linear(latent_size // 2, latent_size),
            upper_softmax()
        )

    def forward(self, input):
        return self.main(input)


class Generator_big(nn.Module):
    def __init__(self, latent_size, img_size, activation_function: nn.Module=upper_softmax()):
        rel_size = int(img_size/latent_size)
        self.latent_size = latent_size
        self.img_size = img_size
        amount_layers = 4
        self.increase = log(rel_size, amount_layers).real
        super(Generator_big, self).__init__()
        self.final_activation_function = activation_function

        layers = [self.get_layer(layer) for layer in range(1, amount_layers)]
        layers += [self.get_layer(amount_layers, last=True)]
        self.main = nn.Sequential(*layers)

    def get_layer(self, layer_num: int, last=False):
        input_size = max(round(pow(self.increase, layer_num - 1) * self.latent_size), 1)
        output_size = max(round(pow(self.increase, layer_num) * self.latent_size), 1)

        layer = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(output_size)
        )
        last_layer = nn.Sequential(
            nn.Linear(input_size, self.img_size),
            #upper_softmax(),
            self.final_activation_function
            #nn.Sigmoid()
        )
        #todo: check that the layers are created correctly.
        return last_layer if last else layer

    def forward(self, input):
        return self.main(input)

class GeneratorUpperSoftmax(Generator_big):
    def __init__(self, latent_size, img_size):
        super().__init__(latent_size, img_size, upper_softmax())

class GeneratorSigmoid(Generator_big):
    def __init__(self, latent_size, img_size):
        super().__init__(latent_size, img_size, nn.Sigmoid())

import torch
import torch.nn as nn
from typing import List, Tuple

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


class BinaryStraightThrough(nn.Module):
    def __init__(self, threshold=0.5):
        super().__init__()
        self.threshold = threshold

    def forward(self, x):
        # Forward: Threshold to 0/1
        x_binary = (x > self.threshold).int()
        # Backward: Pass gradients through sigmoid
        x_binary = x_binary + (x - x.detach())
        return x_binary


class GeneratorSigmoidSTE(GeneratorSigmoid):
    def __init__(self, latent_size, img_size):
        super().__init__(latent_size, img_size)
        self.binarize = BinaryStraightThrough()

    def forward(self, input):
        x = super().forward(input)
        return self.binarize(x)

