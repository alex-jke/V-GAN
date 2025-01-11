from cmath import log

import torch
from torch import nn


# Regular function definition does not appear to work properly within a Sequential definition of a network in Pytorchs
class upper_softmax(nn.Module):
    def __init__(self):
        super().__init__()  # Dummy intialization as there is no parameter to learn

    def forward(self, x):
        x = torch.nn.functional.softmax(x, 1)
        x = torch.less(x, 1/x.shape[1])*x + \
            torch.greater_equal(x, 1/x.shape[1])
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
    def __init__(self, latent_size, img_size):
        rel_size = int(img_size/latent_size)
        layers = 4
        increase_per_layer = int(log(rel_size, layers).real)
        super(Generator_big, self).__init__()

        entry_size = 1
        increase = increase_per_layer

        layer_1 = nn.Sequential(
            nn.Linear(entry_size * latent_size, increase * latent_size),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(increase * latent_size)
        )

        increase *= increase_per_layer
        entry_size *= increase_per_layer

        layer_2 = nn.Sequential(
            nn.Linear(entry_size * latent_size, increase * latent_size),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(increase * latent_size),
        )

        increase *= increase_per_layer
        entry_size *= increase_per_layer
        layer_3 = nn.Sequential(
            nn.Linear(entry_size * latent_size, increase * latent_size),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(increase * latent_size),
        )

        increase *= increase_per_layer
        entry_size *= increase_per_layer
        layer_4 = nn.Sequential(
            nn.Linear(entry_size * latent_size, img_size),
            upper_softmax(),
        )

        gen1 = nn.Sequential(
            layer_1,
            layer_2,
            layer_3,
            layer_4,
        )


        gen2 = nn.Sequential(
            #nn.Linear(latent_size, 2 * latent_size),
            nn.Linear(latent_size, latent_size * increase_per_layer),
            #nn.ReLU(),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(increase_per_layer * latent_size),
            nn.Linear(increase_per_layer * latent_size, 4 * latent_size),
            #nn.ReLU(),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(4 * latent_size),
            nn.Linear(4 * latent_size, 8 * latent_size),
            #nn.ReLU(),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(8 * latent_size),
            nn.Linear(8 * latent_size, img_size),
            upper_softmax() #todo: check is not diferentiable if all points 1/n. Should not be problem since not all points can be selected
            #nn.Sigmoid()
        )
        self.main = gen1

    def forward(self, input):
        return self.main(input)
