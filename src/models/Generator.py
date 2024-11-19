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
        super(Generator_big, self).__init__()
        self.main = nn.Sequential(
            nn.Sequential(
                nn.Linear(latent_size, 2 * latent_size),
                nn.ReLU(),
                #nn.LeakyReLU(0.2),
                nn.BatchNorm1d(2 * latent_size),
                nn.Linear(2 * latent_size, 4 * latent_size),
                nn.ReLU(),
                #nn.LeakyReLU(0.2),
                nn.BatchNorm1d(4 * latent_size),
                nn.Linear(4 * latent_size, 8 * latent_size),
                nn.ReLU(),
                #nn.LeakyReLU(0.2),
                nn.BatchNorm1d(8 * latent_size),
                nn.Linear(8 * latent_size, img_size),
                upper_softmax()
            )
        )

    def forward(self, input):
        return self.main(input)
