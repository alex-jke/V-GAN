import torch 
from torch import nn 

class Encoder(nn.Module): #Not gonna try convolutions yet nor transformers. But should keep them in mind.
    def __init__(self, latent_size, img_size):
        super(Encoder, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(img_size, 8*latent_size),
            nn.Linear(8*latent_size, 4*latent_size),
            nn.Linear(4*latent_size, 2*latent_size),
            nn.Linear(2*latent_size, latent_size)
        )
    def forward(self, input):
        output = self.main(input)
        return output

class Decoder(nn.Module): #Not gonna try convolutions yet nor transformers. But should keep them in mind.
    def __init__(self, latent_size, img_size):
        super(Decoder, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(latent_size, 2*latent_size),
            nn.Linear(2*latent_size, 4*latent_size),
            nn.Linear(4*latent_size, 8*latent_size),
            nn.Linear(8*latent_size, img_size),
        )

    def forward(self, input):
        output = self.main(input)
        return output

class Detector(nn.Module): #Using an ELM as the embedding worked for LUNAR and Deep IForest
    def __init__(self, latent_size, img_size, encoder, decoder):
        super(Detector, self).__init__()
        self.encoder = encoder(latent_size, img_size)
        self.decoder = decoder(latent_size, img_size)

    def forward(self, input):
        enc_X = self.encoder(input)
        dec_X = self.decoder(enc_X)

        enc_X = enc_X.view(input.size(0), -1)
        dec_X = dec_X.view(input.size(0), -1)
        return enc_X, dec_X

def build_mlp(in_dim, out_dim, num_layers=4, dropout=0.1,
              activation=nn.LeakyReLU(0.2), final_activation=None):
    """
    Dynamically builds a stack of linear layers from `in_dim` to `out_dim`,
    with batch normalization, dropout, and an optional final activation.
    """
    # 1) Determine intermediate dimensions via geometric progression
    #    so we go smoothly from in_dim to out_dim in `num_layers` steps.
    #    E.g. for num_layers=4, we have 5 "stops": [in_dim -> mid1 -> mid2 -> mid3 -> out_dim]
    dims = []
    for i in range(num_layers + 1):
        # ratio from 0.0 up to 1.0
        ratio = i / num_layers
        # geometric interpolation: in_dim * (out_dim/in_dim)^(ratio)
        val = in_dim * ((out_dim / in_dim) ** ratio) if in_dim > 0 else out_dim
        dims.append(int(round(val)))
    # Ensure final is exactly out_dim, first is exactly in_dim
    dims[0] = in_dim
    dims[-1] = out_dim

    layers = []
    for i in range(num_layers):
        layer_in = dims[i]
        layer_out = dims[i+1]
        layers.append(nn.Linear(layer_in, layer_out))

        # If this is NOT the final layer, add BN + Activation + Dropout
        # If it IS the final layer, optionally add only final_activation
        is_last = (i == num_layers - 1)
        if not is_last:
            layers.append(nn.BatchNorm1d(layer_out))
            layers.append(activation)
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
        else:
            # Final layer
            if final_activation is not None:
                layers.append(final_activation)

    return nn.Sequential(*layers)


class EncoderExtra(Encoder):
    def __init__(self, img_size, latent_size, num_layers=4, dropout=0.1):
        super().__init__(img_size, latent_size)
        self.main = build_mlp(
            in_dim=img_size,
            out_dim=latent_size,
            num_layers=num_layers,
            dropout=dropout,
            activation=nn.LeakyReLU(0.2),
            final_activation=None  # or nn.Tanh(), etc., if desired
        )

class DecoderExtra(Decoder):
    def __init__(self, img_size, latent_size, num_layers=4, dropout=0.1):
        super().__init__(img_size, latent_size)
        self.main = build_mlp(
            in_dim=latent_size,
            out_dim=img_size,
            num_layers=num_layers,
            dropout=dropout,
            activation=nn.LeakyReLU(0.2),
            final_activation=None  # e.g. nn.Sigmoid() if data in [0,1]
        )

class DetectorExtra(Detector):
    """
    Wraps the Encoder/Decoder as a single module.
    The forward pass returns (encoded, decoded).
    """
    def __init__(self, img_size, latent_size, num_layers=4, dropout=0.1):
        encoder = lambda lat_size, full_size: EncoderExtra(img_size, latent_size, num_layers, dropout)
        decoder = lambda lat_size, full_size: DecoderExtra(img_size, latent_size, num_layers, dropout)
        super().__init__(img_size, latent_size, encoder, decoder)

