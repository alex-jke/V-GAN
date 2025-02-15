import torch 
from torch import nn 

class Encoder(nn.Module): #Not gonna try convolutions yet nor transformers. But should keep them in mind.
    def __init__(self, latent_size, img_size):
        super(Encoder, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(img_size, 8*latent_size),
            nn.LeakyReLU(0.2),
            nn.Linear(8*latent_size, 4*latent_size),
            nn.LeakyReLU(0.2),
            nn.Linear(4*latent_size, 2*latent_size),
            nn.LeakyReLU(0.2),
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
            nn.LeakyReLU(0.2),
            nn.Linear(2*latent_size, 4*latent_size),
            nn.LeakyReLU(0.2),
            nn.Linear(4*latent_size, 8*latent_size),
            nn.LeakyReLU(0.2),
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