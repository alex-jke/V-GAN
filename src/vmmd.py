import inspect
from random import random

import torch
from collections import defaultdict

from matplotlib import pyplot

from VMMDBase import VMMDBase
from colors import VGAN_GREEN, COMPLIMENTARY
from models.Generator import Generator, Generator_big

import torch_two_sample as tts
from models.Mmd_loss_constrained import MMDLossConstrained
from models.Mmd_loss_constrained import RBF as RBFConstrained
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import os
import operator
import datetime


class VMMD(VMMDBase):
    '''
    V-MMD, a Subspace-Generative Moment Matching Network.

    Class for the method VMMD, the application of a GMMN to the problem of Subspace Generation. As a GMMN, no
    kernel learning is performed. The default values for the kernel are
    '''


    def fit(self, X: np.array):
        embedding = lambda x: x
        for _ in self.yield_fit(X, embedding):
            continue

    def yield_fit(self, X: np.array, embedding, yield_epochs: int = None):
        '''
        Fits the model to the data. The model is trained using the MMD loss function. The model is trained using the Adadelta optimizer.
        @param X: A two-dimensional numpy array with the data to be fitted.
        The data should be in the form: n_samples x n_features
        @param embedding: A function that transforms the data. By default, it is the identity function.
        This function is used inside the RBF kernel before calculating the distance.
        @param yield_epochs: The number of epochs between each yield. If None, the model will not yield.
        This is useful for monitoring the training process.
        '''
        cuda = torch.cuda.is_available()
        mps = torch.backends.mps.is_available()
        self._set_seed()

        # MODEL INTIALIZATION#
        epochs = self.epochs
        self._latent_size = latent_size = max(int(X.shape[1] / 16), 1)
        ndims = X.shape[1]
        train_size = X.shape[0]
        self.batch_size = min(self.batch_size, train_size)

        device = self.device
        generator = self.get_the_networks(
            ndims, latent_size, device=device)
        #optimizer = torch.optim.Adadelta(generator.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        optimizer = torch.optim.Adam(generator.parameters(), lr=self.lr, weight_decay=self.weight_decay, betas=(0.5,0.9))
        self.generator_optimizer = optimizer.__class__.__name__
        # loss_function =  tts.MMDStatistic(self.batch_size, self.batch_size)
        kernel = RBFConstrained(embedding=embedding)
        loss_function = MMDLossConstrained(weight=self.weight, kernel=kernel)

        for epoch in range(epochs):
            if self.print_updates:
                print(f'\rEpoch {epoch} of {epochs}')
            generator_loss = 0
            mmd_loss = 0
            gradient = 0

            # DATA LOADER#
            data_loader = self._get_data_loader(X)
            batch_number = data_loader.__len__()

            # GET NOISE TENSORS#
            noise_tensor = self._get_noise_tensor(latent_size)

            # BATCH LOOP#
            for batch in tqdm(data_loader, leave=False):
                # Make sure there is only 1 observation per row.
                batch = batch.view(self.batch_size, -1)
                if cuda:
                    batch = batch.cuda()
                elif mps:
                    #batch = batch.to(torch.float32).to(torch.device('mps'))  # float64 not suported with mps
                    batch.to('mps') # For tokeniz
                # SAMPLE NOISE#
                noise_tensor.normal_()

                # OPTIMIZATION STEP#
                optimizer.zero_grad()
                fake_subspaces = generator(noise_tensor)
                masked_batch = fake_subspaces * batch
                batch_loss = loss_function(masked_batch, batch, fake_subspaces)
                batch_mmd_loss = loss_function.mmd_loss
                self.bandwidth = loss_function.bandwidth
                batch_loss.backward()

                # Apply gradient clipping (e.g., max norm of 1.0)
                if self.apply_gradient_clipping:
                    torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)

                gradient += torch.Tensor([param.grad.norm() for param in generator.parameters()]).mean() / batch_number

                optimizer.step()

                generator_loss += float(batch_loss.to(
                    'cpu').detach().numpy())/batch_number
                mmd_loss += float(batch_mmd_loss.to(
                    'cpu').detach().numpy())/batch_number
                #print("finished batch")

            self._log_epoch(generator_loss, mmd_loss, generator, gradient)
            if yield_epochs is not None and epoch % yield_epochs == 0:
                yield epoch

        self.generator = generator
        self._export(generator)

def model_eval(model, X_data) -> pd.DataFrame:
    device = torch.device('cuda:0' if torch.cuda.is_available(
        ) else 'mps:0' if torch.backends.mps.is_available() else 'cpu')
    sample_amount = min(500, X_data.shape[0])
    X_sample = torch.Tensor(pd.DataFrame(
        X_data).sample(sample_amount).to_numpy()).to(device)
    u = model.generate_subspaces(sample_amount)
    u = u.float()
    uX_data = u * \
              torch.mps.Tensor(X_sample).to(model.device) + \
              torch.mean(X_sample, dim=0) * (1-u)
    # round each value in u to one decimal
    X2_data = torch.Tensor(pd.DataFrame(
        X_data).sample(sample_amount).to_numpy()).to(device)

    #X_data = X_data.to(device)
    uX_data = uX_data.to(device)
    mmd = tts.MMDStatistic(sample_amount, sample_amount)
    mmd_val, distances = mmd(uX_data, X2_data, alphas=[0.01], ret_matrix=True)
    mmd_prop = tts.MMDStatistic(sample_amount, sample_amount)
    mmd_prop_val, distances_prop = mmd_prop(
        X_sample, uX_data, alphas=[1 / model.bandwidth], ret_matrix=True)
    PYDEVD_WARN_EVALUATION_TIMEOUT = 200
    print(f'pval of the MMD two sample test {mmd.pval(distances)}')
    print(
        f'pval of the MMD two sample test with proposed bandwidth {1 / model.bandwidth} is {mmd_prop.pval(distances_prop)}, with MMD {mmd_prop_val}')

    u = torch.round(u * 10) / 10
    unique_subspaces, proba = np.unique(
        np.array(u.to('cpu')), axis=0, return_counts=True)
    proba = proba / np.array(u.to('cpu')).shape[0]
    unique_subspaces = [str(unique_subspaces[i] * 1)
                        for i in range(unique_subspaces.shape[0])]

    subspace_df = pd.DataFrame({'subspace': unique_subspaces, 'probability': proba})
    print(subspace_df)
    print(np.sum(proba))
    return subspace_df

if __name__ == "__main__":
    # mean = [1,1,0,0,0,0,0,0,2,1]
    # cov = [[1,1,0,0,0,0,0,0,0,0],[1,1,0,0,0,0,0,0,0,0],[0,0,1,1,1,0,0,0,0,0],[0,0,1,1,1,0,0,0,0,0],[0,0,1,1,1,0,0,0,0,0],[0,0,0,0,0,1,0,0,0,0],[0,0,0,0,0,0,1,0,0,0],
    #       [0,0,0,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,0,0,1]]
    mean = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    cov = [[1, 0, 0, 0, 0, 0, 0, 0, 500, 500], [0, 1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 1, 0, 0], [500, 0, 0, 0, 0, 0, 0, 0, 1, 500], [500, 0, 0, 0, 0, 0, 0, 0, 500, 1]]
    X_data = np.random.multivariate_normal(mean, cov, 2000)

    model = VMMD(epochs=1500, path_to_directory=Path(os.getcwd()).parent / "experiments" /
                 f"Example_normal_{datetime.datetime.now()}_vmmd", lr=0.01)
    for epoch in model.fit(X_data):
        #print(epoch)
        continue

    model_eval(model, X_data)

