from random import random

import torch
from collections import defaultdict
from models.Generator import Generator, Generator_big
import torch_two_sample as tts
from models.Mmd_loss import MMDLoss
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


class VMMD:
    '''
    V-MMD, a Subspace-Generative Moment Matching Network.

    Class for the method VMMD, the application of a GMMN to the problem of Subspace Generation. As a GMMN, no
    kernel learning is performed. The default values for the kernel are
    '''

    def __init__(self, batch_size=500, epochs=30, lr=0.007, momentum=0.99, seed=777, weight_decay=0.04, path_to_directory=None):
        self.storage = locals()
        self.train_history = defaultdict(list)
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.momentum = momentum
        self.seed = seed
        if self.seed == None:
            self.seed = np.random.randint(10, 10000, 1)
        self.weight_decay = weight_decay
        self.path_to_directory = path_to_directory
        self.generator_optimizer = None
        self.device = torch.device('cuda:0' if torch.cuda.is_available(
        ) else 'mps:0' if torch.backends.mps.is_available() else 'cpu')

    def __plot_loss(self, path_to_directory, show=False):
        train_history = self.train_history
        plt.style.use('ggplot')
        generator_y = train_history['generator_loss']
        x = np.linspace(1, len(generator_y), len(generator_y))
        fig, ax = plt.subplots()
        ax.plot(x, generator_y, color="cornflowerblue",
                label="Generator loss", linewidth=2)

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        ax.legend(loc="upper right")
        plt.savefig(path_to_directory / "train_history.pdf",
                    format="pdf", dpi=1200)

        if show == True:
            print("The show option has been depricated due to lack of utility")

    def get_params(self) -> dict:
        return {'batch size': self.batch_size, 'epochs': self.epochs, 'lr_g': self.lr,
                'momentum': self.momentum, 'weight decay': self.weight_decay,
                'batch_size': self.batch_size, 'seed': self.seed,
                'generator optimizer': self.generator_optimizer}

    def model_snapshot(self, path_to_directory=None, run_number=0, show=False):
        ''' Creates an snapshot of the model

        Saves important information regarding the training of the model
        Args:
            - path_to_directory (Path): Specifies the path to directory (relative to the WD)
            - show (bool): Boolean specifying if a pop-up window should open to show the plot for previsualization.
        '''

        if path_to_directory == None:
            path_to_directory = self.path_to_directory
        path_to_directory = Path(path_to_directory)
        if operator.not_(path_to_directory.exists()):
            os.mkdir(path_to_directory)
        if operator.not_((path_to_directory/"train_history").exists()):
            os.mkdir(path_to_directory / "train_history")

        pd.DataFrame(self.train_history["generator_loss"]).to_csv(
            path_to_directory/'train_history'/f'generator_loss_{run_number}.csv', header=False, index=False)
        if os.path.isfile(path_to_directory/'params.csv') != True:
            pd.DataFrame(self.get_params(), [0]).to_csv(
                path_to_directory / 'params.csv')
        else:
            params = pd.read_csv(path_to_directory / 'params.csv', index_col=0)
            params_new = pd.DataFrame(self.get_params(), [run_number])
            params = params.reindex(params.index.union(params_new.index))
            params.update(params_new)
            params.to_csv(
                path_to_directory / 'params.csv')
        self.__plot_loss(path_to_directory, show)

    def load_models(self, path_to_generator, ndims, device: str = None):
        '''Loads models for prediction

        In case that the generator has already been trained, this method allows to load it (and optionally the discriminator) for generating subspaces
        Args:
            - path_to_generator: Path to the generator (has to be stored as a .keras model)
            - path_to_discriminator: Path to the discriminator (has to be stored as a .keras model) (Optional)
        '''
        if device == None:
            device = self.device
        self.generator = Generator_big(
            img_size=ndims, latent_size=max(int(ndims/16), 1)).to(device)
        self.generator.load_state_dict(torch.load(path_to_generator))
        self.generator.eval()  # This only works for dropout layers
        self.generator_optimizer = f'Loaded Model from {path_to_generator} with {ndims} dimensions in the latent space'
        self.__latent_size = max(int(ndims/16), 1)

    def get_the_networks(self, ndims: int, latent_size: int, device: str = None) -> Generator_big:
        """Object function to obtain the networks' architecture

        Args:
            ndims (int): Number of dimensions of the full space
            latent_size (int): Number of dimensions of the latent space
            device (str, optional): CUDA device to mount the networks to. Defaults to None.

        Returns:
            generator: A generator model (child class from torch.nn.Module)
        """
        if device == None:
            device = self.device
        generator = Generator_big(
            img_size=ndims, latent_size=latent_size).to(device)
        return generator

    def fit(self, X: np.array, embedding = lambda x: x):
        '''
        Fits the model to the data. The model is trained using the MMD loss function. The model is trained using the Adadelta optimizer.
        @param X: A two-dimensional numpy array with the data to be fitted.
        The data should be in the form: n_samples x n_features
        '''

        cuda = torch.cuda.is_available()
        mps = torch.backends.mps.is_available()

        torch.manual_seed(self.seed)
        if cuda:
            torch.cuda.manual_seed(self.seed)
        elif mps:
            torch.mps.manual_seed(self.seed)

        # MODEL INTIALIZATION#
        epochs = self.epochs
        self.__latent_size = latent_size = max(int(X.shape[1]/16), 1)
        ndims = X.shape[1]
        train_size = X.shape[0]
        self.batch_size = min(self.batch_size, train_size)

        device = self.device
        generator = self.get_the_networks(
            ndims, latent_size, device=device)
        optimizer = torch.optim.Adadelta(
            generator.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.generator_optimizer = optimizer.__class__.__name__
        # loss_function =  tts.MMDStatistic(self.batch_size, self.batch_size)
        kernel = RBFConstrained(embedding=embedding)
        loss_function = MMDLossConstrained(weight=10, kernel=kernel)

        for epoch in range(epochs):
            print(f'\rEpoch {epoch} of {epochs}')
            generator_loss = 0

            # DATA LOADER#
            if cuda:
                data_loader = DataLoader(
                    X, batch_size=self.batch_size, drop_last=True, pin_memory=not cuda, shuffle=True)
            else:  # Uses CUDA if Available, other wise MPS or nothing
                data_loader = DataLoader(
                    X, batch_size=self.batch_size, drop_last=True, pin_memory=mps, shuffle=True)
            batch_number = data_loader.__len__()

            # GET NOISE TENSORS#
            if cuda:  # Need to open this if statement as the Tensor function has to be called from diferent modules depending of the device
                #noise_tensor = torch.cuda.FloatTensor(self.batch_size, latent_size)#.to(self.device)#to(torch.device('cuda'))
                noise_tensor = torch.FloatTensor(self.batch_size, latent_size).to(self.device)
            elif mps:
                noise_tensor = torch.mps.Tensor(
                    self.batch_size, latent_size).to(torch.device('mps'))
            else:
                noise_tensor = torch.Tensor(self.batch_size, latent_size)

            # BATCH LOOP#
            for batch in tqdm(data_loader, leave=False):
                # Make sure there is only 1 observation per row.
                batch = batch.view(self.batch_size, -1)
                if cuda:
                    batch = batch.cuda()
                elif mps:
                    batch = batch.to(torch.float32).to(
                        torch.device('mps'))  # float64 not suported with mps

                # SAMPLE NOISE#
                noise_tensor.normal_()

                # OPTIMIZATION STEP#
                optimizer.zero_grad()
                fake_subspaces = generator(noise_tensor)
                # batch_loss = loss_function(batch, fake_subspaces*batch + (fake_subspaces == 1e-08)*torch.mean(batch,dim=0), alphas=[0.1]) #Upper_lower_softmax
                # batch_loss = loss_function(batch, fake_subspaces*batch + torch.less(batch,1/batch.shape[1])*torch.mean(batch,dim=0), alphas=[0.1]) #Upper softmax
                batch_loss = loss_function(batch, fake_subspaces*batch + torch.less(
                    batch, 1/batch.shape[1])*torch.mean(batch, dim=0), fake_subspaces)  # Constrained MMD Loss
                self.bandwidth = loss_function.bandwidth
                batch_loss.backward()
                optimizer.step()
                generator_loss += float(batch_loss.to(
                    'cpu').detach().numpy())/batch_number

            print(f"Average loss in the epoch: {generator_loss}")
            self.train_history["generator_loss"].append(generator_loss)

        if not self.path_to_directory == None:
            path_to_directory = Path(self.path_to_directory)
            if operator.not_(path_to_directory.exists()):
                os.mkdir(path_to_directory)
                if operator.not_(Path(path_to_directory/'models').exists()):
                    os.mkdir(path_to_directory / 'models')
            run_number = int(len(os.listdir(path_to_directory/'models')))
            torch.save(generator.state_dict(),
                       path_to_directory/'models'/f'generator_{run_number}.pt')
            self.model_snapshot(path_to_directory, run_number, show=True)

        self.generator = generator

    def generate_subspaces(self, nsubs):
        # Need to load in cpu as mps Tensor module doesn't properly fix the seed
        noise_tensor = torch.Tensor(nsubs, self.__latent_size).to('cpu')
        if not self.seed == None:
            torch.manual_seed(self.seed)
        noise_tensor.normal_()
        u = self.generator(noise_tensor.to(self.device))
        u = torch.greater_equal(u, 1/u.shape[1])
        return u


def model_eval(model, X_data) -> pd.DataFrame:
    device = torch.device('cuda:0' if torch.cuda.is_available(
        ) else 'mps:0' if torch.backends.mps.is_available() else 'cpu')
    X_sample = torch.Tensor(pd.DataFrame(
        X_data).sample(500).to_numpy()).to(device)
    u = model.generate_subspaces(500)
    uX_data = u * \
              torch.mps.Tensor(X_sample).to(model.device) + \
              torch.mean(X_sample, dim=0) * (~u)
    mmd = tts.MMDStatistic(500, 500)
    mmd_val, distances = mmd(X_sample, uX_data, alphas=[0.01], ret_matrix=True)
    mmd_prop = tts.MMDStatistic(500, 500)
    mmd_prop_val, distances_prop = mmd_prop(
        X_sample, uX_data, alphas=[1 / model.bandwidth], ret_matrix=True)
    PYDEVD_WARN_EVALUATION_TIMEOUT = 200
    print(f'pval of the MMD two sample test {mmd.pval(distances)}')
    print(
        f'pval of the MMD two sample test with proposed bandwidth {1 / model.bandwidth} is {mmd_prop.pval(distances_prop)}, with MMD {mmd_prop_val}')
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
    model.fit(X_data)
    model_eval(model, X_data)

