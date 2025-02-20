import inspect

import torch
from collections import defaultdict
from models.Generator import Generator_big
from models.Detector import Detector, Encoder, Decoder, DetectorExtra
import torch_two_sample as tts
from models.Mmd_loss import MMDLoss
from models.Mmd_loss_constrained import MMDLossConstrained
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import os
import operator
import datetime
from torch.autograd import Variable


class VGAN:
    '''
    V-MMD, a Subspace-Generative Moment Matching Network.

    Class for the method VMMD, the application of a GMMN to the problem of Subspace Generation. As a GMMN, no
    kernel learning is performed. The default values for the kernel are
    '''

    def __init__(self, batch_size=500, penalty=1, epochs=30, lr_G=0.007, lr_D=0.007,
                 iternum_d=1, iternum_g=5, seed=777,
                 weight_decay=0.04, path_to_directory=None,
                 generator= None,
                 #Currently does not do anything:
                 print_updates=False, gradient_clipping=False):
        self.storage = locals()
        self.train_history = defaultdict(list)
        self.batch_size = batch_size
        self.penalty = penalty
        self.epochs = epochs
        self.lr_G = lr_G
        self.lr_D = lr_D
        self.iternum_d = iternum_d
        self.iternum_g = iternum_g
        #self.momentum = momentum
        self.seed = seed
        if seed is None:
            self.seed = 777
        self.weight_decay = weight_decay
        self.path_to_directory = path_to_directory
        self.generator_optimizer = None
        self.__elm = False
        self.device = torch.device('cuda:0' if torch.cuda.is_available(
        ) else 'mps:0' if torch.backends.mps.is_available() else 'cpu')

        self.provided_generator = generator
        self.generator_loss_key = 'generator_loss'
        if generator is None:
            self.provided_generator = Generator_big
        self.print_updates = print_updates

    def __normalize(x, dim=1):
        return x.div(x.norm(2, dim=dim).expand_as(x))

    def __distance(self, x, y, dist):
        """
        Computes distance between corresponding points points in `x` and `y`
        using distance `dist`.
        """
        if dist == 'L2':
            return (x - y).pow(2).mean()
        elif dist == 'L1':
            return (x - y).abs().mean()
        elif dist == 'cos':
            x_n = self.__normalize(x)
            y_n = self.__normalize(y)
            return 2 - (x_n).mul(y_n).mean()
        else:
            assert dist == 'none', 'wtf ?'

    def __weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)
        elif classname.find('Linear') != -1:
            m.weight.data.normal_(0.0, 0.1)
            m.bias.data.fill_(0)

    def __plot_loss(self, path_to_directory, show=False):
        train_history = self.train_history
        plt.style.use('ggplot')
        generator_y = train_history['generator_loss']
        detector_y = train_history['detector_loss']
        x = np.linspace(1, len(generator_y), len(generator_y))
        fig, ax = plt.subplots()
        ax.plot(x, generator_y, color="cornflowerblue",
                label="Generator loss", linewidth=2)
        ax.plot(x, detector_y, color="black",
                label="Detector loss", linewidth=2)

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        ax.legend(loc="upper right")
        plt.savefig(path_to_directory / "train_history.png",
                    format="png", dpi=1200)

        if show == True:
            print("The show option has been depricated due to lack of utility")

    def get_params(self) -> dict:
        return {'batch size': self.batch_size, 'epochs': self.epochs, 'lr_g': self.lr_G,
                #'momentum': self.momentum,
                'weight decay': self.weight_decay,
                'batch_size': self.batch_size, 'seed': self.seed,
                'generator optimizer': self.generator_optimizer,
                "generator": self.generator.__class__.__name__,
                "detector": self.detector.__class__.__name__,}

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
        self.__plot_loss(path_to_directory, show=False)

    def load_models(self, path_to_generator, ndims, device: str = None):
        '''Loads models for prediction

        In case that the generator has already been trained, this method allows to load it (and optionally the discriminator) for generating subspaces
        Args:
            - path_to_generator: Path to the generator (has to be stored as a .keras model)
            - path_to_discriminator: Path to the discriminator (has to be stored as a .keras model) (Optional)
        '''
        if device == None:
            device = self.device
        latent_size = max(int(ndims / 16), 1)
        self.generator =self.get_the_networks(ndims, latent_size, device=device)[0]
        self.generator.load_state_dict(torch.load(
            path_to_generator, map_location=device))
        self.generator.eval()  # This only works for dropout layers
        self.generator_optimizer = f'Loaded Model from {path_to_generator} with {ndims} dimensions in the latent space'
        self.__latent_size = latent_size

    def get_the_networks(self, ndims: int, latent_size: int, device: str = None) -> tuple:
        """Object function to obtain the networks' architecture

        Args:
            ndims (int): Number of dimensions of the full space
            latent_size (int): Number of dimensions of the latent space
            device (str, optional): CUDA device to mount the networks to. Defaults to None.

        Returns:
            tuple: A tuple containing the generator and the detector of the network (child classes from torch.nn.Module)
        """
        if device == None:
            device = self.device
            # Check if only the constructor or a whole generator was passed.
        if inspect.isclass(self.provided_generator):
            generator = self.provided_generator(
                img_size=ndims, latent_size=latent_size).to(device)
        else:
            generator = self.provided_generator

        #detector = Detector(latent_size, ndims, Encoder, Decoder).to(device)
        detector = DetectorExtra(img_size=ndims, latent_size=latent_size).to(device)
        return generator, detector

    def fit(self, X: np.array):
        for _ in self.yield_fit(X):
            continue

    def yield_fit(self, X: np.array, embedding = lambda x: x, yield_epochs: int = None):

        cuda = torch.cuda.is_available()
        mps = torch.backends.mps.is_available()

        torch.manual_seed(self.seed)
        if cuda:
            torch.cuda.manual_seed(self.seed)
        elif mps:
            torch.mps.manual_seed(self.seed)

        # MODEL INTIALIZATION#
        self.__latent_size = latent_size = max(int(X.shape[1]/16), 1)
        ndims = X.shape[1]
        train_size = X.shape[0]
        self.batch_size = min(self.batch_size, train_size)

        device = self.device
        generator, detector = self.get_the_networks(
            ndims, latent_size, device=device)
        generator.apply(self.__weights_init)
        detector.apply(self.__weights_init)

        #gen_optimizer = torch.optim.Adadelta(generator.parameters(), lr=self.lr_G, weight_decay=self.weight_decay)
        #det_optimizer = torch.optim.Adadelta(detector.parameters(), lr=self.lr_D, weight_decay=self.weight_decay)

        gen_optimizer = torch.optim.Adam(generator.parameters(), lr=self.lr_G, weight_decay=self.weight_decay, betas=(0.5,0.9))
        det_optimizer = torch.optim.Adam(detector.parameters(), lr=self.lr_D, weight_decay=self.weight_decay, betas=(0.5,0.9))

        self.generator_optimizer = gen_optimizer.__class__.__name__
        self.detector_optimizer = det_optimizer.__class__.__name__
        # loss_function =  tts.MMDStatistic(self.batch_size, self.batch_size)
        loss_function = MMDLossConstrained(weight=self.penalty)

        # OPTIMIZATION STUFF
        one = torch.mps.Tensor([1])
        minusone = one * -1

        # DATA LOADER#
        if cuda:
            data_loader = DataLoader(
                X, batch_size=self.batch_size, drop_last=True, pin_memory=False, shuffle=True)
        else:  # Uses CUDA if Available, other wise MPS or nothing
            data_loader = DataLoader(
                X, batch_size=self.batch_size, drop_last=True, pin_memory=mps, shuffle=True)
        batch_number = data_loader.__len__()

        # BATCH LOOP#
        iternum_d = 1
        iternum_g = 1
        detector_loss = np.nan
        generator_loss = np.nan
        for epoch in range(self.epochs):
            if self.print_updates:
                print(f'\rEpoch {epoch} of {self.epochs}')

            # GET NOISE TENSORS#
            if cuda:
                noise_tensor = torch.cuda.FloatTensor(
                    self.batch_size, latent_size).to(torch.device('cuda'))
            elif mps:
                noise_tensor = torch.mps.Tensor(
                    self.batch_size, latent_size).to(torch.device('mps'))
            else:
                noise_tensor = torch.Tensor(self.batch_size, latent_size)

            # ELM
            if self.__elm == True:
                for p in detector.encoder.parameters():
                    p.requires_grad = False
            if iternum_d <= self.iternum_d:
                detector_loss = 0
                for batch in tqdm(data_loader, leave=False):
                    # Make sure there is only 1 observation per row.
                    batch = batch.view(self.batch_size, -1)
                    if cuda:
                        batch = batch.cuda()
                    elif mps:
                        batch = batch.to(torch.float32).to(
                            torch.device('mps'))  # float64 not suported with mps

                    # GET SUBSPACES AND ENCODING-DECODING
                    for p in detector.decoder.parameters():
                        p.requires_grad = True
                    batch_enc, batch_dec = detector(batch)
                    with torch.no_grad():
                        #noise_tensor = Variable(noise_tensor.normal_())
                        #fake_subspaces = Variable(generator(noise_tensor).clone().detach())  # Freeze G
                        noise_tensor = noise_tensor.normal_()
                        fake_subspaces = generator(noise_tensor).clone().detach()  # Freeze G

                    projected_batch = fake_subspaces * batch

                    projected_batch_enc, projected_batch_dec = detector(
                        projected_batch
                        #+ torch.less(batch, 1/batch.shape[1])*torch.mean(batch, dim=0)
                    )

                    L2_distance_batch = self.__distance(
                        batch.view(self.batch_size, -1), batch_dec, 'L2')

                    L2_distance_projected_batch = self.__distance(
                        (projected_batch #+ torch.less(batch, 1/batch.shape[1])*torch.mean(batch, dim=0)
                         ).view(self.batch_size, -1)
                        , projected_batch_dec, 'L2')

                    # OPTIMIZATION STEP DETECTOR
                    det_optimizer.zero_grad()
                    batch_loss_D = (minusone.to(self.device)*
                                    (loss_function(batch_enc, projected_batch_enc, fake_subspaces) - .1 *
                                                       L2_distance_batch - .1*L2_distance_projected_batch))  # Constrained MMD Loss
                    self.bandwidth = loss_function.bandwidth
                    batch_loss_D.backward()
                    det_optimizer.step()
                    detector_loss += float(batch_loss_D.to(
                        'cpu').detach().numpy())/batch_number
                iternum_d += 1
                iternum_g = 1

            elif iternum_g <= self.iternum_g:
                generator_loss = 0
                for batch in tqdm(data_loader, leave=False):
                    # Make sure there is only 1 observation per row.
                    batch = batch.view(self.batch_size, -1)
                    if cuda:
                        batch = batch.cuda()
                    elif mps:
                        batch = batch.to(torch.float32).to(
                            torch.device('mps'))  # float64 not suported with mps
                    # GET SUBSPACES AND ENCODING-DECODING
                    batch_enc, batch_dec = detector(batch)

                    #noise_tensor = Variable(noise_tensor.normal_())
                    #fake_subspaces = Variable(generator(noise_tensor))  # Unfreeze G
                    #fake_subspaces.requires_grad = True

                    noise_tensor = torch.normal(mean=0, std=1, size=noise_tensor.shape, device=noise_tensor.device,
                                                requires_grad=True)
                    fake_subspaces = generator(noise_tensor)  # Unfreeze G

                    projected_batch_enc, projected_batch_dec = detector(
                        fake_subspaces*batch #+ torch.less(batch, 1/batch.shape[1])*torch.mean(batch, dim=0)
                        )

                    #L2_distance_batch = self.__distance(
                    #    batch.view(self.batch_size, -1), batch_dec, 'L2')
                    #L2_distance_projected_batch = self.__distance((fake_subspaces*batch + torch.less(
                    #    batch, 1/batch.shape[1])*torch.mean(batch, dim=0)).view(self.batch_size, -1), projected_batch_dec, 'L2')

                    # OPTIMIZATION STEP GENERATOR
                    for p in detector.parameters():
                        p.requires_grad = False
                    gen_optimizer.zero_grad()

                    batch_loss_G = loss_function(
                       projected_batch_enc,  batch_enc, fake_subspaces)  # Constrained MMD Loss
                    self.bandwidth = loss_function.bandwidth
                    batch_loss_G.backward()

                    # Calculate the gradient norm
                    tensors_with_grad = [p for p in generator.parameters() if p.grad is not None]
                    if len(tensors_with_grad) > 0:
                        gradient_norm = torch.norm(torch.stack(
                            [torch.norm(p.grad.detach(), 2) for p in generator.parameters() if p.grad is not None]), 2)
                        # Print the gradient norm
                        #print(f"Gradient norm for Generator: {gradient_norm}")
                    else:
                        raise ValueError("No gradient for the generator")


                    gen_optimizer.step()
                    generator_loss += float(batch_loss_G.to(
                        'cpu').detach().numpy())/batch_number
                iternum_g += 1
                if iternum_g > self.iternum_g:
                    iternum_d = 1
            if self.print_updates:
                print(f"Average loss in the epoch Generator: {generator_loss}")
                print(f"Average loss in the epoch Detector: {detector_loss}")
            self.train_history["generator_loss"].append(generator_loss)
            self.train_history["detector_loss"].append(detector_loss)

            if yield_epochs is not None and epoch % yield_epochs == 0:
                self.generator = generator
                self.detector = detector
                yield epoch

        if not self.path_to_directory == None:
            path_to_directory = Path(self.path_to_directory)
            if operator.not_(path_to_directory.exists()):
                os.makedirs(path_to_directory, exist_ok=True)
            if operator.not_(Path(path_to_directory/'models').exists()):
                os.mkdir(path_to_directory / 'models')
            run_number = int(len(os.listdir(path_to_directory/'models'))/2)
            torch.save(generator.state_dict(),
                       path_to_directory/'models'/f'generator_{run_number}.pt')
            torch.save(generator.state_dict(),
                       path_to_directory/'models'/f'detector_{run_number}.pt')
            self.model_snapshot(path_to_directory, run_number, show=True)

        self.generator = generator
        self.detector = detector

    def generate_subspaces(self, nsubs, round: bool = True):
        noise_tensor = torch.Tensor(nsubs, self.__latent_size).to('cpu')
        if not self.seed == None:
            torch.manual_seed(self.seed)
        noise_tensor.normal_()
        u = self.generator(noise_tensor.to(self.device))
        if round:
            u = torch.greater_equal(u, 0.5) * 1
        u = u.detach()
        return u


if __name__ == "__main__":
    # mean = [1,1,0,0,0,0,0,0,2,1]
    # cov = [[1,1,0,0,0,0,0,0,0,0],[1,1,0,0,0,0,0,0,0,0],[0,0,1,1,1,0,0,0,0,0],[0,0,1,1,1,0,0,0,0,0],[0,0,1,1,1,0,0,0,0,0],[0,0,0,0,0,1,0,0,0,0],[0,0,0,0,0,0,1,0,0,0],
    #       [0,0,0,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,0,0,1]]
    mean = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    cov = [[1, 0, 0, 0, 0, 0, 0, 0, 500, 500], [0, 1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 1, 0, 0], [500, 0, 0, 0, 0, 0, 0, 0, 1, 500], [500, 0, 0, 0, 0, 0, 0, 0, 500, 1]]
    X_data = np.random.multivariate_normal(mean, cov, 2000)

    model = VGAN(epochs=15, penalty=1, path_to_directory=Path() / "experiments" /
                 f"Example_normal_{datetime.datetime.now()}_vgan", lr_G=0.01, lr_D=0.1)
    model.fit(X_data.astype(np.float32))

    X_sample = torch.mps.Tensor(pd.DataFrame(
        X_data).sample(500).to_numpy()).to('cuda:0')
    u = model.generate_subspaces(500)
    uX_data = model.detector.encoder(
        u * torch.mps.Tensor(X_sample).to(model.device) + torch.mean(X_sample, dim=0)*(~u))
    X_sample = model.detector.encoder(X_sample)
    mmd = tts.MMDStatistic(500, 500)
    mmd_val, distances = mmd(X_sample, uX_data, alphas=[0.01], ret_matrix=True)
    mmd_prop = tts.MMDStatistic(500, 500)
    mmd_prop_val, distances_prop = mmd_prop(
        X_sample, uX_data, alphas=[1/model.bandwidth], ret_matrix=True)
    PYDEVD_WARN_EVALUATION_TIMEOUT = 200
    print(
        f'pval of the MMD two sample test {mmd.pval(distances)}, with MMD {mmd_val}')
    print(
        f'pval of the MMD two sample test with proposed bandwidth {1/model.bandwidth} is {mmd_prop.pval(distances_prop)}, with MMD {mmd_prop_val}')
    unique_subspaces, proba = np.unique(
        np.array(u.to('cpu')), axis=0, return_counts=True)
    proba = proba/np.array(u.to('cpu')).shape[0]
    unique_subspaces = [str(unique_subspaces[i]*1)
                        for i in range(unique_subspaces.shape[0])]

    print(pd.DataFrame({'subspace': unique_subspaces, 'probability': proba}))
    print(np.sum(proba))