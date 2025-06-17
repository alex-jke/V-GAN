from vmmd import VMMD
import torch  
import numpy as np 
from pathlib import Path
import datetime
#import torch_two_sample  as tts
import pandas as pd
import argparse
import os
import operator

def covariance_matrix_generator(dim_subspace, sigma, dim_space = 10, return_subspace = False):
    '''Create a covariance matrix
    
    This functions creates, give a dimensionality of a whole space, the desired subspace, and the covariance value (sigma), a simetric covariance matrix with a random subspace embedded
    in it.
    Args:
        -
    '''
    cov_vector = np.sort(np.random.choice(range(dim_space - 1),dim_subspace,replace=False))
    x_matrix = np.diag(np.ones(dim_space))

    for i, feature in enumerate(cov_vector):  
        if i < cov_vector.size - 1:  
            for j in range(cov_vector.size - 1 - i):    
                x_matrix[feature, cov_vector[i+j+1]] = sigma
    x_matrix = x_matrix + np.transpose(x_matrix) - np.diag((np.ones(dim_space))) #Simetric
    #x_matrix = np.matmul(x_matrix, np.transpose(x_matrix)) #Semi-positivelly defined
    if return_subspace == True:
        cov_subspace = np.zeros(dim_space)
        for element in cov_vector:
            cov_subspace[element] = 1    
        return x_matrix, cov_subspace.astype(int)
    else: return x_matrix


def launch_corr_experiments(dim_subspace_vector, sigma_vector, dim_space = 10, repetitions = 10):
    contain_matrix = np.zeros(shape = [len(sigma_vector),len(dim_subspace_vector)])
    mmd_test_matrix = np.zeros(shape = [len(sigma_vector),len(dim_subspace_vector)])
    p_val_matrix = np.zeros(shape = [len(sigma_vector),len(dim_subspace_vector)])
    selected_subsp_dim_matrix = np.zeros(shape = [len(sigma_vector),len(dim_subspace_vector)])
    proba_matrix = np.zeros(shape = [len(sigma_vector),len(dim_subspace_vector)])
    path_for_exiperiments = Path.cwd()/ "experiments" / f"Correlation_extraction_{datetime.datetime.now()}" #So date-time is fixed ONCE only (and we store every individual experiments inside)

    for index_on_sigma, sigma in enumerate(sigma_vector):
        contain_vector_on_d = np.zeros(len(dim_subspace_vector))
        mmd_test_vector_on_d = np.zeros(len(dim_subspace_vector))
        p_value_vector_on_d = np.zeros(len(dim_subspace_vector))
        selected_subsp_dim_vector_on_d = np.zeros(len(dim_subspace_vector))
        proba_vector_on_d = np.zeros(len(dim_subspace_vector))

        for index_on_d, dim_subspace in enumerate(dim_subspace_vector):
            contain_counter = 0
            mmd_test_counter = 0
            p_value_counter = 0
            selected_subsp_dim_counter = 0
            proba_counter = 0
            for i in range(repetitions):
                Cov_matrix, cov_subspace = covariance_matrix_generator(dim_subspace,sigma,dim_space, return_subspace=True)
                mean = np.zeros(dim_space)
                X_data = np.random.multivariate_normal(mean,Cov_matrix,2000)
                
                if operator.not_(path_for_exiperiments.exists()):
                    os.mkdir(path_for_exiperiments)
                
                model = VMMD(epochs = 1500, path_to_directory= path_for_exiperiments/f"run{i}_with_sigma{sigma}_dimsubspace{dim_subspace}", lr = 0.01)
                model.fit(X_data)
                u = model.generate_subspaces(500)
                unique_subspaces, proba= np.unique(np.array(u.to('cpu')), axis=0, return_counts=True)
                proba = proba/np.array(u.to('cpu')).shape[0]
                unique_subspaces = [unique_subspaces[i]*1 for i in range(unique_subspaces.shape[0])]

                index_of_max_subspace = np.argmax(proba)
                control_vector = np.array(unique_subspaces)[index_of_max_subspace, :]  - cov_subspace #To check whether the subspaces are included is easier to check a 1-hot 
                                                                                                      #encoding of them. If the substraction of the encodings yields -1, it is not contained
                contained = 1
                for entry in control_vector:
                    if entry == -1: contained = 0
                if contained == 1:
                    contain_counter += 1/repetitions


                X_sample = torch.mps.Tensor(pd.DataFrame(X_data).sample(500).to_numpy()).to('mps:0')
                uX_data = u *torch.mps.Tensor(X_sample).to(model.device) + torch.mean(X_sample,dim=0)*(~u)
                mmd = tts.MMDStatistic(500, 500)
                mmd_val, distances = mmd(X_sample,uX_data,alphas=[1/model.bandwidth], ret_matrix=True)
                p_value = mmd.pval(distances)
                if p_value <= 0.05:
                    mmd_test_counter += 1/repetitions

                p_value_counter += p_value/repetitions

                selected_subsp_dim_counter += np.sum(np.array(unique_subspaces)[index_of_max_subspace, :])/repetitions

                proba_counter += proba[index_of_max_subspace]/repetitions
            
            contain_vector_on_d[index_on_d] = contain_counter
            mmd_test_vector_on_d[index_on_d] = mmd_test_counter
            p_value_vector_on_d[index_on_d] = p_value_counter
            selected_subsp_dim_vector_on_d[index_on_d] = selected_subsp_dim_counter
            proba_vector_on_d[index_on_d] = proba_counter

        contain_matrix[index_on_sigma,:] = contain_vector_on_d
        mmd_test_matrix[index_on_sigma,:] = mmd_test_vector_on_d
        p_val_matrix[index_on_sigma,:] = p_value_vector_on_d
        selected_subsp_dim_matrix[index_on_sigma,:] = selected_subsp_dim_vector_on_d
        proba_matrix[index_on_sigma,:] = proba_vector_on_d
    contain_matrix = pd.DataFrame(contain_matrix,columns = args.dim_subspace_vector, index = args.sigma_vector)
    mmd_test_matrix = pd.DataFrame(mmd_test_matrix,columns = args.dim_subspace_vector, index = args.sigma_vector)
    p_val_matrix = pd.DataFrame(p_val_matrix,columns = args.dim_subspace_vector, index = args.sigma_vector)
    selected_subsp_dim_matrix = pd.DataFrame(selected_subsp_dim_matrix, columns = args.dim_subspace_vector, index = args.sigma_vector)
    proba_matrix = pd.DataFrame(proba_matrix, columns = args.dim_subspace_vector, index = args.sigma_vector)

    contain_matrix.to_csv(path_for_exiperiments/f"contain_matrix.csv")
    mmd_test_matrix.to_csv(path_for_exiperiments/f"mmd_test_matrix.csv")
    p_val_matrix.to_csv(path_for_exiperiments/f"p_val_matrix.csv")
    selected_subsp_dim_matrix.to_csv(path_for_exiperiments/f"selected_subsp_dim_matrix.csv")
    proba_matrix.to_csv(path_for_exiperiments/f"proba_matrix.csv")

    return contain_matrix, mmd_test_matrix, p_val_matrix, selected_subsp_dim_matrix, proba_matrix


def parse_arguments(): 
   parser = argparse.ArgumentParser(description="Covariance extraction experiments for V-GAN") 
   parser.add_argument("--dim_subspace_vector", nargs='+', type=int, default=[5], help='Array of multiple dimensions for the subspaces')
   parser.add_argument("--sigma_vector", nargs='+', type=float, default=[20,50], help='Array of multiple covariance values for the subspaces')
   parser.add_argument("--repetitions", type=int, default=10, help='Number of repetitions taken place for each dimensionality and covariance value')
   parser.add_argument("--dim_space", type=int, default= 10, help="Dimensionality of the full-space where the experiment is taken place")

   return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    
    contain_matrix, mmd_test_matrix, p_val_matrix , selected_subsp_dim_matrix, proba_matrix = launch_corr_experiments(args.dim_subspace_vector, args.sigma_vector, args.dim_space, args.repetitions)

    
    print(f"The freq. matrix of the contain ratio for each subspace is: \n {contain_matrix} \n The freq. matrix of the pass ratio for the mmd two sample test is: \n {mmd_test_matrix} \n The average p_value matrix is: \n {p_val_matrix} \n, The average dimensionality of the selected subspace is: \n {selected_subsp_dim_matrix}\n The average probability share of the selected subspace is: \n {proba_matrix}")