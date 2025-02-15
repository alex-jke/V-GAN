from vmmd import VMMD
from vgan import VGAN
import numpy as np
import torch
import pandas as pd 
import numpy as np
from pathlib import Path
import datetime
import torch_two_sample  as tts
import pandas as pd
from sklearn.preprocessing import normalize

if __name__ == "__main__":
        
        # X_data = np.load("data/SpamBase.npz")
        # df = pd.DataFrame(X_data["X"])
        # df["outlier"] = X_data["y"]
        # df["id"] = df.index

        # df["outlier"] = pd.factorize(df["outlier"], sort=True)[0] #Keep in mind: 0 inlier, 1 outlier
        # X_data = normalize(df[df["outlier"] == 0].to_numpy(), axis=0)
        X_data = pd.read_parquet("data/p53_mutant_inactive.parquet").to_numpy()
        X_data = normalize(X_data, axis = 0)

        #model = VMMD(epochs = 1500, batch_size= 500, path_to_directory=Path()/ "experiments" / f"Example_dataset_{datetime.datetime.now()}", lr=0.01)
        model = VGAN(epochs = 1500, penalty=10, batch_size= 500, path_to_directory=Path() / "experiments" / f"Example_dataset_{datetime.datetime.now()}", iternum_d=1, iternum_g=5, lr_G = 0.01, lr_D = 0.01)
        model.fit(X_data)

        X_sample = torch.mps.Tensor(pd.DataFrame(X_data).sample(500).to_numpy()).to('mps:0')
        u = model.generate_subspaces(500)
        uX_sample = u *torch.mps.Tensor(X_sample).to(model.device) + torch.mean(X_sample,dim=0)*(~u)
        mmd = tts.MMDStatistic(500, 500) 
        mmd_val, distances = mmd(X_sample,uX_sample,alphas=[0.01], ret_matrix=True)
        mmd_prop = tts.MMDStatistic(500, 500)
        mmd_prop_val, distances_prop = mmd_prop(X_sample,uX_sample,alphas=[1/model.bandwidth], ret_matrix=True)
        PYDEVD_WARN_EVALUATION_TIMEOUT = 200
        print(f'pval of the MMD two sample test {mmd.pval(distances)}')
        print(f'pval of the MMD two sample test with proposed bandwidth {1/model.bandwidth} is {mmd_prop.pval(distances_prop)}, with MMD {mmd_prop_val}' )
        unique_subspaces, proba= np.unique(np.array(u.to('cpu')), axis=0, return_counts=True)
        proba = proba/np.array(u.to('cpu')).shape[0]
        unique_subspaces = [str(unique_subspaces[i]*1) for i in range(unique_subspaces.shape[0])]

        print(pd.DataFrame({'subspace': unique_subspaces, 'probability': proba}))
        print(np.sum(proba))
