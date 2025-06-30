![vgan1-light](https://github.com/user-attachments/assets/770fe2f6-8c42-4e4d-b7bd-bf015c46f993)
============================================================================


# Introduction 
Repository for the V-GAN algorithm in paper "Adversarial Subspace Generation for Knowledge Discovery in High Dimensional Data" for subspace search.

Our proposed algorithm, V-GAN, is capable of identifying a collection of subspaces relevant to a studied population $\mathbf{X}$. We do so by building on a theoretical framework that explains the _Multiple Views_ phenomenom of data. 
Details of this will be added in the future, alongside a pre-print version of the paper.


# Installation

To install, simply use the requirements.txt file 
`pip install -r requirements.txt`
Additionally, if you plan to train VGAN, you should also install the torch-two-sample package: 

```
git clone https://github.com/josipd/torch-two-sample.git
cd torch-two-sample
pip install .
```

Furthermore, when using the PyCharm IDE, the src folder should be marked as a source root. This can be done by right-clicking on the src folder in the PyCharm project explorer and selecting "Mark Directory as" -> "Sources Root". This will allow PyCharm to recognize the modules in the src folder correctly.
Otherwise, errors might ocur when trying to import modules from the src folder.

If running the LLama models, which are run by some experiments, make sure you are logged in into Huggingface and have 
access to the LLama models used here (meta-llama/Llama-3.2-1B and meta-llama/Llama-3.2-3B). To do so, you can use the following command:
```
huggingface-cli login
```

The version of Torch in requirements not quite correct and might give error. If that is the case, the proper version can be installed via:
```
pip install torch==2.2.1+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
```

# Running

There are several pipelines available for running the experiments. These are located in the `src/text` directory:
- `vmmd_experiments.py`: This script runs V-GAN without kernel learning on numeric data.
- `vgan_experiments.py`: This script runs V-GAN with kernel learning on numeric data.
- `od_experiments.py`: This script runs the Outlier Detection experiments.
- `vmmd_lightning_text_experiments.py`: This script runs V-GAN without kernel learning using PyTorch Lightning on textual data.

All of these scripts include a sample usage in the `if __name__ == "__main__":` section. There, the parameters for the experiments can be set, such as the dataset to use, the number of epochs, and other hyperparameters.

# Documentation
A documentation for this project is available inside the `documentation` folder.