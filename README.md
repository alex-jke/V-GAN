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

# Running
If running the LLama models, which are run by some experiments, make sure you are logged in into Huggingface and have 
access to the LLama models used here (meta-llama/Llama-3.2-1B and meta-llama/Llama-3.2-3B). To do so, you can use the following command:
```
huggingface-cli login
```

Torch version in requirements not correct, will give error. To install proper version:
```
pip install torch==2.2.1+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
```