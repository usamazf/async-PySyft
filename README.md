# Introduction

Async-PySyft is a implementation of Asyncrhonous Federated Learning 
using the PySyft library.


## Pre-requisites

In order to run this code, you need to install PySyft and all it's 
pre-requisites. As detailed by the OpenMined here: [PySyft ](https://github.com/OpenMined/PySyft). 
It is recommended to create a virtual environment for installing PySyft. 

Best way to do this with 
[Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/overview.html) Virtual environment. 

```bash
conda create -n pysyft python=3.7
conda activate pysyft # some older version of conda require "source activate pysyft" instead.
conda install jupyter notebook==5.7.8 tornado==4.5.3
```

Finally when you are ready to install simply run

> PySyft supports Python >= 3.6 and PyTorch 1.4

```bash
pip install 'syft[udacity]'
```

## Running the code