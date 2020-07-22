# Introduction

Async-PySyft is a implementation of Asyncrhonous Federated Learning 
using the PySyft library.


## Pre-requisites

In order to run this code, you need to install PySyft and all it's 
pre-requisites. As detailed by the OpenMined here: [PySyft ](https://github.com/OpenMined/PySyft). 
It is recommended to create a virtual environment for installing PySyft. 

Best way to do this with 
[Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/overview.html) virtual environment. 

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

First thing we need to do is to run the Federated Server. In order to do so
we use the following command

```bash
python start_federated_server.py --host "host_address" --port "listen_port"
```

After the server has successfully started next step it to run the Federated Workers. This can be done using following command:

```bash
python start_federated_workers.py --remotehost "server_host_address" --remoteport "server_listen_port" --host "host_for_local_workers" --port "starting_listen_port" --count {number of workers} --id "worker_id_prefix"
```








