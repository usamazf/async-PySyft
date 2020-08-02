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

Running the code has two distinct parts i.e. starting up the server and initiating the clients. Each of these steps are explained below.

### Staring the Federated Server
First thing we need to do is to run the Federated Server. In order to do so
we use the following command

```bash
python start_federated_server.py --host "host_address" --port "listen_port"
```

### Starting the Federated Workers

After the server has successfully started next step it to run the Federated Workers. Let us assume we want to run 10 workers on two machines: Machine 1 
and Machine 2 with 5 workers each. This can be done using following commands:

#### Machine 1:
```bash
python start_federated_workers.py --remotehost "server_ip_address" --remoteport "server_listen_port" --host "machine_1_ip" --port "starting_listen_port" --count 5 --rank 0 --world 10 --id "worker_id_prefix"
```

#### Machine 2:
```bash
python start_federated_workers.py --remotehost "server_ip_address" --remoteport "server_listen_port" --host "machine_2_ip" --port "starting_listen_port" --count 5 --rank 5 --world 10 --id "worker_id_prefix"
```

#### Explanation:

The above commands will run 5 workers each on Machine 1 and Machine 2. Machine 1 will have workers with ranks 0-4 and Machine 2 will have workers with rank 5-9.

> **_NOTE:_**
> The world size (--world flag) should always be the total number of workers across all machines. The starting rank (--rank flag) should always be exclusive and continuous
list of workers.

## Acknowledgements
1) The multiple model and batch gradient computations are from the repo here: https://github.com/owkin/grad-cnns

2) The training implementation for differential privacy based models etc. were inspired from the repo here: https://github.com/heyyjudes/dp-sgd









