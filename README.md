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

After the server has successfully started next step it to run the Federated Workers. Let us assume we want to run 10 workers on two Servers: Server 1 
and Server 2 with 5 workers each. This can be done using following commands:

### Server 1:
```bash
python start_federated_workers.py --remotehost "server_ip_address" --remoteport "server_listen_port" --host "server_1_ip" --port "starting_listen_port" --count 5 --rank 0 --world 10 --id "worker_id_prefix"
```

### Server 2:
```bash
python start_federated_workers.py --remotehost "server_ip_address" --remoteport "server_listen_port" --host "server_2_ip" --port "starting_listen_port" --count 5 --rank 5 --world 10 --id "worker_id_prefix"
```

### Explanation:

The above commands with run 5 workers each on Server 1 and Server 2. Server 1 will have workers with ranks 0-4 and Server 2 will have workers with rank 5-9.

> **_NOTE:_**
> The world size should always be the total number of workers among all servers combined.








