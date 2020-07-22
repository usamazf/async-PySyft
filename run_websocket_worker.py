#-----------------------------------------------------------------------------------------------#
#                                                                                               #
#   I M P O R T     G L O B A L     L I B R A R I E S                                           #
#                                                                                               #
#-----------------------------------------------------------------------------------------------#
import argparse
import torch
import numpy as np

import syft as sy

#-----------------------------------------------------------------------------------------------#
#                                                                                               #
#   I M P O R T     L O C A L     L I B R A R I E S   /   F I L E S                             #
#                                                                                               #
#-----------------------------------------------------------------------------------------------#
from workers import FederatedWorker
from modules.worker_config import setup_worker_config

#***********************************************************************************************#
#                                                                                               #
#   description:                                                                                #
#   starting websocket server for a given client / worker node. Each now worker has a server.   #
#                                                                                               #
#***********************************************************************************************#
def start_websocket_worker(id, host, port, hook, rank, world_size):
    # create a server instance
    worker = FederatedWorker(id=id, host=host, port=port, hook=hook, verbose=True)
    # setup worker configurations
    setup_worker_config(worker, rank, world_size)
    # print a log message, do remember to clean up though
    print("Starting Worker Node {0}".format(id))
    worker.start()
    return worker

#***********************************************************************************************#
#                                                                                               #
#   description:                                                                                #
#   argument parsing and configurations for setting up the websocket server.                    #
#                                                                                               #
#***********************************************************************************************#
if __name__ == "__main__":
    
    # parse arguments
    parser = argparse.ArgumentParser(description="Run websocket server worker.")
    parser.add_argument("--port", type=int, help="port number of federated server, e.g. --port 8778", required=True)
    parser.add_argument("--host", type=str, default="localhost", help="host for the connection")
    parser.add_argument("--id", type=str, help="name (id) of the websocket server worker, e.g. --id alice", required=True)
    parser.add_argument("--rank", type=int, help="rank of current worker process, used purely for dataset loading", required=True)
    parser.add_argument("--world", type=int, help="total number of worker processes, used purely for dataset loading", required=True)
    args = parser.parse_args()
    
    # hook and start server
    hook = sy.TorchHook(torch)
    
    # call server start function
    worker = start_websocket_worker(id=args.id, host=args.host, port=args.port, hook=hook, rank=args.rank, world_size=args.world)
