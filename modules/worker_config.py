#-----------------------------------------------------------------------------------------------#
#                                                                                               #
#   I M P O R T     G L O B A L     L I B R A R I E S                                           #
#                                                                                               #
#-----------------------------------------------------------------------------------------------#
import torch
import torch.nn as nn
import torch.nn.functional as F

import syft as sy

#-----------------------------------------------------------------------------------------------#
#                                                                                               #
#   I M P O R T     L O C A L     L I B R A R I E S   /   F I L E S                             #
#                                                                                               #
#-----------------------------------------------------------------------------------------------#
from workers import FederatedWorker
from modules.model_loader import get_model
from modules.data_loader import load_dataset
from utils.utils import split_dataset_and_return_mine
from configs import globals as glb

#***********************************************************************************************#
#                                                                                               #
#   description:                                                                                #
#   helper function to divide a dataset appropriately if needed.                                #
#                                                                                               #
#***********************************************************************************************#

#***********************************************************************************************#
#                                                                                               #
#   description:                                                                                #
#   a fucntion to setup the worker node with model / dataset.                                   #
#                                                                                               #
#***********************************************************************************************#
def setup_worker_config(worker: FederatedWorker, rank: int, world_size: int):
    # create a model
    model = get_model(model_name=glb.MODEL)
    # add it to worker's local data
    worker.train_manager.add_model(model, key=glb.MODEL)
    # load dataset
    train_set, _ = load_dataset(dataset=glb.DATASET)
    my_dataset = split_dataset_and_return_mine(dataset=train_set,
                                               rank=rank,
                                               world_size=world_size)
    # add the dataset to local worker
    worker.train_manager.add_dataset(my_dataset, key=glb.DATASET_ID)
    # done return from this function
    return