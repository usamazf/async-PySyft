#-----------------------------------------------------------------------------------------------#
#                                                                                               #
#   I M P O R T     G L O B A L     L I B R A R I E S                                           #
#                                                                                               #
#-----------------------------------------------------------------------------------------------#
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import RandomSampler, SequentialSampler
import numpy as np

import syft as sy
from syft.messaging.message import ObjectRequestMessage

#-----------------------------------------------------------------------------------------------#
#                                                                                               #
#   I M P O R T     L O C A L     L I B R A R I E S   /   F I L E S                             #
#                                                                                               #
#-----------------------------------------------------------------------------------------------#
from workers import FederatedWorkerPointer
from utils.utils import model_flatten, model_unflatten, AverageMeter

#***********************************************************************************************#
#                                                                                               #
#   description:                                                                                #
#   a class to hold all the training info and help manage data for Federated Workers.           #
#                                                                                               #
#***********************************************************************************************#
class TrainingManager:
    def __init__(self, worker, datasets, models):
        # setup information sent by the federated worker
        self.owner = worker
        self.datasets = datasets if datasets is not None else dict()
        self.models = models if models is not None else dict()
        
        # dataloader and related information need to sample batches
        self.data_info = dict()
        
    def add_dataset(self, dataset, key: str):
        """Add new dataset to the current federated worker object.
        Args:
            dataset: a new dataset instance to be added.
            key: a unique identifier for the new dataset.
        """
        if key not in self.datasets:
            self.datasets[key] = dataset
        else:
            raise ValueError(f"Key {key} already exists in Datasets")
    
    def remove_dataset(self, key: str):
        """Remove a dataset from current federated worker object.
        Args:
            key: a unique identifier for the dataset to be removed
        """
        if key in self.datasets:
            del self.datasets[key]

    def add_model(self, model, key: str):
        """Add new model to the current federated worker object.
        Args:
            model: a new model instance to be added.
            key: a unique identifier for the new model.
        """
        if key not in self.models:
            self.models[key] = model
        else:
            raise ValueError(f"Key {key} already exists in Models")
    
    def remove_model(self, key: str):
        """Remove a model from current federated worker object.
        Args:
            key: a unique identifier for the model to be removed
        """
        if key in self.models:
            del self.models[key]
    
    def get_train_plan(self):
        """Extract the train plan stored at the federated worker
        """
        return self.owner.get_obj(self.plan_id)

    def get_global_model(self):
        """Extract the latest model parameters stored at the federated worker
        """
        model_params = self.owner.get_obj(self.model_param_id)
        model = self.models[self.model_id]
        # unpack parameters into the locally stored model
        model_unflatten(model, model_params)
        return model
    
    def get_criterion(self):
        """Decide which criterion is required and build it
        """
        if self.criterion == "CrossEntropyLoss":
            criterion = nn.CrossEntropyLoss()
        
        return criterion
        
    def get_optimizer(self, model):
        """Decide which optimizer is required and build it
        """
        if self.optimizer == "SGD":
            optimizer = optim.SGD(model.parameters(), lr=0.0001)
        
        return optimizer

    def setup_configurations(self, config_dict: dict):
        """Setup the train configurations sent from the server
        """
        # training related information
        self.lr = config_dict["lr"]
        self.batch_size = config_dict["batch_size"]
        self.random_sample = config_dict["random_sample"]
        self.max_nr_batches = config_dict["max_nr_batches"]
        self.criterion = config_dict["criterion"]
        self.optimizer = config_dict["optimizer"]
        
        # local storage object related information
        self.plan_id = config_dict["plan_id"]
        self.model_id = config_dict["model_id"]
        self.model_param_id = config_dict["model_param_id"]
        self.result_losses_id = config_dict["result_losses_id"]
        self.result_params_id = config_dict["result_params_id"]
        self.result_differ_id = config_dict["result_differ_id"]
        
        # worker related sampling information
        self.fellow_workers = config_dict["sampled_workers"] #[sample_id, worker_id, host, port]
        self.self_sample_id = config_dict["sampled_worker_id"]
    
    def next_batches(self, dataset_key: str):
        """Return next set of batches for training.
        """
        # raise value error if dataset doesn't exist
        if dataset_key not in self.datasets:
            raise ValueError(f"Dataset {dataset_key} unknown.")
        
        # check if there is a need to create a new dataloader
        if dataset_key not in self.data_info:
            self._create_data_loader(dataset_key=dataset_key)
        
        batches = []
        # sample the required number of batch
        for i in range(self.max_nr_batches):
            try:
                next_batch = next(self.data_info[dataset_key][1])
            except:
                # need to reset the iterator and get new batch
                print("Resetting Train Loader {0}\n".format(self.owner.id))
                self.data_info[dataset_key][1] = iter(self.data_info[dataset_key][0])
                next_batch = next(self.data_info[dataset_key][1])
            # append the new batch to list
            batches.append(next_batch)
        
        # return the requested batches
        return batches
    
    def _create_data_loader(self, dataset_key: str):
        """Helper function to create the dataloader as per our requirements
        """
        data_range = range(len(self.datasets[dataset_key]))
        # check requirements for data sampling
        if self.random_sample:
            sampler = RandomSampler(data_range)
        else:
            sampler = SequentialSampler(data_range)
        # create the dataloader as per our requirments
        data_loader = torch.utils.data.DataLoader(
            self.datasets[dataset_key],
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=0,
            drop_last=True,
        )

        # add it as a local object
        self.data_info[dataset_key] = [data_loader, None]
        
    def store_training_results(self, updated_model, losses):
        """Store the training results as local objects
        """
        # register losses array as a local object
        loss = torch.tensor([1,2,3])#losses)
        loss.id = self.result_losses_id #"loss"
        self.owner.register_obj(loss)
        
        # register updated model as a local object
        updated_params = model_flatten(updated_model)
        updated_params.id = self.result_params_id #"updated_params"
        self.owner.register_obj(updated_params)
        
        # compute change and register it for consumption by the server
        difference = updated_params - (self.owner.get_obj(self.model_param_id))
        difference.id = self.result_differ_id #"differnce"
        self.owner.register_obj(difference)

        
    def additive_secret_sharing(self, updated_model, losses):
        """Apply additive secrete sharing techniques to carry out model average
        """
        # register losses array as a local object
        #loss = torch.tensor([1,2,3])#losses)
        #loss.id = self.result_losses_id #"loss"
        #self.owner.register_obj(loss)
        
        # register updated model as a local object
        updated_params = model_flatten(updated_model)

        # register updated model as a local object
        updated_params = model_flatten(updated_model)
        updated_params.id = self.result_params_id #"updated_params"
        self.owner.register_obj(updated_params)

        # chunk and store split parts locally
        self_chunk = None
        for idx, chunk in enumerate(updated_params.chunk(len(self.fellow_workers))):
            cloned_chunk = chunk.clone().detach()
            cloned_chunk.id = "chunk_{0}".format(idx)
            if idx==self.self_sample_id:
                self_chunk = cloned_chunk
            else:
                self.owner.register_obj(cloned_chunk)
        
        # list to retrieve results into
        chunk_list = []
        
        # retrieve chunks from other workers
        for sample_id, worker_id, host, port in self.fellow_workers:
            # skip if self
            if self.self_sample_id==sample_id:
                continue
            # create a connection
            kwargs_websocket = {"host": host, "hook": self.owner.hook, "verbose": False}
            client_ptr = FederatedWorkerPointer(id=worker_id, port=int(port), **kwargs_websocket)
            
            # fetch desired chunk
            msg = ObjectRequestMessage("chunk_{0}".format(self.self_sample_id), None, "")
            serialized_message = sy.serde.serialize(msg)
            chunk_recvd = sy.serde.deserialize(client_ptr._send_msg(serialized_message))
            
            # append the fetched chunk to the local list
            chunk_list.append(chunk_recvd)
        
        # carry out a sum / average of the all the chunks fetched with local chunk
        for recv_chunk in chunk_list:
            self_chunk.add_(recv_chunk)
        
        self_chunk.div_(len(chunk_list)+1)
        
        # store the final results locally to be fetched by the server
        self_chunk.id = "final_chunk_{0}".format(self.self_sample_id)
        self.owner.register_obj(cloned_chunk)
