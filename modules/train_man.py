#-----------------------------------------------------------------------------------------------#
#                                                                                               #
#   I M P O R T     G L O B A L     L I B R A R I E S                                           #
#                                                                                               #
#-----------------------------------------------------------------------------------------------#
import torch
from torch.utils.data import BatchSampler, RandomSampler, SequentialSampler
import numpy as np

#-----------------------------------------------------------------------------------------------#
#                                                                                               #
#   I M P O R T     L O C A L     L I B R A R I E S   /   F I L E S                             #
#                                                                                               #
#-----------------------------------------------------------------------------------------------#



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

    def get_global_model_params(self):
        """Extract the latest model parameters stored at the federated worker
        """
        model_parameters = []
        for layer in range(self.model_tensors_count):
            tensor_id = self.model_param_id+"_"+self.owner.id+"_{0}".format(layer)
            model_parameters.append(self.owner.get_obj(tensor_id))
        return model_parameters
    
    def setup_configurations(self, config_dict: dict):
        """Setup the train configurations sent from the server
        """
        self.lr = config_dict["lr"]
        self.plan_id = config_dict["plan_id"]
        self.model_id = config_dict["model_id"]+"_"+self.owner.id
        print(self.model_id)
        self.model_param_id = config_dict["model_param_id"]
        self.batch_size = config_dict["batch_size"]
        self.random_sample = config_dict["random_sample"]
        self.max_nr_batches = config_dict["max_nr_batches"]
        self.model_tensors_count = config_dict["model_tensor_count"]
    
    def next_batches(self, dataset_key: str):
        """Return next set of batches for training.
        """
        # raise value error if dataset doesn't exist
        if dataset_key not in self.datasets:
            raise ValueError(f"Dataset {dataset_key} unknown.")
        
        # check if there is a need to create a new dataloader
        if dataset_key not in self.data_info:
            self._create_data_loader(dataset_key=dataset_key)
        
        # check if iterators need to be created
        #if self.data_info[dataset_key][1] is None:
        #    self.data_info[dataset_key][1] = iter(self.data_info[dataset_key][0])
        
        # check if iterators need to be reset
        #if self.data_info[dataset_key][2] < (self.data_info[dataset_key][3]+self.max_nr_batches):
        #    self.data_info[dataset_key][1] = iter(self.data_info[dataset_key][0])
        #    self.data_info[dataset_key][3] = 0
        #    print("Iterator of dataloader reset")
        
        batches = []
        # sample the required number of batch
        for i in range(self.max_nr_batches):
            try:
                next_batch = next(self.data_info[dataset_key][1])
            except:
                # need to reset the iterator and get new batch
                self.data_info[dataset_key][1] = iter(self.data_info[dataset_key][0])
                next_batch = next(self.data_info[dataset_key][1])
            # append the new batch to list
            batches.append(next_batch)
        
        # update call count
        self.data_info[dataset_key][2] += self.max_nr_batches

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
        batch_count = int(len(self.datasets[dataset_key].targets) // self.batch_size)
        self.data_info[dataset_key] = [data_loader, None]
