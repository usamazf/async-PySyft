#-----------------------------------------------------------------------------------------------#
#                                                                                               #
#   I M P O R T     G L O B A L     L I B R A R I E S                                           #
#                                                                                               #
#-----------------------------------------------------------------------------------------------#
import torch
import syft as sy
from timeit import default_timer as timer

from typing import Union
from typing import List

from syft.workers.websocket_server import WebsocketServerWorker
from syft.generic.abstract.tensor import AbstractTensor

#-----------------------------------------------------------------------------------------------#
#                                                                                               #
#   I M P O R T     L O C A L     L I B R A R I E S   /   F I L E S                             #
#                                                                                               #
#-----------------------------------------------------------------------------------------------#
from modules.train_man import TrainingManager

#***********************************************************************************************#
#                                                                                               #
#   description:                                                                                #
#   class that implements the logic for Federated Worker nodes.                                 #
#                                                                                               #
#***********************************************************************************************#
class FederatedWorker(WebsocketServerWorker):
    def __init__(
        self,
        hook,
        host: str,
        port: int,
        id: Union[int, str] = 0,
        log_msgs: bool = False,
        verbose: bool = False,
        data: List[Union[torch.Tensor, AbstractTensor]] = None,
        loop=None,
        cert_path: str = None,
        key_path: str = None,
        datasets = None,
        models = None,
    ):
        """This is a simple extension to normal workers wherein
        all messages are passed over websockets. Note that because
        BaseWorker assumes a request/response paradigm, this worker
        enforces this paradigm by default.
        Args:
            hook (sy.TorchHook): a normal TorchHook object
            id (str or id): the unique id of the worker (string or int)
            log_msgs (bool): whether or not all messages should be
                saved locally for later inspection.
            verbose (bool): a verbose option - will print all messages
                sent/received to stdout
            host (str): the host on which the server should be run
            port (int): the port on which the server should be run
            data (dict): any initial tensors the server should be
                initialized with (such as datasets)
            loop: the asyncio event loop if you want to pass one in
                yourself
            cert_path: path to used secure certificate, only needed for secure connections
            key_path: path to secure key, only needed for secure connections
        """
        
        # create a train manager instance
        self.train_manager = TrainingManager(self, datasets, models)

        # call WebsocketServerWorker constructor
        super().__init__(hook=hook, 
                         host=host, 
                         port=port, 
                         id=id, 
                         data=None, 
                         log_msgs=log_msgs, 
                         verbose=verbose, 
                         loop=loop, 
                         cert_path=cert_path,
                         key_path=key_path,
                        )
    
    def set_train_config(self, **kwargs):
        """Set the training configuration in to the trainconfig object
        Args:
            **kwargs:
                add arguments here
        """
        self.train_manager.setup_configurations(kwargs)
        return "SUCCESS"

    def fit(self, dataset_key: str, epoch: int, device: str = "cpu", **kwargs):
        """Fits a model on the local dataset as specified in the local TrainConfig object.
        Args:
            dataset_key: Identifier of the local dataset that shall be used for training.
            **kwargs: Unused.
        Returns:
            loss: Training loss on the last batch of training data.
        """
        print("Fitting model on worker {0}".format(self.id))
        start_fit_time = timer()
        # get model and it's respective parameters
        model = self.train_manager.get_global_model()
        
        # get criterion and optimizer instances
        criterion = self.train_manager.get_criterion()
        optimizer = self.train_manager.get_optimizer(model)
        
        # get next set of batches to train on
        data_batches = self.train_manager.next_batches(dataset_key=dataset_key)
        
        # local variables for training
        losses = []
        
        # starting training on all batches (need to modify this later to sample)
        for batch_idx, (input, target) in enumerate(data_batches):
            input = input.view(self.train_manager.batch_size, -1)
            # compute output
            output = model(input)
            # clear any previous buffers
            optimizer.zero_grad()
            # compute gradients in a backward pass
            loss = criterion(output, target)
            loss.backward()
            # call step of optimizer to update model params
            optimizer.step()
            # update local stores
            losses.append(loss)
        
        end_fit_time =timer()
        print(f'Time to train 1 epoch on worker XXX')
        # store all the results so that they can requested back by the server
        self.train_manager.store_training_results(model, losses)
        
        return None
