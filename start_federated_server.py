#-----------------------------------------------------------------------------------------------#
#                                                                                               #
#   I M P O R T     G L O B A L     L I B R A R I E S                                           #
#                                                                                               #
#-----------------------------------------------------------------------------------------------#
import asyncio
import websockets
import argparse
import time
from timeit import default_timer as timer

import torch
import torch.nn as nn
import torchvision

import syft as sy
# this hook is needed before the training_plan library import
hook = sy.TorchHook(torch)

#-----------------------------------------------------------------------------------------------#
#                                                                                               #
#   I M P O R T     L O C A L     L I B R A R I E S   /   F I L E S                             #
#                                                                                               #
#-----------------------------------------------------------------------------------------------#
from workers import FederatedWorkerPointer
from modules.model_loader import get_model
from modules.data_loader import load_dataset
from modules.validate import validate
#from modules.training_plan import build_and_get_train_plan
from utils.utils import average_model_parameters, model_flatten, model_unflatten
from configs import globals as glb

#-----------------------------------------------------------------------------------------------#
#                                                                                               #
#   Define global parameters.                                                                   #
#                                                                                               #
#-----------------------------------------------------------------------------------------------#
WORKER_LIST = []
    
#***********************************************************************************************#
#                                                                                               #
#   description:                                                                                #
#   helper fucntions to communicate with client worker and their information.                   #
#                                                                                               #
#***********************************************************************************************#
async def connection_handler(websocket, path):
    # receive information of the worker
    worker_id = await websocket.recv()
    worker_host = await websocket.recv()
    worker_port = await websocket.recv()
    # print log message
    print("connection received from client {0}!!!!".format(worker_id))
    # setup arguments
    kwargs_websocket = {"host": worker_host, "hook": hook, "verbose": True}
    time.sleep(5)
    # create new instance of the websocket server object
    remote_client_ptr = FederatedWorkerPointer(id=worker_id, port=int(worker_port), **kwargs_websocket)
    # update the local dictionary
    WORKER_LIST.append([remote_client_ptr, worker_id, worker_host, int(worker_port)])

#***********************************************************************************************#
#                                                                                               #
#   description:                                                                                #
#   helper fucntions to communicate with client worker.                                         #
#                                                                                               #
#***********************************************************************************************#
async def fit_model_on_worker(worker_ptr: FederatedWorkerPointer, model_params, train_plan, dataset_key, iteration, sampled_id, kwargs):
    """Send the model to the worker and fit the model on the worker's training data.
    Args:
        worker_ptr: Remote location, where the model shall be trained.
        model: Batch size of each training step.
        train_plan: Model which shall be trained.
        iteration: current iteration being run
    Returns:
        A tuple containing:
            * worker_id: Union[int, str], id of the worker.
            * updated_parameter: parameters of the improved model.
            * loss: Loss on last training batch, torch.tensor.
    """
    # clear all remote objects
    worker_ptr.clear_objects_remote()
    
    # setup sampled worker id to kwargs
    kwargs["sampled_worker_id"] = sampled_id
    
    # send the fresh model parameters
    model_params_copy = model_params.clone().detach()
    model_params_copy.id = kwargs["model_param_id"]
    ptr_model = model_params_copy.send(worker_ptr)
    
    # set train configurations on the remote worker
    await worker_ptr.set_train_config(**kwargs)
    
    # run the async fit method and fetch results
    task_object = worker_ptr.async_fit(dataset_key=dataset_key, iteration=iteration, return_ids=[kwargs["result_losses_id"], kwargs["result_differ_id"]])
    loss, worker_update = await task_object
    
    # return results    
    return worker_ptr.id, loss, worker_update

#***********************************************************************************************#
#                                                                                               #
#   description:                                                                                #
#   helper fucntions to build arguments dictionary for training configurations.                 #
#                                                                                               #
#***********************************************************************************************#
def build_training_configurations():
    # create an arguments dictionary
    kwargs = dict()
    kwargs["plan_id"] = glb.PLAN_ID
    kwargs["model_id"] = glb.MODEL
    kwargs["model_param_id"] = glb.MODEL_PARAM_ID
    kwargs["lr"] = glb.INITIAL_LR
    kwargs["batch_size"] = glb.BATCH_SIZE
    kwargs["random_sample"] = glb.RANDOM_SAMPLE_BATCHES
    kwargs["max_nr_batches"] = glb.MAX_NR_BATCHES
    kwargs["dataset_key"] = glb.DATASET_ID
    #kwargs["iterations"] = glb.NUM_ITERS
    kwargs["criterion"] = glb.CRITERION
    kwargs["optimizer"] = glb.OPTIMIZER
    kwargs["diff_privacy"] = glb.USE_DP
    kwargs["result_params_id"] = "result_param"
    kwargs["result_differ_id"] = "result_diff"
    kwargs["result_losses_id"] = "result_loss"

    return kwargs

#***********************************************************************************************#
#                                                                                               #
#   description:                                                                                #
#   helper fucntions to communicate with client worker.                                         #
#                                                                                               #
#***********************************************************************************************#
async def training_handler():
    # yield control to connector class
    await asyncio.sleep(30)
    
    # build and get the train plan
    #train_plan  = build_and_get_train_plan()
    train_plan = None
    
    # create training arguments
    kwargs = build_training_configurations()
    
    # build model
    model = get_model(model_name=glb.MODEL)
    
    # get a loss function
    criterion = nn.CrossEntropyLoss()
    
    # load test dataset
    _, test_loader = load_dataset(dataset=glb.DATASET, loaders=True)
    
    # get some variable
    n_iterations = glb.NUM_ITERS
    
    start = timer()
    print(f"\nStarting the Training Process for {n_iterations} iterations\n")
    # iterate over the workers
    for curr_iter in range(n_iterations):
        # print log message
        print("\n\nRunning iteration {0} of {1}".format(curr_iter+1, n_iterations))
            
        # sample workers based on our logic here
        sampled_workers = [worker[0] for worker in WORKER_LIST] #[WORKER_LIST[0][0]]
        print("Sampled worker count: ", len(sampled_workers))
        
        # extract latest model parameters
        model_params = model_flatten(model)
        
        # run the training on all workers
        start_timer_iter = timer()
        results = await asyncio.gather(
            *[
                fit_model_on_worker(
                    worker_ptr=worker,
                    model_params=model_params,
                    train_plan=train_plan,
                    dataset_key=glb.DATASET_ID,
                    iteration=curr_iter,
                    sampled_id=idx,
                    kwargs=kwargs,
                )
                for idx, worker in enumerate(sampled_workers)
            ])
        end_timer_iter = timer()

        print(f"Iteration: {curr_iter}\nTime to train and await gradients for {len(sampled_workers)} workers: {(end_timer_iter-start_timer_iter):3f}s")

        
        # extract from all workers the updated model parameters
        upd_wrkr = {}
        for worker_id, loss, recvd_update in results:
            upd_wrkr[worker_id] = recvd_update
        
        # get the parameter average
        avgd_update = average_model_parameters(upd_wrkr)
        
        # unpack the new parameters into local model
        model_params.add_(avgd_update)
        model_unflatten(model, model_params)
        
        # evaluate on testset using the new model
        print("Begin Validation @ Iteration {}".format(curr_iter+1))
        val_loss, prec1 = validate(test_loader, model, criterion)
        
    #while True:
    #   continue
    end = timer()
    print(f"Total Training Time for {n_iterations} iterations: {(end-start):3f} seconds")

#***********************************************************************************************#
#                                                                                               #
#   description:                                                                                #
#   argument parsing and configurations for setting up the federated server.                    #
#                                                                                               #
#***********************************************************************************************#
if __name__ == '__main__':
    # parse the arguments
    parser = argparse.ArgumentParser(description="Setup Federated Server Module.")
    parser.add_argument("--port", type=int, help="port number of federated server, e.g. --port 8778", required=True)
    parser.add_argument("--host", type=str, default="localhost", help="host for the connection")
    args = parser.parse_args()
    
    # listen on the listen_port to connect new client
    start_server = websockets.serve(connection_handler, args.host, args.port)
    
    # run forever
    print("REACHED THIS POINT, NOW WAITING FOR WORKER CONNECTION")
    
    # create a forever running event loop
    asyncio.get_event_loop().run_until_complete(start_server)
    asyncio.ensure_future(training_handler())
    asyncio.get_event_loop().run_forever()
