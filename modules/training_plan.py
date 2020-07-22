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
from modules.model_loader import get_model
from configs import globals as glb

#***********************************************************************************************#
#                                                                                               #
#   description:                                                                                #
#   a fucntion to set current model parameters equal to that of the global model.               #
#                                                                                               #
#***********************************************************************************************#
def set_model_params(module, params_list, start_param_idx=0):
    """ Set params list into model recursively
    """
    param_idx = start_param_idx

    for name, param in module._parameters.items():
        module._parameters[name] = params_list[param_idx]
        param_idx += 1

    for name, child in module._modules.items():
        if child is not None:
            param_idx = set_model_params(child, params_list, param_idx)

    return param_idx

#***********************************************************************************************#
#                                                                                               #
#   description:                                                                                #
#   helper function to apply sgd step with given learning rate and parameters.                  #
#                                                                                               #
#***********************************************************************************************#
def naive_sgd(param, **kwargs):
    return param - kwargs['lr'] * param.grad

#***********************************************************************************************#
#                                                                                               #
#   description:                                                                                #
#   helper function to compute softmax cross entropy loss. Only a placeholder for now.          #
#                                                                                               #
#***********************************************************************************************#
def softmax_cross_entropy_with_logits(logits, targets, batch_size):
    """ Calculates softmax entropy
        Args:
            * logits: (NxC) outputs of dense layer
            * targets: (NxC) one-hot encoded labels
            * batch_size: value of N, temporarily required because Plan cannot trace .shape
    """
    # numstable logsoftmax
    norm_logits = logits - logits.max()
    log_probs = norm_logits - norm_logits.exp().sum(dim=1, keepdim=True).log()
    # NLL, reduction = mean
    return -(targets * log_probs).sum() / batch_size

#***********************************************************************************************#
#                                                                                               #
#   description:                                                                                #
#   helper function to build training plan with forward pass, loss and backward pass.           #
#                                                                                               #
#***********************************************************************************************#
@sy.func2plan()
def training_plan(inputs, targets, batch_size, lr, model_params):
    # inject params into the model
    set_model_params(model, model_params)
    
    # forward pass of the model
    logits = model.forward(inputs)
    
    # loss calculation
    loss = softmax_cross_entropy_with_logits(logits, targets, batch_size)
    
    # backpropagation to calculate gradients
    loss.backward()
    
    # update step
    updated_params = [
        naive_sgd(param, lr=lr) for param in model_params
    ]
    
    # computing accuracy 
    pred = torch.argmax(logits, dim=1)
    target = torch.argmax(targets, dim=1)
    acc = pred.eq(target).sum().float() / batch_size
    
    return (loss, acc, *updated_params)

#***********************************************************************************************#
#                                                                                               #
#   description:                                                                                #
#   actual function called to build and get the training plan, model etc.                       #
#                                                                                               #
#***********************************************************************************************#
def build_and_get_train_plan( ):
    # create an arguments dictionary
    kwargs = dict()
    kwargs["plan_id"] = glb.PLAN_ID
    kwargs["model_id"] = glb.MODEL
    kwargs["model_param_id"] = glb.MODEL_PARAM_ID
    kwargs["lr"] = glb.INITIAL_LR
    kwargs["batch_size"] = glb.BATCH_SIZE
    kwargs["max_nr_batches"] = glb.MAX_NR_BATCHES
    kwargs["dataset_key"] = glb.DATASET_ID
    kwargs["epochs"] = glb.NUM_EPOCHS
    
    # create a model
    global model
    model = get_model(model_name=kwargs["model_id"])
    #model.id = kwargs["model_id"]
    
    # dummy input parameters to make the trace
    model_params = [param.data for param in model.parameters()]

    # dummy data to make the trace
    X = torch.randn(3, 28 * 28)
    y = F.one_hot(torch.tensor([1, 2, 3]), 10)
    lr = torch.tensor([0.01])
    batch_size = torch.tensor([3.0])

    # build the actual training plan
    training_plan.build(X, y, batch_size, lr, model_params, trace_autograd=True)
    training_plan.id = kwargs["plan_id"]
    #training_plan = None
    
    # return the requested results
    return training_plan, kwargs
