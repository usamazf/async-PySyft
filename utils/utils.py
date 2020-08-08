#-----------------------------------------------------------------------------------------------#
#                                                                                               #
#   I M P O R T     G L O B A L     L I B R A R I E S                                           #
#                                                                                               #
#-----------------------------------------------------------------------------------------------#
import torch
import syft as sy

#***********************************************************************************************#
#                                                                                               #
#   Description:                                                                                #
#   flatten and unflatten the passed model to be used for synchronization.                      #
#                                                                                               #
#***********************************************************************************************#
def model_flatten(model):
    vec = []
    for param in model.parameters():
        vec.append(param.data.view(-1))
    return torch.cat(vec)

def model_unflatten(model, vec):
    pointer = 0
    for param in model.parameters():
        num_param = torch.prod(torch.LongTensor(list(param.size())))
        param.data = vec[pointer:pointer + num_param].view(param.size())
        pointer += num_param

#***********************************************************************************************#
#                                                                                               #
#   Description:                                                                                #
#   flatten and unflatten the gradients of the model or into the model.                         #
#                                                                                               #
#***********************************************************************************************#
def model_grad_flatten(model):
    vec = []
    for param in model.parameters():
        vec.append(param.grad.view(-1))
    return torch.cat(vec)

def model_grad_unflatten(model, vec):
    pointer = 0
    for param in model.parameters():
        num_param = torch.prod(torch.LongTensor(list(param.size())))
        param.grad = vec[pointer:pointer + num_param].view(param.size())
        pointer += num_param
        
#***********************************************************************************************#
#                                                                                               #
#   description:                                                                                #
#   utility class to help assist the overall program with averaging, sum etc.                   #
#                                                                                               #
#***********************************************************************************************#
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

#***********************************************************************************************#
#                                                                                               #
#   Description:                                                                                #
#   utility function to split the given dataset.                                                #
#                                                                                               #
#***********************************************************************************************#
def split_dataset_and_return_mine(dataset, rank, world_size, split_by_target=False):
    """Utility function to split given dataset and return local worker's share
    """
    final_dataset = None
    # split data based on selected criteria
    if not split_by_target:
        total_samples = len(dataset)
        samples_per_worker = int(total_samples//world_size)
        # get starting and ending indices
        start_index = rank * samples_per_worker
        final_index = (rank+1) * samples_per_worker - 1
        # sample the appropriate portion of data
        selected_data = dataset.data[start_index:final_index]
        selected_targets = dataset.targets[start_index:final_index]
        # create new dataset
        final_dataset = sy.BaseDataset(data=selected_data, 
                                       targets=selected_targets,
                                       transform=dataset.transform)
    else:
        # need to implemenet logic here
        pass
    
    return final_dataset

#***********************************************************************************************#
#                                                                                               #
#   Description:                                                                                #
#   utility function to split the given dataset.                                                #
#                                                                                               #
#***********************************************************************************************#
def scale_model_parameters(model_params, scale_factor):
    pass

#***********************************************************************************************#
#                                                                                               #
#   Description:                                                                                #
#   utility function to add two model parameter lists.                                          #
#                                                                                               #
#***********************************************************************************************#
def add_model_parameters(dst_model_params, src_model_params):
    pass

#***********************************************************************************************#
#                                                                                               #
#   Description:                                                                                #
#   utility function to average a dictionary of models.                                         #
#                                                                                               #
#***********************************************************************************************#
def average_model_parameters(model_params: dict):
    """Calculate the average of a dictionary containing model parameters.
    Args:
        models (Dict): a dictionary of model parameters for which the 
        average is calculated.
    Returns:
        List: the list of averaged parameters.
    """
    nr_models = len(model_params)
    model_list = list(model_params.values())
    avg_model = model_list[0]
    
    # add all models
    for i in range(1, nr_models):
        avg_model += model_list[i]
        
    # scale the summed up models
    avg_model *= (1.0 / nr_models)
    
    # returnt the averaged model
    return avg_model
