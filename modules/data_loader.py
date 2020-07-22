#-----------------------------------------------------------------------------------------------#
#                                                                                               #
#   I M P O R T     G L O B A L     L I B R A R I E S                                           #
#                                                                                               #
#-----------------------------------------------------------------------------------------------#
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as dt

#-----------------------------------------------------------------------------------------------#
#                                                                                               #
#   Define global parameters to be used within this program                                     #
#                                                                                               #
#-----------------------------------------------------------------------------------------------#
WORKERS = 8

#***********************************************************************************************#
#                                                                                               #
#   description:                                                                                #
#   main program for initializing everything and setting up the training process.               #
#                                                                                               #
#***********************************************************************************************#
def load_dataset(dataset="mnist"):
    # load required dataset
    if dataset == "cifar-10":
        return CIFAR10_dataset( )
    elif dataset == "mnist":
        return MNIST_dataset( )

#***********************************************************************************************#
#                                                                                               #
#   description:                                                                                #
#   routine for loading cifar-10 dataset from the data directory.                               #
#                                                                                               #
#***********************************************************************************************#
def CIFAR10_dataset( ):
    
    # Define the transform for the data. Notice, we must resize to 224x224 with this dataset and model.
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # Initialize Datasets. CIFAR10 will automatically download if not present
    trainset = dt.CIFAR10(root='./data/cifar_10/', train=True,download=True, transform=transform)
    testset = dt.CIFAR10(root='./data/cifar_10/', train=False,download=True, transform=transform)

    # Create the Dataloaders to feed data to the training and validation steps
    #train_loader = sy.FederatedDataLoader(trainset.federate((v_workers[0],v_workers[1],v_workers[2],v_workers[3],v_workers[4])), batch_size=n_batch, shuffle=True)
    #test_loader = torch.utils.data.DataLoader(testset, batch_size=n_batch*20, shuffle=True, num_workers=WORKERS) 
    
    return trainset, testset

#***********************************************************************************************#
#                                                                                               #
#   description:                                                                                #
#   routine for loading mnist dataset from the data directory.                                  #
#                                                                                               #
#***********************************************************************************************#
def MNIST_dataset( ):
    
    # Define the transform for the data. Notice, we must resize to 224x224 with this dataset and model.
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.1307,))
                                   ])

    # Initialize Datasets. MNIST will automatically download if not present
    trainset = dt.MNIST(root='./data/mnist', train=True,download=True, transform=transform)
    testset = dt.MNIST(root='./data/mnist', train=False,download=True, transform=transform)
    
    return trainset, testset

