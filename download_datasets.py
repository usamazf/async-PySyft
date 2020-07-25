import torch
import torchvision
import argparse
import torchvision.transforms as transforms
import torchvision.datasets as dt

#***********************************************************************************************#
#                                                                                               #
#   description:                                                                                #
#   Download CIFAR10 Dataset and transform the images                                           #
#                                                                                               #
#***********************************************************************************************#


def download_cifar10():
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    dt.CIFAR10(root='./data/cifar_10/', train=True,download=True, transform=transform)
    dt.CIFAR10(root='./data/cifar_10/', train=False,download=True, transform=transform)


#***********************************************************************************************#
#                                                                                               #
#   description:                                                                                #
#   Download MNIST Dataset and transform the images                                             #
#                                                                                               #
#***********************************************************************************************#

def download_mnist():
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.1307,))
                                   ])

    dt.MNIST(root='./data/mnist', train=True,download=True, transform=transform)
    dt.MNIST(root='./data/mnist', train=False,download=True, transform=transform)


#***********************************************************************************************#
#                                                                                               #
#   description:                                                                                #
#   run download_dataset.py in the terminal and set the --dataset to either "mnist" or "cifar10"#
#                                                                                               #
#***********************************************************************************************#


if __name__ == "__main__":
      # parse the arguments
    parser = argparse.ArgumentParser(description="Download Dataset from torchvision")
    parser.add_argument("--dataset", type=str, help="mnist or cifar10", required=True)
    args = parser.parse_args()
    if args.dataset == "mnist":
        print("Downloading MNIST ...")
        download_mnist()
    if args.dataset == "cifar10":
        print("Downloading CIFAR10")
        download_cifar10()
