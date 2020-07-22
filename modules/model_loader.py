#-----------------------------------------------------------------------------------------------#
#                                                                                               #
#   I M P O R T     L I B R A R I E S                                                           #
#                                                                                               #
#-----------------------------------------------------------------------------------------------#
import torch
import torch.nn as nn
import torch.nn.functional as F

import syft as sy

#-----------------------------------------------------------------------------------------------#
#                                                                                               #
#   Define global parameters to be used through out the program                                 #
#                                                                                               #
#-----------------------------------------------------------------------------------------------#
cfg = { # Configuration for larger models
'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

#***********************************************************************************************#
#                                                                                               #
#   description:                                                                                #
#   create and return the desired model.                                                        #
#                                                                                               #
#***********************************************************************************************#
def get_model(model_name, num_classes=10):
    # check which model is desired and return it
    if model_name == "small":
        return Model()
    elif model_name == "mnist-small":
        return Net()
    elif model_name == "vgg-13":
        return vgg13_model(nClasses=num_classes)
    elif model_name == "vgg-16":
        return vgg16_model(nClasses=num_classes)
    elif model_name == "vgg-19":
        return vgg19_model(nClasses=num_classes)

#***********************************************************************************************#
#                                                                                               #
#   description:                                                                                #
#   create and return the desired model.                                                        #
#                                                                                               #
#***********************************************************************************************#
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(3,6,5)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6,16,5)
        self.fc1 = nn.Linear(16*5*5,120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84,10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

#***********************************************************************************************#
#                                                                                               #
#   description:                                                                                #
#   create and return the desired model.                                                        #
#                                                                                               #
#***********************************************************************************************#
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        #x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

#***********************************************************************************************#
#                                                                                               #
#   description:                                                                                #
#   class to create custom vgg models instead of using torchvision.                             #
#                                                                                               #
#***********************************************************************************************#
class VGG(nn.Module):
    def __init__(self, num_classes, vgg_name="VGG19"):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512*7*7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        #layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

#***********************************************************************************************#
#                                                                                               #
#   description:                                                                                #
#   create vgg-13 model and return it.                                                          #
#                                                                                               #
#***********************************************************************************************#
def vgg13_model(nClasses):
    # Constructs a VGG-13 model.
    model = VGG(num_classes=nClasses, vgg_name="VGG13")
    return model

#***********************************************************************************************#
#                                                                                               #
#   description:                                                                                #
#   create vgg-16 model and return it.                                                          #
#                                                                                               #
#***********************************************************************************************#
def vgg16_model(nClasses):
    # Constructs a VGG-16 model.
    model = VGG(num_classes=nClasses, vgg_name="VGG16")
    return model

#***********************************************************************************************#
#                                                                                               #
#   description:                                                                                #
#   create vgg-19 model and return it.                                                          #
#                                                                                               #
#***********************************************************************************************#
def vgg19_model(nClasses):
    # Constructs a VGG-19 model.
    model = VGG(num_classes=nClasses, vgg_name="VGG19")
    return model
