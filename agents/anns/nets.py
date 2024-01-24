import torch
from torch import nn
from torchvision import models


def get_torch_model(model_name, num_classes):
    model = None
    if model_name == 'resnet18':
        model = models.resnet18(weights=None)
        in_channels = 4
        model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.fc = nn.Linear(in_features=512, out_features=num_classes, bias=True)
    if model_name == 'point_net':
        # model = 
        # change the output
        # model.classifier[1] = nn.Linear(in_features=96, out_features=num_classes, bias=True)
        print(" Antonio: Add here a new model of a net for Point Clouds ")

    return model


def load_model(model_path):
    try:
        return torch.load(model_path).eval()
    except:
        return torch.jit.load(model_path).eval()
