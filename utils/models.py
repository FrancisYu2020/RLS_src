import torch.nn as nn
import torch
import torchvision.models as models
from utils.resnet3d import *
import os

# 10: is just a placeholder for class inheritence
resnet_2d_models = {10: models.resnet18, 18: models.resnet18, 34: models.resnet34}

def get_model(architecture_name, num_classes, window_size):
    '''
    create new model to train
    '''
    if 'resnet' in architecture_name:
        dimension, architecture = architecture_name.split('-')
        if dimension[0] == '2':
            return RLS2DModel(num_classes, int(architecture[-2:]), window_size=window_size)
        else:
            return RLS3DModel(num_classes, int(architecture[-2:]), window_size=window_size)
    else:
        raise NotImplementedError("ViT model part not implemented!")

def load_model(checkpoint_path, num_classes, window_size):
    '''
    load existing model to evaluate
    checkpoint_path: the experiment name of the model
    '''
    architecture_name = checkpoint_path.split('/')[1].split('_')[-1]
    model = get_model(architecture_name, num_classes, window_size)
    model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
    return model

# currently used
class RLS2DModel(nn.Module):
    def __init__(self, num_classes, layers=18, window_size=16):
        super().__init__()
        self.layers = layers
        self.conv = nn.Conv2d(window_size, 3, 3, 3, 1)
        torch.nn.init.kaiming_normal_(self.conv.weight, a=0, mode='fan_out')
        self.resnet = resnet_2d_models[layers](pretrained=False)
        self.classification_head = nn.Sequential(
            nn.Linear(self.resnet.fc.in_features, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
        self.regression_head = nn.Sequential(
            nn.Linear(self.resnet.fc.in_features, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
        self._init_modules()
    
    def _init_modules(self):
        # initialize conv blocks
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                    
    def forward(self, x):
        x = self.conv(x)
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.resnet.avgpool(x)
        x = x.flatten(start_dim=1)
        
        return self.classification_head(x), self.regression_head(x)
    
class RLS3DModel(RLS2DModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv = nn.Sequential(
            nn.Conv3d(1, 3, 7, 1, 3),
#             nn.BatchNorm3d(3),
#             nn.ReLU()
        )
        self.resnet = generate_model(self.layers)
        self._init_modules()
    
class RLSViTModel(nn.Module):
    def __init__(self):
        super().__init__()
        raise NotImplementedError("ViT model for RLS project is not implemented yet!")