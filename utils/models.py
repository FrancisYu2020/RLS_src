import torch.nn as nn
import torch
import torchvision.models as models
import os
from utils.resnet3d import *

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
    elif architecture_name == "2d-conv":
        return RLSConv2D(num_classes, window_size=window_size)
    elif architecture_name == '3d-conv':
        return RLSConv3D(num_classes, window_size=window_size)
    elif architecture_name == '2d-linear':
        return Linear(window_size)
    elif architecture_name == '2d-positional':
        return PositionalMLP(window_size)
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

class Linear(nn.Module):
    def __init__(self, window_size):
        super().__init__()
        self.linear = nn.Linear(window_size * 256, 2)
    
    def forward(self, x):
        return self.linear(x.flatten(start_dim=1))

class MLP(nn.Module):
    def __init__(self, in_features, out_features, hidden_features=64):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, out_features)
        )
    
    def forward(self, x):
        return self.mlp(x)

class PositionalEncoding(nn.Module):
    # modified from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    def __init__(self, max_len: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, flattened_mat_len, window_size]``
        """
        x = x + self.pe
        return self.dropout(x)
    
class PositionalMLP(nn.Module):
    def __init__(self, window_size):
        super().__init__()
        self.pe1 = PositionalEncoding(256, window_size)
        self.pe2 = PositionalEncoding(1, 256)
        self.mlp1 = nn.Sequential(
            MLP(window_size, window_size // 2, hidden_features=window_size),
            MLP(window_size // 2, 1)
        )
        self.mlp2 = nn.Sequential(
            MLP(256, 64, hidden_features=256),
            MLP(64, 2)
        )
    
    def forward(self, x):
        '''
        Arguments:
            x: Tensor, shape ``[batch_size, window_size, mat_size, mat_size]``
        '''
        N, W, M, _ = x.shape
        x = x.flatten(start_dim=2).transpose(1, 2)
        x = self.pe1(x)
        x = self.mlp1(x).transpose(1, 2)
        return self.mlp2(self.pe2(x)).squeeze(-1)
        
# try simple cnn 2d
class RLSConv2D(nn.Module):
    def __init__(self, num_classes, window_size=16, initial_temperature=1.0):
        super().__init__()
        self.window_size = window_size
        self.conv = nn.Sequential(
            nn.Conv2d(window_size, 2 * window_size, 4, 2),
            nn.BatchNorm2d(2 * window_size),
            nn.ReLU(),
            nn.Conv2d(2 * window_size, 4 * window_size, 4, 2),
            nn.BatchNorm2d(4 * window_size),
            nn.ReLU()
        )
        self.temperature = nn.Parameter(torch.ones(1) * initial_temperature)
        self.conv_out = self._get_conv_out_size()
        self.num_classes = num_classes
        self.classification_head = MLP(self.conv_out, self.num_classes)
        self.regression_head = MLP(self.conv_out, self.num_classes)
        self._init_modules()
    
    def _get_conv_out_size(self):
        x = torch.randn(1, self.window_size, 16, 16)
        return self.conv(x).view(-1).size(0)
    
    def _init_modules(self):
        # initialize conv blocks
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                    
    def forward(self, x):
        x = self.conv(x).flatten(start_dim=1)
        return self.classification_head(x) / self.temperature
#         return self.classification_head(x), self.regression_head(x)

# try simple cnn 3d
class RLSConv3D(RLSConv2D):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv = nn.Sequential(
            nn.Conv3d(1, 16, (40, 4, 4), (8, 2, 2)),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.Conv3d(16, 32, (7, 4, 4), (4, 2, 2)),
            nn.BatchNorm3d(32),
            nn.ReLU()
        )
        self.conv_out = self._get_conv3d_out_size()
        self.classification_head = MLP(self.conv_out, self.num_classes)
        self.regression_head = MLP(self.conv_out, self.num_classes)
        self._init_modules()
    
    def _get_conv3d_out_size(self):
        x = torch.randn(1, 1, self.window_size, 16, 16)
        return self.conv(x).view(-1).size(0)
    
# currently used
class RLS2DModel(nn.Module):
    def __init__(self, num_classes, layers=18, window_size=16, initial_temperature=1.0):
        super().__init__()
        self.layers = layers
#         self.conv = nn.Conv2d(window_size, 3, 3, 3, 1)
#         torch.nn.init.kaiming_normal_(self.conv.weight, a=0, mode='fan_out')
        self.resnet = resnet_2d_models[layers]()
        self.resnet.conv1 = nn.Conv2d(window_size, 64, 7, 2, 3, bias=False)
        self.classification_head = MLP(self.resnet.fc.in_features, num_classes)
        self.regression_head = MLP(self.resnet.fc.in_features, num_classes)
        self.temperature = nn.Parameter(torch.ones(1) * initial_temperature)
        self._init_modules()
        self.resnet.conv1.to(torch.device('cuda'))
    
    def _init_modules(self):
        # initialize conv blocks
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                    
    def forward(self, x):
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
        
        return self.classification_head(x) / self.temperature
#         return self.classification_head(x), self.regression_head(x)
        
class RLS3DModel(RLS2DModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
#         self.conv = nn.Sequential(
#             nn.Conv3d(1, 3, 7, 1, 3),
# #             nn.BatchNorm3d(3),
# #             nn.ReLU()
#         )
        self.resnet = generate_model(self.layers)
        self.resnet.conv1 = nn.Sequential(
            nn.Conv3d(1, 64, 7, 1, 3)
        )
        self._init_modules()
    
class RLSViTModel(nn.Module):
    def __init__(self):
        super().__init__()
        raise NotImplementedError("ViT model for RLS project is not implemented yet!")