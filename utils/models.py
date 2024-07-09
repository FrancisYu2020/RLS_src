import torch.nn as nn
import torch
import torchvision.models as models
import os
from utils.resnet3d import *
import timm

# 10: is just a placeholder for class inheritence
resnet_2d_models = {10: models.resnet18, 18: models.resnet18, 34: models.resnet34}

def get_model(args):
    '''
    create new model to train
    '''
    architecture_name = args.architecture
    clip_len = args.clip_len
    print(architecture_name)
    if args.data_type == 'context':
        num_classes = 1
    else:
        num_classes = clip_len
    if 'resnet' in architecture_name:
        dimension, architecture = architecture_name.split('-')
        if dimension[0] == '2':
            return RLS2DModel(num_classes, int(architecture[-2:]), clip_len=clip_len, num_classes=num_classes)
        else:
            return RLS3DModel(num_classes, int(architecture[-2:]), clip_len=clip_len, num_classes=num_classes)
    elif architecture_name == "2d-conv":
        return RLSConv2D(args, clip_len=clip_len, num_classes=num_classes)
    elif architecture_name == '3d-conv':
        return RLSConv3D(args, clip_len=clip_len, num_classes=num_classes)
    elif architecture_name == '2d-linear':
        return Linear(clip_len, args.input_size)
    elif architecture_name == '2d-positional':
        return PositionalMLP(clip_len, args.input_size)
    elif architecture_name == '2d-efficientnet':
        return EfficientNet(args, "efficientnet_b0.ra_in1k", num_classes)
    else:
        raise NotImplementedError("ViT model part not implemented!")

def load_model(args, ckpt_name):
    '''
    load existing model to evaluate
    checkpoint_path: the experiment name of the model
    '''
    checkpoint_path = os.path.join(args.checkpoint_dir, ckpt_name)
    architecture_name = checkpoint_path.split('/')[1].split('_')[-1]
    model = get_model(args, args.clip_len)
    model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
    return model

class Squeeze(nn.Module):
    def __init__(self):
        super(Squeeze, self).__init__()
        pass

    def forward(self, x):
        return x.squeeze()
    
class EfficientNet(nn.Module):
    def __init__(self, args, backbone, num_classes, dropout=0.1, pretrained=False, seq_len=32):
        super().__init__()
        assert args.batch_size >= seq_len, "batch size must be > n_heads"
        assert args.batch_size % seq_len == 0, "batch size must be multiple of n_heads"
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 3, 1, 1),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(3, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.input_size = args.input_size
        self.batch_size = args.batch_size // seq_len
        self.seq_len = seq_len
        hdim = self.get_conv_feature_dim()
        
        self.encoder = timm.create_model(
            backbone,
            num_classes=num_classes,
            features_only=False,
            drop_rate=dropout,
            drop_path_rate=0,
            pretrained=pretrained
        )
        
        if 'efficient' in backbone:
            hdim = self.encoder.conv_head.out_channels
            self.encoder.classifier = nn.Identity()
        elif 'convnext' in backbone:
            hdim = self.encoder.head.fc.in_features
            self.encoder.head.fc = nn.Identity()
            
        lstm_hidden = 64
        self.lstm = nn.LSTM(hdim, lstm_hidden, num_layers=2, dropout=dropout, bidirectional=True, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(2 * lstm_hidden, 256),
#             Squeeze(),
#             nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.LeakyReLU(0.1),
            # nn.ReLU(),
            nn.Linear(256, 1),
        )
    
    def get_conv_feature_dim(self):
        d, u, l, r = self.input_size
        dummy_input = torch.randn(1, 3, r - l, u - d)
        print(f"dummy input size {dummy_input.shape}")
        feat = self.encoder(dummy_input)
        return feat.shape[-1]
        
    def forward(self, x):
#         print(x.shape)
#         exit()
        feat = self.encoder(x).flatten(start_dim=1)
        feat = feat.reshape(self.batch_size, self.seq_len, -1)
#         feat = self.encoder(x).unsqueeze(0)
        feat, _ = self.lstm(feat)
        feat = self.head(feat)
        return feat.flatten()
        
        
class Linear(nn.Module):
    def __init__(self, clip_len, input_size):
        super().__init__()
        H1, H2, W1, W2 = input_size
        H = H2 - H1
        W = W2 - W1
        self.linear = nn.Linear(clip_len * H * W, clip_len)
    
    def forward(self, x):
        return self.linear(x.flatten(start_dim=1))

class MLP(nn.Module):
    def __init__(self, in_features, out_features, hidden_features=256):
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
            x: Tensor, shape ``[batch_size, flattened_mat_len, clip_len]``
        """
        x = x + self.pe
        return self.dropout(x)
    
class PositionalMLP(nn.Module):
    def __init__(self, clip_len, input_size):
        super().__init__()
        H1, H2, W1, W2 = input_size
        H = H2 - H1
        W = W2 - W1
        self.pe1 = PositionalEncoding(H * W, clip_len)
        self.pe2 = PositionalEncoding(1, H * W)
        self.mlp1 = nn.Sequential(
            MLP(clip_len, clip_len // 2, hidden_features=clip_len),
            MLP(clip_len // 2, 1)
        )
        self.mlp2 = nn.Sequential(
            MLP(H * W, 64, hidden_features=H * W),
            MLP(64, clip_len)
        )
    
    def forward(self, x):
        '''
        Arguments:
            x: Tensor, shape ``[batch_size, clip_len, mat_size, mat_size]``
        '''
        N, W, M, _ = x.shape
        x = x.flatten(start_dim=2).transpose(1, 2)
        x = self.pe1(x)
        x = self.mlp1(x).transpose(1, 2)
        return self.mlp2(self.pe2(x)).squeeze()
        
# try simple cnn 2d
class RLSConv2D(nn.Module):
    def __init__(self, args, clip_len=16, initial_temperature=1.0, num_classes=1):
        super().__init__()
        self.clip_len = clip_len
        self.conv = nn.Sequential(
            nn.Conv2d(clip_len, 2 * clip_len, 3, 1, 1),
            nn.BatchNorm2d(2 * clip_len),
            nn.ReLU(),
            nn.Conv2d(2 * clip_len, 4 * clip_len, 3, 1, 1),
            nn.BatchNorm2d(4 * clip_len),
            nn.ReLU(),
            nn.AvgPool2d(2)
        )
#         self.conv = nn.Sequential(
#             nn.Conv2d(3, 16, 3, 1, 1),
#             nn.BatchNorm2d(16),
#             nn.ReLU(),
#             nn.MaxPool2d(2),
#             nn.Conv2d(16, 32, 3, 1, 1)
#         )
        self.temperature = nn.Parameter(torch.ones(1) * initial_temperature)
        H1, H2, W1, W2 = args.input_size
        H = H2 - H1
        W = W2 - W1
        self.input_size = (H, W)
        self.conv_out = self._get_conv_out_size()
        self.classification_head = nn.Linear(self.conv_out, num_classes)
        self._init_modules()
    
    def _conv_forward(self, x):
        x = self.conv(x)
        return x
        
    def _get_conv_out_size(self):
        x = torch.randn(1, self.clip_len, *self.input_size)
#         x = torch.randn(1, 3, *self.input_size)
        return self._conv_forward(x).view(-1).size(0)
    
    def _init_modules(self):
        # initialize conv blocks
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                    
    def forward(self, x):
        x = self._conv_forward(x).flatten(start_dim=1)
        return self.classification_head(x)
#         return self.classification_head(x) / self.temperature

# try simple cnn 3d
class RLSConv3D(RLSConv2D):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv = nn.Sequential(
            nn.Conv3d(1, 16, (25, 3, 3), 1, 1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.AvgPool3d(2),
#             nn.Conv3d(16, 32, (10, 3, 3), 1, 1),
#             nn.BatchNorm3d(32),
#             nn.ReLU(),
#             nn.AvgPool3d(2),
#             nn.Conv3d(32, 64, 1, 1, 0),
#             nn.BatchNorm3d(64),
#             nn.ReLU()
        )
        self.conv_out = self._get_conv3d_out_size()
        self.classification_head = nn.Linear(self.conv_out, self.clip_len)
        self._init_modules()
    
    def _get_conv3d_out_size(self):
        x = torch.randn(1, 1, self.clip_len, *self.input_size)
        return self._conv_forward(x).view(-1).size(0)