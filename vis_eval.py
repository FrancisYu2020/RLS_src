import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingLR
import argparse
import os
import pandas as pd
from sklearn.metrics import confusion_matrix
import timm
from tqdm import tqdm
from utils.models import *
from utils.dataset import *
from utils.metrics import *
from train import *
from utils.vis_utils import plot_validation, inference_and_plot
from utils import get_checkpoint_path, get_val_checkpoint_dir

parser = argparse.ArgumentParser()
parser.add_argument("--lr", default=0.01, help="learning rate used to train the model", type=float)
parser.add_argument("--weight_decay", default=0.1, help="weight decay used to train the model", type=float)
parser.add_argument("--epochs", default=300, help="epochs used to train the model", type=int)
parser.add_argument("--batch_size", default=256, help="batch size used to train the model", type=int)
parser.add_argument("--num_classes", default=2, help="number of classes for the classifier", type=int)
parser.add_argument("--clip_len", default=16, help="window size of the input data", type=int)
parser.add_argument("--input_size", default=[16, 16], nargs="+", help="window size of the input data", type=int)
parser.add_argument("--cross_val_type", default=0, type=int, help="0 for train all val all, 1 for leave patient 1 out")
parser.add_argument("--task", default="classification", type=str, help="indicate what kind of task to be run (regression/classification, etc.)")
parser.add_argument("--alpha", default=0.5, type=float, help="weight of cls loss, default 0.5")
parser.add_argument("--architecture", default="3d-resnet18", choices=["2d-positional", "2d-linear", "3d-conv", "2d-conv", "3d-resnet10", "3d-resnet18", "2d-resnet18", "ViT-tiny"], help="architecture used")
parser.add_argument("--exp_id", default=0, type=str, help="the id of the experiment")
parser.add_argument("--normalize_data", default=0, type=int, help="0 for raw mat input, 1 for normalized to [0, 1] and standardization")
parser.add_argument("--patients", default=[15], type=int, nargs="+", help="patient ids included in the training")
parser.add_argument("--seed", default=1, type=int, help="random seed for torch")
args = parser.parse_args()

args.exp_name = get_checkpoint_path(args)
args.checkpoint_root = os.path.join('checkpoint', args.exp_name)
args.figure_root = os.path.join('figures', args.exp_name)
if not os.path.exists(args.checkpoint_root):
    raise FileNotFoundError(f'{args.checkpoint_root} not found!')
if not os.path.exists(args.figure_root):
    os.mkdir(args.figure_root)
    
# args.exp_name += f'_{args.exp_id}'
args.checkpoint_dir = get_val_checkpoint_dir(args)
args.figure_dir = os.path.join(args.figure_root, str(args.seed))
# print(args.figure_dir)
# exit()
if not os.path.exists(args.checkpoint_dir):
    raise FileNotFoundError(f'{args.checkpoint_dir} not found!')
if not os.path.exists(args.figure_dir):
    os.mkdir(args.figure_dir)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args.device = device

# torch.manual_seed(3407)

# Hyperparameters
num_classes = args.num_classes  # Number of classes in ImageNet
num_epochs = args.epochs
batch_size = args.batch_size
learning_rate = args.lr

# load data
data_prefix = 'win' + str(args.clip_len) + '_'

# data_paths = ['data/patient03-12-12-2023', 'data/patient05-02-15-2024', 'data/patient06-02-17-2024', 'data/patient09-03-01-2024', 'data/patient11-03-15-2024']
# patient_ids = {0:3, 1:5, 2:6, 3:9, 4:11}
data_paths = ['data/patient15-04-12-2024']
patient_ids = {0:15}

H, W = args.input_size
for i in range(len(data_paths)):
    args.patient_id = patient_ids[i]
    leave_out_idx = i
    val_data = np.load(os.path.join(data_paths[leave_out_idx], data_prefix + 'tal_val_data.npy')).astype(np.float32)[:, :, -H:, -W:]
    val_label = np.load(os.path.join(data_paths[leave_out_idx], data_prefix + 'tal_val_label.npy'))
    val_data = np.log(1 + val_data)
    if args.normalize_data:
        val_data /= np.log(4096)

    val_label = val_label.astype(np.int_)
    print(val_data.shape, val_label.shape, val_label.sum())

    # original CNN transformation
    val_transform = get_cnn_transforms(val_data.shape[1], train=False)

    # create dataset and dataloader
    val_dataset = RLSDataset(args.architecture, val_data, val_label, transform=val_transform)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
            
    # main eval code
    # Initialize the model
    inference_and_plot(args, val_loader, 'f1')
    inference_and_plot(args, val_loader, 'f0.5')
    inference_and_plot(args, val_loader, 'miou')