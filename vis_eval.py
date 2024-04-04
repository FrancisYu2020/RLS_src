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
from utils.vis_utils import plot_validation

parser = argparse.ArgumentParser()
parser.add_argument("--lr", default=0.01, help="learning rate used to train the model", type=float)
parser.add_argument("--weight_decay", default=0.1, help="weight decay used to train the model", type=float)
parser.add_argument("--epochs", default=300, help="epochs used to train the model", type=int)
parser.add_argument("--batch_size", default=256, help="batch size used to train the model", type=int)
parser.add_argument("--num_classes", default=1, help="number of classes for the classifier", type=int)
parser.add_argument("--window_size", default=16, help="window size of the input data", type=int)
parser.add_argument("--cross_val_type", default=0, type=int, help="0 for train all val all, 1 for leave patient 1 out")
parser.add_argument("--task", default="classification", type=str, help="indicate what kind of task to be run (regression/classification, etc.)")
parser.add_argument("--normalize_roi", default=1, type=int, help="whether normalize the roi indices between [0, 1]")
parser.add_argument("--alpha", default=0.5, type=float, help="weight of cls loss, default 0.5")
parser.add_argument("--architecture", default="3d-resnet18", choices=["2d-positional", "2d-linear", "3d-conv", "2d-conv", "3d-resnet10", "3d-resnet18", "2d-resnet18", "ViT-tiny"], help="architecture used")
parser.add_argument("--exp_id", default=0, type=str, help="the id of the experiment")
parser.add_argument("--normalize_data", default=0, type=int, help="0 for raw mat input, 1 for normalized to [0, 1] and standardization")
args = parser.parse_args()

args.exp_name = f"win{args.window_size}_epoch{args.epochs}_lr{args.lr}_wd{args.weight_decay}_bs{args.batch_size}_cv{args.cross_val_type}_nd{args.normalize_data}_{args.architecture}"
args.checkpoint_root = os.path.join('checkpoint', args.exp_name)
args.figure_root = os.path.join('figures', args.exp_name)
if not os.path.exists(args.checkpoint_root):
    raise FileNotFoundError(f'{args.checkpoint_root} not found!')
if not os.path.exists(args.figure_root):
    os.mkdir(args.figure_root)
    
args.exp_name += f'_{args.exp_id}'
args.checkpoint_dir = os.path.join(args.checkpoint_root, args.exp_id)
args.figure_dir = os.path.join(args.figure_root, args.exp_id)
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
data_prefix = 'win' + str(args.window_size) + '_'

data_paths = ['data/patient03-12-12-2023', 'data/patient05-02-15-2024', 'data/patient06-02-17-2024']
patient_ids = {0:3, 1:5, 2:6}
for i in range(len(data_paths)):
    args.patient_id = patient_ids[i]
    leave_out_idx = i
    val_data = np.load(os.path.join(data_paths[leave_out_idx], data_prefix + 'sensing_mat_data_val.npy')).astype(np.float32)
    val_label = np.load(os.path.join(data_paths[leave_out_idx], data_prefix + 'EMG_label_val.npy'))
    val_roi_label = np.load(os.path.join(data_paths[leave_out_idx], data_prefix + 'EMG_roi_label_val.npy'))
    if args.normalize_data:
        val_data /= 225
        val_data -= 0.002
        val_data /= 0.012

    positive_idx = val_label > 0
    val_label[positive_idx] = 1
    val_label = val_label.astype(np.int_)
    print(val_data.shape, val_label.shape, val_label.sum())

    # original CNN transformation
    val_transform = get_cnn_transforms(val_data.shape[1], train=False)

    # create dataset and dataloader
    val_dataset = RLSDataset(args.architecture, val_data, val_label, val_roi_label, transform=val_transform, normalize_roi=args.normalize_roi)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
            
    # main eval code
    # Initialize the model
    model = load_model(os.path.join(args.checkpoint_dir, 'best_accuracy.ckpt'), num_classes, args.window_size)
    model.to(device)
    model.eval()
    y_score, y_true = [], []
    for images, labels, _ in tqdm(val_loader):
        images = images.to(args.device)
        labels = labels.to(args.device)
        y_score.append(model(images).squeeze())
        y_true.append(labels)
    y_score = torch.hstack(y_score)
    y_true = torch.hstack(y_true)
    
    num_figures = 10
    for j in tqdm(range(num_figures)):
        threshold = j / num_figures
        plot_validation(args, y_score, y_true, threshold, 'accuracy')
        
    model = load_model(os.path.join(args.checkpoint_dir, 'best_loss.ckpt'), num_classes, args.window_size)
    model.to(device)
    model.eval()
    y_score, y_true = [], []
    for images, labels, _ in tqdm(val_loader):
        images = images.to(args.device)
        labels = labels.to(args.device)
        y_score.append(model(images).squeeze())
        y_true.append(labels)
    y_score = torch.hstack(y_score)
    y_true = torch.hstack(y_true)
    
    for j in tqdm(range(num_figures)):
        threshold = j / num_figures
        plot_validation(args, y_score, y_true, threshold, 'loss')
    
#     model = load_model(os.path.join(args.checkpoint_dir, 'best_f1.ckpt'), num_classes, args.window_size)
#     model.to(device)
#     model.eval()
#     plot_validation(args, model, val_loader, 'f1')
    
#     model = load_model(os.path.join(args.checkpoint_dir, 'best_f0.5.ckpt'), num_classes, args.window_size)
#     model.to(device)
#     model.eval()
#     plot_validation(args, model, val_loader, 'f0.5')
    
#     model = load_model(os.path.join(args.checkpoint_dir, 'best_miou.ckpt'), num_classes, args.window_size)
#     model.to(device)
#     model.eval()
#     plot_validation(args, model, val_loader, 'mIoU')
