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
from utils.class_balanced_loss import CB_loss
import timm
from tqdm import tqdm
from utils.models import *
from utils.dataset import *
from utils.metrics import *
from utils.lossfunc import *
import wandb
from train import *

parser = argparse.ArgumentParser()
parser.add_argument("--lr", default=0.01, help="learning rate used to train the model", type=float)
parser.add_argument("--weight_decay", default=0.1, help="weight decay used to train the model", type=float)
parser.add_argument("--epochs", default=300, help="epochs used to train the model", type=int)
parser.add_argument("--batch_size", default=256, help="batch size used to train the model", type=int)
parser.add_argument("--num_classes", default=2, help="number of classes for the classifier", type=int)
parser.add_argument("--window_size", default=16, help="window size of the input data", type=int)
parser.add_argument("--label_type", default="hard", help="indicate whether use hard one-hot labels or soft numerical labels", choices=["hard", "soft"])
parser.add_argument("--long_tailed", default=0, help="indicate whether use balanced sampled data or the whole long-tailed data, 0 for balanced, 1 for long-tailed", type=int)
parser.add_argument("--exp_label", default=None, help="extra labels to distinguish between different experiments")
parser.add_argument("--cross_val_type", default=0, type=int, help="0 for train all val all, 1 for leave patient 1 out")
parser.add_argument("--task", default="regression", type=str, help="indicate what kind of task to be run (regression/classification, etc.)")
parser.add_argument("--normalize_roi", default=1, type=int, help="whether normalize the roi indices between [0, 1]")
parser.add_argument("--alpha", default=0.5, type=float, help="weight of cls loss, default 0.5")
parser.add_argument("--architecture", default="3d-resnet18", choices=["3d-resnet10", "3d-resnet18", "2d-resnet18", "ViT-tiny"], help="architecture used")
parser.add_argument("--exp_id", default=0, type=str, help="the id of the experiment")
parser.add_argument("--debug_mode", default=0, type=int, help="0 for experiment mode, 1 for debug mode")
args = parser.parse_args()

args.exp_name = f"{args.task}_win{args.window_size}_epoch{args.epochs}_lr{args.lr}_wd{args.weight_decay}_bs{args.batch_size}_alpha{args.alpha}_cv{args.cross_val_type}_nr{args.normalize_roi}_{args.architecture}"
args.checkpoint_root = os.path.join('checkpoint', args.exp_name)
if not os.path.exists(args.checkpoint_root) and (not args.debug_mode):
    os.mkdir(args.checkpoint_root)
    
args.exp_name += f'_{args.exp_id}'
args.checkpoint_dir = os.path.join(args.checkpoint_root, args.exp_id)
if not os.path.exists(args.checkpoint_dir) and (not args.debug_mode):
    os.mkdir(args.checkpoint_dir)

# start a new wandb run to track this script
if not args.debug_mode:
    wandb.init(
        # set the wandb project where this run will be logged
        project="rls",
        
        # track hyperparameters and run metadata
        config={
        "task": args.task,
        "window_size": args.window_size,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "architecture": args.architecture,
        "cross_val_type": args.cross_val_type,
        "normalize_roi": args.normalize_roi,
        "epochs": args.epochs,
        },
    
        # experiment name
        name=args.exp_name
    )

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args.device = device

# torch.manual_seed(1234)

# Hyperparameters
num_classes = args.num_classes  # Number of classes in ImageNet
window_size = args.window_size
num_epochs = args.epochs
batch_size = args.batch_size
learning_rate = args.lr

# load data
args.window_size = 'win' + str(args.window_size) + '_'
if args.long_tailed:
    args.window_size += 'LT_'

data_paths = ['data/12-13-2023', 'data/02-17-2024']
# data_paths = ['data/12-13-2023', 'data/02-15-2024', 'data/02-17-2024']
if args.cross_val_type == 0:
    train_data = np.concatenate([np.load(os.path.join(path, args.window_size + 'sensing_mat_data_train.npy')).astype(np.float32) for path in data_paths], axis=0)
    train_label = np.concatenate([np.load(os.path.join(path, args.window_size + 'EMG_label_train.npy')) for path in data_paths], axis=0)
    train_roi_label = np.concatenate([np.load(os.path.join(path, args.window_size + 'EMG_roi_label_train.npy')) for path in data_paths], axis=0)
    val_data = np.concatenate([np.load(os.path.join(path, args.window_size + 'sensing_mat_data_val.npy')).astype(np.float32) for path in data_paths], axis=0)
    val_label = np.concatenate([np.load(os.path.join(path, args.window_size + 'EMG_label_val.npy')) for path in data_paths], axis=0)
    val_roi_label = np.concatenate([np.load(os.path.join(path, args.window_size + 'EMG_roi_label_val.npy')) for path in data_paths], axis=0)
else:
    leave_out_idx = args.cross_val_type - 1
    train_data = np.concatenate([np.load(os.path.join(data_paths[i], args.window_size + 'sensing_mat_data_train.npy')).astype(np.float32) for i in range(len(data_paths)) if i != leave_out_idx], axis=0)
    train_label = np.concatenate([np.load(os.path.join(data_paths[i], args.window_size + 'EMG_label_train.npy')) for i in range(len(data_paths)) if i != leave_out_idx], axis=0)
    train_roi_label = np.concatenate([np.load(os.path.join(data_paths[i], args.window_size + 'EMG_roi_label_train.npy')) for i in range(len(data_paths)) if i != leave_out_idx], axis=0)
    val_data = np.concatenate([np.load(os.path.join(data_paths[leave_out_idx], args.window_size + 'sensing_mat_data_train.npy')).astype(np.float32), np.load(os.path.join(data_paths[leave_out_idx], args.window_size + 'sensing_mat_data_val.npy')).astype(np.float32)], axis=0)
    val_label = np.concatenate([np.load(os.path.join(data_paths[leave_out_idx], args.window_size + 'EMG_label_train.npy')), np.load(os.path.join(data_paths[leave_out_idx], args.window_size + 'EMG_label_val.npy'))], axis=0)
    val_roi_label = np.concatenate([np.load(os.path.join(data_paths[leave_out_idx], args.window_size + 'EMG_roi_label_train.npy')), np.load(os.path.join(data_paths[leave_out_idx], args.window_size + 'EMG_roi_label_val.npy'))], axis=0)

if args.label_type == "hard":
    positive_idx = train_label > 0
    train_label[positive_idx] = 1
    positive_idx = val_label > 0
    val_label[positive_idx] = 1
    train_label = train_label.astype(np.int_)
    val_label = val_label.astype(np.int_)
else:
    raise NotImplementedError("soft label not implemented yet!")
    
print(train_data.shape, train_data.dtype, train_label.shape, val_data.shape, val_label.shape, val_label.sum())
train_data /= 225
train_data -= 0.002
train_data /= 0.012

# original CNN transformation
train_transform = get_cnn_transforms(train_data.shape[1])
val_transform = get_cnn_transforms(train_data.shape[1], train=False)

# create dataset and dataloader
train_dataset = RLSDataset(args.architecture, train_data, train_label, train_roi_label, transform=train_transform, normalize_roi=args.normalize_roi)
val_dataset = RLSDataset(args.architecture, val_data, val_label, val_roi_label, transform=val_transform, normalize_roi=args.normalize_roi)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

# Initialize the model
model = get_model(args.architecture, num_classes, window_size)
model.to(device)

# Loss and optimizer
img_num_per_cls = np.unique(train_label, return_counts=True)[1]

cls_criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=args.weight_decay)
scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0)
args.best_f1, args.best_fprec, args.best_miou = -np.inf, -np.inf, -np.inf
args.best_f1_epoch, args.best_fprec_epoch, args.best_miou_epoch = None, None, None

# main train val loop
for epoch in range(num_epochs):
    args.epoch = epoch
    train_precision, train_recall, train_f1, train_fprec, train_cls_loss, train_regression_loss, train_miou = train(args, model, train_loader, cls_criterion, regression_criterion, optimizer, scheduler)
    val_precision, val_recall, val_f1, val_fprec, val_cls_loss, val_regression_loss, val_miou = val(args, model, val_loader, cls_criterion, regression_criterion)
    if not args.debug_mode:
        wandb.log({"Classification/train/loss": train_cls_loss, "Regression/train/loss": train_regression_loss, "Classification/train/f1": train_f1, "Classification/train/f0.5": train_fprec, "Classification/train/precision": train_precision, "Classification/train/recall": train_recall, "Regression/train/mIoU": train_miou, "Classification/val/loss": val_cls_loss, "Regression/val/loss": val_regression_loss, "Classification/val/f1": val_f1, "Classification/val/f0.5": val_fprec, "Classification/val/precision": val_precision, "Classification/val/recall": val_recall, "Regression/val/mIoU": val_miou})

if not args.debug_mode:
    wandb.finish()
print(f'The best f1 score is {args.best_f1:.2f} at epoch {args.best_f1_epoch}; The best f0.5 score is {args.best_fprec:.2f} at epoch {args.best_fprec_epoch}; The best mIoU score is {args.best_miou:.2f} at epoch {args.best_miou_epoch}')