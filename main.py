import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, LambdaLR
import argparse
import os
import pandas as pd
from sklearn.metrics import confusion_matrix
import timm
from tqdm import tqdm
from utils import get_checkpoint_path, prepare_folder
from utils.models import *
from utils.dataset import *
from utils.metrics import *
import wandb
from train import *
from utils.lossfunc import *
from utils.scheduler import WarmupCosineScheduler

parser = argparse.ArgumentParser()
parser.add_argument("--lr", default=0.01, help="learning rate used to train the model", type=float)
parser.add_argument("--weight_decay", default=0.1, help="weight decay used to train the model", type=float)
parser.add_argument("--epochs", default=300, help="epochs used to train the model", type=int)
parser.add_argument("--warmup_epochs", default=20, help="linear warmup epochs used to train the model", type=int)
parser.add_argument("--batch_size", default=256, help="batch size used to train the model", type=int)
parser.add_argument("--num_classes", default=2, help="number of classes for the classifier", type=int)
parser.add_argument("--clip_len", default=16, help="window size of the input data", type=int)
parser.add_argument("--input_size", default=[16, 16], nargs="+", help="window size of the input data", type=int)
parser.add_argument("--label_type", default="hard", help="indicate whether use hard one-hot labels or soft numerical labels", choices=["hard", "soft"])
parser.add_argument("--exp_label", default=None, help="extra labels to distinguish between different experiments")
parser.add_argument("--cross_val_type", default=0, type=int, help="0 for train all val all, 1 for leave patient 1 out")
parser.add_argument("--task", default="classification", type=str, choices=["classification", "regression"], help="indicate what kind of task to be run (regression/classification, etc.)")
parser.add_argument("--architecture", default="3d-resnet18", choices=["2d-efficientnet", "2d-positional", "2d-linear", "3d-conv", "2d-conv", "3d-resnet10", "3d-resnet18", "2d-resnet18", "ViT-tiny"], help="architecture used")
parser.add_argument("--exp_id", default=0, type=str, help="the id of the experiment")
parser.add_argument("--debug_mode", default=0, type=int, help="0 for experiment mode, 1 for debug mode")
parser.add_argument("--normalize_data", default=0, type=int, help="0 for raw mat input, 1 for normalized to [0, 1] and standardization")
parser.add_argument("--patients", default=[15], type=int, nargs="+", help="patient ids included in the training")
parser.add_argument("--checkpoint_root", default='checkpoint', type=str, help="checkpoint root path")
parser.add_argument("--seed", default=1, type=int, help="random seed for torch")
parser.add_argument("--split", default=-1, type=float, help="split ratio of train and val")
parser.add_argument("--data_type", default='tal', choices=['tal_pos', 'tal', 'context'], type=str, help="choose which kind of data to use")
args = parser.parse_args()

def main_func(args):
    torch.manual_seed(args.seed)
    
    args.exp_name = get_checkpoint_path(args)
    args.checkpoint_dir = prepare_folder(args)
    
    # start a new wandb run to track this script
    if not args.debug_mode:
        wandb.init(
            # set the wandb project where this run will be logged
            project="rls",
            
            # track hyperparameters and run metadata
            config={
                "task": args.task,
                "clip_len": args.clip_len,
                "learning_rate": args.lr,
                "weight_decay": args.weight_decay,
                "batch_size": args.batch_size,
                "architecture": args.architecture,
                "cross_val_type": args.cross_val_type,
                "epochs": args.epochs,
                "num class": args.num_classes
            },
        
            # experiment name
            name=args.exp_id
        )
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    
    # torch.manual_seed(1234)
    
    # Hyperparameters
    num_classes = args.num_classes  # Number of classes in ImageNet
    clip_len = args.clip_len
    num_epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = args.lr
    
    # load data
    args.clip_len_prefix = 'win' + str(args.clip_len) + '_'
    
    train_data, train_label, val_data, val_label = preprocess_dataset(args)
    # train_data, train_label, val_data, val_label = prepare_datasets(args)
    print(train_data.shape, train_label.shape, train_label.sum(), val_data.shape, val_label.shape, val_label.sum())
    print(train_data.min(), train_data.max(), train_data.mean(), val_data.min(), val_data.max(), val_data.mean())
    
    # original CNN transformation
    train_transform = get_cnn_transforms(args)
    val_transform = get_cnn_transforms(args, train=False)
    
    # create dataset and dataloader
    train_dataset = RLSDataset(args, train_data, train_label, transform=train_transform)
    val_dataset = RLSDataset(args, val_data, val_label, transform=val_transform)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    
    # Initialize the model
    model = get_model(args)
    model.to(device)
    if not args.debug_mode:
        wandb.watch(model, log='all', log_freq=1)
    
    # Loss and optimizer
    cls_num_list = np.unique(train_label, return_counts=True)[1]
    beta = 0.9999
    effective_num = 1.0 - np.power(beta, cls_num_list)
    per_cls_weights = (1.0 - beta) / np.array(effective_num)
    neg_weight, pos_weight = per_cls_weights
    pos_weight = torch.tensor(pos_weight / neg_weight).to(device)
    # print(pos_weight)
    cls_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    # cls_criterion = DiceLoss()
    # cls_criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=args.weight_decay)
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=args.weight_decay)
    
    # Cosine annealing scheduler with warmup
    args.warmup_epochs = max(10, args.epochs // 100)
    scheduler = WarmupCosineScheduler(args, optimizer)
    
    args.best_f1, args.best_fprec, args.best_miou, args.best_pr_auc, args.best_loss, args.best_accuracy = -np.inf, -np.inf, -np.inf, -np.inf, np.inf, -np.inf
    args.best_f1_epoch, args.best_fprec_epoch, args.best_miou_epoch, args.best_pr_auc_epoch, args.best_loss_epoch, args.best_accuracy_epoch = None, None, None, None, None, None
    
    print('start training loop')
    # main train val loop
    for epoch in range(num_epochs):
        args.epoch = epoch
        train_loss, train_f1, train_fprec, train_miou = train(args, model, train_loader, cls_criterion, optimizer)
    #     train_precision, train_recall, train_f1, train_fprec, train_cls_loss, train_regression_loss, train_miou = train(args, model, train_loader, cls_criterion, regression_criterion, optimizer, scheduler)
        precision, recall, f1, fprec, val_cls_loss, accuracy, miou = val(args, model, optimizer, val_loader, cls_criterion)
        #     val_precision, val_recall, val_f1, val_fprec, val_cls_loss, val_regression_loss, val_miou = val(args, model, val_loader, cls_criterion, regression_criterion)
        if not args.debug_mode:
            wandb.log({"Classification/val/loss": val_cls_loss, "Classification/val/f1": f1, "Classification/val/f0.5":fprec, "Classification/val/precision":precision, "Classification/val/recall":recall, "Classification/val/miou": miou, "Classification/train/loss":train_loss, "Classification/train/f1":train_f1, "Classification/train/f0.5":train_fprec, "Classification/train/miou":train_miou})
        if scheduler:
            initial_lr = scheduler.get_last_lr()
            scheduler.step()
            curr_lr = scheduler.get_last_lr()
            print(f'previous lr: {initial_lr} vs. curr lr: {curr_lr}')
    
    if not args.debug_mode:
        wandb.finish()
        torch.save({'f1':args.best_f1, 'f0.5':args.best_fprec, 'accuracy':args.best_accuracy, 'miou':args.best_miou, 'best_f1_epoch':args.best_f1_epoch, 'best_f0.5_epoch':args.best_fprec_epoch, 'best_accuracy_epoch':args.best_accuracy_epoch, 'best_miou_epoch':args.best_miou_epoch}, os.path.join(args.checkpoint_dir, 'val_results.pth'))
    print(f'The best f1 score is {args.best_f1 * 100:.2f} at epoch {args.best_f1_epoch}; The best f0.5 score is {args.best_fprec * 100:.2f} at epoch {args.best_fprec_epoch}; The best miou score is {args.best_miou * 100:.2f}% at epoch {args.best_miou_epoch}', flush=True)
    return args.best_f1

if __name__ == '__main__':
    main_func(args)