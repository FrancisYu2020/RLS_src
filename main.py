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
parser.add_argument("--input_size", default=[0, 16, 0, 16], nargs="+", help="window size of the input data", type=int)
parser.add_argument("--val_type", default='cross+internal-val', choices=['cross-val', 'internal-val', 'cross+internal-val'], type=str, help="choose the method for validation, cross-val for validation on different patient, internal-val for validation on same patient different sleep stage, otherwise validation on both")
# parser.add_argument("--task", default="classification", type=str, choices=["classification", "regression"], help="indicate what kind of task to be run (regression/classification, etc.)")
parser.add_argument("--architecture", default="3d-resnet18", choices=["2d-efficientnet", "2d-positional", "2d-linear", "3d-conv", "2d-conv", "3d-resnet10", "3d-resnet18", "2d-resnet18", "2d-mlp", "2d-baseline"], help="architecture used")
parser.add_argument("--exp_id", default=0, type=str, help="the id of the experiment")
parser.add_argument("--debug_mode", default=0, type=int, help="0 for experiment mode, 1 for debug mode")
parser.add_argument("--patients", default=[15], type=int, nargs="+", help="patient ids included in the training")
parser.add_argument("--checkpoint_root", default='checkpoint', type=str, help="checkpoint root path")
parser.add_argument("--seed", default=1, type=int, help="random seed for torch")
parser.add_argument("--split", default=-1, type=float, help="split ratio of train and val")
parser.add_argument("--data_type", default='tal', choices=['tal_pos', 'tal', 'context', 'rgb_video'], type=str, help="choose which kind of data to use")
parser.add_argument("--subset", default=1.0, type=float, help="training subset / training set")
parser.add_argument("--val_freq", default=10, type=int, help="number of epochs between each validation")
parser.add_argument("--num_workers", default=8, type=int, help="number of workers for pytorch dataloader")
args = parser.parse_args()

def main_func(args):
    torch.manual_seed(args.seed)
    
    args.exp_name = get_checkpoint_path(args)
    args.checkpoint_dir = prepare_folder(args)
    
    # start a new wandb run to track this script
    if not args.debug_mode:
        wandb.init(
            # set the wandb project where this run will be logged
            project="rls_context",
            
            # track hyperparameters and run metadata
            config={
                "clip_len": args.clip_len,
                "learning_rate": args.lr,
                "weight_decay": args.weight_decay,
                "batch_size": args.batch_size,
                "architecture": args.architecture,
                "val_type": args.val_type,
                "epochs": args.epochs
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
    
    train_data, train_label, cross_val_data, cross_val_label, internal_val_data, internal_val_label = preprocess_dataset(args)
    if args.subset < 1:
        train_indices = np.random.choice(np.arange(len(train_data)), int(args.subset * len(train_data)), replace=False)
        train_data = train_data[train_indices]
        train_label = train_label[train_indices]
    # train_data, train_label, val_data, val_label = prepare_datasets(args)
    print("Training metadata:", train_data.shape, train_label.shape, int(train_label.sum()))
    print("Training statistics:", f'{train_data.mean()} ± {train_data.std()}')
    if cross_val_data is not None:
        print("Cross patient validation metadata:", cross_val_label.shape, int(cross_val_label.sum()))
        print("Cross patient validation statistics:", f'{cross_val_data.mean()} ± {cross_val_data.std()}')
    if internal_val_data is not None:
        print("Internal validation metadata:", internal_val_label.shape, int(internal_val_label.sum()))
        print("Internal validation statistics:", f'{internal_val_data.mean()} ± {internal_val_data.std()}')
    
    
    # original CNN transformation
    train_transform = get_cnn_transforms(args)
    val_transform = get_cnn_transforms(args, train=False)
    
    # create dataset and dataloader
    train_dataset = RLSDataset(args, train_data, train_label, transform=train_transform)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=args.num_workers, shuffle=True, pin_memory=True)
    if cross_val_data is not None:
        cross_val_dataset = RLSDataset(args, cross_val_data, cross_val_label, transform=val_transform)
        cross_val_loader = DataLoader(dataset=cross_val_dataset, num_workers=args.num_workers, batch_size=4 * batch_size, shuffle=False, pin_memory=True)
    if internal_val_data is not None:
        internal_val_dataset = RLSDataset(args, internal_val_data, internal_val_label, transform=val_transform)
        internal_val_loader = DataLoader(dataset=internal_val_dataset, num_workers=args.num_workers, batch_size=4 * batch_size, shuffle=False, pin_memory=True)
    
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
    cls_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=args.weight_decay)
    
    # Cosine annealing scheduler with warmup
    args.warmup_epochs = max(10, args.epochs // 100)
    args.cosine_epochs = max(20, int(args.epochs * 0.8))
#     args.warmup_epochs = 1
#     args.cosine_epochs = 3
    scheduler = WarmupCosineScheduler(args, optimizer)
    
    args.best_f1 = {'cross':-np.inf, 'internal':-np.inf}
    args.best_fprec = {'cross':-np.inf, 'internal':-np.inf}
    args.best_miou = {'cross':-np.inf, 'internal':-np.inf}
    args.best_loss = {'cross':np.inf, 'internal':np.inf}
    
    args.best_f1_epoch = {'cross':None, 'internal':None}
    args.best_fprec_epoch = {'cross':None, 'internal':None}
    args.best_miou_epoch = {'cross':None, 'internal':None}
    args.best_loss_epoch = {'cross':None, 'internal':None}
    
    print('start training loop')        
    # main train val loop
    for epoch in range(num_epochs):
        args.epoch = epoch
        
        # train one epoch and get training logs
        train_loss, train_f1, train_fprec, train_miou = train(args, model, train_loader, cls_criterion, optimizer)
        log_data = {"train/loss":train_loss, "train/f1":train_f1, "train/f0.5":train_fprec, "train/miou":train_miou}
        
        if (epoch + 1) % args.val_freq == 0:
            # val one epoch and get val logs
            if cross_val_data is not None:
                precision, recall, f1, fprec, val_loss, miou = val(args, model, optimizer, cross_val_loader, cls_criterion, 'cross')
                log_data.update({"val/cross_loss": val_loss, "val/cross_f1": f1, "val/cross_f0.5":fprec, "val/cross_precision":precision, "val/cross_recall":recall, "val/cross_miou": miou})
            if internal_val_data is not None:
                precision, recall, f1, fprec, val_loss, miou = val(args, model, optimizer, internal_val_loader, cls_criterion, 'internal')
                log_data.update({"val/internal_loss": val_loss, "val/internal_f1": f1, "val/internal_f0.5":fprec, "val/internal_precision":precision, "val/internal_recall":recall, "val/internal_miou": miou})
        
        if not args.debug_mode:
            wandb.log(log_data)
                
        if scheduler:
            initial_lr = scheduler.get_last_lr()
            scheduler.step()
            curr_lr = scheduler.get_last_lr()
            print(f'previous lr: {initial_lr} vs. curr lr: {curr_lr}')
    
    if not args.debug_mode:
        wandb.finish()
        torch.save({'f1':args.best_f1, 'f0.5':args.best_fprec, 'miou':args.best_miou, 'best_f1_epoch':args.best_f1_epoch, 'best_f0.5_epoch':args.best_fprec_epoch, 'best_miou_epoch':args.best_miou_epoch}, os.path.join(args.checkpoint_dir, 'val_results.pth'))
    print(f'The best f1 score is {args.best_f1} at epoch {args.best_f1_epoch}')
    print(f'The best f0.5 score is {args.best_fprec} at epoch {args.best_fprec_epoch}')
    print(f'The best miou score is {args.best_miou}% at epoch {args.best_miou_epoch}')
    return args.best_f1

if __name__ == '__main__':
    main_func(args)