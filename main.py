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
from copy import deepcopy

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
parser.add_argument("--architecture", default="3d-resnet18", choices=["2d-efficientnet", "2d-positional", "2d-linear", "3d-conv", "2d-conv", "3d-resnet10", "3d-resnet18", "2d-resnet18", "2d-mlp", "2d-baseline", "3d-pretrained-resnet"], help="architecture used")
parser.add_argument("--exp_id", default=0, type=str, help="the id of the experiment")
parser.add_argument("--debug_mode", default=0, type=int, help="0 for experiment mode, 1 for debug mode")
parser.add_argument("--patients", default=[15], type=int, nargs="+", help="patient ids included in the training")
parser.add_argument("--checkpoint_root", default='checkpoint', type=str, help="checkpoint root path")
parser.add_argument("--seed", default=1, type=int, help="random seed for torch")
parser.add_argument("--split", default=-1, type=float, help="split ratio of train and val")
parser.add_argument("--data_type", default='tal', choices=['tal_pos', 'tal', 'context', 'context_more_negative', 'rgb_video'], type=str, help="choose which kind of data to use")
parser.add_argument("--subset", default=1.0, type=float, help="training subset / training set")
parser.add_argument("--val_freq", default=1, type=int, help="number of epochs between each validation")
parser.add_argument("--num_workers", default=8, type=int, help="number of workers for pytorch dataloader")
parser.add_argument("--downsample_val", default=1, type=int, help="1 for downsample the validation set")
parser.add_argument("--phase", default='train', type=str, help="identify whether training or testing phase")
parser.add_argument("--num_val_sets", default=1, type=int, help="number of validation set used")
args = parser.parse_args()
args.clip_len_prefix = 'win' + str(args.clip_len) + '_'

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
    train_data, train_label, cross_val_data, cross_val_label, internal_val_data, internal_val_label = preprocess_dataset(args)
    if args.subset < 1:
        train_indices = np.random.choice(np.arange(len(train_data)), int(args.subset * len(train_data)), replace=False)
        train_data = train_data[train_indices]
        train_label = train_label[train_indices]
    # train_data, train_label, val_data, val_label = prepare_datasets(args)
    print("Training metadata:", train_data.shape, train_label.shape, int(train_label.sum()))
    print("Training statistics:", f'{train_data.mean()} ± {train_data.std()}')
    if cross_val_data is not None:
        print("Cross patient validation metadata:", cross_val_label.shape, cross_val_label.sum(axis=-1).astype(int))
        print("Cross patient validation statistics:", f'{cross_val_data.mean()} ± {cross_val_data.std()}')
    if internal_val_data is not None:
        print("Internal validation metadata:", internal_val_label.shape, internal_val_label.sum(axis=-1).astype(int))
        print("Internal validation statistics:", f'{internal_val_data.mean()} ± {internal_val_data.std()}')
    
    
    # original CNN transformation
    train_transform = get_cnn_transforms(args)
    val_transform = get_cnn_transforms(args, train=False)
    
    # create dataset and dataloader
    train_dataset = RLSDataset(args, train_data, train_label, transform=train_transform)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=args.num_workers, shuffle=True, pin_memory=True)
    if cross_val_data is not None:
        cross_val_datasets = [RLSDataset(args, cross_val_data[i], cross_val_label[i], transform=val_transform) for i in range(args.num_val_sets)]
        cross_val_loaders = [DataLoader(dataset=cross_val_datasets[i], num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False, pin_memory=True) for i in range(args.num_val_sets)]
    if internal_val_data is not None:
        internal_val_datasets = [RLSDataset(args, internal_val_data[i], internal_val_label[i], transform=val_transform) for i in range(args.num_val_sets)]
        internal_val_loaders = [DataLoader(dataset=internal_val_datasets[i], num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False, pin_memory=True) for i in range(args.num_val_sets)]
    
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
    scheduler = WarmupCosineScheduler(args, optimizer)
    
    infs = [np.inf] * args.num_val_sets
    neg_infs = [-np.inf] * args.num_val_sets
    args.best_f1 = {'cross':deepcopy(neg_infs), 'internal':deepcopy(neg_infs)}
    args.best_fprec = {'cross':deepcopy(neg_infs), 'internal':deepcopy(neg_infs)}
    args.best_miou = {'cross':deepcopy(neg_infs), 'internal':deepcopy(neg_infs)}
    args.best_loss = {'cross':deepcopy(infs), 'internal':deepcopy(infs)}
    
    nones = [-1] * args.num_val_sets
    args.best_f1_epoch = {'cross':nones, 'internal':nones}
    args.best_fprec_epoch = {'cross':nones, 'internal':nones}
    args.best_miou_epoch = {'cross':nones, 'internal':nones}
    args.best_loss_epoch = {'cross':nones, 'internal':nones}
    
    print('start training loop')        
    # main train val loop
    for epoch in range(num_epochs):
        args.epoch = epoch
        
        # train one epoch and get training logs
        train_loss, train_f1, train_fprec, train_miou = train(args, model, train_loader, cls_criterion, optimizer)
        log_data = {"train/loss":train_loss, "train/f1":train_f1, "train/f0.5":train_fprec, "train/miou":train_miou}
        
        if (epoch + 1) % args.val_freq == 0:
            # val one epoch and get val logs
            for i in range(args.num_val_sets):
                if cross_val_data is not None:
                    precision, recall, f1, fprec, val_loss, miou = val(args, model, optimizer, cross_val_loaders[i], cls_criterion, 'cross', i)
                    log_data.update({f"val{i}/cross_loss": val_loss, f"val{i}/cross_f1": f1, f"val{i}/cross_f0.5":fprec, f"val{i}/cross_precision":precision, f"val{i}/cross_recall":recall, f"val{i}/cross_miou": miou})
                if internal_val_data is not None:
                    precision, recall, f1, fprec, val_loss, miou = val(args, model, optimizer, internal_val_loaders[i], cls_criterion, 'internal', i)
                    log_data.update({f"val{i}/internal_loss": val_loss, f"val{i}/internal_f1": f1, f"val{i}/internal_f0.5":fprec, f"val{i}/internal_precision":precision, f"val{i}/internal_recall":recall, f"val{i}/internal_miou": miou})
        
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