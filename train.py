import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
import argparse
import os
import pandas as pd
# from sklearn.metrics import confusion_matrix, 
from sklearn.metrics import fbeta_score, precision_recall_curve, auc, accuracy_score, precision_score, recall_score
import timm
from tqdm import tqdm
from utils.models import *
from utils.dataset import *
from utils.metrics import *

def train(args, model, train_loader, cls_criterion, optimizer, scheduler=None):
    '''
    args: input arguments from main function
    model: model to be trained
    train_loader: training data dataloader
    epsilon: the laplace smoothing factor used to prevent division by 0
    cls_criterion: classification loss criterion
    regression_criterion: regression loss criterion
    optimizer: training optimizer
    scheduler: training scheduler 
    '''
    # Train the model
    running_cls_loss, running_regression_loss, total_samples = 0, 0, 0
    running_iou, total_iou_samples = 0, 0
    matrix = np.zeros((2, 2))
    model.train()
    train_loss = 0
    train_pred = []
    train_labels = []
    counts = 0
    for i, (images, labels) in tqdm(enumerate(train_loader)):
        images = images.to(args.device)
        labels = labels.to(args.device).float()
#         print(model(images).shape, labels.shape)
        pred = model(images).squeeze()
        loss = cls_criterion(pred, labels)
        train_loss += loss * len(images)
        train_pred.append(pred)
        train_labels.append(labels)
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
#         torch.nn.utils.clip_grad_value_(model.parameters(), 0.5) # clip gradient to [-1, 1]
        optimizer.step()
    y_pred = (torch.sigmoid(torch.cat(train_pred, dim=0).reshape(-1)).cpu().detach().numpy() > 0.5) * 1.0
    y_true = torch.cat(train_labels, dim=0).reshape(-1).cpu().detach().numpy()
    train_f1 = fbeta_score(y_true, y_pred, beta=1)
    train_fprec = fbeta_score(y_true, y_pred, beta=0.5)
    train_miou = np.logical_and(y_true, y_pred).sum() / (np.logical_or(y_true, y_pred).sum() + 1e-8)
    return train_loss, train_f1, train_fprec, train_miou

        
def val(args, model, optimizer, val_loader, cls_criterion):
    '''
    args: input arguments from main function
    model: model to be evaluated 
    train_loader: training data dataloader
    epsilon: the laplace smoothing factor used to prevent division by 0
    cls_criterion: classification loss criterion
    regression_criterion: regression loss criterion
    '''
    model.eval()
    running_loss = 0
    y_score, y_true = [], []
    for i, (images, labels) in tqdm(enumerate(val_loader)):
        images = images.to(args.device)
        labels = labels.to(args.device).float()

        # Forward pass
        with torch.no_grad():
            cls_logits = model(images).squeeze()
            loss = cls_criterion(cls_logits, labels)
        running_loss += loss.item() * labels.shape[0]
        y_score.append(cls_logits.reshape(-1))
        y_true.append(labels.reshape(-1))
    y_score = torch.sigmoid(torch.cat(y_score)).cpu().detach().numpy()
    running_loss /= len(y_score)
    y_pred = y_score > 0.5
    y_true = torch.cat(y_true).cpu().detach().numpy()
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    pr_auc = auc(recall, precision)
    accuracy = accuracy_score(y_true, y_pred) * 100
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = fbeta_score(y_true, y_pred, beta=1)
    fprec = fbeta_score(y_true, y_pred, beta=0.5)
    miou = np.logical_and(y_true, y_pred).sum() / (np.logical_or(y_true, y_pred).sum() + 1e-8)
        
    if f1 > args.best_f1:
        args.best_f1 = f1
        args.best_f1_epoch = args.epoch + 1
        if not args.debug_mode:
            torch.save({'state_dict':model.state_dict(), 'optimizer':optimizer.state_dict(), 'precision':precision, 'recall':recall, 'f1':f1, 'cls_loss':running_loss}, os.path.join(args.checkpoint_dir, 'best_f1.ckpt'))
        tqdm.write('Best F1 model saved!')
            
    if fprec > args.best_fprec:
        args.best_fprec = fprec
        args.best_fprec_epoch = args.epoch + 1
        if not args.debug_mode:
            torch.save({'state_dict':model.state_dict(), 'optimizer':optimizer.state_dict(), 'precision':precision, 'recall':recall, 'f1':f1, 'f0.5':fprec, 'cls_loss': running_loss}, os.path.join(args.checkpoint_dir, 'best_f0.5.ckpt'))
        tqdm.write('Best F0.5 model saved!')
    
    if miou > args.best_miou:
        args.best_miou = miou
        args.best_miou_epoch = args.epoch + 1
        if not args.debug_mode:
            torch.save({'state_dict':model.state_dict(), 'optimizer':optimizer.state_dict(), 'precision':precision, 'recall':recall, 'f1':f1, 'f0.5':fprec, 'cls_loss':running_loss, 'miou':miou}, os.path.join(args.checkpoint_dir, 'best_miou.ckpt'))
        tqdm.write('Best miou model saved!')
    tqdm.write(f'Epoch [{args.epoch+1}/{args.epochs}], CLS Loss: {loss}, F1: {f1 * 100:.2f}%, F0.5: {fprec * 100:.2f}%, precision: {precision * 100:.2f}%, recall: {recall * 100:.2f}%, miou: {miou * 100:.2f}%')
#     tqdm.write(f'Epoch [{args.epoch+1}/{args.epochs}], CLS Loss: {bce_loss}, f loss: {f_loss}, contrastive loss: {contrastive_loss}, rank loss: {rank_loss}, PR AUC: {pr_auc:.4f}')
#     return bce_loss, regression_loss, pr_auc, accuracy
    return precision, recall, f1, fprec, loss, accuracy, miou
        