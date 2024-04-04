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
from sklearn.metrics import fbeta_score, precision_recall_curve, auc, accuracy_score
import timm
from tqdm import tqdm
from utils.models import *
from utils.dataset import *
from utils.metrics import *

def train(args, model, train_loader, cls_criterion, regression_criterion, optimizer, scheduler=None):
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
    for i, (images, labels, roi_labels) in enumerate(train_loader):
        images = images.to(args.device)
        labels = labels.to(args.device).float()
#         roi_labels = roi_labels.to(args.device).float()
    
        # Forward pass
        cls_loss, regression_loss = 0, 0
        if args.task == "classification":
            cls_logits = model(images).squeeze()
            cls_loss += sum(cls_criterion(cls_logits, labels))
#             cls_loss += cls_criterion(cls_logits, labels)
        else:
            raise NotImplementedError("regression task not implemented")
#             cls_logits, regression_logits = model(images)
#             cls_loss += cls_criterion(cls_logits, labels)
#             regression_loss += regression_criterion(regression_logits, roi_labels, labels)
#         loss = args.alpha * cls_loss + (1 - args.alpha) * regression_loss
        loss = cls_loss
    
#         _, predicted = torch.max(cls_logits, 1)
    
        #compute the iou
#         ious = calculate_iou(roi_labels, regression_logits.detach())
#         running_iou += ious.sum()
#         total_iou_samples += ious.size(0)
        #TODO: finish the case when no regression is conducted
    
        # Calculate the number of correct predictions
#         matrix[0][0] += (labels[predicted == 0] == 0).sum()
#         matrix[0][1] += (labels[predicted == 1] == 0).sum()
#         matrix[1][0] += (labels[predicted == 0] == 1).sum()
#         matrix[1][1] += (labels[predicted == 1] == 1).sum()
    
        running_cls_loss += cls_loss.item()
#         running_regression_loss += regression_loss.item()
    
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()
            print(f"lr decayed!")
        
    # Return calculated metrics
#     return get_metric_scores(matrix, running_cls_loss, running_regression_loss, running_iou, total_iou_samples)

        
def val(args, model, val_loader, cls_criterion, regression_criterion):
    '''
    args: input arguments from main function
    model: model to be evaluated 
    train_loader: training data dataloader
    epsilon: the laplace smoothing factor used to prevent division by 0
    cls_criterion: classification loss criterion
    regression_criterion: regression loss criterion
    '''
#     matrix = np.zeros((2, 2))
#     correct = 0
    model.eval()
    running_cls_loss, running_regression_loss = 0, 0
#     running_iou, total_iou_samples = 0, 0
    y_score, y_true = [], []
    for i, (images, labels, roi_labels) in enumerate(val_loader):
        images = images.to(args.device)
        labels = labels.to(args.device).float()
#         roi_labels = roi_labels.to(args.device).float()

        # Forward pass
        cls_loss, regression_loss = 0, 0
        if args.task == 'classification':
            cls_logits = model(images).squeeze()
            bce_loss, f_loss, contrastive_loss, rank_loss = cls_criterion(cls_logits, labels)
#             loss += cls_criterion(cls_logits, labels)
        else:
            cls_logits, regression_logits = model(images)
            mask = roi_labels[:, 0] != -1
            cls_loss += cls_criterion(cls_logits, labels)
            regression_loss += regression_criterion(regression_logits, roi_labels, labels)
        loss = bce_loss + contrastive_loss + f_loss
        running_cls_loss += bce_loss.item()
        y_score.append(cls_logits)
        y_true.append(labels)
#         running_regression_loss += regression_loss.item()
        
#         _, predicted = torch.max(cls_logits, 1)
        # predicted = torch.ones(outputs.size(0))
        # predicted = torch.from_numpy(np.random.choice([0, 1], size=predicted.size(0)))

        #compute the iou
#         ious = calculate_iou(roi_labels, regression_logits)
#         running_iou += ious.sum()
#         total_iou_samples += ious.size(0)
        #TODO: finish the case when no regression is conducted
    bce_loss /= len(y_score)
    contrastive_loss /= len(y_score)
    rank_loss /= len(y_score)
    f_loss /= len(y_score)
    y_score = torch.hstack(y_score).cpu().detach().numpy()
    y_true = torch.hstack(y_true).cpu().detach().numpy()
#     f_prec = fbeta(y_true, y_pred, beta=0.5)
#     f1 = fbeta(y_true, y_pred, beta=1)
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    pr_auc = auc(recall, precision)
    accuracy = accuracy_score(y_true, y_score >= 0) * 100
    
    if pr_auc > args.best_pr_auc:
        args.best_pr_auc = pr_auc
        args.best_pr_auc_epoch = args.epoch + 1
        if not args.debug_mode:
            torch.save({'state_dict':model.state_dict(), 'cls_loss':bce_loss, 'regression_loss':regression_loss, 'pr_auc': pr_auc, 'accuracy': accuracy}, os.path.join(args.checkpoint_dir, 'best_pr_auc.ckpt'))
        print('Best PR AUC model saved!')
    
    if loss < args.best_loss:
        args.best_loss = loss
        args.best_loss_epoch = args.epoch + 1
        if not args.debug_mode:
            torch.save({'state_dict':model.state_dict(), 'cls_loss':bce_loss, 'regression_loss':regression_loss, 'pr_auc': pr_auc, 'accuracy': accuracy}, os.path.join(args.checkpoint_dir, 'best_loss.ckpt'))
        print('Best val loss model saved!')
    
    if accuracy > args.best_accuracy:
        args.best_accuracy = accuracy
        args.best_accuracy_epoch = args.epoch + 1
        if not args.debug_mode:
            torch.save({'state_dict':model.state_dict(), 'cls_loss':bce_loss, 'regression_loss':regression_loss, 'pr_auc': pr_auc, 'accuracy': accuracy}, os.path.join(args.checkpoint_dir, 'best_accuracy.ckpt'))
        print('Best accuracy model saved!')
        
#     if f1 > args.best_f1:
#         args.best_f1 = f1
#         args.best_f1_epoch = args.epoch + 1
#         if not args.debug_mode:
#             torch.save({'state_dict':model.state_dict(), 'precision':precision, 'recall':recall, 'f1':f1, 'f0.5':fprec, 'cls_loss':cls_loss, 'regression_loss':regression_loss, 'miou':miou}, os.path.join(args.checkpoint_dir, 'best_f1.ckpt'))
#         print('Best F1 model saved!')
            
#     if fprec > args.best_fprec:
#         args.best_fprec = fprec
#         args.best_fprec_epoch = args.epoch + 1
#         if not args.debug_mode:
#             torch.save({'state_dict':model.state_dict(), 'precision':precision, 'recall':recall, 'f1':f1, 'f0.5':fprec, 'cls_loss':cls_loss, 'regression_loss':regression_loss, 'miou':miou}, os.path.join(args.checkpoint_dir, 'best_f0.5.ckpt'))
#         print('Best F0.5 model saved!')
    
#     if miou > args.best_miou:
#         args.best_miou = miou
#         args.best_miou_epoch = args.epoch + 1
#         if not args.debug_mode:
#             torch.save({'state_dict':model.state_dict(), 'precision':precision, 'recall':recall, 'f1':f1, 'f0.5':fprec, 'cls_loss':cls_loss, 'regression_loss':regression_loss, 'miou':miou}, os.path.join(args.checkpoint_dir, 'best_miou.ckpt'))
#         print('Best miou model saved!')
    print(f'Epoch [{args.epoch+1}/{args.epochs}], CLS Loss: {bce_loss}, Accuracy: {accuracy:.2f}% PR AUC: {pr_auc:.4f}')
#     print(f'Epoch [{args.epoch+1}/{args.epochs}], CLS Loss: {bce_loss}, f loss: {f_loss}, contrastive loss: {contrastive_loss}, rank loss: {rank_loss}, PR AUC: {pr_auc:.4f}')
    return cls_loss, regression_loss, pr_auc, accuracy
#     return precision, recall, f1, fprec, cls_loss, regression_loss, miou
        