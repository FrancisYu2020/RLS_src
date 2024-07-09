import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from matplotlib.patches import Rectangle
import os
from sklearn.metrics import fbeta_score, precision_score
from .models import *
import numpy as np

def inference_and_plot(args, val_loader, metric):
    assert metric in ['loss', 'pr_auc', 'accuracy', 'f1', 'f0.5', 'miou']
    model_name = f'best_{metric}.ckpt'
    model = load_model(args, model_name)
    model.to(args.device)
    model.eval()
    y_score, y_true = [], []
    with torch.no_grad():
        for images, labels in tqdm(val_loader):
            images = images.to(args.device)
            labels = labels.to(args.device)
            y_score.append(model(images).squeeze())
            y_true.append(labels)
    y_score = torch.cat(y_score, dim=0).reshape(-1)
    y_true = torch.cat(y_true, dim=0).reshape(-1)
    plot_validation(args, y_score, y_true, metric)

def plot_validation(args, y_score, y_true, metric, height=0.3, linewidth=1, offset=-20000, yoffset=0.35, fontsize=15, figsize=(10, 6)):
    '''
    args: input arguments from main function
    model: model to be evaluated 
    train_loader: training data dataloader
    epsilon: the laplace smoothing factor used to prevent division by 0
    cls_criterion: classification loss criterion
    regression_criterion: regression loss criterion
    '''
    assert metric in ['loss', 'pr_auc', 'accuracy', 'f1', 'f0.5', 'miou']
    plt.figure(figsize=figsize)
    y_score = torch.sigmoid(y_score).detach().cpu().numpy()
    y_true = y_true.cpu().numpy()
    y_pred = y_score >= 0.5
    f_prec = fbeta_score(y_true, y_pred, beta=0.5)
    f_1 = fbeta_score(y_true, y_pred, beta=1)
    precision = precision_score(y_true, y_pred, zero_division=0)
    miou = np.logical_and(y_true, y_pred).sum() / (np.logical_or(y_true, y_pred).sum() + 1e-8)
    
    for i in range(len(y_pred)):
            
        # classification ground truth 
        rect0 = Rectangle((i, 0), args.clip_len, height, edgecolor='red' if y_true[i] else 'black', facecolor='red' if y_true[i] else 'black', linewidth=linewidth)
        plt.gca().add_patch(rect0)
            
        # classification prediction
        rect1 = Rectangle((i, 1), args.clip_len, height, edgecolor='red' if y_pred[i] else 'black', facecolor='red' if y_pred[i] else 'black', linewidth=linewidth)
        plt.gca().add_patch(rect1)
                    
    plt.text(offset, 0 + yoffset, 'classification ground truth', fontsize=fontsize)
    plt.text(offset, 1 + yoffset, 'classification prediction', fontsize=fontsize)
    plt.text((plt.xlim()[1] + plt.xlim()[0]) * 3 / 5, 6 * yoffset, f'miou={miou:.4f}   f0.5={f_prec:.4f}\nf1={f_1:.4f}   precision={precision}', fontsize=fontsize)
    plt.xlim(0, 55000)
    plt.ylim(0, 3)
    plt.title(f'Patient{args.patient_id}_' + args.exp_name + f'_best_{metric}')
    plt.savefig(os.path.join(args.figure_dir, f'Patient{args.patient_id}' + f'_best_{metric}.png'))