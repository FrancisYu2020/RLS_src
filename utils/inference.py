import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import fbeta_score, precision_score
from matplotlib.patches import Rectangle
import os
        
def inference(args, model, val_loader):
    '''
    args: input arguments from main function
    model: model to be evaluated 
    val_loader: validation data dataloader
    '''
    model.eval()
    y_score, y_true = [], []
    for i, (images, labels) in tqdm(enumerate(val_loader)):
        images = images.to(args.device)
        labels = labels.to(args.device).float()

        # Forward pass
        with torch.no_grad():
            cls_logits = model(images).squeeze()
        y_score.append(cls_logits.reshape(-1))
        y_true.append(labels.reshape(-1))
    y_score = torch.sigmoid(torch.cat(y_score)).cpu().detach().numpy()
    y_pred = y_score > 0.5
    y_true = torch.cat(y_true).cpu().detach().numpy()
    return y_pred, y_true
        
def plot_inference(args, y_pred, y_true, metric, height=0.3, linewidth=1, offset=-20000, yoffset=0.35, fontsize=15, figsize=(10, 6)):
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
    f_prec = fbeta_score(y_true, y_pred, beta=0.5)
    f_1 = fbeta_score(y_true, y_pred, beta=1)
    precision = precision_score(y_true, y_pred, zero_division=0)
    miou = np.logical_and(y_true, y_pred).sum() / (np.logical_or(y_true, y_pred).sum() + 1e-8)
    
    for i in tqdm(range(len(y_pred))):
            
        # classification ground truth 
        rect0 = Rectangle((i, 0), args.clip_len, height, edgecolor='red' if y_true[i] else 'black', facecolor='red' if y_true[i] else 'black', linewidth=linewidth)
        plt.gca().add_patch(rect0)
            
        # classification prediction
        rect1 = Rectangle((i, 1), args.clip_len, height, edgecolor='red' if y_pred[i] else 'black', facecolor='red' if y_pred[i] else 'black', linewidth=linewidth)
        plt.gca().add_patch(rect1)
                    
    plt.text(offset, 0 + yoffset, 'classification ground truth', fontsize=fontsize)
    plt.text(offset, 1 + yoffset, 'classification prediction', fontsize=fontsize)
    plt.text((plt.xlim()[1] + plt.xlim()[0]) * 3 / 5, 6 * yoffset, f'miou={miou:.4f}   f0.5={f_prec:.4f}\nf1={f_1:.4f}   precision={precision}', fontsize=fontsize)
    plt.xlim(0, len(y_pred))
    plt.ylim(0, 3)
    if args.cross_val_type:
        args.val_type = 'heldout-val'
    else:
        args.val_type = 'individual-val'
    plt.title(f'{args.val_type}' + f'_best_{metric}')
    plt.savefig(os.path.join(args.checkpoint_folder, f'{args.val_type}' + f'_best_{metric}.png'))