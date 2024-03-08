import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from matplotlib.patches import Rectangle
import os

def plot_validation(args, model, val_loader, metric, height=0.3, linewidth=1, offset=-20000, yoffset=0.35, fontsize=15, figsize=(10, 6)):
    '''
    args: input arguments from main function
    model: model to be evaluated 
    train_loader: training data dataloader
    epsilon: the laplace smoothing factor used to prevent division by 0
    cls_criterion: classification loss criterion
    regression_criterion: regression loss criterion
    '''
    plt.figure(figsize=figsize)
    
    def plot_regression(roi_labels, y_position):
        '''
        sub function to plot regression part
        '''
        if roi_labels[i][1] < 0:
            rect2 = Rectangle((curr_start, y_position), args.window_size, height, edgecolor='black', facecolor='black', linewidth=linewidth)
            plt.gca().add_patch(rect2)
        else:
            if roi_labels[i][0] < 0:
                roi_labels[i][0] = 0
            if roi_labels[i][0]:
                rect2 = Rectangle((curr_start, y_position), roi_labels[i][0], height, edgecolor='black', facecolor='black', linewidth=linewidth)
                plt.gca().add_patch(rect2)
            rect2 = Rectangle((roi_labels[i][0] + curr_start, y_position), roi_labels[i][-1] - roi_labels[i][0], height, edgecolor='red', facecolor='red', linewidth=linewidth)
            plt.gca().add_patch(rect2)
            if roi_labels[i][-1] < args.window_size - 1:
                rect2 = Rectangle((roi_labels[i][-1] + curr_start, y_position), args.window_size - 1 - roi_labels[i][-1], height, edgecolor='black', facecolor='black', linewidth=linewidth)
                plt.gca().add_patch(rect2)
            
    curr_start = 0
    for images, labels, roi_labels in tqdm(val_loader):
        images = images.to(args.device)
        labels = labels.to(args.device)
        roi_labels = roi_labels.to(args.device).float()
        if args.normalize_roi:
            roi_labels = (roi_labels * args.window_size).int()

        # Forward pass
        cls_loss, regression_loss = 0, 0
        if args.task == 'classification':
            cls_logits = model(images)
        else:
            cls_logits, regression_logits = model(images)
            mask = roi_labels[:, 0] != -1
            if args.normalize_roi:
                regression_logits = (regression_logits * args.window_size).int().cpu()
            else:
                regression_logits = regression_logits.int().cpu()
        
        _, predicted = torch.max(cls_logits, 1)
        for i in range(len(predicted)):
            roi_labels = roi_labels.cpu()
            
            # classification ground truth 
            rect0 = Rectangle((curr_start, 0), args.window_size, height, edgecolor='red' if labels[i] else 'black', facecolor='red' if labels[i] else 'black', linewidth=linewidth)
            plt.gca().add_patch(rect0)
            
            # classification prediction
            rect1 = Rectangle((curr_start, 1), args.window_size, height, edgecolor='red' if predicted[i] else 'black', facecolor='red' if predicted[i] else 'black', linewidth=linewidth)
            plt.gca().add_patch(rect1)
            
            # regression ground truth
            plot_regression(roi_labels, 2)
            
            # regression prediction
            plot_regression(regression_logits, 3)
                    
            curr_start += args.window_size
        
    plt.text(curr_start + offset, 0 + yoffset, 'classification ground truth', fontsize=fontsize)
    plt.text(curr_start + offset, 1 + yoffset, 'classification prediction', fontsize=fontsize)
    plt.text(curr_start + offset, 2 + yoffset, 'regression ground truth', fontsize=fontsize)
    plt.text(curr_start + offset, 3 + yoffset, 'regression prediction', fontsize=fontsize)
    plt.xlim(0, 55000)
    plt.ylim(0, 4)
    suffix = {'f1': '_f1_model.png', 'f0.5': '_f0.5_model.png', 'mIoU': '_mIoU_model.png'}
    plt.title(f'Patient{args.patient_id}_' + args.exp_name + f'_{metric}_model')
    plt.savefig(os.path.join(args.figure_dir, f'Patient{args.patient_id}' + suffix[metric]))