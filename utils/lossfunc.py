import torch.nn as nn
import torch.nn.functional as F
import torch

def regression_criterion(predict, roi_labels, labels):
    if not labels.sum():
        return (predict - predict).sum()
    mask = labels == 1
    return F.mse_loss(predict[mask], roi_labels[mask]) + (F.relu(predict[mask][:, 0] - predict[mask][:, 1].detach())).mean() + (F.relu(-predict[mask])).mean()

class combined_loss(nn.Module):
    def __init__(self, beta, epsilon=1e-7, margin=1.0, weights=None):
        super().__init__()
        self.beta = beta
        self.epsilon = epsilon
        self.margin = margin
        self.bce_loss = nn.BCEWithLogitsLoss()
     
    def surrogate_precision_recall_loss(self, y_logits, y_true):
        return 0
        y_pred = torch.sigmoid(y_logits)
        TP = (y_pred * y_true).sum()
        FP = ((1 - y_pred) * y_true).sum()
        FN = (y_pred * (1 - y_true)).sum()
        fbeta = (1 + self.beta**2) * TP / ((1 + self.beta**2) * TP + (self.beta**2) * FN + FP + self.epsilon)
        fbeta = fbeta.clamp(min=self.epsilon, max=1 - self.epsilon)
        return 1 - fbeta
    
    def pairwise_ranking_loss(self, y_score, y_true):
        """
        A basic implementation of pairwise ranking loss.
    
        Parameters:
        - y_true: Ground truth labels (1 for positive, 0 for negative).
        - y_score: Predicted scores or probabilities for being positive.
        - margin: Margin for the loss.
        """
        return 0
        # Find the indices of positive and negative examples
        positive_indices = (y_true == 1)
        negative_indices = (y_true == 0)
    
        # Get all pairwise combinations of positive and negative scores
        positive_scores = y_score[positive_indices]
        negative_scores = y_score[negative_indices]
    
        # Calculate the loss for all positive-negative pairs
        differences = positive_scores.unsqueeze(1) - negative_scores.unsqueeze(0)  # Broadcasting to get pairwise differences
        losses = torch.clamp(self.margin - differences, min=0)  # Apply hinge loss
    
        # Average the loss over all pairs
        return losses.mean()
    
    def contrastive_loss(self, y_score, y_true):
        """
        A contrastive loss that emphasizes separating positive and hard-negative samples.
    
        Parameters:
        - y_true: Ground truth labels.
        - y_score: Predicted scores.
        - margin: Margin for the loss.
        """
        return 0
        positive_scores = y_score[y_true == 1]
        negative_scores = y_score[y_true == 0]
    
        # Assuming negative_scores are sorted or filtered to represent hard negatives
        hard_negative_scores = negative_scores[:len(positive_scores)]  # Example strategy
    
        # Calculate the loss for positive and hard-negative pairs
        loss = torch.clamp(self.margin - (positive_scores.unsqueeze(1) - hard_negative_scores.unsqueeze(0)), min=0)
    
        return loss.mean()
    
    def forward(self, y_score, y_true):
        return 1 * self.bce_loss(y_score, y_true), 0 * self.surrogate_precision_recall_loss(y_score, y_true), 0 * self.contrastive_loss(y_score, y_true), 0 * self.pairwise_ranking_loss(y_score, y_true)

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-8):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(logits)
        
        # Flatten the tensors
        probs = probs.view(-1)
        targets = targets.view(-1)
        
        # Compute intersection and union
        intersection = (probs * targets).sum()
        union = probs.sum() + targets.sum()
        
        # Compute Dice coefficient
        dice_coeff = (2. * intersection + self.smooth) / (union + self.smooth)
        
        # Compute Dice loss
        dice_loss = 1. - dice_coeff
        
        return dice_loss
