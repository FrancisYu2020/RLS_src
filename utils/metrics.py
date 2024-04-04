import torch

def calculate_iou(gt, pred, print_input=False):
    """
    Calculate Intersection over Union (IoU) for a single pair of segments
    :param gt: PyTorch tensor of ground truth segments, shape (N, 2)
    :param pred: PyTorch tensor of predicted segments, shape (N, 2)
    :return: IoU for each segment pair
    """
    mask = gt[:, 0] >= 0
    if (mask).sum() == 0:
        return torch.tensor([]).cuda()
    gt = gt[mask]
    pred = pred[mask]
    if print_input:
        print("prediction: ", pred)
        print("ground truth: ", gt)
    # Calculate intersection
    inter_start = torch.max(gt[:, 0], pred[:, 0])
    inter_end = torch.min(gt[:, 1], pred[:, 1])
    intersection = (inter_end - inter_start).clamp(min=0)

    # Calculate union
    union = (gt[:, 1] - gt[:, 0]) + (pred[:, 1] - pred[:, 0]) - intersection

    # Compute IoU, avoid division by zero
    iou = intersection / union.where(union > 0, torch.ones_like(union))
    
    return iou

def calculate_miou(gt, pred):
    """
    Calculate mean Intersection over Union (mIoU)
    :param gt: PyTorch tensor of ground truth segments, shape (N, 2)
    :param pred: PyTorch tensor of predicted segments, shape (N, 2)
    :return: mIoU
    """
    if not gt.size(0):
        return torch.tensor(0)
    iou = calculate_iou(gt, pred)
    miou = iou.mean()
    return miou

def f_beta_score(beta, precision, recall, epsilon):
    '''
    Calculate f beta score,
    :param beta: beta = 1 for f1 score, beta = 0.5 for more weight on precision
    :param epsilon: smoothing factor
    '''
    return (1 + beta**2) * precision * recall / (beta**2 * precision + recall + epsilon)
    
# Example usage:
# gt and pred are PyTorch tensors of shape (N, 2), where N is the number of segments
# Each row in gt and pred is [start, end] for the segments
# gt = torch.tensor([[-1,20],[15,25]], dtype=torch.float32)
# pred = torch.tensor([[15, 25], [20, 30]], dtype=torch.float32)
# miou = calculate_miou(gt, pred)
# print(f"Mean IoU: {miou.item()}")
# print(gt)
