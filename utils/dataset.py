from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np

def get_cnn_transforms(window_size, train=True):
    # cnn transformation
    if train:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomRotation(degrees=(-20, 20)),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
#             transforms.Normalize([0.47] * window_size, [0.2] * window_size),
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
#             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            transforms.Normalize([0.47] * window_size, [0.2] * window_size),
        ])
    return transform

def get_vit_transforms(train=True):
    # ViT transformation
    if train:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    return transform

class RLSDataset(Dataset):
    def __init__(self, architecture_name, data, label, roi, transform=None, normalize_roi=False):
        """
        Args:
            data: (N, C, H, W)
            label: 0 or 1 indicating whether this is a positive region or not
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.transform = transform
        self.images = data
        self.label = label
        self.roi = roi
        self.dimension = int(architecture_name[0])
        if normalize_roi:
            self.roi = self.roi * 1.0 / data.shape[1]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image, label, roi = self.images[idx].T, self.label[idx], self.roi[idx]

        if self.transform:
            image = self.transform(image)
        if self.dimension == 3:
            image = image.unsqueeze(0)
            
        return image, label, roi