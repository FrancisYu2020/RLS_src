from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import os
import torch
import torch.nn.functional as F

def preprocess_data(data, labels, unravel=False, upperbound=255, z=3):
    # use z score to filter outlier values
    if unravel:
        N, C, H, W = data.shape
        labels = labels.reshape(-1)
        data = data.reshape(N * C,  H, W)
    data[data > upperbound] = upperbound
    data /= upperbound
#     flat_data = data.reshape(-1)
#     mask = flat_data > 0
#     mean = flat_data[mask].mean()
#     std = flat_data[mask].std()
#     lower_threshold = max(0, mean - 3 * std)
#     upper_threshold = mean + 3 * std
#     data[data < lower_threshold] = 0
#     data[data > upper_threshold] = upper_threshold
#     data = (np.array(data) * 255).astype(np.uint8)
#     data = data[:len(data)//2]
#     labels = labels[:len(labels)//2]
    return data, labels

def preprocess_dataset(args):
    train_data, train_label, val_data, val_label = prepare_datasets(args)
    print(train_data.max(), val_data.max())
    train_data, train_label = preprocess_data(train_data, train_label, unravel=('conv' not in args.architecture))
    val_data, val_label = preprocess_data(val_data, val_label, unravel=('conv' not in args.architecture))
    return train_data, train_label, val_data, val_label

def get_cnn_transforms(args, train=True):
    # cnn transformation
    if train:
        if 'efficientnet' in args.architecture:
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Grayscale(num_output_channels=3),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=(-30, 30)),
                transforms.ToTensor(),
            ])
        else:
            transform = transforms.Compose([
#             transforms.ToTensor(),
                CustomToTensor(args.clip_len, args.normalize_data),
                transforms.RandomRotation(degrees=(-20, 20)),
                # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
    else:
        if 'efficientnet' in args.architecture:
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
            ])
        else:
            transform = transforms.Compose([
#             transforms.ToTensor(),
                CustomToTensor(args.clip_len, args.normalize_data),
    #             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
    return transform

class CustomToTensor:
    def __init__(self, clip_len, normalize=True, mean=None, std=None):
        self.normalize = normalize

    def __call__(self, img):
        img = img.T
        img = torch.from_numpy(img)
        if self.normalize:
            # Normalize the tensor
            mean = torch.tensor(self.mean).view(-1, 1, 1)
            std = torch.tensor(self.std).view(-1, 1, 1)
            img = (img - mean) / std
        
        return img
    
class RLSDataset(Dataset):
    def __init__(self, args, data, label, transform=None):
        """
        Args:
            data: (N, C, H, W)
            label: 0 or 1 indicating whether this is a positive region or not
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.transform = transform
        self.images = data
        self.label = label
        self.dimension = int(args.architecture[0])
        if not transform:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
            ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image, label = self.images[idx], self.label[idx]
        image = self.transform(image.T)
        if self.dimension == 3:
            image = image.unsqueeze(0)
        return image, label

def prepare_datasets(args):
    data_root = 'data'
    data_type = args.data_type
    patients = {
        15: 'data/patient15-04-12-2024-relabeled', 
        16: 'data/patient16-04-13-2024',
        17: 'data/patient17-04-14-2024',
        18: 'data/patient18-04-15-2024',
        19: 'data/patient19-04-16-2024',
        20: 'data/patient20-04-18-2024',
        21: 'data/patient21-04-26-2024',
        22: 'data/patient22-04-27-2024',
        23: 'data/patient23-04-28-2024',
        24: 'data/patient24-04-29-2024',
        25: 'data/patient25-05-10-2024',
        26: 'data/patient26-05-11-2024',
        27: 'data/patient27-05-13-2024',
        28: 'data/patient28-05-13-2024',
        29: 'data/patient29-05-14-2024-relabeled',
        30: 'data/patient30-05-25-2024',
        31: 'data/patient31-05-27-2024',
        32: 'data/patient32-05-28-2024',
        33: '',
        34: f'{data_root}/patient34-05-30-2024',
        35: f'{data_root}/patient35-06-06-2024',
        36: f'{data_root}/patient36-06-07-2024',
        37: f'{data_root}/patient37-06-08-2024',
        38: f'{data_root}/patient38-06-09-2024',
        39: f'{data_root}/patient39-06-10-2024',
        40: f'{data_root}/patient40-06-11-2024',
        41: f'{data_root}/patient41-06-12-2024',
        42: f'{data_root}/patient42-06-21-2024',
    }

    data_paths = []
    for patient_id in args.patients:
        data_paths.append(patients[patient_id])
    if args.cross_val_type == 0:
        train_data = np.concatenate([np.load(os.path.join(path, args.clip_len_prefix + f'{data_type}_train_data.npy')).astype(np.float32) for path in data_paths], axis=0)
        train_label = np.concatenate([np.load(os.path.join(path, args.clip_len_prefix + f'{data_type}_train_label.npy')) for path in data_paths], axis=0)
        val_data = np.concatenate([np.load(os.path.join(path, args.clip_len_prefix + f'{data_type}_val_data.npy')).astype(np.float32) for path in data_paths], axis=0)
        val_label = np.concatenate([np.load(os.path.join(path, args.clip_len_prefix + f'{data_type}_val_label.npy')) for path in data_paths], axis=0)
    elif args.cross_val_type == 1:
        assert 0 < args.split < 1, "Please set the train split ratio when cross_val_type is 1"
        split = max(1, int(args.split * len(data_paths)))
        print(f'Training set patients: {data_paths[:split]}')
        print(f'Validation set patients: {data_paths[split:]}')
        train_data = np.concatenate([np.load(os.path.join(path, args.clip_len_prefix + f'{data_type}_train_data_full.npy')).astype(np.float32) for path in data_paths[:split]], axis=0)
        train_label = np.concatenate([np.load(os.path.join(path, args.clip_len_prefix + f'{data_type}_train_label_full.npy')) for path in data_paths[:split]], axis=0)
        val_data = np.concatenate([np.load(os.path.join(path, args.clip_len_prefix + f'{data_type}_val_data_full.npy')).astype(np.float32) for path in data_paths[split:]], axis=0)
        val_label = np.concatenate([np.load(os.path.join(path, args.clip_len_prefix + f'{data_type}_val_label_full.npy')) for path in data_paths[split:]], axis=0)
    else:
        leave_out_idx = args.cross_val_type - 1
        train_data = np.concatenate([np.load(os.path.join(data_paths[i], args.clip_len_prefix + f'{data_type}_train_data.npy')).astype(np.float32) for i in range(len(data_paths)) if i != leave_out_idx], axis=0)
        train_label = np.concatenate([np.load(os.path.join(data_paths[i], args.clip_len_prefix + f'{data_type}_train_label.npy')) for i in range(len(data_paths)) if i != leave_out_idx], axis=0)
        val_data = np.concatenate([np.load(os.path.join(data_paths[leave_out_idx], args.clip_len_prefix + f'{data_type}_train_data.npy')).astype(np.float32), np.load(os.path.join(data_paths[leave_out_idx], args.clip_len_prefix + f'{data_type}_val_data.npy')).astype(np.float32)], axis=0)
        val_label = np.concatenate([np.load(os.path.join(data_paths[leave_out_idx], args.clip_len_prefix + f'{data_type}_train_label.npy')), np.load(os.path.join(data_paths[leave_out_idx], args.clip_len_prefix + f'{data_type}_val_label.npy'))], axis=0)
    H1, H2, W1, W2 = args.input_size
    train_data = train_data[:, :, H1:H2, W1:W2]
    val_data = val_data[:, :, H1:H2, W1:W2]
    return train_data, train_label, val_data, val_label