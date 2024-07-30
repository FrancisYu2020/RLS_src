from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import os
import torch
import torch.nn.functional as F

def preprocess_data(data, labels, unravel=False, upperbound=255, z=3):
    if data is None:
        return data, labels
    # use z score to filter outlier values
    if unravel:
        N, C, H, W = data.shape
        labels = labels.reshape(-1)
        data = data.reshape(N * C,  H, W)
    data[data > upperbound] = upperbound
    data /= upperbound
    return data, labels

def preprocess_dataset(args):
    unravel = False
    train_data, train_label, cross_val_data, cross_val_label, internal_val_data, internal_val_label = prepare_datasets(args)
    train_data, train_label = preprocess_data(train_data, train_label, unravel=unravel)
    cross_val_data, cross_val_label = preprocess_data(cross_val_data, cross_val_label, unravel=unravel)
    internal_val_data, internal_val_label = preprocess_data(internal_val_data, internal_val_label, unravel=unravel)
    return train_data, train_label, cross_val_data, cross_val_label, internal_val_data, internal_val_label

def get_cnn_transforms(args, train=True):
    # cnn transformation
    framewise_transforms = ['2d-resnet18', '2d-baseline']
    frame_trans = args.architecture in framewise_transforms
    frame_trans = 1 # currently try all the methods with rgb_video
    if train:
        if frame_trans:
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Grayscale(num_output_channels=3),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=(-30, 30)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            transform = transforms.Compose([
                CustomToTensor(args.clip_len),
                transforms.RandomRotation(degrees=(-20, 20)),
            ])
    else:
        if frame_trans:
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            transform = transforms.Compose([
                CustomToTensor(args.clip_len),
            ])
    return transform

class CustomToTensor:
    def __init__(self, clip_len):
        pass

    def __call__(self, img):
        img = img.T
        img = torch.from_numpy(img)
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
        self.images = (data * 255).astype(np.uint8)
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
        if len(image.shape) == 2:
            image = self.transform(image.T)
        elif len(image.shape) == 3:
            video = []
            for c in range(image.shape[0]):
                video.append(self.transform(image[c].T).unsqueeze(1))
            image = torch.cat(video, dim=1)
        else:
            raise NotImplementedError(f"Invalid image/video shape {image.shape}")
        return image, label

def prepare_datasets(args):
    data_root = 'data'
    data_type = args.data_type
    if data_type == 'rgb_video':
        data_type = 'context'
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
    cross_val_data, cross_val_label = None, None
    internal_val_data, internal_val_label = None, None
    for patient_id in args.patients:
        data_paths.append(patients[patient_id])
    if args.val_type == 'internal-val':
        train_data = np.concatenate([np.load(os.path.join(path, args.clip_len_prefix + f'{data_type}_train_data.npy')).astype(np.float32) for path in data_paths], axis=0)
        train_label = np.concatenate([np.load(os.path.join(path, args.clip_len_prefix + f'{data_type}_train_label.npy')) for path in data_paths], axis=0)
        internal_val_data = np.concatenate([np.load(os.path.join(path, args.clip_len_prefix + f'{data_type}_val_data.npy')).astype(np.float32) for path in data_paths], axis=0)
        internal_val_label = np.concatenate([np.load(os.path.join(path, args.clip_len_prefix + f'{data_type}_val_label.npy')) for path in data_paths], axis=0)
    elif args.val_type == 'cross-val':
        assert 0 < args.split < 1, "Please set the train split ratio when cross patient val is used"
        split = max(1, int(args.split * len(data_paths)))
        print(f'Training set patients: {data_paths[:split]}')
        print(f'Validation set patients: {data_paths[split:]}')
        train_data = np.concatenate([np.load(os.path.join(path, args.clip_len_prefix + f'{data_type}_train_data_full.npy')).astype(np.float32) for path in data_paths[:split]], axis=0)
        train_label = np.concatenate([np.load(os.path.join(path, args.clip_len_prefix + f'{data_type}_train_label_full.npy')) for path in data_paths[:split]], axis=0)
        cross_val_data = np.concatenate([np.load(os.path.join(path, args.clip_len_prefix + f'{data_type}_val_data_full.npy')).astype(np.float32) for path in data_paths[split:]], axis=0)
        cross_val_label = np.concatenate([np.load(os.path.join(path, args.clip_len_prefix + f'{data_type}_val_label_full.npy')) for path in data_paths[split:]], axis=0)
    elif args.val_type == 'cross+internal-val':
        assert 0 < args.split < 1, "Please set the train split ratio when cross patient val is used"
        split = max(1, int(args.split * len(data_paths)))
        print(f'Training set patients: {data_paths[:split]}')
        print(f'Validation set patients: {data_paths[split:]}')
        train_data = np.concatenate([np.load(os.path.join(path, args.clip_len_prefix + f'{data_type}_train_data.npy')).astype(np.float32) for path in data_paths[:split]], axis=0)
        train_label = np.concatenate([np.load(os.path.join(path, args.clip_len_prefix + f'{data_type}_train_label.npy')) for path in data_paths[:split]], axis=0)
        cross_val_data = np.concatenate([np.load(os.path.join(path, args.clip_len_prefix + f'{data_type}_val_data_full.npy')).astype(np.float32) for path in data_paths[split:]], axis=0)
        cross_val_label = np.concatenate([np.load(os.path.join(path, args.clip_len_prefix + f'{data_type}_val_label_full.npy')) for path in data_paths[split:]], axis=0)
        internal_val_data = np.concatenate([np.load(os.path.join(path, args.clip_len_prefix + f'{data_type}_val_data.npy')).astype(np.float32) for path in data_paths[:split]], axis=0)
        internal_val_label = np.concatenate([np.load(os.path.join(path, args.clip_len_prefix + f'{data_type}_val_label.npy')) for path in data_paths[:split]], axis=0)
    else:
        raise NotImplementedError(f'{args.val_type} validation method not implemented!')
    
    train_data, train_label = process_data(args, train_data, train_label, 'train')
    all_cross_val_data, all_cross_val_label = [], []
    all_internal_val_data, all_internal_val_label = [], []
    np.random.seed(args.seed)
    for i in range(args.num_val_sets):
        temp_cross_val_data, temp_cross_val_label = process_data(args, cross_val_data, cross_val_label, 'val')
        temp_internal_val_data, temp_internal_val_label = process_data(args, internal_val_data, internal_val_label, 'val')
        if temp_cross_val_data is not None:
            all_cross_val_data.append(np.expand_dims(temp_cross_val_data, axis=0))
            all_cross_val_label.append(np.expand_dims(temp_cross_val_label, axis=0))
        if temp_internal_val_data is not None:
            all_internal_val_data.append(np.expand_dims(temp_internal_val_data, axis=0))
            all_internal_val_label.append(np.expand_dims(temp_internal_val_label, axis=0))
    cross_val_data = np.concatenate(all_cross_val_data, axis=0) if all_cross_val_data else None
    cross_val_label = np.concatenate(all_cross_val_label, axis=0) if all_cross_val_label else None
    internal_val_data = np.concatenate(all_internal_val_data, axis=0) if all_internal_val_data else None
    internal_val_label = np.concatenate(all_internal_val_label, axis=0) if all_internal_val_data else None

    if args.data_type == 'rgb_video':
        # for single frame evaluation baseline
        center_frame = train_data.shape[1] // 2
        train_data = train_data[:, center_frame, ...]
        cross_val_data = cross_val_data[:, center_frame, ...] if cross_val_data is not None else None
        internal_val_data = internal_val_data[:, center_frame, ...] if internal_val_data is not None else None
    return train_data, train_label, cross_val_data, cross_val_label, internal_val_data, internal_val_label

def process_data(args, data, label, mode):
    # helper function to process different train/val dataset
    assert mode in ['train', 'val']
    if data is None:
        return None, None
    H1, H2, W1, W2 = args.input_size
    if mode == 'train':
        return data[:, :, H1:H2, W1:W2], label
    else:
        data = data[:, :, H1:H2, W1:W2]
        n_chunks = 5
        if args.downsample_val == 1:
            # downsample the validation set due to it is too large
            chunk_length = len(data) // n_chunks
            if chunk_length <= 3000:
                # validation set is small enough, no need to downsample
                return data, label
            downsampled_data, downsampled_label = [], []
            start_indices = []
            for i in range(n_chunks):
                start_idx = np.random.randint(i * chunk_length, (i + 1) * chunk_length - 3000)
                start_indices.append(start_idx)
                downsampled_data.append(data[start_idx:start_idx + 3000, ...])
                downsampled_label.append(label[start_idx:start_idx + 3000, ...])
            print(start_indices)
            return np.concatenate(downsampled_data, axis=0), np.concatenate(downsampled_label, axis=0)
        else:
            return data, label
            
    
    
    
    