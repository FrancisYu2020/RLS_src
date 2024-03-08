import numpy as np
import os

class ArgParser:
    def __init__(self, window_size='win155_'):
        self.window_size = window_size

args = ArgParser()

def quick_print(note, data):
    print(note + ':', data.max(), (data / 255).mean(axis=(0,2,3)), (data / 255).std(axis=(0, 2, 3)))
    print((data / 255).mean(axis=(0,2,3)).mean(), (data / 255).mean(axis=(0,2,3)).std(), (data / 255).std())
    print()

data_paths = ['data/12-13-2023', 'data/02-17-2024']
# data_paths = ['data/12-13-2023', 'data/02-15-2024', 'data/02-17-2024']
for i in range(len(data_paths)):
    train_data = np.load(os.path.join(data_paths[i], args.window_size + 'sensing_mat_data_train.npy')).astype(np.float32)
    val_data = np.load(os.path.join(data_paths[i], args.window_size + 'sensing_mat_data_val.npy')).astype(np.float32)
    whole_data = np.concatenate([np.load(os.path.join(data_paths[i], args.window_size + 'sensing_mat_data_train.npy')).astype(np.float32), np.load(os.path.join(data_paths[i], args.window_size + 'sensing_mat_data_val.npy')).astype(np.float32)], axis=0)
    train_label = np.load(os.path.join(data_paths[i], args.window_size + 'EMG_label_train.npy'))
    val_label = np.load(os.path.join(data_paths[i], args.window_size + 'EMG_label_val.npy')).astype(np.float32)
    whole_label = np.concatenate([np.load(os.path.join(data_paths[i], args.window_size + 'EMG_label_train.npy')), np.load(os.path.join(data_paths[i], args.window_size + 'EMG_label_val.npy'))], axis=0)
    positive_train_idx, negative_train_idx = train_label > 0, train_label == 0
    positive_val_idx, negative_val_idx = val_label > 0, val_label == 0
    positive_whole_idx, negative_whole_idx = whole_label > 0, whole_label == 0
    positive_train_data, negative_train_data = train_data[positive_train_idx], train_data[negative_train_idx]
    positive_val_data, negative_val_data = val_data[positive_val_idx], val_data[negative_val_idx]
    positive_whole_data, negative_whole_data = whole_data[positive_whole_idx], whole_data[negative_whole_idx]

    quick_print('train', train_data)
    quick_print('val', val_data)
    quick_print('whole', whole_data)
    print()
    print()
    quick_print('positive train', positive_train_data)
    quick_print('positive val', positive_val_data)
    quick_print('positive whole', positive_whole_data)
    print()
    print()
    quick_print('negative train', negative_train_data)
    quick_print('negative val', negative_val_data)
    quick_print('negative whole', negative_whole_data)
    print('=================================================================================================')
    print()
    print()
    print()
