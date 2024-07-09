import cv2
import matplotlib.pyplot as plt
import numpy as np
import json
import pandas as pd
import torch
import torch.nn.functional as F
import os
import psutil
from utils.preprocess import *

# Get available memory
available_memory = psutil.virtual_memory().available

# Assuming np.float64
element_size = np.dtype(np.float64).itemsize

# Calculate max array size
max_array_size = available_memory / element_size

print(f"Maximum array size (np.float64): {max_array_size} elements")

# Data loading Version 2 - toy data previously
# data_root = '/scratch/bbsg/hangy6/RLS/data' # delta data root
data_root = 'data' # HAL data root
patient_dirs = [
    '',
    f'{data_root}/patient01-08-27-2023',
    '',
    '',
    '',
    f'{data_root}/patient05-02-15-2024',
    f'{data_root}/patient06-02-17-2024',
    '',
    '',
    f'{data_root}/patient09-03-01-2024',
    '',
    f'{data_root}/patient11-03-15-2024',
    '',
    f'{data_root}/patient13-03-31-2024',
    f'{data_root}/patient14-04-03-2024',
    f'{data_root}/patient15-04-12-2024',
    f'{data_root}/patient16-04-13-2024',
    f'{data_root}/patient17-04-14-2024',
    f'{data_root}/patient18-04-15-2024',
    f'{data_root}/patient19-04-16-2024',
    f'{data_root}/patient20-04-18-2024',
    f'{data_root}/patient21-04-26-2024',
    f'{data_root}/patient22-04-27-2024',
    f'{data_root}/patient23-04-28-2024',
    f'{data_root}/patient24-04-29-2024',
    f'{data_root}/patient25-05-10-2024',
    f'{data_root}/patient26-05-11-2024',
    f'{data_root}/patient27-05-13-2024',
    f'{data_root}/patient28-05-13-2024',
    f'{data_root}/patient29-05-14-2024',
    f'{data_root}/patient30-05-25-2024',
    f'{data_root}/patient31-05-27-2024',
    f'{data_root}/patient32-05-28-2024',
]

from tqdm import tqdm
for i in tqdm(range(25, 26)):
    patient_dir = patient_dirs[i]

    print(f'Memory available before loading data: {psutil.virtual_memory().available}')
    try:
        del matrices, timestamps, data
    except:
        pass
    
    data_file = os.path.join(patient_dir, 'raw_pressure_data.npy')
    timestamps_file = os.path.join(patient_dir, 'raw_timestamps.npy')
    if os.path.exists(data_file) and os.path.exists(timestamps_file):
        print('load existing pressure data...')
        matrices = np.load(data_file).astype(np.float32)
        timestamps = np.load(timestamps_file)
        print(f'successfully loaded pressure data from {data_file} and timestamps from {timestamps_file}')
    else:
        print('Generating pressure data from raw json file...')
        pressure_data = load_json(os.path.join(patient_dir, 'raw_data.json'))
        matrices, timestamps = get_pressure_matrices(pressure_data)
        np.save(data_file, np.array(matrices))
        np.save(timestamps_file, np.array(timestamps))
        print('Successfully saved pressure data and timestamps')
    # matrices = 5 * (matrices / 500)**2
    save_data = True
    balanced = True
    clip_len = 40
    
    EMG_label_path = f'{patient_dir}/positive_timestamps.csv'
    wake_mask = f'{patient_dir}/wake_mask.csv'
    train_data, train_label, val_data, val_label, _ = \
    make_tal_labels(matrices, timestamps, EMG_label_path, filter_timestamp=wake_mask, \
                clip_len=clip_len, balanced=balanced, overlap_train=False, \
                split_ratio=0.44, plot_statistics=False, calibrate_value=4.65)
    print(train_label.shape, train_label.sum(axis=-1), val_label.shape, val_label.sum(axis=-1))
    
    if save_data:
        np.save(f'{patient_dir}/win{clip_len}_tal_train_data.npy', train_data[:, :, :, :])
        np.save(f'{patient_dir}/win{clip_len}_tal_val_data.npy', val_data[:, :, :, :])
        np.save(f'{patient_dir}/win{clip_len}_tal_train_label.npy', train_label)
        np.save(f'{patient_dir}/win{clip_len}_tal_val_label.npy', val_label)
