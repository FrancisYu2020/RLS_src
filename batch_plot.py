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
from tqdm import tqdm

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

def plot_func(patient_dir, matrices, lower_threshold=-1):
    # patient 15
    plt.clf()
    plt.figure(figsize=(10,4))
    min_value = 0
    max_value = 335
    bin_size = 5

    # Create bin edges
    bins = np.arange(min_value, max_value + bin_size, bin_size)

    # Bin the data
    selected_val = matrices.reshape(-1)[matrices.reshape(-1) > lower_threshold]
    hist, bin_edges = np.histogram(selected_val, bins=bins)

    # Plot only the first n bins
    plt.hist(selected_val, bins=bins, edgecolor='black', alpha=0.7)
    plt.xlim([min_value, max_value])
    plt.xlabel('Mat Sensor Pressure (mmHg)')
    plt.ylabel('Pixel Count')
    plt.title(patient_dir.split('/')[-1] + f' has {len(matrices)} frames')
    plt.yscale('log')
    plt.tight_layout()
    root = f'figures/statistics/threshold_{lower_threshold}'
    if not os.path.exists(root):
        os.makedirs(root, exist_ok=True)
    plt.savefig(os.path.join(root, patient_dir.split('/')[-1] + '.png'))

def plot_change(patient_dir, matrices, timestamps, labels, z=3):
    plt.clf()
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))
#     plt.figure(figsize=(10,4))
    
    mean = matrices.mean()
    std = matrices.std()
    lower_threshold = mean - 3 * std
    higher_threshold = mean + 3 * std
    
    matrices[matrices < lower_threshold] = 0
    matrices[matrices > higher_threshold] = higher_threshold
    print(matrices.min(), matrices.max(), matrices.mean(), matrices.std())
    padded_matrices = np.pad(matrices, pad_width=((0, 0), (1, 1), (1, 1)), mode='constant', constant_values=0)
    top_pad, left_pad, bottom_pad, right_pad = padded_matrices[:, :-2, 1:-1], padded_matrices[:, 1:-1, :-2], padded_matrices[:, 2:, 1:-1], padded_matrices[:, 1:-1, 2:]
    large_anomaly_positions = matrices == higher_threshold
    mean_neighbors = (top_pad + bottom_pad + left_pad + right_pad) / 4
    matrices[large_anomaly_positions] = mean_neighbors[large_anomaly_positions]
    print(matrices.min(), matrices.max(), matrices.mean(), matrices.std())
    gradient = matrices[1:] - matrices[:-1]
#     print(gradient.min(), gradient.max(), gradient.mean(), gradient.std())
    mask = gradient != 0
    mean_gradient = gradient.sum(axis=(1, 2)) / (mask.sum(axis=(1, 2)) + 1e-8)
#     print(gradient.sum(axis=(1, 2)).min(), gradient.sum(axis=(1, 2)).max(), gradient.sum(axis=(1, 2)).mean(), gradient.sum(axis=(1, 2)).std())
#     print(mask.sum(axis=(1, 2)).min(), mask.sum(axis=(1, 2)).max(), mask.sum(axis=(1, 2)).mean(), mask.sum(axis=(1, 2)).std())
    no_change_positions = np.where(mask.sum(axis=(1, 2)) == 0)
#     print(gradient.sum(axis=(1, 2))[no_change_positions])
#     print(mean_gradient.min(), mean_gradient.max(), mean_gradient.mean(), mean_gradient.std())

    axs[0].plot(np.arange(len(mean_gradient)), mean_gradient)
    axs[0].set_xlabel('Timestamp ')
    axs[0].set_ylabel('Pressure Change (mmHg)')
    axs[0].set_title(patient_dir.split('/')[-1] + f' has {len(matrices)} frames\n{timestamps[0]} - {timestamps[-1]}')
    
    axs[1].plot(np.arange(len(mean_gradient)), labels[1:])
    axs[1].set_xlabel('Timestamp ')
    axs[1].set_ylabel('0 for non-LM frame, 1 for LM frame')
    axs[1].set_title(patient_dir.split('/')[-1] + f' has {len(matrices)} frames\n{timestamps[0]} - {timestamps[-1]}, Correlation = {pearson_correlation(mean_gradient, labels):.4f}')
    
    plt.tight_layout()
    root = f'figures/statistics/grad-1_z-3'
    if not os.path.exists(root):
        os.makedirs(root, exist_ok=True)
    fig.savefig(os.path.join(root, patient_dir.split('/')[-1] + '.png'))

def plot_interval_change(patient_dir, matrices, timestamps, labels, interval=6, z=3):
    plt.clf()
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))
#     plt.figure(figsize=(10,4))
    
    mean = matrices.mean()
    std = matrices.std()
    lower_threshold = mean - 3 * std
    higher_threshold = mean + 3 * std
    
    matrices[matrices < lower_threshold] = 0
    matrices[matrices > higher_threshold] = higher_threshold
    print(matrices.min(), matrices.max(), matrices.mean(), matrices.std())
    
    padded_matrices = np.pad(matrices, pad_width=((0, 0), (1, 1), (1, 1)), mode='constant', constant_values=0)
    top_pad, left_pad, bottom_pad, right_pad = padded_matrices[:, :-2, 1:-1], padded_matrices[:, 1:-1, :-2], padded_matrices[:, 2:, 1:-1], padded_matrices[:, 1:-1, 2:]
    large_anomaly_positions = matrices == higher_threshold
    mean_neighbors = (top_pad + bottom_pad + left_pad + right_pad) / 4
    matrices[large_anomaly_positions] = mean_neighbors[large_anomaly_positions]
    print(matrices.min(), matrices.max(), matrices.mean(), matrices.std())
    
    
    gradient = matrices[1:] - matrices[:-1]
    num_intervals = len(gradient) // interval
    _, H, W = gradient.shape
    gradient = gradient[:num_intervals * interval].reshape(num_intervals, interval, H, W)
    labels = labels[1:num_intervals * interval + 1].reshape(num_intervals, interval).mean(axis=-1)
    labels = (labels >= 0.5) * 1.0
#     print(gradient.min(), gradient.max(), gradient.mean(), gradient.std())
    mask = gradient != 0
    mean_gradient = gradient.sum(axis=(1, 2, 3)) / (mask.sum(axis=(1, 2, 3)) + 1e-8)
#     print(gradient.sum(axis=(1, 2)).min(), gradient.sum(axis=(1, 2)).max(), gradient.sum(axis=(1, 2)).mean(), gradient.sum(axis=(1, 2)).std())
#     print(mask.sum(axis=(1, 2)).min(), mask.sum(axis=(1, 2)).max(), mask.sum(axis=(1, 2)).mean(), mask.sum(axis=(1, 2)).std())
    no_change_positions = np.where(mask.sum(axis=(1, 2, 3)) == 0)
#     print(gradient.sum(axis=(1, 2))[no_change_positions])
#     print(mean_gradient.min(), mean_gradient.max(), mean_gradient.mean(), mean_gradient.std())
    correlation = pearson_correlation(mean_gradient, labels)

    axs[0].plot(np.arange(len(mean_gradient)), mean_gradient)
    axs[0].set_xlabel('Timestamp ')
    axs[0].set_ylabel('Pressure Change (mmHg)')
    axs[0].set_title(patient_dir.split('/')[-1] + f' has {len(matrices)} frames\n{timestamps[0]} - {timestamps[-1]}')
    
    axs[1].plot(np.arange(len(mean_gradient)), labels)
    axs[1].set_xlabel('Timestamp ')
    axs[1].set_ylabel('Percentage of LM frames in the interval')
    axs[1].set_title(patient_dir.split('/')[-1] + f' has {len(matrices)} frames\n{timestamps[0]} - {timestamps[-1]}, Correlation = {correlation:.4f}')
    
    plt.tight_layout()
    root = f'figures/statistics/interval{interval}_grad-1_z-3'
    if not os.path.exists(root):
        os.makedirs(root, exist_ok=True)
    fig.savefig(os.path.join(root, patient_dir.split('/')[-1] + '.png'))
    return correlation

def plot_interval_max_change(patient_dir, matrices, timestamps, labels, interval=6, z=3):
    plt.clf()
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))
#     plt.figure(figsize=(10,4))
    
    mean = matrices.mean()
    std = matrices.std()
    lower_threshold = mean - 3 * std
    higher_threshold = mean + 3 * std
    
    matrices[matrices < lower_threshold] = 0
    matrices[matrices > higher_threshold] = higher_threshold
    print(matrices.min(), matrices.max(), matrices.mean(), matrices.std())
    
    padded_matrices = np.pad(matrices, pad_width=((0, 0), (1, 1), (1, 1)), mode='constant', constant_values=0)
    top_pad, left_pad, bottom_pad, right_pad = padded_matrices[:, :-2, 1:-1], padded_matrices[:, 1:-1, :-2], padded_matrices[:, 2:, 1:-1], padded_matrices[:, 1:-1, 2:]
    large_anomaly_positions = matrices == higher_threshold
    mean_neighbors = (top_pad + bottom_pad + left_pad + right_pad) / 4
    matrices[large_anomaly_positions] = mean_neighbors[large_anomaly_positions]
    print(matrices.min(), matrices.max(), matrices.mean(), matrices.std())
    
    
    gradient = matrices[1:] - matrices[:-1]
    num_intervals = len(gradient) // interval
    _, H, W = gradient.shape
    gradient = gradient[:num_intervals * interval].reshape(num_intervals, interval, H, W)
    labels = labels[1:num_intervals * interval + 1].reshape(num_intervals, interval).mean(axis=-1)
#     print(gradient.min(), gradient.max(), gradient.mean(), gradient.std())
    mask = gradient != 0
    max_gradient = gradient.max(axis=(1, 2, 3))

    axs[0].plot(np.arange(len(max_gradient)), max_gradient)
    axs[0].set_xlabel('Timestamp ')
    axs[0].set_ylabel('Pressure Change (mmHg)')
    axs[0].set_title(patient_dir.split('/')[-1] + f' has {len(matrices)} frames\n{timestamps[0]} - {timestamps[-1]}')
    
    axs[1].plot(np.arange(len(max_gradient)), labels)
    axs[1].set_xlabel('Timestamp ')
    axs[1].set_ylabel('Percentage of LM frames in the interval')
    axs[1].set_title(patient_dir.split('/')[-1] + f' has {len(matrices)} frames\n{timestamps[0]} - {timestamps[-1]}, Correlation = {pearson_correlation(max_gradient, labels):.4f}')
    
    plt.tight_layout()
    root = f'figures/statistics/interval{interval}_max_grad-1_z-3'
    if not os.path.exists(root):
        os.makedirs(root, exist_ok=True)
    fig.savefig(os.path.join(root, patient_dir.split('/')[-1] + '.png'))

def plot_interval_percentile_change(patient_dir, matrices, timestamps, labels, interval=6, percentile=90, z=3):
    plt.clf()
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))
#     plt.figure(figsize=(10,4))
    
    mean = matrices.mean()
    std = matrices.std()
    lower_threshold = mean - 3 * std
    higher_threshold = mean + 3 * std
    
    matrices[matrices < lower_threshold] = 0
    matrices[matrices > higher_threshold] = higher_threshold
    print(matrices.min(), matrices.max(), matrices.mean(), matrices.std())
    
    padded_matrices = np.pad(matrices, pad_width=((0, 0), (1, 1), (1, 1)), mode='constant', constant_values=0)
    top_pad, left_pad, bottom_pad, right_pad = padded_matrices[:, :-2, 1:-1], padded_matrices[:, 1:-1, :-2], padded_matrices[:, 2:, 1:-1], padded_matrices[:, 1:-1, 2:]
    large_anomaly_positions = matrices == higher_threshold
    mean_neighbors = (top_pad + bottom_pad + left_pad + right_pad) / 4
    matrices[large_anomaly_positions] = mean_neighbors[large_anomaly_positions]
    print(matrices.min(), matrices.max(), matrices.mean(), matrices.std())
    
    
    gradient = np.abs(matrices[1:] - matrices[:-1])
    num_intervals = len(gradient) // interval
    _, H, W = gradient.shape
    gradient = gradient[:num_intervals * interval].reshape(num_intervals, interval, H, W)
    labels = labels[1:num_intervals * interval + 1].reshape(num_intervals, interval).mean(axis=-1)
#     print(gradient.min(), gradient.max(), gradient.mean(), gradient.std())
    mask = gradient != 0
    percentile_gradient = np.percentile(gradient, percentile, axis=(1, 2, 3))

    axs[0].plot(np.arange(len(percentile_gradient)), percentile_gradient)
    axs[0].set_xlabel('Timestamp ')
    axs[0].set_ylabel('Pressure Change (mmHg)')
    axs[0].set_title(patient_dir.split('/')[-1] + f' has {len(matrices)} frames\n{timestamps[0]} - {timestamps[-1]}')
    
    axs[1].plot(np.arange(len(percentile_gradient)), labels)
    axs[1].set_xlabel('Timestamp ')
    axs[1].set_ylabel('Percentage of LM frames in the interval')
    axs[1].set_title(patient_dir.split('/')[-1] + f' has {len(matrices)} frames\n{timestamps[0]} - {timestamps[-1]}, Correlation = {pearson_correlation(percentile_gradient, labels):.4f}')
    
    plt.tight_layout()
    root = f'figures/statistics/interval{interval}_percentile{percentile}_grad-1_z-3'
    if not os.path.exists(root):
        os.makedirs(root, exist_ok=True)
    fig.savefig(os.path.join(root, patient_dir.split('/')[-1] + '.png'))

def plot_interval_averaged_percentile_change(patient_dir, matrices, timestamps, labels, interval=6, percentile=90, z=3):
    plt.clf()
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))
#     plt.figure(figsize=(10,4))
    
    mean = matrices.mean()
    std = matrices.std()
    lower_threshold = mean - 3 * std
    higher_threshold = 5
#     higher_threshold = mean + 3 * std
    
    matrices[matrices < lower_threshold] = 0
    matrices[matrices > higher_threshold] = 0
#     matrices[matrices > higher_threshold] = higher_threshold
    print(matrices.min(), matrices.max(), matrices.mean(), matrices.std())
    
    padded_matrices = np.pad(matrices, pad_width=((0, 0), (1, 1), (1, 1)), mode='constant', constant_values=0)
    top_pad, left_pad, bottom_pad, right_pad = padded_matrices[:, :-2, 1:-1], padded_matrices[:, 1:-1, :-2], padded_matrices[:, 2:, 1:-1], padded_matrices[:, 1:-1, 2:]
    large_anomaly_positions = matrices == higher_threshold
    mean_neighbors = (top_pad + bottom_pad + left_pad + right_pad) / 4
    matrices[large_anomaly_positions] = mean_neighbors[large_anomaly_positions]
    print(matrices.min(), matrices.max(), matrices.mean(), matrices.std())
    
    
    gradient = np.abs(matrices[1:] - matrices[:-1])
    num_intervals = len(gradient) // interval
    _, H, W = gradient.shape
    gradient = gradient[:num_intervals * interval].reshape(num_intervals, interval, H, W)
    labels = labels[1:num_intervals * interval + 1].reshape(num_intervals, interval).mean(axis=-1)
#     print(gradient.min(), gradient.max(), gradient.mean(), gradient.std())
    percentile_gradient = np.percentile(gradient, percentile, axis=(1, 2, 3))
    gradient -= percentile_gradient.reshape(-1, 1, 1, 1)
    gradient[gradient < 0] = 0
    mask = gradient > 0
    percentile_gradient = gradient.sum(axis=(1, 2, 3)) / (mask.sum(axis=(1,2,3)) + 1e-10)
    

    axs[0].plot(np.arange(len(percentile_gradient)), percentile_gradient)
    axs[0].set_xlabel('Timestamp ')
    axs[0].set_ylabel('Pressure Change (mmHg)')
    axs[0].set_title(patient_dir.split('/')[-1] + f' has {len(matrices)} frames\n{timestamps[0]} - {timestamps[-1]}')
    
    axs[1].plot(np.arange(len(percentile_gradient)), labels)
    axs[1].set_xlabel('Timestamp ')
    axs[1].set_ylabel('Percentage of LM frames in the interval')
    axs[1].set_title(patient_dir.split('/')[-1] + f' has {len(matrices)} frames\n{timestamps[0]} - {timestamps[-1]}, Correlation = {pearson_correlation(percentile_gradient, labels):.4f}')
    
    plt.tight_layout()
    root = f'figures/statistics/interval{interval}_averaged-percentile{percentile}_grad-1_z-3'
    if not os.path.exists(root):
        os.makedirs(root, exist_ok=True)
    fig.savefig(os.path.join(root, patient_dir.split('/')[-1] + '.png'))


def plot_percentile(patient_dir, matrices):
    plt.clf()
    plt.figure(figsize=(10, 6))
    positive_matrices = matrices.reshape(-1)[matrices.reshape(-1) > 0]
    percentiles = np.percentile(positive_matrices, 0.1 * np.arange(1000))
    percentile_change = percentiles[1:] - percentiles[:-1]
    max_change = percentile_change.argmax()
    plt.bar(0.1 * np.arange(1000), percentiles)
    plt.yscale('log')
    plt.xlabel('Timestamp ')
    plt.ylabel('Pressure Change (mmHg)')
    plt.title(patient_dir.split('/')[-1] + f' has {len(matrices)} frames\n{timestamps[0]} - {timestamps[-1]}\n max change percentile-{max_change}={percentiles[max_change]}, delta={percentile_change[max_change]}')
        
    plt.tight_layout()
    root = f'figures/statistics/find_reasonable_values'
    if not os.path.exists(root):
        os.makedirs(root, exist_ok=True)
    plt.savefig(os.path.join(root, patient_dir.split('/')[-1] + '.png'))


    
def pearson_correlation(x, y):
    assert len(x) == len(y), "input must have the same length!"
    x_mean = x.mean()
    y_mean = y.mean()
    n = len(x)
    numerator = (x * y).sum() - n * x_mean * y_mean
    denominator = np.sqrt(((x**2).sum() - n * x_mean**2) * ((y**2).sum() - n * y_mean**2))
    return numerator / (denominator + 1e-10)

correlations = []

for i in tqdm(range(15, 33)):
    patient_dir = patient_dirs[i]

    print(f'Memory available before loading data: {psutil.virtual_memory().available}')
    try:
        del matrices, timestamps, data
    except:
        pass

    clip_len = 156
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
    matrices = 5 * (matrices / 500)**2
    print(matrices.shape, matrices.max(), np.unravel_index(matrices.argmax(), shape=matrices.shape))
    save_data = False
    balanced = False
    clip_len = 40
        
    EMG_label_path = f'{patient_dir}/positive_timestamps.csv'
    wake_mask = None
#     wake_mask = f'{patient_dir}/wake_mask.csv'
    train_data, train_label, _, _, timestamps = \
    make_tal_labels(matrices, timestamps, EMG_label_path, filter_timestamp=wake_mask, \
                clip_len=clip_len, balanced=balanced, overlap_train=False, \
                split_ratio=1.0, plot_statistics=False, calibrate_value=4.65)
    print(train_data.min(), train_data.max())
    print(train_label.reshape(-1).shape, np.concatenate(train_data, axis=0).shape)
#     plot_percentile(patient_dir, matrices)
#     plot_interval_averaged_percentile_change(patient_dir, np.concatenate(train_data, axis=0), timestamps, train_label.reshape(-1), percentile=75)
    correlations.append(plot_interval_change(patient_dir, np.concatenate(train_data, axis=0), timestamps, train_label.reshape(-1), interval=100))
    #     plot_func(patient_dir, matrices, lower_threshold=0)
print(np.array(correlations).mean())

    