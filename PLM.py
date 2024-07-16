import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def count_groups(arr):
    # Ensure the array is a numpy array
    arr = np.asarray(arr)
    
    # Check for the change points in the array
    change_points = np.diff(arr)
    
    # A group of 1s starts when there's a change from 0 to 1
    starts = np.where(change_points == 1)[0]
    
    # Additionally, if the array starts with a 1, it's also a group start
    if arr[0] == 1:
        starts = np.insert(starts, 0, -1)  # Add a start at the beginning
    
    # The number of groups is the number of start points
    num_groups = len(starts)
    
    return num_groups

def pearson_correlation(x, y):
    # Ensure the inputs are numpy arrays
    x = np.asarray(x)
    y = np.asarray(y)
    
    # Compute the correlation matrix
    corr_matrix = np.corrcoef(x, y)
    
    # The Pearson correlation coefficient is the off-diagonal element
    pearson_corr = corr_matrix[0, 1]
    
    return pearson_corr

all_lm_counts = []
all_mean_data = []

for i in tqdm(range(15, 43)):
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
        f'{data_root}/patient15-04-12-2024-relabeled',
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
        f'{data_root}/patient29-05-14-2024-relabeled',
        f'{data_root}/patient30-05-25-2024',
        f'{data_root}/patient31-05-27-2024',
        f'{data_root}/patient32-05-28-2024',
        '',
        f'{data_root}/patient34-05-30-2024',
        f'{data_root}/patient35-06-06-2024',
        f'{data_root}/patient36-06-07-2024',
        f'{data_root}/patient37-06-08-2024',
        f'{data_root}/patient38-06-09-2024',
        f'{data_root}/patient39-06-10-2024',
        f'{data_root}/patient40-06-11-2024',
        f'{data_root}/patient41-06-12-2024',
        f'{data_root}/patient42-06-21-2024',
    ]
    
    clip_len = 6
    patient_dir = patient_dirs[i]
    if not patient_dir:
        continue
    data = np.load(f'{patient_dir}/win{clip_len}_context_val_data_full.npy')[:, 3, ...]
    label = np.load(f'{patient_dir}/win{clip_len}_context_val_label_full.npy')
    gradient = np.abs(data[1:] - data[:-1])
    label = label[1:]
    n_hours = len(gradient) // 21600
    _, H, W = gradient.shape
    mean_data = list(gradient[:n_hours * 21600].reshape(n_hours, 21600, H, W).mean(axis=(1, 2, 3)))
    temp = label[:n_hours * 21600].reshape(n_hours, 21600)
    lm_counts = []
    for h in range(n_hours):
        lm_counts.append(count_groups(temp[h]))
    lm_counts.append(count_groups(label[n_hours * 21600:]))
    mean_data.append(gradient[n_hours * 21600:].mean())
    all_lm_counts += lm_counts
    all_mean_data += mean_data
    lm_counts = np.array(lm_counts)
    mean_data = np.array(mean_data)
    cor = pearson_correlation(mean_data, lm_counts)
    indices = np.argsort(lm_counts)
    plt.clf()
    plt.plot(lm_counts[indices], mean_data[indices])
    plt.title(f'Patient{i}: correlation = {cor}')
    plt.xlabel('Leg movements / h')
    plt.ylabel('average of mat value change')
    plt.savefig(f'figures/hourly/{i}.png')

all_lm_counts = np.array(all_lm_counts)
all_mean_data = np.array(all_mean_data)
indices = np.argsort(all_lm_counts)
cor = pearson_correlation(all_lm_counts, all_mean_data)
plt.clf()
plt.plot(all_lm_counts[indices], all_mean_data[indices])
plt.title(f'Patient 15 - 42: correlation = {cor}')
plt.xlabel('Leg movements / h')
plt.ylabel('average of mat value change')
plt.savefig(f'figures/hourly/all_patient.png')