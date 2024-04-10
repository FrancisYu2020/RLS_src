import numpy as np
import matplotlib.pyplot as plt
import json


def load_json(filename):
    try:
        with open(filename, 'r') as f:
            content = json.load(f)
    except:
        content = []
        with open(filename, 'r') as file:
            for line in file:
                # Process each line here
                line = line.strip()
                content.append(json.loads(line[:-1] if line[-1] == ',' else line))
        content = {'pressureData': content}
    
    pressure_data = content['pressureData']
    print(f'{len(pressure_data)} data records loaded from {filename}!')
    return pressure_data

def get_pressure_matrices(pressure_data):
    timestamps = []
    matrices = []
    for i in range(1, len(pressure_data)):
        timestamps.append(pressure_data[i]['dateTime'])
        matrices.append(np.array(pressure_data[i]['pressureMatrix']))
    matrices = np.array(matrices)
    timestamps = np.array(timestamps)
    return matrices, timestamps

def sliding_window_view(arr, window_size, stride):
    """
    Create a view of `arr` with a sliding window of size `window_size` moved by `stride`.
    
    Parameters:
    - arr: numpy array of 1 dimension.
    - window_size: size of the sliding window.
    - stride: step size between windows.
    
    Returns:
    - A 2D numpy array where each row is a window.
    """
    n = arr.shape[0]
    num_windows = (n - window_size) // stride + 1
    indices = np.arange(window_size)[None, :] + stride * np.arange(num_windows)[:, None]
    return arr[indices]

def get_roi_labels(labels):
    first_ones = np.full(labels.shape[0], -1)  # Fill with -1 to indicate rows with no 1s
    last_ones = np.full(labels.shape[0], -1)
    for i, row in enumerate(labels):
        # Find the first 1
        first_idx = np.argmax(row)
        if row[first_idx] == 1:
            first_ones[i] = first_idx
        # Find the last 1
        reversed_idx = np.argmax(row[::-1])
        if row[-reversed_idx-1] == 1:
            last_ones[i] = len(row) - reversed_idx - 1
    roi = np.vstack([first_ones, last_ones]).T
    return roi

def converted_timestamp(timestamp):
    ts = timestamp.split('T')[-1].split('-')[0].split(':')
    if int(ts[0]) < 20:
        ts[0] = int(ts[0]) + 24
    ts = int(ts[0]) * 3600 + int(ts[1]) * 60 + float(ts[2])
    return ts

def make_labels(data, timestamps, positive_csv, filter_timestamp=None, window_size=16, split_ratio=0.9, overlap_train=True, balanced=True, overnight=True, plot_statistics=False):
    '''
    data: sensing mat pressure matrices, dimension = (N, 16, 16)
    timestamps: the time stamps for each frame of pressure matrix, dimension = (N, ), each element is a formatted string
    positive_csv: path to the positive time intervals
    window_size: the time window size for each data sample
    split_ratio: train data : val data ratio
    overnight: indicate whether the experiment conducted is overnight, if True, the hours < 12 need to add 24 since they are overnight
    '''
    # filter out unuseful part
    if filter_timestamp:
        filter_ts = converted_timestamp(filter_timestamp)
        i = 0
        while i < len(timestamps) and converted_timestamp(timestamps[i]) <= filter_ts:
            i += 1
        data = data[i:]
        timestamps = timestamps[i:]
            
    # abandon the last few frames
    _, H, W = data.shape
    num_samples = len(timestamps) // window_size
    num_frames = num_samples * window_size
    data = data[:num_frames].reshape(-1, window_size, H, W)
    print(data.shape)
    timestamps = timestamps[:num_frames]
    labels = np.zeros(num_frames)
    
    # main loop variables, i for iteration through each frame, j for iteration through the positive regions
    i, j = 0, 0
    import pandas as pd
    # load positive regions
    positive_region = pd.read_csv(positive_csv)
    
    # initialize the time stamps of the very first positive region
    start_h, start_min, start_s, end_h, end_min, end_s = positive_region.iloc[j]
    start_ts, end_ts = start_h * 3600 + start_min * 60 + start_s, end_h * 3600 + end_min * 60 + end_s
    
    # loop over the whole data and annotate the data
    while i < len(timestamps):
        ts = converted_timestamp(timestamps[i])
        
        # label positive signals
        if start_ts <= ts <= end_ts:
            labels[i] = 1
        elif ts > end_ts:
            j += 1
            if j >= len(positive_region):
                print(f'timestamps at break: {timestamps[i]}')
                break
            start_h, start_min, start_s, end_h, end_min, end_s = positive_region.iloc[j]
            if start_h < 20:
                start_h += 24
                end_h += 24
            start_ts, end_ts = start_h * 3600 + start_min * 60 + start_s, end_h * 3600 + end_min * 60 + end_s
            
        i += 1
    if i % window_size:
        i = (i // window_size + 1) * window_size
    num_samples = i // window_size
    data = data[:num_samples]
    labels = labels[:i]
    labels = labels.reshape(-1, window_size)
    # filter_idx = data.max(axis=(1, 2, 3)) < 160
    # data = data[filter_idx]
    # labels = labels[filter_idx]
    # num_samples = filter_idx.sum()
    # print(f'Number of samples after filtering is {num_samples}')
    original_labels = labels
    
    # get roi labels
    roi = get_roi_labels(labels)
    labels = labels.sum(axis=1) / window_size
    
    # plot max value of each positive and negative data point
    if plot_statistics:
        positive_idx = labels > 0
        positive_data = data[positive_idx]
        max_vals, counts = np.unique(positive_data.max(axis=(1,2,3)), return_counts=True)
        print(f'leg movement regions max value is {max_vals.max()}, min max value is {max_vals.min()}')
        plt.plot(max_vals, counts, label='positive region max values')
    
        negative_idx = labels == 0
        negative_data = data[negative_idx]
        assert len(positive_data) + len(negative_data) == len(data)
        max_vals, counts = np.unique(negative_data.max(axis=(1,2,3)), return_counts=True)
        print(f'non leg movement regions max value is {max_vals.max()}, min max value is {max_vals.min()}')
        plt.plot(max_vals, counts, label='negative region max values')
        plt.title('max value statistics for ' + '/'.join(positive_csv.split('/')[:-1]))
        plt.legend()
        plt.savefig('/'.join(positive_csv.split('/')[:-1] + ['max_vals.png']))
    
    # split train and val data, label
    # use all the held out data samples for validation
    indices = np.arange(num_samples)
    val_idx = indices[int(split_ratio * num_samples):]
    val_data = data[val_idx]
    val_label = labels[val_idx]
    val_roi_label = roi[val_idx]
        
    # take all the positive samples while randomly sample the same amount of negative samples
    train_idx = indices[:int(split_ratio * num_samples)]
    train_data = data[train_idx]
    train_label = labels[train_idx]
    train_roi_label = roi[train_idx]
    if overlap_train:
        stride = window_size // 2
        train_data = sliding_window_view(train_data.reshape(-1, H, W), window_size, stride)
        train_label = sliding_window_view(original_labels[:int(split_ratio * num_samples)].reshape(-1), window_size, stride)
        train_roi_label = get_roi_labels(train_label)
        train_label = train_label.sum(axis=1) / window_size
        print(train_data.shape, train_label.shape, train_roi_label.shape)
    
    # if balanced is false then we directly take all the training data for training
    if not balanced:
        print(train_data.shape, (train_label > 0).sum(), val_data.shape, (val_label > 0).sum())
        return train_data, train_label, train_roi_label, val_data, val_label, val_roi_label
    positive_train_idx = train_label > 0
    positive_train_data = train_data[positive_train_idx]
    positive_train_label = train_label[positive_train_idx]
    
    negative_train_idx = train_label == 0
    negative_train_data = train_data[negative_train_idx]
    negative_train_label = train_label[negative_train_idx]
    negative_train_idx = np.random.choice(np.arange(len(negative_train_label)), size=positive_train_idx.sum(), replace=False)
    negative_train_data = negative_train_data[negative_train_idx]
    negative_train_label = negative_train_label[negative_train_idx]
    train_data = np.vstack((positive_train_data, negative_train_data))
    train_label = np.hstack((positive_train_label, negative_train_label))
    print(negative_train_data.shape, positive_train_data.shape, train_data.shape, val_data.shape, (val_label > 0).sum())
    print(len(train_label), (labels[int(len(labels) * split_ratio):] > 0).sum())
    return train_data, train_label, train_roi_label, val_data, val_label, val_roi_label

def cosine_similarity(A, B):
    '''
    Arguments:
        A: N x vector_size
        B: M x vector_size
    '''
    N, M = A.shape[0], B.shape[0]
    X = A.reshape(N, -1)
    Y = B.reshape(M, -1)
    X_norm = np.linalg.norm(X, axis=1, keepdims=True)
    Y_norm = np.linalg.norm(Y, axis=1, keepdims=True)
    # print(X_norm)
    # smooth denominator
    # X_norm += 1e-20
    # Y_norm += 1e-20
    normalized_X = X / X_norm
    normalized_Y = Y / Y_norm
    return normalized_X @ normalized_Y.T # (N, M)

def get_data_chunks(matrices, timestamps, window_size, start_timestamp=None, end_timestamp=None):
    if start_timestamp:
        start_timestamp = converted_timestamp(start_timestamp)
        i = 0
        while converted_timestamp(timestamps[i]) < start_timestamp:
            i += 1
        timestamps = timestamps[i:]
        matrices = matrices[:i]
    
    if end_timestamp:
        end_timestamp = converted_timestamp(end_timestamp)
        i = len(timestamps) - 1
        while converted_timestamp(timestamps[i]) > end_timestamp:
            i -= 1
        timestamps = timestamps[:i + 1]
        matrices = matrices[:i + 1]
    
    N, H, W = matrices.shape
    num_samples = N // window_size
    data = matrices[:num_samples * window_size].reshape(num_samples, window_size, H, W)
    return data
    
    