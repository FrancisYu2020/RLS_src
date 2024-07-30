from main import args
from utils.dataset import *
from utils.models import *
from utils import get_checkpoint_path, prepare_folder
from train import val_loop, compute_metrics
import os
import torch
import matplotlib.pyplot as plt

# adjust argument for testing
args.downsample_val = 0
args.val_type = 'cross+internal-val'
args.exp_name = get_checkpoint_path(args)
args.checkpoint_dir = prepare_folder(args, skip_results=False)
args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        
train_data, train_label, _, _, _, _ = preprocess_dataset(args)
# option 1
pressure_change = np.abs(train_data[:, 1:, ...] - train_data[:, :-1, ...]).mean(axis=(1,2,3))
# # option 2
# pressure_change = np.abs(train_data[:, 1:, ...] - train_data[:, :-1, ...])
# sum_pressure_change = pressure_change.sum(axis=(1,2,3))
# mask = (pressure_change > 0).sum(axis=(1,2,3))
# pressure_change = sum_pressure_change / (mask + 1e-8)

positive = train_label > 0
negative = train_label == 0
values, counts = np.unique(pressure_change, return_counts=True)
mem = {}
for val, count in zip(values, counts):
    mem[val] = count

threshold = 0.001
pos_values, pos_counts = np.unique(pressure_change[positive], return_counts=True)
pos_mask = pos_values > threshold
print(pos_counts[pos_mask].sum() / pos_counts.sum())

neg_values, neg_counts = np.unique(pressure_change[negative], return_counts=True)
neg_mask = neg_values > threshold
print(neg_counts[neg_mask].sum() / neg_counts.sum())

pos_mem = {}
for val, count in zip(pos_values, pos_counts):
    pos_mem[val] = count
    
pos_counts, neg_counts = [], []
for val in values:
    pos_count = pos_mem.get(val, 0)
    pos_counts.append(pos_count)
    neg_counts.append(mem[val] - pos_count)

def cosine_similarity(x, y):
    x = np.array(x) * 1.0
    y = np.array(y) * 1.0
    x /= x.sum()
    y /= y.sum()
    return (x @ y) / (np.linalg.norm(x) * np.linalg.norm(y))

print(cosine_similarity(pos_counts, neg_counts))

# Create a figure with two subplots (2 rows, 1 column)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))

# First plot
ax1.plot(values, pos_counts, label='pos')
ax1.set_title('leg movement mean pressure change count')
ax1.set_xlabel('mean pressure change')
ax1.set_ylabel('count')
# ax1.set_ylim(0, 3)
ax1.legend()
ax1.grid(True)

# Second plot
ax2.plot(values, pos_counts, label='pos')
ax2.plot(values, neg_counts, label='neg', alpha = 0.5)
ax2.set_title('non-leg movement mean pressure change count')
ax2.set_xlabel('mean pressure change')
ax2.set_ylabel('count')
ax2.legend()
ax2.grid(True)

# Adjust layout to prevent overlap
plt.tight_layout()

# Save the figure to a PNG file
plt.savefig('figures/average_over_all.png')

# values, counts = np.unique(pressure_change[positive], return_counts=True)
# print(len(values))
# plt.plot(values, counts, label='positive')
# plt.xlabel('mean pressure change')
# plt.ylabel('counts')
# plt.legend()
# plt.title('Leg movements mean pressure change counts in training set')
# plt.savefig('figures/positive1.png')

# values, counts = np.unique(pressure_change[negative], return_counts=True)
# print(len(values))
# plt.plot(values, counts, label='negative')
# plt.legend()
# plt.title('Non-leg movements mean pressure change counts in training set')
# plt.savefig('figures/negative1.png')