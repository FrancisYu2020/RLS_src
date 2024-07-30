# script to study the regional prediction quality for a window for original method

from utils.models import *
from utils.dataset import *
from utils.inference import inference, plot_inference
from utils import prepare_folder, get_checkpoint_path
from main import args
import os
import torch
from sklearn.metrics import fbeta_score

args.exp_name = get_checkpoint_path(args)
args.checkpoint_folder = prepare_folder(args, skip_results=True)

print(args.checkpoint_folder)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args.device = device

model = get_model(args)
model.load_state_dict(torch.load(os.path.join(args.checkpoint_folder, 'best_f1.ckpt'))['state_dict'])
model.to(device)

args.clip_len_prefix = 'win' + str(args.clip_len) + '_'
_, _, val_data, val_label = preprocess_dataset(args)
print(val_data.shape, val_label.shape, val_label.sum())
print(val_data.min(), val_data.max(), val_data.mean())
    
# original CNN transformation
train_transform = get_cnn_transforms(args)
val_transform = get_cnn_transforms(args, train=False)
val_dataset = RLSDataset(args, val_data, val_label, transform=val_transform)
val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

y_pred, y_true = inference(args, model, val_loader)
del model, val_loader
plot_inference(args, y_pred, y_true, 'f1')

# Naive method error analysis
y_pred = y_pred.reshape(-1, args.clip_len)
y_true = y_true.reshape(-1, args.clip_len)
L = args.clip_len // 3

left_pred = y_pred[:, :L]
left_true = y_true[:, :L]
middle_pred = y_pred[:, L:-L]
middle_true = y_true[:, L:-L]
right_pred = y_pred[:, -L:]
right_true = y_true[:, -L:]

f1_left = np.round(100 * fbeta_score(left_true.reshape(-1), left_pred.reshape(-1), beta=1), 2)
f1_middle = np.round(100 * fbeta_score(middle_true.reshape(-1), middle_pred.reshape(-1), beta=1), 2)
f1_right = np.round(100 * fbeta_score(right_true.reshape(-1), right_pred.reshape(-1), beta=1), 2)
print(f1_left, f1_middle, f1_right)
