from main import args
from utils.dataset import *
from utils.models import *
from utils import get_checkpoint_path, prepare_folder
from train import val_loop, compute_metrics
import os
import torch

# adjust argument for testing
args.downsample_val = 0
args.val_type = 'cross+internal-val'
args.exp_name = get_checkpoint_path(args)
args.checkpoint_dir = prepare_folder(args, skip_results=True)
args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

def test_model(model, mode):
    assert mode in ['cross', 'internal']
    results_path = os.path.join(args.checkpoint_dir, f'test_{mode}.pth')
    if os.path.exists(results_path):
        print(f'test_{mode}.pth already exist:')
        print(torch.load(results_path))
        print()
        return
    # val one epoch and get val logs
    criterion = torch.nn.BCEWithLogitsLoss()
    log_data = {}
    if cross_val_data is not None:
        val_loss, y_pred, y_true = val_loop(args, model, cross_val_loader, criterion)
        precision, recall, f1, fprec, miou = compute_metrics(y_pred, y_true)
        log_data.update({"val/cross_loss": val_loss, "val/cross_f1": f1, "val/cross_f0.5":fprec, "val/cross_precision":precision, "val/cross_recall":recall, "val/cross_miou": miou})
    if internal_val_data is not None:
        val_loss, y_pred, y_true = val_loop(args, model, internal_val_loader, criterion)
        precision, recall, f1, fprec, miou = compute_metrics(y_pred, y_true)
        log_data.update({"val/internal_loss": val_loss, "val/internal_f1": f1, "val/internal_f0.5":fprec, "val/internal_precision":precision, "val/internal_recall":recall, "val/internal_miou": miou})
    torch.save(log_data, results_path)
    print(f'{mode} test results:')
    print(log_data)
    print()

def baseline(mode, inter_label, intra_label):
    assert mode in ['all_one', 'random']
    log_data = {}
    if mode == 'random':
        inter_y_pred = torch.randint(0, 2, (len(inter_label),)).numpy()
        intra_y_pred = torch.randint(0, 2, (len(intra_label),)).numpy()
    elif mode == 'all_one':
        inter_y_pred = np.ones(len(inter_label)).astype(int)
        intra_y_pred = np.ones(len(intra_label)).astype(int)
        
    if inter_label is not None:
        precision, recall, f1, fprec, miou = compute_metrics(inter_y_pred, inter_label)
        log_data.update({"val/cross_f1": f1, "val/cross_f0.5":fprec, "val/cross_precision":precision, "val/cross_recall":recall, "val/cross_miou": miou})
    if intra_label is not None:
        precision, recall, f1, fprec, miou = compute_metrics(intra_y_pred, intra_label)
        log_data.update({"val/internal_f1": f1, "val/internal_f0.5":fprec, "val/internal_precision":precision, "val/internal_recall":recall, "val/internal_miou": miou})
    print(f'{mode} test results:')
    print(log_data)
    print()
        
        
_, _, cross_val_data, cross_val_label, internal_val_data, internal_val_label = preprocess_dataset(args)
print(cross_val_data.shape, internal_val_data.shape)

# prepare test dataset
val_transform = get_cnn_transforms(args, train=False)
if cross_val_data is not None:
    cross_val_dataset = RLSDataset(args, cross_val_data, cross_val_label, transform=val_transform)
    cross_val_loader = DataLoader(dataset=cross_val_dataset, num_workers=args.num_workers, batch_size=4 * args.batch_size, shuffle=False, pin_memory=True)
if internal_val_data is not None:
    internal_val_dataset = RLSDataset(args, internal_val_data, internal_val_label, transform=val_transform)
    internal_val_loader = DataLoader(dataset=internal_val_dataset, num_workers=args.num_workers, batch_size=4 * args.batch_size, shuffle=False, pin_memory=True)
labels = {'cross':cross_val_label, 'internal':internal_val_label}

model = get_model(args)
model.to(args.device)

# test baseline 
baseline('random', cross_val_label, internal_val_label)
baseline('all_one', cross_val_label, internal_val_label)

# test inter patient model
model.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, "cross_best_f1.ckpt"))['state_dict'])
model.eval()
test_model(model, 'cross')

# test intra patient model
model.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, "internal_best_f1.ckpt"))['state_dict'])
model.eval()
test_model(model, 'internal')

