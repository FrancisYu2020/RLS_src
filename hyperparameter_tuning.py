from main import main_func
import optuna
import json
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--lr", default=0.01, help="learning rate used to train the model", type=float)
parser.add_argument("--weight_decay", default=0.1, help="weight decay used to train the model", type=float)
parser.add_argument("--epochs", default=300, help="epochs used to train the model", type=int)
parser.add_argument("--warmup_epochs", default=20, help="linear warmup epochs used to train the model", type=int)
parser.add_argument("--batch_size", default=256, help="batch size used to train the model", type=int)
parser.add_argument("--num_classes", default=2, help="number of classes for the classifier", type=int)
parser.add_argument("--clip_len", default=16, help="window size of the input data", type=int)
parser.add_argument("--input_size", default=[16, 16], nargs="+", help="window size of the input data", type=int)
parser.add_argument("--label_type", default="hard", help="indicate whether use hard one-hot labels or soft numerical labels", choices=["hard", "soft"])
parser.add_argument("--exp_label", default=None, help="extra labels to distinguish between different experiments")
parser.add_argument("--cross_val_type", default=0, type=int, help="0 for train all val all, 1 for leave patient 1 out")
parser.add_argument("--task", default="classification", type=str, choices=["classification", "regression"], help="indicate what kind of task to be run (regression/classification, etc.)")
parser.add_argument("--architecture", default="3d-resnet18", choices=["2d-efficientnet", "2d-positional", "2d-linear", "3d-conv", "2d-conv", "3d-resnet10", "3d-resnet18", "2d-resnet18", "ViT-tiny"], help="architecture used")
parser.add_argument("--exp_id", default=0, type=str, help="the id of the experiment")
parser.add_argument("--debug_mode", default=0, type=int, help="0 for experiment mode, 1 for debug mode")
parser.add_argument("--normalize_data", default=0, type=int, help="0 for raw mat input, 1 for normalized to [0, 1] and standardization")
parser.add_argument("--patients", default=[15], type=int, nargs="+", help="patient ids included in the training")
parser.add_argument("--checkpoint_root", default='checkpoint', type=str, help="checkpoint root path")
parser.add_argument("--seed", default=1, type=int, help="random seed for torch")
parser.add_argument("--split", default=-1, type=float, help="split ratio of train and val")
parser.add_argument("--data_type", default='tal', choices=['tal_pos', 'tal', 'context'], type=str, help="choose which kind of data to use")
args = parser.parse_args()

# Define the objective function for Optuna
def objective(trial):
    # Hyperparameter search space
    lr = trial.suggest_categorical('lr', [5e-4, 1e-3, 5e-3, 1e-2])
    batch_size = trial.suggest_categorical('batch_size', [128, 256, 512])
    weight_decay = trial.suggest_categorical('weight_decay', [1e-5, 1e-4, 1e-3, 1e-2, 1e-1])
    
    args.lr = lr
    args.batch_size = batch_size
    args.weight_decay = weight_decay
    
    f1 = main_func(args)
    
    return f1

# Create the study and optimize
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=2)

# Print the best hyperparameters
print("Best hyperparameters: ", study.best_params)
results_json = os.path.join('checkpoint', args.exp_id, 'results.json')
with open(results_json, 'w') as f:
    json.dump(study.best_params, f)