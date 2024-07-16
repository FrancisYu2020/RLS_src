import torch
import os

def get_checkpoint_path(args):
    path = f"{args.exp_id}/win{args.clip_len}_epoch{args.epochs}_lr{args.lr}_wd{args.weight_decay}_bs{args.batch_size}_{args.architecture}"
    return path

def get_val_checkpoint_dir(args):
    '''
    get the folder path to the saved checkpoint for validation/plot
    '''
    args.checkpoint_root = os.path.join('checkpoint', args.exp_name)
    checkpoint_dir = os.path.join(args.checkpoint_root, str(args.seed))
    return checkpoint_dir

def prepare_folder(args, skip_results=False):
    '''
    skip_results: if set to True, continue the python script by ignoring the saved results
    '''
    args.checkpoint_root = os.path.join('checkpoint', args.exp_name)
    if not os.path.exists(args.checkpoint_root) and (not args.debug_mode):
        os.makedirs(args.checkpoint_root, exist_ok=True)
    
    checkpoint_dir = os.path.join(args.checkpoint_root, str(args.seed))

    if not os.path.exists(checkpoint_dir):
        if not args.debug_mode:
            os.mkdir(checkpoint_dir)
    val_results_path = os.path.join(checkpoint_dir, 'val_results.pth')
    if os.path.exists(val_results_path) and not skip_results:
        print(f"Already exist {val_results_path}:", torch.load(val_results_path))
        exit()
    return checkpoint_dir