import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

# Custom scheduler that applies warm-up first, then cosine annealing
class WarmupCosineScheduler:
    def __init__(self, args, optimizer):
        self.warmup_epochs = args.warmup_epochs
        self.warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=self.warmup_lr)
        self.cosine_scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs - args.warmup_epochs)
        self.epoch = 0
    
    def step(self):
        if self.epoch < self.warmup_epochs:
            self.warmup_scheduler.step()
        else:
            self.cosine_scheduler.step()
        self.epoch += 1
    
    def get_last_lr(self):
        if self.epoch < self.warmup_epochs:
            return self.warmup_scheduler.get_last_lr()
        else:
            return self.cosine_scheduler.get_last_lr()
    
    def warmup_lr(self, epoch):
        if epoch < self.warmup_epochs:
            return epoch / self.warmup_epochs
        return 1