import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR

# Custom scheduler that applies warm-up first, then cosine annealing
class WarmupCosineScheduler:
    def __init__(self, args, optimizer):
        self.epoch = 0
        self.warmup_epochs = args.warmup_epochs
        self.cosine_epochs = args.cosine_epochs
        self.warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=self.warmup_lr)
        self.cosine_scheduler = CosineAnnealingLR(optimizer, T_max=args.cosine_epochs - args.warmup_epochs, eta_min=0.2 * args.lr)
        self.lambda_scheduler = LambdaLR(optimizer, lr_lambda=self.lambda_lr)
    
    def step(self):
        if self.epoch < self.warmup_epochs:
            self.warmup_scheduler.step()
        elif self.epoch < self.cosine_epochs:
            self.cosine_scheduler.step()
        else:
            self.lambda_scheduler.step()
        self.epoch += 1
    
    def get_last_lr(self):
        if self.epoch < self.warmup_epochs:
            return self.warmup_scheduler.get_last_lr()
        elif self.epoch < self.cosine_epochs:
            return self.cosine_scheduler.get_last_lr()
        else:
            return self.lambda_scheduler.get_last_lr()
    
    def warmup_lr(self, epoch):
        if epoch < self.warmup_epochs:
            return epoch / self.warmup_epochs
        return 1
    
    def lambda_lr(self, epoch):
        return 0.02

    