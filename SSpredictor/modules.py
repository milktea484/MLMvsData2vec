import numpy as np
import torch
from torch.optim.lr_scheduler import _LRScheduler


class CosineScheduler(_LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int = 10000,
        total_steps: int = 200000,
        max_lr: float = 1e-5,
        min_lr: float = 1e-6,
        last_iter: int = -1,
    ):
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.max_lr = max_lr
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch=last_iter)

    def get_lr(self):
        last_iter = self.last_epoch
        if last_iter < self.warmup_steps:
            return [base_lr * last_iter / self.warmup_steps for base_lr in self.base_lrs]
        
        if last_iter > self.total_steps:
            return [base_lr * self.min_lr / self.max_lr for base_lr in self.base_lrs]
        
        decay_ratio = (last_iter - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        coeff = 0.5 * (1.0 + np.cos(np.pi * decay_ratio))
        
        return [self.min_lr + coeff * (base_lr - self.min_lr) for base_lr in self.base_lrs]