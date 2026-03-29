import numpy as np
import torch
import torch.nn as nn
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
    

class ResNet2DBlock(nn.Module):
    """
    ResNetの2D畳み込みブロック. RNA-LLMの論文参照.
    Luciano I Zablocki, Leandro A Bugnon, Matias Gerard, Leandro Di Persia, Georgina Stegmayer, Diego H Milone, 
    Comprehensive benchmarking of large language models for RNA secondary structure prediction, 
    Briefings in Bioinformatics, Volume 26, Issue 2, March 2025, bbaf137, https://doi.org/10.1093/bib/bbaf137
    """
    def __init__(self, embed_dim, kernel_size=3, bias=False):
        super().__init__()

        # Bottleneck architecture
        self.conv_net = nn.Sequential(
            nn.Conv2d(
                in_channels=embed_dim, out_channels=embed_dim, kernel_size=1, bias=bias
            ),
            nn.InstanceNorm2d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=embed_dim,
                out_channels=embed_dim,
                kernel_size=kernel_size,
                bias=bias,
                padding="same",
            ),
            nn.InstanceNorm2d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=embed_dim, out_channels=embed_dim, kernel_size=1, bias=bias
            ),
            nn.InstanceNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        residual = x

        x = self.conv_net(x)
        x = x + residual

        return x


class ResNet2D(nn.Module):
    """
    ResNetの2D畳み込みブロックを複数積み重ねたネットワーク. RNA-LLMの論文参照.
    Luciano I Zablocki, Leandro A Bugnon, Matias Gerard, Leandro Di Persia, Georgina Stegmayer, Diego H Milone, 
    Comprehensive benchmarking of large language models for RNA secondary structure prediction, 
    Briefings in Bioinformatics, Volume 26, Issue 2, March 2025, bbaf137, https://doi.org/10.1093/bib/bbaf137
    """
    def __init__(self, embed_dim, num_blocks, kernel_size=3, bias=False):
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                ResNet2DBlock(embed_dim, kernel_size, bias=bias)
                for _ in range(num_blocks)
            ]
        )

    def forward(self, x):
        for block in self.blocks:
            x = block(x)

        return x
    
class SimpleArch(nn.Module):
    """
    シンプルなアーキテクチャ. 線形層1枚.
    """
    def __init__(self, embedding_dim: int, use_attention: bool):
        super().__init__()
        self.use_attention = use_attention
        self.linear = nn.Linear(embedding_dim, 1) if use_attention else nn.Linear(2 * embedding_dim, 1)
    
    def forward(self, x: torch.Tensor):
        return self.linear(x)
    
class CNNArch(nn.Module):
    """
    CNNベースのアーキテクチャ. ResNet2Dを使用.
    """
    def __init__(
        self,
        arch: dict,
        embedding_dim: int,
        use_attention: bool,
    ):
        super().__init__()
        self.use_attention = use_attention
        self.linear_in = nn.Linear(embedding_dim, arch["conv_dim"]) if use_attention else nn.Linear(embedding_dim, int(arch["conv_dim"] / 2))
        self.resnet = ResNet2D(arch["conv_dim"], arch["n_residual_blocks"], kernel_size=arch["kernel_size"])
        self.conv_out = nn.Conv2d(arch["conv_dim"], 1, kernel_size=arch["kernel_size"], padding="same")
        
    def forward(self, x: torch.Tensor):
        x = self.linear_in(x)
        
        x = x.permute(0, 3, 1, 2) # (B, L, L, E) -> (B, E, L, L)
        
        x = self.resnet(x)
        x = self.conv_out(x)
        
        x = x.squeeze(1) # (B, 1, L, L) -> (B, L, L)
        
        return x