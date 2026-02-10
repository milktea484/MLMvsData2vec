from dataclasses import dataclass
from typing import Any

from omegaconf import MISSING


# frameworkconfig内部で使用するarchitecture設定クラス
@dataclass
class ArchitectureConfig:
    embed_dim: int = MISSING
    ffn_dim: int = MISSING
    n_layers: int = MISSING
    n_heads: int = MISSING
    attention_dropout: float = MISSING
    rope_max_length: int = MISSING
    k_layers: int | None = None
    n_head_layers: int | None = None

# framework設定の基底クラス
@dataclass
class FrameworkConfig:
    _target_: str = MISSING
    name: str = MISSING
    wandb_tags: list[str] = MISSING
    lr: float = MISSING
    weight_decay: float = MISSING
    sptoken_prob: float = MISSING
    """入力配列のうちspecial tokenとする確率"""
    mask_prob: float = MISSING
    """special tokenのうち"<mask>"トークンに置換する確率"""
    ernie_rna_alpha: float = MISSING
    
    arch: ArchitectureConfig = MISSING
    
    def __post_init__(self):
        assert self.name in ["mlm", "data2vec"], \
            "framework.name must be either 'mlm' or 'data2vec'."

# MLM用のframework設定クラス
@dataclass
class MLMConfig(FrameworkConfig):
    pass

# 量子化する場合の設定クラス
@dataclass
class QuantizeConfig:
    use_quantize: bool = MISSING
    """data2vecで量子化を使用するかどうか"""
    codebook_dim: int = MISSING
    n_codebooks: int = MISSING

# data2vec用のframework設定クラス
@dataclass
class data2vecConfig(FrameworkConfig):
    ema_decay: float = MISSING
    ema_end_decay: float = MISSING
    ema_anneal_end_steps: int = MISSING
    loss_beta: float = MISSING
    
    quantize_cfg: QuantizeConfig = MISSING
        
    def __post_init__(self):
        super().__post_init__()
        assert self.arch.k_layers is not None and self.arch.n_head_layers is not None, \
            "k_layers and n_head_layers must be specified for data2vec framework."

# model_size設定の基底クラス
@dataclass
class ModelSizeConfig:
    name: str = MISSING
    gradient_accumulation_steps: int = MISSING
    gradient_accumulation_steps_for_test: int = MISSING
    embed_dim: int = MISSING
    ffn_dim: int = MISSING
    n_layers: int = MISSING
    n_heads: int = MISSING
    attention_dropout: float = MISSING
    k_layers: int = MISSING
    n_head_layers: int = MISSING
    
    def __post_init__(self):
        assert self.embed_dim % self.n_heads == 0, \
            "embed_dim must be divisible by n_heads"
        
# optimizer設定の基底クラス
@dataclass
class OptimizerConfig:
    _target_: str = MISSING
    lr: float = MISSING
    
# AdamW設定クラス 
@dataclass
class AdamWConfig(OptimizerConfig):
    weight_decay: float = MISSING

# LR scheduler設定の基底クラス
@dataclass
class LRSchedulerConfig:
    _target_: str = MISSING
    
# Cosine scheduler設定クラス
@dataclass
class CosineSchedulerConfig(LRSchedulerConfig):
    warmup_steps: int = MISSING
    total_steps: int = MISSING
    max_lr: float = MISSING
    min_lr: float = MISSING

# 共通設定クラス
@dataclass
class CommonConfig:
    seed: int = MISSING
    batch_size: int = MISSING
    max_steps: int = MISSING
    eval_interval: int = MISSING
    eval_steps: int = MISSING
    use_gpu: bool = MISSING
    validation: bool = MISSING
    num_workers: int = MISSING
    
# checkpoint設定クラス
@dataclass
class CheckpointConfig:
    model_save_interval: int = MISSING

# path設定クラス
@dataclass
class PathConfig:
    data_dir: str = MISSING
    test_data_dir: str = MISSING
    output_dir: str = MISSING

# dataset設定クラス
@dataclass
class DatasetConfig:
    tokens: list[str] = MISSING
    rna_tokens: list[str] = MISSING
    other_tokens: list[str] = MISSING
    
    max_length: int = MISSING
    """CLS, EOSトークンを含んだ最大シーケンス長"""
    
    train_file: str = MISSING
    validation_file: str = MISSING
    test_file: str = MISSING
    
    def __post_init__(self):
        assert "<pad>" in self.tokens, \
            "'<pad>' token must be in tokens."
        assert "<mask>" in self.tokens, \
            "'<mask>' token must be in tokens."
        assert self.rna_tokens == self.tokens[:len(self.rna_tokens)], \
            "rna_tokens must be the prefix of tokens."
    
# その他実験に関する設定クラス
@dataclass
class ExperimentConfig:
    extract_repr_layers: int = MISSING
    """
    特徴表現を抽出するTransformer層の番号
    0からn_layersの範囲で, 0は埋め込み表現, それ以降は各Transformer層の出力を指す
    """
    use_ernie_rna: bool = MISSING
    """ERNIE-RNAの手法を使用するかどうか"""
    use_additional_token: bool = MISSING
    """
    CLS, EOSトークンを使用するかどうか. use_ernie_rnaがTrueの場合, 自動的にTrueになる
    """
    
    def __post_init__(self):
        if self.use_ernie_rna:
            self.use_additional_token = True

# メインの設定クラス
@dataclass
class MainConfig:
    # defaultsリストで指定される各グループを定義
    framework: Any = MISSING
    model_size: ModelSizeConfig = MISSING
    optimizer: Any = MISSING
    lr_scheduler: Any = MISSING
    common: CommonConfig = MISSING
    checkpoint: CheckpointConfig = MISSING
    path: PathConfig = MISSING
    dataset: DatasetConfig = MISSING
    experiment: ExperimentConfig = MISSING
    