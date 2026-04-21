from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omegaconf import MISSING, ListConfig, OmegaConf


@dataclass
class ModelConfig:
    _target_: str = MISSING
    name: str = MISSING
    lr: float = MISSING
    min_lr: float = MISSING
    weight_decay: float = MISSING
    
@dataclass
class KnotFoldArchConfig:
    use_simple: bool = MISSING
    """線形層1層のみのシンプルなアーキテクチャを使用するかどうか"""
    conv_dim: int = MISSING
    kernel_size: int = MISSING
    n_residual_blocks: int = MISSING

@dataclass
class KnotFoldConfig(ModelConfig):
    kf_lambda: float = MISSING
    max_ref_epochs: int = MISSING
    arch: KnotFoldArchConfig = MISSING

@dataclass
class OptimizeConfig:
    _target_: str = MISSING
    name: str = MISSING
    lr: float = MISSING
    
@dataclass
class AdamWConfig(OptimizeConfig):
    weight_decay: float = MISSING

@dataclass
class CommonConfig:
    seed: int = MISSING
    batch_size: int = MISSING
    max_epochs: int = MISSING
    eval_per_epoch: int = MISSING
    eval_steps: int = MISSING
    use_gpu: bool = MISSING
    validation: bool = MISSING
    iterations: int = MISSING

@dataclass
class PretrainConfig:
    framework: list[str] | str | None = None
    """使用する事前学習モデルのフレームワーク. Noneでない場合, 単体, 複数どちらの場合もlistとして渡される."""
    
    timestamp: list[str] | str | None = None
    """使用する事前学習モデルのタイムスタンプ. Noneでない場合, 単体, 複数どちらの場合もlistとして渡される."""
    
    checkpoint: list[str] | str = MISSING
    """使用するモデルのチェックポイント. 単体, 複数どちらの場合もlistとして渡される.デフォルトは事前学習モデルの最終ステップ(final)"""

    def __post_init__(self):
        # ListConfigをlistに変換
        if isinstance(self.framework, ListConfig):
            self.framework = OmegaConf.to_container(self.framework, resolve=True)
        if isinstance(self.timestamp, ListConfig):
            self.timestamp = OmegaConf.to_container(self.timestamp, resolve=True)
        if isinstance(self.checkpoint, ListConfig):
            self.checkpoint = OmegaConf.to_container(self.checkpoint, resolve=True)
        
        # strの場合もlistに変換
        if isinstance(self.framework, str):
            self.framework = [self.framework]
        if isinstance(self.timestamp, str):
            self.timestamp = [self.timestamp]
        if isinstance(self.checkpoint, str):
            self.checkpoint = [self.checkpoint]
    
@dataclass
class PathConfig:
    data_dir: str = MISSING
    embedding_dir: str = MISSING
    pretrain_model_dir: str = MISSING
    output_dir: str = MISSING
    timestamp: str = MISSING    

@dataclass
class DatasetConfig:
    max_length: int = MISSING
    sequence_file: str = MISSING
    embedding_file: list[str] | str | None = None
    """すでにh5形式で保存されている配列特徴量のファイル名. 事前学習モデルの出力を使用する場合に必要. Noneでない場合, 単体, 複数どちらの場合もlistとして渡される."""
    train_file: str = MISSING
    validation_file: str = MISSING
    test_file: str = MISSING

    def __post_init__(self):
        # ListConfigをlistに変換
        if isinstance(self.embedding_file, ListConfig):
            self.embedding_file = OmegaConf.to_container(self.embedding_file, resolve=True)
        
        # strの場合もlistに変換
        if isinstance(self.embedding_file, str):
            self.embedding_file = [self.embedding_file]
    
@dataclass
class ExperimentConfig:
    name: str = MISSING
    """二次構造予測に使用する実験の名前 (必須)"""
    
    additional_experiment_info: str | None = None
    """実験の追加情報 (famfoldのfamilyやkfoldのk). 訓練, テストファイルのパス指定に使用する"""
    
    use_teacher: bool = MISSING
    """教師モデルの出力を使用して二次構造予測を行うかどうか (data2vecのみ)"""
    
    use_attention: bool = MISSING
    """使用する特徴表現をattentionにするかどうか. falseなら配列特徴量になる"""

@dataclass
class MainConfig:
    model: Any = MISSING
    optimizer: Any = MISSING
    common: CommonConfig = MISSING
    pretrain: PretrainConfig = MISSING
    path: PathConfig = MISSING
    dataset: DatasetConfig = MISSING
    experiment: ExperimentConfig = MISSING
