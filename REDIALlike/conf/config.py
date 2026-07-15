from dataclasses import dataclass
from typing import Any

from omegaconf import MISSING, ListConfig, OmegaConf


@dataclass
class CommonConfig:
    seed: int = MISSING
    use_gpu: bool = MISSING

@dataclass
class PretrainConfig:
    framework: str | None = None
    """使用する事前学習モデルのフレームワーク. Noneでない場合, 単体, 複数どちらの場合もlistとして渡される."""
    
    timestamp: str | None = None
    """使用する事前学習モデルのタイムスタンプ. Noneでない場合, 単体, 複数どちらの場合もlistとして渡される."""
    
    checkpoint: str = MISSING
    """使用するモデルのチェックポイント. 単体, 複数どちらの場合もlistとして渡される.デフォルトは事前学習モデルの最終ステップ(final)"""
    
@dataclass
class PathConfig:
    data_dir: str = MISSING
    pretrain_model_dir: str = MISSING
    output_dir: str = MISSING
    timestamp: str = MISSING    

@dataclass
class DatasetConfig:
    max_length: int = MISSING
    rna_tokens: list[str] = MISSING
    sequence_file: str = MISSING
    train_file: str = MISSING
    validation_file: str = MISSING
    test_file: str = MISSING
    
@dataclass
class ExperimentConfig:
    use_teacher: bool = MISSING
    """教師モデルの出力を使用するかどうか (data2vecのみ)"""

    return_attn: bool = MISSING
    """REDIALlikeのgenerate_contact_mapでAttention mapを返すかどうか"""
    
    extract_repr_layers: list[int] | None = MISSING
    """事前学習モデルのどのレイヤの特徴表現を抽出するか。リストで指定可能。Noneの場合は事前学習モデルの設定に従う。"""
    

@dataclass
class MainConfig:
    common: CommonConfig = MISSING
    pretrain: PretrainConfig = MISSING
    path: PathConfig = MISSING
    dataset: DatasetConfig = MISSING
    experiment: ExperimentConfig = MISSING
