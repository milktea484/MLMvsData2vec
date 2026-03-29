from dataclasses import dataclass
from typing import Any

from omegaconf import MISSING


@dataclass
class CommonConfig:
    seed: int = MISSING
    batch_size: int = MISSING
    use_gpu: bool = MISSING
    iterations: int = None
    """デフォルトはtrain_configの設定に従う"""

@dataclass
class PretrainConfig:
    framework: str = None
    """使用する事前学習モデルのフレームワーク"""
    
    timestamp: str = None
    """使用する事前学習モデルのタイムスタンプ"""

@dataclass
class PathConfig:
    output_dir: str = MISSING
    timestamp: str = MISSING
    
@dataclass
class SStrainModelPathConfig:
    model_name: str = MISSING
    timestamp: str = MISSING
    
@dataclass
class DatasetConfig:
    embedding_file: str = None
    test_file: str = MISSING
    
@dataclass
class EvaluationConfig:
    auc_step: float = MISSING
    """ROC AUCとPR AUCを計算する際の閾値の刻み幅. 例えば0.01なら0, 0.01, 0.02, ..., 1の100点で評価することになる"""
    
@dataclass
class ExperimentConfig:
    name: str = MISSING
    additional_experiment_info: str = None
    kf_lambda: list[float] = MISSING
    """KnotFoldのmin cost flowアルゴリズムで使用するλのリスト. デフォルトはtrain_configの設定に従う"""
    
@dataclass
class MainConfig:
    common: CommonConfig = MISSING
    pretrain: PretrainConfig = MISSING
    path: PathConfig = MISSING
    SStrain_model_path: SStrainModelPathConfig = MISSING
    dataset: DatasetConfig = MISSING
    evaluation: EvaluationConfig = MISSING
    experiment: ExperimentConfig = MISSING