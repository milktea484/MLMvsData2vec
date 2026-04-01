from dataclasses import dataclass
from typing import Any

from omegaconf import MISSING


@dataclass
class CommonConfig:
    seed: int = MISSING
    batch_size: int = MISSING
    use_gpu: bool = MISSING
    iterations: int | None = None
    """デフォルトはtrain_configの設定に従う"""
    evaluation: bool = MISSING
    """評価を行うかどうか. デフォルトはTrue"""
    prediction: bool = MISSING
    """予測を行うかどうか. デフォルトはTrue"""
    save_probability_matrix: bool = MISSING
    """予測された塩基対確率行列を保存するかどうか. デフォルトはTrue"""
    
    def __post_init__(self):
        if self.evaluation is False and self.prediction is False:
            raise ValueError("At least one of evaluation or prediction must be True.")

@dataclass
class PretrainConfig:
    framework: str | None = None
    """使用する事前学習モデルのフレームワーク"""
    
    timestamp: str | None = None
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
    embedding_file: str | None = None
    test_file: str = MISSING
    
@dataclass
class EvaluationConfig:
    auc_step: float = MISSING
    """ROC AUCとPR AUCを計算する際の閾値の刻み幅. 例えば0.01なら0, 0.01, 0.02, ..., 1の100点で評価することになる"""

@dataclass
class KfLambdaConfig:
    min: float | None = None
    """kf_lambdaの最小値"""
    max: float | None = None
    """kf_lambdaの最大値"""
    step: float | None = None
    """kf_lambdaの刻み幅"""

@dataclass
class ExperimentConfig:
    name: str = MISSING
    additional_experiment_info: str | None = None
    kf_lambda_cfg: KfLambdaConfig = MISSING
    """KnotFoldの最小フローアルゴリズムのlambdaの設定. minからmaxまでstep刻みで複数のkf_lambdaを試す. デフォルトはtrain_config(knotfold.yaml)の設定に従う"""
    
@dataclass
class MainConfig:
    common: CommonConfig = MISSING
    pretrain: PretrainConfig = MISSING
    path: PathConfig = MISSING
    SStrain_model_path: SStrainModelPathConfig = MISSING
    dataset: DatasetConfig = MISSING
    evaluation: EvaluationConfig = MISSING
    experiment: ExperimentConfig = MISSING