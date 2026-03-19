# hydraでモデルとoptimizer選択, 学習率スケジューラはcosine schedulerで固定
# 事前学習モデルを読み込んで, batchごとに表現抽出して二次構造予測に回す
# その関係でモデルの_trainメソッドはbatchを入力とする

import logging
import random
import warnings
from pathlib import Path

import hydra
import numpy as np
import torch
from conf.config import MainConfig
from omegaconf import OmegaConf
from utils import setup_config

setup_config()

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: MainConfig):
    
    # 出力ディレクトリの設定
    output_dir = Path(cfg.path.output_dir) / f"{cfg.framework.name}" / f"{cfg.path.timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # logの設定
    logging.basicConfig(
        level=logging.INFO,  # Set the minimum log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format="%(asctime)s - %(name)s.%(lineno)d - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),  # Log to console
            logging.FileHandler(output_dir / "log_train.txt", mode="w"),
        ],
    )
    logger = logging.getLogger(__name__)
    
    # warningsの設定 (UserWarningを無視)
    warnings.filterwarnings("ignore", category=UserWarning)

    # 使用デバイスの設定
    if torch.cuda.is_available() and cfg.common.use_gpu:
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
        logger.info("Using GPU for training.")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU for training.")
    ctx = torch.autocast(device_type=device.type, dtype=torch.bfloat16)

    # seed固定
    random.seed(cfg.common.seed)
    np.random.seed(cfg.common.seed)
    torch.manual_seed(cfg.common.seed)
    torch.backends.cudnn.benchmark = False  # 再現性を無視してでも畳み込み演算速度を上げるオプション
    torch.backends.cudnn.deterministic = True  # pytorchで非決定的な操作を決定的なものにするオプション
