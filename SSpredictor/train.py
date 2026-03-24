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
from dataset import create_batch_iterator, create_dataloader
from modules import CosineScheduler
from omegaconf import OmegaConf
from utils import setup_config

import pretrain.models as PretrainModels

setup_config()

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: MainConfig):
    
    # 出力ディレクトリの設定
    output_dir = Path(cfg.path.output_dir)
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
    
    # 表現学習モデルの準備
    pretrain_cfg_path = Path(cfg.pretrain.model_path) / f"train_config/.hydra/config.yaml"
    pretrain_cfg = OmegaConf.load(pretrain_cfg_path)
    
    pretrain_model: PretrainModels.BaseModel = hydra.utils.instantiate(
        pretrain_cfg.framework,
        padding_idx=pretrain_cfg.dataset.tokens.index("<pad>"),
        num_tokens=len(pretrain_cfg.dataset.tokens),
        experiment_cfg=pretrain_cfg.experiment,
        device=device
    )
    
    ## 事前学習モデルの重みの読み込み
    checkpoint = pretrain_cfg.common.max_steps if cfg.pretrain.checkpoint == "final" else cfg.pretrain.checkpoint
    pretrain_weight = f"weight_{checkpoint}.pth" if not cfg.experiment.use_teacher else f"teacher_weight_{checkpoint}.pth"
    
    pretrain_model._load_state_dict(torch.load(Path(cfg.pretrain.model_path) / pretrain_weight, map_location=device))
    
    logger.info(f"Loaded pretrain model from {cfg.pretrain.model_path}")
    
    # データローダーの設定
    train_loader = create_dataloader(config=cfg, pretrain_config=pretrain_cfg, split="train")
    val_loader = {
        "train": create_dataloader(config=cfg, pretrain_config=pretrain_cfg, split="train"),
    }
    if cfg.common.validation:
        val_loader["validation"] = create_dataloader(config=cfg, pretrain_config=pretrain_cfg, split="validation")
    
    if cfg.model.name == "knotfold":
        ref_loader = create_dataloader(config=cfg, pretrain_config=pretrain_cfg, split="reference")
        
    total_steps = cfg.common.max_epochs * len(train_loader) # 二次構造予測モデルの学習ステップ数

    # 二次構造予測モデルの動的インポートと初期化
    model = hydra.utils.instantiate(
        cfg.model,
        pretrain_model=pretrain_model,
        use_attention=cfg.experiment.use_attention,
        device=device)
    
    optimizer: torch.optim.Optimizer = hydra.utils.instantiate(cfg.optimizer, model.parameters())
    
    lr_scheduler = CosineScheduler(
        optimizer=optimizer,
        total_steps=total_steps,
        warmup_steps=total_steps // 10,  # 学習ステップの10%をウォームアップに使用
        max_lr=cfg.model.lr,
        min_lr=cfg.model.min_lr,
    )