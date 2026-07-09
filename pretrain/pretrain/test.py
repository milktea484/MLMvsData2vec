import datetime
import logging
import json
import random
import sys
import warnings
from pathlib import Path

import hydra
import h5py
import numpy as np
import torch
import wandb
from omegaconf import OmegaConf
from tqdm import tqdm

from .conf.config import MainConfig
from .dataset import create_dataloader
from .models import BaseModel
from .utils import setup_config, validate_config

setup_config()


def main():
    # コマンドライン引数の取得
    sys_args = sys.argv[1:]
    args = {}
    for arg in sys_args:
        key, value = arg.split("=", 1)
        
        if value.lower() == "true":
            value = True
        elif value.lower() == "false":
            value = False
        args[key] = value
    
    # warningsの設定 (UserWarningを無視)
    warnings.filterwarnings("ignore", category=UserWarning)
    
    # logの設定
    logger = logging.getLogger(__name__)
    
    # 使用デバイスの設定
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
        logger.info("Using GPU for training.")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU for training.")
    ctx = torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    
    # 学習済みモデルが存在するディレクトリの設定, 存在確認
    pretrain_dir = Path("./results/pretrain_results") / args["framework"] / args["timestamp"]
    if not pretrain_dir.exists():
        raise FileNotFoundError(f"Pretrained model directory '{pretrain_dir}' does not exist.")

    # 設定の読み込み
    config_path = pretrain_dir / "train_config" / ".hydra" / "config.yaml"
    cfg: MainConfig = OmegaConf.load(config_path)
    
    # 設定の上書き
    cfg.dataset.test_file = args["test_file"]
    extract_repr_layers = json.loads(args["extract_repr_layers"])
    cfg.experiment.extract_repr_layers = extract_repr_layers

    # モデルの読み込み
    # 互換パッチ: 古い pretrain 実験では `_target_` が "models.*" になっていることがあるため
    # 現在のパッケージ構成に合わせてフルパスに書き換える
    target = getattr(cfg.framework, "_target_", None)
    if target == "models.data2vecModel":
        cfg.framework._target_ = "pretrain.models.data2vecModel"
    elif target == "models.MLMModel":
        cfg.framework._target_ = "pretrain.models.MLMModel"
    
    pretrain_model: BaseModel = hydra.utils.instantiate(
        cfg.framework,
        padding_idx=cfg.dataset.tokens.index("<pad>"),
        num_tokens=len(cfg.dataset.tokens),
        experiment_cfg=cfg.experiment,
        device=device
    )
    
    # 事前学習モデルの重みの読み込み
    if args["checkpoint"] == "final":
        checkpoint = cfg.common.max_steps
    else:
        checkpoint = args["checkpoint"]

    pretrain_weight = f"weight_{checkpoint}.pth" if not args["is_teacher"] else f"teacher_weight_{checkpoint}.pth"
    pretrain_model._load_state_dict(torch.load(pretrain_dir / pretrain_weight, map_location=device))
    
    # データローダーの設定
    test_loader = create_dataloader(
        config=cfg,
        split="test",
    )
    
    # 埋め込みの抽出と保存
    embedding_output_dir = Path(f"{pretrain_dir}/embeddings")
    embedding_output_dir.mkdir(parents=True, exist_ok=True)
    
    ## 抽出する埋め込みの種類
    embedding_type = "attn" if args["is_attention"] else "repr"
    if args["is_teacher"]:
        embedding_type = "teacher_" + embedding_type
    
    h5_files = {layer: h5py.File(embedding_output_dir / f"{Path(args['test_file']).stem}_{embedding_type}_layer{layer}.h5", "w") for layer in extract_repr_layers}
        
    pretrain_model.eval()
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Extracting embeddings"):
            seq_ids = batch["seq_ids"]
            lengths = batch["lengths"]
            
            # 埋め込みの抽出
            representations = pretrain_model._test(batch, extract_repr_layers=extract_repr_layers)[embedding_type]  # (len(extract_repr_layers), B, L, E)
            
            # 各層の埋め込みを保存
            for index, layer in enumerate(extract_repr_layers):
                layer_embeddings = representations[index]  # (B, L, E)
                for i in range(layer_embeddings.size(0)):
                    seq_length = lengths[i]
                    h5_files[layer].create_dataset(f"{seq_ids[i]}", data=layer_embeddings[i, :seq_length].cpu().numpy())
    
    # h5_fileのクローズ
    for h5_file in h5_files.values():
        h5_file.close()

if __name__ == "__main__":
    main()
