import datetime
import logging
import random
import warnings
from pathlib import Path

import hydra
import numpy as np
import torch
import wandb
from conf.config import MainConfig
from dataset import create_batch_iterator
from models import BaseModel
from omegaconf import OmegaConf
from tqdm import tqdm
from utils import setup_config, validate_config

setup_config()


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: MainConfig):
    
    # 設定の妥当性確認
    validate_config(cfg)
    
    # 出力ディレクトリの設定
    timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    output_dir = Path(cfg.path.output_dir) / f"{cfg.framework.name}" / f"{timestamp}"
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
        
    # wandbの初期化
    wandb.init(
        project=f"MLMvsData2vec",
        config=OmegaConf.to_container(cfg, resolve=True),
        name=f"{cfg.framework.name}_{timestamp}",
        tags=cfg.framework.wandb_tags,
        dir=output_dir.resolve(),
    )
    
    # モデルの動的インポートと初期化
    model: BaseModel = hydra.utils.instantiate(
        cfg.framework,
        padding_idx=cfg.dataset.tokens.index("<pad>"),
        num_tokens=len(cfg.dataset.tokens),
        experiment_cfg=cfg.experiment,
        device=device
    )
    optimizer: torch.optim.Optimizer = hydra.utils.instantiate(cfg.optimizer, params=model.parameters())
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler = hydra.utils.instantiate(cfg.lr_scheduler, optimizer=optimizer)
    
    # データローダーの設定
    train_batch_iterator = create_batch_iterator(config=cfg, split="train")
    val_batch_iterator = {
        "train": create_batch_iterator(config=cfg, split="train"),
    }
    if cfg.common.validation:
        val_batch_iterator["validation"] = create_batch_iterator(config=cfg, split="validation")
    
    # トレーニングの実行
    logger.info(f"Run on {output_dir}, with device {device}")
    logger.info(f"Framework: {cfg.framework.name}")
    logger.info(f"Setting seed: {cfg.common.seed}")
    logger.info(f"Input for training: {cfg.dataset.train_file}")
    if cfg.common.validation:
        logger.info(f"Input for validation: {cfg.dataset.validation_file}")
    
    logger.info("--------Start training--------")

    # gradient_accumulation_stepsを考慮した実質の学習ステップ数
    step = 0
    # progress bar用のloss蓄積変数
    accumulated_loss = 0.0
    
    model = torch.compile(model)
    
    # 初期モデルの保存
    model.save_model(save_path=output_dir, step=step)
    logger.info(f"Initial model saved to {output_dir / 'weight_0.pth'}")
    
    # トレーニングループ
    model.train()
    with tqdm(total=cfg.common.max_steps, desc="Training") as pbar:
        for iteration in range(cfg.common.max_steps * cfg.model_size.gradient_accumulation_steps):
            
            # 損失計算と勾配計算
            train_batch = next(train_batch_iterator)
            with ctx:
                loss = model._train(train_batch)
            loss = loss / cfg.model_size.gradient_accumulation_steps
            loss.backward()
            accumulated_loss += loss.item()
            
            if (iteration + 1) % cfg.model_size.gradient_accumulation_steps == 0:
                # モデルの学習を進める前にvalidationとwandbへのログ出力
                if step % cfg.common.eval_interval == 0 or step == cfg.common.max_steps - 1:
                    # validation
                    model.eval()
                    
                    # trainデータとvalデータの両方で評価
                    for split, val_iterator in val_batch_iterator.items():
                        
                        # 損失
                        val_loss = 0.0
                        
                        # 分散 (垂直・水平方向)
                        val_var = torch.zeros((2,), requires_grad=False, device=device)
                        val_var_teacher = torch.zeros((2,), requires_grad=False, device=device)
                            
                        for eval_iteration in range(cfg.common.eval_steps):
                            # 検証結果の取得
                            val_batch = next(val_iterator)
                            with ctx, torch.inference_mode():
                                val_results = model._validate(val_batch)
                                
                            # 損失
                            val_loss += val_results["loss"].item()
                            
                            # 分散
                            val_var += val_results["var"]
                            val_var_teacher += val_results.get("var_teacher", torch.zeros((2,), device=device, requires_grad=False))
                            
                            # data2vecで離散化させる場合はここに追記
                                
                        
                        # lossを平均してwandbにログ出力
                        val_loss = val_loss / cfg.common.eval_steps
                        wandb.log({f"{split}_loss": val_loss}, step=step + 1)
                        
                        # 分散を平均してwandbにログ出力
                        val_var = val_var / cfg.common.eval_steps
                        wandb.log({f"{split}_var_vertical": val_var[0].item()}, step=step + 1)
                        wandb.log({f"{split}_var_horizontal": val_var[1].item()}, step=step + 1)
                        
                        # MLMの場合はteacher分散は0として出力される
                        val_var_teacher = val_var_teacher / cfg.common.eval_steps
                        wandb.log({f"{split}_var_teacher_vertical": val_var_teacher[0].item()}, step=step + 1)
                        wandb.log({f"{split}_var_teacher_horizontal": val_var_teacher[1].item()}, step=step + 1)
                    
                    model.train()
                
                # パラメータの更新（とprogress barの更新）
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                model._step()
                
                pbar.update()
                pbar.set_postfix({"loss": f"{accumulated_loss:.4f}"})
                accumulated_loss = 0.0
                step += 1
                
                # モデルの保存
                if step % cfg.checkpoint.model_save_interval == 0 or step == cfg.common.max_steps:
                    model.save_model(save_path=output_dir, step=step)
                    logger.info(f"Model saved at step {step} to {output_dir / f'weight_{step}.pth'}")
                
        
    wandb.finish()
    
                

if __name__ == "__main__":
    main()
