# hydraでモデルとoptimizer選択, 学習率スケジューラはcosine schedulerで固定
# 事前学習モデルを読み込んで, batchごとに表現抽出して二次構造予測に回す
# その関係でモデルの_trainメソッドはbatchを入力とする

import copy
import logging
import random
import warnings
from pathlib import Path

import hydra
import numpy as np
import torch
import wandb
from conf.config import MainConfig
from dataset import create_dataloader
from models import KnotFoldModel
from modules import CosineScheduler
from omegaconf import OmegaConf
from tqdm import tqdm
from utils import get_embedding_dim, setup_config, validate_config

import pretrain.models as PretrainModels
from pretrain.conf.config import MainConfig as PretrainMainConfig

setup_config()

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: MainConfig):
    
    # 設定の妥当性確認
    validate_config(cfg)
    
    # 出力ディレクトリの設定
    output_dir = Path(cfg.path.output_dir) 
    
    ## 事前学習モデルがある時
    if cfg.pretrain.framework is not None and cfg.pretrain.timestamp is not None:
        output_dir = output_dir / cfg.pretrain.framework / cfg.pretrain.timestamp
        
    ## embedding fileがある時
    elif cfg.dataset.embedding_file is not None:
        embedding_name = cfg.dataset.embedding_file.split(".")[0]
        output_dir = output_dir / embedding_name
    
    output_dir = output_dir / cfg.model.name / cfg.path.timestamp
    
    ## additional_experiment_infoがある時
    if cfg.experiment.additional_experiment_info is not None:
        output_dir = output_dir / cfg.experiment.additional_experiment_info
        
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
        project=f"SSprediction",
        config=OmegaConf.to_container(cfg, resolve=True),
        name=f"{cfg.experiment.name}_{cfg.path.timestamp}" if cfg.experiment.additional_experiment_info is None else f"{cfg.experiment.name}_{cfg.experiment.additional_experiment_info}_{cfg.path.timestamp}",
        tags=[cfg.experiment.name, cfg.experiment.additional_experiment_info, cfg.model.name],
        dir=output_dir.resolve(),
    )
    
    # 表現学習モデルの準備
    pretrain_model_path = None
    if cfg.pretrain.framework is not None and cfg.pretrain.timestamp is not None:
        pretrain_model_path = Path(cfg.path.pretrain_model_dir) / cfg.pretrain.framework / cfg.pretrain.timestamp
        if not pretrain_model_path.exists():
            raise FileNotFoundError(f"Pretrain model path {pretrain_model_path} does not exist.")
    
    pretrain_model = None
    pretrain_cfg = None
    if pretrain_model_path is not None:
        pretrain_cfg_path = pretrain_model_path / f"train_config/.hydra/config.yaml"
        pretrain_cfg: PretrainMainConfig = OmegaConf.load(pretrain_cfg_path)
        
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
        
        pretrain_model._load_state_dict(torch.load(pretrain_model_path / pretrain_weight, map_location=device))
        
        logger.info(f"Loaded pretrain model from {pretrain_model_path}, checkpoint: {checkpoint}")
    else:
        # 事前学習モデルが指定されていない場合, embedding_fileで指定されている埋め込み表現を使用
        logger.info("No pretrain model specified. Will use embeddings from embedding_file provided.")
    
    # データローダーの設定
    train_loader = create_dataloader(config=cfg, split="train", pretrain_config=pretrain_cfg)
    val_loader = {
        "train": create_dataloader(config=cfg, split="train", pretrain_config=pretrain_cfg),
    }
    if cfg.common.validation:
        val_loader["validation"] = create_dataloader(config=cfg, split="validation", pretrain_config=pretrain_cfg)
    
    if cfg.model.name == "knotfold":
        ref_loader = create_dataloader(config=cfg, split="reference", pretrain_config=pretrain_cfg)
        
    total_steps = cfg.common.max_epochs * len(train_loader) # 二次構造予測モデルの学習ステップ数
    eval_interval = max(1, len(train_loader) // cfg.common.eval_per_epoch) # 評価の頻度 (train_loaderの長さに基づいて決定)

    # embedding次元数の取得
    embedding_dim = None
    if cfg.dataset.embedding_file is None:
        assert pretrain_model is not None, "Either a pretrain model or an embedding file must be specified to determine the embedding dimension."
        if cfg.experiment.use_attention:
            embedding_dim = pretrain_cfg.framework.n_layers * pretrain_cfg.framework.n_heads
        else:
            embedding_dim = pretrain_cfg.model_size.embed_dim
    else:
        embedding_dim = get_embedding_dim(train_loader, cfg.experiment.use_attention)
        
    logger.info(f"embedding dimension: {embedding_dim}")
    
    logger.info(f"Run on {output_dir}, with device {device}")
    logger.info(f"Model: {cfg.model.name}")
    logger.info(f"Setting seed: {cfg.common.seed}")
    logger.info(f"Experiment: {cfg.experiment.name}")
    
    # iteration
    for iteration in range(cfg.common.iterations):
        logger.info(f"--------Iteration {iteration + 1}/{cfg.common.iterations}--------")

        # 二次構造予測モデルの動的インポートと初期化
        model: KnotFoldModel = hydra.utils.instantiate(
            cfg.model,
            pretrain_model=pretrain_model,
            embedding_dim=embedding_dim,
            use_attention=cfg.experiment.use_attention,
            device=device
        )
        
        optimizer: torch.optim.Optimizer = hydra.utils.instantiate(cfg.optimizer, model.parameters())
        
        lr_scheduler = CosineScheduler(
            optimizer=optimizer,
            total_steps=total_steps,
            warmup_steps=total_steps // 10,  # 学習ステップの10%をウォームアップに使用
            max_lr=cfg.model.lr,
            min_lr=cfg.model.min_lr,
        )
        
        # reference用のモデルの動的インポートと初期化 (今はknotfoldのみ)
        if cfg.model.name == "knotfold":
            ref_model: KnotFoldModel = hydra.utils.instantiate(
                cfg.model,
                pretrain_model=pretrain_model,
                embedding_dim=embedding_dim,
                use_attention=cfg.experiment.use_attention,
                device=device,
                reference=True
            )
            
            ref_optimizer: torch.optim.Optimizer = hydra.utils.instantiate(cfg.optimizer, ref_model.parameters())
            
            ref_total_steps = cfg.model.max_ref_epochs * len(ref_loader) # リファレンスモデルの学習ステップ数
            ref_lr_scheduler = CosineScheduler(
                optimizer=ref_optimizer,
                total_steps=ref_total_steps,
                warmup_steps=ref_total_steps // 10,  # 学習ステップの10%をウォームアップに使用
                max_lr=cfg.model.lr,
                min_lr=cfg.model.min_lr,
            )
        
        # 学習ループの開始
        model = torch.compile(model)
        if cfg.model.name == "knotfold":
            ref_model = torch.compile(ref_model)
        
        # 初めにreference modelの学習を行う (knotfoldのみ)
        if cfg.model.name == "knotfold":
            # referenceモデルの保存用
            min_loss = float("inf")
            best_model_state_dict = None
            save_epoch = 0
            
            ref_model.train()
            logger.info("--------Start training reference model--------")
            with tqdm(ref_total_steps, desc="Reference Model Training") as ref_pbar:
                for ref_epoch in range(cfg.model.max_ref_epochs):
                    for ref_batch_idx, ref_batch in enumerate(ref_loader):
                        with ctx:
                            ref_loss = ref_model._train(ref_batch)
                            
                        ref_loss.backward()
                        
                        step = ref_epoch * len(ref_loader) + ref_batch_idx
                        
                        # 学習を進める前にvalidationとwandbへのログ出力
                        if step % eval_interval == 0 or step == ref_total_steps - 1:
                            wandb.log({f"ref_loss_{iteration}": ref_loss.item()}, step=step)
                            
                            if cfg.common.validation:
                                ref_model.eval()
                                with torch.no_grad():
                                    val_loader = val_loader["validation"]
                                    val_loss = 0.0
                                    for val_step in range(cfg.common.eval_steps):
                                        val_batch = next(iter(val_loader))
                                        val_loss += ref_model._test(val_batch)["loss"].item()
                                    val_loss /= len(val_loader)
                                    wandb.log({f"ref_val_loss_{iteration}": val_loss}, step=step)
                                    
                                    # モデルの保存
                                    if val_loss < min_loss:
                                        min_loss = val_loss
                                        best_model_state_dict = copy.deepcopy(ref_model.state_dict())
                                        save_epoch = step / eval_interval
                            else:
                                if ref_loss < min_loss:
                                    min_loss = ref_loss
                                    best_model_state_dict = copy.deepcopy(ref_model.state_dict())
                                    save_epoch = step / eval_interval
                                    
                                ref_model.train()
                        
                        ref_optimizer.step()
                        ref_lr_scheduler.step()
                        ref_optimizer.zero_grad()
                        
                        ref_pbar.update(1)
                        ref_pbar.set_postfix({"loss": ref_loss.item()})
                        
            # 最終的に最も性能の良かったモデルを保存
            assert best_model_state_dict is not None, "No model was saved during training. Please check if the training loop is working correctly."
            logger.info(f"Best reference model found at epoch {save_epoch:.2f} with validation loss {min_loss:.4f}. Saving model...")
            torch.save(best_model_state_dict, output_dir / f"weights/reference_{iteration}.pth")
        
        
        # 二次構造予測モデルの学習
        # モデルの保存用
        min_loss = float("inf")
        best_model_state_dict = None
        save_epoch = 0
        
        model.train()
        logger.info("--------Start training main model--------")
        with tqdm(total=total_steps, desc="Main Model Training") as pbar:
            for epoch in range(cfg.common.max_epochs):
                for train_batch_idx, batch in enumerate(train_loader):
                    with ctx:
                        loss = model._train(batch)
                        
                    loss.backward()
                    
                    step = epoch * len(train_loader) + train_batch_idx
                    
                    # 学習を進める前にvalidationとwandbへのログ出力
                    if step % eval_interval == 0 or step == total_steps - 1:
                        for split, val_loader in val_loader.items():
                            model.eval()
                            with torch.no_grad():
                                val_loss = 0.0
                                for val_step in range(cfg.common.eval_steps):
                                    val_batch = next(iter(val_loader))
                                    val_loss += model._test(val_batch)["loss"].item()
                                val_loss /= len(val_loader)
                                wandb.log({f"{split}_loss_{iteration}": val_loss}, step=step)
                                
                                # モデルの保存
                                if split == "validation":
                                    if val_loss < min_loss:
                                        min_loss = val_loss
                                        best_model_state_dict = copy.deepcopy(model.state_dict())
                                        save_epoch = step / eval_interval
                                elif split == "train":
                                    if val_loss < min_loss:
                                        min_loss = val_loss
                                        best_model_state_dict = copy.deepcopy(model.state_dict())
                                        save_epoch = step / eval_interval

                            model.train()
                    
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    
                    pbar.update(1)
                    pbar.set_postfix({"loss": loss.item()})
                    
        # 最終的に最も性能の良かったモデルを保存
        assert best_model_state_dict is not None, "No model was saved during training. Please check if the training loop is working correctly."
        logger.info(f"Best model found at epoch {save_epoch:.2f} with validation loss {min_loss:.4f}. Saving model...")
        torch.save(best_model_state_dict, output_dir / f"weights/prior_{iteration}.pth")