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
from dataset import create_batch_iterator, create_dataloader
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

    # DictConfigをpythonオブジェクトに変換 (listの読み込みのため)
    structured_cfg = OmegaConf.merge(OmegaConf.structured(MainConfig), cfg)
    main_cfg : MainConfig = OmegaConf.to_object(structured_cfg)
    
    # 設定の妥当性確認
    validate_config(main_cfg)
    
    # 出力ディレクトリの設定
    output_dir_path = Path(main_cfg.path.output_dir) 

    ## 事前学習モデルとembedding fileの両方が指定されている時
    if main_cfg.pretrain.framework is not None and main_cfg.pretrain.timestamp is not None and main_cfg.dataset.embedding_file is not None:
        output_dir_path /= "combined_representation"
    
    ## 事前学習モデルのみの時
    elif main_cfg.pretrain.framework is not None and main_cfg.pretrain.timestamp is not None:
        if len(main_cfg.pretrain.framework) == 1 and len(main_cfg.pretrain.timestamp) == 1:
            output_dir_path /= Path(main_cfg.pretrain.framework[0]) / main_cfg.pretrain.timestamp[0]
        else:
            output_dir_path /= "combined_representation"
        
    ## embedding fileがある時
    elif main_cfg.dataset.embedding_file is not None:
        if len(main_cfg.dataset.embedding_file) == 1:
            output_dir_path /= Path(main_cfg.dataset.embedding_file[0]).stem
        else:
            output_dir_path /= "combined_representation"
    
    output_dir_path /= Path(main_cfg.model.name) / main_cfg.path.timestamp
    
    ## additional_experiment_infoがある時
    if main_cfg.experiment.additional_experiment_info is not None:
        output_dir_path /= main_cfg.experiment.additional_experiment_info
        
    output_dir_path.mkdir(parents=True, exist_ok=True)
    weight_dir = output_dir_path / "weights"
    weight_dir.mkdir(parents=True, exist_ok=True)
    
    # logの設定
    logger = logging.getLogger(__name__)
    
    # warningsの設定 (UserWarningを無視)
    warnings.filterwarnings("ignore", category=UserWarning)

    # 使用デバイスの設定
    if torch.cuda.is_available() and main_cfg.common.use_gpu:
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
        logger.info("Using GPU for training.")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU for training.")
    ctx = torch.autocast(device_type=device.type, dtype=torch.bfloat16)

    # seed固定
    random.seed(main_cfg.common.seed)
    np.random.seed(main_cfg.common.seed)
    torch.manual_seed(main_cfg.common.seed)
    torch.backends.cudnn.benchmark = False  # 再現性を無視してでも畳み込み演算速度を上げるオプション
    torch.backends.cudnn.deterministic = True  # pytorchで非決定的な操作を決定的なものにするオプション
    
    # wandbの初期化
    wandb.init(
        project=f"SSprediction",
        config=main_cfg,
        name=f"{main_cfg.experiment.name}_{main_cfg.path.timestamp}" if main_cfg.experiment.additional_experiment_info is None else f"{main_cfg.experiment.name}_{main_cfg.experiment.additional_experiment_info}_{main_cfg.path.timestamp}",
        tags=[main_cfg.experiment.name, main_cfg.experiment.additional_experiment_info, main_cfg.model.name],
        dir=output_dir_path.resolve(),
    )

    # 事前学習モデルのembedding次元の合計
    total_pretrain_embedding_dim = 0
    
    # 表現学習モデルの準備
    ## 事前学習モデルのパスのリストを作成. 事前学習モデルが指定されていない場合は空リストのままにする
    pretrain_model_paths = []
    if main_cfg.pretrain.framework is not None and main_cfg.pretrain.timestamp is not None:
        for framework, timestamp in zip(main_cfg.pretrain.framework, main_cfg.pretrain.timestamp):
            pretrain_model_path = Path(main_cfg.path.pretrain_model_dir) / framework / timestamp
            if not pretrain_model_path.exists():
                raise FileNotFoundError(f"Pretrain model path {pretrain_model_path} does not exist.")
            pretrain_model_paths.append(pretrain_model_path)
    
    ## 事前学習モデルの読み込みと初期化
    pretrain_model_infos = []

    if pretrain_model_paths:
        # checkpointsの設定
        checkpoints: list[str] = []
        # checkpointがfinal単体の場合, pretrain_model_pathsの数だけfinalを追加. そうでない場合はmain_cfg.pretrain.checkpointをそのまま使用
        if len(main_cfg.pretrain.checkpoint) == 1 and main_cfg.pretrain.checkpoint[0] == "final":
            checkpoints = ["final"] * len(pretrain_model_paths)
        else:
            checkpoints = main_cfg.pretrain.checkpoint
        
        if len(checkpoints) != len(pretrain_model_paths):
            raise ValueError("Length of pretrain.checkpoint must be the same as the number of pretrain models specified by pretrain.framework and pretrain.timestamp.")

        for pretrain_model_path, checkpoint in zip(pretrain_model_paths, checkpoints):
            # 事前学習モデルのconfigの読み込み
            pretrain_cfg_path = pretrain_model_path / f"train_config/.hydra/config.yaml"
            pretrain_cfg: PretrainMainConfig = OmegaConf.load(pretrain_cfg_path)
            
            # 互換パッチ: 古い pretrain 実験では `_target_` が "models.*" になっていることがあるため
            # 現在のパッケージ構成に合わせてフルパスに書き換える
            target = getattr(pretrain_cfg.framework, "_target_", None)
            if target == "models.data2vecModel":
                pretrain_cfg.framework._target_ = "pretrain.models.data2vecModel"
            elif target == "models.MLMModel":
                pretrain_cfg.framework._target_ = "pretrain.models.MLMModel"
            
            pretrain_model: PretrainModels.BaseModel = hydra.utils.instantiate(
                pretrain_cfg.framework,
                padding_idx=pretrain_cfg.dataset.tokens.index("<pad>"),
                num_tokens=len(pretrain_cfg.dataset.tokens),
                experiment_cfg=pretrain_cfg.experiment,
                device=device
            )
            
            # 事前学習モデルの重みの読み込み
            if checkpoint == "final":
                checkpoint = pretrain_cfg.common.max_steps

            pretrain_weight = f"weight_{checkpoint}.pth" if not main_cfg.experiment.use_teacher else f"teacher_weight_{checkpoint}.pth"
            pretrain_model._load_state_dict(torch.load(pretrain_model_path / pretrain_weight, map_location=device))

            # 事前学習モデルの情報を保存
            pretrain_model_infos.append({
                "model": pretrain_model,
                "config": pretrain_cfg,
            })
            
            logger.info(f"Loaded pretrain model from {pretrain_model_path}, checkpoint: {checkpoint}")

        # 事前学習モデルが複数指定されている場合は,それらの出力を結合して二次構造予測モデルに入力することを想定しているため, additional tokenの設定が一致しているか確認
        pretrain_model_cfgs = [info["config"] for info in pretrain_model_infos]

        use_additional_token_list = []
        for pretrain_cfg in pretrain_model_cfgs:
            use_additional_token_list.append(pretrain_cfg.experiment.use_additional_token)
            if main_cfg.experiment.use_attention:
                embed_dim = pretrain_cfg.framework.arch.n_layers * pretrain_cfg.framework.arch.n_heads
            else:
                embed_dim = pretrain_cfg.model_size.embed_dim
            total_pretrain_embedding_dim += embed_dim
        
        if len(set(use_additional_token_list)) > 1:
            raise ValueError("When using multiple pretrain models, the setting of experiment.use_additional_token must be the same across all pretrain models.")

    # embedding_fileが指定されている場合, その埋め込み表現を使用
    if main_cfg.dataset.embedding_file is not None:
        logger.info(f"Using embedding file {main_cfg.dataset.embedding_file} for training.")
    
    # データローダーの設定
    train_loader = create_dataloader(
        config=main_cfg,
        split="train",
        pretrain_cfgs=[info["config"] for info in pretrain_model_infos] if pretrain_model_infos else None,
    )

    ## 評価用はイテレータ
    val_iterators = {
        "train": create_batch_iterator(
            config=main_cfg,
            split="train",
            pretrain_cfgs=[info["config"] for info in pretrain_model_infos] if pretrain_model_infos else None
        ),
    }
    if main_cfg.common.validation:
        val_iterators["validation"] = create_batch_iterator(
            config=main_cfg,
            split="validation",
            pretrain_cfgs=[info["config"] for info in pretrain_model_infos] if pretrain_model_infos else None
        )

    # knotfoldを使用する場合はreference用のデータローダーも作成
    if main_cfg.model.name == "knotfold":
        ref_loader = create_dataloader(
            config=main_cfg,
            split="reference",
            pretrain_cfgs=[info["config"] for info in pretrain_model_infos] if pretrain_model_infos else None,
        )

    total_steps = main_cfg.common.max_epochs * len(train_loader) # 二次構造予測モデルの学習ステップ数
    eval_interval = max(1, len(train_loader) // main_cfg.common.eval_per_epoch) # 評価の頻度 (train_loaderの長さに基づいて決定)

    # embedding次元の取得
    embedding_dim = 0

    ## 事前学習モデルの埋め込み次元の合計
    if pretrain_model_infos:
        embedding_dim += total_pretrain_embedding_dim
    
    ## embedding_fileから得られる埋め込みの次元.
    if main_cfg.dataset.embedding_file is not None:
        embedding_dim += get_embedding_dim(train_loader)
        
    logger.info(f"embedding dimension: {embedding_dim} (from number of pretrain models: {len(pretrain_model_infos)}, from embedding file: {len(main_cfg.dataset.embedding_file) if main_cfg.dataset.embedding_file is not None else 0})")
    
    logger.info(f"Run on {output_dir_path}, with device {device}")
    logger.info(f"Model: {main_cfg.model.name}")
    logger.info(f"Setting seed: {main_cfg.common.seed}")
    logger.info(f"Experiment: {main_cfg.experiment.name}")
    
    # iteration
    for iteration in range(main_cfg.common.iterations):
        logger.info(f"--------Iteration {iteration + 1}/{main_cfg.common.iterations}--------")

        # 二次構造予測モデルの動的インポートと初期化
        model: KnotFoldModel = hydra.utils.instantiate(
            main_cfg.model,
            pretrain_models=[info["model"] for info in pretrain_model_infos] if pretrain_model_infos else None,
            embedding_dim=embedding_dim,
            use_attention=main_cfg.experiment.use_attention,
            device=device
        )
        
        optimizer: torch.optim.Optimizer = hydra.utils.instantiate(main_cfg.optimizer, model.parameters())
        
        lr_scheduler = CosineScheduler(
            optimizer=optimizer,
            total_steps=total_steps,
            warmup_steps=total_steps // 10,  # 学習ステップの10%をウォームアップに使用
            max_lr=main_cfg.model.lr,
            min_lr=main_cfg.model.min_lr,
        )
        
        # reference用のモデルの動的インポートと初期化 (今はknotfoldのみ)
        if main_cfg.model.name == "knotfold":
            ref_model: KnotFoldModel = hydra.utils.instantiate(
                main_cfg.model,
                pretrain_models=[info["model"] for info in pretrain_model_infos] if pretrain_model_infos else None,
                embedding_dim=embedding_dim,
                use_attention=main_cfg.experiment.use_attention,
                device=device,
                reference=True
            )
            
            ref_optimizer: torch.optim.Optimizer = hydra.utils.instantiate(main_cfg.optimizer, ref_model.parameters())
            
            ref_total_steps = main_cfg.model.max_ref_epochs * len(ref_loader) # リファレンスモデルの学習ステップ数
            ref_lr_scheduler = CosineScheduler(
                optimizer=ref_optimizer,
                total_steps=ref_total_steps,
                warmup_steps=ref_total_steps // 10,  # 学習ステップの10%をウォームアップに使用
                max_lr=main_cfg.model.lr,
                min_lr=main_cfg.model.min_lr,
            )
        
        # 学習ループの開始
        model = torch.compile(model)
        if main_cfg.model.name == "knotfold":
            ref_model = torch.compile(ref_model)
        
        # 初めにreference modelの学習を行う (knotfoldのみ)
        if main_cfg.model.name == "knotfold":
            # referenceモデルの保存用
            min_loss = float("inf")
            best_model_state_dict = None
            save_epoch = 0
            
            # wandbのログ出力用
            ref_loss_log = {}
            ref_val_loss_log = {}
            
            ref_model.train()
            logger.info("--------Start training reference model--------")
            with tqdm(total=ref_total_steps, desc="Reference Model Training") as ref_pbar:
                for ref_epoch in range(main_cfg.model.max_ref_epochs):
                    for ref_batch_idx, ref_batch in enumerate(ref_loader):
                        with ctx:
                            ref_loss = ref_model._train(ref_batch)
                            
                        ref_loss.backward()
                        
                        step = ref_epoch * len(ref_loader) + ref_batch_idx
                        
                        # 学習を進める前にvalidation
                        if step % eval_interval == 0 or step == ref_total_steps - 1:
                            ref_loss_log[step] = ref_loss.item()
                            
                            if main_cfg.common.validation:
                                ref_model.eval()
                                with torch.no_grad():
                                    val_iterator = val_iterators["validation"]
                                    val_loss = 0.0
                                    for val_step in range(main_cfg.common.eval_steps):
                                        # create_batch_iterator は (batch, epoch) を返すので batch のみ取り出す
                                        val_batch, _ = next(val_iterator)
                                        val_loss += ref_model._test(val_batch)["loss"].item()
                                    val_loss /= main_cfg.common.eval_steps
                                    ref_val_loss_log[step] = val_loss
                                    
                                    # モデルの保存
                                    if val_loss < min_loss:
                                        min_loss = val_loss
                                        best_model_state_dict = copy.deepcopy(ref_model.state_dict())
                                        save_epoch = step / (eval_interval * main_cfg.common.eval_per_epoch)
                            else:
                                if ref_loss < min_loss:
                                    min_loss = ref_loss
                                    best_model_state_dict = copy.deepcopy(ref_model.state_dict())
                                    save_epoch = step / (eval_interval * main_cfg.common.eval_per_epoch)
                                    
                                ref_model.train()
                        
                        ref_optimizer.step()
                        ref_lr_scheduler.step()
                        ref_optimizer.zero_grad()
                        
                        ref_pbar.update(1)
                        ref_pbar.set_postfix({"loss": ref_loss.item()})
                        
            # 最終的に最も性能の良かったモデルを保存
            assert best_model_state_dict is not None, "No model was saved during training. Please check if the training loop is working correctly."
            logger.info(f"Best reference model found at epoch {save_epoch:.2f} with validation loss {min_loss:.4f}. Saving model...")
            torch.save(best_model_state_dict, output_dir_path / f"weights/reference_{iteration}.pth")
        
        
        # 二次構造予測モデルの学習
        # モデルの保存用
        min_loss = float("inf")
        best_model_state_dict = None
        save_epoch = 0
        
        model.train()
        logger.info("--------Start training main model--------")
        with tqdm(total=total_steps, desc="Main Model Training") as pbar:
            for epoch in range(main_cfg.common.max_epochs):
                for train_batch_idx, batch in enumerate(train_loader):
                    with ctx:
                        loss = model._train(batch)
                        
                    loss.backward()
                    
                    step = epoch * len(train_loader) + train_batch_idx
                    
                    # 学習を進める前にvalidationとwandbへのログ出力
                    if step % eval_interval == 0 or step == total_steps - 1:
                        # referenceモデルのwandbへのログ出力 (knotfoldのみ)
                        if main_cfg.model.name == "knotfold":
                            ref_loss = ref_loss_log.get(step, None)
                            ref_val_loss = ref_val_loss_log.get(step, None)
                            if ref_loss is not None:
                                wandb.log({f"reference_loss_{iteration}": ref_loss}, step=step)
                            if ref_val_loss is not None:
                                wandb.log({f"reference_validation_loss_{iteration}": ref_val_loss}, step=step)
                        
                        for split, val_iterator in val_iterators.items():
                            model.eval()
                            with torch.no_grad():
                                val_loss = 0.0
                                for val_step in range(main_cfg.common.eval_steps):
                                    # create_batch_iterator は (batch, epoch) を返すので batch のみ取り出す
                                    val_batch, _ = next(val_iterator)
                                    val_loss += model._test(val_batch)["loss"].item()
                                val_loss /= main_cfg.common.eval_steps
                                wandb.log({f"{split}_loss_{iteration}": val_loss}, step=step)
                                
                                # モデルの保存
                                if split == "validation":
                                    if val_loss < min_loss:
                                        min_loss = val_loss
                                        best_model_state_dict = copy.deepcopy(model.state_dict())
                                        save_epoch = step / (eval_interval * main_cfg.common.eval_per_epoch)
                                elif split == "train":
                                    if val_loss < min_loss:
                                        min_loss = val_loss
                                        best_model_state_dict = copy.deepcopy(model.state_dict())
                                        save_epoch = step / (eval_interval * main_cfg.common.eval_per_epoch)

                            model.train()
                    
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    
                    pbar.update(1)
                    pbar.set_postfix({"loss": loss.item()})
                    
        # 最終的に最も性能の良かったモデルを保存
        assert best_model_state_dict is not None, "No model was saved during training. Please check if the training loop is working correctly."
        logger.info(f"Best model found at epoch {save_epoch:.2f} with validation loss {min_loss:.4f}. Saving model...")
        torch.save(best_model_state_dict, output_dir_path / f"weights/prior_{iteration}.pth")
        
if __name__ == "__main__":
    main()
