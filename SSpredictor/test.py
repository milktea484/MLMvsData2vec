import copy
import logging
import random
import warnings
from pathlib import Path

import hydra
import numpy as np
import pretrain.models as PretrainModels
import torch
from conf.config import MainConfig as TrainMainConfig
from conf.test_config import MainConfig
from dataset import create_dataloader
from metrics import calculate_auc, update_confusion_matrix
from models import KnotFoldModel
from modules import CosineScheduler
from omegaconf import OmegaConf
from pretrain.conf.config import MainConfig as PretrainMainConfig
from tqdm import tqdm
from utils import get_embedding_dim, setup_test_config

setup_test_config()

@hydra.main(version_base=None, config_path="conf", config_name="test_config")
def main(cfg: MainConfig):
    
    # 訓練済みモデルのPathの設定
    train_model_path = Path(cfg.path.output_dir)
    
    ## 事前学習モデルがある時
    if cfg.pretrain.framework is not None and cfg.pretrain.timestamp is not None:
        train_model_path = train_model_path / cfg.pretrain.framework / cfg.pretrain.timestamp
        
    ## embedding fileがある時
    elif cfg.dataset.embedding_file is not None:
        embedding_name = cfg.dataset.embedding_file.split(".")[0]
        train_model_path = train_model_path / embedding_name
    
    train_model_path = train_model_path / cfg.SStrain_model_path.model_name / cfg.SStrain_model_path.timestamp
    
    ## additional_experiment_infoがある時
    if cfg.experiment.additional_experiment_info is not None:
        train_model_path = train_model_path / cfg.experiment.additional_experiment_info

    ## ディレクトリの存在確認
    if not train_model_path.exists():
        raise FileNotFoundError(f"Model path {train_model_path} does not exist.")
    
    # 訓練済みモデルと対応するhydraconfigの読み込み
    train_cfg_path = Path(cfg.path.output_dir) / cfg.path.timestamp / "train_config" / "config.yaml"
    if not train_cfg_path.exists():
        raise FileNotFoundError(f"Train config path {train_cfg_path} does not exist.")
    
    train_cfg: TrainMainConfig = OmegaConf.load(train_cfg_path)
    
    # 設定の上書き
    train_cfg.dataset.test_file = cfg.dataset.test_file
    if cfg.common.iterations is None:
        cfg.common.iterations = train_cfg.common.iterations
    if cfg.experiment.kf_lambda is None:
        cfg.experiment.kf_lambda = [train_cfg.model.kf_lambda]

    # logの設定
    logging.basicConfig(
        level=logging.INFO,  # Set the minimum log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format="%(asctime)s - %(name)s.%(lineno)d - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),  # Log to console
            logging.FileHandler(train_model_path / "log_test.txt", mode="w"),
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
    pretrain_model_path = None
    if train_cfg.pretrain.framework is not None and train_cfg.pretrain.timestamp is not None:
        pretrain_model_path = Path(train_cfg.path.pretrain_model_dir) / train_cfg.pretrain.framework / train_cfg.pretrain.timestamp
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
        checkpoint = pretrain_cfg.common.max_steps if train_cfg.pretrain.checkpoint == "final" else train_cfg.pretrain.checkpoint
        pretrain_weight = f"weight_{checkpoint}.pth" if not train_cfg.experiment.use_teacher else f"teacher_weight_{checkpoint}.pth"
        
        pretrain_model._load_state_dict(torch.load(pretrain_model_path / pretrain_weight, map_location=device))
        
        logger.info(f"Loaded pretrain model from {pretrain_model_path}, checkpoint: {checkpoint}")
    else:
        # 事前学習モデルが指定されていない場合, embedding_fileで指定されている埋め込み表現を使用
        logger.info("No pretrain model specified. Will use embeddings from embedding_file provided.")
        
    # データローダーの設定
    test_loader = create_dataloader(config=cfg, split="test", pretrain_config=pretrain_cfg)
    
    # embedding次元数の取得
    embedding_dim = None
    if train_cfg.dataset.embedding_file is None:
        assert pretrain_model is not None, "Either a pretrain model or an embedding file must be specified to determine the embedding dimension."
        if train_cfg.experiment.use_attention:
            embedding_dim = pretrain_cfg.framework.n_layers * pretrain_cfg.framework.n_heads
        else:
            embedding_dim = pretrain_cfg.model_size.embed_dim
    else:
        embedding_dim = get_embedding_dim(test_loader, train_cfg.experiment.use_attention)
        
    logger.info(f"embedding dimension: {embedding_dim}")
    
        
    logger.info(f"Run on {train_model_path}, with device {device}")
    logger.info(f"Setting seed: {cfg.common.seed}")
    logger.info(f"Experiment name: {cfg.experiment.name}")
    
    # iterationごとのモデルのリスト
    models = []
    if train_cfg.model.name == "knotfold":
        ref_models = []
        
    # ROCやPRを用いた評価をするための混合行列. 各要素はサイズlen(thresholds)のnumpy配列になる
    confusion_dict = {
        "tp": 0,
        "tn": 0,
        "fp": 0,
        "fn": 0,
    }
    
    for iteration in range(cfg.common.iterations):
        logger.info(f"--------Iteration {iteration + 1}/{cfg.common.iterations}--------")

        # 二次構造予測モデルの動的インポートと初期化
        model: KnotFoldModel = hydra.utils.instantiate(
            train_cfg.model,
            pretrain_model=pretrain_model,
            embedding_dim=embedding_dim,
            use_attention=train_cfg.experiment.use_attention,
            device=device
        )
        
        # 訓練済みモデルの重みの読み込み
        model._load_state_dict(torch.load(train_model_path / f"weights/prior_{iteration}.pth", map_location=device))
        models.append(copy.deepcopy(model))
        
        # reference用のモデルの動的インポートと初期化 (今はknotfoldのみ)
        if train_cfg.model.name == "knotfold":
            ref_model: KnotFoldModel = hydra.utils.instantiate(
                train_cfg.model,
                pretrain_model=pretrain_model,
                embedding_dim=embedding_dim,
                use_attention=train_cfg.experiment.use_attention,
                device=device,
                reference=True
            )
            
            # reference用のモデルの重みの読み込み
            ref_model._load_state_dict(torch.load(train_model_path / f"weights/reference_{iteration}.pth", map_location=device))
            ref_models.append(copy.deepcopy(ref_model))

    # 全体の結果の保存
    overall_results = {
        "y_probs": [],
        "test_losses": [],
    }

    # バッチごとに処理
    for batch in tqdm(test_loader, desc="Testing"):
        
        # iterationごとの結果を保存するための変数
        iter_results = {
            "y_probs": [],
            "test_losses": [],
        }
        if train_cfg.model.name == "knotfold":
            iter_results["ref_y_probs"] = []
            iter_results["ref_test_losses"] = []
        
        with ctx, torch.inference_mode():
            # iterationごとにモデルのテストを実行
            for iteration in range(cfg.common.iterations):
                outputs = models[iteration]._test(batch)
                iter_results["y_probs"].append(outputs["y_prob"])
                iter_results["test_losses"].append(outputs["loss"])
                
                if train_cfg.model.name == "knotfold":
                    ref_outputs = ref_models[iteration]._test(batch)
                    iter_results["ref_y_probs"].append(ref_outputs["y_prob"])
                    iter_results["ref_test_losses"].append(ref_outputs["loss"])
            
            # iterationごとの予測確率と損失を平均
            mean_y_prob = torch.mean(torch.stack(iter_results["y_probs"], dim=0), dim=0) # (iterations, B, L, L) -> (B, L, L)
            mean_test_loss = torch.mean(torch.stack(iter_results["test_losses"], dim=0))
            overall_results["y_probs"].append(mean_y_prob.detach().cpu())
            overall_results["test_losses"].append(mean_test_loss.detach().cpu().item())
            
            if train_cfg.model.name == "knotfold":
                mean_ref_y_prob = torch.mean(torch.stack(iter_results["ref_y_probs"], dim=0), dim=0) # (iterations, B, L, L) -> (B, L, L)
                mean_ref_test_loss = torch.mean(torch.stack(iter_results["ref_test_losses"], dim=0))
                # overall_results["ref_y_probs"].append(mean_ref_y_prob.detach().cpu())
                # overall_results["ref_test_losses"].append(mean_ref_test_loss.detach().cpu().item())
            
            # 評価と予測
            for batch_idx, length, seq_id, sequence, gt_bp_matrix, pred_bp_matrix in enumerate(zip(batch["lengths"], batch["seq_ids"], batch["sequences"], batch["bp_matrices"], mean_y_prob)):
                # paddingの除去
                gt_bp_matrix = gt_bp_matrix[:length, :length]
                pred_bp_matrix = pred_bp_matrix[:length, :length]
                
                # pred_bp_matrixはロジットなのでシグモイド関数を通して確率に変換
                pred_bp_matrix = torch.sigmoid(pred_bp_matrix)

                # 混合行列の更新
                confusion_dict = update_confusion_matrix(gt_bp_matrix.cpu(), pred_bp_matrix.cpu(), step=cfg.evaluation.auc_step, confusion_dict=confusion_dict)
                
                # predictの引数の用意
                predict_args = {
                    "gt_bp_matrix": gt_bp_matrix,
                    "pred_bp_matrix": pred_bp_matrix
                }

                # knotfoldの場合
                if train_cfg.model.name == "knotfold":
                    ref_bp_matrix = mean_ref_y_prob[batch_idx][:length, :length]
                    ref_bp_matrix = torch.sigmoid(ref_bp_matrix)
                    predict_args["ref_bp_matrix"] = ref_bp_matrix
                    predict_args["kf_lambda"] = cfg.experiment.kf_lambda
                
                # 予測 (ここからつづき)
                model.predict(**predict_args)
            
            
    # 全体のROC AUCやPR AUCの計算
    figure_materials, auc_dict = calculate_auc(confusion_dict)
    
    overall_results["roc_auc"] = auc_dict["roc_auc"]
    overall_results["pr_auc"] = auc_dict["pr_auc"]