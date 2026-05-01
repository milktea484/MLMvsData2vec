import logging
import random
import warnings
from pathlib import Path

import h5py
import hydra
import numpy as np
import pandas as pd
import torch
from conf.config import MainConfig as TrainMainConfig
from conf.test_config import MainConfig
from dataset import create_dataloader
from metrics import calculate_auc, calculate_confusion_matrix
from models import KnotFoldModel
from omegaconf import OmegaConf
from tqdm import tqdm
from utils import (get_embedding_dim, load_past_SSpredictor_cfg, setup_test_config, visualize_auc,
                   visualize_probability_matrix)

from pretrain.conf.config import MainConfig as PretrainMainConfig
from pretrain.models import BaseModel as PretrainBaseModel

setup_test_config()

@hydra.main(version_base=None, config_path="conf", config_name="test_config")
def main(cfg: MainConfig):

    # DictConfigをpythonオブジェクトに変換 (listの読み込みのため)
    structured_cfg = OmegaConf.merge(OmegaConf.structured(MainConfig), cfg)
    main_cfg : MainConfig = OmegaConf.to_object(structured_cfg)

    # 訓練済みモデルのPathの設定
    train_model_path = Path(main_cfg.path.output_dir)

    ## 事前学習モデルとembedding_fileの両方がある時
    if main_cfg.pretrain.framework is not None and main_cfg.pretrain.timestamp is not None and main_cfg.dataset.embedding_file is not None:
        train_model_path /= "combined_representation"
    
    ## 事前学習モデルのみの時
    elif main_cfg.pretrain.framework is not None and main_cfg.pretrain.timestamp is not None:
        if len(main_cfg.pretrain.framework) == 1 and len(main_cfg.pretrain.timestamp) == 1:
            train_model_path /= Path(main_cfg.pretrain.framework[0]) / main_cfg.pretrain.timestamp[0]
        else:
            train_model_path /= "combined_representation"
        
    ## embedding fileのみの時
    elif main_cfg.dataset.embedding_file is not None:
        if len(main_cfg.dataset.embedding_file) == 1:
            train_model_path /= Path(main_cfg.dataset.embedding_file[0]).stem
        else:
            train_model_path /= "combined_representation"
    
    train_model_path /= Path(main_cfg.SStrain_model_path.model_name) / main_cfg.SStrain_model_path.timestamp
    
    ## additional_experiment_infoがある時
    if main_cfg.experiment.additional_experiment_info is not None:
        train_model_path /= main_cfg.experiment.additional_experiment_info

    ## ディレクトリの存在確認
    if not train_model_path.exists():
        raise FileNotFoundError(f"Model path {train_model_path} does not exist.")
    
    # 出力ディレクトリの作成
    output_dir_path = train_model_path / "test_results" / f"{main_cfg.path.timestamp}"
    output_dir_path.mkdir(parents=True, exist_ok=True)
    
    # logの設定
    logging.basicConfig(
        level=logging.INFO,  # Set the minimum log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format="%(asctime)s - %(name)s.%(lineno)d - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),  # Log to console
            logging.FileHandler(output_dir_path / "log_test.txt", mode="w"),
        ],
    )
    logger = logging.getLogger(__name__)
    
    # warningsの設定 (UserWarningを無視)
    warnings.filterwarnings("ignore", category=UserWarning)
    
    # 訓練済みモデルと対応するhydraconfigの読み込み
    train_cfg_path = Path(main_cfg.path.output_dir) / "hydra_configs" / main_cfg.SStrain_model_path.timestamp / "train_config/.hydra/config.yaml"
    if not train_cfg_path.exists():
        raise FileNotFoundError(f"Train config path {train_cfg_path} does not exist.")
    
    train_cfg = load_past_SSpredictor_cfg(train_cfg_path)
    
    # 設定の上書き
    ## test_file
    train_cfg.dataset.test_file = main_cfg.dataset.test_file
    
    ## iterations
    if main_cfg.common.iterations is None:
        main_cfg.common.iterations = train_cfg.common.iterations
    else:
        if main_cfg.common.iterations > train_cfg.common.iterations:
            raise ValueError(f"Test config iterations {main_cfg.common.iterations} cannot be greater than train config iterations {train_cfg.common.iterations}.")
    if main_cfg.common.iterations < 1:
        raise ValueError(f"Test config iterations must be a positive integer.")
    
    ## knotfoldのkf_lambdaのリストの作成
    kf_lambda_list = None
    if train_cfg.model.name == "knotfold":
        kf_lambda_list = np.arange(main_cfg.experiment.kf_lambda_cfg.min, main_cfg.experiment.kf_lambda_cfg.max + main_cfg.experiment.kf_lambda_cfg.step, main_cfg.experiment.kf_lambda_cfg.step).tolist()

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
    
    # embedding次元数のリスト
    total_pretrain_embedding_dim = 0

    # 表現学習モデルの準備
    ## 事前学習モデルのパスのリストを作成. 事前学習モデルが指定されていない場合は空リストのままにする
    pretrain_model_paths = []
    if train_cfg.pretrain.framework is not None and train_cfg.pretrain.timestamp is not None:
        for framework, timestamp in zip(train_cfg.pretrain.framework, train_cfg.pretrain.timestamp):
            pretrain_model_path = Path(train_cfg.path.pretrain_model_dir) / framework / timestamp
            if not pretrain_model_path.exists():
                raise FileNotFoundError(f"Pretrain model path {pretrain_model_path} does not exist.")
            pretrain_model_paths.append(pretrain_model_path)

    ## 事前学習モデルの読み込みと初期化
    pretrain_model_infos = []

    if pretrain_model_paths:
        # checkpointsの設定
        checkpoints: list[str] = []
        # checkpointがfinal単体の場合, pretrain_model_pathsの数だけfinalを追加. そうでない場合はcfg.pretrain.checkpointをそのまま使用
        if len(train_cfg.pretrain.checkpoint) == 1 and train_cfg.pretrain.checkpoint[0] == "final":
            checkpoints = ["final"] * len(pretrain_model_paths)
        else:
            checkpoints = train_cfg.pretrain.checkpoint
        
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
            
            pretrain_model: PretrainBaseModel = hydra.utils.instantiate(
                pretrain_cfg.framework,
                padding_idx=pretrain_cfg.dataset.tokens.index("<pad>"),
                num_tokens=len(pretrain_cfg.dataset.tokens),
                experiment_cfg=pretrain_cfg.experiment,
                device=device
            )
        
            # 事前学習モデルの重みの読み込み
            if checkpoint == "final":
                checkpoint = pretrain_cfg.common.max_steps

            pretrain_weight = f"weight_{checkpoint}.pth" if not train_cfg.experiment.use_teacher else f"teacher_weight_{checkpoint}.pth"
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
            if train_cfg.experiment.use_attention:
                embed_dim = pretrain_cfg.framework.arch.n_layers * pretrain_cfg.framework.arch.n_heads
            else:
                embed_dim = pretrain_cfg.model_size.embed_dim
            total_pretrain_embedding_dim += embed_dim
        
        if len(set(use_additional_token_list)) > 1:
            raise ValueError("When using multiple pretrain models, the setting of experiment.use_additional_token must be the same across all pretrain models.")

    # embedding_fileが指定されている場合, その埋め込み表現を使用
    if train_cfg.dataset.embedding_file is not None:
        logger.info(f"Using embedding file {train_cfg.dataset.embedding_file} for training.")
        
    # データローダーの設定
    test_loader = create_dataloader(
        config=train_cfg,
        split="test",
        pretrain_cfgs=[info["config"] for info in pretrain_model_infos] if pretrain_model_infos else None,
    )
    
    # embedding次元数の取得
    embedding_dim = 0

    ## 事前学習モデルの埋め込み次元の合計
    if pretrain_model_infos:
        embedding_dim += total_pretrain_embedding_dim

    ## embedding_fileから得られる埋め込みの次元
    if train_cfg.dataset.embedding_file is not None:
        embedding_dim += get_embedding_dim(test_loader)

    logger.info(f"embedding dimension: {embedding_dim}")
    
    logger.info(f"Run on {train_model_path}, with device {device}")
    logger.info(f"Model: {train_cfg.model.name}")
    logger.info(f"Setting seed: {main_cfg.common.seed}")
    logger.info(f"Experiment name: {main_cfg.experiment.name}")
    
    # iterationごとのモデルのリスト
    models = []
    if train_cfg.model.name == "knotfold":
        ref_models = []
    
    for iteration in range(main_cfg.common.iterations):
        logger.info(f"--------Iteration {iteration + 1}/{main_cfg.common.iterations}--------")

        # 二次構造予測モデルの動的インポートと初期化
        model: KnotFoldModel = hydra.utils.instantiate(
            train_cfg.model,
            pretrain_models=[info["model"] for info in pretrain_model_infos] if pretrain_model_infos else None,
            embedding_dim=embedding_dim,
            use_attention=train_cfg.experiment.use_attention,
            device=device
        )
        model.eval()
        
        # 訓練済みモデルの重みの読み込み
        model._load_state_dict(torch.load(train_model_path / f"weights/prior_{iteration}.pth", map_location=device))
        models.append(model)
        
        # reference用のモデルの動的インポートと初期化 (今はknotfoldのみ)
        if train_cfg.model.name == "knotfold":
            ref_model: KnotFoldModel = hydra.utils.instantiate(
                train_cfg.model,
                pretrain_models=[info["model"] for info in pretrain_model_infos] if pretrain_model_infos else None,
                embedding_dim=embedding_dim,
                use_attention=train_cfg.experiment.use_attention,
                device=device,
                reference=True
            )
            
            # reference用のモデルの重みの読み込み
            ref_model._load_state_dict(torch.load(train_model_path / f"weights/reference_{iteration}.pth", map_location=device))
            ref_models.append(ref_model)

    # 全体の結果の保存
    overall_results = {"test_losses": []}
    if train_cfg.model.name == "knotfold":
        overall_results["ref_test_losses"] = []
    
    # probability_matrixの保存用
    if main_cfg.common.save_probability_matrix:
        probability_matrix_path = output_dir_path / "predicted_probability_matrices.h5"
        hdf5_file = h5py.File(probability_matrix_path, "w")
    
    # ROCやPRを用いた評価をするための混合行列. 各要素はサイズlen(thresholds)のnumpy配列になる
    confusion_dict = {
        "tp": 0,
        "tn": 0,
        "fp": 0,
        "fn": 0,
    }
    
    # predictの引数の用意
    # cpuメモリに保存することになるので, メモリエラーに注意
    predict_args = {
        "batch_list": [],
    }
    
    if train_cfg.model.name == "knotfold":
        predict_args["kf_lambda_list"] = kf_lambda_list

    # バッチごとに処理
    for i, batch in enumerate(tqdm(test_loader, desc="Testing")):
        
        # iterationごとの結果を保存するための変数
        iter_results = {
            "logits": [],
            "test_losses": [],
        }
        if train_cfg.model.name == "knotfold":
            iter_results["ref_logits"] = []
            iter_results["ref_test_losses"] = []
        
        with ctx, torch.inference_mode():
            # iterationごとにモデルのテストを実行
            for iteration in range(main_cfg.common.iterations):
                outputs = models[iteration]._test(batch)
                iter_results["logits"].append(outputs["logits"])
                iter_results["test_losses"].append(outputs["loss"])
                
                if train_cfg.model.name == "knotfold":
                    ref_outputs = ref_models[iteration]._test(batch)
                    iter_results["ref_logits"].append(ref_outputs["logits"])
                    iter_results["ref_test_losses"].append(ref_outputs["loss"])
            
            # iterationごとの予測確率と損失を平均
            mean_logits = torch.mean(torch.stack(iter_results["logits"], dim=0), dim=0) # (iterations, B, L, L) -> (B, L, L)
            mean_test_loss = torch.mean(torch.stack(iter_results["test_losses"], dim=0))
            
            if train_cfg.model.name == "knotfold":
                mean_ref_logits = torch.mean(torch.stack(iter_results["ref_logits"], dim=0), dim=0) # (iterations, B, L, L) -> (B, L, L)
                mean_ref_test_loss = torch.mean(torch.stack(iter_results["ref_test_losses"], dim=0))
                # overall_results["ref_y_probs"].append(mean_ref_y_prob.detach().cpu())
                overall_results["ref_test_losses"].append(mean_ref_test_loss.detach().cpu().item())
            
            # 評価と予測
            for batch_idx, zip_tuple in enumerate(zip(batch["seq_ids"], batch["sequences"], batch["lengths"], batch["bp_matrices"], mean_logits)):
                seq_id, sequence, length, gt_bp_matrix, pred_bp_prob = zip_tuple
                
                # paddingの除去
                gt_bp_matrix = gt_bp_matrix[:length, :length]
                pred_bp_prob = pred_bp_prob[:length, :length]
                
                # pred_bp_probはロジットなのでシグモイド関数を通して確率に変換
                pred_bp_prob = torch.sigmoid(pred_bp_prob.fill_diagonal_(float("-inf")))

                # 全体の結果の保存
                overall_results["test_losses"].append(mean_test_loss.detach().cpu().item())
                if main_cfg.common.save_probability_matrix:
                    hdf5_file.create_dataset(seq_id, data=pred_bp_prob.to(torch.float32).detach().cpu().numpy())
                    
                # 初回ならばシーケンスごとの予測確率行列を可視化して保存
                if i == 0 and batch_idx == 0:
                    probability_matrix_path = output_dir_path / f"{seq_id}_probability_matrix.png"
                    visualize_probability_matrix(
                        gt_bp_matrix=gt_bp_matrix.to(torch.float32).detach().cpu(),
                        probability_matrix=pred_bp_prob.to(torch.float32).detach().cpu(),
                        output_path=probability_matrix_path
                    )

                # 混合行列の更新
                if main_cfg.common.evaluation:
                    batch_confusion_dict = calculate_confusion_matrix(
                        gt_bp_matrix=gt_bp_matrix.to(torch.float32).detach().cpu(),
                        probability_matrix=pred_bp_prob.to(torch.float32).detach().cpu(),
                        step=main_cfg.evaluation.auc_step
                    )
                    confusion_dict["tp"] += batch_confusion_dict["tp"]
                    confusion_dict["tn"] += batch_confusion_dict["tn"]
                    confusion_dict["fp"] += batch_confusion_dict["fp"]
                    confusion_dict["fn"] += batch_confusion_dict["fn"]

                # predictの引数の用意
                if main_cfg.common.prediction:
                    predict_args_batch = {
                        "seq_id": seq_id,
                        "sequence": sequence,
                        "length": length,
                        "gt_bp_matrix": gt_bp_matrix, # (L, L)
                        "pred_bp_prob": pred_bp_prob, # (L, L)
                    }

                    # knotfoldの場合
                    if train_cfg.model.name == "knotfold":
                        ref_bp_prob = mean_ref_logits[batch_idx][:length, :length]
                        ref_bp_prob = torch.sigmoid(ref_bp_prob)
                        predict_args_batch["ref_bp_prob"] = ref_bp_prob # (L, L)
                        
                    predict_args["batch_list"].append(predict_args_batch)
    
    overall_results["test_losses"] = np.mean(overall_results["test_losses"])
    if train_cfg.model.name == "knotfold":
        overall_results["ref_test_losses"] = np.mean(overall_results["ref_test_losses"])
    
    # probability_matrixの保存を閉じる
    if main_cfg.common.save_probability_matrix:
        hdf5_file.close()
    
    # 評価
    if main_cfg.common.evaluation:
        # 全体のROC AUCやPR AUCの計算
        figure_materials, auc_dict = calculate_auc(confusion_dict)
        auc_figure_path = output_dir_path / "auc_curve.png"
        
        # ROC曲線とPR曲線の描画
        visualize_auc(figure_materials, output_path=auc_figure_path)
        logger.info(f"Saved ROC and PR curve figure to {auc_figure_path}")
        overall_results["roc_auc"] = auc_dict["roc_auc"]
        overall_results["pr_auc"] = auc_dict["pr_auc"]
        logger.info(f"ROC AUC: {auc_dict['roc_auc']:.4f}, PR AUC: {auc_dict['pr_auc']:.4f}")
    
    # 予測
    if main_cfg.common.prediction:
        results = model.predict(**predict_args)
        
        # 結果の保存
        ## knotfoldのkf_lambdaごとの結果の保存
        if train_cfg.model.name == "knotfold":
            # kf_lambda_resultsをcsvに保存
            kf_lambda_results = results["kf_lambda_results"]
            kf_lambda_results_list = []
            for kf_lambda, scores in kf_lambda_results.items():
                kf_lambda_results_list.append({
                    "kf_lambda": kf_lambda,
                    "is_optimal": scores["is_optimal"],
                    "precision": scores["precision"],
                    "recall": scores["recall"],
                    "f1": scores["f1"],
                })
            kf_lambda_results_path = output_dir_path / "kf_lambda_results.csv"
            kf_lambda_results_df = pd.DataFrame(kf_lambda_results_list)
            kf_lambda_results_df.to_csv(kf_lambda_results_path, index=False)
            logger.info(f"Saved kf_lambda results to {kf_lambda_results_path}")
            
            #is_optimalがTrueのスコアをoverall_resultsに保存
            optimal_kf_lambda_results = kf_lambda_results_df[kf_lambda_results_df["is_optimal"] == True]
            if len(optimal_kf_lambda_results) > 0:
                overall_results["kf_lambda"] = optimal_kf_lambda_results.iloc[0]["kf_lambda"]
                overall_results["precision"] = optimal_kf_lambda_results.iloc[0]["precision"]
                overall_results["recall"] = optimal_kf_lambda_results.iloc[0]["recall"]
                overall_results["f1"] = optimal_kf_lambda_results.iloc[0]["f1"]
        
        ## 予測結果の保存
        prediction_to_csv_info = []
        for pred_result in results["prediction_results"]:
            pred_result_info = {
                "id": pred_result["seq_id"],
                "sequence": pred_result["sequence"],
                "base_pairs": pred_result["pairs"],
                "len": pred_result["length"],
                "precision": pred_result["scores"]["precision"],
                "recall": pred_result["scores"]["recall"],
                "f1": pred_result["scores"]["f1"],
            }
            prediction_to_csv_info.append(pred_result_info)
        
        prediction_results_path = output_dir_path / "prediction_results.csv"
        pd.DataFrame(prediction_to_csv_info).to_csv(prediction_results_path, index=False)
        logger.info(f"Saved prediction results to {prediction_results_path}")
        
        ## 全体の結果の保存
        overall_results_path = output_dir_path / "overall_results.csv"
        pd.DataFrame([overall_results]).to_csv(overall_results_path, index=False)
        logger.info(f"Saved overall results to {overall_results_path}")
            
if __name__ == "__main__":
    main()
