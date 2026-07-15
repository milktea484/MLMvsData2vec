import torch
from omegaconf import OmegaConf
import hydra
import random
import numpy as np
import logging
import warnings
import h5py

import pretrain
from pretrain.conf.config import MainConfig as PretrainMainConfig

from pathlib import Path
from tqdm import tqdm

from redial import REDIALlike
from conf.config import MainConfig
from utils import assess_attention, setup_config, validate_config
from dataset import create_dataloader

setup_config()

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: MainConfig):
    """
    REDIALlikeから得られたcontact mapが二次構造情報をどれだけ含んでいるかを確認する。
    また、Attention mapとの比較も行う。
    """

    # DictConfigをpythonオブジェクトに変換 (listの読み込みのため)
    structured_cfg = OmegaConf.merge(OmegaConf.structured(MainConfig), cfg)
    main_cfg : MainConfig = OmegaConf.to_object(structured_cfg)

    validate_config(main_cfg)

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
    ctx = torch.autocast(device_type=device.type, dtype=torch.float32)

    # seed固定
    random.seed(main_cfg.common.seed)
    np.random.seed(main_cfg.common.seed)
    torch.manual_seed(main_cfg.common.seed)
    torch.backends.cudnn.benchmark = False  # 再現性を無視してでも畳み込み演算速度を上げるオプション
    torch.backends.cudnn.deterministic = True  # pytorchで非決定的な操作を決定的なものにするオプション

    # pathなどの設定
    output_dir = Path(main_cfg.path.output_dir) / main_cfg.path.timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    
    # 事前学習モデルの準備
    pretrain_model_path = Path(main_cfg.path.pretrain_model_dir) / main_cfg.pretrain.framework / main_cfg.pretrain.timestamp
    if not pretrain_model_path.exists():
        raise FileNotFoundError(f"Pretrain model path {pretrain_model_path} does not exist.")
    
    # 事前学習モデルの読み込みと初期化
    # 事前学習モデルのconfigの読み込み
    pretrain_cfg_path = pretrain_model_path / f"train_config/.hydra/config.yaml"
    pretrain_cfg: PretrainMainConfig = OmegaConf.load(pretrain_cfg_path)

    # バッチサイズの変更（強制的に1にする）
    pretrain_cfg.common.batch_size = pretrain_cfg.model_size.gradient_accumulation_steps_for_test
    
    # 互換パッチ: 古い pretrain 実験では `_target_` が "models.*" になっていることがあるため
    # 現在のパッケージ構成に合わせてフルパスに書き換える
    target = getattr(pretrain_cfg.framework, "_target_", None)
    if target == "models.data2vecModel":
        pretrain_cfg.framework._target_ = "pretrain.models.data2vecModel"
    elif target == "models.MLMModel":
        pretrain_cfg.framework._target_ = "pretrain.models.MLMModel"
    
    pretrain_model: pretrain.models.BaseModel = hydra.utils.instantiate(
        pretrain_cfg.framework,
        padding_idx=pretrain_cfg.dataset.tokens.index("<pad>"),
        num_tokens=len(pretrain_cfg.dataset.tokens),
        experiment_cfg=pretrain_cfg.experiment,
        device=device
    )
    
    # 事前学習モデルの重みの読み込み
    checkpoint = main_cfg.pretrain.checkpoint
    if checkpoint == "final":
        checkpoint = pretrain_cfg.common.max_steps

    pretrain_weight = f"weight_{checkpoint}.pth" if not main_cfg.experiment.use_teacher else f"teacher_weight_{checkpoint}.pth"
    pretrain_model._load_state_dict(torch.load(pretrain_model_path / pretrain_weight, map_location=device))
    
    logger.info(f"Loaded pretrain model from {pretrain_model_path}, checkpoint: {checkpoint}")

    # データローダーの準備
    test_dataloader = create_dataloader(config=pretrain_cfg)

    # REDIALlikeの初期化
    rediallike = REDIALlike(
        pretrain_model=pretrain_model,
        pretrain_config=pretrain_cfg,
        rna_tokens=main_cfg.dataset.rna_tokens,
        device=device,
    )
    # もし、main_cfg.experiment.extract_repr_layersがNoneでない場合は、REDIALlikeのextract_repr_layersを上書きする
    if main_cfg.experiment.extract_repr_layers is not None:
        rediallike.set_extract_repr_layers(main_cfg.experiment.extract_repr_layers)
        logger.info(f"Overriding extract_repr_layers to {main_cfg.experiment.extract_repr_layers}")
    
    # バッチごとに処理
    with torch.inference_mode(), ctx:
        with h5py.File(output_dir / "scores.h5", "w") as hdf:
            hdf.create_group("scores")
            
            for batch in tqdm(test_dataloader, desc="Processing batches"):
                seq_id = batch["seq_ids"][0]  # バッチサイズは1なので、最初の要素を取得
                
                hdf.create_group(f"scores/{seq_id}")
                
                # 正解構造の取得
                bp_matrix = batch["bp_matrices"].squeeze(0).detach().cpu().numpy()    # (L, L)

                # REDIALlikeによるcontact mapの生成
                contact_map, attention_map = rediallike.generate_contact_map(batch, return_attn=main_cfg.experiment.return_attn)
                
                # contact mapの正規化
                # attention mapに合わせて，softmaxを行う (attention mapはsoftmaxを通している)
                contact_map = torch.softmax(contact_map, dim=-1)  # (len(extract_repr_layers), L, L)

                contact_map = contact_map.detach().cpu().numpy()  # (len(extract_repr_layers), L, L)
                attention_map = attention_map.detach().cpu().numpy() if attention_map is not None else None  # (n_layer*n_heads, L, L)

                for layer_idx, layer in enumerate(rediallike.extract_repr_layers):
                    hdf.create_group(f"scores/{seq_id}/layer_{layer}")
                    
                    # contact mapの評価
                    contact_map_layer = contact_map[layer_idx]
                    contact_score, _ = assess_attention(contact_map_layer, bp_matrix, single_map=True)

                    # スコアの保存
                    hdf[f"scores/{seq_id}/layer_{layer}"].create_dataset("contact_map", data=contact_score)

                    if main_cfg.experiment.return_attn:
                        # attention mapの評価
                        attn_score, max_hl = assess_attention(attention_map, bp_matrix, n_heads=pretrain_cfg.model_size.n_heads, n_layers=pretrain_cfg.model_size.n_layers, designated_layer=layer)
                        hdf[f"scores/{seq_id}/layer_{layer}"].create_dataset("attention_map", data=attn_score)
                        hdf[f"scores/{seq_id}/layer_{layer}"].create_dataset("max_head", data=max_hl)
                
            logger.info(f"Scores saved to {output_dir / 'scores.h5'}")
    
if __name__ == "__main__":
    main()
