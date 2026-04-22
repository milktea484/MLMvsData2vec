from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from conf.config import AdamWConfig, KnotFoldConfig, MainConfig
from conf.test_config import MainConfig as TestMainConfig
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf


def seq2token(
    sequences: list[str],
    tokens: list[str] = ["A", "C", "G", "U", "N", "<mask>", "<pad>", "<cls>", "<eos>"],
    other_tokens: list[str] = ["B", "D", "F", "I", "H", "K", "M", "S", "R", "W", "V", "Y", "X"],
    use_additional_token: bool = False
) -> list[torch.Tensor]:
    """
    文字列のシーケンスをトークンIDのテンソルに変換する関数
    Args:
        sequences (list[str]): 文字列のシーケンスのリスト
        tokens (list[str]): トークンのリスト
        other_tokens (list[str]): その他Nに変換される塩基のリスト
        use_additional_token (bool): CLS, EOSトークンを使用するかどうか
    Returns:
        list[torch.Tensor]: トークンIDのテンソルのリスト
    """
    mapping = {nt: idx for idx, nt in enumerate(tokens)}
    mapping.update({nt: tokens.index("N") for nt in other_tokens})
    mapping["T"] = mapping["U"]
    
    token_seqs = []
    for seq in sequences:
        token_seq = [mapping.get(nt) for nt in seq]
        if use_additional_token:
            token_seq = [mapping["<cls>"]] + token_seq + [mapping["<eos>"]]
        
        if any(v is None for v in token_seq):
            raise ValueError("Invalid nucleotide found")
        token_seqs.append(torch.tensor(token_seq, dtype=torch.uint8))
        
    return token_seqs

def bp2matrix(L, base_pairs) -> torch.Tensor:
    """
    リスト形式のBase pairsを行列に変換する関数
    Args:
        L (int): シーケンスの長さ
        base_pairs (list[tuple[int, int]]): Base pairのリスト
    Returns:
        torch.Tensor: Base pairの行列

    """
    
    matrix = torch.zeros((L, L))
    if base_pairs != []:
        # base pairs are 1-based
        bp = torch.tensor(base_pairs) - 1
        matrix[bp[:, 0], bp[:, 1]] = 1
        matrix[bp[:, 1], bp[:, 0]] = 1

    return matrix

def get_embedding_dim(loader: torch.utils.data.DataLoader) -> int:
    """
    embedding_fileから得られる埋め込みの次元のみを返す関数. 事前学習モデルの埋め込み次元は別途用意する必要あり
    Args:
        loader: データローダー
    Returns:
        int: 埋め込みの次元数
    """
    batch_elem = next(iter(loader)) # (B, L, E) or (B, E, L, L)

    # embeddingsがあるときはその次元数を返す. ないときは0を返す
    if batch_elem["embeddings"] is not None:
        use_attention = batch_elem["embeddings"].dim() == 4
        return batch_elem["embeddings"].shape[1] if use_attention else batch_elem["embeddings"].shape[-1]
    
    else:
        return 0

def outer_concat(t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
    """
    t1とt2のouter concatを計算する関数. 配列特徴量を入力とする時に使用
    Args:
        t1: shape = B x L x E
        t2: shape = B x L x E
    Returns:
        torch.Tensor: shape = B x L x L x 2E
    """
    
    assert t1.shape == t2.shape, f"Shapes of input tensors must match! ({t1.shape} != {t2.shape})"

    seq_len = t1.shape[1]
    a = t1.unsqueeze(-2).expand(-1, -1, seq_len, -1)
    b = t2.unsqueeze(-3).expand(-1, seq_len, -1, -1)

    return torch.concat((a, b), dim=-1)

def symmetrize(x: torch.Tensor, zero_diagonal: bool = True) -> torch.Tensor:
    """
    行列を対称化する関数. 対角線は0にする.
    Args:
        x (torch.Tensor): shape = B x E x L x L or B x L x L
        zero_diagonal (bool): 対角線を0にするかどうか
    Returns:
        torch.Tensor: shape = B x E x L x L or B x L x L
    """
    
    output = (x + x.transpose(-2, -1)) / 2
    if zero_diagonal:
        output_triu = torch.triu(output, diagonal=1)
        output = output_triu + output_triu.transpose(-2, -1)
    
    return output

def apc(x: torch.Tensor) -> torch.Tensor:
    """
    Perform average product correct, used for contact prediction.
    (by https://github.com/facebookresearch/esm/blob/2b369911bb5b4b0dda914521b9475cad1656b2ac/esm/modules.py#L32)
    
    Args:
        x: shape = B x E x L x L
    Returns:
        torch.Tensor: shape = B x E x L x L
    """
    
    a1 = x.sum(-1, keepdims=True)
    a2 = x.sum(-2, keepdims=True)
    a12 = x.sum((-1, -2), keepdims=True)

    avg = a1 * a2
    avg.div_(a12)
    normalized = x - avg
    
    return normalized

def visualize_auc(fig_materials: dict, output_path: Path):
    """
    ROC曲線とPR曲線を描画する関数
    Args:
        fig_materials (dict): ROC曲線とPR曲線の図示するための材料を格納する辞書
        output_path (Path): 出力ファイルのパス
    """
    
    fpr = fig_materials["fpr"]
    precision = fig_materials["precision"]
    recall = fig_materials["recall"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # ROC曲線の描画
    axes[0].plot(fpr, fpr, linestyle="--", label="Random")
    axes[0].plot(fpr, recall, marker=".", label="ROC curve")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve")
    axes[0].legend()

    # PR曲線の描画
    axes[1].plot(recall, precision, marker=".", label="PR curve")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve")
    axes[1].legend()

    fig.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
def visualize_probability_matrix(gt_bp_matrix: torch.Tensor, probability_matrix: torch.Tensor, output_path: Path):
    """
    予測された塩基対確率行列を描画する関数
    Args:
        gt_bp_matrix (torch.Tensor): shape = L x Lの真の二次構造行列
        probability_matrix (torch.Tensor): shape = L x Lの塩基対確率行列
        output_path (Path): 出力ファイルのパス
    """
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    im1 = axes[0].imshow(gt_bp_matrix, cmap="viridis", vmin=0, vmax=1)
    axes[0].set_title("Ground Truth Base Pair Matrix")
    fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    
    im2 = axes[1].imshow(probability_matrix, cmap="viridis", vmin=0, vmax=1)
    axes[1].set_title("Predicted Base Pair Probability Matrix")
    fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    
    fig.tight_layout()
    plt.savefig(output_path)
    plt.close()
    

def setup_config():
    """
    OmegaconfのカスタムリゾルバとHydraのConfigStoreへの設定登録を行う関数
    """
    if not OmegaConf._get_resolver("div"):
        OmegaConf.register_new_resolver("div", lambda x, y: int(x / y))

    if not OmegaConf._get_resolver("mul"):
        OmegaConf.register_new_resolver("mul", lambda x, y: x * y)

    if not hasattr(setup_config, "is_registered"):
        cs = ConfigStore.instance()

        cs.store(name="base_config_schema", node=MainConfig)
        
        cs.store(group="model", name="knotfold_schema", node=KnotFoldConfig)
        
        cs.store(group="optimizer", name="adamw_schema", node=AdamWConfig)
        
        setup_config.is_registered = True

def validate_config(cfg: MainConfig):
    """
    設定の妥当性を確認する関数
    Args:
        cfg (MainConfig): 検証する設定オブジェクト
    """
    
    # "dataset.embedding_file"または"pretrain.framework, pretrain.timestampの両方"のどちらも指定されていない時はエラー
    if (cfg.pretrain.framework is None or cfg.pretrain.timestamp is None) and cfg.dataset.embedding_file is None:
        raise ValueError("Either pretrain.framework and pretrain.timestamp or dataset.embedding_file must be specified.")
    
    # pretrain.frameworkとpretrain.timestampの両方が指定されている時
    if cfg.pretrain.framework is not None and cfg.pretrain.timestamp is not None:
        # 両方ともlistで同じ長さでなければエラー
        assert isinstance(cfg.pretrain.framework, list) and isinstance(cfg.pretrain.timestamp, list), f"pretrain.framework and pretrain.timestamp must both be lists if one of them is a list. type(pretrain.framework): {type(cfg.pretrain.framework)}, type(pretrain.timestamp): {type(cfg.pretrain.timestamp)}"
        if len(cfg.pretrain.framework) != len(cfg.pretrain.timestamp):
            raise ValueError("Length of pretrain.framework and pretrain.timestamp lists must be the same.")

    
def setup_test_config():
    """
    OmegaconfのカスタムリゾルバとHydraのConfigStoreへの設定登録を行う関数
    """
    if not OmegaConf._get_resolver("div"):
        OmegaConf.register_new_resolver("div", lambda x, y: int(x / y))

    if not OmegaConf._get_resolver("mul"):
        OmegaConf.register_new_resolver("mul", lambda x, y: x * y)

    if not hasattr(setup_test_config, "is_registered"):
        cs = ConfigStore.instance()

        cs.store(name="test_config_schema", node=TestMainConfig)
        
        setup_test_config.is_registered = True
