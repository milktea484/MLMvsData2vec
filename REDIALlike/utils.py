from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from conf.config import MainConfig
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
        token_seq = [mapping.get(nt) for nt in seq.upper()]
        if use_additional_token:
            token_seq = [mapping["<cls>"]] + token_seq + [mapping["<eos>"]]
        
        if any(v is None for v in token_seq):
            raise ValueError("Invalid nucleotide found")
        token_seqs.append(torch.tensor(token_seq, dtype=torch.uint8))
        
    return token_seqs

def create_attention_bias(
    token_seq: torch.Tensor,
    token_seq_masked: torch.Tensor = None,
    use_ernie_rna: bool = False,
    ernie_rna_alpha: float = 0.8,
    tokens: list[str] = ["A", "C", "G", "U", "N", "<mask>", "<pad>", "<cls>", "<eos>"]
) -> tuple[torch.Tensor, torch.Tensor] | tuple[None, None]:
    """
    事前学習モデル用にAttentionマスクを作成する関数. use_ernie_rnaがTrueの場合, ernie_rna_alphaを使用する.
    
    Args:
        token_seq (torch.Tensor): 元のトークン配列 (1次元テンソル)
        token_seq_masked (torch.Tensor): マスクされたトークン配列 (1次元テンソル)
        use_ernie_rna (bool): ERNIE-RNAの戦略を使用するかどうか (default=False)
        ernie_rna_alpha (float): ERNIE-RNAのalpha値 (default=0.8)
        tokens (list[str]): トークンのリスト
    Returns:
        tuple[torch.Tensor, torch.Tensor] | tuple[None, None]: attentionバイアスとマスクされた配列のattentionバイアス  
        ernie_rnaを使わない場合, (None, None)を返す
    """
    if not use_ernie_rna:
        return None, None
    
    L = token_seq.shape[0]
    seq_row = token_seq.view(L, 1)  # (L, 1)
    seq_col = token_seq.view(1, L)  # (1, L)
    
    attention_bias = torch.zeros((L, L), device=token_seq.device)
    
    A, C, G, U, MASK = [
        tokens.index(x) for x in ["A", "C", "G", "U", "<mask>"]
    ]

    attention_bias += ((seq_row == A) & (seq_col == U)) * 2.0
    attention_bias += ((seq_row == U) & (seq_col == A)) * 2.0
    attention_bias += ((seq_row == C) & (seq_col == G)) * 3.0
    attention_bias += ((seq_row == G) & (seq_col == C)) * 3.0
    attention_bias += ((seq_row == G) & (seq_col == U)) * ernie_rna_alpha
    attention_bias += ((seq_row == U) & (seq_col == G)) * ernie_rna_alpha

    # <mask> の行・列を-1.0に設定
    attention_bias_masked = attention_bias.clone().detach()
    if token_seq_masked is not None:
        mask_positions = (token_seq_masked == MASK).nonzero(as_tuple=True)[0]
        if mask_positions.numel() > 0:
            attention_bias_masked[mask_positions, :] = -1.0
            attention_bias_masked[:, mask_positions] = -1.0
    
    return attention_bias, attention_bias_masked

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

def assess_attention(attn, gt, n_heads=12, n_layers=12, designated_layer=None, single_map=False):
    """
    Assess the quality of attention maps against ground truth contact maps.
    Args:
        attn (np.Ndarray): Attention maps of shape (n_heads*n_layers, L, L).
        gt (np.Ndarray): Ground truth contact map of shape (L, L).
        designated_layer (int, optional): If specified, only assess the attention maps of this layer. Defaults to None.
        single_map (bool, optional): If True, assess the attention maps as a single map. Defaults to False.
    Returns:
        (float, int): the percentage of attention weights that indicate the pairing of structural tokens within each attention head of the architecture
            int: the index of the attention head that has the highest score
    """
    if single_map:
        # attn shape: (L, L)
        sum_attn = np.sum(attn)
        if sum_attn == 0:
            return 0.0, None
        score = np.sum(attn * gt) / sum_attn
        return score.astype(float), None

    else:
        max_score = 0.0
        max_hl = 0
        if designated_layer is not None:
            start_idx = (designated_layer - 1) * n_heads
            end_idx = designated_layer * n_heads

        iteration_range = range(n_heads * n_layers) if designated_layer is None else range(n_heads)
        for idx in iteration_range:
            hl = idx if designated_layer is None else idx + start_idx
            sum_attn_hl = np.sum(attn[hl])
            if sum_attn_hl == 0:
                continue
            score = np.sum(attn[hl] * gt) / sum_attn_hl
            if score > max_score:
                max_score = score.astype(float)
                max_hl = hl

        return max_score, max_hl

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
        
        setup_config.is_registered = True

def validate_config(cfg: MainConfig):
    """
    設定の妥当性を確認する関数
    Args:
        cfg (MainConfig): 検証する設定オブジェクト
    """
    pass
