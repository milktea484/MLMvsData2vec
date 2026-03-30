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

def bp2matrix(L, base_pairs):
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

def get_embedding_dim(loader: torch.utils.data.DataLoader, use_attention: bool) -> int:
    """
    ローダーから埋め込み次元数を取得する関数. embedding_fileが指定されている場合のみ使用
    Args:
        loader: データローダー
        use_attention: attentionかどうか
    Returns:
        int: 埋め込みの次元数
    """
    batch_elem = next(iter(loader)) # (B, L, E) or (B, E, L, L)
    
    return batch_elem["embeddings"].shape[1] if use_attention else batch_elem["embeddings"].shape[-1]

def outer_concat(t1: torch.Tensor, t2: torch.Tensor):
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

def symmetrize(x: torch.Tensor, zero_diagonal: bool = True):
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

def apc(x: torch.Tensor):
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
    
    if (cfg.pretrain.framework is None or cfg.pretrain.timestamp is None) and cfg.dataset.embedding_file is None:
        raise ValueError("Either pretrain.framework and pretrain.timestamp or dataset.embedding_file must be specified.")
    
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