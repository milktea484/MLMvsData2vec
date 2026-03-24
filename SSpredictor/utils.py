import torch
from conf.config import AdamWConfig, KnotFoldConfig, MainConfig
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
    matrix = torch.zeros((L, L))
    if base_pairs != []:
        # base pairs are 1-based
        bp = torch.tensor(base_pairs) - 1
        matrix[bp[:, 0], bp[:, 1]] = 1
        matrix[bp[:, 1], bp[:, 0]] = 1

    return matrix

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
