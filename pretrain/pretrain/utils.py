import torch
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from .conf.config import (AdamWConfig, CosineSchedulerConfig, MainConfig,
                          MLMConfig, SwitchConfig, data2vecConfig)


def masking(
    token_seq: torch.Tensor,
    mask_idx: int,
    rna_tokens: list[str] = ["A", "C", "G", "U", "N"],
    sptoken_prob: float = 0.15,
    mask_prob: float = 0.8,
    use_additional_token: bool = False
) -> tuple[torch.Tensor, list[int]]:
    """
    入力をマスクする関数．デフォルトなら以下の通り．
    入力配列のうち15%の確率でマスク対象(special token), そのうち80%を"<mask>"トークン, 10%をランダムな他トークン, 残り10%はそのまま
    
    Args:
        token_seq (torch.Tensor): 元のトークン配列 (1次元テンソル)
        mask_idx (int): "<mask>"トークンのインデックス
        rna_tokens (list[str]): RNAトークンのリスト (default=["A", "C", "G", "U", "N"])
        sptoken_prob (float): マスク対象とする確率 (default=0.15)
        mask_prob (float): マスク対象のうち"<mask>"トークンに置換する確率 (default=0.8)
        use_additional_token (bool): CLS, EOSトークンを使用するかどうか (default=False)
    Returns:
        tuple[torch.Tensor, list[int]]: マスクされたトークン配列とマスク対象のインデックスリスト
    """
    
    masked_token_seq = token_seq.clone().detach()
    
    # 各位置ごとに確率 sptoken_prob で special token 対象にする（ベルヌーイ試行）
    probability_matrix = torch.full(
        masked_token_seq.shape,
        sptoken_prob,
        device=masked_token_seq.device,
    )
    # CLS, EOSトークンをマスク対象から除外する
    if use_additional_token:
        probability_matrix[0] = 0.0  # CLSトークン
        probability_matrix[-1] = 0.0  # EOSトークン

    sptoken_idxes = torch.bernoulli(probability_matrix).bool() # ベルヌーイ試行（確率判定）
    sptoken_idxes = sptoken_idxes.nonzero(as_tuple=True)[0].tolist()
    
    # マスク種類の決定
    probs = torch.rand(len(sptoken_idxes), device=masked_token_seq.device).tolist()
    rna_idxes = torch.arange(len(rna_tokens), dtype=masked_token_seq.dtype, device=masked_token_seq.device)
    
    for idx, prob in zip(sptoken_idxes, probs):
        if prob < mask_prob:
            masked_token_seq[idx] = mask_idx
        elif prob > 0.5 + mask_prob / 2.0:
            # 元のトークン以外のトークンを一つ選ぶ
            other_idxes = rna_idxes[rna_idxes != masked_token_seq[idx]]
            rand_idx = torch.randint(0, len(other_idxes), (1,)).item()
            masked_token_seq[idx] = other_idxes[rand_idx]
            
    return masked_token_seq, sptoken_idxes

def span_masking(
    token_seq: torch.Tensor,
    mask_idx: int,
    rna_tokens: list[str] = ["A", "C", "G", "U", "N"],
    sptoken_prob: float = 0.15,
    mask_prob: float = 0.8,
    min_span_length: int = 4,
    max_span_length: int = 8,
    use_additional_token: bool = False
) -> tuple[torch.Tensor, list[int]]:
    """
    入力をk-mer単位でマスクする関数。デフォルトなら以下の通り。
    入力配列のうち15%の割合をマスク対象(special token), そのうち80%を"<mask>"トークン, 10%をランダムな他トークン, 残り10%はそのまま

    Args:
        token_seq (torch.Tensor): 元のトークン配列 (1次元テンソル)
        mask_idx (int): "<mask>"トークンのインデックス
        rna_tokens (list[str]): RNAトークンのリスト (default=["A", "C", "G", "U", "N"])
        sptoken_prob (float): マスク対象とする確率 (default=0.15)
        mask_prob (float): マスク対象のうち"<mask>"トークンに置換する確率 (default=0.8)
        min_span_length (int): マスクする際の最小スパン長 (default=4)
        max_span_length (int): マスクする際の最大スパン長 (default=8)
        use_additional_token (bool): CLS, EOSトークンを使用するかどうか (default=False)
    Returns:
        tuple[torch.Tensor, list[int]]: マスクされたトークン配列とマスク対象のインデックスリスト
    """

    masked_token_seq = token_seq.clone().detach()

    # 各位置ごとに確率 sptoken_prob で special token 対象にする（ベルヌーイ試行）
    probability_matrix = torch.full(
        masked_token_seq.shape,
        sptoken_prob * 2 / (max_span_length + min_span_length), # スパン長を考慮してスパン長の平均で割って確率を調整
        device=masked_token_seq.device,
    )

    # spanの最小長を考慮して、最後のmin_span_length個のトークンはマスク対象から除外する
    probability_matrix[-min_span_length:] = 0.0

    # CLS, EOSトークンをマスク対象から除外する
    if use_additional_token:
        probability_matrix[0] = 0.0  # CLSトークン
        probability_matrix[-(min_span_length+1)] = 0.0  # spanの分は除外されているので追加でEOSトークン分

    sptoken_idxes = torch.bernoulli(probability_matrix).bool() # ベルヌーイ試行（確率判定）
    sptoken_idxes = sptoken_idxes.nonzero(as_tuple=True)[0].tolist()

    # マスク種類の決定
    probs = torch.rand(len(sptoken_idxes)).tolist()
    rna_idxes = torch.arange(len(rna_tokens), dtype=masked_token_seq.dtype, device=masked_token_seq.device)

    # スパンの決定
    span_lengths = torch.randint(min_span_length, max_span_length + 1, (len(sptoken_idxes),)).tolist()

    for idx, prob in zip(sptoken_idxes, probs):
        span_length = span_lengths[sptoken_idxes.index(idx)]

        # スパンが配列の範囲を超えないように調整
        # probability_matrixの設定で最後のmin_span_length個はマスク対象から除外されているので、調整後のspan_lengthがmin_span_length未満になることはない
        if idx + span_length >= len(masked_token_seq):
            span_length = len(masked_token_seq) - idx
            # EOSトークンを考慮
            if use_additional_token:
                span_length -= 1
            span_lengths[sptoken_idxes.index(idx)] = span_length

        if prob < mask_prob:
            masked_token_seq[idx : idx + span_length] = mask_idx
        elif prob > 0.5 + mask_prob / 2.0:
            # 元のトークン以外のトークンを一つ選ぶ
            other_idxes = torch.tensor([rna_idxes[rna_idxes != token_seq[i]].tolist() for i in range(idx, idx + span_length)], device=masked_token_seq.device)   # shape: (span_length, num_tokens-1)
            rand_idx = torch.randint(0, other_idxes.shape[1], (span_length,), device=masked_token_seq.device)
            masked_token_seq[idx : idx + span_length] = other_idxes[torch.arange(span_length), rand_idx]

    # スパンを考慮したsptoken_idxesの更新
    updated_sptoken_idxes = []
    for idx, sptoken_idx in enumerate(sptoken_idxes):
        updated_sptoken_idxes.extend(range(sptoken_idx, sptoken_idx + span_lengths[idx]))
    # 重複を削除
    updated_sptoken_idxes = list(set(updated_sptoken_idxes))

    return masked_token_seq, updated_sptoken_idxes

def create_attention_bias(
    token_seq: torch.Tensor,
    token_seq_masked: torch.Tensor = None,
    use_ernie_rna: bool = False,
    ernie_rna_alpha: float = 0.8,
    tokens: list[str] = ["A", "C", "G", "U", "N", "<mask>", "<pad>", "<cls>", "<eos>"]
) -> tuple[torch.Tensor, torch.Tensor] | tuple[None, None]:
    """
    Attentionマスクを作成する関数. use_ernie_rnaがTrueの場合, ernie_rna_alphaを使用する.
    
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
        
        cs.store(group="framework", name="mlm_schema", node=MLMConfig)
        cs.store(group="framework", name="data2vec_schema", node=data2vecConfig)
        cs.store(group="framework", name="switch_schema", node=SwitchConfig)
        
        cs.store(group="optimizer", name="adamw_schema", node=AdamWConfig)
        
        cs.store(group="lr_scheduler", name="cosine_schema", node=CosineSchedulerConfig)

        setup_config.is_registered = True
        
def validate_config(cfg: MainConfig):
    """
    設定の妥当性を確認する関数
    Args:
        cfg (MainConfig): 検証する設定オブジェクト
    """
    
    if not (cfg.experiment.extract_repr_layers >= 0 and cfg.experiment.extract_repr_layers <= cfg.model_size.n_layers):
        raise ValueError("extract_repr_layers must be between 0 and n_layers.")

    if cfg.framework.name == "data2vec":
        if not (cfg.model_size.k_layers >= 1 and cfg.model_size.k_layers <= cfg.model_size.n_layers):
            raise ValueError("k_layers must be between 1 and n_layers for data2vec framework.")
