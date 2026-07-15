import random
from pathlib import Path
import json

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from pretrain.conf.config import MainConfig as PretrainMainConfig
from utils import create_attention_bias, seq2token, bp2matrix

            
class REDIALlikeDataset(Dataset):
    """
    REDIALlike用データセットクラス
    Args:
        dataset_path (Path): データセットのパス (.csvファイル)
        tokens (list[str]): トークンのリスト
        other_tokens (list[str]): その他トークンのリスト
        use_additional_token (bool): CLS, EOSトークンを使用するかどうか
        use_ernie_rna (bool): ERNIE-RNAの戦略を使用するかどうか
        ernie_rna_alpha (float): ERNIE-RNAのalpha値
    """
    
    def __init__(
        self,
        dataset_path: Path,
        tokens: list[str] = ["A", "C", "G", "U", "N", "<mask>", "<pad>", "<cls>", "<eos>"],
        other_tokens: list[str] = ["B", "D", "F", "I", "H", "K", "M", "S", "R", "W", "V", "Y", "X"],
        use_additional_token: bool = False,
        use_ernie_rna: bool = False,
        ernie_rna_alpha: float = 0.8,
    ):
        if dataset_path.suffix != ".csv":
            raise ValueError("TestDataset only supports .csv files.")

        data = pd.read_csv(dataset_path)
        sequences = data["sequence"].tolist()
        self.base_pairs = [json.loads(data.base_pairs.iloc[i]) for i in range(len(data))]
        
        self.seq_ids = data["id"].tolist()
        self.token_seqs = seq2token(
            sequences,
            tokens=tokens,
            other_tokens=other_tokens,
            use_additional_token=use_additional_token,
        )

        self.attn_biases = None
        if use_ernie_rna:
            self.attn_biases = []
            for token_seq in self.token_seqs:
                attn_bias, _ = create_attention_bias(
                    token_seq,
                    token_seq_masked=None,
                    use_ernie_rna=use_ernie_rna,
                    ernie_rna_alpha=ernie_rna_alpha,
                    tokens=tokens,
                )
                self.attn_biases.append(attn_bias)

        self.tokens = tokens
        
    def __len__(self):
        return len(self.seq_ids)
    
    def __getitem__(self, idx: int):
        # 正解二次構造の取得
        length = len(self.token_seqs[idx])
        bp_matrix = bp2matrix(L=length, base_pairs=self.base_pairs[idx])

        return {
            "seq_id": self.seq_ids[idx],
            "token_seq": self.token_seqs[idx],
            "attn_bias": self.attn_biases[idx] if self.attn_biases is not None else None,
            "attn_biases_masked": self.attn_biases[idx] if self.attn_biases is not None else None,
            "bp_matrix": bp_matrix,
            "length": length,
        }
        
    def pad_batch(self, batch: list[dict]) -> dict:
        seq_ids = [b["seq_id"] for b in batch]
        token_seqs = [b["token_seq"] for b in batch]
        attn_biases = [b["attn_bias"] for b in batch] if self.attn_biases is not None else None
        attn_biases_masked = [b["attn_biases_masked"] for b in batch] if self.attn_biases is not None else None
        bp_matrices = [b["bp_matrix"] for b in batch]
        lengths = [b["length"] for b in batch]

        # バディング用にサイズを取得
        batch_size = len(batch)
        max_length = max(lengths)
        
        # バッチ用のテンソルを初期化
        token_seqs_padded = torch.full((batch_size, max_length), fill_value=self.tokens.index("<pad>"), dtype=torch.long)

        # attentionマスクの初期化
        attn_mask = torch.full((batch_size, 1, max_length, max_length), fill_value=-1e6)
        
        # attentionバイアスの初期化
        if self.attn_biases is not None:
            attn_biases_padded = torch.zeros((batch_size, 1, max_length, max_length))
            attn_biases_masked_padded = torch.zeros((batch_size, 1, max_length, max_length))
        
        # パディング
        for k in range(batch_size):
            token_seqs_padded[k, :lengths[k]] = token_seqs[k]
            attn_mask[k, :, :lengths[k], :lengths[k]] = 0
            if self.attn_biases is not None:
                attn_biases_padded[k, :, :lengths[k], :lengths[k]] = attn_biases[k]
                attn_biases_masked_padded[k, :, :lengths[k], :lengths[k]] = attn_biases_masked[k]
            
        return {
            "seq_ids": seq_ids,
            "token_seqs": token_seqs_padded,
            "attn_mask": attn_mask,
            "attn_biases": attn_biases_padded if self.attn_biases is not None else None,
            "attn_biases_masked": attn_biases_masked_padded if self.attn_biases is not None else None,
            "bp_matrices": torch.stack(bp_matrices, dim=0),  # (B, L, L)
            "lengths": lengths,
        }
    
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def create_dataloader(config: PretrainMainConfig, ) -> torch.utils.data.DataLoader:
    """
    データローダーの作成関数
    Args:
        config (PretrainMainConfig): 設定情報
    
    Returns:
        torch.utils.data.DataLoader: データローダー
    """
    
    # データセットの選択
    dataset = REDIALlikeDataset(
        dataset_path=Path(config.path.test_data_dir) / config.dataset.test_file,
        tokens=config.dataset.tokens,
        other_tokens=config.dataset.other_tokens,
        use_additional_token=config.experiment.use_additional_token,
        use_ernie_rna=config.experiment.use_ernie_rna,
        ernie_rna_alpha=config.framework.ernie_rna_alpha,
    )
    gradient_accumulation_steps = config.model_size.gradient_accumulation_steps_for_test  # テスト時は勾配を蓄積しない
    
    g = torch.Generator()
    g.manual_seed(config.common.seed)
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.common.batch_size // gradient_accumulation_steps,
        worker_init_fn=seed_worker,
        generator=g,
        shuffle=False,
        num_workers=config.common.num_workers,
        pin_memory=True if config.common.use_gpu else False,
        collate_fn=dataset.pad_batch,
    )
    
    return dataloader
