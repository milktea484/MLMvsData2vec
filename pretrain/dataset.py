import random
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from conf.config import MainConfig
from torch.utils.data import Dataset
from utils import create_attention_bias, masking, seq2token


class TrainingDataset(Dataset):
    """
    訓練用 (or 検証用) データセットクラス
    Args:
        dataset_path (Path): データセットのパス (.h5ファイル)
        tokens (list[str]): トークンのリスト
        rna_tokens (list[str]): RNAトークンのリスト
        sptoken_prob (float): マスク対象とする確率
        mask_prob (float): マスク対象のうち"<mask>"トークンに置換する確率
        use_additional_token (bool): CLS, EOSトークンを使用するかどうか
        use_ernie_rna (bool): ERNIE-RNAの戦略を使用するかどうか
        ernie_rna_alpha (float): ERNIE-RNAのalpha値
    """
    
    def __init__(
        self,
        dataset_path: Path,
        tokens: list[str] = ["A", "C", "G", "U", "N", "<mask>", "<pad>", "<cls>", "<eos>"],
        rna_tokens: list[str] = ["A", "C", "G", "U", "N"],
        sptoken_prob: float = 0.15,
        mask_prob: float = 0.8,
        use_additional_token: bool = False,
        use_ernie_rna: bool = False,
        ernie_rna_alpha: float = 0.8,
    ):
        assert dataset_path.suffix == ".h5", "TrainingDataset only supports .h5 files."
        self.cache = dataset_path
        self.tokens = tokens
        self.rna_tokens = rna_tokens
        self.sptoken_prob = sptoken_prob
        self.mask_prob = mask_prob
        self.use_additional_token = use_additional_token
        self.use_ernie_rna = use_ernie_rna
        self.ernie_rna_alpha = ernie_rna_alpha
        self.cls_token = tokens.index("<cls>")
        self.eos_token = tokens.index("<eos>")
        
        with h5py.File(self.cache, "r") as hdf:
            self.dataset_length = len(hdf["/seqs"])
        
        self._hdf = None
        
    def _get_hdf(self):
        if self._hdf is None:
            self._hdf = h5py.File(self.cache, "r")
        return self._hdf
        
    def __len__(self):
        return self.dataset_length
        
    def __getitem__(self, idx: int):
        hdf = self._get_hdf()
        token_seq = hdf["/seqs"][idx].decode()
        
        # token_seqは文字列なのでintに変換後，torch.tensorに変換
        token_seq = [int(t) for t in list(token_seq)]
        if self.use_additional_token:
            token_seq = [self.cls_token] + token_seq + [self.eos_token]
        token_seq = torch.tensor(token_seq, dtype=torch.uint8)
        
        # マスクされたトークン配列の作成
        token_seq_masked, masked_idxes = masking(
            token_seq,
            mask_idx=self.tokens.index("<mask>"),
            rna_tokens=self.rna_tokens,
            sptoken_prob=self.sptoken_prob,
            mask_prob=self.mask_prob
        )
        
        # attentionバイアスの作成
        attn_bias, attn_bias_masked = create_attention_bias(
            token_seq,
            token_seq_masked,
            use_ernie_rna=self.use_ernie_rna,
            ernie_rna_alpha=self.ernie_rna_alpha,
            tokens=self.tokens,
        )
        
        return {
            "token_seq": token_seq,
            "token_seq_masked": token_seq_masked,
            "attn_bias": attn_bias,
            "attn_bias_masked": attn_bias_masked,
            "masked_idxes": masked_idxes,
            "length": len(token_seq),
        }
        
    def pad_batch(self, batch: list[dict]) -> dict:
        token_seqs = [b["token_seq"] for b in batch]
        token_seqs_masked = [b["token_seq_masked"] for b in batch]
        attn_biases = [b["attn_bias"] for b in batch]
        attn_biases_masked = [b["attn_bias_masked"] for b in batch]
        masked_idxes = [b["masked_idxes"] for b in batch]
        lengths = [b["length"] for b in batch]
        
        # バディング用にサイズを取得
        batch_size = len(token_seqs)
        max_length = max(lengths)
        
        # バッチ用のテンソルを初期化
        token_seqs_padded = torch.full((batch_size, max_length), fill_value=self.tokens.index("<pad>"), dtype=torch.uint8)
        token_seqs_masked_padded = torch.full((batch_size, max_length), fill_value=self.tokens.index("<pad>"), dtype=torch.uint8)
        
        # attentionマスクの初期化
        attn_mask = torch.full((batch_size, 1, max_length, max_length), fill_value=-1e6, dtype=torch.bfloat16)
        
        # attentionバイアスの初期化
        if self.use_ernie_rna:
            attn_biases_padded = torch.zeros((batch_size, 1, max_length, max_length), dtype=torch.bfloat16)
            attn_biases_masked_padded = torch.zeros((batch_size, 1, max_length, max_length), dtype=torch.bfloat16)
        else:
            attn_biases_padded = None
            attn_biases_masked_padded = None
        
        # パディング
        # attentionバイアスは, ERNIE-RNAを使わない場合Noneになる可能性があるので注意
        for k in range(batch_size):
            token_seqs_padded[k, :lengths[k]] = token_seqs[k]
            token_seqs_masked_padded[k, :lengths[k]] = token_seqs_masked[k]
            attn_mask[k, :, :lengths[k], :lengths[k]] = 0
            if self.use_ernie_rna:
                attn_biases_padded[k, :, :lengths[k], :lengths[k]] = attn_biases[k]
                attn_biases_masked_padded[k, :, :lengths[k], :lengths[k]] = attn_biases_masked[k]
            
        return {
            "token_seqs": token_seqs_padded,
            "token_seqs_masked": token_seqs_masked_padded,
            "attn_mask": attn_mask,
            "attn_biases": attn_biases_padded,
            "attn_biases_masked": attn_biases_masked_padded,
            "masked_idxes": masked_idxes,
        }
            
class TestDataset(Dataset):
    """
    テスト用データセットクラス
    Args:
        dataset_path (Path): データセットのパス (.csvファイル)
        tokens (list[str]): トークンのリスト
        other_tokens (list[str]): その他トークンのリスト
        use_additional_token (bool): CLS, EOSトークンを使用するかどうか
    """
    
    def __init__(
        self,
        dataset_path: Path,
        tokens: list[str] = ["A", "C", "G", "U", "N", "<mask>", "<pad>", "<cls>", "<eos>"],
        other_tokens: list[str] = ["B", "D", "F", "I", "H", "K", "M", "S", "R", "W", "V", "Y", "X"],
        use_additional_token: bool = False,
    ):
        assert dataset_path.suffix == ".csv", "TestDataset only supports .csv files."
        data = pd.read_csv(dataset_path)
        sequences = data["sequence"].tolist()
        
        self.seq_ids = data["id"].tolist()
        self.token_seqs = seq2token(
            sequences,
            tokens=tokens,
            other_tokens=other_tokens,
            use_additional_token=use_additional_token,
        )
        self.tokens = tokens
        
    def __len__(self):
        return len(self.seq_ids)
    
    def __getitem__(self, idx: int):
        return {
            "seq_id": self.seq_ids[idx],
            "token_seq": self.token_seqs[idx],
            "length": len(self.token_seqs[idx]),
        }
        
    def pad_batch(self, batch: list[dict]) -> dict:
        seq_ids = [b["seq_id"] for b in batch]
        token_seqs = [b["token_seq"] for b in batch]
        lengths = [b["length"] for b in batch]
        
        # バディング用にサイズを取得
        batch_size = len(token_seqs)
        max_length = max(lengths)
        
        # バッチ用のテンソルを初期化
        token_seqs_padded = torch.full((batch_size, max_length), fill_value=self.tokens.index("<pad>"), dtype=torch.uint8)
        
        # attentionマスクの初期化
        attn_mask = torch.full((batch_size, 1, max_length, max_length), fill_value=-1e6, dtype=torch.bfloat16)
        
        # パディング
        for k in range(batch_size):
            token_seqs_padded[k, :lengths[k]] = token_seqs[k]
            attn_mask[k, :, :lengths[k], :lengths[k]] = 0
            
        return {
            "seq_ids": seq_ids,
            "token_seqs": token_seqs_padded,
            "attn_mask": attn_mask,
            "lengths": lengths,
        }
    
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def create_dataloader(config: MainConfig, split: str):
    """
    データローダーの作成関数
    Args:
        config (MainConfig): 設定情報
        split (str): データ分割 ("train", "validation", or "test")
    
    Returns:
        torch.utils.data.DataLoader: データローダー
    """
    
    # データセットの選択
    if split == "train":
        dataset = TrainingDataset(
            dataset_path=Path(config.path.data_dir) / config.dataset.train_file,
            tokens=config.dataset.tokens,
            rna_tokens=config.dataset.rna_tokens,
            sptoken_prob=config.framework.sptoken_prob,
            mask_prob=config.framework.mask_prob,
            use_additional_token=config.experiment.use_additional_token,
            use_ernie_rna=config.experiment.use_ernie_rna,
            ernie_rna_alpha=config.framework.ernie_rna_alpha,
        )
    elif split == "validation":
        dataset = TrainingDataset(
            dataset_path=Path(config.path.data_dir) / config.dataset.validation_file,
            tokens=config.dataset.tokens,
            rna_tokens=config.dataset.rna_tokens,
            sptoken_prob=config.framework.sptoken_prob,
            mask_prob=config.framework.mask_prob,
            use_additional_token=config.experiment.use_additional_token,
            use_ernie_rna=config.experiment.use_ernie_rna,
            ernie_rna_alpha=config.framework.ernie_rna_alpha,
        )
    elif split == "test":
        dataset = TestDataset(
            dataset_path=Path(config.path.data_dir) / config.dataset.test_file,
            tokens=config.dataset.tokens,
            other_tokens=config.dataset.other_tokens,
            use_additional_token=config.experiment.use_additional_token,
        )
    else:
        raise ValueError(f"Invalid split name: {split}")
    
    g = torch.Generator()
    g.manual_seed(config.common.seed)
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.common.batch_size // config.model_size.gradient_accumulation_steps,
        worker_init_fn=seed_worker,
        generator=g,
        shuffle=(split == "train"),
        num_workers=config.common.num_workers,
        pin_memory=True if config.common.use_gpu else False,
        collate_fn=dataset.pad_batch,
    )
    
    return dataloader


def create_batch_iterator(
    config: MainConfig,
    split: str,
):
    """
    バッチイテレータの作成関数
    Args:
        config (MainConfig): 設定情報
        split (str): データ分割 ("train", "validation", or "test")
    
    Returns:
        torch.utils.data.DataLoader: バッチイテレータ
    """
    
    loader = create_dataloader(config=config, split=split)
    
    while True:
        for batch in loader:
            batch["token_seqs"] = batch["token_seqs"].to(torch.long)
            if split != "test":
                batch["token_seqs_masked"] = batch["token_seqs_masked"].to(torch.long)
            yield batch
            
        if split != "train":
            break
    
    