import json
import random
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from conf.config import MainConfig
from pretrain.conf.config import MainConfig as PretrainMainConfig
from torch.utils.data import Dataset
from utils import bp2matrix, seq2token


class EmbeddingDataset(Dataset):
    """
    二次構造予測用データセットクラス
    Args:
        dataset_path (Path): データセットのパス (.csvファイル)
        tokens (list[str]): トークンのリスト
        other_tokens (list[str]): その他トークンのリスト
        use_additional_token (bool): CLS, EOSトークンを使用するかどうか
        use_attention (bool): 使用する特徴表現をattentionにするかどうか. falseなら配列特徴量になる
        embedding_path (Path): すでにh5形式で保存されている配列特徴量のファイルパス. 事前学習モデルの出力を使用する場合に必要
        reference_embedding_dim (int): 参照埋め込みの次元数. これが指定されている場合, データセットは参照埋め込みを返すようになる
        use_pretrain_model (bool): 事前学習モデルを使用して埋め込みを生成するかどうか. embedding_pathが指定されている場合にのみ, Falseに設定可能
    """
    
    def __init__(
        self,
        dataset_path: Path,
        tokens: list[str] = ["A", "C", "G", "U", "N", "<mask>", "<pad>", "<cls>", "<eos>"],
        other_tokens: list[str] = ["B", "D", "F", "I", "H", "K", "M", "S", "R", "W", "V", "Y", "X"],
        use_additional_token: bool = False,
        use_attention: bool = False,
        embedding_path: Path = None,
        reference_embedding_dim: int = None,
        use_pretrain_model: bool = True,
    ):
        assert use_pretrain_model or embedding_path is not None, "Either use_pretrain_model must be True or embedding_path must be specified."

        if dataset_path.suffix != ".csv":
            raise ValueError("EmbeddingDataset only supports .csv files.")
        if embedding_path is not None and embedding_path.suffix != ".h5":
            raise ValueError("embedding_path must be a .h5 file.")

        # データセットの読み込み
        data = pd.read_csv(dataset_path)
        self.sequences = data["sequence"].tolist()
        self.base_pairs = [json.loads(data.base_pairs.iloc[i]) for i in range(len(data))]
        self.seq_ids = data["id"].tolist()
        
        # トークン化されたシーケンスの生成 (事前学習モデルを使用する場合のみ)
        self.token_seqs = seq2token(
            sequences=self.sequences,
            tokens=tokens,
            other_tokens=other_tokens,
            use_additional_token=use_additional_token,
        ) if use_pretrain_model else None
        
        # 埋め込みのキャッシュファイルのパス
        self.cache_embedding_file = embedding_path
        
        # 参照埋め込みの次元数 (KnotFoldで必要な場合)
        self.reference_embedding_dim = reference_embedding_dim
        
        # その他
        self.tokens = tokens
        self.use_attention = use_attention
    
    def __len__(self):
        return len(self.seq_ids)
    
    def __getitem__(self, idx: int):
        
        seq_id = self.seq_ids[idx]
        
        if self.token_seqs is not None:
            # 事前学習モデルを使用するときのlength
            length = len(self.token_seqs[idx])  # CLS, EOSトークンを使用している場合は実際のシーケンス長より2大きくなっている
        else:
            length = -1  # lengthは埋め込みから取得する必要があるため, ダミーの値を設定
        
        # キャッシュされた埋め込みがある場合はそれを読み込む
        embedding = None
        if self.cache_embedding_file is not None:
            with h5py.File(self.cache_embedding_file, "r") as hdf:
                embedding = torch.from_numpy(hdf[seq_id][()])  # shape(L, E) or (E, L, L)
                
                # use_attentionの設定変更 
                if embedding.ndim == 3:
                    self.use_attention = True
                else:
                    self.use_attention = False
                
                # 埋め込みの次元数を取得
                if self.reference_embedding_dim == -1:
                    self.reference_embedding_dim = embedding.shape[0] if self.use_attention else embedding.shape[-1]
                
                # 配列長の取得
                if length == -1:
                    length = embedding.shape[-1] if self.use_attention else embedding.shape[0]
        
        # 塩基対行列の生成
        bp_matrix = bp2matrix(length, self.base_pairs[idx])
        
        # 参照埋め込みの生成 (KnotFoldで必要な場合)
        reference_embedding = None
        if self.reference_embedding_dim is not None:
            # 参照埋め込みはシーケンス長に依存する必要があるため, データセット内で乱数生成器を初期化してシーケンスごとに固定の乱数を生成
            g = torch.Generator()
            g.manual_seed(idx)
            
            if self.use_attention:
                reference_embedding = torch.randn((self.reference_embedding_dim, length, length), generator=g)
            else:
                reference_embedding = torch.randn((length, self.reference_embedding_dim), generator=g)
        
        return {
            "seq_id": seq_id,
            "sequence": self.sequences[idx],
            "token_seq": self.token_seqs[idx] if self.token_seqs is not None else None,
            "bp_matrix": bp_matrix,
            "length": length,
            "embedding": embedding,
            "reference_embedding": reference_embedding,
        }
        
        
    def pad_batch(self, batch: list[dict]) -> dict:
        seq_ids = [b["seq_id"] for b in batch]
        sequences = [b["sequence"] for b in batch]
        token_seqs = [b["token_seq"] for b in batch] if self.token_seqs is not None else None
        bp_matrices = [b["bp_matrix"] for b in batch]
        lengths = [b["length"] for b in batch]
        embeddings = [b["embedding"] for b in batch] if self.cache_embedding_file is not None else None
        reference_embeddings = [b["reference_embedding"] for b in batch] if self.reference_embedding_dim is not None else None
        
        # バディング用にサイズを取得
        batch_size = len(batch)
        max_length = max(lengths)
        
        # バッチ用のテンソルを初期化
        token_seqs_padded = torch.full((batch_size, max_length), fill_value=self.tokens.index("<pad>"), dtype=torch.long)
        
        # attentionマスクの初期化
        attn_mask = torch.full((batch_size, 1, max_length, max_length), fill_value=-1e6)
        
        # 正解二次構造matrixを-1で初期化
        bp_matrices_padded = -torch.ones((batch_size, max_length, max_length), dtype=torch.int8)
        
        # embeddingの初期化
        if self.cache_embedding_file is not None:
            # reference_embedding_dimと同じになるが, referenceがない場合も考えてembedding_dimを取得
            embedding_dim = embeddings[0].shape[0] if self.use_attention else embeddings[0].shape[-1]
            if self.use_attention:
                embeddings_padded = torch.zeros((batch_size, embedding_dim, max_length, max_length))
            else:
                embeddings_padded = torch.zeros((batch_size, max_length, embedding_dim))

        # reference_embeddingの初期化
        if self.reference_embedding_dim is not None:
            if self.use_attention:
                reference_embeddings_padded = torch.zeros((batch_size, self.reference_embedding_dim, max_length, max_length))
            else:
                reference_embeddings_padded = torch.zeros((batch_size, max_length, self.reference_embedding_dim))
        
        # パディング
        for k in range(batch_size):
            token_seqs_padded[k, :lengths[k]] = token_seqs[k]
            attn_mask[k, :, :lengths[k], :lengths[k]] = 0
            bp_matrices_padded[k, :lengths[k], :lengths[k]] = bp_matrices[k]
            
            if self.cache_embedding_file is not None:
                if self.use_attention:
                    embeddings_padded[k, :, :lengths[k], :lengths[k]] = embeddings[k] # (B, E, L, L)
                else:
                    embeddings_padded[k, :lengths[k], :] = embeddings[k] # (B, L, E)
            
            if self.reference_embedding_dim is not None:
                if self.use_attention:
                    reference_embeddings_padded[k, :, :lengths[k], :lengths[k]] = reference_embeddings[k] # (B, E, L, L)
                else:
                    reference_embeddings_padded[k, :lengths[k], :] = reference_embeddings[k] # (B, L, E)
                    
        return {
            "seq_ids": seq_ids,
            "sequences": sequences,
            "token_seqs": token_seqs_padded,
            "attn_mask": attn_mask,
            "bp_matrices": bp_matrices_padded,
            "lengths": lengths,
            "embeddings": embeddings_padded,
            "reference_embeddings": reference_embeddings_padded,
        }
        
        
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    

def create_dataloader(config: MainConfig, split: str, pretrain_config: PretrainMainConfig=None) -> torch.utils.data.DataLoader:
    """
    データローダーの作成関数
    Args:
        config (MainConfig): 二次構造予測モデルの設定情報
        split (str): データ分割 ("train", "validation", "test", or "reference")
        pretrain_config (PretrainMainConfig): 事前学習モデルの設定情報
    
    Returns:
        torch.utils.data.DataLoader: データローダー
    """
    
    # データセットの選択
    assert split in ["train", "validation", "test", "reference"], "split must be 'train', 'validation', 'test', or 'reference'"

    if split == "train" or split == "reference":
        dataset_file = config.dataset.train_file
    elif split == "validation":
        dataset_file = config.dataset.validation_file
    else:
        dataset_file = config.dataset.test_file
    
    # ArchiveII_famfoldなどではadditional_experiment_infoを使用してdatasetのパスを指定
    if config.experiment.additional_experiment_info is not None:
        dataset_path = Path(config.path.data_dir) / config.experiment.name.lower() / config.experiment.additional_experiment_info / dataset_file
    else:
        dataset_path = Path(config.path.data_dir) / config.experiment.name.lower() / dataset_file
    
    # reference配列用に次元を指定
    reference_embedding_dim = None
    if split == "reference":
        # embedding_fileが指定されている場合, reference次元はembeddingの次元に合わせる必要があるため, reference_embedding_dimは-1に設定
        if config.dataset.embedding_file is not None:
            reference_embedding_dim = -1 
        else:
            if config.experiment.use_attention:
                reference_embedding_dim = pretrain_config.framework.n_layers * pretrain_config.framework.n_heads
            else:
                reference_embedding_dim = pretrain_config.model_size.embed_dim
                
            assert reference_embedding_dim > 0, "reference_embedding_dim must be positive unless embedding_file is specified"

    # embedding_fileが指定されている場合はそちらを使用
    if config.dataset.embedding_file is not None:
        embedding_path = Path(config.path.embedding_dir) / config.dataset.embedding_file
    else:
        embedding_path = None

    if pretrain_config is not None:
        dataset = EmbeddingDataset(
            dataset_path=dataset_path,
            tokens=pretrain_config.dataset.tokens,
            other_tokens=pretrain_config.dataset.other_tokens,
            use_additional_token=pretrain_config.experiment.use_additional_token,
            use_attention=config.experiment.use_attention,
            embedding_path=embedding_path,
            reference_embedding_dim=reference_embedding_dim,
        )
    else:
        # pretrain_configが指定されていない場合, 事前学習モデルを使用しないことを明示的に指定
        dataset = EmbeddingDataset(
            dataset_path=dataset_path,
            use_attention=config.experiment.use_attention,
            embedding_path=embedding_path,
            reference_embedding_dim=reference_embedding_dim,
            use_pretrain_model=False,
        )
    
    g = torch.Generator()
    g.manual_seed(config.common.seed)
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.common.batch_size,
        shuffle=(split=="train"),
        collate_fn=dataset.pad_batch,
        worker_init_fn=seed_worker,
        generator=g,
    )
    
    return dataloader

def create_batch_iterator(config: MainConfig, pretrain_config: PretrainMainConfig, split: str):
    """
    バッチイテレータの作成関数
    Args:
        config (MainConfig): 二次構造予測モデルの設定情報
        pretrain_config (PretrainMainConfig): 事前学習モデルの設定情報
        split (str): データ分割 ("train", "validation", "test", or "reference")
    
    Returns:
        Iterator: バッチイテレータ. batchとepochを返す
    """
    
    assert split in ["train", "validation", "test", "reference"], "split must be 'train', 'validation', 'test', or 'reference'"
    
    loader = create_dataloader(config=config, pretrain_config=pretrain_config, split=split)
    
    epoch = 0
    while True:
        for batch in loader:
            batch["token_seqs"] = batch["token_seqs"].to(torch.long)
            yield batch, epoch
        epoch += 1
            
        if split == "test":
            break
    