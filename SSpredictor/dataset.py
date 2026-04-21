import json
import random
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from conf.config import MainConfig
from torch.utils.data import Dataset
from utils import bp2matrix, seq2token

from pretrain.conf.config import MainConfig as PretrainMainConfig


class EmbeddingDataset(Dataset):
    """
    二次構造予測用データセットクラス
    Args:
        dataset_path (Path): データセットのパス (.csvファイル)
        tokens (list[str]): トークンのリスト
        other_tokens (list[str]): その他トークンのリスト
        use_additional_token (bool): CLS, EOSトークンを使用するかどうか
        use_attention (bool): 使用する特徴表現をattentionにするかどうか. falseなら配列特徴量になる
        embedding_paths (list[Path]): すでにh5形式で保存されている配列特徴量のファイルパス. 事前学習モデルの出力を使用する場合に必要. 単体, 複数どちらの場合もlistとして渡される.
        reference_embedding_dim (int | None): 参照埋め込みの次元. これが指定されている場合, データセットは参照埋め込みを返すようになる.
        use_pretrain_model (bool): 事前学習モデルを使用して埋め込みを生成するかどうか. embedding_pathsが指定されている場合にのみ, Falseに設定可能
    """
    
    def __init__(
        self,
        dataset_path: Path,
        tokens: list[str] = ["A", "C", "G", "U", "N", "<mask>", "<pad>", "<cls>", "<eos>"],
        other_tokens: list[str] = ["B", "D", "F", "I", "H", "K", "M", "S", "R", "W", "V", "Y", "X"],
        use_additional_token: bool = False,
        use_attention: bool = False,
        embedding_paths: list[Path] | None = None,
        reference_embedding_dim: int | None = None,
        use_pretrain_model: bool = True,
    ):
        # 引数の検査
        assert use_pretrain_model or embedding_paths is not None, "Either use_pretrain_model must be True or embedding_paths must be specified."

        if dataset_path.suffix != ".csv":
            raise ValueError("EmbeddingDataset only supports .csv files.")
        if embedding_paths is not None and not all(path.suffix == ".h5" for path in embedding_paths):
            raise ValueError("All paths in embedding_paths must be .h5 files.")

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
        
        # 埋め込みのキャッシュファイルのパスのリスト
        self.cache_embedding_paths = embedding_paths if embedding_paths is not None else []
        
        # 参照埋め込みの次元 (KnotFoldで必要な場合)
        self.reference_embedding_dim = reference_embedding_dim
        self.is_reference_embedding_dim_fixed = False # reference_embedding_dimを固定したかどうか
        
        # その他
        self.tokens = tokens
        self.use_additional_token = use_additional_token
        self.use_attention = use_attention
        self.use_pretrain_model = use_pretrain_model
        
        self._cache_embedding_hdf = [None] * len(self.cache_embedding_paths)  # 埋め込みのhdfファイルのハンドル. 最初はNoneで, 必要になったときに開く
    
    def _get_hdf(self, index: int):
        if self._cache_embedding_hdf[index] is None:
            self._cache_embedding_hdf[index] = h5py.File(self.cache_embedding_paths[index], "r")
        return self._cache_embedding_hdf[index]
    
    def __len__(self):
        return len(self.seq_ids)
    
    def __getitem__(self, idx: int):
        
        seq_id = self.seq_ids[idx]
        
        sequence_length = len(self.sequences[idx])

        # additional tokenを使用するときのtoken_seqの長さは, 実際のシーケンス長より2大きくなっていることに注意
        if self.token_seqs is not None and sequence_length != len(self.token_seqs[idx]):
            if sequence_length + 2 != len(self.token_seqs[idx]):
                raise ValueError("The length of token_seq should be equal to the length of sequence plus 2 if use_additional_token (cls, eos) is True.")

        # キャッシュされた埋め込みがある場合はそれを読み込む
        embedding_list = []
        embedding_ndims = [] # 埋め込みの次元数を保存するリスト. attentionか配列特徴量かの判定に使用
        for i, cache_path in enumerate(self.cache_embedding_paths):
            hdf = self._get_hdf(i)
            emb = torch.from_numpy(hdf[seq_id][()])  # shape(L, E) or (E, L, L)

            # use_attentionの確認. pretrain modelも使用する場合は, あらかじめuse_attentionを明記しなければならない.
            # また, 埋め込み表現が[CLS]や[EOS]などのadditional tokenを含む場合, それらを除く
            if emb.ndim == 3:
                if self.use_pretrain_model and not self.use_attention:
                    raise ValueError("The loaded embedding has 3 dimensions, which is not compatible with use_attention=False (cannot concatenate). Please set use_attention=True or provide 2-dimensional embeddings.")
                
                # 埋め込み表現のadditional tokenを確認し, あれば除く
                if sequence_length != emb.shape[1] or sequence_length != emb.shape[2]:
                    if sequence_length + 2 == emb.shape[1] and sequence_length + 2 == emb.shape[2]:
                        emb = emb[:, 1:-1, 1:-1] # (E, L, L)にする
                    else:
                        raise ValueError("The length of the loaded embedding does not match the length of the sequence. Please check the use of additional tokens and the consistency between the lengths of the loaded embeddings and the sequences.")
            else:
                if self.use_pretrain_model and self.use_attention:
                    raise ValueError("The loaded embedding has 2 dimensions, which is not compatible with use_attention=True (cannot concatenate). Please set use_attention=False or provide 3-dimensional embeddings.")

                # 埋め込み表現のadditional tokenを確認し, あれば除く
                if sequence_length != emb.shape[0]:
                    if sequence_length + 2 == emb.shape[0]:
                        emb = emb[1:-1, :] # (L, E)にする
                    else:
                        raise ValueError("The length of the loaded embedding does not match the length of the sequence. Please check the use of additional tokens and the consistency between the lengths of the loaded embeddings and the sequences.")

            embedding_list.append(emb)
            embedding_ndims.append(emb.ndim)

        # 埋め込みが複数指定されている場合は, それらを結合することを想定しているため, 埋め込みの次元数が一致しているか確認
        if len(set(embedding_ndims)) > 1:
            raise ValueError("When using multiple embedding files, the dimensions of the embeddings must be the same.")
        # use_attentionの設定が明記されていない場合は, 埋め込みの次元数から判断する
        elif embedding_ndims != []:
            self.use_attention = (embedding_ndims[0] == 3) # 埋め込みの次元数が3ならattention, 2なら配列特徴量

        # 埋め込みの結合
        embedding = None
        if embedding_list != []:
            if self.use_attention:
                embedding = torch.cat(embedding_list, dim=0)  # (E, L, L)同士を結合して ((embeddingファイル数)*E, L, L) にする
            else:
                embedding = torch.cat(embedding_list, dim=-1)  # (L, E)同士を結合して (L, (embeddingファイル数)*E) にする

        # referenceの次元の設定
        if not self.is_reference_embedding_dim_fixed and self.reference_embedding_dim is not None and embedding is not None:
            self.reference_embedding_dim += embedding.shape[0] if self.use_attention else embedding.shape[-1]
            self.is_reference_embedding_dim_fixed = True
        
        # 塩基対行列の生成
        bp_matrix = bp2matrix(sequence_length, self.base_pairs[idx])
        
        # 参照埋め込みの生成 (KnotFoldで必要な場合)
        reference_embedding = None
        if self.reference_embedding_dim is not None:
            # 参照埋め込みはシーケンス長に依存する必要があるため, データセット内で乱数生成器を初期化してシーケンスごとに固定の乱数を生成
            g = torch.Generator()
            g.manual_seed(idx)
            
            if self.use_attention:
                reference_embedding = torch.randn((self.reference_embedding_dim, sequence_length, sequence_length), generator=g)
            else:
                reference_embedding = torch.randn((sequence_length, self.reference_embedding_dim), generator=g)
        
        return {
            "seq_id": seq_id,
            "sequence": self.sequences[idx],
            "token_seq": self.token_seqs[idx] if self.token_seqs is not None else None,
            "bp_matrix": bp_matrix,
            "length": sequence_length,
            "embedding": embedding,
            "reference_embedding": reference_embedding,
        }
        
        
    def pad_batch(self, batch: list[dict]) -> dict:
        seq_ids = [b["seq_id"] for b in batch]
        sequences = [b["sequence"] for b in batch]
        token_seqs = [b["token_seq"] for b in batch] if self.token_seqs is not None else None
        bp_matrices = [b["bp_matrix"] for b in batch]
        lengths = [b["length"] for b in batch]
        embeddings = [b["embedding"] for b in batch] if self.cache_embedding_paths else None
        reference_embeddings = [b["reference_embedding"] for b in batch] if self.reference_embedding_dim is not None else None
        
        # バディング用にサイズを取得
        batch_size = len(batch)
        max_length = max(lengths)

        # トークン長 (additional tokenを含むことがあるので, lengths とは別にする必要がある)
        token_lengths = None
        if self.token_seqs is not None:
            if self.use_additional_token:
                token_lengths = [length + 2 for length in lengths]
            else:
                token_lengths = lengths

        # バッチ用のテンソルとattentionマスクを初期化
        token_seqs_padded = None
        attn_mask = None
        if self.token_seqs is not None:
            # additional tokenを使用する場合は, CLSとEOSの分だけmax_lengthより2大きいサイズでパディングする必要がある
            max_token_length = max_length + 2 if self.use_additional_token else max_length

            # token_seqsはadditional tokenを含む場合があるため, それらを考慮してパディングする必要がある. (lengthsはadditional tokenを含まないシーケンスの長さ)
            token_seqs_padded = torch.full((batch_size, max_token_length), fill_value=self.tokens.index("<pad>"), dtype=torch.long)
        
            # attentionマスクも同様にadditional tokenを考慮して初期化
            attn_mask = torch.full((batch_size, 1, max_token_length, max_token_length), fill_value=-1e6)
        
        # 正解二次構造matrixを-1で初期化
        bp_matrices_padded = -torch.ones((batch_size, max_length, max_length), dtype=torch.int8)
        
        # embeddingの初期化
        embeddings_padded = None
        if self.cache_embedding_paths:
            embedding_dim = embeddings[0].shape[0] if self.use_attention else embeddings[0].shape[-1]
            if self.use_attention:
                embeddings_padded = torch.zeros((batch_size, embedding_dim, max_length, max_length))
            else:
                embeddings_padded = torch.zeros((batch_size, max_length, embedding_dim))

        # reference_embeddingの初期化
        reference_embeddings_padded = None
        if self.reference_embedding_dim is not None:
            if self.use_attention:
                reference_embeddings_padded = torch.zeros((batch_size, self.reference_embedding_dim, max_length, max_length))
            else:
                reference_embeddings_padded = torch.zeros((batch_size, max_length, self.reference_embedding_dim))
        
        assert token_seqs_padded is not None or embeddings_padded is not None or reference_embeddings_padded is not None, "At least one of token_seqs, embeddings, or reference_embeddings must be returned by the dataset."
        
        # パディング
        for k in range(batch_size):
            if self.token_seqs is not None:
                token_seqs_padded[k, :token_lengths[k]] = token_seqs[k]
                attn_mask[k, :, :token_lengths[k], :token_lengths[k]] = 0

            bp_matrices_padded[k, :lengths[k], :lengths[k]] = bp_matrices[k]
            
            if self.cache_embedding_paths:
                if self.use_attention:
                    embeddings_padded[k, :, :lengths[k], :lengths[k]] = embeddings[k] # (B, E, L, L)
                else:
                    embeddings_padded[k, :lengths[k], :] = embeddings[k] # (B, L, E)
            
            if self.reference_embedding_dim is not None:
                if self.use_attention:
                    reference_embeddings_padded[k, :, :lengths[k], :lengths[k]] = reference_embeddings[k] # (B, ref_E, L, L)
                else:
                    reference_embeddings_padded[k, :lengths[k], :] = reference_embeddings[k] # (B, L, ref_E)
                    
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
    

def create_dataloader(config: MainConfig, split: str, pretrain_cfgs: list[PretrainMainConfig]=None) -> torch.utils.data.DataLoader:
    """
    データローダーの作成関数
    Args:
        config (MainConfig): 二次構造予測モデルの設定情報
        split (str): データ分割 ("train", "validation", "test", or "reference")
        pretrain_cfgs (list[PretrainMainConfig]): 事前学習モデルの設定情報のリスト
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
        reference_embedding_dim = 0
        # 事前学習モデルが指定されている場合
        if pretrain_cfgs is not None:
            if config.experiment.use_attention:
                for pretrain_cfg in pretrain_cfgs:
                    reference_embedding_dim += pretrain_cfg.framework.arch.n_layers * pretrain_cfg.framework.arch.n_heads
            else:
                for pretrain_cfg in pretrain_cfgs:
                    reference_embedding_dim += pretrain_cfg.model_size.embed_dim
                
            assert reference_embedding_dim > 0, "reference_embedding_dim must be positive unless embedding_file is specified"

    # embedding_fileが指定されている場合はそちらを使用
    embedding_paths = []
    if config.dataset.embedding_file is not None:
        for embedding_file in config.dataset.embedding_file:
            embedding_paths.append(Path(config.path.embedding_dir) / embedding_file)

    if pretrain_cfgs is not None:
        dataset = EmbeddingDataset(
            dataset_path=dataset_path,
            tokens=pretrain_cfgs[0].dataset.tokens,
            other_tokens=pretrain_cfgs[0].dataset.other_tokens,
            use_additional_token=pretrain_cfgs[0].experiment.use_additional_token,
            use_attention=config.experiment.use_attention,
            embedding_paths=embedding_paths,
            reference_embedding_dim=reference_embedding_dim,
        )
    else:
        # pretrain_configが指定されていない場合, 事前学習モデルを使用しないことを明示的に指定
        dataset = EmbeddingDataset(
            dataset_path=dataset_path,
            use_attention=config.experiment.use_attention,
            embedding_paths=embedding_paths,
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

def create_batch_iterator(config: MainConfig, split: str, pretrain_cfgs: list[PretrainMainConfig]=None) -> iter:
    """
    バッチイテレータの作成関数
    Args:
        config (MainConfig): 二次構造予測モデルの設定情報
        pretrain_cfgs (list[PretrainMainConfig]): 事前学習モデルの設定情報のリスト
        split (str): データ分割 ("train", "validation", or "test")
    
    Returns:
        Iterator: バッチイテレータ. batchとepochを返す
    """
    
    assert split in ["train", "validation", "test"], "split must be 'train', 'validation', or 'test'"
    
    loader = create_dataloader(config=config, split=split, pretrain_cfgs=pretrain_cfgs)
    
    epoch = 0
    while True:
        for batch in loader:
            batch["token_seqs"] = batch["token_seqs"].to(torch.long)
            yield batch, epoch
        epoch += 1
            
        if split == "test":
            break
    