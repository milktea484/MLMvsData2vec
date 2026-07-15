import torch

from pretrain.models import BaseModel
from utils import create_attention_bias
from pretrain.conf.config import MainConfig


class REDIALlike:
    """
    REDIALlikeは、REDIALの戦略に則って、変異を加えた配列を入力とした事前学習モデルの出力から、その変異による特徴表現の変化をcontact mapとして行列に変換するクラス。
    基本的には、事前学習モデルの配列特徴表現をcontact mapに変換するためのメソッドを提供する。
    """

    def __init__(self, pretrain_model: BaseModel, pretrain_config: MainConfig, rna_tokens: list[str] = ["A", "C", "G", "U"], device: torch.device = torch.device("cpu")):
        """"
        Args:
            pretrain_model (BaseModel): REDIALlikeで使用する事前学習モデル
            pretrain_config (MainConfig): 事前学習モデルのconfig
            rna_tokens (list[str]): RNAトークンのリスト (default=["A", "C", "G", "U"])
            device (torch.device): 使用するデバイス (default=torch.device("cpu"))
        """
        self.pretrain_model = pretrain_model
        self.rna_tokens = rna_tokens
        self.n_layers = pretrain_config.model_size.n_layers
        self.ernie_rna_alpha = pretrain_config.framework.ernie_rna_alpha
        self.device = device
        self.set_extract_repr_layers(pretrain_config.experiment.extract_repr_layers)
        
    def set_extract_repr_layers(self, layers: list[int] | int | str):
        """
        事前学習モデルのどのレイヤの特徴表現を抽出するかを設定するメソッド。
        
        Args:
            layers (list[int] | int | str): 抽出するレイヤ番号のリスト, 単体のレイヤ番号, または"all"
        """
        if isinstance(layers, int):
            layers = [layers]
        elif isinstance(layers, str) and layers == "all":
            layers = list(range(self.n_layers+1))
        
        self.extract_repr_layers = layers

    def generate_contact_map(self, batch, return_attn=False):
        """
        REDIALlikeのメインメソッド。入力された配列のバッチ（バッチサイズ1）に対して、事前学習モデルを用いてcontact mapを生成する。
        バッチはpretrain.datasetの形式に従っている。パディングを考慮するのが大変なので、バッチサイズは1で固定されていること前提。
        Args:
            batch (dict): pretrain.datasetの形式に従ったバッチ。バッチサイズは1であることが前提。
            return_attn (bool): Attention mapを返すかどうか。
        Returns:
            contact_map (torch.Tensor): 変異による特徴表現の変化をcontact mapとして表現した行列。shapeは(len(extract_repr_layers), L, L)で、Lは配列長。
            変異前後の特徴表現の差分を計算/平均し、対称化とapcを行っている（正規化はしていない）。
            attn_map (torch.Tensor, optional): Attention map。return_attnがTrueの場合に返される。shapeは(len(extract_repr_layers), L, L)。
        """
        # バッチの内容の抽出
        token_seq = batch["token_seqs"] # (B, L)
        length = batch["lengths"][0]

        new_batch = {
            "seq_ids": batch["seq_ids"] * 4,  # 変異前後の配列をconcatするため、バッチサイズを4倍にする
            "token_seqs": torch.cat([token_seq, token_seq, token_seq, token_seq], dim=0),  # (4B, L)
            "attn_mask": batch["attn_mask"].repeat(4, 1, 1, 1),  # (4B, 1, L, L)
            "attn_biases": batch["attn_biases"] if batch["attn_biases"] is not None else None,  # (B, 1, L, L) or None (Noneじゃない場合は後で変異後の配列に対応するattention biasを求めるのでrepeatしない)
            "attn_biases_masked": batch["attn_biases_masked"] if batch["attn_biases_masked"] is not None else None,  # (B, 1, L, L) or None (Noneじゃない場合は後で変異後の配列に対応するattention biasを求めるのでrepeatしない)
            "lengths": batch["lengths"] * 4,  # (4B,)
        }

        # contact mapのベース行列を初期化(レイヤー数, 配列長, 配列長)
        contact_map_list = [[] for _ in range(len(self.extract_repr_layers))]

        # 以下の操作を配列長だけループ
        ## 変異配列の生成
        ## 変異前の配列と変異後の配列をconcatして、batchをサイズ4に拡張
        ## 配列の特徴表現を事前学習モデルで取得
        ## 変異前後での特徴表現の差分を計算し、contact mapに反映
        for i in range(length):
            # 変異配列の生成
            mutated_batch = self._mutate_batch(new_batch, mutation_index=i)
            # 変異前後の特徴表現を取得
            representations = self.pretrain_model._test(mutated_batch, extract_repr_layers=self.extract_repr_layers)["repr"]  # len(extract_repr_layers, 4B, L, E) or (4B, L, E) if len(extract_repr_layers)==1
            if representations.dim() == 3:  # len(extract_repr_layers)==1の場合は、(4B, L, E)の形になるので、(1, 4B, L, E)に変換する
                representations = representations.unsqueeze(0)  # (1, 4B, L, E)

            for layer_idx, layer_repr in enumerate(representations):
                # 変異前後の特徴表現を正規化
                normalized_repr = torch.nn.functional.layer_norm(layer_repr, (layer_repr.shape[-1],))  # (4B, L, E)
                
                # 変異前後の特徴表現の差分を計算
                diff_repr_list = [torch.linalg.norm(normalized_repr[mutation_batch_idx + 1] - normalized_repr[0], dim=-1) for mutation_batch_idx in range(3)]  # (3, L)
                diff_repr = torch.stack(diff_repr_list, dim=0).mean(dim=0)  # (L,)
                contact_map_list[layer_idx].append(diff_repr)

        # contact mapに反映
        contact_map = torch.stack([torch.stack(layer_diff_list, dim=0) for layer_diff_list in contact_map_list], dim=0)  # (len(extract_repr_layers), L, L)

        # 対称化、apc
        contact_map = torch.triu(contact_map, diagonal=1) + torch.tril(contact_map, diagonal=-1)    # 対角成分を0として対称化
        contact_map = self._apc(contact_map)

        # return_attnがTrueの場合は、attention mapも返す
        attention_map = None
        if return_attn:
            attention_map = self.pretrain_model._test(mutated_batch, extract_repr_layers=self.extract_repr_layers)["attn"][0] # (n_layers*n_heads, L, L) 変異前の配列のattention map

        # 出力
        return contact_map, attention_map
    
    def _mutate_batch(self, batch, mutation_index):
        """
        REDIALlikeの内部メソッド。入力されたバッチに対して、指定されたインデックスの塩基を変異させた新しいバッチを生成する。
        Args:
            batch (dict): pretrain.datasetの形式に従ったバッチ
            mutation_index (int): 変異させる塩基のインデックス
        Returns:
            mutated_batch (dict): 変異後の配列を含む新しいバッチ。
            変異は、指定されたインデックスの塩基をランダムに選ばれた別の塩基に置き換えることで行う。変異後の配列は、元の配列と同じ長さを持つ。
        """
        # 変異前の配列を取得
        token_seq = batch["token_seqs"].clone()  # (4B, L)

        # 変異後の配列を生成
        original_token = token_seq[0, mutation_index].item()
        mutation_token_list = [self.rna_tokens.index(token) for token in self.rna_tokens if self.rna_tokens.index(token) != original_token]

        # original_tokenと異なる3つの塩基に変異
        for mutation_batch_idx, mutation_token in enumerate(mutation_token_list):
            token_seq[mutation_batch_idx + 1, mutation_index] = mutation_token

        # attention biasがNoneでない場合は、変異後の配列に対応するattention biasを更新する必要がある。
        attn_biases = batch["attn_biases"]
        if attn_biases is not None:
            # 変異後の配列に対応するattention biasを更新
            attn_bias_list = [attn_biases[0]]  # 変異前の配列のattention bias
            attn_bias_list += [
                create_attention_bias(
                    token_seq[mutation_batch_idx + 1],
                    use_ernie_rna=True,
                    ernie_rna_alpha=self.ernie_rna_alpha
                ) for mutation_batch_idx in range(3)
            ]  # 変異後の配列のattention bias

            attn_biases = torch.stack(attn_bias_list, dim=0)  # (4B, 1, L, L)

        # 新しいバッチを生成
        mutated_batch = {
            "seq_ids": batch["seq_ids"],
            "token_seqs": token_seq,
            "attn_mask": batch["attn_mask"],
            "attn_biases": attn_biases,
            "attn_biases_masked": attn_biases,
            "lengths": batch["lengths"],
        }
        
        return mutated_batch

    def _apc(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform average product correct, used for contact prediction.
        (by https://github.com/facebookresearch/esm/blob/2b369911bb5b4b0dda914521b9475cad1656b2ac/esm/modules.py#L32)
        
        Args:
            x: shape = ... x L x L
        Returns:
            torch.Tensor: shape = ... x L x L
        """
        
        a1 = x.sum(-1, keepdims=True)
        a2 = x.sum(-2, keepdims=True)
        a12 = x.sum((-1, -2), keepdims=True)

        avg = a1 * a2
        avg.div_(a12)
        normalized = x - avg
        
        return normalized
