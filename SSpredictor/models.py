from typing import Any

import torch
import torch.nn as nn
from modules import ResNet2D
from utils import apc, outer_concat, symmetrize

from pretrain.models import BaseModel as PretrainModel


class KnotFoldModel(nn.Module):
    """
    predictでKnotFoldのアルゴリズムを使用するモデル.
    
    Args:
        arch: アーキテクチャの設定. conf/config.pyのKnotFoldArchConfigを参照.
        pretrain_model: 事前学習モデル. BaseModelを継承したモデルを想定
        use_attention: 入力の形状がattention weightかどうか.
        device: モデルを配置するデバイス.
        reference: 参照モデルかどうか. 
    """
    def __init__(
        self,
        arch: dict[str, Any],
        pretrain_model: PretrainModel | None,
        embedding_dim: int,
        use_attention: bool,
        device: torch.device,
        reference: bool = False,
        **framework_kwargs,
    ):
        super().__init__()
        
        # 事前学習モデル
        self.pretrain_model = pretrain_model
        if self.pretrain_model is not None:
            self.pretrain_model.to(device=device)
            self.pretrain_model.eval()
        
        # アーキテクチャの構築
        self.norm = nn.LayerNorm(embedding_dim) 
        
        if arch["use_simple"]:
            self.linear = nn.Linear(embedding_dim, 1) if use_attention else nn.Linear(2 * embedding_dim, 1)
        else:
            self.linear_in = nn.Linear(embedding_dim, arch["conv_dim"]) if use_attention else nn.Linear(embedding_dim, int(arch["conv_dim"] / 2))
            self.resnet = ResNet2D(arch["conv_dim"], arch["n_residual_blocks"], kernel_size=arch["kernel_size"])
            self.conv_out = nn.Conv2d(arch["conv_dim"], 1, kernel_size=arch["kernel_size"], padding="same")
            
        # その他
        self.use_simple = arch["use_simple"]
        self.use_attention = use_attention
        self.reference = reference
        
        self.device = device
        self.to(device=device)
        
    def loss_func(self, pred: torch.Tensor, target: torch.Tensor):
        """
        Cross entropy loss. L1正規化は一旦保留
        Args:
            pred: 予測値. shape = (B, L, L)
            target: 真の二次構造行列. shape = (B, L, L), -1 for ignore, 0 for unpaired, 1 for paired
        Returns:
            torch.Tensor: 損失値
        """

        binary_target = target.clone()
        binary_target[binary_target == -1] = 0
        
        # マスク対象の位置の重みを0にするためのテンソル
        weight = torch.ones_like(binary_target)
        weight[target == -1] = 0

        # L1正規化は一旦保留
        # l1 = torch.tensor(0, device=self.device)
        # for param in self.parameters():
        #     l1 += torch.sum(torch.abs(param))

        # loss = nn.functional.binary_cross_entropy_with_logits(pred, binary_target.float()) + self.alpha * l1
        
        loss = nn.functional.binary_cross_entropy_with_logits(pred, binary_target.float(), weight=weight)

        return loss
        
    def forward(self, x: torch.Tensor, bp_matrix: torch.Tensor, split: str = "train"):
        """
        順伝播の関数. 入力は事前学習モデルの特徴量とデータセットの特徴量を結合したものを想定.
        Args:
            x (torch.Tensor): 入力特徴量. shape = (B, L, E) or (B, E, L, L)
            bp_matrix (torch.Tensor): 真の二次構造行列. shape = (B, L, L)
            split (str): データの分割. "train" または "test".

        Returns:
            Union: splitによって異なる.
            
            - train: torch.Tensor（損失値）
            - test:
            {
                "loss": torch.Tensor（損失値）
                "logits": torch.Tensor（予測された二次構造行列のロジット. shape = (B, L, L)）
            }
        """
        
        # 正規化
        x = self.norm(x)
            
        # attentionの前処理
        if self.use_attention:
            x = symmetrize(x)
            x = apc(x)
            x = x.permute(0, 2, 3, 1) # (B, E, L, L) -> (B, L, L, E)
        
        # アーキテクチャの適用
        # シンプルな場合
        if self.use_simple:
            if not self.use_attention:
                x = outer_concat(x, x) # (B, L, E) -> (B, L, L, 2E)
            
            x = self.linear(x).squeeze(-1)
            x = symmetrize(x) # (B, L, L)
            
        # CNNの場合
        else:
            x = self.linear_in(x)
            
            if not self.use_attention:
                x = outer_concat(x, x) # (B, L, M/2) -> (B, L, L, M)
            
            x = x.permute(0, 3, 1, 2) # (B, L, L, M) -> (B, M, L, L)
            
            x = self.resnet(x)
            x = self.conv_out(x)
            x = x.squeeze(1) # (B, 1, L, L) -> (B, L, L)
            x = symmetrize(x) # (B, L, L)
        
        # 損失計算
        loss = self.loss_func(x, bp_matrix)
        
        if split == "train":
            return loss

        elif split == "test":
            return {
                "loss": loss,
                "logits": x,
            }
        
        return x
        
    def _train(self, batch):
        """
        学習時の順伝播と損失計算を行う関数.
        Args:
            batch: バッチデータ
        Returns:
            torch.Tensor: 損失値
        """
        # 事前学習モデルからの特徴量の取得
        pretrain_model_embeddings = None
        if self.pretrain_model is not None and not self.reference:
            with torch.inference_mode():
                pretrain_outputs = self.pretrain_model._test(batch)
                
            if self.use_attention:
                pretrain_model_embeddings = pretrain_outputs["attn"]
            else:
                pretrain_model_embeddings = pretrain_outputs["repr"]
        
        # 読み込んだembeddingの取得
        dataset_embeddings = None
        if self.reference:
            dataset_embeddings = batch["reference_embeddings"]
        elif batch["embeddings"] is not None:
            dataset_embeddings = batch["embeddings"]
            
        # embeddingを結合するかどうか (めんどうなのでしない)
        # embeddings = None
        # if pretrain_model_embeddings is not None and dataset_embeddings is not None:
        #     if pretrain_model_embeddings.shape[1] != dataset_embeddings.shape[1]:
        #         raise ValueError(f"Dimension of pretrain model embeddings and dataset embeddings must match! ({pretrain_model_embeddings.shape} vs {dataset_embeddings.shape})")
            
        #     if self.use_attention:
        #         embeddings = torch.stack([pretrain_model_embeddings, dataset_embeddings], dim=1)
        #     else:
        #         embeddings = torch.cat([pretrain_model_embeddings, dataset_embeddings], dim=-1)
                
        # elif pretrain_model_embeddings is not None:
        #     embeddings = pretrain_model_embeddings
        # else:
        #     embeddings = dataset_embeddings
        
        embeddings = pretrain_model_embeddings if pretrain_model_embeddings is not None else dataset_embeddings
            
        assert embeddings is not None, "Embeddings could not be constructed!"

        # forward
        loss = self(
            embeddings.to(device=self.device),
            batch["bp_matrices"].to(device=self.device),
            split="train",
        )
        
        return loss
    
    def _test(self, batch):
        """
        基本は_trainと同じ. 評価時の順伝播と損失計算を行い, 損失とlogitsを返す.
        Args:
            batch: バッチデータ
        Returns:
            dict:
            {  
                "loss": torch.Tensor（損失値）   
                "logits": torch.Tensor（予測された二次構造行列のロジット. shape = (B, L, L)）  
            }
            
        """
        # 事前学習モデルからの特徴量の取得
        pretrain_model_embeddings = None
        if self.pretrain_model is not None:
            with torch.inference_mode():
                pretrain_outputs = self.pretrain_model._test(batch)
                
            if self.use_attention:
                pretrain_model_embeddings = pretrain_outputs["attn"]
            else:
                pretrain_model_embeddings = pretrain_outputs["repr"]
        
        # 読み込んだembeddingの取得
        dataset_embeddings = None
        if batch["embeddings"] is not None:
            dataset_embeddings = batch["embeddings"]

        embeddings = pretrain_model_embeddings if pretrain_model_embeddings is not None else dataset_embeddings
            
        assert embeddings is not None, "Embeddings could not be constructed!"

        # forward
        results = self(
            embeddings.to(device=self.device),
            batch["bp_matrices"].to(device=self.device),
            split="test",
        )
        
        return {
            "loss": results["loss"],
            "logits": results["logits"],
        }
    
    @staticmethod
    def predict(
        gt_bp_matrix: torch.Tensor,
        pred_bp_matrix: torch.Tensor,
        ref_bp_matrix: torch.Tensor,
        kf_lambda: list[float]
    ):
        """
        二次構造行列の予測を行う関数. KnotFoldのアルゴリズムを使用する.
        Args:
            gt_bp_matrix (torch.Tensor): 真の二次構造行列. shape = (B, L, L)
            pred_bp_matrix (torch.Tensor): 予測された二次構造確率行列. shape = (B, L, L)
            ref_bp_matrix (torch.Tensor): 参照モデルの二次構造行列確率行列. shape = (B, L, L)
            kf_lambda (list[float]): min cost flow アルゴリズムで使用するλ.
        Returns:
            torch.Tensor: 予測された二次構造行列. shape = (B, L, L)
        """
        
        # ここにKnotFoldのアルゴリズムを実装する
        
        if kf_lambda is None:
            kf_lambda = self.kf_lambda
        
        # --- IGNORE ---
    
    def _load_state_dict(self, state_dict: dict):
        """
        torch.compileを考慮したload_state_dictの実装
        
        Args:
            state_dict (dict): ロードするstate_dict
        """
        
        return self.load_state_dict(
            {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}
        )
        