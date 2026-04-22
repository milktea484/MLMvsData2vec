import os
import subprocess
import tempfile
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from metrics import precision_recall_f1
from modules import ResNet2D
from tqdm import tqdm
from utils import apc, bp2matrix, outer_concat, symmetrize

import pretrain


class KnotFoldModel(nn.Module):
    """
    predictでKnotFoldのアルゴリズムを使用するモデル.
    
    Args:
        arch: アーキテクチャの設定. conf/config.pyのKnotFoldArchConfigを参照.
        pretrain_models: 事前学習モデルのリスト. BaseModelを継承したモデルを想定
        embedding_dim: 埋め込み次元数. 事前学習モデルの特徴量とデータセットの特徴量を結合した後の次元数を指定する必要がある.
        use_attention: 入力の形状がattention weightかどうか.
        device: モデルを配置するデバイス.
        reference: 参照モデルかどうか. 
    """
    def __init__(
        self,
        arch: dict[str, Any],
        pretrain_models: list[pretrain.models.BaseModel] | None,
        embedding_dim: int,
        use_attention: bool,
        device: torch.device,
        reference: bool = False,
        **framework_kwargs,
    ):
        super().__init__()
        
        # 事前学習モデル
        if pretrain_models is not None:
            for pretrain_model in pretrain_models:
                pretrain_model.eval()
                pretrain_model.to(device=device)
        self.pretrain_models = pretrain_models
        
        # アーキテクチャの構築
        self.norm = nn.LayerNorm(embedding_dim) 
        
        if arch["use_simple"]:
            self.linear = nn.Linear(embedding_dim, 1) if use_attention else nn.Linear(2 * embedding_dim, 1)
        else:
            self.linear_in = nn.Linear(embedding_dim, arch["conv_dim"]) if use_attention else nn.Linear(embedding_dim, int(arch["conv_dim"] / 2))
            self.resnet = ResNet2D(arch["conv_dim"], arch["n_residual_blocks"], kernel_size=arch["kernel_size"])
            self.conv_out = nn.Conv2d(arch["conv_dim"], 1, kernel_size=arch["kernel_size"], padding="same")
            
        # その他
        self.embedding_dim = embedding_dim
        self.use_simple = arch["use_simple"]
        self.use_attention = use_attention
        self.reference = reference
        
        self.device = device
        self.to(device=device)
    
    def get_pretrain_model_embeddings(self, batch) -> torch.Tensor | None:
        """
        事前学習モデルからの特徴量を取得する関数. 事前学習モデルが複数ある場合は結合して返す. モデルがない場合はNoneを返す.
        Args:
            batch: バッチデータ
        Returns:
            torch.Tensor | None: 事前学習モデルからの特徴量. shape = (B, L, E) or (B, E, L, L)
        """
        
        pretrain_model_embeddings = None
        pretrain_model_embeddings_list = []
        if self.pretrain_models is not None:
            for pretrain_model in self.pretrain_models:
                with torch.inference_mode():
                    pretrain_outputs = pretrain_model._test(batch)
                
                output_embeddings = pretrain_outputs["attn"] if self.use_attention else pretrain_outputs["repr"]
                
                # additional tokensを除去 (必要な場合)
                max_length = max(batch["lengths"])
                ## attentionの場合
                if self.use_attention:
                    if output_embeddings.shape[3] > max_length:
                        assert output_embeddings.shape[3] == max_length + 2, f"Output embeddings length does not match expected length! ({output_embeddings.shape[3]} vs {max_length} + 2)"

                        output_embeddings = output_embeddings[:, :, 1:-1, 1:-1] if self.use_attention else output_embeddings[:, 1:-1, :] # (B, E, L+2, L+2) -> (B, E, L, L)
                ## attentionでない場合
                else:
                    if output_embeddings.shape[1] > max_length:
                        assert output_embeddings.shape[1] == max_length + 2, f"Output embeddings length does not match expected length! ({output_embeddings.shape[1]} vs {max_length} + 2)"

                        output_embeddings = output_embeddings[:, 1:-1, :] # (B, L+2, E) -> (B, L, E)

                pretrain_model_embeddings_list.append(output_embeddings)
            
            if self.use_attention:
                pretrain_model_embeddings = torch.cat(pretrain_model_embeddings_list, dim=1) # (B, E, L, L) * num_models -> (B, E_total, L, L)
            else:
                pretrain_model_embeddings = torch.cat(pretrain_model_embeddings_list, dim=-1) # (B, L, E) * num_models -> (B, L, E_total)
        
        return pretrain_model_embeddings
    
    def concatenate_embeddings(self, pretrain_model_embeddings: torch.Tensor | None, dataset_embeddings: torch.Tensor | None, max_length: int) -> torch.Tensor:
        """
        事前学習モデルからの特徴量とデータセットの特徴量を結合する関数.
        Args:
            pretrain_model_embeddings: 事前学習モデルからの特徴量. shape = (B, L, E) or (B, E, L, L)
            dataset_embeddings: データセットからの特徴量. shape = (B, L, E) or (B, E, L, L)
            max_length: バッチ内最大シーケンス長
        Returns:
            torch.Tensor: 結合された特徴量. shape = (B, L, E_total) or (B, E_total, L, L)
        """
        
        embeddings = None
        if pretrain_model_embeddings is not None and dataset_embeddings is not None:
            if pretrain_model_embeddings.dim() != dataset_embeddings.dim():
                raise ValueError(f"Dimension of pretrain model embeddings and dataset embeddings must match! ({pretrain_model_embeddings.dim()} vs {dataset_embeddings.dim()})")
            
            if self.use_attention:
                embeddings = torch.cat([pretrain_model_embeddings, dataset_embeddings], dim=1)
            else:
                embeddings = torch.cat([pretrain_model_embeddings, dataset_embeddings], dim=-1)
                
        elif pretrain_model_embeddings is not None:
            embeddings = pretrain_model_embeddings
        elif dataset_embeddings is not None:
            embeddings = dataset_embeddings
        
        # embeddings = pretrain_model_embeddings if pretrain_model_embeddings is not None else dataset_embeddings
        
        # assertion
        assert embeddings is not None, "Embeddings could not be constructed!"
        if self.use_attention:
            assert embeddings.dim() == 4, f"Embeddings dimension must be 4 for attention! (got {embeddings.dim()})"
            assert embeddings.shape[2] == max_length and embeddings.shape[3] == max_length, f"Embeddings spatial dimensions must match max sequence length! (got {embeddings.shape[2:]} vs {max_length})"
            assert embeddings.shape[1] == self.embedding_dim, f"Embeddings channel dimension must match embedding_dim! (got {embeddings.shape[1]} vs {self.embedding_dim})"
        else:
            assert embeddings.dim() == 3, f"Embeddings dimension must be 3 for non-attention! (got {embeddings.dim()})"
            assert embeddings.shape[1] == max_length, f"Embeddings sequence dimension must match max sequence length! (got {embeddings.shape[1]} vs {max_length})"
            assert embeddings.shape[2] == self.embedding_dim, f"Embeddings feature dimension must match embedding_dim! (got {embeddings.shape[2]} vs {self.embedding_dim})"
        
        return embeddings
        
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
        # attentionの前処理
        if self.use_attention:
            x = symmetrize(x)
            x = apc(x)
            x = x.permute(0, 2, 3, 1) # (B, E, L, L) -> (B, L, L, E)
        
        # 正規化
        x = self.norm(x)
        
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
        pretrain_model_embeddings = self.get_pretrain_model_embeddings(batch) if not self.reference else None

        # 読み込んだembeddingの取得
        dataset_embeddings = None
        if self.reference:
            dataset_embeddings = batch["reference_embeddings"]
        elif batch["embeddings"] is not None:
            dataset_embeddings = batch["embeddings"]
            
        # embeddingの結合
        embeddings = self.concatenate_embeddings(pretrain_model_embeddings, dataset_embeddings, max_length=max(batch["lengths"]))

        # forward
        loss = self(
            embeddings.to(device=self.device),
            batch["bp_matrices"].to(device=self.device),
            split="train",
        )
        
        return loss
    
    def _test(self, batch):
        """
        基本は_trainと同じだが, referenceは考慮しない. 評価時の順伝播と損失計算を行い, 損失とlogitsを返す.
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
        pretrain_model_embeddings = self.get_pretrain_model_embeddings(batch)

        # 読み込んだembeddingの取得
        dataset_embeddings = None
        if batch["embeddings"] is not None:
            dataset_embeddings = batch["embeddings"]
            
        # embeddingの結合
        embeddings = self.concatenate_embeddings(pretrain_model_embeddings, dataset_embeddings, max_length=max(batch["lengths"]))

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
    @torch.inference_mode()
    def predict(
        batch_list: list[dict[str, Any]],
        kf_lambda_list: list[float]
    ):
        """
        二次構造行列の予測を行う関数. KnotFoldのアルゴリズムを使用する. 最適なkf_lambdaを選択するように
        Args:
            batch_list (list[dict[str, Any]]): バッチデータのリスト (len = B). 各バッチは以下のキーを持つことを想定.
                - seq_id: 配列のID
                - sequence: 配列の文字列
                - length: 配列の長さ
                - gt_bp_matrix: 真の二次構造行列. shape = (L, L)
                - pred_bp_prob: 予測された二次構造行列の確率. shape = (L, L)
                - ref_bp_prob: 参照二次構造行列の確率. shape = (L, L)
            kf_lambda_list (list[float]): 最小コストフローアルゴリズムで使用するλのリスト.
            
        Returns:
            dict:
            {  
                "kf_lambda_results": kf_lambdaの値をキー, f1スコア,precision,recallをもつ辞書を値とする辞書.   
                "prediction_results": kf_lambdaが最適な場合の予測結果のリスト (len = B). 各要素は以下のキーを持つことを想定.
                    - seq_id: 配列のID
                    - sequence: 配列の文字列
                    - gt_bp_matrix: 真の二次構造行列. shape = (L, L)
                    - pred_bp_matrix: 予測された二次構造行列. shape = (L, L)
                    - pairs: 予測された塩基対のリスト. 各要素は (i, j) の形式で, i番目とj番目の塩基がペアであることを示す.
                    - scores: 予測された二次構造と真の二次構造のF1スコア, Precision, Recall.
            }
        """
        
        # 結果の保存用
        results = {
            "kf_lambda_results": {},
            "prediction_results": [],
        }
        
        # 最大のF1スコアを出すkf_lambdaを選択するための変数
        best_kf_lambda = None
        best_f1 = -1.0
        
        # 前のkf_lambdaのF1スコアと比較して, 連続して3回改善がない場合は終了するための変数
        prev_f1 = -1.0
        no_improve_count = 0
        
        # kf_lambdaのリストをループして最適なものを選択する
        with tqdm(total=len(kf_lambda_list)*len(batch_list), desc="Testing kf_lambda values") as pbar:
            for kf_lambda in kf_lambda_list:
                # batchごとの結果の保存用
                prediction_results = []
                
                # スコアの平均
                avg_f1 = 0.0
                avg_pre = 0.0
                avg_rec = 0.0
                
                here = os.path.dirname(__file__)
                with tempfile.TemporaryDirectory() as d:
                    # 各配列について予測を行う
                    for batch in batch_list:
                        seq_id = batch["seq_id"]
                        sequence = batch["sequence"]
                        length = batch["length"]
                        gt_bp_matrix = batch["gt_bp_matrix"].to(torch.float32).detach().cpu() # (L, L)
                        pred_bp_prob = batch["pred_bp_prob"].to(torch.float32).detach().cpu() # (L, L)
                        ref_bp_prob = batch["ref_bp_prob"].to(torch.float32).detach().cpu() # (L, L)

                        fg: np.ndarray = pred_bp_prob.numpy()
                        bg: np.ndarray = ref_bp_prob.numpy()
                        with open(os.path.join(d, "prior.mat"), 'w') as fp:
                            for i in range(fg.shape[0]):
                                for j in range(fg.shape[0]):
                                    fp.write("%.10f" % fg[i][j])
                                    fp.write("\t")
                                fp.write("\n")
                        with open(os.path.join(d, "reference.mat"), 'w') as fp:
                            for i in range(bg.shape[0]):
                                for j in range(bg.shape[0]):
                                    fp.write("%.10f" % bg[i][j])
                                    fp.write("\t")
                                fp.write("\n")

                        mincostflowcmd = f"{here}/knotfold/KnotFold_mincostflow {d}/prior.mat {d}/reference.mat {kf_lambda}"
                        p = subprocess.run(mincostflowcmd, shell=True, capture_output=True)
                        assert p.returncode == 0
                        pairs = []
                        for line in p.stdout.decode().split("\n"):
                            if len(line) == 0:
                                continue
                            l, r = line.split()
                            pairs.append([int(l), int(r)])
                        
                        pred_bp_matrix = bp2matrix(length, pairs).detach().cpu() # (L, L)
                        
                        # スコアの計算
                        pre, rec, f1 = precision_recall_f1(gt_bp_matrix, pred_bp_matrix)
                        avg_f1 += f1
                        avg_pre += pre
                        avg_rec += rec
                        
                        prediction_results.append({
                            "seq_id": seq_id,
                            "sequence": sequence,
                            "length": length,
                            "gt_bp_matrix": gt_bp_matrix,
                            "pred_bp_matrix": pred_bp_matrix,
                            "pairs": pairs,
                            "scores": {
                                "f1": f1,
                                "precision": pre,
                                "recall": rec,
                            },
                        })
                        
                        pbar.update(1)
                        pbar.set_postfix({
                            "kf_lambda": kf_lambda,
                            "batch_f1": f1,
                        })

                # スコアの平均を計算
                avg_f1 /= len(batch_list)
                avg_pre /= len(batch_list)
                avg_rec /= len(batch_list)
                
                # 今回のkf_lambdaのスコアを保存
                results["kf_lambda_results"][kf_lambda] = {
                    "is_optimal": False,
                    "f1": avg_f1,
                    "precision": avg_pre,
                    "recall": avg_rec,
                }
                
                # 最適なkf_lambdaを更新
                if avg_f1 > best_f1:
                    best_f1 = avg_f1
                    best_kf_lambda = kf_lambda
                    results["prediction_results"] = prediction_results
                    
                # 直前のkf_lambdaのスコアと比較して改善がない場合が3回続いたら終了
                if prev_f1 != -1.0 and avg_f1 <= prev_f1:
                    no_improve_count += 1
                    if no_improve_count >= 3:
                        break
                else:
                    no_improve_count = 0

                prev_f1 = avg_f1
            
        results["kf_lambda_results"][best_kf_lambda]["is_optimal"] = True
            
        return results

    
    def _load_state_dict(self, state_dict: dict):
        """
        torch.compileを考慮したload_state_dictの実装
        
        Args:
            state_dict (dict): ロードするstate_dict
        """
        
        return self.load_state_dict(
            {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}
        )
        