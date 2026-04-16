import math
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conf.config import ExperimentConfig, MLMLossConfig, QuantizeConfig
from .modules import CommonModule


class BaseModel(nn.Module):
    """
    MLMModelとdata2vecModelの基底クラス
    
    Args:
        padding_idx (int): パディングトークンのインデックス
        num_tokens (int): トークンの総数
        device: 使用デバイス
    """
    def __init__(self, padding_idx: int, num_tokens: int, device):
        super().__init__()
        self.device = device
        self.padding_idx = padding_idx
        self.num_tokens = num_tokens
        
    def loss_func(self, *args, **kwargs):
        raise NotImplementedError("loss_func method not implemented.")
    
    def _train(self, batch):
        raise NotImplementedError("_train method not implemented.")
    
    def _validate(self, batch):
        raise NotImplementedError("_validate method not implemented.")
    
    def _test(self, batch):
        raise NotImplementedError("_test method not implemented.")

    def _step(self):
        raise NotImplementedError("_step method not implemented.")
    
    def save_model(self, save_path: Path, step: int):
        raise NotImplementedError("save_model method not implemented.")
            
    def _load_state_dict(self, state_dict: dict):
        """
        torch.compileを考慮したload_state_dictの実装
        
        Args:
            state_dict (dict): ロードするstate_dict
        """
        
        return self.load_state_dict(
            {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}
        )


class MLMModel(BaseModel):
    """
    Masked Language Modelingの実装
    
    Args:
        arch (dict[str, Any]): モデルアーキテクチャの設定情報
        padding_idx (int): パディングトークンのインデックス
        num_tokens (int): トークンの総数
        experiment_cfg (ExperimentConfig): 実験に関する設定情報
        device: 使用デバイス
    """
    def __init__(
        self,
        arch: dict[str, Any],
        padding_idx: int,
        num_tokens: int,
        experiment_cfg: ExperimentConfig,
        device,
        **framework_kwargs
    ):
        super().__init__(padding_idx=padding_idx, num_tokens=num_tokens, device=device)
        
        # レイヤ数など
        self.n_layers = arch["n_layers"]

        # モデルの構築
        self.model = CommonModule(arch, self.padding_idx, self.num_tokens, device)

        # regression headの構築 (MLMでは単純に線形層)
        curr_dim = arch["embed_dim"]
        projs = []
        for i in range(arch["n_head_layers"] - 1):
            next_dim = arch["embed_dim"] * 2 if i == 0 else curr_dim
            projs.append(nn.Linear(curr_dim, next_dim))
            projs.append(nn.GELU())
            curr_dim = next_dim
        projs.append(nn.Linear(curr_dim, arch["embed_dim"]))
        self.regression_head = nn.Sequential(*projs)
        
        # 分類器の構築
        self.classifier = nn.Linear(arch["embed_dim"], self.num_tokens)
        
        # その他に関する設定
        self.extract_repr_layers= experiment_cfg.extract_repr_layers
        self.to(self.device)
        
    def loss_func(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        損失関数の計算
        
        Args:
            logits (torch.Tensor): モデルの出力ロジット (num_masked_positions, vocab_size)
            target (torch.Tensor): 正解ラベル (num_masked_positions,)
        Returns:
            torch.Tensor: cross_entropyを用いて計算された損失
        """
        return F.cross_entropy(logits, target)
        
    def forward(
        self,
        token_input: torch.Tensor,
        attn_mask: torch.Tensor,
        target: torch.Tensor = None,
        attn_biases_masked: torch.Tensor = None,
        masked_idxes: list[list[int]] = None,
        mode: str = "train",
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """
        train, validation, testで共通のforward処理.
        trainとvalidationでは損失計算まで行い, testでは出力のみ取得する.
        Args:
            token_input (torch.Tensor): 入力テンソル (B, L)  
            attn_mask (torch.Tensor): biasを含んだアテンションマスク (B, 1, L, L)  
            target (torch.Tensor | None): 正解ラベルテンソル (B, L)  
            attn_biases_masked (torch.Tensor | None): マスクされた入力に対するアテンションのバイアス (B, 1, L, L)
            masked_idxes (list[list[int]] | None): マスクされた位置のインデックスリスト (B, [mask_positions])  
            mode (str): "train", "validation", "test"のいずれか
        Returns:
            Union: モードによって返り値が異なる  
            
            - trainの場合: loss (torch.Tensor)  
            - validationの場合: 以下の辞書  
                {  
                "loss": torch.Tensor  
                "var": torch.Tensor (2,)  
                }
            - testの場合: 以下の辞書  
                {  
                "repr": torch.Tensor (B, L, embed_dim)  
                "attn": torch.Tensor (B, n_layers*n_heads, L, L)  
                }
        """
        assert mode in ["train", "validation", "test"], f"Unsupported mode: {mode}"
        
        # 中間表現を取得するレイヤの指定
        # train時は最終層のみ, test時は全レイヤ
        repr_layers = [self.n_layers] if mode != "test" else list(range(self.n_layers+1))
        
        # モデルの順伝搬
        hidden_reprs = self.model(
            token_input,
            attn_mask,
            attn_biases=attn_biases_masked,
            repr_layers=repr_layers,
            return_fc=False,
            test_mode=(mode == "test"),
        )
        
        # 最終層の出力のうち, 配列特徴のみを取得
        x = [x_and_attn[0] for x_and_attn in hidden_reprs][-1]  # (B, L, embed_dim)
        x = F.layer_norm(x, (x.shape[-1],))
        
        # testの場合はextract_repr_layers層の配列特徴と, 全レイヤのアテンションを取得
        if mode == "test":
            repr_output = hidden_reprs[self.extract_repr_layers][0]  # (B, L, embed_dim)
            repr_output = F.layer_norm(repr_output, (repr_output.shape[-1],))
            attn_output = [x_and_attn[1] for x_and_attn in hidden_reprs[1:]] # 最初の層はアテンションを持たない
            attn_output = torch.cat(attn_output, dim=1)  # (B, n_layers*n_heads, L, L)
            
            return {
                "repr": repr_output,
                "attn": attn_output,
            }
            
        # validattion時は分散を計算
        if mode == "validation":
            var = torch.stack([
                torch.var(x.view(-1, x.shape[-1]), dim=0).mean(),
                torch.var(x.view(-1, x.shape[-1]), dim=1).mean(),
            ])
        
        # マスクされた位置のロジットと正解ラベルを抽出
        masked_idxes_tensor = torch.zeros_like(token_input, dtype=torch.bool)  # (B, L)
        for batch_idx, idxes in enumerate(masked_idxes):
            masked_idxes_tensor[batch_idx, idxes] = True
            
        x = x[masked_idxes_tensor]  # (num_masked_positions, embed_dim)
        target = target[masked_idxes_tensor]  # (num_masked_positions,)

        # regression headの適用
        x = self.regression_head(x)  # (num_masked_positions, embed_dim)
            
        # 分類器の適用
        logits = self.classifier(x)  # (num_masked_positions, vocab_size)
        
        # 損失計算用の形状に変換
        logits = logits.view(-1, logits.shape[-1])  # (num_masked_positions, vocab_size)
        target = target.view(-1)  # (num_masked_positions,)
        
        # 損失計算
        loss = self.loss_func(logits, target)
        
        # train時とvalidation時で返り値を分ける
        if mode == "train":
            return loss
        else:  # validation
            assert mode == "validation"
            return {
                "loss": loss,
                "var": var,
            }
    
    def _train(self, batch):
        """
        訓練用
        Args:
            batch: バッチデータ
        Returns:
            loss: torch.Tensor
        """
        token_input = batch["token_seqs_masked"].to(self.device)
        target = batch["token_seqs"].to(self.device)
        attn_mask = batch["attn_mask"].to(self.device)
        masked_idxes = batch["masked_idxes"]
        
        attn_biases_masked = batch["attn_biases_masked"]
        if isinstance(attn_biases_masked, torch.Tensor):
            attn_biases_masked = attn_biases_masked.to(self.device)
        
        loss = self(
            token_input,
            attn_mask,
            target=target,
            attn_biases_masked=attn_biases_masked,
            masked_idxes=masked_idxes,
            mode="train",
        )
        
        return loss
    
    def _validate(self, batch):
        """
        検証用
        Args:
            batch: バッチデータ
        Returns:
            dict:
                - loss: torch.Tensor
                - var: torch.Tensor (2,)
        """
        token_input = batch["token_seqs_masked"].to(self.device)
        target = batch["token_seqs"].to(self.device)
        attn_mask = batch["attn_mask"].to(self.device)
        masked_idxes = batch["masked_idxes"]
        
        attn_biases_masked = batch["attn_biases_masked"]
        if isinstance(attn_biases_masked, torch.Tensor):
            attn_biases_masked = attn_biases_masked.to(self.device)
        
        results = self(
            token_input,
            attn_mask,
            target=target,
            attn_biases_masked=attn_biases_masked,
            masked_idxes=masked_idxes,
            mode="validation",
        )
        
        return results
    
    def _test(self, batch):
        """
        テスト用
        Args:
            batch: バッチデータ
        Returns:
            dict:
                - repr: torch.Tensor (B, L, embed_dim)
                - attn: torch.Tensor (B, n_layers*n_heads, L, L)
        """
        token_input = batch["token_seqs"].to(self.device)
        attn_mask = batch["attn_mask"].to(self.device)
        
        results = self(
            token_input,
            attn_mask,
            target=None,
            masked_idxes=None,
            mode="test",
        )
        
        return results

    def _step(self):
        """
        学習ステップの更新 (optimizerとlr_schedulerは外部に設定しているのでここでは何もしない)
        """
        pass
    
    def save_model(self, save_path: Path, step: int):
        """
        モデルの保存
        
        Args:
            save_path (Path): 保存先ディレクトリ
            step (int): 現在のステップ数
        """
        torch.save(self.state_dict(), save_path / f"weight_{step}.pth")
    

class data2vecModel(BaseModel):
    """
    data2vecフレームワークの実装
    
    Args:
        arch (dict[str, Any]): モデルアーキテクチャの設定情報
        ema_decay (float): EMAの減衰率
        ema_end_decay (float): EMAの最終減衰率
        ema_anneal_end_steps (int): EMAの減衰率を最終値に到達させるステップ数
        loss_beta (float): SmoothL1Lossのbeta値
        quantize_cfg (QuantizeConfig): 量子化に関する設定情報
        padding_idx (int): パディングトークンのインデックス
        num_tokens (int): トークンの総数
        experiment_cfg (ExperimentConfig): 実験に関する設定情報
        device: 使用デバイス
    """
    def __init__(
        self,
        arch: dict[str, Any],
        ema_decay: float,
        ema_end_decay: float,
        ema_anneal_end_steps: int,
        loss_beta: float,
        quantize_cfg: QuantizeConfig,
        mlm_loss_cfg: MLMLossConfig,
        padding_idx: int,
        num_tokens: int,
        experiment_cfg: ExperimentConfig,
        device,
        **framework_kwargs
    ):
        super().__init__(padding_idx=padding_idx, num_tokens=num_tokens, device=device)
        
        # quantizeの有無
        self.use_quantize = quantize_cfg.use_quantize
        
        # MLMのlossをdata2vecのlossに追加するかどうか
        self.use_mlm_loss = mlm_loss_cfg.use_mlm_loss
        self.mlm_loss_weight = mlm_loss_cfg.mlm_loss_weight
        
        # 学習に関するパラメータ
        self.num_updates = 0
        self.ema_decay = ema_decay
        self.ema_end_decay = ema_end_decay
        self.ema_anneal_end_steps = ema_anneal_end_steps
        
        # レイヤ数など
        self.n_layers = arch["n_layers"]
        self.k_layers = arch["k_layers"]
        
        # モデルの構築
        from .modules import EMAModule
        self.student_model = CommonModule(arch, self.padding_idx, self.num_tokens, device)
        self.ema = EMAModule(self.student_model, self.ema_decay, device)
        
        # regression headの構築
        curr_dim = arch["embed_dim"]
        projs = []
        for i in range(arch["n_head_layers"] - 1):
            next_dim = arch["embed_dim"] * 2 if i == 0 else curr_dim
            projs.append(nn.Linear(curr_dim, next_dim))
            projs.append(nn.GELU())
            curr_dim = next_dim
        if self.use_quantize:
            projs.append(nn.Linear(curr_dim, quantize_cfg.n_codebooks))
        else:
            projs.append(nn.Linear(curr_dim, arch["embed_dim"]))
        self.regression_head = nn.Sequential(*projs)
        
        # MLMのlossをdata2vecのlossに追加する場合の分類器
        if self.use_mlm_loss:
            self.classifier = nn.Linear(arch["embed_dim"], self.num_tokens)
        
        # 損失関数
        if self.use_quantize:
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = nn.SmoothL1Loss(reduction="none", beta=loss_beta)
        
        # 離散化アリの場合のコードブックと量子化行列
        if self.use_quantize:
            self.register_buffer(
                "codebook",
                F.normalize(
                    torch.randn(quantize_cfg.n_codebooks, quantize_cfg.codebook_dim),
                    p=2,
                    dim=1,
                ),
            )
            self.register_buffer(
                "quant_matrix",
                nn.init.xavier_uniform_(torch.empty(arch["embed_dim"], quantize_cfg.codebook_dim)),
            )
            
        # その他に関する設定
        self.extract_repr_layers= experiment_cfg.extract_repr_layers
        self.to(self.device)
        
    def loss_func(self, logits: torch.Tensor, target: torch.Tensor, logits_mlm: torch.Tensor=None, target_mlm: torch.Tensor=None) -> torch.Tensor:
        """
        損失関数の計算
        
        Args:
            logits (torch.Tensor): モデルの出力ロジット(num_masked_positions, embed_dim)
            target (torch.Tensor): 正解ラベル (num_masked_positions, embed_dim)
            logits_mlm (torch.Tensor | None): MLMロジット (num_masked_positions, vocab_size)
            target_mlm (torch.Tensor | None): MLM正解ラベル (num_masked_positions,)
        Returns:
            torch.Tensor: 計算された損失
        """
        
        loss = torch.tensor(0.0, device=self.device)
        
        # 量子化する場合はここに追記
        if self.use_quantize:
            raise NotImplementedError("Quantization not implemented yet.")
            
        # 量子化しない場合は通常のSmoothL1Lossを計算
        else:
            sz = logits.shape[-1]
            d2v_loss = self.criterion(
                logits.float(),
                target.float(),
            ).sum(dim=-1) / math.sqrt(sz)
        
        # MLMのlossをdata2vecのlossに追加する場合
        if self.use_mlm_loss:
            mlm_loss = F.cross_entropy(logits_mlm, target_mlm)
            loss = (1 - self.mlm_loss_weight) * d2v_loss + self.mlm_loss_weight * mlm_loss
        else:
            loss = d2v_loss
        
        return loss
        
    def forward(
        self,
        student_input: torch.Tensor,
        attn_mask: torch.Tensor,
        teacher_input: torch.Tensor = None,
        attn_biases: torch.Tensor = None,
        attn_biases_masked: torch.Tensor = None,
        masked_idxes: list[list[int]] = None,
        mode: str = "train",
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """
        train, validation, testで共通のforward処理. 
        trainとvalidationでは損失計算まで行い, testでは出力のみ取得する.
        Args:
            student_input (torch.Tensor): 入力テンソル (B, L)
            attn_mask (torch.Tensor): アテンションのパディングマスク (B, 1, L, L)
            teacher_input (torch.Tensor | None): マスクされていない入力テンソル (B, L) (test時はNone)
            attn_biases (torch.Tensor | None): アテンションのバイアス (B, 1, L, L)
            attn_biases_masked (torch.Tensor | None): マスクされた入力に対するアテンションのバイアス (B, 1, L, L)
            masked_idxes (list[list[int]] | None): マスクされた位置のインデックスリスト (B, [mask_positions])
            mode (str): "train", "validation", "test"のいずれか
            
        Returns:
            Union: モードによって返り値が異なる  
            
            - trainの場合: loss (torch.Tensor)  
            - validationの場合: 以下の辞書  
                {  
                "loss": torch.Tensor  
                "var": torch.Tensor (2,)  
                "var_teacher": torch.Tensor (2,)  
                }  
            - testの場合: 以下の辞書  
                {  
                "repr": torch.Tensor (B, L, embed_dim)  
                "attn": torch.Tensor (B, n_layers*n_heads, L, L)  
                "teacher_repr": torch.Tensor (B, L, embed_dim)  
                "teacher_attn": torch.Tensor (B, n_layers*n_heads, L, L)  
                }
        """
        
        assert mode in ["train", "validation", "test"], f"Unsupported mode: {mode}"
        
        # 中間表現を取得するレイヤの指定
        # studnetでは, train時は最終層のみ, test時は全レイヤ
        repr_layers = [self.n_layers] if mode != "test" else list(range(self.n_layers+1))
        
        # student modelの順伝搬
        hidden_reprs = self.student_model(
            student_input,
            attn_mask,
            attn_biases=attn_biases_masked,
            repr_layers=repr_layers,
            return_fc=False,
            test_mode=(mode == "test"),
        )
        
        # 最終層の出力のうち, 配列特徴のみを取得
        x = [x_and_attn[0] for x_and_attn in hidden_reprs][-1]  # (B, L, embed_dim)
        x = F.layer_norm(x, (x.shape[-1],))
        
        # testの場合はextract_repr_layers層の配列特徴と, 全レイヤのアテンションを取得
        if mode == "test":
            repr_output = hidden_reprs[self.extract_repr_layers][0]  # (B, L, embed_dim)
            repr_output = F.layer_norm(repr_output, (repr_output.shape[-1],))
            attn_output = [x_and_attn[1] for x_and_attn in hidden_reprs[1:]] # 最初の層はアテンションを持たない
            attn_output = torch.cat(attn_output, dim=1)  # (B, n_layers*n_heads, L, L)
        
        # teacher modelの順伝搬
        with torch.no_grad():
            self.ema.model.eval()
            
            # teacher modelではK層の中間表現が必要になるため, repr_layersを調整
            if mode != "test":
                repr_layers = list(range(self.n_layers - self.k_layers + 1, self.n_layers + 1)) # 最終K層
            
            # test時はstudent_inputを使用
            hidden_reprs = self.ema.model(
                teacher_input if mode != "test" else student_input,
                attn_mask,
                attn_biases=attn_biases,
                repr_layers=repr_layers,
                return_fc=True,
                test_mode=(mode == "test"),
            )
            
            # 最終K層の中間表現を取得し, レイヤ正規化と平均化
            y = hidden_reprs[-self.k_layers:]
            y = torch.stack([l[0].float() for l in y], dim=0) # (K, B, L, embed_dim)

            y = F.layer_norm(y, (y.shape[-1],))
            y = y.mean(dim=0) # (B, L, embed_dim)
            y = F.layer_norm(y, (y.shape[-1],))
        
        # test時は損失計算を行わずに出力を返す
        if mode == "test":
            teacher_repr = hidden_reprs[self.extract_repr_layers][0]  # (B, L, embed_dim)
            teacher_attn = [y_and_attn[1] for y_and_attn in hidden_reprs[1:]] # 最初の層はアテンションを持たない
            teacher_attn = torch.cat(teacher_attn, dim=1)  # (B, n_layers*n_heads, L, L)
            
            return {
                "repr": repr_output,
                "attn": attn_output,
                "teacher_repr": teacher_repr,
                "teacher_attn": teacher_attn,
            }
            
        # validattion時は分散を計算
        if mode == "validation":
            var = torch.stack([
                torch.var(x.view(-1, x.shape[-1]), dim=0).mean(),
                torch.var(x.view(-1, x.shape[-1]), dim=1).mean(),
                ])
            var_teacher = torch.stack([
                torch.var(y.view(-1, y.shape[-1]), dim=0).mean(),
                torch.var(y.view(-1, y.shape[-1]), dim=1).mean(),
            ])
            
        # 損失計算に向けて, マスクされた位置の特徴を抽出
        masked_idxes_tensor = torch.zeros_like(student_input, dtype=torch.bool)  # (B, L)
        for batch_idx, idxes in enumerate(masked_idxes):
            masked_idxes_tensor[batch_idx, idxes] = True
            
        x = x[masked_idxes_tensor]  # (num_masked_positions, embed_dim)
        y = y[masked_idxes_tensor]  # (num_masked_positions, embed_dim)
        
        # 損失計算
        # 量子化する場合はここに追記
        if self.use_quantize:
            raise NotImplementedError("Quantization not implemented yet.")
        
        # 通常のSmoothL1LossまたはMLMのcross_entropyを計算
        else:
            # regression headの適用
            d2v_logits = self.regression_head(x)  # (num_masked_positions, embed_dim)
            
            if self.use_mlm_loss:
                mlm_logits = self.classifier(x)  # (num_masked_positions, vocab_size)
                loss = self.loss_func(d2v_logits, y, logits_mlm=mlm_logits, target_mlm=teacher_input[masked_idxes_tensor])
            else:
                loss = self.loss_func(d2v_logits, y)
        
        # train時とvalidation時で返り値を分ける
        if mode == "train":
            return loss.mean()
        
        else:  # validation
            assert mode == "validation"
            return {
                "loss": loss.mean(),
                "var": var,
                "var_teacher": var_teacher,
            }
    
    def _train(self, batch):
        """
        訓練用
        Args:
            batch: バッチデータ
        Returns:
            loss: torch.Tensor
        """
        student_input = batch["token_seqs_masked"].to(self.device)
        teacher_input = batch["token_seqs"].to(self.device)
        attn_mask = batch["attn_mask"].to(self.device)
        masked_idxes = batch["masked_idxes"]
        attn_biases = batch["attn_biases"]
        attn_biases_masked = batch["attn_biases_masked"]
        if isinstance(attn_biases, torch.Tensor):
            attn_biases = attn_biases.to(self.device)
            attn_biases_masked = attn_biases_masked.to(self.device)
        
        loss = self(
            student_input,
            attn_mask,
            teacher_input=teacher_input,
            attn_biases=attn_biases,
            attn_biases_masked=attn_biases_masked,
            masked_idxes=masked_idxes,
            mode="train",
        )
        
        return loss
    
    def _validate(self, batch):
        """
        検証用
        Args:
            batch: バッチデータ
        Returns:
            dict:
                - loss: torch.Tensor
                - var: torch.Tensor (2,)
                - var_teacher: torch.Tensor (2,)
        """
        student_input = batch["token_seqs_masked"].to(self.device)
        teacher_input = batch["token_seqs"].to(self.device)
        attn_mask = batch["attn_mask"].to(self.device)
        masked_idxes = batch["masked_idxes"]
        
        attn_biases = batch["attn_biases"]
        attn_biases_masked = batch["attn_biases_masked"]
        if isinstance(attn_biases, torch.Tensor):
            attn_biases = attn_biases.to(self.device)
            attn_biases_masked = attn_biases_masked.to(self.device)
        
        results = self(
            student_input,
            attn_mask,
            teacher_input=teacher_input,
            attn_biases=attn_biases,
            attn_biases_masked=attn_biases_masked,
            masked_idxes=masked_idxes,
            mode="validation",
        )
        
        return results
    
    def _test(self, batch):
        """
        テスト用
        Args:
            batch: バッチデータ
        Returns:
            dict:
                - repr: torch.Tensor (B, L, embed_dim)
                - attn: torch.Tensor (B, n_layers*n_heads, L, L)
                - teacher_repr: torch.Tensor (B, L, embed_dim)
                - teacher_attn: torch.Tensor (B, n_layers*n_heads, L, L)
        """
        student_input = batch["token_seqs"].to(self.device)
        attn_mask = batch["attn_mask"].to(self.device)
        
        results = self(
            student_input,
            attn_mask,
            teacher_input=None,
            masked_idxes=None,
            mode="test",
        )
        
        return results
    
    def set_num_updates(self):
        """
        EMAのステップ数の初期化および更新メソッド
        """
        
        # モデルがtrainingモードかつEMAが有効な場合にのみ更新
        if self.training and self.ema is not None:
            # warmup中はema_decayを線形増加させ, それ以降はema_end_decayに固定
            if self.ema_decay != self.ema_end_decay:
                if self.num_updates >= self.ema_anneal_end_steps:
                    decay = self.ema_end_decay
                else:
                    # 線形に増加させる
                    r = self.ema_end_decay - self.ema_decay
                    pct_remaining = 1 - self.num_updates / self.ema_anneal_end_steps
                    decay = self.ema_end_decay - r * pct_remaining
                self.ema.set_decay(decay)
            
            # EMAの更新
            if self.ema.get_decay() < 1:
                self.ema.step(self.student_model)
        
        # ステップ数の更新
        self.num_updates += 1

    def _step(self):
        """
        学習ステップの更新 (optimizerとlr_schedulerは外部に設定しているのでEMAの更新のみ)
        """
        # EMAとステップ数の更新
        self.set_num_updates()
    
    def save_model(self, save_path: Path, step: int):
        """
        モデルの保存
        
        Args:
            save_path (Path): 保存先ディレクトリ
            step (int): 現在のステップ数
        """
        
        torch.save(self.state_dict(), save_path / f"weight_{step}.pth")
        torch.save(self.ema.model.state_dict(), save_path / f"teacher_weight_{step}.pth")

        