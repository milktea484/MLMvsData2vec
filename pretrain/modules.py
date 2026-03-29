import copy
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import _LRScheduler


class CosineScheduler(_LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int = 10000,
        total_steps: int = 200000,
        max_lr: float = 1e-5,
        min_lr: float = 1e-6,
        last_iter: int = -1,
    ):
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.max_lr = max_lr
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch=last_iter)

    def get_lr(self):
        last_iter = self.last_epoch
        if last_iter < self.warmup_steps:
            return [base_lr * last_iter / self.warmup_steps for base_lr in self.base_lrs]
        
        if last_iter > self.total_steps:
            return [base_lr * self.min_lr / self.max_lr for base_lr in self.base_lrs]
        
        decay_ratio = (last_iter - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        coeff = 0.5 * (1.0 + np.cos(np.pi * decay_ratio))
        
        return [self.min_lr + coeff * (base_lr - self.min_lr) for base_lr in self.base_lrs]


class RotaryPositionalEmbedding(nn.Module):
    """
    RoPE (Rotary Positional Embedding) の実装
    
    Args:
        max_length (int): 最大シーケンス長
        head_size (int): ヘッドあたりの埋め込み次元数
        device: 使用デバイス
    """
    
    def __init__(self, max_length: int, head_size: int, device=None):
        super().__init__()
        # パラメータの保存
        assert head_size % 2 == 0, "head_size must be even number for RoPE"
        self.head_size = head_size

        # RoPEの計算に使用するcosineとsineの事前計算
        theta = 1.0 / (10000 ** (torch.arange(0, head_size, 2, device=device) / head_size))
        pos = torch.arange(max_length, device=device)
        x = torch.outer(pos, theta).repeat(1, 2)
        self.register_buffer(
            "rope_cos", torch.cos(x), persistent=False
        )  # (max_length, head_size)
        self.register_buffer(
            "rope_sin", torch.sin(x), persistent=False
        )  # (max_length, head_size)
        
        self.to(device)
        
    def forward(self, x: torch.Tensor, seq_length: int) -> torch.Tensor:
        """
        RoPEの順伝搬処理
        
        Args:
            x (torch.Tensor): 入力テンソル (B, n_heads, L, head_size)
            seq_length (int): シーケンス長
        
        Returns:
            torch.Tensor: RoPEが適用されたテンソル (B, n_heads, L, head_size)
        """
        
        # RoPEの適用
        rope_cos = self.rope_cos[:seq_length]
        rope_sin = self.rope_sin[:seq_length]
        
        x1 = x[..., : self.head_size // 2]
        x2 = x[..., self.head_size // 2 :]
        rotated = torch.cat((-x2, x1), dim=-1)
        
        roped = (x * rope_cos) + (rotated * rope_sin)
        
        return roped.type_as(x)  # (B, n_head, L, head_size)


class TransformerLayer(nn.Module):
    """
    Transformer層
    Args:
        arch (dict[str, Any]): モデルアーキテクチャの設定情報
        device: 使用デバイス
    """
    def __init__(self, arch: dict[str, Any], device=None):
        super().__init__()
        
        # モデルサイズの設定
        self.n_heads = arch["n_heads"]
        self.embed_dim = arch["embed_dim"]
        self.ffn_dim = arch["ffn_dim"]
        self.attention_dropout = arch["attention_dropout"]
        
        # RoPE
        self.position_embedding = RotaryPositionalEmbedding(
            max_length=arch["rope_max_length"],
            head_size=arch["embed_dim"] // arch["n_heads"],
            device=device,
        )
        
        # アテンション層
        self.layer_norm1 = nn.LayerNorm(arch["embed_dim"])
        self.layer_norm2 = nn.LayerNorm(arch["embed_dim"])
        self.scale = (arch["embed_dim"] // arch["n_heads"]) ** -0.5
        self.c_attn = nn.Linear(arch["embed_dim"], 3 * arch["embed_dim"])
        self.proj_attn = nn.Linear(arch["embed_dim"], arch["embed_dim"])
        
        # FFN
        self.fc1 = nn.Linear(arch["embed_dim"], arch["ffn_dim"])
        self.fc2 = nn.Linear(arch["ffn_dim"], arch["embed_dim"])
        self.activation = nn.GELU()
        
        self.device = device
        self.to(device)
        
    def manual_scaled_dot_product_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: torch.Tensor,
        dropout_p: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        attention weightのlogitを取得するためのscaled dot product attentionの実装
        
        Args:
            query (torch.Tensor): クエリテンソル (B, n_heads, L, head_size)
            key (torch.Tensor): キーテンソル (B, n_heads, L, head_size)
            value (torch.Tensor): バリューテンソル (B, n_heads, L, head_size)
            attn_mask (torch.Tensor): バイアスを含んだアテンションマスク (B, 1, L, L)
            dropout_p (float): ドロップアウト確率
        Returns:
            tuple:
            attention出力テンソル (B, n_heads, L, head_size), attention weight (B, n_heads, L, L), attention weightのlogit (B, n_heads, L, L)
        """
        
        query = query * self.scale
        attn_logits = torch.matmul(query, key.transpose(-2, -1))  # (B, n_heads, L, L)
        attn_logits = attn_logits + attn_mask  # (B, n_heads, L, L)
        attn = attn_logits.softmax(dim=-1)  # (B, n_heads, L, L)
        attn = torch.dropout(attn, dropout_p, train=self.training)
        out = torch.matmul(attn, value)  # (B, n_heads, L, head_size)
        
        return out, attn, attn_logits
    
    def original_scaled_dot_product_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: torch.Tensor,
        dropout_p: float = 0.0,
        return_attn: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        PyTorch組み込みのscaled dot product attentionを使用する関数.
        attention weightのlogitは取得できない.
        Args:
            query (torch.Tensor): クエリテンソル (B, n_heads, L, head_size)
            key (torch.Tensor): キーテンソル (B, n_heads, L, head_size)
            value (torch.Tensor): バリューテンソル (B, n_heads, L, head_size)
            attn_mask (torch.Tensor): バイアスを含んだアテンションマスク (B, 1, L, L)
            dropout_p (float): ドロップアウト確率
            return_attn (bool): attention weightを返すかどうか
        Returns:
            tuple:
            attention出力テンソル (B, n_heads, L, head_size), attention weight (B, n_heads, L, L) | None
        """
        L = query.shape[2]
        
        out = F.scaled_dot_product_attention(
            query=query,
            key=key,
            value=value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
        )
        
        if return_attn:
            v_identity = torch.eye(L, device=self.device).unsqueeze(0).unsqueeze(0) # (1, 1, L, L)
            attn = F.scaled_dot_product_attention(
                query=query,
                key=key,
                value=v_identity,
                attn_mask=attn_mask,
                dropout_p=0.0,
            )  # (B, n_heads, L, L)
        
            return out, attn
        else:
            return out, None
        
    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor,
        return_fc: bool = False,
        test_mode: bool = False,
        use_ernie_rna: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        """
        Transformer層の順伝搬処理
        
        Args:
            x (torch.Tensor): 入力テンソル (B, L, embed_dim)
            attn_mask (torch.Tensor): バイアスが付加されたアテンションマスク (B, 1, L, L)
            return_fc (bool): 残差接続前の中間表現を返すかどうか
            test_mode (bool): テストモードかどうか
            use_ernie_rna (bool): ERNIE-RNAの戦略を使用するかどうか
        
        Returns:
            tuple:
            出力テンソル (B, L, embed_dim),  
            残差接続前の中間表現 (B, L, embed_dim) | None,  
            attention weight (B, n_layers*n_heads, L, L),  
            attentionのlogits(B, n_layers*n_heads, L, L) | None
        """
        
        B, L, E = x.shape
        
        # 残差接続と正規化
        residual = x
        x = self.layer_norm1(x)
        
        # qkvの計算と分割
        qkv = self.c_attn(x)  # (B, L, 3*embed_dim)
        q, k, v = qkv.split(self.embed_dim, dim=-1) # 各々 (B, L, embed_dim)
        q = q.view(B, L, self.n_heads, E // self.n_heads).transpose(1, 2)  # (B, n_heads, L, head_size)
        k = k.view(B, L, self.n_heads, E // self.n_heads).transpose(1, 2)  # (B, n_heads, L, head_size)
        v = v.view(B, L, self.n_heads, E // self.n_heads).transpose(1, 2)  # (B, n_heads, L, head_size)
        
        # RoPEの適用
        q = self.position_embedding(q, L)
        k = self.position_embedding(k, L)
        
        # self-attention計算
        assert attn_mask.is_floating_point(), \
            "attn_mask must be float type"
        
        attn, attn_logits = None, None
        if use_ernie_rna:
            # ERNIE-RNAの戦略を使用する場合, attention weightのlogitを取得する必要がある
            x, attn, attn_logits = self.manual_scaled_dot_product_attention(
                query=q,
                key=k,
                value=v,
                attn_mask=attn_mask,
                dropout_p=self.attention_dropout if self.training else 0.0
            )
            if not test_mode:
                attn = None  # テストモード以外ではattention weightは返さない
        else:
            # 通常のscaled dot product attention
            x, attn = self.original_scaled_dot_product_attention(
                query=q,
                key=k,
                value=v,
                attn_mask=attn_mask,
                dropout_p=self.attention_dropout if self.training and not test_mode else 0.0,
                return_attn=test_mode,
            )
            
        x = x.transpose(1, 2).contiguous().view(B, L, E)  # (B, L, embed_dim)
        
        # アテンション出力の線形変換と残差接続
        x = self.proj_attn(x)  # (B, L, embed_dim)
        x = residual + x  # 残差接続
        
        # FFN
        residual = x
        x = self.layer_norm2(x)
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        
        # 残差接続前の中間表現
        fc_result = x if return_fc else None
        
        x = residual + x  # 残差接続
        
        return x, fc_result, attn, attn_logits
        

class CommonModule(nn.Module):
    """
    MLMとdata2vecで共通のモジュール
    
    Args:
        arch (dict[str, Any]): モデルアーキテクチャの設定情報
        padding_idx: パディングトークンのインデックス
        num_tokens: トークンの総数
        device: 使用デバイス
    """
    def __init__(self, arch: dict[str, Any], padding_idx: int, num_tokens: int, device):
        super().__init__()
        self.device = device
        
        # 埋め込み層
        self.embedding = nn.Embedding(
            num_embeddings=num_tokens,
            embedding_dim=arch["embed_dim"],
            padding_idx=padding_idx,
        )
        
        # レイヤ正規化
        self.layer_norm = nn.LayerNorm(arch["embed_dim"])
        
        # Transformer層
        self.layers = nn.ModuleList(
            [TransformerLayer(arch=arch,device=device,) for _ in range(arch["n_layers"])]
        )
        
        self.to(device)
        
    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor,
        attn_biases: torch.Tensor | None = None,
        repr_layers: list[int] = None,
        return_fc: bool = False,
        test_mode: bool = False,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """
        MLMやdata2vecで共通の順伝搬処理
        Args:
            x (torch.Tensor): 入力テンソル (B, L)
            attn_mask (torch.Tensor): バイアスが付加されたアテンションマスク (B, 1, L, L)
            attn_biases (torch.Tensor | None): attentionバイアス (B, 1, L, L) | None
            repr_layers (list[int]): 取得する中間表現のレイヤリスト. Noneの場合は全レイヤ
            return_fc (bool): 残差接続前の中間表現を返すかどうか
            test_mode (bool): テストモード
        Returns:
            hidden_reprs (list[tuple[torch.Tensor, torch.Tensor]]): 各層の出力とアテンションのタプルリスト（正規化無し）  
            repr_layersで指定された層の出力のみを含む (指定されていない場合は全層分).  
            indexとしては, embedding層の出力を0とし, 最終層をn_layersとする.
        """
        # ERNIE-RNAのバイアスを加えるかどうか
        use_ernie_rna = attn_biases is not None
        
        # 埋め込み層
        x = self.embedding(x) # (B, L, embed_dim)

        # レイヤ正規化
        x = self.layer_norm(x) # (B, L, embed_dim)
        
        # 埋め込み層の出力を保存
        hidden_reprs = []
        if (repr_layers is None) or (0 in repr_layers):
            hidden_reprs.append((x, None))
        
        # アテンションマスクにERNIE-RNAのバイアスを加える (初期層)
        if use_ernie_rna:
            attn_mask_and_bias = attn_mask + attn_biases  # (B, 1, L, L)
        else:
            attn_mask_and_bias = attn_mask
        
        # Transformer層の順伝搬
        for layer_idx, layer in enumerate(self.layers):
            x, fc_result, attn, attn_logits = layer(
                x,
                attn_mask=attn_mask_and_bias,
                return_fc=return_fc,
                test_mode=test_mode,
                use_ernie_rna=use_ernie_rna,
            )
            
            # 指定された層の出力を保存. 指定されていない場合は全層保存
            if (repr_layers is None) or ((layer_idx + 1) in repr_layers):
                if return_fc:
                    hidden_reprs.append((fc_result, attn))
                else:
                    hidden_reprs.append((x, attn))
                    
            # ERNIE-RNA戦略に則り, 前層のattentionを次層のバイアスとする
            if use_ernie_rna:
                attn_mask_and_bias = attn_logits  # (B, 1, L, L)
        
        return hidden_reprs
    
    
class EMAModule():
    """
    data2vecのteacherモデルで使用するEMAModule
    fairseqの実装から必要な部分のみ抜粋
    
    Args:
        model: student model (CommonModule)
        ema_decay: EMAの減衰率
        device: 使用デバイス
        skip_keys: student modelのparameterのうち, 追跡をしない部分 (基本はNone)
    """
    def __init__(
        self, model: CommonModule,
        ema_decay: float = 0.9999,
        device=None,
        skip_keys: list[str] | None = None,
    ):
        # modelはstudent modelを渡す
        self.model = copy.deepcopy(model)
        self.model.requires_grad_(False)
        self.model.to(device)
        self.decay = ema_decay
        self.skip_keys = (skip_keys or set())
        
    def set_decay(self, decay: float):
        """EMAの減衰率を設定"""
        self.decay = decay
    
    def get_decay(self) -> float:
        """EMAの減衰率を取得"""
        return self.decay
    
    def _step_internal(self, new_model: nn.Module):
        """
        new_model (student model) のパラメータを用いてEMA更新を行う内部関数
        """
        decay = self.decay
        
        # 更新後のteacherモデルのパラメータ
        ema_state_dict = {}
        
        # 更新前のteacherモデルのパラメータ
        ema_params = self.model.state_dict()

        # EMA更新の実行
        for key, param in new_model.named_parameters():
            
            # パラメータのうち,dict型のものは無視 (入れ子なので)
            if isinstance(param, dict):
                continue
            
            try:
                ema_param = ema_params[key]
            except KeyError:
                ema_param = (
                    param.float().clone() if param.ndim == 1 else copy.deepcopy(param)
                )
                ema_params[key] = ema_param

            if param.shape != ema_param.shape:
                raise ValueError(
                    "incompatible tensor shapes between model param and ema param"
                    + "{} vs. {}".format(param.shape, ema_param.shape)
                )

            if "version" in key:
                # Do not decay a model.version pytorch param
                continue

            # skip_keysに含まれるか, 学習不要なパラメータはコピーのみ
            if key in self.skip_keys or not param.requires_grad:
                ema_params[key].copy_(param.to(dtype=ema_param.dtype).data)
                ema_param = ema_params[key]
            else:
                ema_param.mul_(decay)
                ema_param.add_(param.data.to(dtype=ema_param.dtype), alpha=1 - decay)

            ema_state_dict[key] = ema_param

        # バッファのものも更新
        for key, param in new_model.named_buffers():
            ema_state_dict[key] = param

        self.model.load_state_dict(ema_state_dict, strict=False)
    
    @torch.no_grad()
    def step(self, new_model: nn.Module):
        """student modelのパラメータを用いてEMA更新を行う"""
        self._step_internal(new_model)
        
    def reverse(self, model: nn.Module):
        """
        Load the model parameters from EMA model.
        Useful for inference or fine-tuning from the EMA model.
        """
        d = self.model.state_dict()
        if "_ema" in d:
            del d["_ema"]

        model.load_state_dict(d, strict=False)
        return model