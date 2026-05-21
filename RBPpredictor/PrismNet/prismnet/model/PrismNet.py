from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import hydra  # type: ignore
    from omegaconf import OmegaConf  # type: ignore
except Exception:  # pragma: no cover
    hydra = None
    OmegaConf = None
from .resnet import ResidualBlock1D, ResidualBlock2D
from .se import SEBlock


class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, relu=True, same_padding=False, bn=False):
        super(Conv2d, self).__init__()
        p0 = int((kernel_size[0] - 1) / 2) if same_padding else 0
        p1 = int((kernel_size[1] - 1) / 2) if same_padding else 0
        padding = (p0, p1)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class PrismNet(nn.Module):
    def __init__(
        self,
        mode: str = "pu",
        *,
        structure_source: Literal["shape", "pretrain"] = "shape",
        pretrain_concat_mode: Literal["proj1d", "raw"] = "proj1d",
        pretrain_model_root: str | None = None,
        pretrain_framework: Literal["data2vec", "mlm"] | None = None,
        pretrain_timestamp: str | None = None,
        pretrain_checkpoint: str = "final",
        pretrain_use_teacher: bool = False,
        pretrain_repr_key: str = "repr",
        pretrain_amp: bool = False,
        device: torch.device | str | None = None,
    ):
        super(PrismNet, self).__init__()
        self.mode = mode
        self.structure_source = structure_source
        self.pretrain_repr_key = pretrain_repr_key
        self.pretrain_concat_mode = pretrain_concat_mode
        h_p, h_k = 2, 5 
        if mode=="pu":
            self.n_features = 5
        elif mode=="seq":
            self.n_features = 4
            h_p, h_k = 1, 3 
        elif mode=="str":
            self.n_features = 1
            h_p, h_k = 0, 1
        else:
            raise "mode error"

        # Optional: frozen pretrain model features that replace/augment structure channel.
        # - proj1d: frozen pretrain repr -> trainable Linear(embed_dim->1) as structure-like 1ch
        # - raw:   frozen pretrain repr (embed_dim) concatenated as-is to onehot (no compression)
        self.pretrain_model: nn.Module | None = None
        self.pretrain_proj: nn.Linear | None = None
        self._pretrain_tokens: list[str] | None = None
        self._pretrain_token_to_idx: dict[str, int] | None = None
        self._pretrain_use_additional_token: bool = False
        self._pretrain_embed_dim: int | None = None
        self._pretrain_device = torch.device(device) if device is not None else torch.device("cpu")
        self._pretrain_amp = bool(pretrain_amp)
        self.register_buffer("_pretrain_attn_mask_base", torch.empty(0), persistent=False)

        if self.structure_source == "pretrain":
            if pretrain_framework is None or pretrain_timestamp is None:
                raise ValueError("pretrain_framework and pretrain_timestamp must be provided when structure_source='pretrain'.")

            if pretrain_model_root is None:
                raise ValueError("pretrain_model_root must be provided when structure_source='pretrain'.")

            pretrain_model_dir = Path(pretrain_model_root) / pretrain_framework / pretrain_timestamp
            cfg_path = pretrain_model_dir / "train_config" / ".hydra" / "config.yaml"
            if not cfg_path.exists():
                raise FileNotFoundError(f"Pretrain config not found: {cfg_path}")

            if hydra is None or OmegaConf is None:
                raise ImportError("hydra-core and omegaconf are required to use structure_source='pretrain'.")

            # Lazy import to avoid hard dependency unless requested
            import pretrain.models as PretrainModels  # type: ignore

            # Pretrain configs may rely on custom OmegaConf resolvers (e.g., ${div:...})
            if not OmegaConf._get_resolver("div"):
                OmegaConf.register_new_resolver("div", lambda x, y: int(x / y))
            if not OmegaConf._get_resolver("mul"):
                OmegaConf.register_new_resolver("mul", lambda x, y: x * y)

            pretrain_cfg = OmegaConf.load(str(cfg_path))

            # Compatibility patch: older experiments sometimes store short _target_
            target = getattr(getattr(pretrain_cfg, "framework", None), "_target_", None)
            if target == "models.data2vecModel":
                pretrain_cfg.framework._target_ = "pretrain.models.data2vecModel"
            elif target == "models.MLMModel":
                pretrain_cfg.framework._target_ = "pretrain.models.MLMModel"

            tokens: list[str] = list(pretrain_cfg.dataset.tokens)
            token_to_idx = {t: i for i, t in enumerate(tokens)}

            pretrain_model: PretrainModels.BaseModel = hydra.utils.instantiate(
                pretrain_cfg.framework,
                padding_idx=token_to_idx["<pad>"],
                num_tokens=len(tokens),
                experiment_cfg=pretrain_cfg.experiment,
                device=self._pretrain_device,
            )

            ckpt: Any = pretrain_checkpoint
            if str(ckpt) == "final":
                ckpt = int(pretrain_cfg.common.max_steps)
            weight_name = f"teacher_weight_{ckpt}.pth" if pretrain_use_teacher else f"weight_{ckpt}.pth"
            weight_path = pretrain_model_dir / weight_name
            if not weight_path.exists():
                raise FileNotFoundError(f"Pretrain weight not found: {weight_path}")

            state = torch.load(str(weight_path), map_location=self._pretrain_device)
            pretrain_model._load_state_dict(state)

            pretrain_model.eval()
            for p in pretrain_model.parameters():
                p.requires_grad_(False)

            embed_dim = int(pretrain_cfg.model_size.embed_dim)
            self.pretrain_model = pretrain_model
            self._pretrain_embed_dim = embed_dim
            if self.pretrain_concat_mode == "proj1d":
                self.pretrain_proj = nn.Linear(embed_dim, 1)
            elif self.pretrain_concat_mode == "raw":
                self.pretrain_proj = None
                if self.mode == "pu":
                    self.n_features = 4 + embed_dim
                elif self.mode == "str":
                    self.n_features = embed_dim
            else:
                raise ValueError(f"Unknown pretrain_concat_mode: {self.pretrain_concat_mode}")
            self._pretrain_tokens = tokens
            self._pretrain_token_to_idx = token_to_idx
            self._pretrain_use_additional_token = bool(pretrain_cfg.experiment.use_additional_token)
        
        base_channel = 8
        self.conv    = Conv2d(1, base_channel, kernel_size=(11, h_k), bn = True, same_padding=True)
        self.se      = SEBlock(base_channel)
        self.res2d   = ResidualBlock2D(base_channel, kernel_size=(11, h_k), padding=(5, h_p)) 
        self.res1d   = ResidualBlock1D(base_channel*4) 
        self.avgpool = nn.AvgPool2d((1,self.n_features))
        self.gpool   = nn.AdaptiveAvgPool1d(1)
        self.fc      = nn.Linear(base_channel*4*8, 1)
        self._initialize_weights()

    def train(self, mode: bool = True):
        # Keep frozen pretrain model in eval even during PrismNet training
        super().train(mode)
        if self.pretrain_model is not None:
            self.pretrain_model.eval()
        return self

    def state_dict(self, *args, **kwargs):  # type: ignore[override]
        """Return a state_dict excluding frozen pretrain model weights.

        The pretrain model is treated as an external, fixed dependency. We keep
        PrismNet's own parameters (including the trainable projection
        `pretrain_proj`) but drop `pretrain_model.*` entries to avoid bloated
        checkpoints.
        """
        sd = super().state_dict(*args, **kwargs)
        for k in list(sd.keys()):
            if k.startswith("pretrain_model."):
                sd.pop(k)
        return sd

    def load_state_dict(self, state_dict, strict: bool = True):  # type: ignore[override]
        """Load state_dict while ignoring frozen pretrain model weights.

        - New checkpoints (after this change) do not contain `pretrain_model.*`.
        - Older checkpoints might contain them; we ignore those keys so the
          referenced pretrain model is always loaded from the configured
          pretrain directory, not from the PrismNet checkpoint.
        """
        if isinstance(state_dict, dict):
            filtered = {k: v for k, v in state_dict.items() if not k.startswith("pretrain_model.")}
        else:
            filtered = state_dict

        incompatible = super().load_state_dict(filtered, strict=False)

        missing = [k for k in incompatible.missing_keys if not k.startswith("pretrain_model.")]
        unexpected = [k for k in incompatible.unexpected_keys if not k.startswith("pretrain_model.")]
        if strict and (missing or unexpected):
            raise RuntimeError(
                "Error(s) in loading state_dict for PrismNet:\n"
                + ("\n".join(["\tMissing key(s): " + ", ".join(missing)]) if missing else "")
                + ("\n" if missing and unexpected else "")
                + ("\n".join(["\tUnexpected key(s): " + ", ".join(unexpected)]) if unexpected else "")
            )

        return incompatible

    def _onehot_to_pretrain_tokens(self, onehot: torch.Tensor) -> torch.Tensor:
        """Convert PrismNet onehot (B, 1, L, 4) to pretrain token IDs (B, L)."""
        if self._pretrain_token_to_idx is None:
            raise RuntimeError("Pretrain token mapping is not initialized.")

        # onehot: float tensor
        # argmax gives 0 when all-zeros; fix those as N
        base_idx = onehot.argmax(dim=-1).squeeze(1).to(dtype=torch.long)  # (B, L)
        is_unknown = (onehot.sum(dim=-1).squeeze(1) == 0)

        tok_A = self._pretrain_token_to_idx["A"]
        tok_C = self._pretrain_token_to_idx["C"]
        tok_G = self._pretrain_token_to_idx["G"]
        tok_U = self._pretrain_token_to_idx.get("U")
        if tok_U is None:
            # allow "T" as U equivalent, but default tokens include "U"
            tok_U = self._pretrain_token_to_idx["T"]
        tok_N = self._pretrain_token_to_idx["N"]

        mapping = torch.tensor([tok_A, tok_C, tok_G, tok_U], device=onehot.device, dtype=torch.long)
        token_seqs = mapping[base_idx]  # (B, L)
        token_seqs[is_unknown] = tok_N

        if self._pretrain_use_additional_token:
            tok_cls = self._pretrain_token_to_idx["<cls>"]
            tok_eos = self._pretrain_token_to_idx["<eos>"]
            cls_col = torch.full((token_seqs.shape[0], 1), tok_cls, device=onehot.device, dtype=torch.long)
            eos_col = torch.full((token_seqs.shape[0], 1), tok_eos, device=onehot.device, dtype=torch.long)
            token_seqs = torch.cat([cls_col, token_seqs, eos_col], dim=1)

        return token_seqs

    def _compute_pretrain_features(self, onehot: torch.Tensor) -> torch.Tensor:
        """Compute pretrain-derived features to concatenate with onehot.

        Args:
            onehot: (B, 1, L, 4)
        Returns:
            (B, 1, L, K) where K=1 for proj1d, K=embed_dim for raw
        """
        assert self.pretrain_model is not None
        if self.pretrain_concat_mode == "proj1d":
            assert self.pretrain_proj is not None
        elif self.pretrain_concat_mode == "raw":
            pass
        else:
            raise ValueError(f"Unknown pretrain_concat_mode: {self.pretrain_concat_mode}")

        # Ensure pretrain model uses the same device as input
        device = onehot.device
        if hasattr(self.pretrain_model, "device"):
            setattr(self.pretrain_model, "device", device)

        token_seqs = self._onehot_to_pretrain_tokens(onehot)  # (B, L or L+2)
        Ltok = token_seqs.shape[1]

        # Reuse a cached all-zero attention mask to avoid per-step allocation.
        # Pretrain code expects a floating-point bias mask (B, 1, L, L).
        if self._pretrain_amp and device.type == "cuda":
            amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            amp_dtype = torch.float32

        if (
            self._pretrain_attn_mask_base.numel() == 0
            or self._pretrain_attn_mask_base.shape[-1] != Ltok
            or self._pretrain_attn_mask_base.dtype != amp_dtype
            or self._pretrain_attn_mask_base.device != device
        ):
            self._pretrain_attn_mask_base = torch.zeros((1, 1, Ltok, Ltok), device=device, dtype=amp_dtype)
        attn_mask = self._pretrain_attn_mask_base.expand(token_seqs.shape[0], -1, -1, -1)

        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=amp_dtype)
            if (self._pretrain_amp and device.type == "cuda")
            else torch.autocast(device_type="cpu", enabled=False)
        )
        with torch.no_grad(), autocast_ctx:
            outputs = self.pretrain_model._test({"token_seqs": token_seqs, "attn_mask": attn_mask})

        if self.pretrain_repr_key not in outputs:
            raise KeyError(f"Requested pretrain_repr_key='{self.pretrain_repr_key}' not in outputs keys={list(outputs.keys())}")
        reprs: torch.Tensor = outputs[self.pretrain_repr_key]

        # Remove additional tokens if present
        L = onehot.shape[2]
        if reprs.shape[1] == L + 2:
            reprs = reprs[:, 1:-1, :]
        elif reprs.shape[1] != L:
            raise ValueError(f"Pretrain repr length mismatch: got {reprs.shape[1]}, expected {L} (or {L}+2 with additional tokens).")

        if self.pretrain_concat_mode == "raw":
            feats = reprs.unsqueeze(1)  # (B, 1, L, embed_dim)
        else:
            feats = self.pretrain_proj(reprs).unsqueeze(1)  # (B, 1, L, 1)

        # Keep dtype aligned with onehot to avoid implicit casts in cat/conv.
        return feats.to(dtype=onehot.dtype)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, input):
        """[forward]
        
        Args:
            input ([tensor],N,C,W,H): input features
        """
        # Replace structure channel (index 4) if requested
        if self.structure_source == "pretrain":
            if input.shape[-1] < 4:
                raise ValueError(f"Input last-dim must be >=4 for pretrain tokenization, got {input.shape}")
            onehot = input[:, :, :, :4]
            pretrain_feats = self._compute_pretrain_features(onehot)
            input = torch.cat([onehot, pretrain_feats], dim=-1)

        if self.mode=="seq":
            input = input[:,:,:,:4]
        elif self.mode=="str":
            input = input[:,:,:,4:]
        x = self.conv(input)
        x = F.dropout(x, 0.1, training=self.training)
        z = self.se(x)
        x = self.res2d(x*z)
        x = F.dropout(x, 0.5, training=self.training)
        x = self.avgpool(x)
        x = x.view(x.shape[0], x.shape[1], x.shape[2])
        x = self.res1d(x)
        x = F.dropout(x, 0.3, training=self.training)
        x = self.gpool(x)
        x = x.view(x.shape[0], x.shape[1])
        x = self.fc(x)
        return x


class PrismNet_large(nn.Module):
    def __init__(
        self,
        mode: str = "pu",
        *,
        structure_source: Literal["shape", "pretrain"] = "shape",
        pretrain_concat_mode: Literal["proj1d", "raw"] = "proj1d",
        pretrain_model_root: str | None = None,
        pretrain_framework: Literal["data2vec", "mlm"] | None = None,
        pretrain_timestamp: str | None = None,
        pretrain_checkpoint: str = "final",
        pretrain_use_teacher: bool = False,
        pretrain_repr_key: str = "repr",
        pretrain_amp: bool = False,
        device: torch.device | str | None = None,
    ):
        super(PrismNet_large, self).__init__()
        self.mode = mode
        self.structure_source = structure_source
        self.pretrain_repr_key = pretrain_repr_key
        self.pretrain_concat_mode = pretrain_concat_mode
        h_p, h_k = 2, 5 
        if mode=="pu":
            self.n_features = 5
        elif mode=="seq":
            self.n_features = 4
            h_p, h_k = 1, 3 
        elif mode=="str":
            self.n_features = 1
            h_p, h_k = 0, 1
        else:
            raise "mode error"

        self.pretrain_model: nn.Module | None = None
        self.pretrain_proj: nn.Linear | None = None
        self._pretrain_tokens: list[str] | None = None
        self._pretrain_token_to_idx: dict[str, int] | None = None
        self._pretrain_use_additional_token: bool = False
        self._pretrain_embed_dim: int | None = None
        self._pretrain_device = torch.device(device) if device is not None else torch.device("cpu")
        self._pretrain_amp = bool(pretrain_amp)
        self.register_buffer("_pretrain_attn_mask_base", torch.empty(0), persistent=False)

        if self.structure_source == "pretrain":
            if pretrain_framework is None or pretrain_timestamp is None:
                raise ValueError("pretrain_framework and pretrain_timestamp must be provided when structure_source='pretrain'.")

            if pretrain_model_root is None:
                raise ValueError("pretrain_model_root must be provided when structure_source='pretrain'.")

            pretrain_model_dir = Path(pretrain_model_root) / pretrain_framework / pretrain_timestamp
            cfg_path = pretrain_model_dir / "train_config" / ".hydra" / "config.yaml"
            if not cfg_path.exists():
                raise FileNotFoundError(f"Pretrain config not found: {cfg_path}")

            if hydra is None or OmegaConf is None:
                raise ImportError("hydra-core and omegaconf are required to use structure_source='pretrain'.")

            import pretrain.models as PretrainModels  # type: ignore

            if not OmegaConf._get_resolver("div"):
                OmegaConf.register_new_resolver("div", lambda x, y: int(x / y))
            if not OmegaConf._get_resolver("mul"):
                OmegaConf.register_new_resolver("mul", lambda x, y: x * y)

            pretrain_cfg = OmegaConf.load(str(cfg_path))
            target = getattr(getattr(pretrain_cfg, "framework", None), "_target_", None)
            if target == "models.data2vecModel":
                pretrain_cfg.framework._target_ = "pretrain.models.data2vecModel"
            elif target == "models.MLMModel":
                pretrain_cfg.framework._target_ = "pretrain.models.MLMModel"

            tokens: list[str] = list(pretrain_cfg.dataset.tokens)
            token_to_idx = {t: i for i, t in enumerate(tokens)}

            pretrain_model: PretrainModels.BaseModel = hydra.utils.instantiate(
                pretrain_cfg.framework,
                padding_idx=token_to_idx["<pad>"],
                num_tokens=len(tokens),
                experiment_cfg=pretrain_cfg.experiment,
                device=self._pretrain_device,
            )

            ckpt: Any = pretrain_checkpoint
            if str(ckpt) == "final":
                ckpt = int(pretrain_cfg.common.max_steps)
            weight_name = f"teacher_weight_{ckpt}.pth" if pretrain_use_teacher else f"weight_{ckpt}.pth"
            weight_path = pretrain_model_dir / weight_name
            if not weight_path.exists():
                raise FileNotFoundError(f"Pretrain weight not found: {weight_path}")

            state = torch.load(str(weight_path), map_location=self._pretrain_device)
            pretrain_model._load_state_dict(state)

            pretrain_model.eval()
            for p in pretrain_model.parameters():
                p.requires_grad_(False)

            embed_dim = int(pretrain_cfg.model_size.embed_dim)
            self.pretrain_model = pretrain_model
            self._pretrain_embed_dim = embed_dim
            if self.pretrain_concat_mode == "proj1d":
                self.pretrain_proj = nn.Linear(embed_dim, 1)
            elif self.pretrain_concat_mode == "raw":
                self.pretrain_proj = None
                if self.mode == "pu":
                    self.n_features = 4 + embed_dim
                elif self.mode == "str":
                    self.n_features = embed_dim
            else:
                raise ValueError(f"Unknown pretrain_concat_mode: {self.pretrain_concat_mode}")
            self._pretrain_tokens = tokens
            self._pretrain_token_to_idx = token_to_idx
            self._pretrain_use_additional_token = bool(pretrain_cfg.experiment.use_additional_token)
        
        base_channel = 64
        self.conv    = Conv2d(1, base_channel, kernel_size=(11, h_k), bn = True, same_padding=True)
        self.se      = SEBlock(base_channel)
        self.res2d   = ResidualBlock2D(base_channel, kernel_size=(11, h_k), padding=(5, h_p)) 
        self.res1d   = ResidualBlock1D(base_channel*4) 
        self.avgpool = nn.AvgPool2d((1,self.n_features))
        self.gpool   = nn.AdaptiveAvgPool1d(1)
        self.fc      = nn.Linear(base_channel*4*8, 1)
        self._initialize_weights()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.pretrain_model is not None:
            self.pretrain_model.eval()
        return self

    def state_dict(self, *args, **kwargs):  # type: ignore[override]
        sd = super().state_dict(*args, **kwargs)
        for k in list(sd.keys()):
            if k.startswith("pretrain_model."):
                sd.pop(k)
        return sd

    def load_state_dict(self, state_dict, strict: bool = True):  # type: ignore[override]
        if isinstance(state_dict, dict):
            filtered = {k: v for k, v in state_dict.items() if not k.startswith("pretrain_model.")}
        else:
            filtered = state_dict

        incompatible = super().load_state_dict(filtered, strict=False)

        missing = [k for k in incompatible.missing_keys if not k.startswith("pretrain_model.")]
        unexpected = [k for k in incompatible.unexpected_keys if not k.startswith("pretrain_model.")]
        if strict and (missing or unexpected):
            raise RuntimeError(
                "Error(s) in loading state_dict for PrismNet_large:\n"
                + ("\n".join(["\tMissing key(s): " + ", ".join(missing)]) if missing else "")
                + ("\n" if missing and unexpected else "")
                + ("\n".join(["\tUnexpected key(s): " + ", ".join(unexpected)]) if unexpected else "")
            )

        return incompatible

    def _onehot_to_pretrain_tokens(self, onehot: torch.Tensor) -> torch.Tensor:
        if self._pretrain_token_to_idx is None:
            raise RuntimeError("Pretrain token mapping is not initialized.")

        base_idx = onehot.argmax(dim=-1).squeeze(1).to(dtype=torch.long)  # (B, L)
        is_unknown = (onehot.sum(dim=-1).squeeze(1) == 0)

        tok_A = self._pretrain_token_to_idx["A"]
        tok_C = self._pretrain_token_to_idx["C"]
        tok_G = self._pretrain_token_to_idx["G"]
        tok_U = self._pretrain_token_to_idx.get("U")
        if tok_U is None:
            tok_U = self._pretrain_token_to_idx["T"]
        tok_N = self._pretrain_token_to_idx["N"]

        mapping = torch.tensor([tok_A, tok_C, tok_G, tok_U], device=onehot.device, dtype=torch.long)
        token_seqs = mapping[base_idx]
        token_seqs[is_unknown] = tok_N

        if self._pretrain_use_additional_token:
            tok_cls = self._pretrain_token_to_idx["<cls>"]
            tok_eos = self._pretrain_token_to_idx["<eos>"]
            cls_col = torch.full((token_seqs.shape[0], 1), tok_cls, device=onehot.device, dtype=torch.long)
            eos_col = torch.full((token_seqs.shape[0], 1), tok_eos, device=onehot.device, dtype=torch.long)
            token_seqs = torch.cat([cls_col, token_seqs, eos_col], dim=1)

        return token_seqs

    def _compute_pretrain_features(self, onehot: torch.Tensor) -> torch.Tensor:
        assert self.pretrain_model is not None
        if self.pretrain_concat_mode == "proj1d":
            assert self.pretrain_proj is not None
        elif self.pretrain_concat_mode == "raw":
            pass
        else:
            raise ValueError(f"Unknown pretrain_concat_mode: {self.pretrain_concat_mode}")

        device = onehot.device
        if hasattr(self.pretrain_model, "device"):
            setattr(self.pretrain_model, "device", device)

        token_seqs = self._onehot_to_pretrain_tokens(onehot)
        Ltok = token_seqs.shape[1]

        if self._pretrain_amp and device.type == "cuda":
            amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            amp_dtype = torch.float32

        if (
            self._pretrain_attn_mask_base.numel() == 0
            or self._pretrain_attn_mask_base.shape[-1] != Ltok
            or self._pretrain_attn_mask_base.dtype != amp_dtype
            or self._pretrain_attn_mask_base.device != device
        ):
            self._pretrain_attn_mask_base = torch.zeros((1, 1, Ltok, Ltok), device=device, dtype=amp_dtype)
        attn_mask = self._pretrain_attn_mask_base.expand(token_seqs.shape[0], -1, -1, -1)

        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=amp_dtype)
            if (self._pretrain_amp and device.type == "cuda")
            else torch.autocast(device_type="cpu", enabled=False)
        )
        with torch.no_grad(), autocast_ctx:
            outputs = self.pretrain_model._test({"token_seqs": token_seqs, "attn_mask": attn_mask})

        if self.pretrain_repr_key not in outputs:
            raise KeyError(
                f"Requested pretrain_repr_key='{self.pretrain_repr_key}' not in outputs keys={list(outputs.keys())}"
            )
        reprs: torch.Tensor = outputs[self.pretrain_repr_key]

        L = onehot.shape[2]
        if reprs.shape[1] == L + 2:
            reprs = reprs[:, 1:-1, :]
        elif reprs.shape[1] != L:
            raise ValueError(
                f"Pretrain repr length mismatch: got {reprs.shape[1]}, expected {L} (or {L}+2 with additional tokens)."
            )

        if self.pretrain_concat_mode == "raw":
            feats = reprs.unsqueeze(1)
        else:
            feats = self.pretrain_proj(reprs).unsqueeze(1)
        return feats.to(dtype=onehot.dtype)

    def _compute_pretrain_structure_channel(self, onehot: torch.Tensor) -> torch.Tensor:
        # Backward-compat shim: kept for older call sites (should not be used in new code).
        return self._compute_pretrain_features(onehot)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, input):
        """[summary]
        
        Args:
            input ([tensor],N,C,W,H): input features
        """
        if self.structure_source == "pretrain":
            if input.shape[-1] < 4:
                raise ValueError(f"Input last-dim must be >=4 for pretrain tokenization, got {input.shape}")
            onehot = input[:, :, :, :4]
            pretrain_feats = self._compute_pretrain_features(onehot)
            input = torch.cat([onehot, pretrain_feats], dim=-1)

        if self.mode=="seq":
            input = input[:,:,:,:4]
        elif self.mode=="str":
            input = input[:,:,:,4:]
        x = self.conv(input)    # 
        x = F.dropout(x, 0.1, training=self.training)
        z = self.se(x)
        x = self.res2d(x*z)
        x = F.dropout(x, 0.5, training=self.training)
        x = self.avgpool(x)
        x = x.view(x.shape[0], x.shape[1], x.shape[2])
        x = self.res1d(x)
        x = F.dropout(x, 0.3, training=self.training)
        x = self.gpool(x)
        x = x.view(x.shape[0], x.shape[1])
        x = self.fc(x)
        return x
