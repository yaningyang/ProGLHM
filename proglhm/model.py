# Copyright (c) 2026 ProGLHM contributors. All rights reserved.
"""ProGLHM classifier for frozen Nucleotide Transformer v2 embeddings."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


MODEL_NAME = "ProGLHM"
MODEL_ARCHITECTURE_VERSION = "proglhm-v1-ntv2-clean128-shared-bimamba"
GLOBAL_DIMENSION = 64
MAMBA_D_STATE = 16
MAMBA_D_CONV = 4
MAMBA_EXPAND = 2
ALPHA_SCALE_MAX = 0.20
ALPHA_INITIAL_VALUE = 0.05
ALPHA_RAW_INITIAL_VALUE = -1.0986122886681098
LAMBDA_SCALE_MAX = 0.05
LAMBDA_INITIAL_VALUE = 0.01
LAMBDA_RAW_INITIAL_VALUE = -1.3862943611198906
FILM_OUTPUT_WEIGHT_STD = 1e-3


def validate_inputs(data):
    """Validate a nonempty batch with real tokens followed by right padding."""
    embeddings, valid_mask = data
    if embeddings.ndim != 3 or valid_mask.ndim != 2:
        raise ValueError(
            "Expected embeddings [batch, tokens, 1024] and mask [batch, tokens]"
        )
    if embeddings.shape[:2] != valid_mask.shape:
        raise ValueError(
            f"Embedding/mask shape mismatch: {embeddings.shape} vs {valid_mask.shape}"
        )
    if embeddings.shape[-1] != 1024:
        raise ValueError("Expected 1024 NT-v2 features per token")
    if embeddings.shape[0] < 1:
        raise ValueError("Each batch must contain at least one sequence")
    if not 8 <= embeddings.shape[1] <= 128 or embeddings.shape[1] % 8:
        raise ValueError("Token dimension must be a multiple of 8 between 8 and 128")
    if embeddings.device != valid_mask.device:
        raise ValueError("Embeddings and masks must be on the same device")
    valid_mask = valid_mask.bool()
    if not torch.all(valid_mask.any(dim=1)):
        raise ValueError("Each sequence must contain at least one valid token")
    seen_padding = (~valid_mask).cumsum(dim=1) > 0
    if torch.any(valid_mask & seen_padding):
        raise ValueError("Mask must contain contiguous real tokens then right padding")
    return embeddings, valid_mask


def _make_mamba(d_model, d_state, d_conv, expand):
    try:
        from mamba_ssm import Mamba
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "ProGLHM requires mamba-ssm==1.2.0.post1 and "
            "causal-conv1d==1.2.0.post2 with compatible PyTorch/CUDA libraries. "
            "A Linux/NVIDIA CUDA environment is required."
        ) from exc
    return Mamba(
        d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand
    )


def _reverse_valid_prefix(values, valid_mask):
    """Reverse valid-token prefixes while keeping padding on the right."""
    if values.ndim != 3 or valid_mask.ndim != 2:
        raise ValueError("Expected values [B,L,D] and mask [B,L]")
    if values.shape[:2] != valid_mask.shape:
        raise ValueError("Value/mask shapes do not match")
    valid_mask = valid_mask.bool()
    lengths = valid_mask.sum(dim=1)
    positions = torch.arange(values.shape[1], device=values.device)
    positions = positions.unsqueeze(0).expand(values.shape[0], -1)
    reversed_positions = lengths.unsqueeze(1) - 1 - positions
    gather_positions = torch.where(
        valid_mask, reversed_positions, positions
    ).clamp_min(0)
    reversed_values = values.gather(
        1, gather_positions.unsqueeze(-1).expand_as(values)
    )
    return reversed_values.masked_fill(~valid_mask.unsqueeze(-1), 0.0)



class _LocalBackbone(nn.Module):
    """CNN compression, Transformer encoding and attention readout."""

    pool_count = 3

    def __init__(
        self,
        ntv2_hidden_size=1024,
        target_embedding_dim=256,
        num_classes=2,
        hidden_size=64,
        num_transformer_layers=2,
        attention_dim=64,
        dropout_rate=0.3,
        cnn_dim=128,
        cnn_layer_num=3,
        cnn_kernel_size=8,
        fc_dim=128,
        use_two_stage_proj=True,
    ):
        super().__init__()
        if cnn_layer_num != 3:
            raise ValueError("The formal architecture requires exactly three CNNs")
        self.use_two_stage_proj = use_two_stage_proj
        if use_two_stage_proj:
            self.ntv2_proj1 = nn.Linear(ntv2_hidden_size, 512)
            self.ntv2_proj2 = nn.Linear(512, target_embedding_dim)
            nn.init.xavier_uniform_(self.ntv2_proj1.weight)
            nn.init.zeros_(self.ntv2_proj1.bias)
            nn.init.xavier_uniform_(self.ntv2_proj2.weight)
            nn.init.zeros_(self.ntv2_proj2.bias)
        else:
            self.ntv2_proj_single = nn.Linear(ntv2_hidden_size, target_embedding_dim)
            nn.init.xavier_uniform_(self.ntv2_proj_single.weight)
            nn.init.zeros_(self.ntv2_proj_single.bias)

        self.norm = nn.LayerNorm(target_embedding_dim)
        self.embedding_dim = target_embedding_dim
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.num_transformer_layers = num_transformer_layers
        self.attention_dim = attention_dim
        self.dropout = dropout_rate
        self.cnn_dim = cnn_dim
        self.cnn_layer_num = cnn_layer_num
        self.cnn_kernel_size = cnn_kernel_size
        self.fc_dim = fc_dim
        self.cnn_layers = nn.ModuleList()
        for index in range(self.cnn_layer_num):
            in_channels = (
                self.embedding_dim
                if index == 0
                else int(self.cnn_dim / (2 ** (index - 1)))
            )
            out_channels = int(self.cnn_dim / (2 ** index))
            self.cnn_layers.append(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=self.cnn_kernel_size,
                    padding="same",
                )
            )

        final_channels = int(self.cnn_dim / (2 ** (self.cnn_layer_num - 1)))
        self.maxpool_layer = nn.MaxPool1d(kernel_size=2)
        self.fc_cnn_to_transformer = nn.Linear(final_channels, self.hidden_size)
        nhead = 4 if self.hidden_size == 64 else 6 if self.hidden_size == 96 else 4
        if self.hidden_size % nhead != 0:
            raise ValueError(
                f"hidden_size {self.hidden_size} is not divisible by {nhead}"
            )
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_size,
            nhead=nhead,
            dropout=self.dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            self.transformer_encoder_layer,
            num_layers=self.num_transformer_layers,
        )
        self.d1 = nn.Linear(self.hidden_size, self.fc_dim)
        self.d2 = nn.Linear(self.fc_dim, self.num_classes)
        self.attention_fc = nn.Linear(self.hidden_size, self.attention_dim)
        self.attention_out = nn.Linear(self.attention_dim, 1)
        self.reset_diagnostics()

    @staticmethod
    def _pool_valid_mask(valid_mask):
        return F.max_pool1d(
            valid_mask.float().unsqueeze(1), kernel_size=2
        ).squeeze(1).bool()

    def reset_diagnostics(self):
        self.last_gamma_mean = None
        self.last_gamma_std = None
        self.last_gamma_abs_max = None
        self.last_delta_mean = None
        self.last_delta_std = None
        self.last_delta_abs_max = None
        self.last_global_attention_entropy = None
        self.last_global_attention_max_weight = None

    def _project_embeddings(self, embeddings, valid_mask):
        if self.use_two_stage_proj:
            embeddings = self.ntv2_proj2(F.relu(self.ntv2_proj1(embeddings)))
        else:
            embeddings = self.ntv2_proj_single(embeddings)
        embeddings = self.norm(embeddings)
        return embeddings.masked_fill(~valid_mask.unsqueeze(-1), 0.0)

    def _conv_pool_stage(self, values, valid_mask, cnn_layer, use_dropout):
        values = cnn_layer(values)
        values = values.masked_fill(
            ~valid_mask.unsqueeze(1), torch.finfo(values.dtype).min
        )
        values = self.maxpool_layer(values)
        valid_mask = self._pool_valid_mask(valid_mask)
        values = F.relu(values).masked_fill(~valid_mask.unsqueeze(1), 0.0)
        if use_dropout:
            values = F.dropout(values, p=self.dropout, training=self.training)
        return values, valid_mask

    def attention(self, transformer_output, valid_mask):
        logits = self.attention_out(
            torch.tanh(self.attention_fc(transformer_output))
        ).squeeze(-1)
        logits = logits.masked_fill(
            ~valid_mask, torch.finfo(logits.dtype).min
        )
        weights = F.softmax(logits, dim=1).unsqueeze(-1)
        return torch.sum(weights * transformer_output, dim=1)

    def _classify_sequence(self, sequence, valid_mask):
        sequence = sequence.masked_fill(~valid_mask.unsqueeze(-1), 0.0)
        transformer_out = self.transformer(
            sequence, src_key_padding_mask=~valid_mask
        )
        return self.d2(F.relu(self.d1(self.attention(transformer_out, valid_mask))))



class ProGLHM(_LocalBackbone):
    """Hybrid promoter-window classifier with background/promoter output logits.

    ``forward((embeddings, valid_mask))`` accepts ``[B,T,1024]`` floating-point
    NT-v2 features and ``[B,T]`` masks. ``T`` is a multiple of eight, at most
    128. Column 0 is background and column 1 is promoter. ``mamba_layer`` can
    supply a Mamba module; otherwise the installed Mamba implementation is used.
    """

    model_name = MODEL_NAME
    architecture_version = MODEL_ARCHITECTURE_VERSION

    def __init__(self, mamba_layer=None):
        super().__init__()
        self.global_projection = nn.Linear(128, GLOBAL_DIMENSION)
        self.global_norm = nn.LayerNorm(GLOBAL_DIMENSION)
        self.global_attention_fc = nn.Linear(GLOBAL_DIMENSION, GLOBAL_DIMENSION)
        self.global_attention_out = nn.Linear(GLOBAL_DIMENSION, 1)
        self.film_query_norm = nn.LayerNorm(GLOBAL_DIMENSION)
        self.film_mlp = nn.Sequential(
            nn.Linear(GLOBAL_DIMENSION, GLOBAL_DIMENSION * 2),
            nn.GELU(),
            nn.Linear(GLOBAL_DIMENSION * 2, GLOBAL_DIMENSION * 2),
        )
        nn.init.normal_(
            self.film_mlp[-1].weight, mean=0.0, std=FILM_OUTPUT_WEIGHT_STD
        )
        nn.init.zeros_(self.film_mlp[-1].bias)
        self.lambda_raw = nn.Parameter(
            torch.tensor(LAMBDA_RAW_INITIAL_VALUE, dtype=torch.float32)
        )

        self.mamba = mamba_layer or _make_mamba(
            GLOBAL_DIMENSION, MAMBA_D_STATE, MAMBA_D_CONV, MAMBA_EXPAND
        )
        self.direction_raw = nn.Parameter(torch.zeros((), dtype=torch.float32))
        self.alpha_raw = nn.Parameter(
            torch.tensor(ALPHA_RAW_INITIAL_VALUE, dtype=torch.float32)
        )

    def effective_alpha(self):
        value = ALPHA_SCALE_MAX * torch.sigmoid(self.alpha_raw)
        strict_upper = torch.nextafter(
            value.new_tensor(ALPHA_SCALE_MAX), value.new_zeros(())
        )
        return value.clamp_max(strict_upper)

    def effective_lambda(self):
        value = LAMBDA_SCALE_MAX * torch.sigmoid(self.lambda_raw)
        strict_upper = torch.nextafter(
            value.new_tensor(LAMBDA_SCALE_MAX), value.new_zeros(())
        )
        return value.clamp_max(strict_upper)

    def direction_weight(self):
        return torch.sigmoid(self.direction_raw)

    def _shared_bidirectional_mamba(self, base, valid_mask):
        forward_values = self.mamba(base).masked_fill(
            ~valid_mask.unsqueeze(-1), 0.0
        )
        reverse_inputs = _reverse_valid_prefix(base, valid_mask)
        backward_values = _reverse_valid_prefix(
            self.mamba(reverse_inputs), valid_mask
        )
        direction = self.direction_weight().to(dtype=base.dtype)
        return direction * forward_values + (1.0 - direction) * backward_values

    def _masked_global_attention(self, tokens, valid_mask):
        logits = self.global_attention_out(
            torch.tanh(self.global_attention_fc(tokens))
        ).squeeze(-1)
        logits = logits.masked_fill(~valid_mask, torch.finfo(logits.dtype).min)
        weights = F.softmax(logits, dim=1)
        probabilities = weights.float().clamp_min(1e-12)
        self.last_global_attention_entropy = (
            -(probabilities * probabilities.log()).sum(dim=1).mean().detach()
        )
        self.last_global_attention_max_weight = (
            probabilities.max(dim=1).values.mean().detach()
        )
        return torch.sum(weights.unsqueeze(-1) * tokens, dim=1)

    def _global_summary(self, highres_tokens, highres_mask):
        base = self.global_norm(self.global_projection(highres_tokens))
        base = base.masked_fill(~highres_mask.unsqueeze(-1), 0.0)
        update = self._shared_bidirectional_mamba(base, highres_mask)
        tokens = base + self.effective_alpha().to(base.dtype) * update
        tokens = tokens.masked_fill(~highres_mask.unsqueeze(-1), 0.0)
        return self._masked_global_attention(tokens, highres_mask)

    def _film_modulation(self, query, lowres_mask, summary):
        gamma_raw, delta_raw = self.film_mlp(summary).chunk(2, dim=-1)
        gamma = torch.tanh(gamma_raw)
        delta = torch.tanh(delta_raw)
        self.last_gamma_mean = gamma.float().mean().detach()
        self.last_gamma_std = gamma.float().std(unbiased=False).detach()
        self.last_gamma_abs_max = gamma.float().abs().max().detach()
        self.last_delta_mean = delta.float().mean().detach()
        self.last_delta_std = delta.float().std(unbiased=False).detach()
        self.last_delta_abs_max = delta.float().abs().max().detach()
        normalized_query = self.film_query_norm(query)
        modulation = normalized_query * gamma.unsqueeze(1) + delta.unsqueeze(1)
        fused = query + self.effective_lambda().to(query.dtype) * modulation
        return fused.masked_fill(~lowres_mask.unsqueeze(-1), 0.0)

    def forward(self, data):
        self.reset_diagnostics()
        embeddings, highres_mask = validate_inputs(data)
        projected = self._project_embeddings(embeddings, highres_mask)
        cnn1 = F.relu(self.cnn_layers[0](projected.transpose(1, 2)))
        cnn1 = cnn1.masked_fill(~highres_mask.unsqueeze(1), 0.0)
        summary = self._global_summary(cnn1.transpose(1, 2), highres_mask)

        lowres = cnn1.masked_fill(
            ~highres_mask.unsqueeze(1), torch.finfo(cnn1.dtype).min
        )
        lowres = self.maxpool_layer(lowres)
        lowres_mask = self._pool_valid_mask(highres_mask)
        lowres = lowres.masked_fill(~lowres_mask.unsqueeze(1), 0.0)
        lowres, lowres_mask = self._conv_pool_stage(
            lowres, lowres_mask, self.cnn_layers[1], use_dropout=True
        )
        lowres, lowres_mask = self._conv_pool_stage(
            lowres, lowres_mask, self.cnn_layers[2], use_dropout=True
        )
        query = self.fc_cnn_to_transformer(lowres.transpose(1, 2))
        query = query.masked_fill(~lowres_mask.unsqueeze(-1), 0.0)
        fused = self._film_modulation(query, lowres_mask, summary)
        return self._classify_sequence(fused, lowres_mask)

    def tensor_diagnostics(self):
        return {
            "alpha": self.effective_alpha(),
            "lambda": self.effective_lambda(),
            "direction_weight": self.direction_weight(),
            "gamma_mean": self.last_gamma_mean,
            "gamma_std": self.last_gamma_std,
            "gamma_abs_max": self.last_gamma_abs_max,
            "delta_mean": self.last_delta_mean,
            "delta_std": self.last_delta_std,
            "delta_abs_max": self.last_delta_abs_max,
            "global_attention_entropy": self.last_global_attention_entropy,
            "global_attention_max_weight": self.last_global_attention_max_weight,
        }

    def scalar_diagnostics(self):
        return {
            name: None if value is None else float(value.detach().float().cpu())
            for name, value in self.tensor_diagnostics().items()
        }

