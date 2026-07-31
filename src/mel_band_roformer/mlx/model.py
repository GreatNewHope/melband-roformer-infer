"""MelBandRoformerMLX -- the Mel-Band RoFormer model architecture ported to Apple MLX.

Vendored (not depended on) because the upstream package this ships from,
`mlx-audio-separator`, bundles a full model catalog, a CLI, Demucs/VR/MDX
architectures, and a compiled `mlx-audio-io` extension that pins `mlx==0.31.2`
exactly -- installing it would drag all of that in for one model class.

Two source files are combined here, because upstream's own MelBand-Roformer file
imports its transformer/attention/band-split/mask-estimator primitives from its
sibling BS-Roformer file rather than redefining them (the two architectures share
almost everything except the band layout). This module vendors both: the shared
primitives (everything through `BSRoformerBlock`) verbatim from upstream's
`bs_roformer.py`, then upstream's `mel_band_roformer.py` unchanged below it, with
its `from .bs_roformer import (...)` replaced by nothing since both now live in one
file. bs-roformer-infer (this package's fork sibling) vendors the same shared
primitives independently for its own BS-Roformer port -- see that package's
`src/bs_roformer/mlx/model.py`; the duplication is real and is called out in this
package's CLAUDE.md rather than hidden.

Two deliberate deviations from upstream, beyond the header:

1. `exact_zero_safe_rfft` (below) wraps the STFT call inside `__call__`. MLX
   0.31.2's Metal rfft kernel is not exactly zero on an all-zero frame; see its
   docstring for the failure mode and measured numbers. Ported from
   bs-roformer-infer's identical fix for the same MLX kernel bug on the sibling
   architecture -- read its docstring before touching it.
2. Upstream's `MelBandRoformerMLX` also carries a trunk-level `self.final_norm`,
   applied once after the transformer stack, in addition to each individual
   Transformer's own trailing norm (`norm_output=True`). This package's own
   PyTorch `MelBandRoformer` (`mel_band_roformer.py`) has no such layer and no
   such call -- each Transformer's own norm is the only normalization the trunk
   ever applies. Keeping upstream's extra norm would not merely mismatch
   numerically: PyTorch checkpoints never trained a `final_norm` weight, so
   `load_converted_weights()` below would refuse to load at all (an unmatched
   model parameter, never a silent partial load). It is omitted here rather than
   filled with an untrained tensor.

Vendored from:
    Project:  mlx-audio-separator (MIT License)
    Author:   ssmall256 (as named in upstream LICENSE)
    Repo:     https://github.com/ssmall256/mlx-audio-separator
    Files:    mlx_audio_separator/separator/models/roformer/bs_roformer.py
              (shared primitives) and
              mlx_audio_separator/separator/models/roformer/mel_band_roformer.py
              (MelBandRoformerMLX and its mel-filter-bank helpers)
    Revision: 0ddc8cf5507906b52ac45a9cd9e6d26e881a93f8
    Copyright (c) 2024-2026 ssmall256. Permission is hereby granted, free of
    charge, to any person obtaining a copy of this software and associated
    documentation files (the "Software"), to deal in the Software without
    restriction, subject to the MIT License terms in upstream's LICENSE file.

Reads: mlx.core, mlx.nn, numpy, mlx_spectro (get_transform_mlx), packaging
"""

import math
import os
from collections import defaultdict
from contextlib import contextmanager
from typing import List, Optional, Tuple  # noqa: UP035

import mlx.core as mx
import mlx.nn as nn  # noqa: PLR0402
import numpy as np
from mlx_spectro import get_transform_mlx
from packaging import version

_USE_SAFE_SLICE_ACCUMULATION = version.parse(mx.__version__) >= version.parse("0.31.2")

# ---------------------------------------------------------------------------
# Shared primitives, vendored verbatim from upstream's bs_roformer.py (the
# subset mel_band_roformer.py actually imports: pack/unpack helpers, rearrange,
# L2Norm, FeedForward, Attention, LinearAttention, Transformer, BandSplit,
# MaskEstimator, BSRoformerBlock). BSRoformerMLX itself, its DEFAULT_FREQS_
# PER_BANDS constant, and create_compiled_model are BS-Roformer-only and not
# needed by the MelBand model, so they are not vendored here.
# ---------------------------------------------------------------------------


def exists(val):
    return val is not None


def default(v, d):
    return v if exists(v) else d


def env_enabled(name: str, default_value: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default_value)
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def batched_group_linear(x_group: mx.array, weights: mx.array, biases: mx.array | None = None) -> mx.array:
    """Apply per-group linear layers in a single batched matmul.

    Args:
        x_group: Shape (B, T, G, In)
        weights: Shape (G, Out, In)
        biases: Optional shape (G, Out)
    """
    out = mx.einsum("btgi,goi->btgo", x_group, weights)
    if biases is not None:
        out = out + biases[None, None, :, :]
    return out


# MLX-native replacements for einops operations
def pack(tensors: List[mx.array], pattern: str) -> Tuple[mx.array, List]:  # noqa: UP006
    """
    Pack tensors by flattening dimensions according to pattern.

    Patterns:
    - "b * d" : keep first and last dim, flatten middle dims
    - "* t d" : flatten leading dims, keep last two dims
    - "* f d" : flatten leading dims, keep last two dims
    """
    if len(tensors) == 1:
        x = tensors[0]
        original_shape = x.shape

        if pattern == "b * d":
            # Keep first and last dim, flatten middle
            # (b, f, t, d) -> (b, f*t, d)
            if len(x.shape) == 4:
                b, f, t, d = x.shape
                x = mx.reshape(x, (b, f * t, d))
            elif len(x.shape) == 3:
                # Already in the right shape
                pass
            return x, [original_shape]

        elif pattern == "* t d" or pattern == "* f d":
            # Flatten leading dims, keep last two
            # (b, f, t, d) -> (b*f, t, d)
            if len(x.shape) >= 3:
                *leading, t, d = x.shape
                leading_size = int(np.prod(leading))
                x = mx.reshape(x, (leading_size, t, d))
            return x, [original_shape]

        else:
            raise NotImplementedError(f"Pack pattern not implemented: {pattern}")
    else:
        # Stack multiple tensors
        packed = mx.stack(tensors, axis=0)
        shapes = [t.shape for t in tensors]
        return packed, shapes


def unpack(tensor: mx.array, shapes: List, pattern: str) -> List[mx.array]:  # noqa: UP006
    """
    Unpack tensor by restoring original shape.
    """
    if len(shapes) == 1:
        original_shape = shapes[0]
        x = mx.reshape(tensor, original_shape)
        return [x]
    else:
        # Unstack multiple tensors
        return [tensor[i] for i in range(len(shapes))]


def pack_one(tensors: List[mx.array], pattern: str) -> Tuple[mx.array, List]:  # noqa: UP006
    """Pack single tensor - alias for pack."""
    return pack(tensors, pattern)


def unpack_one(tensor: mx.array, shapes: List, pattern: str) -> mx.array:  # noqa: UP006
    """Unpack single tensor - returns first element."""
    return unpack(tensor, shapes, pattern)[0]


def rearrange(x: mx.array, pattern: str, **axes_lengths) -> mx.array:
    """
    MLX-native implementation of einops rearrange for BS-Roformer/MelBand-Roformer
    patterns.
    """
    if "->" not in pattern:
        raise ValueError(f"Invalid pattern: {pattern}")

    input_pattern, output_pattern = pattern.split("->")
    input_pattern = input_pattern.strip()
    output_pattern = output_pattern.strip()

    # Pattern: "b n (qkv h d) -> qkv b h n d" with qkv=3, h=heads
    if input_pattern == "b n (qkv h d)" and output_pattern == "qkv b h n d":
        b, n, _ = x.shape
        qkv = axes_lengths['qkv']
        h = axes_lengths['h']
        d = x.shape[-1] // (qkv * h)
        x = mx.reshape(x, (b, n, qkv, h, d))
        x = mx.transpose(x, (2, 0, 3, 1, 4))  # qkv, b, h, n, d
        return x

    # Pattern: "b n (qkv h d) -> qkv b h d n" with qkv=3, h=heads (for freq transformer)
    if input_pattern == "b n (qkv h d)" and output_pattern == "qkv b h d n":
        b, n, _ = x.shape
        qkv = axes_lengths['qkv']
        h = axes_lengths['h']
        d = x.shape[-1] // (qkv * h)
        x = mx.reshape(x, (b, n, qkv, h, d))
        x = mx.transpose(x, (2, 0, 3, 4, 1))  # qkv, b, h, d, n
        return x

    # Pattern: "b h n d -> b n h d"
    if input_pattern == "b h n d" and output_pattern == "b n h d":
        return mx.transpose(x, (0, 2, 1, 3))

    # Pattern: "b h d n -> b n h d"
    if input_pattern == "b h d n" and output_pattern == "b n h d":
        return mx.transpose(x, (0, 3, 1, 2))

    # Pattern: "b n h d -> b h n d"
    if input_pattern == "b n h d" and output_pattern == "b h n d":
        return mx.transpose(x, (0, 2, 1, 3))

    # Pattern: "b h n d -> b n (h d)"
    if input_pattern == "b h n d" and output_pattern == "b n (h d)":
        b, h, n, d = x.shape
        return mx.reshape(mx.transpose(x, (0, 2, 1, 3)), (b, n, h * d))

    # Pattern: "b h d n -> b n (h d)"
    if input_pattern == "b h d n" and output_pattern == "b n (h d)":
        b, h, d, n = x.shape
        return mx.reshape(mx.transpose(x, (0, 3, 1, 2)), (b, n, h * d))

    # Pattern: "b n h -> b h n 1"
    if input_pattern == "b n h" and output_pattern == "b h n 1":
        return mx.transpose(x, (0, 2, 1))[..., None]

    # Pattern: "b c t -> (b c) t"
    if input_pattern == "b c t" and output_pattern == "(b c) t":
        b, c, t = x.shape
        return mx.reshape(x, (b * c, t))

    # Pattern: "(b c) f t complex -> b (f c) t complex" with c=channels
    if input_pattern == "(b c) f t complex" and output_pattern == "b (f c) t complex":
        c = axes_lengths['c']
        bc, f, t, complex_dim = x.shape
        b = bc // c
        x = mx.reshape(x, (b, c, f, t, complex_dim))
        x = mx.transpose(x, (0, 2, 1, 3, 4))  # b, f, c, t, complex
        x = mx.reshape(x, (b, f * c, t, complex_dim))
        return x

    # Pattern: "b n (f c) t -> (b n c) f t" with c=channels
    if input_pattern == "b n (f c) t" and output_pattern == "(b n c) f t":
        c = axes_lengths['c']
        b, n, fc, t = x.shape
        f = fc // c
        x = mx.reshape(x, (b, n, f, c, t))
        x = mx.transpose(x, (0, 1, 3, 2, 4))  # b, n, c, f, t
        x = mx.reshape(x, (b * n * c, f, t))
        return x

    # Pattern: "(b n c) t -> b n c t" with b=batch, n=stems, c=channels
    if input_pattern == "(b n c) t" and output_pattern == "b n c t":
        b = axes_lengths['b']
        n = axes_lengths['n']
        c = axes_lengths['c']
        bnc, t = x.shape  # noqa: RUF059
        return mx.reshape(x, (b, n, c, t))

    # Pattern: "b 1 c t -> b c t"
    if input_pattern == "b 1 c t" and output_pattern == "b c t":
        return mx.squeeze(x, axis=1)

    # Pattern: "b f t c -> b t (f c)"
    if input_pattern == "b f t c" and output_pattern == "b t (f c)":
        b, f, t, c = x.shape
        x = mx.transpose(x, (0, 2, 1, 3))  # b, t, f, c
        return mx.reshape(x, (b, t, f * c))

    # Pattern: "b t f d -> b f t d"
    if input_pattern == "b t f d" and output_pattern == "b f t d":
        return mx.transpose(x, (0, 2, 1, 3))

    # Pattern: "b f t d -> b t f d"
    if input_pattern == "b f t d" and output_pattern == "b t f d":
        return mx.transpose(x, (0, 2, 1, 3))

    # Pattern: "b n t (f c) -> b n f t c" with c=2
    if input_pattern == "b n t (f c)" and output_pattern == "b n f t c":
        c = axes_lengths['c']
        b, n, t, fc = x.shape
        f = fc // c
        x = mx.reshape(x, (b, n, t, f, c))
        return mx.transpose(x, (0, 1, 3, 2, 4))  # b, n, f, t, c

    raise NotImplementedError(f"Rearrange pattern not implemented: {pattern}")


@contextmanager
def exact_zero_safe_rfft():
    """Route `mx.fft.rfft` through the CPU stream for one STFT. NOT upstream code.

    MLX 0.31.2's Metal rfft kernel packs two real FFTs into one complex FFT; in
    float32 that cancellation is not bit-exact, so a frame whose true value is
    exactly zero comes back as roughly 4.5e-07 instead of 0. That matters far more
    than its size suggests: `L2Norm` discards magnitude, and its eps of 1e-12 is
    five orders below the artifact, so the clamp never engages and pure numerical
    noise is normalized into a full-scale, essentially random feature vector --
    about a millionfold amplification. Time-axis attention then spreads that one
    corrupted frame across every position, which is why silence at the end of a
    chunk corrupts the output at the beginning.

    This is not a corner case: every track's final chunk is padded, and music has
    rests. bs-roformer-infer measured, on its own real checkpoint, a zero-padded
    chunk diverging from Torch by 1.455e-02 max abs without this workaround, and
    2.2e-07 with it -- the same noise floor as a chunk with no silence at all.
    Ported here verbatim since the underlying MLX kernel bug is identical.

    Raising `L2Norm`'s eps was considered and rejected: genuinely quiet audio has
    legitimate band norms in the same range, so it would trade this bug for a
    quieter one that only shows up on soft material.

    Caveat, stated rather than hidden: this swaps a module-level attribute, so it
    is not thread-safe. Inference here is single-threaded per session.

    Delete this once MLX's kernel is fixed.
    """
    original = mx.fft.rfft

    def cpu_stream_rfft(*args, **kwargs):
        with mx.stream(mx.cpu):
            result = original(*args, **kwargs)
            mx.eval(result)
        return result

    mx.fft.rfft = cpu_stream_rfft
    try:
        yield
    finally:
        mx.fft.rfft = original


class L2Norm(nn.Module):
    """PyTorch-compatible norm: x / max(||x||, eps) * sqrt(dim) * weight.

    Matches this package's own torch RMSNorm (mel_band_roformer.py), which is
    `F.normalize(x, dim=-1) * scale * gamma` -- the same formula under a
    different class name; state-dict keys line up via the `.gamma` -> `.weight`
    rename in convert.py, not via matching class names.
    """

    def __init__(self, dim, eps=1e-12):
        super().__init__()
        self.eps = eps
        self.scale = dim ** 0.5
        self.weight = mx.ones((dim,))
        self.use_fast_norm = str(os.environ.get("MLX_AUDIO_SEPARATOR_ROFORMER_FAST_NORM", "0")).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

    def __call__(self, x):
        if self.use_fast_norm:
            # Equivalent to L2 normalization with sqrt(dim) scaling.
            return mx.fast.rms_norm(x, self.weight, self.eps) * self.scale
        norm = mx.sqrt(mx.sum(x * x, axis=-1, keepdims=True))
        denom = mx.maximum(norm, self.eps)
        return (x / denom) * self.scale * self.weight


class ExactGELU(nn.Module):
    """Exact GELU to match PyTorch's default (erf-based) implementation."""

    def __call__(self, x):
        return 0.5 * x * (1.0 + mx.erf(x / math.sqrt(2.0)))


class FeedForward(nn.Module):
    """
    Feed-forward network with RMSNorm.
    Matches PyTorch structure with nn.Sequential for weight compatibility.
    """

    def __init__(self, dim, mult=4, dropout=0.0):
        super().__init__()
        dim_inner = int(dim * mult)

        self.net = nn.Sequential(
            L2Norm(dim),            # net.layers.0
            nn.Linear(dim, dim_inner),  # net.layers.1
            ExactGELU(),            # net.layers.2
            nn.Dropout(dropout),    # net.layers.3
            nn.Linear(dim_inner, dim),  # net.layers.4
            nn.Dropout(dropout)     # net.layers.5
        )

    def __call__(self, x):
        return self.net(x)


class Attention(nn.Module):
    """
    Multi-head attention with rotary embeddings and gating.
    """

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0, rotary_embed=None):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        dim_inner = heads * dim_head

        # rotary_embed is now a boolean flag indicating whether to use RoPE
        self.use_rotary_embed = rotary_embed if isinstance(rotary_embed, bool) else (rotary_embed is not None)
        self.norm = L2Norm(dim)
        self.to_qkv = nn.Linear(dim, dim_inner * 3, bias=False)
        self.to_gates = nn.Linear(dim, heads)

        self.to_out = nn.Sequential(
            nn.Linear(dim_inner, dim, bias=False),  # to_out.layers.0
            nn.Dropout(dropout)                      # to_out.layers.1
        )

    def __call__(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x)
        qkv = rearrange(qkv, "b n (qkv h d) -> qkv b h n d", qkv=3, h=self.heads)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.use_rotary_embed:
            # fast.rope with traditional=True matches PyTorch's RotaryEmbedding.
            q = mx.fast.rope(q, dims=self.dim_head, traditional=True, base=10000.0, scale=1.0, offset=0)
            k = mx.fast.rope(k, dims=self.dim_head, traditional=True, base=10000.0, scale=1.0, offset=0)

        if os.environ.get("MLX_USE_FAST_SDP") == "1":
            # Optional fast path; may change numerics vs PyTorch.
            out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
        else:
            # Manual attention to match PyTorch behavior; mx.fast kernel diverges in practice.
            attn_scores = mx.matmul(q, mx.transpose(k, (0, 1, 3, 2))) * self.scale
            attn_scores = attn_scores - mx.max(attn_scores, axis=-1, keepdims=True)
            attn = mx.softmax(attn_scores, axis=-1)
            out = mx.matmul(attn, v)

        gates = self.to_gates(x)
        gates = mx.sigmoid(gates)
        gates = rearrange(gates, "b n h -> b h n 1")
        out = out * gates

        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class LinearAttention(nn.Module):
    """Linear attention variant, kept for structural parity with upstream/
    bs-roformer-infer; melband-roformer-infer's own torch model never sets
    linear_transformer_depth > 0, so this path is unused by any checkpoint here.
    """

    def __init__(self, dim, dim_head=32, heads=8, dropout=0.0):
        super().__init__()
        dim_inner = dim_head * heads
        self.heads = heads
        self.dim_head = dim_head

        self.norm = L2Norm(dim)
        self.to_qkv = nn.Linear(dim, dim_inner * 3, bias=False)

        self.temperature = mx.ones((heads, 1, 1))

        self.to_out = nn.Linear(dim_inner, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def __call__(self, x):
        b, n, _ = x.shape  # noqa: RUF059

        x = self.norm(x)

        qkv = self.to_qkv(x)
        qkv = rearrange(qkv, "b n (qkv h d) -> qkv b h d n", qkv=3, h=self.heads)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q / mx.sqrt(mx.sum(q * q, axis=-2, keepdims=True) + 1e-8)
        k = k / mx.sqrt(mx.sum(k * k, axis=-2, keepdims=True) + 1e-8)

        q = q * mx.exp(self.temperature)

        context = mx.matmul(k, v.transpose(0, 1, 3, 2))
        out = mx.matmul(q, context)

        out = rearrange(out, "b h d n -> b n (h d)")
        out = self.to_out(out)
        return self.dropout(out)


class TransformerLayer(nn.Module):
    """Single transformer layer with attention and feedforward."""
    def __init__(self, attn, ff):
        super().__init__()
        self.attn = attn
        self.ff = ff

    def __call__(self, x):
        x = self.attn(x) + x
        x = self.ff(x) + x
        return x


class Transformer(nn.Module):
    """Transformer block with attention and feed-forward layers."""

    def __init__(
        self,
        *,
        dim,
        depth,
        dim_head=64,
        heads=8,
        attn_dropout=0.0,
        ff_dropout=0.0,
        ff_mult=4,
        norm_output=True,
        rotary_embed=None,
        linear_attn=False
    ):
        super().__init__()
        self.depth = depth

        for i in range(depth):
            if linear_attn:
                attn = Attention(
                    dim=dim,
                    dim_head=dim_head,
                    heads=heads,
                    dropout=attn_dropout,
                    rotary_embed=rotary_embed
                )
            else:
                attn = Attention(
                    dim=dim,
                    dim_head=dim_head,
                    heads=heads,
                    dropout=attn_dropout,
                    rotary_embed=rotary_embed
                )

            ff = FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout)
            setattr(self, f'layers_{i}', TransformerLayer(attn, ff))

        self.norm = L2Norm(dim) if norm_output else nn.Identity()

    def __call__(self, x):
        for i in range(self.depth):
            layer = getattr(self, f'layers_{i}')
            x = layer(x)

        return self.norm(x)


class BandSplitModule(nn.Module):
    """Single band processing module."""
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.norm = L2Norm(dim_in)
        self.linear = nn.Linear(dim_in, dim_out)

    def __call__(self, x):
        return self.linear(self.norm(x))


class BandSplit(nn.Module):
    """Band-split module that splits frequency bins into bands and projects to
    feature dimension."""

    def __init__(self, dim, dim_inputs: Tuple[int, ...], use_grouped: bool = False):  # noqa: UP006
        super().__init__()
        self.dim_inputs = dim_inputs
        self.num_bands = len(dim_inputs)
        self.split_points = np.cumsum(self.dim_inputs)[:-1].tolist()
        self.use_grouped = bool(use_grouped)
        self._band_module_names: list[str] = []
        grouped: dict[int, list[int]] = defaultdict(list)

        for i, dim_in in enumerate(dim_inputs):
            module_name = f"to_features_{i}"
            setattr(self, module_name, BandSplitModule(dim_in, dim))
            self._band_module_names.append(module_name)
            grouped[int(dim_in)].append(i)
        self._grouped_band_indices = tuple((dim_in, tuple(indices)) for dim_in, indices in grouped.items())
        self.use_grouped_weight_cache = env_enabled(
            "MLX_AUDIO_SEPARATOR_ROFORMER_GROUPED_WEIGHT_CACHE",
            default_value=False,
        )
        self._grouped_pack_cache: dict[tuple[int, ...], dict[str, object]] = {}

    def _get_grouped_pack(self, band_indices: tuple[int, ...]) -> dict[str, object]:
        modules = [getattr(self, self._band_module_names[idx]) for idx in band_indices]
        signature_items: list[int] = []
        for module in modules:
            signature_items.append(id(module.norm.weight))
            signature_items.append(id(module.linear.weight))
            bias = getattr(module.linear, "bias", None)
            signature_items.append(id(bias) if bias is not None else 0)
        signature = tuple(signature_items)

        if self.use_grouped_weight_cache:
            cached = self._grouped_pack_cache.get(band_indices)
            if cached is not None and cached.get("signature") == signature:
                return cached

        norm_weights = mx.stack([module.norm.weight for module in modules], axis=0)
        linear_weights = mx.stack([module.linear.weight for module in modules], axis=0)
        biases = [getattr(module.linear, "bias", None) for module in modules]
        has_bias = all(bias is not None for bias in biases)
        linear_bias = mx.stack(biases, axis=0) if has_bias else None
        packed = {
            "signature": signature,
            "eps": float(modules[0].norm.eps),
            "scale": float(modules[0].norm.scale),
            "norm_weights": norm_weights,
            "linear_weights": linear_weights,
            "linear_bias": linear_bias,
        }
        if self.use_grouped_weight_cache:
            self._grouped_pack_cache[band_indices] = packed
        return packed

    def _forward_grouped(self, splits: list[mx.array]) -> mx.array:
        outs: list[mx.array | None] = [None] * self.num_bands

        for _, band_indices in self._grouped_band_indices:
            if len(band_indices) <= 1:
                band_idx = int(band_indices[0])
                to_feature = getattr(self, self._band_module_names[band_idx])
                outs[band_idx] = to_feature(splits[band_idx])
                continue

            grouped_input = mx.stack([splits[idx] for idx in band_indices], axis=2)  # (B, T, G, D)
            packed = self._get_grouped_pack(band_indices)

            eps = float(packed["eps"])
            scale = float(packed["scale"])
            norm = mx.sqrt(mx.sum(grouped_input * grouped_input, axis=-1, keepdims=True))
            denom = mx.maximum(norm, eps)
            normalized = (grouped_input / denom) * scale
            norm_weights = packed["norm_weights"]
            normalized = normalized * norm_weights[None, None, :, :]
            linear_weights = packed["linear_weights"]
            linear_bias = packed["linear_bias"]
            grouped_out = batched_group_linear(normalized, linear_weights, linear_bias)
            for local_idx, band_idx in enumerate(band_indices):
                outs[int(band_idx)] = grouped_out[:, :, local_idx, :]

        return mx.stack([out for out in outs if out is not None], axis=-2)

    def __call__(self, x):
        splits = mx.split(x, self.split_points, axis=-1)
        if self.use_grouped:
            return self._forward_grouped(splits)

        outs = []
        for i, split_input in enumerate(splits):
            to_feature = getattr(self, self._band_module_names[i])
            split_output = to_feature(split_input)
            outs.append(split_output)

        return mx.stack(outs, axis=-2)


def MLP(dim_in, dim_out, dim_hidden=None, depth=1):
    """Helper to create MLP with MLX Sequential (stores layers in self.layers list).

    NOT UPSTREAM (deviation from bs_roformer.py's vendored MLP): this package's
    own torch `MelBandRoformer.MLP()` (mel_band_roformer.py) builds
    `dims = (dim_in, *((dim_hidden,) * depth), dim_out)` -- `depth` hidden
    layers, not `depth - 1`. bs-roformer-infer's torch `MLP()` uses `depth - 1`
    (one fewer hidden layer for the same `depth` value), and its vendored MLX
    `MLP()` matches that. Copying it verbatim here would build a shallower MLP
    than the checkpoint was trained with; `load_converted_weights()` in
    convert.py caught the resulting shape mismatch (a dropped `to_freqs_*.
    layers.2.*` tensor) against a real state_dict.
    """
    dim_hidden = default(dim_hidden, dim_in)

    layers = []
    dims = (dim_in, *((dim_hidden,) * depth), dim_out)

    for ind, (layer_dim_in, layer_dim_out) in enumerate(zip(dims[:-1], dims[1:])):  # noqa: RUF007
        is_last = ind == (len(dims) - 2)

        layers.append(nn.Linear(layer_dim_in, layer_dim_out))

        if not is_last:
            layers.append(nn.Tanh())

    return nn.Sequential(*layers)


class MaskEstimator(nn.Module):
    """Mask estimator that generates frequency masks for each stem."""

    def __init__(self, dim, dim_inputs: Tuple[int, ...], depth, mlp_expansion_factor=4, use_grouped: bool = False):  # noqa: UP006
        super().__init__()
        self.dim_inputs = dim_inputs
        self.num_bands = len(dim_inputs)
        self.use_grouped = bool(use_grouped)
        dim_hidden = dim * mlp_expansion_factor
        self._mlp_module_names: list[str] = []
        grouped: dict[int, list[int]] = defaultdict(list)

        for i, dim_in in enumerate(dim_inputs):
            module_name = f"to_freqs_{i}"
            setattr(self, module_name, MLP(dim, dim_in * 2, dim_hidden=dim_hidden, depth=depth))
            self._mlp_module_names.append(module_name)
            grouped[int(dim_in)].append(i)
        self._grouped_band_indices = tuple((dim_in, tuple(indices)) for dim_in, indices in grouped.items())
        self.use_grouped_weight_cache = env_enabled(
            "MLX_AUDIO_SEPARATOR_ROFORMER_GROUPED_WEIGHT_CACHE",
            default_value=False,
        )
        self._grouped_mlp_pack_cache: dict[tuple[int, ...], dict[str, object] | None] = {}

    def _get_grouped_mlp_pack(self, band_indices: tuple[int, ...]) -> dict[str, object] | None:
        mlps = [getattr(self, self._mlp_module_names[idx]) for idx in band_indices]
        if not mlps:
            return None

        layer_lists = [getattr(mlp, "layers", None) for mlp in mlps]
        if any(layers is None for layers in layer_lists):
            return None
        depth = len(layer_lists[0])
        if any(len(layers) != depth for layers in layer_lists):
            return None

        metadata: list[tuple[str, list[mx.array] | None, list[mx.array | None] | None, bool]] = []
        signature_items: list[int] = []
        for layer_idx in range(depth):
            proto = layer_lists[0][layer_idx]
            proto_name = proto.__class__.__name__.lower()
            if hasattr(proto, "weight"):
                weights = []
                biases = []
                has_bias = True
                for layers in layer_lists:
                    layer = layers[layer_idx]
                    weight = getattr(layer, "weight", None)
                    if weight is None:
                        return None
                    weights.append(weight)
                    signature_items.append(id(weight))
                    bias = getattr(layer, "bias", None)
                    if bias is None:
                        has_bias = False
                    biases.append(bias)
                    signature_items.append(id(bias) if bias is not None else 0)
                metadata.append(("linear", weights, biases, has_bias))
            elif proto_name == "tanh":
                metadata.append(("tanh", None, None, False))
            else:
                return None

        signature = tuple(signature_items)
        if self.use_grouped_weight_cache:
            cached = self._grouped_mlp_pack_cache.get(band_indices)
            if cached is not None and cached.get("signature") == signature:
                return cached

        ops: list[dict[str, object]] = []
        for kind, weights, biases, has_bias in metadata:
            if kind == "tanh":
                ops.append({"kind": "tanh"})
                continue
            assert weights is not None
            linear_weights = mx.stack(weights, axis=0)
            linear_bias = mx.stack(biases, axis=0) if has_bias and biases is not None else None
            ops.append({"kind": "linear", "weights": linear_weights, "bias": linear_bias})

        packed = {"signature": signature, "ops": ops}
        if self.use_grouped_weight_cache:
            self._grouped_mlp_pack_cache[band_indices] = packed
        return packed

    def _run_grouped_mlp(self, grouped_input: mx.array, band_indices: tuple[int, ...]) -> mx.array | None:
        packed = self._get_grouped_mlp_pack(band_indices)
        if packed is None:
            return None
        x = grouped_input
        for op in packed["ops"]:
            if op["kind"] == "tanh":
                x = mx.tanh(x)
            else:
                x = batched_group_linear(x, op["weights"], op["bias"])
        return x

    def __call__(self, x):
        x_bands = [x[..., i, :] for i in range(x.shape[-2])]

        if self.use_grouped:
            outs_by_band: list[mx.array | None] = [None] * self.num_bands
            for _, band_indices in self._grouped_band_indices:
                if len(band_indices) <= 1:
                    band_idx = int(band_indices[0])
                    mlp = getattr(self, self._mlp_module_names[band_idx])
                    freq_out_before_glu = mlp(x_bands[band_idx])
                    freq_out = mx.split(freq_out_before_glu, 2, axis=-1)
                    outs_by_band[band_idx] = freq_out[0] * mx.sigmoid(freq_out[1])
                    continue

                grouped_input = mx.stack([x_bands[idx] for idx in band_indices], axis=2)  # (B, T, G, D)
                grouped_out = self._run_grouped_mlp(grouped_input, band_indices)
                if grouped_out is None:
                    for band_idx in band_indices:
                        mlp = getattr(self, self._mlp_module_names[int(band_idx)])
                        freq_out_before_glu = mlp(x_bands[int(band_idx)])
                        freq_out = mx.split(freq_out_before_glu, 2, axis=-1)
                        outs_by_band[int(band_idx)] = freq_out[0] * mx.sigmoid(freq_out[1])
                    continue

                values, gates = mx.split(grouped_out, 2, axis=-1)
                grouped_masks = values * mx.sigmoid(gates)
                for local_idx, band_idx in enumerate(band_indices):
                    outs_by_band[int(band_idx)] = grouped_masks[:, :, local_idx, :]

            return mx.concatenate([out for out in outs_by_band if out is not None], axis=-1)

        outs = []
        for i, band_features in enumerate(x_bands):
            mlp = getattr(self, self._mlp_module_names[i])
            freq_out_before_glu = mlp(band_features)

            freq_out = mx.split(freq_out_before_glu, 2, axis=-1)
            freq_out = freq_out[0] * mx.sigmoid(freq_out[1])

            outs.append(freq_out)

        return mx.concatenate(outs, axis=-1)


class BSRoformerBlock(nn.Module):
    """One trunk block, containing time and frequency (and optionally linear)
    transformers. Container only -- the loop over blocks lives in
    MelBandRoformerMLX._forward_transformers."""
    def __init__(self, linear_transformer, time_transformer, freq_transformer):
        super().__init__()
        self.has_linear = linear_transformer is not None

        if self.has_linear:
            self.linear_transformer = linear_transformer
        self.time_transformer = time_transformer
        self.freq_transformer = freq_transformer

    def __call__(self, x):
        return x


# ---------------------------------------------------------------------------
# Mel-specific pieces, vendored from upstream's mel_band_roformer.py.
# ---------------------------------------------------------------------------


def _hz_to_mel(freq: np.ndarray, htk: bool = False) -> np.ndarray:
    """Convert Hz to mel scale."""
    if htk:
        return 2595.0 * np.log10(1.0 + freq / 700.0)
    # Slaney formula
    f_min = 0.0
    f_sp = 200.0 / 3
    mel = (freq - f_min) / f_sp
    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_min) / f_sp
    logstep = np.log(6.4) / 27.0
    log_mask = freq >= min_log_hz
    mel[log_mask] = min_log_mel + np.log(freq[log_mask] / min_log_hz) / logstep
    return mel


def _mel_to_hz(mel: np.ndarray, htk: bool = False) -> np.ndarray:
    """Convert mel to Hz."""
    if htk:
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)
    f_min = 0.0
    f_sp = 200.0 / 3
    freq = f_min + f_sp * mel
    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_min) / f_sp
    logstep = np.log(6.4) / 27.0
    log_mask = mel >= min_log_mel
    freq[log_mask] = min_log_hz * np.exp(logstep * (mel[log_mask] - min_log_mel))
    return freq


def create_mel_filter_bank(
    sample_rate: int,
    n_fft: int,
    n_mels: int,
    fmin: float = 0.0,
    fmax: Optional[float] = None,  # noqa: UP045
    htk: bool = False,
    norm: Optional[str] = None,  # noqa: UP045
) -> np.ndarray:
    """Create a mel filter bank matching librosa.filters.mel output.

    Args:
        sample_rate: Audio sample rate
        n_fft: FFT size
        n_mels: Number of mel bands
        fmin: Minimum frequency
        fmax: Maximum frequency (default: sample_rate / 2)
        htk: Use HTK formula (default: Slaney)
        norm: Normalization type (None or "slaney")

    Returns:
        (n_mels, n_fft // 2 + 1) filter bank matrix
    """
    if fmax is None:
        fmax = float(sample_rate) / 2

    n_freqs = n_fft // 2 + 1
    fft_freqs = np.linspace(0, float(sample_rate) / 2, n_freqs)

    min_mel = _hz_to_mel(np.array([fmin]), htk=htk)[0]
    max_mel = _hz_to_mel(np.array([fmax]), htk=htk)[0]
    mels = np.linspace(min_mel, max_mel, n_mels + 2)
    mel_freqs = _mel_to_hz(mels, htk=htk)

    fdiff = np.diff(mel_freqs)
    ramps = np.subtract.outer(mel_freqs, fft_freqs)

    weights = np.zeros((n_mels, n_freqs))
    for i in range(n_mels):
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i + 2] / fdiff[i + 1]
        weights[i] = np.maximum(0, np.minimum(lower, upper))

    if norm == "slaney":
        enorm = 2.0 / (mel_freqs[2 : n_mels + 2] - mel_freqs[:n_mels])
        weights *= enorm[:, np.newaxis]

    return weights


class MelBandRoformerMLX(nn.Module):
    """
    MelBand-Roformer for music source separation.

    Uses mel-scale frequency bands instead of BS-Roformer's explicit linear
    frequency splits; overlapping mel bands share frequency bins, and their
    estimated masks are averaged back onto those shared bins (scatter-add).
    Reuses the transformer, attention, and mask-estimation primitives vendored
    above from upstream's BS-Roformer file.
    """

    def __init__(
        self,
        dim,
        *,
        depth,
        stereo=False,
        num_stems=1,
        time_transformer_depth=2,
        freq_transformer_depth=2,
        linear_transformer_depth=0,
        num_bands=60,
        dim_head=64,
        heads=8,
        attn_dropout=0.0,
        ff_dropout=0.0,
        mlp_expansion_factor=4,
        mask_estimator_depth=2,
        sample_rate=44100,
        stft_n_fft=2048,
        stft_hop_length=512,
        stft_win_length=2048,
        stft_normalized=False,
        chunk_seconds: float = 8.0,
        overlap_seconds: float = 1.0,
        match_input_audio_length: bool = False,
        **kwargs  # Accept and ignore other PyTorch-specific params
    ):
        super().__init__()

        self.dim = dim
        self.depth = depth
        self.stereo = stereo
        self.audio_channels = 2 if stereo else 1
        self.num_stems = num_stems
        self.mlp_expansion_factor = mlp_expansion_factor
        self.num_bands = num_bands
        self.sample_rate = sample_rate
        self.match_input_audio_length = match_input_audio_length

        # STFT
        self.stft_n_fft = stft_n_fft
        self.stft_hop_length = stft_hop_length
        self.stft_win_length = stft_win_length
        self.stft_normalized = stft_normalized
        self._stft_transform = get_transform_mlx(
            n_fft=stft_n_fft,
            hop_length=stft_hop_length,
            win_length=stft_win_length,
            window_fn="hann",
            window=None,
            periodic=True,
            center=True,
            normalized=stft_normalized,
        )

        self.chunk_seconds = float(chunk_seconds)
        self.overlap_seconds = float(overlap_seconds)

        # Build mel filter bank
        mel_fb = create_mel_filter_bank(
            sample_rate=sample_rate,
            n_fft=stft_n_fft,
            n_mels=num_bands,
        )

        # Match PyTorch: ensure DC and Nyquist are covered
        mel_fb[0, 0] = 1.0
        mel_fb[-1, -1] = 1.0

        freqs_per_band_mask = mel_fb > 0  # (num_bands, n_freqs)

        assert freqs_per_band_mask.any(axis=0).all(), (
            "Not all frequencies covered by mel bands"
        )

        num_freqs_per_band = freqs_per_band_mask.sum(axis=1).astype(np.int32)  # (num_bands,)
        num_bands_per_freq = freqs_per_band_mask.sum(axis=0).astype(np.int32)  # (n_freqs,)

        freq_indices_list = []
        for band_idx in range(num_bands):
            freq_idx = np.where(freqs_per_band_mask[band_idx])[0]
            freq_indices_list.append(freq_idx)
        freq_indices = np.concatenate(freq_indices_list)

        if stereo:
            freq_indices_expanded = np.stack(
                [freq_indices * 2, freq_indices * 2 + 1], axis=-1
            )
            freq_indices = freq_indices_expanded.reshape(-1)

        self.freq_indices = mx.array(freq_indices)
        self.num_freqs_per_band = mx.array(num_freqs_per_band)
        self.num_bands_per_freq = mx.array(num_bands_per_freq)

        freqs_per_bands_with_complex = tuple(
            int(2 * f * self.audio_channels)
            for f in num_freqs_per_band.tolist()
        )

        transformer_kwargs = dict(  # noqa: C408
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            ff_mult=mlp_expansion_factor,
            # This package's own torch MelBandRoformer has no trunk-level final
            # norm -- each Transformer normalizes its own output (see module
            # docstring point 2), so norm_output stays True here.
            norm_output=True,
        )

        rotary_embed = True

        for i in range(depth):
            linear_tran = None
            if linear_transformer_depth > 0:
                linear_tran = Transformer(
                    depth=linear_transformer_depth,
                    rotary_embed=rotary_embed,
                    linear_attn=False,
                    **transformer_kwargs,
                )
            time_tran = Transformer(
                depth=time_transformer_depth,
                rotary_embed=rotary_embed,
                **transformer_kwargs,
            )
            freq_tran = Transformer(
                depth=freq_transformer_depth,
                rotary_embed=rotary_embed,
                **transformer_kwargs,
            )
            setattr(self, f"layers_{i}", BSRoformerBlock(linear_tran, time_tran, freq_tran))

        # NOT UPSTREAM: no trunk-level final_norm -- see module docstring point 2.

        self.band_split = BandSplit(dim=dim, dim_inputs=freqs_per_bands_with_complex)

        for i in range(num_stems):
            setattr(
                self,
                f"mask_estimators_{i}",
                MaskEstimator(
                    dim=dim,
                    dim_inputs=freqs_per_bands_with_complex,
                    depth=mask_estimator_depth,
                    mlp_expansion_factor=mlp_expansion_factor,
                ),
            )

        if os.environ.get("MLX_ENABLE_COMPILE") == "1":
            self._forward_transformers = mx.compile(self._forward_transformers)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def __call__(self, raw_audio):
        """Forward pass: raw audio -> STFT -> mel gather -> transform -> scatter -> iSTFT.

        Args:
            raw_audio: (batch, channels, time) or (batch, time)

        Returns:
            Separated audio: (batch, num_stems, channels, time) or (batch, channels, time)
        """
        if raw_audio.ndim == 2:
            raw_audio = mx.expand_dims(raw_audio, axis=1)

        batch_size, channels, time_samples = raw_audio.shape

        fixed_len_env = os.environ.get("MLX_FIXED_CHUNK_SAMPLES")
        if fixed_len_env:
            fixed_len = int(fixed_len_env)
            if time_samples > fixed_len:
                raise ValueError(
                    f"Input length {time_samples} exceeds MLX_FIXED_CHUNK_SAMPLES={fixed_len}"
                )
            if time_samples < fixed_len:
                raw_audio = mx.pad(raw_audio, [(0, 0), (0, 0), (0, fixed_len - time_samples)])

        if (self.stereo and channels != 2) or (not self.stereo and channels != 1):
            raise ValueError(
                f"Config mismatch: stereo={self.stereo} but input has {channels} channel(s)"
            )

        audio_flat = rearrange(raw_audio, "b c t -> (b c) t")

        # NOT UPSTREAM: exact_zero_safe_rfft wraps this call -- see its docstring.
        # Upstream's mel_band_roformer.py calls mx.fft.rfft unguarded here.
        with exact_zero_safe_rfft():
            stft_complex = self._stft_transform.stft(audio_flat)  # (b*c, F, T) complex
            mx.eval(stft_complex)
        stft_real = mx.stack([stft_complex.real, stft_complex.imag], axis=-1)  # (b*c, F, T, 2)

        stft_repr = mx.reshape(
            stft_real,
            (batch_size, channels, stft_real.shape[1], stft_real.shape[2], 2),
        )
        stft_repr = mx.transpose(stft_repr, (0, 2, 1, 3, 4))  # (b, F, c, T, 2)
        stft_repr = mx.reshape(
            stft_repr,
            (batch_size, stft_repr.shape[1] * channels, stft_repr.shape[3], 2),
        )

        masks = self._forward_model(stft_repr)
        # masks: (b, n_stems, num_gathered_freqs, T, 2)

        n_freqs_total = self.stft_n_fft // 2 + 1
        if self.stereo:
            n_freqs_total *= 2

        _, num_stems, _, time_steps, _ = masks.shape
        masks_summed = mx.zeros(
            (batch_size, num_stems, n_freqs_total, time_steps, 2),
            dtype=masks.dtype,
        )
        masks_summed = masks_summed.at[:, :, self.freq_indices, :, :].add(masks)

        denom = self.num_bands_per_freq
        if self.stereo:
            denom = mx.repeat(denom, 2)
        denom = denom.astype(masks_summed.dtype).reshape(1, 1, -1, 1, 1)
        masks_averaged = masks_summed / mx.maximum(denom, mx.array(1e-8, dtype=masks_summed.dtype))

        stft_repr_expanded = mx.expand_dims(stft_repr, axis=1)  # (b, 1, F*c, T, 2)
        stft_complex = stft_repr_expanded[..., 0] + 1j * stft_repr_expanded[..., 1]
        mask_complex = masks_averaged[..., 0] + 1j * masks_averaged[..., 1]
        stft_masked = stft_complex * mask_complex  # (b, n, F*c, T)

        stft_masked = rearrange(
            stft_masked, "b n (f c) t -> (b n c) f t", c=self.audio_channels
        )

        original_length = raw_audio.shape[-1]
        istft_length = original_length if self.match_input_audio_length else None
        recon_audio = self._stft_transform.istft(stft_masked, length=istft_length)
        recon_audio = rearrange(
            recon_audio,
            "(b n c) t -> b n c t",
            b=batch_size,
            n=self.num_stems,
            c=self.audio_channels,
        )

        if self.num_stems == 1:
            recon_audio = rearrange(recon_audio, "b 1 c t -> b c t")

        return recon_audio

    # ------------------------------------------------------------------
    # Chunked inference (upstream's own convenience path -- unused by this
    # package's backend, which drives chunking itself in backends/mlx_backend.py
    # to reuse ChunkingPlan and stay overlap-add-identical with the Torch path;
    # kept for parity with upstream and for callers composing this model
    # directly)
    # ------------------------------------------------------------------

    def separate_audio_chunked(
        self,
        raw_audio: mx.array,
        *,
        sr: int = 44100,
        chunk_seconds: Optional[float] = None,  # noqa: UP045
        overlap_seconds: Optional[float] = None,  # noqa: UP045
        use_hann_window: bool = True,
        batch_hops: int = 1,
    ) -> mx.array:
        """Chunked overlap-add inference for long audio."""
        if raw_audio.ndim == 1:
            raw_audio = raw_audio[None, None, :]
        elif raw_audio.ndim == 2:
            if raw_audio.shape[0] in (1, 2):
                raw_audio = raw_audio[None, ...]
            else:
                raw_audio = raw_audio[:, None, :]
        elif raw_audio.ndim != 3:
            raise ValueError(f"Expected audio with 1-3 dims, got shape {tuple(raw_audio.shape)}")

        B, C, T = raw_audio.shape

        chunk_s = float(self.chunk_seconds if chunk_seconds is None else chunk_seconds)
        overlap_s = float(self.overlap_seconds if overlap_seconds is None else overlap_seconds)

        chunk_len = int(round(chunk_s * sr))  # noqa: RUF046
        overlap_len = int(round(overlap_s * sr))  # noqa: RUF046
        if chunk_len <= 0:
            raise ValueError(f"chunk_seconds too small -> chunk_len={chunk_len}")
        if overlap_len >= chunk_len:
            raise ValueError(f"overlap ({overlap_len}) must be < chunk_len ({chunk_len})")

        hop_len = chunk_len - overlap_len

        n_hops = int(math.ceil(max(T - overlap_len, 1) / hop_len))  # noqa: RUF046
        total_len = (n_hops - 1) * hop_len + chunk_len
        pad_len = total_len - T
        padded = mx.pad(raw_audio, [(0, 0), (0, 0), (0, pad_len)]) if pad_len > 0 else raw_audio

        if use_hann_window and chunk_len > 1:
            w = np.hanning(chunk_len).astype(np.float32)
        else:
            w = np.ones((chunk_len,), dtype=np.float32)
        w_mx = mx.array(w, dtype=mx.float32)
        w_view_single = w_mx.reshape(1, 1, -1)
        w_view_multi = w_mx.reshape(1, 1, 1, -1)

        if self.num_stems == 1:
            out_acc = mx.zeros((B, C, total_len), dtype=mx.float32)
            w_acc = mx.zeros((1, 1, total_len), dtype=mx.float32)
        else:
            out_acc = mx.zeros((B, self.num_stems, C, total_len), dtype=mx.float32)
            w_acc = mx.zeros((1, 1, 1, total_len), dtype=mx.float32)

        starts = [hop * hop_len for hop in range(n_hops)]
        eval_flush_interval = max(8, int(batch_hops) * 2)
        pending_updates = 0
        arange_chunk = mx.arange(chunk_len, dtype=mx.int32)
        all_starts_mx = mx.array(starts, dtype=mx.int32)
        all_gather_idx = all_starts_mx[:, None] + arange_chunk[None, :]
        use_gather_batching = env_enabled(
            "MLX_AUDIO_SEPARATOR_ROFORMER_CHUNK_GATHER_BATCHING",
            default_value=False,
        )

        for i in range(0, n_hops, batch_hops):
            hops = list(range(i, min(i + batch_hops, n_hops)))
            H = len(hops)
            if use_gather_batching:
                try:
                    gather_idx = all_gather_idx[i : i + H]
                    chunk_batch = mx.transpose(padded[:, :, gather_idx], (2, 0, 1, 3))
                    chunk_batch = chunk_batch.reshape(H * B, C, chunk_len)
                except Exception:  # noqa: BLE001
                    chunk_list = [padded[..., starts[h] : starts[h] + chunk_len] for h in hops]
                    chunk_batch = mx.concatenate(chunk_list, axis=0)
            else:
                chunk_list = [padded[..., starts[h] : starts[h] + chunk_len] for h in hops]
                chunk_batch = mx.concatenate(chunk_list, axis=0)

            batch_out = self(chunk_batch)

            if self.num_stems == 1:
                batch_out = batch_out.reshape(H, B, C, chunk_len)
                for j, hop in enumerate(hops):
                    start = starts[hop]
                    end = start + chunk_len
                    out_update = batch_out[j] * w_view_single
                    if _USE_SAFE_SLICE_ACCUMULATION:
                        start_mx = mx.array([start])
                        out_acc = mx.slice_update(
                            out_acc,
                            out_acc[..., start:end] + out_update,
                            start_mx,
                            axes=(out_acc.ndim - 1,),
                        )
                        w_acc = mx.slice_update(
                            w_acc,
                            w_acc[..., start:end] + w_view_single,
                            start_mx,
                            axes=(w_acc.ndim - 1,),
                        )
                    else:
                        out_acc = out_acc.at[..., start:end].add(out_update)
                        w_acc = w_acc.at[..., start:end].add(w_view_single)
            else:
                batch_out = batch_out.reshape(H, B, self.num_stems, C, chunk_len)
                for j, hop in enumerate(hops):
                    start = starts[hop]
                    end = start + chunk_len
                    out_update = batch_out[j] * w_view_multi
                    if _USE_SAFE_SLICE_ACCUMULATION:
                        start_mx = mx.array([start])
                        out_acc = mx.slice_update(
                            out_acc,
                            out_acc[..., start:end] + out_update,
                            start_mx,
                            axes=(out_acc.ndim - 1,),
                        )
                        w_acc = mx.slice_update(
                            w_acc,
                            w_acc[..., start:end] + w_view_multi,
                            start_mx,
                            axes=(w_acc.ndim - 1,),
                        )
                    else:
                        out_acc = out_acc.at[..., start:end].add(out_update)
                        w_acc = w_acc.at[..., start:end].add(w_view_multi)

            pending_updates += H
            if pending_updates >= eval_flush_interval:
                mx.eval(out_acc, w_acc)
                pending_updates = 0

        mx.eval(out_acc, w_acc)
        out_acc = out_acc / mx.maximum(w_acc, 1e-8)
        out_acc = out_acc[..., :T]
        mx.eval(out_acc)
        return out_acc

    def separate(self, wav: mx.array, *, sr: int = 44100) -> mx.array:
        """Convenience wrapper: handles shape normalization and chunking."""
        if wav.ndim == 1:
            wav_bct = wav[None, None, :]
            squeeze_b = True
        elif wav.ndim == 2:
            if wav.shape[0] in (1, 2):
                wav_bct = wav[None, ...]
                squeeze_b = True
            else:
                wav_bct = wav[:, None, :]
                squeeze_b = False
        elif wav.ndim == 3:
            wav_bct = wav
            squeeze_b = wav.shape[0] == 1
        else:
            raise ValueError(f"Expected wav with 1-3 dims, got shape {tuple(wav.shape)}")

        chunk_len = int(round(self.chunk_seconds * sr))  # noqa: RUF046
        if wav_bct.shape[-1] > chunk_len:
            out = self.separate_audio_chunked(wav_bct, sr=sr)
        else:
            out = self(wav_bct)

        if squeeze_b:
            out = out[0]
        return out

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _forward_transformers(self, x):
        """Process through transformer stack: (batch, time, bands, dim) -> same."""
        for i in range(self.depth):
            block = getattr(self, f"layers_{i}")

            if block.has_linear:
                x, ft_ps = pack([x], "b * d")
                x = block.linear_transformer(x)
                (x,) = unpack(x, ft_ps, "b * d")

            x = rearrange(x, "b t f d -> b f t d")
            x, ps = pack([x], "* t d")
            x = block.time_transformer(x)
            (x,) = unpack(x, ps, "* t d")

            x = rearrange(x, "b f t d -> b t f d")
            x, ps = pack([x], "* f d")
            x = block.freq_transformer(x)
            (x,) = unpack(x, ps, "* f d")

        # NOT UPSTREAM: no trunk-level final_norm call -- see module docstring
        # point 2. Each Transformer above already normalized its own output.
        return x

    def _estimate_masks(self, x):
        """Generate masks from transformer output."""
        masks = []
        for i in range(self.num_stems):
            estimator = getattr(self, f"mask_estimators_{i}")
            masks.append(estimator(x))
        masks = mx.stack(masks, axis=1)
        masks = rearrange(masks, "b n t (f c) -> b n f t c", c=2)
        return masks

    def _forward_model(self, stft_repr):
        """Gather mel freqs -> band split -> transform -> estimate masks."""
        x_gathered = mx.take(stft_repr, self.freq_indices, axis=1)
        x = rearrange(x_gathered, "b f t c -> b t (f c)")
        x = self.band_split(x)
        x = self._forward_transformers(x)
        return self._estimate_masks(x)
