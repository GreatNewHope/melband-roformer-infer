"""PyTorch -> MLX weight conversion for MelBandRoformerMLX, plus a strict load gate.

`convert_torch_to_mlx_weights` renames and restructures a PyTorch `state_dict`'s
keys to match the MLX module tree built by `model.py` -- gamma->weight,
`layers.N.K` block indices, Sequential `layers.N` insertion, band-split/
mask-estimator submodule naming. This package's torch `MelBandRoformer` has no
linear-attention branch and no mask-estimator variation (unlike bs-roformer-infer,
which has both), so the block-index mapping here is simpler: `layers.{i}.0` is
always the time transformer and `layers.{i}.1` is always the freq transformer.

Upstream's own loader calls `model.load_weights(list(weights.items()),
strict=False)`, which **silently discards** any key that doesn't match the module
tree: a checkpoint can "load" with whole layers left at random initialization and
produce plausible-looking garbage with no error. `load_converted_weights()` below
diffs the model's own parameter keys (via `mlx.utils.tree_flatten`) against the
converted weight keys and raises a `ValueError` naming the mismatch *before*
calling `load_weights` -- callers should use this instead of calling
`load_weights` directly. It caught the vendored model's stray `final_norm` (see
`model.py`'s docstring) within minutes of being wired against the real checkpoint.

`convert_torch_to_mlx_weights`'s renaming rules are adapted from
bs-roformer-infer's identical converter for the shared BS-Roformer/MelBand-
Roformer primitives (this package's fork sibling); `load_converted_weights` is
this org's own addition on both packages, not upstream code.

Reads: mlx.core, mlx.nn, mlx.utils (tree_flatten), numpy
"""

import logging
import re
from typing import Any

import mlx.core as mx
import numpy as np
from mlx import nn
from mlx.utils import tree_flatten

logger = logging.getLogger(__name__)

_TRANSFORMER_NAMES = ("time_transformer", "freq_transformer")


def convert_torch_to_mlx_weights(state_dict: dict[str, Any]) -> dict[str, mx.array]:
    """Convert a PyTorch MelBandRoformer state dict to MLX weights format.

    Handles:
    1. Parameter name mapping (gamma -> weight for norms)
    2. Module path restructuring for MLX module tree
    3. Sequential layer indexing (module.N -> module.layers.N)
    4. Band-split / mask-estimator submodule naming
    """
    mlx_weights = {}

    for key, value in state_dict.items():
        # rotary_embed is a shared nn.Module referenced from every attention
        # layer in one Transformer stack; its buffer is skipped (mx.fast.rope
        # computes the same frequencies from theta=10000 directly).
        if "rotary_embed.freqs" in key:
            continue

        numpy_weight = _to_numpy(value)
        mlx_key = key.replace(".gamma", ".weight")

        if "band_split.to_features." in mlx_key:
            parts = mlx_key.split(".")
            if len(parts) >= 5 and parts[2].isdigit():
                band_idx = parts[2]
                submodule_idx = parts[3]
                param_name = ".".join(parts[4:])
                if submodule_idx == "0":
                    mlx_key = f"band_split.to_features_{band_idx}.norm.{param_name}"
                elif submodule_idx == "1":
                    mlx_key = f"band_split.to_features_{band_idx}.linear.{param_name}"

        # Main block: layers.{i}.{0|1}.{rest} -> layers_{i}.{time|freq}_transformer.{rest}
        main_block_match = re.match(r"^layers\.(\d+)\.([01])\.(.+)$", mlx_key)
        if main_block_match:
            block_idx = main_block_match.group(1)
            transformer_idx = int(main_block_match.group(2))
            rest = main_block_match.group(3)
            mlx_key = f"layers_{block_idx}.{_TRANSFORMER_NAMES[transformer_idx]}.{rest}"

        # Individual transformer layer: .layers.{j}.{0|1}. -> .layers_{j}.{attn|ff}.
        transformer_match = re.search(r"(\.layers)\.(\d+)\.([01])\.", mlx_key)
        if transformer_match:
            prefix = mlx_key[: transformer_match.start()]
            layer_idx = transformer_match.group(2)
            submodule = transformer_match.group(3)
            suffix = mlx_key[transformer_match.end() :]

            if submodule == "0":
                mlx_key = f"{prefix}.layers_{layer_idx}.attn.{suffix}"
            elif submodule == "1":
                mlx_key = f"{prefix}.layers_{layer_idx}.ff.{suffix}"

        if "mask_estimators." in mlx_key:
            mlx_key = re.sub(r"mask_estimators\.(\d+)", r"mask_estimators_\1", mlx_key)
            mlx_key = re.sub(r"to_freqs\.(\d+)\.0\.", r"to_freqs_\1.", mlx_key)

        mlx_key = re.sub(r"\.net\.(\d+)\.", r".net.layers.\1.", mlx_key)
        mlx_key = re.sub(r"\.to_out\.(\d+)\.", r".to_out.layers.\1.", mlx_key)
        mlx_key = re.sub(r"(to_freqs_\d+)\.(\d+)\.", r"\1.layers.\2.", mlx_key)

        mlx_weights[mlx_key] = mx.array(numpy_weight)

    logger.debug(f"Converted {len(mlx_weights)} tensors from PyTorch to MLX format")
    return mlx_weights


def _to_numpy(value) -> np.ndarray:
    """Convert a weight value to numpy array, handling torch tensors."""
    try:
        return value.cpu().numpy()
    except AttributeError:
        return np.array(value)


#: MelBandRoformerMLX's mel-filter-bank bookkeeping (freq_indices,
#: num_freqs_per_band, num_bands_per_freq): plain mx.array attributes on the
#: module, which MLX's parameter tree walk cannot distinguish from a learned
#: weight. They are the MLX analogue of the torch model's own `register_buffer(
#: ..., persistent=False)` arrays of the same names -- deterministic functions
#: of (sample_rate, stft_n_fft, num_bands) computed at construction time, never
#: part of a training checkpoint. Excluded from the audit below for the same
#: reason torch's state_dict() never includes them.
NON_CHECKPOINT_MODEL_KEYS = frozenset(
    {"freq_indices", "num_freqs_per_band", "num_bands_per_freq"}
)


def load_converted_weights(
    model: nn.Module,
    mlx_weights: dict[str, mx.array],
    *,
    ignore_model_keys: frozenset[str] = NON_CHECKPOINT_MODEL_KEYS,
) -> None:
    """Load `mlx_weights` into `model`, refusing a silent partial load.

    `model.load_weights(..., strict=False)` on its own accepts any degree of
    mismatch between the checkpoint and the module tree, dropping whatever
    doesn't line up without a warning. This checks first: every one of the
    model's own parameter keys (from `mlx.utils.tree_flatten(model.parameters())`,
    minus `ignore_model_keys`) must be present in `mlx_weights`, and every key in
    `mlx_weights` must be consumed by the model -- otherwise a `ValueError` is
    raised naming counts and up to 5 example keys on each side, so a conversion
    bug or a mismatched checkpoint fails loudly instead of loading a partially-
    random model. This caught two real bugs in this package's own vendored model
    within minutes of being wired against a real state_dict: a stray trunk-level
    `final_norm` upstream applies that this package's torch model never trained,
    and an `MLP()` hidden-layer-count off-by-one inherited from copying
    bs-roformer-infer's vendored helper instead of matching this package's own
    torch `MLP()`. See model.py's module docstring and `MLP()`'s docstring.
    """
    model_keys = {key for key, _ in tree_flatten(model.parameters())} - ignore_model_keys
    weight_keys = set(mlx_weights.keys())

    unmatched_model = sorted(model_keys - weight_keys)
    dropped_weights = sorted(weight_keys - model_keys)

    if unmatched_model or dropped_weights:
        parts = []
        if unmatched_model:
            example = ", ".join(unmatched_model[:5])
            parts.append(
                f"{len(unmatched_model)} model parameters unmatched (e.g. {example})"
            )
        if dropped_weights:
            example = ", ".join(dropped_weights[:5])
            parts.append(
                f"{len(dropped_weights)} converted tensors dropped (e.g. {example})"
            )
        raise ValueError("MLX weight conversion incomplete: " + ", ".join(parts))

    model.load_weights(list(mlx_weights.items()), strict=False)
