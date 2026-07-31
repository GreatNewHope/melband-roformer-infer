"""Thin barrel for the vendored MLX MelBand-RoFormer backend -- lazy on purpose.

`mel_band_roformer.mlx` is only ever imported on demand by the (caller-owned) MLX
compute backend, never by the package's default import path, so `import
mel_band_roformer` stays MLX-free even with this subpackage present. Attribute
access is deferred via `__getattr__` so that merely importing this package
(e.g. for introspection) doesn't eagerly import `mlx.core`/`mlx.nn` -- the
cost of that import is paid only when a name is actually used.

Reads: .model (MelBandRoformerMLX, lazily), .convert (convert_torch_to_mlx_weights,
load_converted_weights, lazily)
"""

from __future__ import annotations

__all__ = [
    "MelBandRoformerMLX",
    "convert_torch_to_mlx_weights",
    "load_converted_weights",
]


def __getattr__(name: str):
    if name == "MelBandRoformerMLX":
        from .model import MelBandRoformerMLX

        return MelBandRoformerMLX
    if name in ("convert_torch_to_mlx_weights", "load_converted_weights"):
        from . import convert

        return getattr(convert, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
