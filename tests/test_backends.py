"""Backend seam contract: resolution, import purity, and honest failure.

These are offline and hardware-independent. They guard the two properties the
seam exists to protect -- that a requested backend is honoured or refused, never
silently swapped, and that the default import path stays free of optional
frameworks. Ported from bs-roformer-infer's identical test_backends.py (this
package's fork sibling), adapted for melband-roformer-infer's registry, which
currently declares no mask-estimator variations at all.
"""
import sys

import pytest

from mel_band_roformer.backends import (
    BACKEND_NAMES,
    DEFAULT_BACKEND,
    BackendUnavailable,
    get_backend,
    resolve_backend_name,
)


def test_default_and_none_resolve_to_torch():
    assert resolve_backend_name(None) == DEFAULT_BACKEND == "torch"
    assert resolve_backend_name("torch") == "torch"


def test_auto_falls_back_to_torch_when_nothing_accelerated_is_installed():
    """`auto` is the one place a fallback is what the caller asked for."""
    assert resolve_backend_name("auto") in BACKEND_NAMES


def test_unknown_backend_name_raises_value_error():
    with pytest.raises(ValueError):
        resolve_backend_name("cuda")


def test_unavailable_backend_raises_rather_than_substituting():
    """An explicit request is honoured or fails loudly -- never downgraded.

    A silent substitution is only ever discovered by noticing the wrong hardware
    was busy, which is exactly the failure the org's device contract forbids.
    """
    from mel_band_roformer.backends import mlx_backend

    if mlx_backend.MLXBackend.is_available():
        pytest.skip("MLX extra is actually installed on this host")
    with pytest.raises(BackendUnavailable):
        resolve_backend_name("mlx")


def test_torch_backend_satisfies_the_protocol_surface():
    backend = get_backend("torch")
    assert backend.name == "torch"
    assert backend.is_available() is True
    for method in ("separate", "release", "resolved_device"):
        assert hasattr(backend, method), f"TorchBackend is missing {method}"


def test_mlx_refuses_a_variation_it_has_no_head_for():
    """Refusing beats mis-running.

    This package's registry declares no variations at all today, so this backend
    only knows the stock head -- anything else must refuse by name rather than
    silently constructing the stock MaskEstimator (upstream's own model swallows
    unknown constructor arguments and would do exactly that).
    """
    from mel_band_roformer.backends.mlx_backend import MLXBackend

    with pytest.raises(BackendUnavailable) as excinfo:
        MLXBackend.assert_supports_variation("no_such_head")
    assert "no_such_head" in str(excinfo.value)


@pytest.mark.parametrize("variation", [None, "", "mlp"])
def test_mlx_accepts_the_stock_mask_estimator(variation):
    from mel_band_roformer.backends.mlx_backend import MLXBackend

    MLXBackend.assert_supports_variation(variation)


def test_every_registry_variation_has_an_mlx_head():
    """The org's per-checkpoint MLX-coverage goal, asserted rather than assumed:
    every mask-estimator variation any registry checkpoint declares must be
    buildable under MLX. melband-roformer-infer's 21-model registry declares
    none today, so this currently passes vacuously -- it exists so the day a
    variant checkpoint is added, this fails instead of silently mis-running."""
    from pathlib import Path

    import tomllib

    from mel_band_roformer.backends.mlx_backend import supported_variations

    registry = Path("src/mel_band_roformer/config/checkpoints.toml")
    if not registry.exists():
        pytest.skip("registry TOML not found from this working directory")
    models = tomllib.loads(registry.read_text())["models"]
    declared = {m.get("variation") for m in models.values()} - {None, ""}
    missing = sorted(declared - supported_variations())
    assert not missing, f"registry variations with no MLX head: {missing}"


def test_mlx_refuses_chunking_that_is_not_hop_aligned():
    """The chunked path's length bookkeeping assumes chunk_size % hop == 0."""
    from ml_collections import ConfigDict

    from mel_band_roformer.backends.mlx_backend import MLXBackend

    config = ConfigDict(
        {
            "inference": {"chunk_size": 588801, "num_overlap": 2},
            "model": {"stft_hop_length": 512},
        }
    )
    with pytest.raises(BackendUnavailable):
        MLXBackend._reject_unaligned_chunking(config)


def test_auto_skips_a_backend_that_cannot_run_this_checkpoint(monkeypatch):
    """`auto` means "pick one that works", so it must not settle on a backend that
    has no head for this checkpoint and then fail at construction -- Torch would
    simply have worked."""
    from mel_band_roformer.backends import mlx_backend

    monkeypatch.setattr(mlx_backend.MLXBackend, "is_available", classmethod(lambda cls: True))
    assert resolve_backend_name("auto", variation=None) == "mlx"
    assert resolve_backend_name("auto", variation="no_such_head") == "torch"


def test_explicit_backend_is_never_downgraded_for_an_unsupported_checkpoint(monkeypatch):
    """An explicit request is honoured or refused -- `auto`'s fallback is not it."""
    from mel_band_roformer.backends import mlx_backend

    monkeypatch.setattr(mlx_backend.MLXBackend, "is_available", classmethod(lambda cls: True))
    with pytest.raises(BackendUnavailable):
        resolve_backend_name("mlx", variation="no_such_head")


def test_importing_the_package_does_not_pull_in_an_optional_framework():
    """`pip install melband-roformer-infer` must stay MLX-free and import-clean.

    Runs in a subprocess with a fresh sys.modules, not in-process. On a host
    where the `[mlx]` extra really is installed (as this suite's own
    `test_auto_*`/`resolve_backend_name("auto")` calls above legitimately do,
    by probing `is_available()`), Python's import cache keeps `mlx`/`mlx_spectro`
    in `sys.modules` for the rest of the process once any earlier test
    successfully imports them -- an in-process check would then see that
    pre-existing pollution and report a false leak that has nothing to do with
    `import mel_band_roformer` itself.
    """
    import subprocess

    probe = (
        "import sys; import mel_band_roformer; "
        "optional = {'mlx', 'mlx_spectro', 'mlx_audio_io', 'mlx_audio_separator'}; "
        "leaked = sorted({m.split('.')[0] for m in sys.modules} & optional); "
        "sys.exit(1 if leaked else 0)"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"import mel_band_roformer pulled in an optional framework "
        f"(stdout={result.stdout!r} stderr={result.stderr!r})"
    )
