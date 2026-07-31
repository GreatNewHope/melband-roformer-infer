"""Torch-vs-MLX output parity on the real checkpoint, end to end, including silence.

Marked ``realweights`` and deselected by default: needs the MLX extra, an Apple
Silicon Mac, and the default checkpoint already on disk. It never downloads.

Run explicitly:  pytest -m realweights tests/test_mlx_parity.py -v

The silence case is the point of this file. Every track's final chunk is padded
or simply quiet (music has rests), so a benchmark built only from clean signal
would ship a wrong answer that measures beautifully. Root cause and the
workaround it guards live in ``mel_band_roformer/mlx/model.py::exact_zero_safe_rfft``
(ported from bs-roformer-infer's identical fix for the same MLX 0.31.2 kernel bug
on the sibling BS-Roformer architecture).

Unlike bs-roformer-infer's equivalent file (which compares raw module forward
passes for the silence cases and only exercises the full session API on a
separate, silence-free fixture), every case here runs end to end through the
public ``MelBandRoformerSession`` API on a real WAV file -- read, chunk, demix,
write -- so the parity numbers below reflect exactly what a caller gets, not
just what the model math produces in isolation.

Recorded measurement (2026-07-31, arm64 host, torch/MPS vs mlx 0.31.2), max abs
error end to end through MelBandRoformerSession on a real WAV file, vocals stem:

    clean signal        8.428e-08
    zero-padded tail     1.807e-07
    near-silent tail     4.889e-09

Remove-the-fix validation performed by hand (monkeypatching
`mel_band_roformer.mlx.model.exact_zero_safe_rfft` to `contextlib.nullcontext`,
not by editing this file's fixture): with the fix disabled the same three cases
measured 2.618e-03 / 4.487e-02 / 1.029e-04 -- the zero-padded case alone degrades
by roughly 250000x. Restored immediately after.
"""
import numpy as np
import pytest
import soundfile as sf

from mel_band_roformer.download import ensure_model_assets
from mel_band_roformer.inference import mps_available

pytestmark = pytest.mark.realweights

MODEL = "melband-roformer-kim-vocals"
SEED = 1
SAMPLE_RATE = 44100
#: This model's packaged config: chunk_size=352800 (8s @ 44100Hz), num_overlap=2.
#: A track exactly chunk_size long keeps this fixture to one chunk with no
#: reflect-padding at the edges (border == chunk_size/2, and the top-level pad
#: only fires when length > 2*border) -- so the silence in "zeros"/"near_silent"
#: below is exactly the WAV content, not an artifact of the chunker's own edge
#: handling.
CHUNK_SIZE = 352800
SIGNAL_SAMPLES = 220500
MAX_ABS_TOLERANCE = 1e-5


def _mlx_available():
    try:
        import mlx.core  # noqa: F401
        import mlx_spectro  # noqa: F401
    except ImportError:
        return False
    return True


def _make_track(tail: str) -> np.ndarray:
    """One chunk_size-length stereo track: real signal up front, `tail` after it."""
    rng = np.random.default_rng(SEED)
    track = (rng.standard_normal((CHUNK_SIZE, 2)) * 0.1).astype(np.float32)
    if tail == "zeros":
        track[SIGNAL_SAMPLES:, :] = 0.0
    elif tail == "near_silent":
        track[SIGNAL_SAMPLES:, :] *= 1e-6
    return track


@pytest.fixture(scope="module")
def checkpoint_ready():
    if not _mlx_available():
        pytest.skip("MLX extra not installed: pip install 'melband-roformer-infer[mlx]'")
    if not mps_available():
        pytest.skip("needs an Apple Silicon Mac and an arm64 torch build")
    try:
        ensure_model_assets(MODEL, download_missing=False)
    except (FileNotFoundError, KeyError):
        pytest.skip(f"{MODEL} is not cached; run melband-roformer-download first")


@pytest.mark.parametrize("tail", ["signal", "zeros", "near_silent"])
def test_mlx_matches_torch_end_to_end_including_silence(checkpoint_ready, tail, tmp_path):
    """Torch-vs-MLX agreement through the real public API: session.infer() on a
    real WAV file, not a bare module forward pass."""
    from mel_band_roformer import MelBandRoformerSession

    inputs = tmp_path / "in"
    inputs.mkdir()
    sf.write(inputs / f"{tail}.wav", _make_track(tail), SAMPLE_RATE, subtype="FLOAT")

    produced = {}
    for backend, device in (("torch", "mps"), ("mlx", None)):
        with MelBandRoformerSession(backend=backend, device=device) as session:
            manifest = session.infer(inputs, store_dir=tmp_path / backend)
        produced[backend] = {
            entry["output_id"]: sf.read(entry["output_path"])[0] for entry in manifest
        }

    assert set(produced["torch"]) == set(produced["mlx"])
    for output_id, reference in produced["torch"].items():
        candidate = produced["mlx"][output_id]
        worst = float(np.abs(reference - candidate).max())
        assert worst < MAX_ABS_TOLERANCE, (
            f"tail={tail} output={output_id}: Torch-vs-MLX max abs {worst:.3e} "
            f"exceeds {MAX_ABS_TOLERANCE:.0e}. If this fired only for a silent "
            f"tail, suspect exact_zero_safe_rfft in mel_band_roformer/mlx/model.py "
            f"-- investigate rather than widen the tolerance"
        )
