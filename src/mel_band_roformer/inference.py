"""CLI entry point for Mel-Band Roformer inference -- folder-batch separation, chunked overlap-add.

Weights auto-resolve: when --model_path/--config_path are omitted, the requested
registry model (default: the recommended Kim vocals model) is looked up in the
models dir ($MELBAND_ROFORMER_MODELS_PATH > ~/.cache/melband-roformer-infer >
legacy ./models) and downloaded sha256-verified on first use via
download.ensure_model_assets -- explicit paths always win and skip all of that.
separate_folder_with() owns everything backend-agnostic (folder iteration, stem
naming, instrumental derivation, the manifest) so no backend can drift on any of
it; run_folder() keeps its exact signature and is the Torch entry into it -- it
also returns the JSON-serializable manifest of the files it actually wrote, so
Python callers can consume exact output paths without guessing from model
defaults or filename conventions, while the CLI keeps its prior behavior by
ignoring that return value.
Training configs sometimes embed `!!python/tuple` YAML tags; SafeLoaderWithTuple
downgrades those to plain lists so `yaml.load` never has to execute an arbitrary
Python-object constructor, and utils.get_model_from_config converts the needed
params back to tuples afterward. `_resolve_device`/`_select_device` resolve
`None`/""/"auto" to cuda-else-cpu (legacy behaviour) and accept explicit `cpu`,
`cuda`, `cuda:N`, and `mps`, raising on an explicitly requested accelerator that
is unavailable rather than silently downgrading it.

Reads: .backends (resolve_backend_name, get_backend), .backends.torch_backend
(TorchBackend), .utils (get_model_from_config, load_checkpoint_state),
.download (ensure_model_assets), .checkpoints (checkpoint_metadata),
.model_registry (DEFAULT_MODEL), yaml, ml_collections, torch
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Iterable, TypedDict

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import yaml
from ml_collections import ConfigDict
from tqdm import tqdm

from .checkpoints import checkpoint_metadata
from .download import ensure_model_assets
from .model_registry import DEFAULT_MODEL
from .utils import get_model_from_config, load_checkpoint_state


class SafeLoaderWithTuple(yaml.SafeLoader):
    """YAML loader that treats !!python/tuple as a list (which utils.py converts to tuple)."""
    pass


def _tuple_constructor(loader, node):
    """Convert !!python/tuple to a list; utils.py will convert to tuple later."""
    return loader.construct_sequence(node)


SafeLoaderWithTuple.add_constructor('tag:yaml.org,2002:python/tuple', _tuple_constructor)


def _ensure_wav_inputs(input_folder: Path) -> list[Path]:
    if not input_folder.exists():
        raise FileNotFoundError(f"Input folder not found: {input_folder}")
    wav_files = sorted(input_folder.glob("*.wav"))
    if not wav_files:
        raise FileNotFoundError(f"No .wav files found in {input_folder}")
    return wav_files


def _resolve_output_dir(store_dir: Path) -> Path:
    store_dir.mkdir(parents=True, exist_ok=True)
    return store_dir


def _format_iterable(paths: Iterable[Path], verbose: bool) -> Iterable[Path]:
    return tqdm(paths, desc="Tracks", unit="track") if not verbose else paths


class OutputManifestEntry(TypedDict):
    """JSON-serializable record of one file written by run_folder()."""

    input_path: str
    track_id: str
    output_id: str
    output_path: str


OutputManifest = list[OutputManifestEntry]


def _resolve_output_ids(config: ConfigDict) -> list[str]:
    instruments = [str(instr) for instr in config.training.instruments]
    target_instrument = getattr(config.training, "target_instrument", None)
    if target_instrument is not None:
        return [str(target_instrument)]
    return instruments


def _resolve_residual_output_id(config: ConfigDict) -> str | None:
    target_instrument = getattr(config.training, "target_instrument", None)
    instruments = [str(instr) for instr in config.training.instruments]
    if target_instrument is None:
        if len(instruments) == 1 and instruments[0].lower() == "vocals":
            return "instrumental"
        return None

    target = str(target_instrument)
    complements = [instr for instr in instruments if instr.lower() != target.lower()]

    if complements:
        complement = complements[0]
        if target.lower() == "vocals" and complement.lower() == "other":
            return "instrumental"
        return complement

    if target.lower() == "vocals":
        return "instrumental"

    return None


def _record_written_output(
    manifest: OutputManifest,
    *,
    input_path: Path,
    output_id: str,
    output_path: Path,
) -> None:
    manifest.append(
        {
            "input_path": str(input_path),
            "track_id": input_path.stem,
            "output_id": output_id,
            "output_path": str(output_path),
        }
    )


def run_folder(model, args, config, device, verbose: bool = False) -> OutputManifest:
    """Torch entry point: separate every WAV in a folder. Signature unchanged.

    Delegates the Torch-specific work to TorchBackend and the backend-agnostic
    work -- folder iteration, stem naming, instrumental derivation, the manifest --
    to separate_folder_with(), so any backend drives the identical output logic.
    """
    from .backends.torch_backend import TorchBackend

    model.eval()
    backend = TorchBackend(model, config, device)
    return separate_folder_with(backend.separate, args, config, verbose=verbose)


def separate_folder_with(separate, args, config, verbose: bool = False) -> OutputManifest:
    """Backend-agnostic folder run: read, delegate one mixture, write, manifest.

    `separate` receives a `(channels, samples)` float32 array and returns a mapping
    of stem id to an array of the same shape. Everything a stem's filename, the
    derived residual stem, and the returned manifest depend on is decided here,
    once, so no backend can drift on any of it.
    """
    start_time = time.time()

    input_folder = Path(args.input_folder).expanduser()
    store_dir = _resolve_output_dir(Path(args.store_dir).expanduser())
    all_mixtures_path = _ensure_wav_inputs(input_folder)
    total_tracks = len(all_mixtures_path)
    print(f"Total tracks found: {total_tracks}")

    instruments = _resolve_output_ids(config)

    iterable = _format_iterable(all_mixtures_path, verbose)
    manifest: OutputManifest = []

    for track_number, path in enumerate(iterable, 1):
        print(f"\nProcessing track {track_number}/{total_tracks}: {path.name}")

        mix, sr = sf.read(path)
        original_mono = False
        if len(mix.shape) == 1:
            original_mono = True
            mix = np.stack([mix, mix], axis=-1)

        res = separate(mix.T)

        for instr in instruments:
            vocals_output = res[instr].T
            if original_mono:
                vocals_output = vocals_output[:, 0]

            vocals_path = store_dir / f"{path.stem}_{instr}.wav"
            sf.write(vocals_path, vocals_output, sr, subtype="FLOAT")
            _record_written_output(
                manifest,
                input_path=path,
                output_id=instr,
                output_path=vocals_path,
            )

        residual_output_id = _resolve_residual_output_id(config)
        if residual_output_id and instruments:
            vocals_output = res[instruments[0]].T
            if original_mono:
                vocals_output = vocals_output[:, 0]

            original_mix, _ = sf.read(path)
            instrumental = original_mix - vocals_output

            instrumental_path = store_dir / f"{path.stem}_{residual_output_id}.wav"
            sf.write(instrumental_path, instrumental, sr, subtype="FLOAT")
            _record_written_output(
                manifest,
                input_path=path,
                output_id=residual_output_id,
                output_path=instrumental_path,
            )

    time.sleep(1)
    print("Elapsed time: {:.2f} sec".format(time.time() - start_time))
    return manifest


def proc_folder(args):
    parser = argparse.ArgumentParser(description="Mel-Band Roformer inference runner")
    parser.add_argument("--model_type", type=str, default="mel_band_roformer")
    parser.add_argument("--config_path", type=Path, default=None,
                        help="path to config yaml file (omit to auto-resolve from --model)")
    parser.add_argument("--model_path", type=Path, default=None,
                        help="path to the .ckpt weights (omit to auto-resolve from --model)")
    parser.add_argument("--model", type=str, default=None,
                        help="registry model slug/name to auto-resolve when paths are omitted "
                             f"(default: {DEFAULT_MODEL})")
    parser.add_argument("--models_dir", type=Path, default=None,
                        help="directory holding downloaded model assets "
                             "(default: $MELBAND_ROFORMER_MODELS_PATH or ~/.cache/melband-roformer-infer; "
                             "a legacy ./models directory is also searched)")
    parser.add_argument("--input_folder", type=Path, required=True, help="folder with songs to process")
    parser.add_argument("--store_dir", type=Path, default=Path("outputs"), help="path to store model outputs")
    parser.add_argument("--device", type=str, default=None, help="torch device string, defaults to auto")
    parser.add_argument("--backend", type=str, default=None,
                        help="compute backend: 'torch' (default), 'mlx', or 'auto'")
    parser.add_argument("--device_ids", nargs='+', type=int, help='optional list of gpu ids for DataParallel')
    if args is None:
        args = parser.parse_args()
    elif isinstance(args, argparse.Namespace):
        # Already parsed, use as is
        pass
    else:
        args = parser.parse_args(args)

    from .backends import get_backend, resolve_backend_name

    # Availability first, so an unavailable backend fails before a checkpoint is
    # downloaded and verified rather than after.
    resolve_backend_name(getattr(args, "backend", None))

    _resolve_model_assets(args, parser)

    # Resolve for real now that the checkpoint's variation is known: `auto` must
    # be able to skip a backend that has no head for this model.
    backend_name = resolve_backend_name(
        getattr(args, "backend", None), variation=getattr(args, "model_variation", None)
    )

    torch.backends.cudnn.benchmark = True

    with open(args.config_path) as f:
        config = ConfigDict(yaml.load(f, Loader=SafeLoaderWithTuple))

    if backend_name == "torch":
        model = get_model_from_config(args.model_type, config)
        print(f"Using model weights: {args.model_path}")
        model.load_state_dict(load_checkpoint_state(args.model_path, map_location=torch.device("cpu")))

        device = _select_device(args)

        if args.device_ids:
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is required for --device_ids usage")
            model = nn.DataParallel(model, device_ids=args.device_ids).to(device)
        else:
            model = model.to(device)

        backend = get_backend(backend_name)(model, config, device)
    else:
        # MLX builds and converts its own model straight from the checkpoint --
        # it never touches get_model_from_config/DataParallel/.to(device), which
        # are Torch-specific.
        print(f"Using model weights: {args.model_path}")
        backend = get_backend(backend_name).from_checkpoint(
            config=config,
            checkpoint_path=args.model_path,
            variation=getattr(args, "model_variation", None),
            device=_select_mlx_device(getattr(args, "device", None)),
        )

    return separate_folder_with(backend.separate, args, config, verbose=False)


def _resolve_model_assets(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Fill in args.model_path/args.config_path, auto-downloading when both are omitted."""
    model_path = getattr(args, "model_path", None)
    config_path = getattr(args, "config_path", None)
    if (model_path is None) != (config_path is None):
        parser.error("--model_path and --config_path must be given together "
                     "(or both omitted to auto-resolve)")
    if model_path is not None:
        if getattr(args, "model", None):
            parser.error("--model selects a registry model to auto-resolve; "
                         "it cannot be combined with explicit --model_path/--config_path")
        args.model_variation = None
        return
    model_key = getattr(args, "model", None) or DEFAULT_MODEL
    args.model_path, args.config_path = ensure_model_assets(
        model_key, models_dir=getattr(args, "models_dir", None)
    )
    args.model = model_key
    args.model_variation = _safe_checkpoint_metadata(model_key).get("variation")


def _safe_checkpoint_metadata(model_key: str) -> dict:
    """checkpoint_metadata(), but tolerant of models outside the TOML registry.

    Most of this package's 99-model legacy JSON registry (model_registry.py)
    predates config/checkpoints.toml's smaller, sha256-verified 21-model set, so
    a valid --model key here can still miss the TOML lookup.
    """
    try:
        return checkpoint_metadata(model_key)
    except KeyError:
        return {}


def _resolve_device(device: str | torch.device | None) -> torch.device:
    """Resolve legacy auto selection and validate explicit device requests."""
    if device is None or device == "" or device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        print("CUDA is not available. Falling back to CPU. This will be slow.")
        return torch.device("cpu")

    resolved = torch.device(device)
    if resolved.type == "cpu" and resolved.index is None:
        return resolved
    if resolved.type == "mps" and resolved.index is None:
        mps = getattr(torch.backends, "mps", None)
        if mps is None or not mps.is_available():
            raise RuntimeError("MPS was explicitly requested but is not available")
        return resolved
    if resolved.type != "cuda":
        raise ValueError("device must be None, 'auto', 'cpu', 'cuda', 'cuda:N', or 'mps'")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA was explicitly requested but is not available")
    if resolved.index is not None and resolved.index >= torch.cuda.device_count():
        raise RuntimeError(f"CUDA device index {resolved.index} is not available")
    return resolved


def _select_device(args: argparse.Namespace) -> torch.device:
    return _resolve_device(args.device)


def mps_available() -> bool:
    """True when this torch build exposes a usable Apple Silicon MPS backend."""
    backend = getattr(torch.backends, "mps", None)
    return bool(backend is not None and backend.is_available())


def _select_mlx_device(device) -> str:
    """Validate a device request against the MLX backend's resolution table.

    Per the org's `backend` x `device` contract (bs-roformer-infer's
    brain/architecture.md, reused verbatim here): `backend="mlx"` owns its own
    execution target and accepts only `None`, `"auto"`, or `"mps"` -- an explicit
    Torch device string it cannot honour (e.g. `"cuda"`) is refused rather than
    silently reinterpreted.
    """
    if device is None or device in ("", "auto", "mps"):
        return "mps"
    raise ValueError(
        f"backend='mlx' only accepts device None, 'auto', or 'mps'; got {device!r} "
        f"-- device keeps its Torch meaning and is not overloaded to select MLX"
    )


def main():
    """Main entry point for the mel-band-roformer inference script."""
    proc_folder(None)


if __name__ == "__main__":
    main()
