#!/usr/bin/env python3
"""Strict-load and short-forward probe for every direct pcunwa MelBand checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from ml_collections import ConfigDict

from mel_band_roformer import MODEL_REGISTRY, ensure_model_assets
from mel_band_roformer.checkpoints import load_checkpoints
from mel_band_roformer.inference import SafeLoaderWithTuple
from mel_band_roformer.utils import get_model_from_config


def pcunwa_slugs():
    data = load_checkpoints()["models"]
    return [
        slug
        for slug, model in data.items()
        if any(
            artifact["kind"] == "checkpoint"
            and "huggingface.co/pcunwa/" in artifact["url"]
            for artifact in model["artifacts"]
        )
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", dest="models")
    parser.add_argument("--models-dir", type=Path)
    parser.add_argument("--samples", type=int, default=4096)
    parser.add_argument("--skip-forward", action="store_true")
    args = parser.parse_args()
    slugs = args.models or pcunwa_slugs()

    for slug in slugs:
        entry = MODEL_REGISTRY.get(slug)
        checkpoint, config_path = ensure_model_assets(entry, models_dir=args.models_dir)
        with config_path.open() as handle:
            config = ConfigDict(yaml.load(handle, Loader=SafeLoaderWithTuple))
        model = get_model_from_config("mel_band_roformer", config)
        model.load_state_dict(torch.load(checkpoint, map_location="cpu"), strict=True)
        model.eval()
        if not args.skip_forward:
            with torch.no_grad():
                output = model(torch.zeros(1, 2, args.samples))
            if not torch.isfinite(output).all():
                raise RuntimeError(f"non-finite forward output for {slug}")
        print(f"OK {slug}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
