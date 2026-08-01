"""Acceptance inventory for the pcunwa Mel-Band Roformer repositories."""

from mel_band_roformer.checkpoints import load_checkpoints
from mel_band_roformer.model_registry import MODEL_REGISTRY


PCUNWA_CHECKPOINTS = {
    "melband_roformer_big_beta1.ckpt",
    "melband_roformer_big_beta2.ckpt",
    "melband_roformer_big_beta3.ckpt",
    "melband_roformer_big_beta4.ckpt",
    "big_beta5e.ckpt",
    "big_beta6.ckpt",
    "big_beta6x.ckpt",
    "big_beta7.ckpt",
    "melband_roformer_small_v1.ckpt",
    "melband_roformer_inst_v1.ckpt",
    "inst_v1e.ckpt",
    "inst_v1e_plus.ckpt",
    "inst_v1_plus_test.ckpt",
    "melband_roformer_inst_v2.ckpt",
    "kimmel_unwa_ft.ckpt",
    "kimmel_unwa_ft2.ckpt",
    "kimmel_unwa_ft2_bleedless.ckpt",
    "kimmel_unwa_ft3_prev.ckpt",
    "melband_roformer_instvoc_duality_v1.ckpt",
    "melband_roformer_instvox_duality_v2.ckpt",
}


def _checkpoint_artifacts():
    data = load_checkpoints()["models"]
    return {
        artifact["name"]: (slug, model, artifact)
        for slug, model in data.items()
        for artifact in model["artifacts"]
        if artifact["kind"] == "checkpoint"
    }


def test_every_pcunwa_melband_checkpoint_is_registered_once():
    artifacts = _checkpoint_artifacts()
    assert PCUNWA_CHECKPOINTS <= artifacts.keys()
    assert len([name for name in artifacts if name in PCUNWA_CHECKPOINTS]) == 20
    for checkpoint in PCUNWA_CHECKPOINTS:
        slug, model, artifact = artifacts[checkpoint]
        assert MODEL_REGISTRY.get(slug).checkpoint == checkpoint
        assert "huggingface.co/pcunwa/" in artifact["url"]
        assert len(artifact["sha256"]) == 64
        assert artifact["size"] > 0
