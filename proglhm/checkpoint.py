"""Portable, task-bound ProGLHM checkpoints.

Copyright (c) 2026 ProGLHM contributors. All rights reserved.
"""
from __future__ import annotations

import os
from pathlib import Path

from .data import normalize_species, normalize_subtype, task_name

FORMAT_VERSION = 1


def atomic_save(payload, path):
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".partial.{os.getpid()}")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def read_checkpoint(path, species=None, subtype=None):
    import torch
    # Best-model checkpoints contain tensors and builtin metadata only.
    try:
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError(
            f"Cannot safely read a ProGLHM best-model checkpoint: {path}. "
            "Use best.pt for testing/prediction; last.pt is only for explicit training --resume."
        ) from exc
    if not isinstance(payload, dict) or payload.get("format_version") != FORMAT_VERSION or payload.get("model_name") != "ProGLHM":
        raise ValueError(f"Not a ProGLHM release checkpoint: {path}")
    config = payload.get("config", {})
    required = {"species", "subtype", "architecture_version", "encoder", "labels"}
    if not isinstance(config, dict) or not required <= config.keys() or not isinstance(payload.get("model_state_dict"), dict):
        raise ValueError(f"Incomplete ProGLHM checkpoint: {path}")
    if not isinstance(config["encoder"], dict) or not isinstance(config["architecture_version"], str):
        raise ValueError(f"Invalid ProGLHM checkpoint configuration: {path}")
    if normalize_species(config["species"]) != config["species"] or normalize_subtype(config["subtype"]) != config["subtype"]:
        raise ValueError("Checkpoint species/subtype must use the canonical task names")
    epoch = payload.get("epoch")
    if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 1:
        raise ValueError("A selected best-model checkpoint needs a positive epoch; last.pt is only for --resume")
    if species is not None and normalize_species(species) != config["species"]:
        raise ValueError(f"Checkpoint species {config['species']} does not match requested {species}")
    if subtype is not None and normalize_subtype(subtype) != config["subtype"]:
        raise ValueError(f"Checkpoint subtype {config['subtype']} does not match requested {subtype}")
    if config["labels"] != {"background": 0, "promoter": 1}:
        raise ValueError("Unsupported checkpoint label mapping")
    return payload


def load_model(path, species=None, subtype=None, device="cuda"):
    """Return ``(model_in_eval_mode, metadata)``; NT-v2 remains separate."""
    from .model import ProGLHM
    payload = read_checkpoint(path, species, subtype)
    if payload["config"]["architecture_version"] != ProGLHM.architecture_version:
        raise ValueError("Checkpoint architecture version does not match this release")
    model = ProGLHM()
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, payload


def select_checkpoint(checkpoint, model_dir, species, subtype):
    path = Path(checkpoint) if checkpoint else Path(model_dir) / task_name(species, subtype) / "best.pt"
    if not path.is_file():
        raise FileNotFoundError(f"ProGLHM checkpoint not found: {path}. Train this species/subtype first or supply its checkpoint.")
    return path
