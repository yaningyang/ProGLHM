# Copyright (c) 2026 ProGLHM contributors. All rights reserved.
"""ProGLHM training, validation and epoch-boundary checkpoints."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Subset

from .checkpoint import atomic_save, read_checkpoint
from .data import collate_batch, train_val_indices
from .utils import metrics_from_counts, write_csv, write_json


HISTORY_COLUMNS = (
    "epoch", "timestamp", "train_loss", "train_ACC", "train_F1", "train_MCC",
    "val_loss", "val_ACC", "val_F1", "val_MCC", "best_val_ACC", "best_epoch",
    "alpha", "lambda", "direction_weight", "learning_rate", "epoch_seconds",
)


def seed_all(seed):
    """Seed Python, NumPy, PyTorch and CUDA random generators."""
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def optimizer_parameter_groups(network, learning_rate, weight_decay):
    """Group parameters by Mamba's no-weight-decay attribute."""
    decay, no_decay = [], []
    for parameter in network.parameters():
        if parameter.requires_grad:
            (no_decay if getattr(parameter, "_no_weight_decay", False) else decay).append(parameter)
    if not decay and not no_decay:
        raise ValueError("The network has no trainable parameters")
    groups = []
    if decay:
        groups.append({"params": decay, "lr": float(learning_rate), "weight_decay": float(weight_decay)})
    if no_decay:
        groups.append({"params": no_decay, "lr": float(learning_rate), "weight_decay": 0.0})
    return groups


def _rng_state(device):
    return {
        "python": random.getstate(), "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
    }


def _restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state.get("cuda") is not None:
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def _dataset_identity(dataset):
    labels = [int(value) for value in dataset.labels]
    if len(labels) != len(dataset) or not labels or not set(labels) <= {0, 1}:
        raise ValueError("Training requires one binary label per nonempty dataset record")
    identity = {"n": len(dataset), "labels": labels, "metadata": getattr(dataset, "metadata", None)}
    return hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _move(batch, device):
    (features, mask), labels = batch
    return (
        features.to(device, non_blocking=True),
        mask.to(device, non_blocking=True),
    ), labels.to(device, non_blocking=True)


def _counts(labels, predictions):
    return torch.stack((
        ((labels == 1) & (predictions == 1)).sum(),
        ((labels == 1) & (predictions == 0)).sum(),
        ((labels == 0) & (predictions == 1)).sum(),
        ((labels == 0) & (predictions == 0)).sum(),
    ))


def _metrics(counts, loss_sum, sample_count):
    if not sample_count:
        raise ValueError("The data loader produced no samples")
    result = metrics_from_counts(*[int(value) for value in counts.cpu().tolist()])
    result["loss"] = float((loss_sum / sample_count).item())
    return result


def _check_loss(loss, split):
    if not bool(torch.isfinite(loss).all().item()):
        raise FloatingPointError(f"Non-finite {split} loss; no checkpoint was committed for this epoch")


def _evaluate(loader, network, criterion, device):
    """Compute validation loss and metrics in full precision."""
    network.eval()
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    counts = torch.zeros(4, device=device, dtype=torch.long)
    n = 0
    diagnostics = {"alpha": [], "lambda": [], "direction_weight": []}
    with torch.inference_mode():
        for batch in loader:
            inputs, labels = _move(batch, device)
            logits = network(inputs)
            loss = criterion(logits, labels)
            _check_loss(loss, "validation")
            loss_sum += loss.double() * labels.shape[0]
            counts += _counts(labels, logits.argmax(dim=1))
            n += labels.shape[0]
            if hasattr(network, "scalar_diagnostics"):
                for key, value in network.scalar_diagnostics().items():
                    if key in diagnostics and value is not None:
                        diagnostics[key].append(float(value))
    diagnostic_means = {
        key: sum(values) / len(values) if values else None
        for key, values in diagnostics.items()
    }
    return _metrics(counts, loss_sum, n), diagnostic_means


def _history_rows(path):
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != list(HISTORY_COLUMNS):
            raise ValueError(f"Training history columns do not match this release: {path}")
        rows = list(reader)
    epochs = [int(row["epoch"]) for row in rows]
    if epochs != sorted(set(epochs)):
        raise ValueError(f"Training history has repeated or out-of-order epochs: {path}")
    return rows


def fit(dataset, network, config, args, run_dir, device):
    """Fit the training pool and return the validation-selected checkpoint.

    Seed before constructing the network. Settings come from config['training'];
    args supplies the resume flag. Interrupted epochs restart at their boundary.
    """
    config = json.loads(json.dumps(config, allow_nan=False))
    settings = config["training"]
    seed = int(settings["seed"])
    epochs = int(settings["epochs"])
    batch_size = int(settings["batch_size"])
    learning_rate = float(settings["learning_rate"])
    weight_decay = float(settings["weight_decay"])
    workers = int(settings["num_workers"])
    patience = int(settings["patience"])
    if epochs < 1 or batch_size < 1 or workers < 0 or patience < 1:
        raise ValueError("Epochs, batch size and patience must be positive; workers cannot be negative")
    if not math.isfinite(learning_rate) or learning_rate <= 0 or not math.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError("Learning rate must be finite and positive; weight decay must be finite and nonnegative")
    train_indices, val_indices = train_val_indices(len(dataset), seed, settings["val_fraction"])
    if len(train_indices) < batch_size:
        raise ValueError(
            f"Only {len(train_indices)} training records remain, fewer than batch_size={batch_size}. "
            "drop_last=True would produce zero batches; use more records or a smaller batch size."
        )
    identity = _dataset_identity(dataset)
    device = torch.device(device)
    amp_enabled = bool(settings["amp"]) and device.type == "cuda"
    runtime = {"device": str(device), "amp_enabled": amp_enabled}
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    best_path, last_path = run_dir / "best.pt", run_dir / "last.pt"
    history_path = run_dir / "history.csv"
    resume = bool(args if isinstance(args, bool) else getattr(args, "resume", False))
    if resume and not last_path.is_file():
        raise FileNotFoundError(f"--resume requires an existing trusted local checkpoint: {last_path}")
    if not resume and any(path.exists() for path in (best_path, last_path, history_path)):
        raise FileExistsError(f"Training outputs already exist in {run_dir}; use --resume or choose a new output directory")

    loader_generator = torch.Generator().manual_seed(seed)
    loader_kwargs = {"num_workers": workers, "pin_memory": device.type == "cuda", "collate_fn": collate_batch}
    if workers:
        loader_kwargs["persistent_workers"] = True
    train_loader = DataLoader(
        Subset(dataset, train_indices), batch_size=batch_size,
        shuffle=True, drop_last=True, generator=loader_generator, **loader_kwargs,
    )
    val_loader = DataLoader(
        Subset(dataset, val_indices), batch_size=batch_size,
        shuffle=False, drop_last=False, **loader_kwargs,
    )
    network.to(device)
    optimizer = torch.optim.AdamW(
        optimizer_parameter_groups(network, learning_rate, weight_decay), lr=learning_rate
    )
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=10)
    scaler = torch.cuda.amp.GradScaler() if amp_enabled else None
    criterion = torch.nn.CrossEntropyLoss()
    start_epoch = 0
    best_acc, best_mcc, best_epoch, stall = -1.0, 0.0, 0, 0
    best_payload = None
    history = []

    if resume:
        # RNG restoration requires loading a trusted local resume checkpoint.
        last = torch.load(last_path, map_location="cpu", weights_only=False)
        if last.get("format_version") != 1 or last.get("model_name") != "ProGLHM":
            raise ValueError("The resume file is not a ProGLHM release checkpoint")
        if last.get("config") != config:
            raise ValueError("Resume configuration differs from the saved run; start a new output directory")
        if last.get("runtime") != runtime:
            raise ValueError("Resume device or AMP setting differs from the saved run")
        if last.get("dataset_identity") != identity:
            raise ValueError("Resume dataset order, labels or embedding metadata differ from the saved run")
        if last.get("train_indices") != train_indices or last.get("val_indices") != val_indices:
            raise ValueError("Resume training/validation split differs from the saved run")
        start_epoch = int(last["next_epoch"])
        if not 0 <= start_epoch <= epochs:
            raise ValueError("Resume epoch is outside this run's configured range")
        best_acc, best_mcc = float(last["best_val_acc"]), float(last["best_val_mcc"])
        best_epoch, stall = int(last["best_epoch"]), int(last["epochs_without_improvement"])
        best_payload = last.get("best_checkpoint")
        if best_epoch:
            if not isinstance(best_payload, dict) or best_payload.get("config") != config or int(best_payload.get("epoch", -1)) != best_epoch:
                raise ValueError("Resume checkpoint lacks its matching best-model snapshot")
            # Restore the last committed best if a crash occurred between
            # writing best.pt and committing the next epoch to last.pt.
            atomic_save(best_payload, best_path)
        history = [row for row in _history_rows(history_path) if int(row["epoch"]) <= start_epoch]
        if [int(row["epoch"]) for row in history] != list(range(1, start_epoch + 1)):
            raise ValueError("Training history is incomplete for the saved resume epoch")
        write_csv(history_path, history, HISTORY_COLUMNS)
        network.load_state_dict(last["model_state_dict"], strict=True)
        optimizer.load_state_dict(last["optimizer_state_dict"])
        scheduler.load_state_dict(last["scheduler_state_dict"])
        if scaler is not None:
            if last.get("scaler_state_dict") is None:
                raise ValueError("The AMP resume checkpoint lacks gradient-scaler state")
            scaler.load_state_dict(last["scaler_state_dict"])
        loader_generator.set_state(last["loader_generator_state"].cpu())
        if workers and start_epoch:
            # Keep worker restart seeding separate from the restored sampler.
            train_loader.generator = torch.Generator().manual_seed(seed)
            val_loader.generator = torch.Generator().manual_seed(seed)
        _restore_rng(last["rng_state"])
        if last.get("completed"):
            if not best_epoch:
                raise ValueError("Completed training checkpoint has no selected best model")
            payload = read_checkpoint(best_path, config["species"], config["subtype"])
            network.load_state_dict(payload["model_state_dict"], strict=True)
            network.eval()
            return best_path, payload
        print(f"Resuming epoch {start_epoch + 1}/{epochs}; best validation ACC={best_acc:.6f}", flush=True)
    else:
        write_json(run_dir / "split.json", {
            "seed": seed, "val_fraction": settings["val_fraction"],
            "dataset_identity": identity,
            "train_indices": train_indices, "val_indices": val_indices,
        })

    def save_last(next_epoch, completed):
        atomic_save({
            "format_version": 1, "model_name": "ProGLHM", "config": config,
            "runtime": runtime, "dataset_identity": identity,
            "train_indices": train_indices, "val_indices": val_indices,
            "completed": bool(completed), "next_epoch": int(next_epoch),
            "model_state_dict": network.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
            "best_val_acc": best_acc, "best_val_mcc": best_mcc,
            "best_epoch": best_epoch, "epochs_without_improvement": stall,
            "best_checkpoint": best_payload,
            "loader_generator_state": loader_generator.get_state(),
            "rng_state": _rng_state(device),
        }, last_path)

    if not resume:
        save_last(0, completed=False)
    for epoch in range(start_epoch, epochs):
        started = time.perf_counter()
        network.train()
        loss_sum = torch.zeros((), device=device, dtype=torch.float64)
        counts = torch.zeros(4, device=device, dtype=torch.long)
        n = 0
        for batch in train_loader:
            inputs, labels = _move(batch, device)
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    logits = network(inputs)
                    loss = criterion(logits, labels)
                _check_loss(loss, "training")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = network(inputs)
                loss = criterion(logits, labels)
                _check_loss(loss, "training")
                loss.backward()
                optimizer.step()
            loss_sum += loss.detach().double() * labels.shape[0]
            counts += _counts(labels, logits.detach().argmax(dim=1))
            n += labels.shape[0]
        train_metrics = _metrics(counts, loss_sum, n)
        val_metrics, diagnostics = _evaluate(val_loader, network, criterion, device)
        scheduler.step(val_metrics["ACC"])
        if val_metrics["ACC"] > best_acc:
            best_acc, best_mcc = val_metrics["ACC"], val_metrics["MCC"]
            best_epoch, stall = epoch + 1, 0
            best_payload = {
                "format_version": 1, "model_name": "ProGLHM", "config": config,
                "model_state_dict": {
                    name: value.detach().cpu().clone()
                    for name, value in network.state_dict().items()
                },
                "epoch": best_epoch, "val_acc": float(best_acc), "val_mcc": float(best_mcc),
            }
            atomic_save(best_payload, best_path)
        else:
            stall += 1
        row = {
            "epoch": epoch + 1, "timestamp": datetime.now(timezone.utc).isoformat(),
            "train_loss": train_metrics["loss"], "train_ACC": train_metrics["ACC"],
            "train_F1": train_metrics["F1"], "train_MCC": train_metrics["MCC"],
            "val_loss": val_metrics["loss"], "val_ACC": val_metrics["ACC"],
            "val_F1": val_metrics["F1"], "val_MCC": val_metrics["MCC"],
            "best_val_ACC": best_acc, "best_epoch": best_epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "epoch_seconds": time.perf_counter() - started, **diagnostics,
        }
        history.append(row)
        write_csv(history_path, history, HISTORY_COLUMNS)
        completed = stall >= patience or epoch + 1 == epochs
        save_last(epoch + 1, completed)
        print(
            f"Epoch {epoch + 1}/{epochs} train_loss={train_metrics['loss']:.6f} "
            f"val_ACC={val_metrics['ACC']:.6f} val_F1={val_metrics['F1']:.6f} "
            f"val_MCC={val_metrics['MCC']:.6f} best_epoch={best_epoch}", flush=True,
        )
        if completed:
            break
    payload = read_checkpoint(best_path, config["species"], config["subtype"])
    network.load_state_dict(payload["model_state_dict"], strict=True)
    network.eval()
    return best_path, payload
