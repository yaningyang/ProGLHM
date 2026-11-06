"""Independent task training, held-out testing and FASTA prediction.

Copyright (c) 2026 ProGLHM contributors. All rights reserved.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from . import __version__
from .checkpoint import load_model, read_checkpoint, select_checkpoint
from .data import EXPECTED_LENGTHS, collate_batch, normalize_species, normalize_subtype, read_fasta, read_task, sequence_digest, task_name
from .encoder import FrozenNTv2, ensure_cache, resolve_encoder
from .utils import environment_info, file_sha256, metrics_from_counts, write_csv, write_json


def runtime_device(requested="auto", require_mamba=False):
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Install the documented NVIDIA/CUDA environment.")
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Only cpu or cuda devices are supported")
    if require_mamba:
        if device.type != "cuda":
            raise RuntimeError("Mamba requires an NVIDIA CUDA device. CPU is supported for NT-v2 extraction only.")
        try:
            from mamba_ssm import Mamba  # noqa: F401
        except (ImportError, OSError) as exc:
            raise RuntimeError("Install mamba-ssm==1.2.0.post1 and causal-conv1d==1.2.0.post2 in the documented Linux CUDA environment.") from exc
    return device


def encode_task(args):
    records = read_task(args.data_root, args.species, args.subtype, args.split)
    device = runtime_device(args.device)
    spec = resolve_encoder(args.encoder, args.encoder_revision, args.local_files_only)
    dataset = ensure_cache(records, spec, args.cache_dir, task_name(args.species, args.subtype), args.split,
                           str(device), args.batch_size, args.local_files_only)
    metadata = dataset.metadata
    dataset.close()
    return metadata


def train_task(args):
    from .model import ProGLHM
    from .training import fit, seed_all
    species, subtype = normalize_species(args.species), normalize_subtype(args.subtype)
    task = task_name(species, subtype)
    device = runtime_device(args.device, require_mamba=True)
    records = read_task(args.data_root, species, subtype, "train")
    run_dir = Path(args.output_dir) / task
    if not args.resume and any((run_dir / name).exists() for name in ("best.pt", "last.pt")):
        raise FileExistsError(f"Existing run in {run_dir}; use --resume with the same settings or choose another output directory.")
    spec = resolve_encoder(args.encoder, args.encoder_revision, args.local_files_only)
    dataset = ensure_cache(records, spec, args.cache_dir, task, "train", str(device), args.encode_batch_size, args.local_files_only)
    seed_all(args.seed)
    network = ProGLHM()
    config = {"species": species, "subtype": subtype, "task": task,
              "architecture_version": ProGLHM.architecture_version,
              "software_version": __version__, "encoder": spec,
              "labels": {"background": 0, "promoter": 1},
              "dataset": {"sha256": sequence_digest(records), "records": len(records)},
              "training": {"seed": args.seed, "epochs": args.epochs, "batch_size": args.batch_size,
                           "learning_rate": args.learning_rate, "weight_decay": args.weight_decay,
                           "val_fraction": args.val_fraction, "patience": args.patience,
                           "num_workers": args.num_workers, "amp": not args.no_amp},
              "environment": environment_info()}
    try:
        best_path, payload = fit(dataset, network, config, args, run_dir, device)
    finally:
        dataset.close()
    print(f"Best checkpoint: {best_path}; epoch={payload['epoch']} validation ACC={payload['val_acc']:.6f}", flush=True)
    print("Training complete. Run the test command separately for held-out evaluation.", flush=True)
    return {"task": task, "checkpoint": str(best_path), "best_epoch": payload["epoch"], "val_acc": payload["val_acc"]}


def checkpoint_encoder(config, args):
    expected = config.get("encoder")
    required = {"source", "fingerprint", "identity", "resolved_revision"}
    if not isinstance(expected, dict) or not required <= expected.keys():
        raise ValueError("Checkpoint encoder metadata is incomplete")
    identity = expected["identity"]
    if not isinstance(identity, dict) or identity.get("kind") not in {"hub", "local"}:
        raise ValueError("Checkpoint encoder identity must describe a pinned Hub model or local files")
    if not isinstance(expected["source"], str) or not expected["source"] or not isinstance(expected["fingerprint"], str):
        raise ValueError("Checkpoint encoder source or fingerprint is invalid")
    if identity["kind"] == "hub" and (not identity.get("repo_id") or not identity.get("commit") or not expected["resolved_revision"]):
        raise ValueError("Checkpoint Hub encoder has no pinned repository/commit")
    override = getattr(args, "encoder", None)
    if override:
        spec = resolve_encoder(override, expected.get("resolved_revision"), args.local_files_only)
        if spec["fingerprint"] != expected["fingerprint"]:
            raise ValueError("The supplied encoder differs from the checkpoint encoder. Use the same pinned Hub model or an identical local encoder copy.")
        return spec
    if identity["kind"] == "local":
        spec = resolve_encoder(expected["source"], local_files_only=args.local_files_only)
        if spec["fingerprint"] != expected["fingerprint"]:
            raise ValueError("Local encoder contents changed since training")
        return spec
    return expected


def _prediction_rows(network, dataset, records, device, batch_size, task):
    rows = []
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)
    offset = 0
    network.eval()
    with torch.inference_mode():
        for (features, masks), labels in loader:
            logits = network((features.to(device), masks.to(device)))
            if not torch.isfinite(logits).all():
                raise RuntimeError("Model produced non-finite logits")
            probabilities = logits.softmax(dim=1)[:, 1].cpu().tolist()
            predictions = logits.argmax(dim=1).cpu().tolist()
            values = logits.cpu().tolist()
            for j, (probability, prediction, value, label) in enumerate(zip(probabilities, predictions, values, labels.tolist())):
                rows.append({"id": records[offset + j]["id"], "task": task, "true_label": label,
                             "promoter_probability": probability, "predicted_label": prediction,
                             "predicted_class": "promoter" if prediction else "background",
                             "background_logit": value[0], "promoter_logit": value[1]})
            offset += len(predictions)
    return rows


def test_task(args):
    species, subtype = normalize_species(args.species), normalize_subtype(args.subtype)
    task = task_name(species, subtype)
    path = select_checkpoint(args.checkpoint, args.model_dir, species, subtype)
    payload = read_checkpoint(path, species, subtype)  # Reject task mismatch before GPU/encoding.
    device = runtime_device(args.device, require_mamba=True)
    records = read_task(args.data_root, species, subtype, "test")
    spec = checkpoint_encoder(payload["config"], args)
    dataset = ensure_cache(records, spec, args.cache_dir, task, "test", str(device), args.encode_batch_size, args.local_files_only)
    try:
        network, payload = load_model(path, species, subtype, str(device))
        rows = _prediction_rows(network, dataset, records, device, args.batch_size, task)
    finally:
        dataset.close()
    tp = sum(row["true_label"] == row["predicted_label"] == 1 for row in rows)
    fn = sum(row["true_label"] == 1 and row["predicted_label"] == 0 for row in rows)
    fp = sum(row["true_label"] == 0 and row["predicted_label"] == 1 for row in rows)
    tn = sum(row["true_label"] == row["predicted_label"] == 0 for row in rows)
    result = {"task": task, **metrics_from_counts(tp, fn, fp, tn),
              "checkpoint_sha256": file_sha256(path), "test_sha256": sequence_digest(records),
              "positive_class": 1, "decision_rule": "argmax(background_logit, promoter_logit)",
              "checkpoint_epoch": payload["epoch"], "encoder_fingerprint": spec["fingerprint"]}
    out = Path(args.output_dir) / task
    write_csv(out / "predictions.csv", rows, list(rows[0]))
    write_json(out / "metrics.json", result)
    print(json.dumps(result, indent=2), flush=True)
    return result


def predict_fasta(args):
    species, subtype = normalize_species(args.species), normalize_subtype(args.subtype)
    task = task_name(species, subtype)
    path = select_checkpoint(args.checkpoint, args.model_dir, species, subtype)
    payload = read_checkpoint(path, species, subtype)
    output = Path(args.output).resolve()
    if output in {Path(args.input).resolve(), path.resolve()}:
        raise ValueError("Prediction output must differ from the input FASTA and checkpoint")
    records = read_fasta(args.input, EXPECTED_LENGTHS[species])
    device = runtime_device(args.device, require_mamba=True)
    spec = checkpoint_encoder(payload["config"], args)
    # Alternate encoder and classifier GPU residency for each batch.
    encoder = FrozenNTv2(spec, device=str(device), local_files_only=args.local_files_only)
    rows = []
    network = None
    try:
        for start in range(0, len(records), args.batch_size):
            part = records[start:start + args.batch_size]
            features, masks = encoder.encode([record["sequence"] for record in part])
            encoder.model.to("cpu")
            if network is None:
                network, _ = load_model(path, species, subtype, str(device))
            else:
                network.to(device)
            from .encoder import EmbeddingDataset
            dataset = EmbeddingDataset(features, masks, [-1] * len(part), {})
            rows.extend(_prediction_rows(network, dataset, part, device, args.batch_size, task))
            network.to("cpu")
            encoder.model.to(device)
    finally:
        encoder.close()
    for row in rows:
        del row["true_label"]
    write_csv(args.output, rows, list(rows[0]))
    write_json(str(args.output) + ".metadata.json", {"model": "ProGLHM", "task": task,
               "input_sha256": file_sha256(args.input), "checkpoint_sha256": file_sha256(path),
               "encoder": spec, "records": len(rows), "positive_class": 1,
               "decision_rule": "argmax(background_logit, promoter_logit)"})
    print(f"Wrote {len(rows)} predictions to {args.output}", flush=True)
    return {"task": task, "predictions": len(rows), "output": str(args.output)}
