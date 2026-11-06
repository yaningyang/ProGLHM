"""Frozen NT-v2 embeddings and disk-backed NumPy caches.

Copyright (c) 2026 ProGLHM contributors. All rights reserved.
"""
from __future__ import annotations

import gc
import json
import os
import re
from pathlib import Path

import numpy as np

from .data import sequence_digest
from .utils import file_sha256, json_digest, write_json

DEFAULT_ENCODER = "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species"
MAX_TOKENS = 128
HIDDEN_SIZE = 1024
PIPELINE_VERSION = "proglhm-ntv2-clean-n-mask128-f32-v1"


def resolve_encoder(source=DEFAULT_ENCODER, revision=None, local_files_only=False):
    """Bind a Hub revision or local file contents, without loading model weights."""
    source = str(source)
    local = Path(source).expanduser()
    if local.is_dir():
        local = local.resolve()
        if not (local / "config.json").is_file():
            raise ValueError(f"Local encoder has no config.json: {local}")
        files = []
        for path in sorted(local.rglob("*")):
            if path.is_file() and path.suffix in {".json", ".py", ".txt", ".model", ".bin", ".safetensors"} and not any(x.startswith(".") for x in path.relative_to(local).parts):
                files.append({"path": path.relative_to(local).as_posix(), "sha256": file_sha256(path)})
        identity = {"kind": "local", "files": files}
        resolved = None
        source = str(local)
    else:
        revision = revision or "main"
        if re.fullmatch(r"[0-9a-f]{40}", revision):
            resolved = revision
        elif local_files_only:
            from huggingface_hub import try_to_load_from_cache
            config = try_to_load_from_cache(source, "config.json", revision=revision)
            if not isinstance(config, str):
                raise ValueError(f"No cached encoder config for {source}@{revision}; supply a local model directory or a cached commit.")
            resolved = Path(config).parent.name
        else:
            from huggingface_hub import HfApi
            resolved = HfApi().model_info(source, revision=revision).sha
        identity = {"kind": "hub", "repo_id": source, "commit": resolved}
    # Store the resolved commit so branch names and pinned commits compare equally.
    return {"source": source, "requested_revision": resolved or revision, "resolved_revision": resolved,
            "identity": identity, "fingerprint": json_digest(identity),
            "pipeline_version": PIPELINE_VERSION, "max_tokens": MAX_TOKENS, "hidden_size": HIDDEN_SIZE}


class FrozenNTv2:
    def __init__(self, spec, device="cuda", local_files_only=False):
        import torch
        from transformers import AutoModelForMaskedLM, AutoTokenizer
        self.torch = torch
        self.spec = spec
        self.device = torch.device(device)
        kwargs = {"trust_remote_code": True, "local_files_only": local_files_only}
        if spec["resolved_revision"]:
            kwargs["revision"] = spec["resolved_revision"]
        self.tokenizer = AutoTokenizer.from_pretrained(spec["source"], **kwargs)
        self.tokenizer.padding_side = "right"
        self.model = AutoModelForMaskedLM.from_pretrained(spec["source"], **kwargs)
        if int(self.model.config.hidden_size) != HIDDEN_SIZE:
            raise ValueError(f"Expected NT-v2 hidden size {HIDDEN_SIZE}, found {self.model.config.hidden_size}")
        self.model.to(self.device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def encode(self, sequences):
        from .data import clean_sequence
        sequences = [clean_sequence(sequence) for sequence in sequences]
        if not sequences:
            raise ValueError("Cannot encode an empty batch")
        tokenized = self.tokenizer(sequences, padding=False, truncation=False)
        longest = max(map(len, tokenized["input_ids"]))
        if longest > MAX_TOKENS:
            raise ValueError(f"NT-v2 input needs {longest} tokens (maximum {MAX_TOKENS}); no truncation was performed.")
        tokens = self.tokenizer(sequences, return_tensors="pt", padding="max_length", max_length=MAX_TOKENS, truncation=False)
        ids = tokens["input_ids"].to(self.device)
        mask = tokens["attention_mask"].to(self.device)
        with self.torch.inference_mode():
            out = self.model(input_ids=ids, attention_mask=mask, output_hidden_states=True)
            hidden = out.hidden_states[-1] * mask.unsqueeze(-1).to(out.hidden_states[-1].dtype)
        embeddings = hidden.to(device="cpu", dtype=self.torch.float32).numpy()
        masks = mask.to(device="cpu", dtype=self.torch.uint8).numpy()
        if embeddings.shape != (len(sequences), MAX_TOKENS, HIDDEN_SIZE):
            raise RuntimeError(f"Unexpected encoder output: {embeddings.shape}")
        return embeddings, masks

    def close(self):
        self.model.to("cpu")
        del self.model, self.tokenizer
        gc.collect()
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()


class EmbeddingDataset:
    def __init__(self, features, masks, labels, metadata):
        self.features, self.masks = features, masks
        self.labels = np.asarray(labels, dtype=np.int64)
        self.metadata = metadata

    def __len__(self):
        return len(self.labels)

    def close(self):
        """Release disk-backed arrays before removing or replacing cache files."""
        for values in (self.features, self.masks):
            mapping = getattr(values, "_mmap", None)
            if mapping is not None:
                mapping.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __getitem__(self, index):
        import torch
        # Copy one record: readonly mmap views cannot safely back mutable tensors.
        feature = torch.from_numpy(np.array(self.features[index], dtype=np.float32, copy=True))
        mask = torch.from_numpy(np.array(self.masks[index], dtype=np.bool_, copy=True))
        return (feature, mask), torch.tensor(int(self.labels[index]), dtype=torch.long)


def ensure_cache(records, spec, cache_dir, task, split, device="cuda", batch_size=4, local_files_only=False, encoder_factory=None):
    """Build or load embeddings identified by their sequences and encoder."""
    if not records or batch_size < 1:
        raise ValueError("Embedding requires records and a positive batch size")
    expected = {"format_version": 1, "pipeline_version": PIPELINE_VERSION,
                "dataset_sha256": sequence_digest(records), "encoder_fingerprint": spec["fingerprint"],
                "n": len(records), "embedding_shape": [len(records), MAX_TOKENS, HIDDEN_SIZE],
                "mask_shape": [len(records), MAX_TOKENS], "embedding_dtype": "float32", "mask_dtype": "uint8"}
    prefix = Path(cache_dir) / task / f"{split}-{expected['dataset_sha256'][:16]}-{spec['fingerprint'][:16]}"
    prefix.mkdir(parents=True, exist_ok=True)
    metadata_path = prefix / "metadata.json"
    embedding_path, mask_path = prefix / "embeddings.npy", prefix / "masks.npy"

    def load():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata["cache"] != expected:
            raise ValueError(f"Cache metadata mismatch: {metadata_path}")
        features = np.load(embedding_path, mmap_mode="r", allow_pickle=False)
        masks = np.load(mask_path, mmap_mode="r", allow_pickle=False)
        if list(features.shape) != expected["embedding_shape"] or features.dtype != np.float32:
            raise ValueError(f"Invalid embedding cache: {embedding_path}")
        if list(masks.shape) != expected["mask_shape"] or masks.dtype != np.uint8:
            raise ValueError(f"Invalid mask cache: {mask_path}")
        if np.any((masks != 0) & (masks != 1)) or np.any(masks.sum(axis=1) == 0) or np.any(np.diff(masks.astype(np.int16), axis=1) > 0):
            raise ValueError(f"Cache has invalid token masks: {mask_path}")
        return EmbeddingDataset(features, masks, [record.get("label", -1) for record in records], metadata)

    if metadata_path.exists():
        print(f"Using embedding cache: {prefix}", flush=True)
        return load()
    lock = prefix / "encoding.lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError(f"Cache is being encoded, or a previous encoding stopped: {lock}. Remove this lock only after confirming no encoder is running.") from exc
    os.close(descriptor)
    features = masks = encoder = None
    try:
        # Another process may have completed the cache between the initial
        # existence check and our successful lock acquisition.
        if metadata_path.exists():
            return load()
        features = np.lib.format.open_memmap(embedding_path, mode="w+", dtype=np.float32, shape=tuple(expected["embedding_shape"]))
        masks = np.lib.format.open_memmap(mask_path, mode="w+", dtype=np.uint8, shape=tuple(expected["mask_shape"]))
        factory = encoder_factory or FrozenNTv2
        encoder = factory(spec, device=device, local_files_only=local_files_only)
        for start in range(0, len(records), batch_size):
            end = min(start + batch_size, len(records))
            values, valid = encoder.encode([record["sequence"] for record in records[start:end]])
            features[start:end], masks[start:end] = values, valid
            print(f"{task} {split}: encoded {end}/{len(records)}", flush=True)
        features.flush()
        masks.flush()
        del features, masks
        features = masks = None
        write_json(metadata_path, {"cache": expected, "encoder": spec})
    finally:
        if encoder is not None:
            encoder.close()
        if features is not None:
            del features
        if masks is not None:
            del masks
        lock.unlink(missing_ok=True)
    return load()
