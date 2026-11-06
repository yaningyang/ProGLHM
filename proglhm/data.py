"""Task selection, FASTA input and batching for ProGLHM.

Copyright (c) 2026 ProGLHM contributors. All rights reserved.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
import re


class DataError(ValueError):
    """Invalid sequence input or task specification."""


SPECIES = {
    "A.thaliana": "Arabidopsis thaliana",
    "D.melanogaster": "Drosophila melanogaster",
    "D.rerio": "Danio rerio",
    "G.gallus": "Gallus gallus",
    "H.sapiens": "Homo sapiens",
    "M.musculus": "Mus musculus",
    "R.norvegicus": "Rattus norvegicus",
    "Z.mays": "Zea mays",
}
EXPECTED_LENGTHS = {
    name: 251 if name in {"A.thaliana", "Z.mays"} else 300 for name in SPECIES
}
SUBTYPES = ("TATA", "Non_TATA")


def _name_key(value):
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


_SPECIES_ALIASES = {
    _name_key(value): short
    for short, full in SPECIES.items()
    for value in (short, full)
}
_SPECIES_ALIASES.update({
    "arabidopsis": "A.thaliana", "fruitfly": "D.melanogaster",
    "zebrafish": "D.rerio", "chicken": "G.gallus",
    "human": "H.sapiens", "mouse": "M.musculus",
    "rat": "R.norvegicus", "maize": "Z.mays",
})


def normalize_species(value):
    """Resolve an abbreviated, full Latin or supported common species name."""
    result = _SPECIES_ALIASES.get(_name_key(value))
    if result is None:
        raise DataError(
            f"Unknown species {value!r}. Choose one of: {', '.join(SPECIES)}"
        )
    return result


def normalize_subtype(value):
    """Return the dataset directory spelling, TATA or Non_TATA."""
    key = _name_key(value)
    if key == "tata":
        return "TATA"
    if key == "nontata":
        return "Non_TATA"
    raise DataError(f"Unknown subtype {value!r}; choose TATA or Non-TATA")


def task_name(species, subtype):
    return f"{normalize_species(species)}_{normalize_subtype(subtype)}"


def clean_sequence(sequence):
    """Uppercase the sequence and map non-ATCG characters to N."""
    return "".join(base if base in "ATCG" else "N" for base in str(sequence).upper())


def read_fasta(path, expected_length=None):
    """Read FASTA records and validate their IDs and lengths.

    Sequence lines may contain ASCII letters, including ambiguous bases. All
    non-ATCG letters become N in ``sequence``; ``raw_sequence`` preserves the
    concatenated source letters and their case. Headers must have unique IDs.
    """
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise DataError(f"FASTA file not found: {path}")
    records = []
    ids = set()
    header = None
    parts = []

    def finish_record():
        if header is None:
            return
        record_id = header.split()[0]
        raw = "".join(parts)
        if not raw:
            raise DataError(f"Empty FASTA record: {path}::{record_id}")
        if expected_length is not None and len(raw) != int(expected_length):
            raise DataError(
                f"Length mismatch for {path}::{record_id}: {len(raw)} bp; "
                f"expected {int(expected_length)} bp. No sequence was truncated."
            )
        records.append({
            "id": record_id, "description": header,
            "sequence": clean_sequence(raw), "raw_sequence": raw,
            "source": str(path),
        })

    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            text = line.strip()
            if not text:
                continue
            if text.startswith(">"):
                finish_record()
                header = text[1:].strip()
                if not header:
                    raise DataError(f"Empty FASTA header: {path}:{line_number}")
                record_id = header.split()[0]
                if record_id in ids:
                    raise DataError(f"Duplicate FASTA ID {record_id!r}: {path}")
                ids.add(record_id)
                parts = []
            else:
                if header is None:
                    raise DataError(
                        f"Sequence before first FASTA header: {path}:{line_number}"
                    )
                if not text.isascii() or not text.isalpha():
                    raise DataError(
                        f"Non-letter sequence content: {path}:{line_number}"
                    )
                parts.append(text)
    finish_record()
    if not records:
        raise DataError(f"No FASTA records found: {path}")
    return records


def read_task(data_root, species, subtype, split):
    """Read the fixed training pool or test set, positives followed by negatives."""
    short = normalize_species(species)
    canonical_subtype = normalize_subtype(subtype)
    if split not in {"train", "test"}:
        raise DataError(f"Unknown split {split!r}; choose train or test")
    directory = Path(data_root).expanduser() / task_name(short, canonical_subtype)
    records = []
    for prefix, label in (("Pos", 1), ("Neg", 0)):
        for record in read_fasta(
            directory / f"{prefix}_{split}.fasta", EXPECTED_LENGTHS[short]
        ):
            records.append({**record, "label": label})
    return records


def sequence_digest(records):
    """Hash ordered IDs, cleaned sequences and labels independently of paths."""
    digest = hashlib.sha256(b"proglhm-ordered-sequences-v1\n")
    for record in records:
        payload = [str(record["id"]), str(record["sequence"]), record.get("label")]
        digest.update(json.dumps(payload, ensure_ascii=True, separators=(",", ":")).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def train_val_indices(n, seed=200, fraction=0.1):
    """Select an unstratified validation subset with a local random generator."""
    if not isinstance(n, int) or isinstance(n, bool) or n < 2:
        raise DataError("At least two training-pool records are required")
    if not 0.0 < float(fraction) < 1.0:
        raise DataError("Validation fraction must lie strictly between zero and one")
    count = int(n * float(fraction))
    if count < 1 or count >= n:
        raise DataError(
            "The requested validation fraction leaves an empty training or "
            "validation subset; use more records or a different fraction"
        )
    selected = set(random.Random(int(seed)).sample(range(n), count))
    return (
        [index for index in range(n) if index not in selected],
        [index for index in range(n) if index in selected],
    )


def collate_batch(batch):
    """Stack embedding/mask samples and trim common right padding in units of 8."""
    import torch
    from torch.utils.data import default_collate

    (features, mask), labels = default_collate(batch)
    mask = mask.bool()
    if features.ndim != 3 or mask.ndim != 2 or features.shape[:2] != mask.shape:
        raise DataError("Expected features [B,T,D] and valid-token mask [B,T]")
    if features.shape[1] < 8 or features.shape[1] % 8:
        raise DataError("Cached token length must be at least 8 and divisible by 8")
    lengths = mask.sum(dim=1)
    if torch.any(lengths < 1):
        raise DataError("Every sample must contain at least one valid token")
    if torch.any(mask & ((~mask).cumsum(dim=1) > 0)):
        raise DataError("Masks must contain a contiguous valid prefix and right padding")
    stop = ((int(lengths.max().item()) + 7) // 8) * 8
    return (
        features[:, :stop].contiguous().float(),
        mask[:, :stop].contiguous(),
    ), labels.long()


def audit_dataset(data_root):
    """Audit packaged sequence/RC invariants and manifest hashes without a GPU."""
    from .audit import audit_dataset as run_audit

    return run_audit(data_root, verify_manifest=True)
