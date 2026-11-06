"""Sequence-identity and manifest auditing for the distributed ProGLHM tasks.

Copyright (c) 2026 ProGLHM contributors. All rights reserved.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

from .data import (
    DataError, EXPECTED_LENGTHS, SPECIES, SUBTYPES, normalize_subtype,
    read_fasta, task_name,
)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reverse_complement(sequence):
    return sequence.translate(str.maketrans("ACGTN", "TGCAN"))[::-1]


def canonical(sequence):
    return min(sequence, reverse_complement(sequence))


def _metadata(header):
    return dict(item.split("=", 1) for item in header.split()[1:] if "=" in item)


def audit(data_root):
    data_root = Path(data_root).expanduser().resolve()
    files = []
    tasks = []
    species_audits = []
    for species in SPECIES:
        species_sets = {"train": set(), "test": set()}
        species_chromosomes = {"train": set(), "test": set()}
        background_keys = []
        positive_keys = []
        for subtype in SUBTYPES:
            name = task_name(species, subtype)
            row = {"task": name, "species": species, "subtype": subtype,
                   "length_bp": EXPECTED_LENGTHS[species]}
            exact = {"train": set(), "test": set()}
            canonical_sets = {"train": set(), "test": set()}
            chromosomes = {"train": set(), "test": set()}
            all_ids = set()
            for split in ("train", "test"):
                for prefix, label in (("Pos", 1), ("Neg", 0)):
                    relative = f"{name}/{prefix}_{split}.fasta"
                    path = data_root / relative
                    records = read_fasta(path, EXPECTED_LENGTHS[species])
                    sequences = [r["sequence"] for r in records]
                    keys = [canonical(s) for s in sequences]
                    counts = Counter("".join(r["raw_sequence"].upper() for r in records))
                    header_mismatches = []
                    for r in records:
                        metadata = _metadata(r["description"])
                        for field, expected in (("species", species), ("task", subtype), ("split", split)):
                            observed = metadata.get(field)
                            if field == "task":
                                try:
                                    observed = normalize_subtype(observed)
                                except DataError:
                                    pass
                            if observed != expected:
                                header_mismatches.append({"id": r["id"], "field": field,
                                    "expected": expected, "observed": metadata.get(field)})
                    ids = {r["id"] for r in records}
                    file_chromosomes = sorted({_metadata(r["description"]).get("chrom", "") for r in records})
                    chromosomes[split].update(file_chromosomes)
                    exact[split].update(sequences)
                    canonical_sets[split].update(keys)
                    species_sets[split].update(keys)
                    species_chromosomes[split].update(file_chromosomes)
                    if label == 0:
                        background_keys.extend(keys)
                    else:
                        positive_keys.extend(keys)
                    row[f"{'positive' if label else 'background'}_{split}"] = len(records)
                    files.append({
                        "path": relative, "sha256": sha256_file(path),
                        "bytes": path.stat().st_size, "records": len(records),
                        "label": label, "split": split, "task": name,
                        "length_min": min(map(len, sequences)), "length_max": max(map(len, sequences)),
                        "base_counts_raw_uppercase": dict(sorted(counts.items())),
                        "max_n_fraction": max(s.count("N") / len(s) for s in sequences),
                        "records_exceeding_10_percent_N": sum(s.count("N") > len(s)*.1 for s in sequences),
                        "non_ACGTN_raw_bases": {k:v for k,v in counts.items() if k not in "ACGTN"},
                        "exact_duplicate_records_within_file": len(sequences)-len(set(sequences)),
                        "exact_or_RC_duplicate_records_within_file": len(keys)-len(set(keys)),
                        "duplicate_IDs_across_task_files": sorted(ids & all_ids),
                        "chromosomes_from_headers": file_chromosomes,
                        "strands_from_headers": sorted({_metadata(r["description"]).get("strand", "") for r in records}),
                        "header_metadata_mismatches": header_mismatches,
                    })
                    all_ids.update(ids)
            row["total_positive"] = row["positive_train"]+row["positive_test"]
            row["total_background"] = row["background_train"]+row["background_test"]
            row["train_classes_balanced"] = row["positive_train"] == row["background_train"]
            row["test_classes_balanced"] = row["positive_test"] == row["background_test"]
            row["train_test_exact_shared_sequence_keys"] = len(exact["train"] & exact["test"])
            row["train_test_exact_or_RC_shared_sequence_keys"] = len(canonical_sets["train"] & canonical_sets["test"])
            row["train_chromosomes_from_headers"] = sorted(chromosomes["train"])
            row["test_chromosomes_from_headers"] = sorted(chromosomes["test"])
            row["shared_chromosomes_from_headers"] = sorted(chromosomes["train"] & chromosomes["test"])
            tasks.append(row)
        species_audits.append({
            "species": species,
            "train_test_exact_or_RC_shared_sequence_keys_across_both_subtypes": len(species_sets["train"] & species_sets["test"]),
            "background_exact_or_RC_duplicate_records_across_subtypes_and_splits": len(background_keys)-len(set(background_keys)),
            "positive_exact_or_RC_duplicate_records_across_subtypes_and_splits": len(positive_keys)-len(set(positive_keys)),
            "positive_background_exact_or_RC_shared_sequence_keys": len(set(positive_keys) & set(background_keys)),
            "train_chromosomes_from_headers": sorted(species_chromosomes["train"]),
            "test_chromosomes_from_headers": sorted(species_chromosomes["test"]),
            "shared_chromosomes_from_headers": sorted(species_chromosomes["train"] & species_chromosomes["test"]),
        })
    totals = {"tasks":len(tasks), "fasta_files":len(files),
              "positive_sequences":sum(r["total_positive"] for r in tasks),
              "background_sequences":sum(r["total_background"] for r in tasks),
              "training_pool_sequences":sum(r["positive_train"]+r["background_train"] for r in tasks),
              "test_sequences":sum(r["positive_test"]+r["background_test"] for r in tasks),
              "bytes":sum(r["bytes"] for r in files)}
    totals["sequences"] = totals["positive_sequences"]+totals["background_sequences"]
    totals["TATA_promoters"] = sum(r["total_positive"] for r in tasks if r["subtype"] == "TATA")
    totals["Non_TATA_promoters"] = sum(r["total_positive"] for r in tasks if r["subtype"] == "Non_TATA")
    return {"format_version":1, "dataset":"ProGLHM eight-species promoter-recognition benchmark",
            "label_mapping":{"0":"background", "1":"promoter"},
            "totals":totals, "tasks":tasks, "species_audits":species_audits,
            "files":files,
            "audit_scope":"Sequence/RC identity and chromosome labels in distributed FASTA headers. No original genomic coordinates, assembly mapping or chromosome assignment provenance is independently verified."}


def validation_issues(result):
    """Collect sequence and split inconsistencies for each task."""
    issues = []
    for row in result["files"]:
        for field in (
            "records_exceeding_10_percent_N", "non_ACGTN_raw_bases",
            "exact_or_RC_duplicate_records_within_file",
            "duplicate_IDs_across_task_files", "header_metadata_mismatches",
        ):
            if row[field]:
                issues.append(f"{row['path']}: {field}")
    for row in result["tasks"]:
        if not row["train_classes_balanced"] or not row["test_classes_balanced"]:
            issues.append(f"{row['task']}: a fixed split is not class-balanced")
        if row["train_test_exact_or_RC_shared_sequence_keys"]:
            issues.append(f"{row['task']}: shared train/test sequence or RC keys")
        if row["shared_chromosomes_from_headers"]:
            issues.append(f"{row['task']}: shared train/test chromosome labels")
    for row in result["species_audits"]:
        for field in (
            "train_test_exact_or_RC_shared_sequence_keys_across_both_subtypes",
            "background_exact_or_RC_duplicate_records_across_subtypes_and_splits",
            "positive_exact_or_RC_duplicate_records_across_subtypes_and_splits",
            "positive_background_exact_or_RC_shared_sequence_keys",
        ):
            if row[field]:
                issues.append(f"{row['species']}: {field}")
    return issues


def audit_dataset(data_root, verify_manifest=True):
    """Audit task sequences and, by default, compare hashes with manifest.json."""
    data_root = Path(data_root).expanduser().resolve()
    result = audit(data_root)
    issues = validation_issues(result)
    if verify_manifest:
        path = data_root / "manifest.json"
        if not path.is_file():
            raise DataError(f"Dataset manifest not found: {path}")
        try:
            expected = json.loads(path.read_text(encoding="utf-8"))
            expected_hashes = {r["path"]: r["sha256"] for r in expected["files"]}
        except (ValueError, KeyError, TypeError) as exc:
            raise DataError(f"Invalid dataset manifest: {path}") from exc
        observed_hashes = {r["path"]: r["sha256"] for r in result["files"]}
        if observed_hashes != expected_hashes:
            changed = sorted(
                key for key in set(observed_hashes) | set(expected_hashes)
                if observed_hashes.get(key) != expected_hashes.get(key)
            )
            issues.append("FASTA paths or SHA-256 hashes differ from manifest.json: " + ", ".join(changed))
    if issues:
        raise DataError("Dataset audit failed: " + "; ".join(issues))
    result["validation"] = {"passed": True, "manifest_hashes_checked": bool(verify_manifest)}
    return result
