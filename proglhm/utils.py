"""File I/O, classification metrics and environment metadata.

Copyright (c) 2026 ProGLHM contributors. All rights reserved.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".partial.{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_csv(path, rows, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".partial.{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def metrics_from_counts(tp, fn, fp, tn):
    def divide(a, b):
        return float(a / b) if b else 0.0
    total = tp + fn + fp + tn
    accuracy = divide(tp + tn, total)
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {"TP": tp, "FN": fn, "FP": fp, "TN": tn,
            "SN": divide(tp, tp + fn), "SP": divide(tn, tn + fp),
            "ACC": accuracy, "MCC": divide(tp * tn - fp * fn, denominator),
            "F1": divide(2 * tp, 2 * tp + fp + fn), "PRE": divide(tp, tp + fp),
            "Err": 1.0 - accuracy, "n": total}


def environment_info():
    versions = {}
    for name in ("numpy", "biopython", "transformers", "torch", "mamba-ssm", "causal-conv1d"):
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    return {"python": sys.version, "platform": platform.platform(), "packages": versions}
