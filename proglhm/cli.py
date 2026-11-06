"""ProGLHM command-line interface; metadata commands do not require CUDA.

Copyright (c) 2026 ProGLHM contributors. All rights reserved.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .data import EXPECTED_LENGTHS, SPECIES, normalize_species, normalize_subtype, task_name


def positive_int(value):
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Must be a positive integer") from exc
    if number < 1:
        raise argparse.ArgumentTypeError("Must be a positive integer")
    return number


def positive_float(value):
    import math
    try:
        number = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Must be a positive finite number") from exc
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("Must be a positive finite number")
    return number


def build_parser():
    parser = argparse.ArgumentParser(prog="proglhm", description="ProGLHM: task-specific promoter recognition")
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list-tasks", help="List all 16 independent species/subtype tasks")
    audit = commands.add_parser("audit", help="Audit packaged FASTA sequences and manifest")
    audit.add_argument("--data-root", type=Path, default=Path("datasets"))
    audit.add_argument("--output", type=Path, default=None, help="Optional JSON audit output")

    def task_arguments(command):
        command.add_argument("--species", type=normalize_species, required=True, help="e.g. H.sapiens or Homo sapiens")
        command.add_argument("--subtype", type=normalize_subtype, required=True, help="TATA or non-TATA")

    def common(command, encoder_default=None):
        command.add_argument("--device", default="auto", help="auto, cuda, cuda:0, or cpu (encoding only)")
        command.add_argument("--encoder", default=encoder_default, help="HF model ID or local encoder directory")
        command.add_argument("--local-files-only", action="store_true", help="Use previously downloaded encoder files only")

    def checkpoint_arguments(command):
        choice = command.add_mutually_exclusive_group(required=True)
        choice.add_argument("--checkpoint", type=Path, help="Task-specific best.pt")
        choice.add_argument("--model-dir", type=Path, help="Find SPECIES_SUBTYPE/best.pt below this directory")

    encoder_id = "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species"
    encode = commands.add_parser("encode", help="Build/reuse a float32 NT-v2 embedding and token-mask cache")
    task_arguments(encode)
    common(encode, encoder_id)
    encode.add_argument("--encoder-revision", default=None, help="Pin a Hub commit or revision; resolved commit is recorded")
    encode.add_argument("--split", choices=("train", "test"), required=True)
    encode.add_argument("--data-root", type=Path, default=Path("datasets"))
    encode.add_argument("--cache-dir", type=Path, default=Path("cache"))
    encode.add_argument("--batch-size", type=positive_int, default=4)

    for name in ("train", "train-all"):
        command = commands.add_parser(name, help="Train one task" if name == "train" else "Train 16 separate models sequentially")
        if name == "train":
            task_arguments(command)
        common(command, encoder_id)
        command.add_argument("--encoder-revision", default=None)
        command.add_argument("--data-root", type=Path, default=Path("datasets"))
        command.add_argument("--cache-dir", type=Path, default=Path("cache"))
        command.add_argument("--output-dir", type=Path, default=Path("runs"))
        command.add_argument("--epochs", type=positive_int, default=200)
        command.add_argument("--batch-size", type=positive_int, default=32)
        command.add_argument("--encode-batch-size", type=positive_int, default=4)
        command.add_argument("--learning-rate", type=positive_float, default=1e-4)
        command.add_argument("--weight-decay", type=positive_float, default=1e-4)
        command.add_argument("--val-fraction", type=positive_float, default=0.1)
        command.add_argument("--patience", type=positive_int, default=20)
        command.add_argument("--seed", type=int, default=200)
        command.add_argument("--num-workers", type=int, default=2)
        command.add_argument("--no-amp", action="store_true", help="Disable CUDA mixed-precision training")
        command.add_argument("--resume", action="store_true", help="Restore task/last.pt with matching settings")

    for name in ("test", "test-all"):
        command = commands.add_parser(name, help="Evaluate fixed held-out test records")
        if name == "test":
            task_arguments(command)
            checkpoint_arguments(command)
        else:
            command.add_argument("--checkpoint-root", type=Path, required=True)
        common(command)
        command.add_argument("--data-root", type=Path, default=Path("datasets"))
        command.add_argument("--cache-dir", type=Path, default=Path("cache"))
        command.add_argument("--output-dir", type=Path, default=Path("results"))
        command.add_argument("--batch-size", type=positive_int, default=64)
        command.add_argument("--encode-batch-size", type=positive_int, default=4)

    predict = commands.add_parser("predict", help="Predict unlabeled, length-matched FASTA windows")
    task_arguments(predict)
    checkpoint_arguments(predict)
    common(predict)
    predict.add_argument("--input", type=Path, required=True)
    predict.add_argument("--output", type=Path, required=True)
    predict.add_argument("--batch-size", type=positive_int, default=4, help="Encoder/prediction batch size")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "val_fraction", 0.1) >= 1:
        parser.error("--val-fraction must lie strictly between zero and one")
    if getattr(args, "num_workers", 0) < 0:
        parser.error("--num-workers must be nonnegative")
    try:
        if args.command == "list-tasks":
            for species, full in SPECIES.items():
                for subtype in ("TATA", "Non_TATA"):
                    print(f"{task_name(species, subtype)}\t{full}\t{EXPECTED_LENGTHS[species]} bp")
            return 0
        if args.command == "audit":
            from .data import audit_dataset
            report = audit_dataset(args.data_root)
            if args.output:
                from .utils import write_json
                write_json(args.output, report)
            print(json.dumps(report, indent=2))
            return 0
        from . import workflows
        if args.command == "encode":
            workflows.encode_task(args)
        elif args.command == "train":
            workflows.train_task(args)
        elif args.command == "test":
            workflows.test_task(args)
        elif args.command == "predict":
            workflows.predict_fasta(args)
        else:
            results = []
            for species in SPECIES:
                for subtype in ("TATA", "Non_TATA"):
                    args.species, args.subtype = species, subtype
                    if args.command == "train-all":
                        task_args = argparse.Namespace(**vars(args))
                        if args.resume and not (Path(args.output_dir) / task_name(species, subtype) / "last.pt").exists():
                            task_args.resume = False
                        results.append(workflows.train_task(task_args))
                    else:
                        args.checkpoint, args.model_dir = None, args.checkpoint_root
                        results.append(workflows.test_task(args))
            from .utils import write_json
            write_json(Path(args.output_dir) / (args.command + "_summary.json"), results)
        return 0
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError, OSError, ArithmeticError) as exc:
        print(f"ProGLHM error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
