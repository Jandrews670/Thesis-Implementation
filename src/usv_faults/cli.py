from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from usv_faults.clustering.fault_dictionary import build_fault_dictionary
from usv_faults.data_sources.synthetic_usv import SyntheticUSVSource
from usv_faults.preprocessing.datasets import make_dataset
from usv_faults.storage.preview import write_preview_csv
from usv_faults.storage.trials import quality_check_trial
from usv_faults.training.train_sdae import train_sdae


def _not_implemented(name: str, milestone: str) -> int:
    print(f"{name} is planned for {milestone} and is not implemented in objective 1.", file=sys.stderr)
    return 2


def _cmd_attach_data(args: argparse.Namespace) -> int:
    if args.source != "synthetic":
        print("Only the synthetic source is implemented in objective 1.", file=sys.stderr)
        return 2
    created = SyntheticUSVSource.from_config_path(args.config).attach(args.out)
    print(f"Attached {len(created)} synthetic trials under {args.out}")
    return 0


def _cmd_qc(args: argparse.Namespace) -> int:
    report = quality_check_trial(args.trial)
    print(
        f"{report.trial_id}: {report.status} "
        f"({report.sample_count} samples, {report.sample_rate_hz_estimate:.2f} Hz)"
    )
    for warning in report.warnings:
        print(f"WARNING: {warning}", file=sys.stderr)
    for error in report.errors:
        print(f"ERROR: {error}", file=sys.stderr)
    return 1 if report.errors else 0


def _cmd_preview(args: argparse.Namespace) -> int:
    out = write_preview_csv(args.trial, args.out)
    print(f"Wrote telemetry preview to {out}")
    print(f"Wrote preview summary to {out.with_name(out.stem + '_summary.csv')}")
    return 0


def _cmd_make_dataset(args: argparse.Namespace) -> int:
    result = make_dataset(args.config, args.out)
    print(
        f"Created dataset {result['dataset_id']} with {result['window_count']} windows, "
        f"{result['input_dim']} features, and {result['trial_count']} trials at {result['out_dir']}"
    )
    return 0


def _cmd_train_sdae(args: argparse.Namespace) -> int:
    result = train_sdae(args.dataset, args.config, args.out)
    print(
        f"Trained {result['run_id']} for {result['epochs']} epochs on "
        f"{result['train_windows']} healthy windows; threshold={result['threshold']:.6f}; "
        f"artifacts={result['out_dir']}"
    )
    return 0


def _cmd_build_dictionary(args: argparse.Namespace) -> int:
    result = build_fault_dictionary(args.model, args.dataset, args.config, args.out)
    print(
        f"Built dictionary {result['dictionary_id']} with "
        f"{result['cluster_count']} non-noise clusters and "
        f"{result['dictionary_entry_count']} entries from "
        f"{result['candidate_window_count']} candidate anomaly windows; artifacts={result['out_dir']}"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="usv-faults",
        description="USV fault detection proof-of-concept CLI.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    attach = subparsers.add_parser("attach-data", help="Attach data into raw trial folders.")
    attach.add_argument("--source", required=True)
    attach.add_argument("--config", required=True, type=Path)
    attach.add_argument("--out", required=True, type=Path)
    attach.set_defaults(func=_cmd_attach_data)

    qc = subparsers.add_parser("qc", help="Run quality checks for a raw trial folder.")
    qc.add_argument("--trial", required=True, type=Path)
    qc.set_defaults(func=_cmd_qc)

    preview = subparsers.add_parser("preview", help="Write a lightweight telemetry CSV preview.")
    preview.add_argument("--trial", required=True, type=Path)
    preview.add_argument("--out", required=False, type=Path, default=None)
    preview.set_defaults(func=_cmd_preview)

    dataset = subparsers.add_parser("make-dataset", help="Create processed windows from raw trials.")
    dataset.add_argument("--config", required=True, type=Path)
    dataset.add_argument("--out", required=True, type=Path)
    dataset.set_defaults(func=_cmd_make_dataset)

    train = subparsers.add_parser("train-sdae", help="Train the baseline SDAE.")
    train.add_argument("--dataset", required=True, type=Path)
    train.add_argument("--config", required=True, type=Path)
    train.add_argument("--out", required=True, type=Path)
    train.set_defaults(func=_cmd_train_sdae)

    dictionary = subparsers.add_parser("build-dictionary", help="Build a latent fault dictionary.")
    dictionary.add_argument("--model", required=True, type=Path)
    dictionary.add_argument("--dataset", required=True, type=Path)
    dictionary.add_argument("--config", required=True, type=Path)
    dictionary.add_argument("--out", required=True, type=Path)
    dictionary.set_defaults(func=_cmd_build_dictionary)

    for name, milestone in [
        ("evaluate", "Milestone 5"),
        ("run", "Milestone 5"),
    ]:
        placeholder = subparsers.add_parser(name)
        placeholder.set_defaults(func=lambda args, n=name, m=milestone: _not_implemented(n, m))

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
