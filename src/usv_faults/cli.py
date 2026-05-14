from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from usv_faults.data_sources.synthetic_usv import SyntheticUSVSource
from usv_faults.storage.trials import quality_check_trial


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

    for name, milestone in [
        ("preview", "Milestone 2"),
        ("make-dataset", "Milestone 2"),
        ("train-sdae", "Milestone 3"),
        ("build-dictionary", "Milestone 4"),
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
