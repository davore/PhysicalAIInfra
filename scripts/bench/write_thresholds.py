#!/usr/bin/env python3
"""Write mean+3σ thresholds back into a suite of scenario yaml files."""

from __future__ import annotations

import argparse
from pathlib import Path

from robogate.bench.calib import thresholds_from_eval, write_suite_thresholds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("suite", type=Path)
    parser.add_argument("--eval", dest="eval_parquet", type=Path, required=True)
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument("--slices", type=Path, default=Path("slices"))
    parser.add_argument("--eval-id", default=None)
    parser.add_argument(
        "--bounds-pad",
        type=float,
        default=0.10,
        help="Pad recorded action envelope by this fraction (0.10 = +10%).",
    )
    parser.add_argument(
        "--version-contains",
        default=None,
        help="Only use runs whose target_version contains this when measuring lag.",
    )
    args = parser.parse_args()
    eval_id = args.eval_id or args.eval_parquet.stem
    thresholds = thresholds_from_eval(
        args.eval_parquet,
        args.runs,
        args.suite,
        slice_root=args.slices,
        bounds_pad=args.bounds_pad,
        version_contains=args.version_contains,
    )
    written = write_suite_thresholds(args.suite, thresholds, eval_id=eval_id)
    print(f"ok wrote {len(written)} scenarios max_l2={thresholds['max_l2']}")


if __name__ == "__main__":
    main()
