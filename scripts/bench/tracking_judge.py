#!/usr/bin/env python3
"""Decide whether a suite of runs tracks demonstrations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from robogate.eval import find_run
from robogate.run import Run
from robogate.suite import load_suite


def judge(
    suite: Path,
    runs_root: Path,
    *,
    inter_demo_l2: float = 0.918,
    version_contains: str | None = None,
) -> dict:
    ids = [scenario.id for _, scenario in load_suite(suite)]
    frame0 = []
    corrs = []
    l2s = []
    for sid in ids:
        run = Run.load(
            find_run(runs_root, sid, version_contains=version_contains, unperturbed=True)
        )
        pred = np.asarray(run.output("action").get_column("value").to_list(), dtype=np.float64)
        rec = np.asarray(
            run.output("action", recorded=True).get_column("value").to_list(), dtype=np.float64
        )
        n = min(len(pred), len(rec))
        pred, rec = pred[:n], rec[:n]
        frame0.append(pred[0])
        l2s.append(float(np.mean(np.linalg.norm(pred - rec, axis=1))))
        dim_corr = []
        for d in range(pred.shape[1]):
            if pred[:, d].std() < 1e-8 or rec[:, d].std() < 1e-8:
                dim_corr.append(0.0)
                continue
            dim_corr.append(float(np.corrcoef(pred[:, d], rec[:, d])[0, 1]))
        corrs.append(float(np.median(dim_corr)))
    frame0_std = float(np.stack(frame0).std())
    median_corr = float(np.median(corrs))
    mean_l2 = float(np.mean(l2s))
    ok = frame0_std > 0.01 and median_corr > 0.5 and mean_l2 < inter_demo_l2 * 2
    return {
        "ok": ok,
        "frame0_std": frame0_std,
        "median_corr": median_corr,
        "mean_l2": mean_l2,
        "l2_limit": inter_demo_l2 * 2,
        "by_scenario_l2": dict(zip(ids, l2s, strict=True)),
        "by_scenario_corr": dict(zip(ids, corrs, strict=True)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("suite", type=Path)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--inter-demo-l2", type=float, default=0.918)
    parser.add_argument("--version-contains", default=None)
    args = parser.parse_args()
    print(
        json.dumps(
            judge(
                args.suite,
                args.runs,
                inter_demo_l2=args.inter_demo_l2,
                version_contains=args.version_contains,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
