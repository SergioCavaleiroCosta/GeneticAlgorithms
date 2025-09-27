"""Analyze all example benchmark run outputs.

For each subfolder in examples/ that contains output_* run directories, this script:
  - Runs the analysis (using analyze_example.run_analysis)
  - Writes examples/<case>/analysis_summary.json
  - Writes examples/<case>/analysis_runs.csv (per-run details)

It also consolidates a summary CSV at scripts/all_examples_summary.csv for quick comparison.

Usage (PowerShell):
    uv run python scripts/analyze_all_examples.py --tol 1e-6

Options:
    --tol          Success tolerance (default 1e-6)
    --limit        Limit to N most recent runs per example
    --examples     Comma-separated subset (default: detect all)
    --quantiles    Quantiles string passed to underlying analysis
    --out-csv      Path for consolidated summary CSV (default scripts/all_examples_summary.csv)
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import List, Tuple, Dict, Any, Set

from analyze_example import run_analysis, EXAMPLES_DIR  # type: ignore


def discover_examples() -> List[str]:
    names: List[str] = []
    for p in EXAMPLES_DIR.iterdir():
        if p.is_dir() and not p.name.startswith("_"):
            # Heuristic: treat as example if it has at least one output_ dir
            has_output = any(c.name.startswith("output_") for c in p.iterdir() if c.is_dir())
            if has_output:
                names.append(p.name)
    names.sort()
    return names


def main() -> int:
    ap = argparse.ArgumentParser(description="Analyze all examples")
    ap.add_argument("--tol", type=float, default=1e-6, help="Success tolerance")
    ap.add_argument("--limit", type=int, default=0, help="Limit to N most recent runs per example")
    ap.add_argument("--examples", help="Comma-separated subset of examples to analyze")
    ap.add_argument("--quantiles", default="0.25,0.5,0.75,0.95", help="Quantiles list for best objectives")
    ap.add_argument("--out-csv", default=str(Path(__file__).parent / "all_examples_summary.csv"), help="Path for consolidated summary CSV")
    args = ap.parse_args()

    if args.examples:
        example_list = [e.strip() for e in args.examples.split(",") if e.strip()]
    else:
        example_list = discover_examples()

    if not example_list:
        print("No examples discovered.")
        return 1

    consolidated: List[Tuple[str, Dict[str, Any]]] = []
    for ex_name in example_list:
        print(f"\n=== {ex_name} ===")
        try:
            summary, per_run = run_analysis(ex_name, tol=args.tol, limit=args.limit, quantiles=args.quantiles)
        except Exception as e:  # noqa: BLE001
            print(f"Skipping {ex_name}: {e}")
            continue
        consolidated.append((ex_name, summary))
        ex_dir = EXAMPLES_DIR / ex_name
        # Write summary JSON
        with (ex_dir / "analysis_summary.json").open("w") as jf:
            json.dump(summary, jf, indent=2)
        # Write per-run CSV
        with (ex_dir / "analysis_runs.csv").open("w", newline="") as cf:
            w = csv.writer(cf)
            w.writerow(["run_dir", "best_objective", "final_iteration", "first_success_iteration"])        
            for name, b, it, fs in per_run:
                w.writerow([name, b, it, fs if fs is not None else ""])   
        print(f"Wrote analysis_summary.json and analysis_runs.csv for {ex_name}")

    if not consolidated:
        print("No example analyses completed.")
        return 2

    # Build unified CSV header (include dynamic quantile keys)
    # Collect all keys
    all_keys: Set[str] = set()
    for _, summ in consolidated:
        all_keys.update(summ.keys())
    # Order some core keys first
    core_order = [
        "example","best_min","best_max","best_mean","best_median","best_stdev","problem_dimension"
    ]
    quantile_keys = sorted([k for k in all_keys if k.startswith("q") and k[1:].isdigit()], key=lambda s: int(s[1:]))
    remaining = [k for k in all_keys if k not in core_order and k not in quantile_keys]
    header = core_order + quantile_keys + remaining

    out_csv_path = Path(args.out_csv)
    with out_csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for name, summ in consolidated:
            row = [summ.get(k, "") for k in header]
            w.writerow(row)
    print(f"\nConsolidated summary CSV written to {out_csv_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
