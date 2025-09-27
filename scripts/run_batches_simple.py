"""Minimal batched multiprocessing runner for GA examples.

Exactly what was requested:
  - Discover example case directories containing run_ga.py
  - Define run_case(case_path)
  - Build path_list
  - for each round (default 100): create a Pool(size = len(path_list) or specified) and pool.map(run_case, path_list)

No extra aggregation, minimal output.

Usage:
    uv run python scripts/run_batches_simple.py            # 100 rounds, auto pool size
    uv run python scripts/run_batches_simple.py --rounds 5
    uv run python scripts/run_batches_simple.py --rounds 50 --pool-size 8

Exit code is non-zero if any case fails in any round.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = REPO_ROOT / "examples"


def discover_case_scripts() -> List[Path]:
    scripts: List[Path] = []
    for p in sorted(EXAMPLES_DIR.iterdir()):
        run_file = p / "run_ga.py"
        if p.is_dir() and run_file.is_file():
            scripts.append(run_file)
    return scripts


def run_case(script_path: Path) -> int:
    # Run the script with the current Python executable; inherit stdout/stderr.
    proc = subprocess.run([sys.executable, str(script_path)], cwd=str(REPO_ROOT))
    return proc.returncode or 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Simple batched multiprocessing runner")
    parser.add_argument("--rounds", type=int, default=100, help="Number of rounds (batches)")
    parser.add_argument("--pool-size", type=int, default=0, help="Pool size (default: min(cpu_count, number of cases))")
    parser.add_argument("--stop-on-fail", action="store_true", help="Stop immediately on first failing round")
    args = parser.parse_args(argv)

    if args.rounds < 1:
        print("--rounds must be >= 1", file=sys.stderr)
        return 2

    scripts = discover_case_scripts()
    if not scripts:
        print("No example scripts found.", file=sys.stderr)
        return 3

    pool_size = args.pool_size if args.pool_size > 0 else min(cpu_count(), len(scripts))
    print(f"Discovered {len(scripts)} cases. Rounds={args.rounds} PoolSize={pool_size}")

    any_fail = False
    for round_idx in range(1, args.rounds + 1):
        print(f"=== Round {round_idx}/{args.rounds} ===")
        # New pool each round per user specification
        with Pool(processes=pool_size) as pool:
            exit_codes = pool.map(run_case, scripts)
        failed_cases = [scripts[i] for i, code in enumerate(exit_codes) if code != 0]
        if failed_cases:
            any_fail = True
            print(f"Round {round_idx} FAIL: {len(failed_cases)} case(s) failed")
            for path in failed_cases:
                print(f"  - {path.parent.name}")
            if args.stop_on_fail:
                print("Stopping early due to failure and --stop-on-fail")
                break
        else:
            print(f"Round {round_idx} OK")

    if any_fail:
        return 5
    print("All rounds completed successfully.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
