"""Run GA example cases using a per-batch multiprocessing pool.

Behavior:
- Discover all subdirectories in ../examples containing run_ga.py
- For each batch index (0..runs-1):
    * Launch a multiprocessing.Pool (size = min(num_cases, concurrency or cpu count))
    * In that pool, execute each run_ga.py once (one process per case)
    * Wait for all to finish; record success/failure and duration
- Aggregate a summary at the end.

Rationale:
The outer loop represents repeated experimental batches; in each batch all cases are run once concurrently.
This differs from spawning N * cases processes overall; instead we reuse a fresh pool per batch for isolation.

Usage:
    uv run python scripts/run_batches_multiproc.py --runs 5
    uv run python scripts/run_batches_multiproc.py --runs 100 --concurrency 12
    python scripts/run_batches_multiproc.py --runs 20 --cases ackley rastrigin sphere

Options:
    --runs / -r            Number of batches (default 1)
    --concurrency / -c     Max parallel workers (default: min(cpu_count, number of cases))
    --cases                Optional explicit subset of case folder names to include
    --python               Python executable (default: sys.executable)
    --stop-on-fail         Abort remaining batches if any case fails in a batch
    --quiet                Reduce per-case stdout (only errors + summary)
    --list                 Only list discovered cases and exit

Exit code: 0 if all succeeded; non-zero if any failure.
"""
from __future__ import annotations

import argparse
import sys
import time
import subprocess
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import List, Sequence, Optional, Dict

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = REPO_ROOT / "examples"

@dataclass
class CaseResult:
    case: str
    ok: bool
    exit_code: int
    duration: float
    batch: int
    error: Optional[str] = None


def discover_cases() -> List[str]:
    cases: List[str] = []
    for p in sorted(EXAMPLES_DIR.iterdir()):
        if p.is_dir() and (p / "run_ga.py").is_file():
            cases.append(p.name)
    return cases


def run_case(args: tuple[str, str, bool, bool]) -> CaseResult:
    case, python_exec, quiet, capture_err = args
    script = EXAMPLES_DIR / case / "run_ga.py"
    start = time.time()
    proc = subprocess.Popen(
        [python_exec, str(script)],
        stdout=subprocess.PIPE if quiet else None,
        stderr=subprocess.PIPE if capture_err else None,
        text=True,
        cwd=str(REPO_ROOT),
    )
    stdout_data, stderr_data = proc.communicate()
    duration = time.time() - start
    ok = proc.returncode == 0
    err_text = stderr_data if (capture_err and stderr_data) else None
    if quiet and not ok and stdout_data:
        # include suppressed stdout on failure for diagnostics
        if err_text:
            err_text = stdout_data + "\n" + err_text
        else:
            err_text = stdout_data
    return CaseResult(case=case, ok=ok, exit_code=proc.returncode or 0, duration=duration, batch=-1, error=err_text)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run GA examples in batched multiprocessing pools")
    parser.add_argument("--runs", "-r", type=int, default=1, help="Number of batches (outer loop iterations)")
    parser.add_argument("--concurrency", "-c", type=int, default=0, help="Max concurrent processes per batch")
    parser.add_argument("--cases", nargs="*", help="Subset of case folder names to include")
    parser.add_argument("--python", default=sys.executable, help="Python executable to use")
    parser.add_argument("--stop-on-fail", action="store_true", help="Abort remaining batches on first failure")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-case stdout (still shows failures)")
    parser.add_argument("--list", action="store_true", help="List discovered (or filtered) cases and exit")
    parser.add_argument("--capture-stderr", action="store_true", help="Capture stderr for each case and show on failure")

    args = parser.parse_args(argv)

    if args.runs < 1:
        print("--runs must be >= 1", file=sys.stderr)
        return 2

    all_cases = discover_cases()
    if not all_cases:
        print("No cases discovered in examples/", file=sys.stderr)
        return 3

    if args.cases:
        missing = [c for c in args.cases if c not in all_cases]
        if missing:
            print(f"Unknown cases requested: {missing}", file=sys.stderr)
            return 4
        selected = args.cases
    else:
        selected = all_cases

    if args.list:
        print("Cases:")
        for c in selected:
            print(f"  {c}")
        return 0

    # Determine concurrency
    max_workers = args.concurrency if args.concurrency > 0 else min(cpu_count(), len(selected))

    print(f"Discovered {len(selected)} case(s). Running {args.runs} batch(es) with concurrency={max_workers}.")
    print(f"Python: {args.python}\n")

    summary: List[CaseResult] = []
    overall_fail = False

    for batch_idx in range(1, args.runs + 1):
        print(f"=== Batch {batch_idx}/{args.runs} ===")
        batch_start = time.time()
        # Prepare work args
        work = [(case, args.python, args.quiet, args.capture_stderr) for case in selected]
        # Using a context manager ensures processes terminate each batch
        with Pool(processes=max_workers) as pool:
            results: List[CaseResult] = pool.map(run_case, work)
        # Annotate batch index
        for r in results:
            r.batch = batch_idx
            summary.append(r)
        batch_duration = time.time() - batch_start
        # Report batch
        ok_count = sum(1 for r in results if r.ok)
        fail_count = len(results) - ok_count
        print(f"Batch {batch_idx} completed in {batch_duration:.2f}s: OK={ok_count} FAIL={fail_count}")
        if fail_count:
            overall_fail = True
            for r in results:
                if not r.ok:
                    print(f"  - {r.case} exit={r.exit_code} time={r.duration:.2f}s")
                    if r.error:
                        print("    stderr:")
                        for line in r.error.strip().splitlines():
                            print(f"      {line}")
            if args.stop_on_fail:
                print("Stopping early due to --stop-on-fail")
                break

    # Final summary
    print("\n=== Final Summary ===")
    by_case: Dict[str, List[CaseResult]] = {c: [] for c in selected}
    for r in summary:
        by_case[r.case].append(r)
    for case, rows in by_case.items():
        total = len(rows)
        fails = sum(1 for r in rows if not r.ok)
        avg_time = sum(r.duration for r in rows) / total if total else 0.0
        print(f"{case:20s} runs={total:3d} fails={fails:3d} avg_time={avg_time:6.2f}s")

    if overall_fail:
        return 5
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
