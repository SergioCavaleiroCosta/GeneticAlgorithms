"""Analyze GA run outputs for a given example.

Assumptions:
- Each run creates examples/<example>/output_<timestamp>/ with a population logger subdir 'data'.
- Final iteration CSV has the highest iteration number: iter_<k>.csv
- Column schema: iteration,candidate_index,objective,<norm cols>,<real cols>

Metrics computed:
- total_runs: number of output_* directories
- successes: runs where best objective <= --tol
- success_rate: successes / total_runs
- best_objective stats: min, max, mean, median, std
- iteration_of_best (per run): iteration value of row with best objective; summarized mean/median.

Usage:
    uv run python scripts/analyze_example.py --example ackley --tol 1e-6
    uv run python scripts/analyze_example.py --example rastrigin --tol 1e-3 --limit 50

Options:
    --example / -e   Example folder name under examples/
    --tol            Success tolerance on objective (default 1e-6)
    --limit          Limit number of most recent runs (sorted by timestamp) to analyze
    --csv            Output detailed per-run CSV summary path
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics as stats
from pathlib import Path
from typing import List, Tuple, Sequence, Dict, Optional, Callable, Any

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = REPO_ROOT / "examples"


def list_run_dirs(example: str) -> List[Path]:
    base = EXAMPLES_DIR / example
    if not base.is_dir():
        raise SystemExit(f"Example directory not found: {base}")
    runs = [p for p in base.iterdir() if p.is_dir() and p.name.startswith("output_")]
    # Sort by timestamp substring (after output_)
    runs.sort(key=lambda p: p.name.split("output_")[-1])
    return runs


def parse_run(run_dir: Path, success_tol: float) -> Tuple[float, int, Optional[int], List[float], List[str]]:
    """Return (best_objective, final_iteration, first_success_iteration | None, best_params, aliases).

    best_params are the real-domain parameter values for the best objective within the run.
    aliases are the parameter names (consistent across iterations / runs for the same example).
    """
    data_dir = run_dir / "data"
    if not data_dir.is_dir():
        raise FileNotFoundError(f"data directory missing in {run_dir}")
    indexed: List[Tuple[int, Path]] = []
    for p in data_dir.glob("iter_*.csv"):
        stem = p.stem
        if not stem.startswith("iter_"):
            continue
        try:
            it = int(stem.split("iter_")[-1])
        except ValueError:
            continue
        indexed.append((it, p))
    if not indexed:
        raise FileNotFoundError(f"No iter_*.csv files in {data_dir}")
    indexed.sort(key=lambda t: t[0])
    best_obj: Optional[float] = None
    best_params: List[float] = []
    first_success: Optional[int] = None
    aliases: List[str] = []
    final_iteration = indexed[-1][0]
    for iteration, file_path in indexed:  # sequential by iteration
        with file_path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            header = reader.fieldnames or []
            if not aliases:
                # infer aliases: header = iteration,candidate_index,objective, <alias*_n x dim>, <alias x dim>
                if len(header) < 5:
                    raise RuntimeError("Unexpected CSV header length")
                dim = (len(header) - 3) // 2
                aliases = list(header[-dim:])  # ensure concrete list[str]
            for row in reader:
                try:
                    obj = float(row["objective"])  # type: ignore[index]
                except Exception:
                    continue
                if best_obj is None or obj < best_obj:
                    best_obj = obj
                    # capture real values (last dim columns)
                    dim = len(aliases)
                    best_params = [float(row[a]) for a in aliases]
                if first_success is None and obj <= success_tol:
                    first_success = iteration
    if best_obj is None:
        raise RuntimeError(f"Could not determine best objective in {run_dir}")
    return best_obj, final_iteration, first_success, best_params, aliases


def run_analysis(example: str, tol: float = 1e-6, limit: int = 0, quantiles: str = "0.25,0.5,0.75,0.95") -> Tuple[Dict[str, Any], List[Tuple[str, float, int, Optional[int]]]]:
    """Programmatic entrypoint returning (summary, per_run).

    per_run entries: (run_dir_name, best_objective, final_iteration, first_success_iteration)
    """
    runs = list_run_dirs(example)
    if limit > 0:
        runs = runs[-limit:]
    if not runs:
        raise ValueError("No runs found")

    per_run: List[Tuple[str, float, int, Optional[int]]] = []
    best_param_vectors: List[List[float]] = []
    aliases_ref: List[str] = []
    for r in runs:
        try:
            best_obj, it, first_success, best_params, aliases = parse_run(r, tol)
        except Exception:  # noqa: BLE001
            # Skip corrupted run directories but continue others
            continue  # silently skip; caller can infer by counts
        per_run.append((r.name, best_obj, it, first_success))
        if not aliases_ref:
            aliases_ref = aliases
        else:
            if aliases_ref != aliases:
                raise RuntimeError("Alias mismatch across runs; inconsistent parameterization")
        best_param_vectors.append(best_params)

    if not per_run:
        raise ValueError("No valid runs parsed")

    best_values: List[float] = [b for _, b, _, _ in per_run]
    # Removed fields (iterations, success counts) per new requirements; keep only best_values for objective stats.

    def safe_stat(fn: Callable[[Sequence[float]], float], data: Sequence[float], default: float) -> float:
        try:
            return fn(data)
        except Exception:
            return default

    # Parse quantiles
    q_vals: List[float] = []
    if quantiles:
        for q in quantiles.split(","):
            try:
                qf = float(q)
                if 0.0 < qf < 1.0:
                    q_vals.append(qf)
            except ValueError:
                continue
        q_vals = sorted(set(q_vals))

    def quantile(data: Sequence[float], q: float) -> float:
        if not data:
            return float("nan")
        idx = int(round((len(data) - 1) * q))
        return sorted(data)[idx]

    quantile_results: Dict[str, float] = {f"q{int(q*100)}": quantile(best_values, q) for q in q_vals}

    summary: Dict[str, Any] = {
        "example": example,
        "best_min": min(best_values),
        "best_max": max(best_values),
        "best_mean": safe_stat(stats.fmean, best_values, float("nan")),
        "best_median": safe_stat(stats.median, best_values, float("nan")),
        "best_stdev": safe_stat(stats.pstdev, best_values, float("nan")),
    }
    summary.update(quantile_results)

    # Parameter statistics (using best solution vector from each run)
    if best_param_vectors:
        dim = len(best_param_vectors[0])
        summary["problem_dimension"] = dim
        # transpose
        param_stats: Dict[str, Dict[str, float]] = {}
        # Precompute per-parameter lists
        cols: List[List[float]] = [[vec[i] for vec in best_param_vectors if len(vec) == dim] for i in range(dim)]
        def quant(data: Sequence[float], q: float) -> float:
            if not data:
                return float("nan")
            idx = int(round((len(data) - 1) * q))
            return sorted(data)[idx]
        for i, alias in enumerate(aliases_ref):
            data = cols[i]
            if not data:
                continue
            # Specific quantiles requested: 0.25,0.5,0.74,0.95
            ps = {
                "min": min(data),
                "max": max(data),
                "mean": safe_stat(stats.fmean, data, float("nan")),
                "median": safe_stat(stats.median, data, float("nan")),
                "stdev": safe_stat(stats.pstdev, data, float("nan")),
                "q25": quant(data, 0.25),
                "q50": quant(data, 0.50),
                "q74": quant(data, 0.74),
                "q95": quant(data, 0.95),
            }
            param_stats[alias] = ps
        summary["param_stats"] = param_stats

    return summary, per_run


def main() -> int:
    ap = argparse.ArgumentParser(description="Analyze GA runs for an example")
    ap.add_argument("--example", "-e", required=True, help="Example name (folder under examples)")
    ap.add_argument("--tol", type=float, default=1e-6, help="Success tolerance on objective value")
    ap.add_argument("--limit", type=int, default=0, help="Limit to N most recent runs (after sorting)")
    ap.add_argument("--csv", help="Optional path to write per-run detailed CSV")
    ap.add_argument("--quantiles", default="0.25,0.5,0.75,0.95", help="Comma-separated quantiles to report for best objectives")
    ap.add_argument("--summary-json", help="Optional path to write summary JSON")
    args = ap.parse_args()

    try:
        summary, per_run = run_analysis(args.example, args.tol, args.limit, args.quantiles)
    except ValueError as e:  # no runs or no valid runs
        print(str(e))
        return 1

    print("=== Analysis Summary ===")
    for k, v in summary.items():
        print(f"{k:15s}: {v}")

    if args.csv:
        out_path = Path(args.csv)
        with out_path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["run_dir", "best_objective", "final_iteration", "first_success_iteration", "success"])
            for name, b, it, fs in per_run:
                w.writerow([name, b, it, fs if fs is not None else "", b <= args.tol])
        print(f"Per-run details written to {out_path}")

    if args.summary_json:
        js_path = Path(args.summary_json)
        with js_path.open("w") as jf:
            json.dump(summary, jf, indent=2)
        print(f"Summary JSON written to {js_path}")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
