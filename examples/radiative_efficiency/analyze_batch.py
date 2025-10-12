"""Analyze batch optimization results.

This script computes statistics from batch optimization runs and generates
summary reports.

Usage:
    python analyze_batch.py <batch_directory>
    uv run python examples/radiative_efficiency/analyze_batch.py examples/radiative_efficiency/batch_<timestamp>_<hash>
"""
from __future__ import annotations

from pathlib import Path
import sys
import json
from typing import Any
import numpy as np


def compute_statistics(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute statistics from multiple optimization runs.
    
    Args:
        results: List of result dictionaries from individual runs
        
    Returns:
        Dictionary with aggregated statistics
    """
    successful_runs = [r for r in results if r.get("success", False)]
    n_successful = len(successful_runs)
    n_total = len(results)
    
    if n_successful == 0:
        return {
            "n_runs": n_total,
            "n_successful": 0,
            "success_rate": 0.0,
            "error": "No successful runs"
        }
    
    # Extract arrays for statistics
    efficiencies = np.array([r["radiative_efficiency"] for r in successful_runs])
    execution_times = np.array([r["execution_time"] for r in successful_runs])
    evaluations = np.array([r["evaluations"] for r in successful_runs])
    iterations = np.array([r["iterations"] for r in successful_runs])
    
    # Get parameter names from first successful run
    param_names = list(successful_runs[0]["parameters"].keys())
    
    # Collect parameter values
    param_arrays = {
        name: np.array([r["parameters"][name] for r in successful_runs])
        for name in param_names
    }
    
    # Find best run
    best_idx = np.argmax(efficiencies)
    best_run = successful_runs[best_idx]
    
    # Find worst run
    worst_idx = np.argmin(efficiencies)
    worst_run = successful_runs[worst_idx]
    
    # Compute statistics
    stats = {
        "summary": {
            "n_runs": n_total,
            "n_successful": n_successful,
            "success_rate": n_successful / n_total
        },
        "radiative_efficiency": {
            "best": float(np.max(efficiencies)),
            "worst": float(np.min(efficiencies)),
            "mean": float(np.mean(efficiencies)),
            "median": float(np.median(efficiencies)),
            "std": float(np.std(efficiencies)),
            "q25": float(np.percentile(efficiencies, 25)),
            "q75": float(np.percentile(efficiencies, 75))
        },
        "execution_time": {
            "mean": float(np.mean(execution_times)),
            "std": float(np.std(execution_times)),
            "min": float(np.min(execution_times)),
            "max": float(np.max(execution_times)),
            "total": float(np.sum(execution_times))
        },
        "evaluations": {
            "mean": float(np.mean(evaluations)),
            "std": float(np.std(evaluations)),
            "min": int(np.min(evaluations)),
            "max": int(np.max(evaluations)),
            "total": int(np.sum(evaluations))
        },
        "iterations": {
            "mean": float(np.mean(iterations)),
            "std": float(np.std(iterations)),
            "min": int(np.min(iterations)),
            "max": int(np.max(iterations))
        },
        "parameters": {
            name: {
                "mean": float(np.mean(param_arrays[name])),
                "std": float(np.std(param_arrays[name])),
                "min": float(np.min(param_arrays[name])),
                "max": float(np.max(param_arrays[name])),
                "median": float(np.median(param_arrays[name])),
                "q25": float(np.percentile(param_arrays[name], 25)),
                "q75": float(np.percentile(param_arrays[name], 75))
            }
            for name in param_names
        },
        "best_run": {
            "run_id": best_run["run_id"],
            "radiative_efficiency": best_run["radiative_efficiency"],
            "parameters": best_run["parameters"],
            "execution_time": best_run["execution_time"],
            "evaluations": best_run["evaluations"]
        },
        "worst_run": {
            "run_id": worst_run["run_id"],
            "radiative_efficiency": worst_run["radiative_efficiency"],
            "parameters": worst_run["parameters"],
            "execution_time": worst_run["execution_time"],
            "evaluations": worst_run["evaluations"]
        }
    }
    
    return stats


def main() -> None:
    """Analyze batch optimization results."""
    if len(sys.argv) < 2:
        print("Usage: python analyze_batch.py <batch_directory>")
        print("Example: python analyze_batch.py examples/radiative_efficiency/batch_20251010-120000_abc123")
        sys.exit(1)
    
    batch_dir = Path(sys.argv[1])
    
    if not batch_dir.exists():
        print(f"Error: Directory not found: {batch_dir}")
        sys.exit(1)
    
    results_file = batch_dir / "all_results.json"
    if not results_file.exists():
        print(f"Error: Results file not found: {results_file}")
        sys.exit(1)
    
    print("=" * 70)
    print("Batch Optimization Analysis")
    print("=" * 70)
    print(f"Batch directory: {batch_dir}")
    print(f"Loading results from: {results_file}")
    
    # Load results
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    print(f"Total runs found: {len(results)}")
    print("\nComputing statistics...")
    
    # Compute statistics
    stats = compute_statistics(results)
    
    # Save statistics
    stats_file = batch_dir / "statistics.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 70)
    print("Analysis Complete!")
    print("=" * 70)
    print(f"Successful runs: {stats['summary']['n_successful']}/{stats['summary']['n_runs']}")
    print(f"Success rate: {stats['summary']['success_rate']:.1%}")
    
    if stats['summary']['n_successful'] > 0:
        print(f"\nRadiative Efficiency Statistics:")
        print(f"  Best:   {stats['radiative_efficiency']['best']:.6f}")
        print(f"  Mean:   {stats['radiative_efficiency']['mean']:.6f} ± {stats['radiative_efficiency']['std']:.6f}")
        print(f"  Median: {stats['radiative_efficiency']['median']:.6f}")
        print(f"  Worst:  {stats['radiative_efficiency']['worst']:.6f}")
        print(f"  Q25-Q75: [{stats['radiative_efficiency']['q25']:.6f}, {stats['radiative_efficiency']['q75']:.6f}]")
        
        print(f"\nExecution Time:")
        print(f"  Total: {stats['execution_time']['total']:.1f}s ({stats['execution_time']['total']/60:.1f} min)")
        print(f"  Mean:  {stats['execution_time']['mean']:.2f}s ± {stats['execution_time']['std']:.2f}s")
        print(f"  Range: [{stats['execution_time']['min']:.2f}s, {stats['execution_time']['max']:.2f}s]")
        
        print(f"\nEvaluations:")
        print(f"  Total: {stats['evaluations']['total']:,}")
        print(f"  Mean:  {stats['evaluations']['mean']:.0f} ± {stats['evaluations']['std']:.0f}")
        print(f"  Range: [{stats['evaluations']['min']}, {stats['evaluations']['max']}]")
        
        print(f"\nBest Run (ID: {stats['best_run']['run_id']}):")
        print(f"  Efficiency: {stats['best_run']['radiative_efficiency']:.6f}")
        print(f"  Time: {stats['best_run']['execution_time']:.2f}s")
        print(f"  Evaluations: {stats['best_run']['evaluations']}")
        print(f"  Parameters:")
        for name, value in stats['best_run']['parameters'].items():
            param_stats = stats['parameters'][name]
            print(f"    {name:12s} = {value:10.6f}  (mean: {param_stats['mean']:.6f} ± {param_stats['std']:.6f})")
    
    print(f"\nStatistics saved to: {stats_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()
