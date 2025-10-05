"""Analyze all multi-objective examples and create per-example analysis files.

This script runs the multi-objective analysis for all available NSGA-II examples and creates
analysis files in each example directory (following the same pattern as single-objective examples):
- examples/<name>/analysis_runs.csv: Per-run detailed metrics
- examples/<name>/analysis_summary.json: Statistical summary

Also creates a unified summary CSV for easy comparison across problems.

Usage:
    uv run python scripts/analyze_all_multiobjective.py
    uv run python scripts/analyze_all_multiobjective.py --limit 5 --output custom_summary.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import List, Dict, Union
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scripts.analyze_multiobjective import run_multiobjective_analysis

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = REPO_ROOT / "examples"


def find_multiobjective_examples() -> List[str]:
    """Find all examples that appear to be multi-objective (have NSGA-II implementation)."""
    mo_examples: List[str] = []
    
    for example_dir in EXAMPLES_DIR.iterdir():
        if not example_dir.is_dir():
            continue
            
        # Check if it has NSGA-II implementation
        nsga_ii_file = example_dir / "run_nsga_ii.py"
        if nsga_ii_file.exists():
            mo_examples.append(example_dir.name)
    
    return sorted(mo_examples)


def get_reference_points() -> Dict[str, str]:
    """Get appropriate reference points for known multi-objective problems."""
    return {
        "zdt1": "1.1,11.0",      # ZDT1: f1 ∈ [0,1], f2 ∈ [0,9]
        "zdt2": "1.1,11.0",      # ZDT2: f1 ∈ [0,1], f2 ∈ [0,9]  
        "zdt3": "1.1,11.0",      # ZDT3: f1 ∈ [0,1], f2 ∈ [-1,9]
        "dtlz2": "1.1,1.1",      # DTLZ2: f1,f2 ∈ [0,1] (2-objective case)
        "schaffer_n1": "11.0,11.0"  # Schaffer N.1: f1,f2 can be large
    }


def flatten_summary(summary: Dict[str, Any]) -> Dict[str, Union[str, int, float]]:
    """Flatten nested summary structure for CSV export."""
    flat = {}
    
    for key, value in summary.items():
        if isinstance(value, dict):
            # Nested statistics - flatten with prefix
            for sub_key, sub_value in value.items():
                flat[f"{key}_{sub_key}"] = sub_value
        else:
            flat[key] = value
    
    return flat


def analyze_all_multiobjective_examples(limit: int = 0) -> List[Dict[str, Union[str, int, float]]]:
    """Analyze all multi-objective examples and create per-example analysis files."""
    examples = find_multiobjective_examples()
    reference_points = get_reference_points()
    
    if not examples:
        print("No multi-objective examples found!")
        return []
    
    print(f"Found {len(examples)} multi-objective examples: {', '.join(examples)}")
    
    all_results: List[Dict[str, Union[str, int, float]]] = []
    
    for example in examples:
        print(f"\nAnalyzing {example}...")
        
        try:
            ref_point = reference_points.get(example)
            summary, per_run = run_multiobjective_analysis(example, ref_point, limit)
            
            # Write per-example files (same pattern as single-objective)
            example_dir = EXAMPLES_DIR / example
            
            # Write per-run CSV to example directory
            csv_path = example_dir / "analysis_runs.csv"
            if per_run:
                fieldnames = list(per_run[0].keys())
                with csv_path.open("w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(per_run)
                print(f"  ✓ Per-run analysis: {csv_path.relative_to(REPO_ROOT)}")
            
            # Write summary JSON to example directory  
            json_path = example_dir / "analysis_summary.json"
            with json_path.open("w") as f:
                json.dump(summary, f, indent=2)
            print(f"  ✓ Summary: {json_path.relative_to(REPO_ROOT)}")
            
            # Flatten summary for CSV and add to unified results
            flat_summary = flatten_summary(summary)
            all_results.append(flat_summary)
            
            print(f"  ✓ {summary['valid_runs']} valid runs analyzed")
            
            # Print key metrics (using nested structure)
            if 'hypervolume' in summary and isinstance(summary['hypervolume'], dict):
                print(f"  ✓ Hypervolume (mean): {summary['hypervolume']['mean']:.6f}")
            if 'spacing' in summary and isinstance(summary['spacing'], dict):
                print(f"  ✓ Spacing (mean): {summary['spacing']['mean']:.6f}")
            if 'non_dominated_solutions' in summary and isinstance(summary['non_dominated_solutions'], dict):
                print(f"  ✓ Non-dominated solutions (mean): {summary['non_dominated_solutions']['mean']:.1f}")
                
        except Exception as e:
            print(f"  ✗ Error analyzing {example}: {e}")
            # Add error entry
            all_results.append({
                "example": example,
                "error": str(e),
                "total_runs": 0,
                "valid_runs": 0
            })
    
    return all_results


def write_summary_csv(results: List[Dict[str, Union[str, int, float]]], output_path: Path) -> None:
    """Write results to CSV file."""
    if not results:
        print("No results to write!")
        return
    
    # Determine all possible fieldnames
    fieldnames: set[str] = set()
    for result in results:
        fieldnames.update(result.keys())
    
    # Order fieldnames logically  
    ordered_fields = ["example", "total_runs", "valid_runs"]
    metric_fields = [f for f in sorted(fieldnames) if f not in ordered_fields and not f.startswith("error")]
    error_fields = [f for f in fieldnames if f.startswith("error")]
    
    final_fieldnames = ordered_fields + metric_fields + error_fields
    
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=final_fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\nSummary written to: {output_path}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Analyze all multi-objective examples")
    ap.add_argument("--limit", type=int, default=0, help="Limit to N most recent runs per example")
    ap.add_argument("--output", "-o", default="all_multiobjective_summary.csv", help="Output CSV file name")
    args = ap.parse_args()
    
    try:
        results = analyze_all_multiobjective_examples(args.limit)
        
        if results:
            output_path = REPO_ROOT / "scripts" / args.output
            write_summary_csv(results, output_path)
            
            # Print summary statistics
            valid_examples = [r for r in results if "error" not in r]
            error_examples = [r for r in results if "error" in r]
            
            print(f"\n=== Summary ===")
            print(f"✓ {len(valid_examples)} examples analyzed successfully")
            if error_examples:
                print(f"✗ {len(error_examples)} examples had errors:")
                for err_result in error_examples:
                    print(f"  - {err_result['example']}: {err_result['error']}")
            
            if valid_examples:
                print(f"\n=== Key Multi-Objective Metrics ===")
                
                # Hypervolume comparison
                hv_results = [(r['example'], r.get('hypervolume_mean', 0)) for r in valid_examples if 'hypervolume_mean' in r]
                if hv_results:
                    best_hv = max(hv_results, key=lambda x: x[1])
                    print(f"Best Hypervolume: {best_hv[0]} ({best_hv[1]:.6f})")
                
                # Spacing comparison (lower is better)
                spacing_results = [(r['example'], r.get('spacing_mean', float('inf'))) for r in valid_examples if 'spacing_mean' in r]
                if spacing_results:
                    best_spacing = min(spacing_results, key=lambda x: x[1])
                    print(f"Best Spacing: {best_spacing[0]} ({best_spacing[1]:.6f})")
                
                # Non-dominated solutions
                nd_results = [(r['example'], r.get('non_dominated_solutions_mean', 0)) for r in valid_examples if 'non_dominated_solutions_mean' in r]
                if nd_results:
                    most_nd = max(nd_results, key=lambda x: x[1])
                    print(f"Most Non-dominated Solutions: {most_nd[0]} ({most_nd[1]:.1f})")
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())