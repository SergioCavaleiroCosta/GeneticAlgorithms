"""Run multiple NSGA-II experiments for multi-objective benchmark problems.

This script runs multiple NSGA-II experiments in batch mode for all multi-objective examples.
Similar to run_batches_simple.py but specifically for multi-objective optimization.

Usage:
    uv run python scripts/run_multiobjective_batches.py --runs 10
    uv run python scripts/run_multiobjective_batches.py --runs 5 --example zdt1
"""
import argparse
import subprocess
import sys
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = REPO_ROOT / "examples"


def find_multiobjective_examples() -> List[str]:
    """Find all examples that have NSGA-II implementation."""
    mo_examples: List[str] = []
    
    for example_dir in EXAMPLES_DIR.iterdir():
        if not example_dir.is_dir():
            continue
            
        # Check if it has NSGA-II implementation
        nsga_ii_file = example_dir / "run_nsga_ii.py"
        if nsga_ii_file.exists():
            mo_examples.append(example_dir.name)
    
    return sorted(mo_examples)


def run_example_batch(example: str, num_runs: int) -> None:
    """Run multiple experiments for a single multi-objective example."""
    print(f"\n=== Running {num_runs} experiments for {example} ===")
    
    example_dir = EXAMPLES_DIR / example
    if not example_dir.exists():
        print(f"Example directory not found: {example_dir}")
        return
    
    nsga_ii_script = example_dir / "run_nsga_ii.py"
    if not nsga_ii_script.exists():
        print(f"NSGA-II script not found: {nsga_ii_script}")
        return
    
    success_count = 0
    
    for run_num in range(1, num_runs + 1):
        print(f"  Run {run_num}/{num_runs}...", end=" ", flush=True)
        
        try:
            # Run the NSGA-II script
            result = subprocess.run(
                [sys.executable, str(nsga_ii_script)],
                cwd=str(REPO_ROOT),
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode == 0:
                print("✓")
                success_count += 1
            else:
                print(f"✗ (exit code {result.returncode})")
                if result.stderr:
                    print(f"    Error: {result.stderr.strip()}")
                    
        except subprocess.TimeoutExpired:
            print("✗ (timeout)")
        except Exception as e:
            print(f"✗ (exception: {e})")
    
    print(f"  Summary: {success_count}/{num_runs} successful runs")


def main() -> int:
    ap = argparse.ArgumentParser(description="Run multiple NSGA-II experiments")
    ap.add_argument("--runs", "-r", type=int, default=10, help="Number of runs per example")
    ap.add_argument("--example", "-e", help="Run only specific example (default: all)")
    args = ap.parse_args()
    
    if args.example:
        examples = [args.example]
        if not (EXAMPLES_DIR / args.example / "run_nsga_ii.py").exists():
            print(f"Error: {args.example} does not have NSGA-II implementation")
            return 1
    else:
        examples = find_multiobjective_examples()
        if not examples:
            print("No multi-objective examples found!")
            return 1
        print(f"Found {len(examples)} multi-objective examples: {', '.join(examples)}")
    
    total_runs = len(examples) * args.runs
    print(f"Will run {total_runs} total experiments ({args.runs} runs × {len(examples)} examples)")
    
    try:
        for example in examples:
            run_example_batch(example, args.runs)
        
        print(f"\n=== Batch Complete ===")
        print(f"All experiments finished. Use analyze_all_multiobjective.py to analyze results.")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\nBatch interrupted by user.")
        return 1
    except Exception as e:
        print(f"\nError: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())