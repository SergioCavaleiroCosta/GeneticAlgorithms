#!/usr/bin/env python3
"""Master script to run comprehensive optimization framework analysis.

This script performs all major tasks in sequence:
1. Clean existing output directories
2. Run single-objective batch execution (100 runs × 21 examples)
3. Run multi-objective batch execution (100 runs × 5 examples)
4. Generate comprehensive analysis summaries

Usage:
    uv run python scripts/run_complete_analysis.py
    uv run python scripts/run_complete_analysis.py --concurrency 24 --runs 50
    
Options:
    --runs, -r          Number of runs per example (default: 100)
    --concurrency, -c   Number of concurrent processes for single-objective (default: 22)
    --skip-cleanup      Skip cleaning existing output directories
    --skip-single       Skip single-objective batch execution
    --skip-multi        Skip multi-objective batch execution  
    --skip-analysis     Skip analysis generation
    --dry-run           Show what would be executed without running
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"


class TaskRunner:
    """Manages execution of the complete analysis pipeline."""
    
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.start_time = time.time()
        self.task_times: List[tuple[str, float]] = []
    
    def log(self, message: str, level: str = "INFO") -> None:
        """Log a message with timestamp."""
        elapsed = time.time() - self.start_time
        print(f"[{elapsed:6.1f}s] {level}: {message}")
    
    def run_command(self, cmd: List[str], description: str, cwd: Optional[Path] = None) -> bool:
        """Run a command and return success status."""
        if self.args.dry_run:
            self.log(f"DRY RUN: Would execute: {' '.join(cmd)}")
            return True
        
        self.log(f"Starting: {description}")
        task_start = time.time()
        
        try:
            if cwd is None:
                cwd = REPO_ROOT
            
            result = subprocess.run(
                cmd,
                cwd=str(cwd),
                capture_output=True,
                text=True,
                timeout=7200  # 2 hour timeout
            )
            
            task_time = time.time() - task_start
            self.task_times.append((description, task_time))
            
            if result.returncode == 0:
                self.log(f"✅ Completed: {description} ({task_time:.1f}s)")
                if result.stdout.strip():
                    # Show key lines from output
                    lines = result.stdout.strip().split('\n')
                    if len(lines) > 10:
                        self.log(f"Output preview: {lines[0]}")
                        self.log(f"... ({len(lines)} total lines)")
                        self.log(f"Final: {lines[-1]}")
                    else:
                        for line in lines:
                            self.log(f"  {line}")
                return True
            else:
                self.log(f"❌ Failed: {description} (exit code {result.returncode})", "ERROR")
                if result.stderr:
                    self.log(f"Error output: {result.stderr.strip()}", "ERROR")
                if result.stdout:
                    self.log(f"Stdout: {result.stdout.strip()}", "ERROR")
                return False
                
        except subprocess.TimeoutExpired:
            task_time = time.time() - task_start
            self.task_times.append((f"{description} (TIMEOUT)", task_time))
            self.log(f"⏰ Timeout: {description} after {task_time:.1f}s", "ERROR")
            return False
        except Exception as e:
            task_time = time.time() - task_start
            self.task_times.append((f"{description} (ERROR)", task_time))
            self.log(f"💥 Exception in {description}: {e}", "ERROR")
            return False
    
    def clean_outputs(self) -> bool:
        """Clean existing output directories."""
        if self.args.skip_cleanup:
            self.log("Skipping cleanup (--skip-cleanup)")
            return True
        
        # Use PowerShell command for Windows compatibility
        if sys.platform == "win32":
            cmd = ["pwsh", "-Command", "Remove-Item -Path .\\examples\\*\\output_* -Recurse -Force -ErrorAction SilentlyContinue"]
        else:
            cmd = ["rm", "-rf", "examples/*/output_*"]
        
        return self.run_command(
            cmd,
            f"Clean existing output directories"
        )
    
    def run_single_objective_batch(self) -> bool:
        """Run single-objective batch execution."""
        if self.args.skip_single:
            self.log("Skipping single-objective batch (--skip-single)")
            return True
        
        cmd = [
            sys.executable, 
            str(SCRIPTS_DIR / "run_batches_multiproc.py"),
            "-r", str(self.args.runs),
            "-c", str(self.args.concurrency)
        ]
        
        return self.run_command(
            cmd,
            f"Single-objective batch execution ({self.args.runs} runs, {self.args.concurrency} cores)"
        )
    
    def run_multi_objective_batch(self) -> bool:
        """Run multi-objective batch execution."""
        if self.args.skip_multi:
            self.log("Skipping multi-objective batch (--skip-multi)")
            return True
        
        cmd = [
            sys.executable,
            str(SCRIPTS_DIR / "run_multiobjective_batches.py"),
            "--runs", str(self.args.runs)
        ]
        
        return self.run_command(
            cmd,
            f"Multi-objective batch execution ({self.args.runs} runs)"
        )
    
    def generate_analysis(self) -> bool:
        """Generate comprehensive analysis summaries."""
        if self.args.skip_analysis:
            self.log("Skipping analysis generation (--skip-analysis)")
            return True
        
        # Single-objective analysis
        single_cmd = [sys.executable, str(SCRIPTS_DIR / "analyze_all_examples.py")]
        single_success = self.run_command(
            single_cmd,
            "Generate single-objective analysis summaries"
        )
        
        # Multi-objective analysis (if the script exists)
        multi_script = SCRIPTS_DIR / "analyze_all_multiobjective.py"
        if multi_script.exists():
            multi_cmd = [sys.executable, str(multi_script)]
            multi_success = self.run_command(
                multi_cmd,
                "Generate multi-objective analysis summaries"
            )
        else:
            self.log("Multi-objective analysis script not found, skipping", "WARNING")
            multi_success = True
        
        return single_success and multi_success
    
    def print_summary(self, overall_success: bool) -> None:
        """Print execution summary."""
        total_time = time.time() - self.start_time
        
        print("\n" + "="*60)
        print("🏁 EXECUTION SUMMARY")
        print("="*60)
        
        if overall_success:
            print("✅ All tasks completed successfully!")
        else:
            print("❌ Some tasks failed!")
        
        print(f"\n📊 Total execution time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
        
        if self.task_times:
            print(f"\n⏱️  Task breakdown:")
            for task, duration in self.task_times:
                status = "✅" if not task.endswith(("TIMEOUT", "ERROR")) else "❌"
                print(f"  {status} {task:<50} {duration:>8.1f}s")
        
        if not self.args.dry_run and overall_success:
            print(f"\n📁 Output locations:")
            print(f"  • Single-objective analysis: scripts/all_examples_summary.csv")
            print(f"  • Per-example summaries: examples/*/analysis_summary.json")
            print(f"  • Per-example runs: examples/*/analysis_runs.csv")
            
            # Count output directories
            examples_dir = REPO_ROOT / "examples"
            if examples_dir.exists():
                total_outputs = 0
                for example_dir in examples_dir.iterdir():
                    if example_dir.is_dir():
                        outputs = list(example_dir.glob("output_*"))
                        total_outputs += len(outputs)
                        if outputs:
                            print(f"  • {example_dir.name}: {len(outputs)} runs")
                print(f"\n📈 Total optimization runs generated: {total_outputs}")
        
        print("="*60)
    
    def run_all(self) -> bool:
        """Execute the complete analysis pipeline."""
        self.log("🚀 Starting comprehensive optimization framework analysis")
        self.log(f"Configuration: {self.args.runs} runs, {self.args.concurrency} cores")
        
        tasks = [
            ("Clean outputs", self.clean_outputs),
            ("Single-objective batch", self.run_single_objective_batch),
            ("Multi-objective batch", self.run_multi_objective_batch),
            ("Generate analysis", self.generate_analysis),
        ]
        
        success = True
        for task_name, task_func in tasks:
            self.log(f"📋 Task: {task_name}")
            if not task_func():
                self.log(f"❌ Task failed: {task_name}", "ERROR")
                success = False
                if not self.args.dry_run:
                    user_input = input("\nContinue with remaining tasks? (y/N): ")
                    if user_input.lower() not in ['y', 'yes']:
                        break
            else:
                self.log(f"✅ Task completed: {task_name}")
        
        self.print_summary(success)
        return success


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run comprehensive optimization framework analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard analysis (100 runs, 22 cores)
  uv run python scripts/run_complete_analysis.py
  
  # Quick test (10 runs, 4 cores)  
  uv run python scripts/run_complete_analysis.py --runs 10 --concurrency 4
  
  # Skip cleanup and only run analysis
  uv run python scripts/run_complete_analysis.py --skip-cleanup --skip-single --skip-multi
  
  # Dry run to see what would execute
  uv run python scripts/run_complete_analysis.py --dry-run
        """
    )
    
    parser.add_argument("--runs", "-r", type=int, default=100, 
                       help="Number of runs per example (default: 100)")
    parser.add_argument("--concurrency", "-c", type=int, default=22,
                       help="Number of concurrent processes for single-objective (default: 22)")
    parser.add_argument("--skip-cleanup", action="store_true",
                       help="Skip cleaning existing output directories")
    parser.add_argument("--skip-single", action="store_true", 
                       help="Skip single-objective batch execution")
    parser.add_argument("--skip-multi", action="store_true",
                       help="Skip multi-objective batch execution")
    parser.add_argument("--skip-analysis", action="store_true",
                       help="Skip analysis generation")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be executed without running")
    
    args = parser.parse_args()
    
    # Validation
    if args.runs < 1:
        print("Error: --runs must be >= 1", file=sys.stderr)
        return 1
    
    if args.concurrency < 1:
        print("Error: --concurrency must be >= 1", file=sys.stderr)
        return 1
    
    # Check if we're in the right directory
    if not (REPO_ROOT / "examples").exists():
        print(f"Error: examples directory not found. Please run from: {REPO_ROOT}", file=sys.stderr)
        return 1
    
    runner = TaskRunner(args)
    success = runner.run_all()
    
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())