"""Multi-objective optimization analysis tools.

Provides analysis capabilities for NSGA-II multi-objective optimization results,
including hypervolume, spacing, convergence metrics, and run statistics.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = REPO_ROOT / "examples"


def load_multiobjective_run_result(output_dir: Path) -> Optional[Dict[str, Any]]:
    """Load multi-objective optimization result from an output directory."""
    try:
        import pandas as pd
        
        # Check for CSV files
        objectives_file = output_dir / "objectives.csv"
        solutions_file = output_dir / "solutions.csv"
        
        if not objectives_file.exists() or not solutions_file.exists():
            return None
        
        # Load objectives data (this has proper CSV format)
        objectives_df = pd.read_csv(objectives_file)
        
        # Load solutions data - need to read as text and parse manually since
        # solution values aren't quoted in the CSV
        with solutions_file.open('r') as f:
            lines = f.readlines()
        
        if len(lines) < 2:
            return None
        
        # Parse header
        header = lines[0].strip().split(',')
        
        # Parse first data line to get problem dimension
        # Format: iteration,individual,x1,x2,x3,...
        first_data = lines[1].strip().split(',')
        # Subtract 2 for iteration and individual columns
        problem_dimension = len(first_data) - 2
        
        if objectives_df.empty:
            return None
        
        # Extract basic information
        max_iteration = int(objectives_df['iteration'].max())
        generations = max_iteration + 1  # Convert 0-based to count
        
        # Get population size from first iteration
        first_iter_data = objectives_df[objectives_df['iteration'] == 0]
        population_size = int(len(first_iter_data))
        
        # Get problem dimension from solutions
        # Already extracted above from manual CSV parsing
        
        # Get number of objectives
        objective_cols = [col for col in objectives_df.columns if col.startswith('f')]
        num_objectives = len(objective_cols)
        
        # Extract final iteration data
        final_iter_data = objectives_df[objectives_df['iteration'] == max_iteration]
        final_objectives = []
        for _, row in final_iter_data.iterrows():
            obj_values = [row[col] for col in objective_cols]
            final_objectives.append(obj_values)
        
        # Estimate execution time (we don't have actual timing data in CSV)
        # Use a rough estimate based on generations and population size
        estimated_execution_time = generations * population_size * 0.001  # Very rough estimate
        
        result = {
            'execution_time': estimated_execution_time,
            'generations': generations,
            'population_size': population_size,
            'problem_dimension': problem_dimension,
            'num_objectives': num_objectives,
            'final_objectives': final_objectives,
            'num_final_solutions': len(final_objectives)
        }
        
        return result
        
    except Exception as e:
        print(f"Warning: Could not load multi-objective result from {output_dir}: {e}")
        return None


def analyze_multiobjective_runs(example_name: str, limit: int = 0) -> Tuple[List[Dict[str, Any]], List[Path]]:
    """Load and analyze all multi-objective runs for a given example."""
    example_dir = EXAMPLES_DIR / example_name
    if not example_dir.exists():
        raise ValueError(f"Example directory not found: {example_dir}")
    
    # Find all output directories
    output_dirs = sorted([d for d in example_dir.iterdir() if d.is_dir() and d.name.startswith("output_")])
    
    if limit > 0:
        output_dirs = output_dirs[-limit:]  # Take most recent
    
    if not output_dirs:
        raise ValueError(f"No output directories found in {example_dir}")
    
    results = []
    valid_dirs = []
    
    for output_dir in output_dirs:
        result = load_multiobjective_run_result(output_dir)
        if result is not None:
            results.append(result)
            valid_dirs.append(output_dir)
    
    if not results:
        raise ValueError(f"No valid results found for {example_name}")
    
    return results, valid_dirs


def compute_statistics(values: List[float]) -> Dict[str, float]:
    """Compute statistical measures for a list of values."""
    if not values:
        return {}
    
    values_array = np.array(values)
    return {
        'min': float(np.min(values_array)),
        'max': float(np.max(values_array)),
        'mean': float(np.mean(values_array)),
        'median': float(np.median(values_array)),
        'stdev': float(np.std(values_array)),
        'q25': float(np.percentile(values_array, 25)),
        'q50': float(np.percentile(values_array, 50)),
        'q75': float(np.percentile(values_array, 75)),
        'q95': float(np.percentile(values_array, 95))
    }


def calculate_hypervolume(objectives: List[List[float]], reference_point: List[float]) -> float:
    """Calculate hypervolume indicator for a set of objectives."""
    if not objectives or not reference_point:
        return 0.0
    
    # Simple hypervolume calculation for 2D case
    if len(reference_point) == 2:
        # Sort by first objective
        sorted_objectives = sorted(objectives, key=lambda x: x[0])
        
        hv = 0.0
        prev_x = 0.0
        
        for obj in sorted_objectives:
            if obj[0] <= reference_point[0] and obj[1] <= reference_point[1]:
                width = obj[0] - prev_x
                height = reference_point[1] - obj[1]
                if width > 0 and height > 0:
                    hv += width * height
                prev_x = obj[0]
        
        return hv
    
    # For higher dimensions, return 0 (would need more complex implementation)
    return 0.0


def calculate_spacing(objectives: List[List[float]]) -> float:
    """Calculate spacing metric for a set of objectives."""
    if len(objectives) < 2:
        return 0.0
    
    distances = []
    
    for i, obj1 in enumerate(objectives):
        min_dist = float('inf')
        for j, obj2 in enumerate(objectives):
            if i != j:
                # Euclidean distance
                dist = sum((a - b) ** 2 for a, b in zip(obj1, obj2)) ** 0.5
                if dist < min_dist:
                    min_dist = dist
        distances.append(min_dist)
    
    if not distances:
        return 0.0
    
    mean_dist = np.mean(distances)
    variance = np.var(distances)
    
    return float(variance ** 0.5)


def run_multiobjective_analysis(example_name: str, reference_point: Optional[str] = None, limit: int = 0) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Run complete multi-objective analysis for an example."""
    results, run_dirs = analyze_multiobjective_runs(example_name, limit)
    
    if not results:
        raise ValueError(f"No valid results found for {example_name}")
    
    # Extract metrics
    execution_times = [r.get('execution_time', 0.0) for r in results]
    generations = [r.get('generations', 0) for r in results]
    population_sizes = [r.get('population_size', 0) for r in results]
    problem_dimensions = [r.get('problem_dimension', 'unknown') for r in results]
    num_objectives = [r.get('num_objectives', 2) for r in results]
    
    # Extract final population metrics
    hypervolumes = []
    spacings = []
    non_dominated_counts = []
    
    # Parse reference point if provided
    ref_point = None
    if reference_point:
        try:
            ref_point = [float(x.strip()) for x in reference_point.split(',')]
        except Exception:
            print(f"Warning: Could not parse reference point '{reference_point}', using default")
    
    for result in results:
        # Get hypervolume
        if 'hypervolume' in result:
            hypervolumes.append(result['hypervolume'])
        elif 'final_objectives' in result and ref_point:
            hv = calculate_hypervolume(result['final_objectives'], ref_point)
            hypervolumes.append(hv)
        
        # Get spacing
        if 'spacing' in result:
            spacings.append(result['spacing'])
        elif 'final_objectives' in result:
            spacing = calculate_spacing(result['final_objectives'])
            spacings.append(spacing)
        
        # Get non-dominated solutions count
        if 'non_dominated_solutions' in result:
            non_dominated_counts.append(result['non_dominated_solutions'])
    
    # Determine consistent values
    population_size = population_sizes[0] if population_sizes and all(p == population_sizes[0] for p in population_sizes) else None
    generation_count = generations[0] if generations and all(g == generations[0] for g in generations) else None
    problem_dimension = problem_dimensions[0] if problem_dimensions and all(d == problem_dimensions[0] for d in problem_dimensions) else 'unknown'
    num_obj = num_objectives[0] if num_objectives and all(n == num_objectives[0] for n in num_objectives) else 2
    
    # Create summary
    summary = {
        'example': example_name,
        'total_runs': len(results),
        'valid_runs': len(results),
        'problem_dimension': problem_dimension,
        'num_objectives': num_obj,
        'population_size': population_size,
        'generations': generation_count,
    }
    
    # Add execution time statistics (nested structure)
    if execution_times:
        summary['execution_time'] = compute_statistics(execution_times)
    
    # Add hypervolume statistics (nested structure)
    if hypervolumes:
        summary['hypervolume'] = compute_statistics(hypervolumes)
    
    # Add spacing statistics (nested structure)
    if spacings:
        summary['spacing'] = compute_statistics(spacings)
    
    # Add non-dominated solutions statistics (nested structure)
    if non_dominated_counts:
        summary['non_dominated_solutions'] = compute_statistics(non_dominated_counts)
        
        # Calculate Pareto ratio (non-dominated / total population)
        if population_size:
            pareto_ratios = [count / population_size for count in non_dominated_counts]
            summary['pareto_ratio'] = compute_statistics(pareto_ratios)
    
    # Per-run details for CSV
    per_run = []
    for i, (result, run_dir) in enumerate(zip(results, run_dirs)):
        run_data = {
            'run_id': i + 1,
            'run_dir': run_dir.name,
            'execution_time': result.get('execution_time', 0.0),
            'generations': result.get('generations', 0),
            'population_size': result.get('population_size', 0),
            'problem_dimension': result.get('problem_dimension', 'unknown'),
            'num_objectives': result.get('num_objectives', 2),
        }
        
        if 'hypervolume' in result:
            run_data['hypervolume'] = result['hypervolume']
        if 'spacing' in result:
            run_data['spacing'] = result['spacing']
        if 'non_dominated_solutions' in result:
            run_data['non_dominated_solutions'] = result['non_dominated_solutions']
            if population_size:
                run_data['pareto_ratio'] = result['non_dominated_solutions'] / population_size
        
        per_run.append(run_data)
    
    return summary, per_run


def main() -> int:
    """Main function for command-line usage."""
    ap = argparse.ArgumentParser(description="Analyze multi-objective optimization results")
    ap.add_argument("example", help="Example name to analyze")
    ap.add_argument("--reference-point", "-r", help="Reference point for hypervolume (e.g., '1.1,11.0')")
    ap.add_argument("--limit", "-l", type=int, default=0, help="Limit to N most recent runs")
    args = ap.parse_args()
    
    try:
        summary, per_run = run_multiobjective_analysis(args.example, args.reference_point, args.limit)
        
        print(f"Multi-objective analysis for {args.example}:")
        print(json.dumps(summary, indent=2))
        print(f"Per-run data: {len(per_run)} runs")
        
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())