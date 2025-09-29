"""Analysis functions for single-objective benchmark results.

This module provides analysis functionality for single-objective optimization results
stored in output_*/result.csv files created by the batch runners.
"""
from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = REPO_ROOT / "examples"


def load_run_result(output_dir: Path) -> Optional[Dict[str, Any]]:
    """Load a single run result from either result.csv or data/iter_*.csv files."""
    # First try the new result.csv format
    result_path = output_dir / "result.csv"
    if result_path.exists():
        try:
            with result_path.open() as f:
                reader = csv.DictReader(f)
                row = next(reader)  # Should be only one row
                return {
                    'execution_time': float(row['execution_time']),
                    'iterations': int(row['iterations']),
                    'population_size': int(row['population_size']),
                    'best_objective': float(row['best_objective']),
                    'evaluations': int(row['evaluations']),
                    'problem_dimension': None  # Not available in result.csv format
                }
        except Exception as e:
            print(f"Warning: Could not load {result_path}: {e}")
    
    # Fall back to parsing data/iter_*.csv files
    data_dir = output_dir / "data"
    if not data_dir.exists():
        return None
    
    try:
        # Find the highest iteration file
        iter_files = [f for f in data_dir.iterdir() if f.name.startswith("iter_") and f.name.endswith(".csv")]
        if not iter_files:
            return None
        
        # Sort by iteration number
        iter_files.sort(key=lambda f: int(f.stem.split("_")[1]))
        final_iter_file = iter_files[-1]
        
        # Parse the final iteration file
        with final_iter_file.open() as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            
            if not rows:
                return None
            
            # Extract metrics
            final_iteration = int(rows[0]['iteration'])
            population_size = len(rows)
            best_objective = min(float(row['objective']) for row in rows)
            
            # Determine problem dimension from solution vector
            problem_dimension = None
            first_row = rows[0]
            # Count columns that contain coordinate values (not iteration, candidate_index, objective)
            exclude_cols = {'iteration', 'candidate_index', 'objective'}
            solution_cols = [col for col in first_row.keys() if col not in exclude_cols]
            
            # For normalized solutions, count x_n, y_n, z_n, etc. columns
            normalized_cols = [col for col in solution_cols if col.endswith('_n')]
            if normalized_cols:
                problem_dimension = len(normalized_cols)
            else:
                # For real solutions, count x, y, z, etc. columns (excluding any other metadata)
                coordinate_cols = [col for col in solution_cols if len(col) <= 2 and col.isalpha()]
                if coordinate_cols:
                    problem_dimension = len(coordinate_cols)
                else:
                    # Fallback: assume all remaining columns are coordinates
                    problem_dimension = len(solution_cols)
            
            # Rough estimates for missing data
            estimated_evaluations = (final_iteration + 1) * population_size
            estimated_execution_time = 60.0  # Default estimate
            
            # Extract best solution parameters from the final iteration
            best_solution = None
            if rows:
                # Find the row with the best (minimum) objective
                best_row = min(rows, key=lambda row: float(row['objective']))
                
                # Extract parameter columns (x0, x1, x2, ... or x, y, z, ...)
                param_cols = []
                
                # First try numbered parameters (x0, x1, x2, ...)
                numbered_cols = [col for col in best_row.keys() if col.startswith('x') and len(col) > 1 and col[1:].isdigit()]
                if numbered_cols:
                    param_cols = sorted(numbered_cols, key=lambda x: int(x[1:]))
                else:
                    # Try single-letter parameters (x, y, z, ...)
                    single_cols = [col for col in best_row.keys() if len(col) == 1 and col.isalpha()]
                    if single_cols:
                        param_cols = sorted(single_cols)
                
                if param_cols:
                    best_solution = [float(best_row[col]) for col in param_cols]
            
            return {
                'execution_time': estimated_execution_time,
                'iterations': final_iteration + 1,  # Convert 0-based to count
                'population_size': population_size,
                'best_objective': best_objective,
                'evaluations': estimated_evaluations,
                'problem_dimension': problem_dimension,
                'best_solution': best_solution
            }
            
    except Exception as e:
        print(f"Warning: Could not parse data files in {output_dir}: {e}")
        return None


def analyze_runs(example_name: str, tol: float = 1e-6, limit: int = 0) -> Tuple[List[Dict[str, float]], List[Path]]:
    """Load and analyze all runs for a given example.
    
    Returns:
        (results, run_dirs): List of result dictionaries and corresponding run directory paths
    """
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
        result = load_run_result(output_dir)
        if result is not None:
            results.append(result)
            valid_dirs.append(output_dir)
    
    if not results:
        raise ValueError(f"No valid results found for {example_name}")
    
    return results, valid_dirs


def compute_statistics(values: List[float], quantiles: str = "0.25,0.5,0.75,0.95") -> Dict[str, float]:
    """Compute statistical metrics for a list of values."""
    if not values:
        return {}
    
    stats = {
        'min': min(values),
        'max': max(values),
        'mean': statistics.mean(values),
        'median': statistics.median(values),
        'stdev': statistics.stdev(values) if len(values) > 1 else 0.0
    }
    
    # Add quantiles
    quantile_vals = [float(q) for q in quantiles.split(",")]
    for q in quantile_vals:
        if 0 <= q <= 1:
            stats[f'q{int(q*100)}'] = statistics.quantiles(values, n=100)[int(q*100)-1] if len(values) > 1 else values[0]
    
    return stats


def run_analysis(example_name: str, tol: float = 1e-6, limit: int = 0, quantiles: str = "0.25,0.5,0.75,0.95") -> Tuple[Dict[str, Any], List[Tuple[str, float, int, Optional[int]]]]:
    """Run complete analysis for an example.
    
    Returns:
        (summary_dict, per_run_list): Summary statistics and per-run details
    """
    results, run_dirs = analyze_runs(example_name, tol, limit)
    
    # Extract metrics
    best_objectives = [r['best_objective'] for r in results]
    execution_times = [r['execution_time'] for r in results]
    iterations = [r['iterations'] for r in results]
    evaluations = [r['evaluations'] for r in results]
    population_sizes = [r['population_size'] for r in results]
    
    # Extract parameter vectors (best solutions)
    parameter_vectors = []
    for r in results:
        if 'best_solution' in r and r['best_solution'] is not None:
            parameter_vectors.append(r['best_solution'])
    
    # Extract problem dimension (should be consistent across runs)
    dimensions = [r.get('problem_dimension') for r in results if r.get('problem_dimension') is not None]
    if dimensions:
        problem_dimension = dimensions[0]
    else:
        # Fallback: known dimensions for benchmark problems
        known_dimensions = {
            # 2D problems
            'beale': 2, 'booth': 2, 'bukin6': 2, 'cross_in_tray': 2, 'easom': 2,
            'goldstein_price': 2, 'himmelblau': 2, 'holder_table': 2, 'levi13': 2,
            'matyas': 2, 'mccormick': 2, 'schaffer_n2': 2, 'schaffer_n4': 2,
            'three_hump_camel': 2,
            # 10D problems  
            'ackley': 10, 'griewank': 10, 'rastrigin': 10, 'rosenbrock': 10,
            'sphere': 10, 'styblinski_tang': 10,
            # Other
            'eggholder': 2,  # Actually 2D despite being complex
        }
        problem_dimension = known_dimensions.get(example_name, 'unknown')
    
    # Compute statistics
    summary = {
        'example': example_name,
        'total_runs': len(results),
        'problem_dimension': problem_dimension,
        
        # Nested statistics for better organization
        'best_objective': compute_statistics(best_objectives, quantiles),
        'execution_time': compute_statistics(execution_times, quantiles),
        'iterations': compute_statistics(iterations, quantiles),  
        'evaluations': compute_statistics(evaluations, quantiles),
        
        # Configuration
        'population_size': population_sizes[0] if population_sizes else None,
        
        # Success analysis (runs that achieved tolerance)
        'success': {
            'rate': 0.0,
            'runs': 0,
            'tolerance': tol
        }
    }
    
    # Analyze parameters if available
    if parameter_vectors and problem_dimension != 'unknown':
        try:
            # Convert to numpy array for easier analysis
            import numpy as np
            param_array = np.array(parameter_vectors)
            
            # Calculate statistics for each parameter dimension
            parameters_stats = {}
            for dim in range(int(problem_dimension)):
                dim_values = param_array[:, dim]
                parameters_stats[f'x{dim+1}'] = compute_statistics(dim_values.tolist(), quantiles)
            
            summary['parameters'] = parameters_stats
        except Exception as e:
            print(f"Warning: Could not analyze parameters: {e}")
            summary['parameters'] = 'not_available'
    else:
        summary['parameters'] = 'not_available'
    
    # Success analysis (runs that achieved tolerance)
    min_known_optimum = 0.0  # Most test functions have global optimum at 0
    successful_runs = [i for i, obj in enumerate(best_objectives) if abs(obj - min_known_optimum) <= tol]
    summary['success']['rate'] = len(successful_runs) / len(results)
    summary['success']['runs'] = len(successful_runs)
    
    # Per-run details for CSV output
    per_run = []
    for i, (result, run_dir) in enumerate(zip(results, run_dirs)):
        is_success = abs(result['best_objective'] - min_known_optimum) <= tol
        first_success_iter = result['iterations'] if is_success else None
        
        per_run.append((
            run_dir.name,
            result['best_objective'],
            result['iterations'], 
            first_success_iter
        ))
    
    return summary, per_run


if __name__ == "__main__":
    # Test the analysis on a single example
    import sys
    if len(sys.argv) > 1:
        example = sys.argv[1]
        try:
            summary, per_run = run_analysis(example)
            print(f"Analysis for {example}:")
            print(json.dumps(summary, indent=2))
            print(f"Per-run data: {len(per_run)} runs")
        except Exception as e:
            print(f"Error: {e}")
    else:
        print("Usage: python analyze_example.py <example_name>")