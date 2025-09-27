# Rosenbrock Function Optimization Example

This example demonstrates genetic algorithm optimization on the Rosenbrock function, also known as "Rosenbrock's Valley" or "Banana function" - a classic benchmark with a narrow curved valley.

## Problem Description

The Rosenbrock function is defined as:

```
f(x) = sum(100*(x_{i+1} - x_i²)² + (1 - x_i)²)
```

Where:
- **Dimensions**: 10D (configurable)
- **Domain**: x_i ∈ [-2.048, 2.048]
- **Global minimum**: x* = (1, 1, ..., 1) with f(x*) = 0
- **Characteristics**: Narrow curved valley leading to global minimum, challenging for many algorithms

## Key Features

- **Banana-shaped valley**: Distinctive curved valley that's difficult to follow
- **Scale separation**: Large condition number makes optimization challenging
- **Dynamic slicing visualization**: Shows 2D slice revealing valley structure
- **Classic benchmark**: One of the most famous optimization test functions

## Running the Example

```bash
uv run examples/rosenbrock/run_ga.py
```

## Output Files

- `analysis_runs.csv`: Statistical summary of multiple runs
- `analysis_summary.json`: Detailed analysis results
- `output_YYYYMMDD-HHMMSS/`: Timestamped run directory containing:
  - Population data CSV files
  - 2D contour plots showing the banana-shaped valley
  - Final optimization results

## Files

- `problem.py`: Rosenbrock function implementation and problem class
- `plotting.py`: Dynamic slicing visualization utilities
- `run_ga.py`: Main execution script
- `README.md`: This documentation