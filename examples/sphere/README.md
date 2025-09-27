# Sphere Function Optimization Example

This example demonstrates genetic algorithm optimization on the Sphere function, the simplest benchmark function used to test basic convergence properties.

## Problem Description

The Sphere function is defined as:

```
f(x) = sum(x_i²)
```

Where:
- **Dimensions**: 10D (configurable)
- **Domain**: x_i ∈ [-5.12, 5.12]
- **Global minimum**: x* = (0, 0, ..., 0) with f(x*) = 0
- **Characteristics**: Convex, unimodal, separable - the simplest optimization landscape

## Key Features

- **Simplest benchmark**: No local minima, perfectly convex
- **Separable**: Each dimension can be optimized independently
- **Fast convergence**: Any reasonable algorithm should solve this quickly
- **Baseline test**: Used to verify basic algorithm functionality
- **Dynamic slicing visualization**: Shows 2D slice through high-dimensional quadratic bowl

## Running the Example

```bash
uv run examples/sphere/run_ga.py
```

## Output Files

- `analysis_runs.csv`: Statistical summary of multiple runs
- `analysis_summary.json`: Detailed analysis results
- `output_YYYYMMDD-HHMMSS/`: Timestamped run directory containing:
  - Population data CSV files
  - 2D contour plots showing the quadratic bowl
  - Final optimization results

## Files

- `problem.py`: Sphere function implementation and problem class
- `plotting.py`: Dynamic slicing visualization utilities
- `run_ga.py`: Main execution script
- `README.md`: This documentation