# Matyas Function Optimization Example

This example demonstrates genetic algorithm optimization on the Matyas function, a simple 2D benchmark with a single global minimum.

## Problem Description

The Matyas function is defined as:

```
f(x, y) = 0.26*(x² + y²) - 0.48*x*y
```

Where:
- **Dimensions**: 2D
- **Domain**: x,y ∈ [-10, 10]
- **Global minimum**: (x*, y*) = (0, 0) with f(x*, y*) = 0
- **Characteristics**: Convex function with elliptical contours, single global minimum

## Key Features

- **Simple convex landscape**: No local minima, ideal for testing convergence
- **Elliptical contours**: Non-circular level sets due to cross term
- **Fast convergence**: Algorithm should quickly find the global minimum
- **Analytical properties**: Well-understood mathematical structure

## Running the Example

```bash
uv run examples/matyas/run_ga.py
```

## Output Files

- `analysis_runs.csv`: Statistical summary of multiple runs
- `analysis_summary.json`: Detailed analysis results
- `output_YYYYMMDD-HHMMSS/`: Timestamped run directory containing:
  - Population data CSV files
  - 2D contour plots showing elliptical landscape
  - Final optimization results

## Files

- `problem.py`: Matyas function implementation and problem class
- `plotting.py`: 2D contour visualization utilities
- `run_ga.py`: Main execution script
- `README.md`: This documentation