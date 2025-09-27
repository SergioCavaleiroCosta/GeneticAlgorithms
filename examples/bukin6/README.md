# Bukin N.6 Function Optimization Example

This example demonstrates genetic algorithm optimization on the Bukin N.6 function, a challenging 2D benchmark with an unusual domain and narrow global minimum.

## Problem Description

The Bukin N.6 function is defined as:

```
f(x, y) = 100 * sqrt(|y - 0.01*x²|) + 0.01 * |x + 10|
```

Where:
- **Dimensions**: 2D
- **Domain**: x ∈ [-15, -5], y ∈ [-3, 3] (unusual asymmetric domain)
- **Global minimum**: (x*, y*) = (-10, 1) with f(x*, y*) = 0
- **Characteristics**: Narrow valley, non-differentiable, challenging landscape

## Key Features

- **Unusual domain**: Demonstrates handling of asymmetric search spaces
- **Narrow global minimum**: Tests algorithm's precision and local search capability
- **Non-smooth function**: Contains absolute values making it non-differentiable
- **Challenging convergence**: Requires careful balance of exploration and exploitation

## Running the Example

```bash
uv run examples/bukin6/run_ga.py
```

## Output Files

- `analysis_runs.csv`: Statistical summary of multiple runs
- `analysis_summary.json`: Detailed analysis results
- `output_YYYYMMDD-HHMMSS/`: Timestamped run directory containing:
  - Population data CSV files
  - 2D contour plots showing the challenging landscape
  - Final optimization results

## Files

- `problem.py`: Bukin N.6 function implementation and problem class
- `plotting.py`: 2D contour visualization utilities
- `run_ga.py`: Main execution script
- `README.md`: This documentation