# Levi N.13 Function Optimization Example

This example demonstrates genetic algorithm optimization on the Levi N.13 function, a multimodal 2D benchmark with ripple-like patterns.

## Problem Description

The Levi N.13 function is defined as:

```
f(x, y) = sin²(3πx) + (x-1)²*(1 + sin²(3πy)) + (y-1)²*(1 + sin²(2πy))
```

Where:
- **Dimensions**: 2D
- **Domain**: x,y ∈ [-10, 10]
- **Global minimum**: (x*, y*) = (1, 1) with f(x*, y*) = 0
- **Characteristics**: Ripple-like landscape with multiple local minima

## Key Features

- **Ripple patterns**: Trigonometric terms create wave-like structure
- **Multiple local minima**: Many local optima due to oscillatory nature
- **Smooth gradients**: Continuous and differentiable everywhere
- **Challenging convergence**: Algorithm must navigate through ripples to reach global minimum

## Running the Example

```bash
uv run examples/levi13/run_ga.py
```

## Output Files

- `analysis_runs.csv`: Statistical summary of multiple runs
- `analysis_summary.json`: Detailed analysis results
- `output_YYYYMMDD-HHMMSS/`: Timestamped run directory containing:
  - Population data CSV files
  - 2D contour plots showing the ripple landscape
  - Final optimization results

## Files

- `problem.py`: Levi N.13 function implementation and problem class
- `plotting.py`: 2D contour visualization utilities
- `run_ga.py`: Main execution script
- `README.md`: This documentation