# Rastrigin Function Optimization Example

This example demonstrates genetic algorithm optimization on the Rastrigin function, a highly multimodal benchmark with many local minima arranged in a regular pattern.

## Problem Description

The Rastrigin function is defined as:

```
f(x) = A*n + sum(x_i² - A*cos(2π*x_i))
```

Where:
- **Parameters**: A = 10
- **Dimensions**: 10D (configurable)
- **Domain**: x_i ∈ [-5.12, 5.12]
- **Global minimum**: x* = (0, 0, ..., 0) with f(x*) = 0
- **Characteristics**: Highly multimodal with approximately 10^n local minima

## Key Features

- **Extreme multimodality**: Thousands of local minima in regular grid pattern
- **Cosine modulation**: Creates wave-like perturbations over quadratic base
- **Dynamic slicing visualization**: Shows 2D slice through high-dimensional space
- **Classic benchmark**: One of the most famous multimodal test functions

## Running the Example

```bash
uv run examples/rastrigin/run_ga.py
```

## Output Files

- `analysis_runs.csv`: Statistical summary of multiple runs
- `analysis_summary.json`: Detailed analysis results
- `output_YYYYMMDD-HHMMSS/`: Timestamped run directory containing:
  - Population data CSV files
  - 2D contour plots showing the wave pattern
  - Final optimization results

## Files

- `problem.py`: Rastrigin function implementation and problem class
- `plotting.py`: Dynamic slicing visualization utilities
- `run_ga.py`: Main execution script
- `README.md`: This documentation