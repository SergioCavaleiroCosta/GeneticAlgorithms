# Styblinski-Tang Function Optimization Example

This example demonstrates genetic algorithm optimization on the Styblinski-Tang function, a multimodal benchmark with both polynomial and trigonometric components.

## Problem Description

The Styblinski-Tang function is defined as:

```
f(x) = 0.5 * sum(x_i^4 - 16*x_i² + 5*x_i)
```

Where:
- **Dimensions**: 10D (configurable)
- **Domain**: x_i ∈ [-5, 5]
- **Global minimum**: x* = (-2.903534, -2.903534, ..., -2.903534) with f(x*) ≈ -39.16616*n
- **Characteristics**: Multimodal with a dominant global minimum basin

## Key Features

- **Polynomial landscape**: Fourth-order polynomial creates interesting topology
- **Dominant global basin**: Strong global attractor with some local minima
- **Asymmetric optimum**: Global minimum not at origin
- **Moderate multimodality**: Some local minima but not extremely challenging
- **Dynamic slicing visualization**: Shows 2D slice through high-dimensional space

## Running the Example

```bash
uv run examples/styblinski_tang/run_ga.py
```

## Output Files

- `analysis_runs.csv`: Statistical summary of multiple runs
- `analysis_summary.json`: Detailed analysis results
- `output_YYYYMMDD-HHMMSS/`: Timestamped run directory containing:
  - Population data CSV files
  - 2D contour plots showing the polynomial landscape
  - Final optimization results

## Files

- `problem.py`: Styblinski-Tang function implementation and problem class
- `plotting.py`: Dynamic slicing visualization utilities
- `run_ga.py`: Main execution script
- `README.md`: This documentation