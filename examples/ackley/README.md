# Ackley Function Optimization Example

This example demonstrates genetic algorithm optimization on the Ackley function, a highly multimodal benchmark problem.

## Problem Description

The Ackley function is defined as:

```
f(x) = -a * exp(-b * sqrt(1/n * sum(x_i^2))) - exp(1/n * sum(cos(c*x_i))) + a + e
```

Where:
- **Parameters**: a=20, b=0.2, c=2π
- **Dimensions**: 10D (configurable)
- **Domain**: x_i ∈ [-10, 10] (reduced from typical [-32.768, 32.768] for clearer visualization)
- **Global minimum**: x* = (0, 0, ..., 0) with f(x*) = 0
- **Characteristics**: Highly multimodal with many local minima

## Key Features

- **Dynamic slicing visualization**: For high-dimensional problems, shows 2D contour plot over selected parameter pair while fixing other dimensions to current best solution values
- **Population tracking**: Logs population data and generates plots showing evolution progress
- **Timestamped outputs**: Each run creates a unique output directory with timestamp
- **Frame saving**: Generates animation frames for visualization

## Running the Example

```bash
uv run examples/ackley/run_ga.py
```

## Output Files

- `analysis_runs.csv`: Statistical summary of multiple runs
- `analysis_summary.json`: Detailed analysis results
- `output_YYYYMMDD-HHMMSS/`: Timestamped run directory containing:
  - Population data CSV files
  - Visualization plots and animation frames
  - Final results

## Files

- `problem.py`: Ackley function implementation and problem class
- `plotting.py`: Visualization utilities with dynamic slicing
- `run_ga.py`: Main execution script
- `README.md`: This documentation