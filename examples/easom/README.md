# Easom Function Optimization Example

This example demonstrates genetic algorithm optimization on the Easom function, a deceptive 2D benchmark with a very narrow global minimum in a nearly flat landscape.

## Problem Description

The Easom function is defined as:

```
f(x, y) = -cos(x) * cos(y) * exp(-((x-π)² + (y-π)²))
```

Where:
- **Dimensions**: 2D
- **Domain**: x,y ∈ [-10, 10] (reduced from typical [-100, 100] for visualization)
- **Global minimum**: (x*, y*) = (π, π) with f(x*, y*) = -1
- **Characteristics**: Nearly flat with a very narrow global minimum, highly deceptive

## Key Features

- **Deceptive landscape**: Most of the search space provides little gradient information
- **Needle in haystack**: Global minimum is extremely narrow compared to search space
- **Exploration challenge**: Tests algorithm's ability to explore vast flat regions
- **Precision requirement**: Once near minimum, requires fine-tuned exploitation

## Running the Example

```bash
uv run examples/easom/run_ga.py
```

## Output Files

- `analysis_runs.csv`: Statistical summary of multiple runs
- `analysis_summary.json`: Detailed analysis results
- `output_YYYYMMDD-HHMMSS/`: Timestamped run directory containing:
  - Population data CSV files
  - 2D contour plots showing the deceptive landscape
  - Final optimization results

## Files

- `problem.py`: Easom function implementation and problem class
- `plotting.py`: 2D contour visualization utilities
- `run_ga.py`: Main execution script
- `README.md`: This documentation