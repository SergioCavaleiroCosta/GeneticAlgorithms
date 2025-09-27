# Griewank Function Optimization Example

This example demonstrates genetic algorithm optimization on the Griewank function, a multimodal benchmark that becomes more challenging in higher dimensions.

## Problem Description

The Griewank function is defined as:

```
f(x) = sum(x_i²/4000) - prod(cos(x_i/sqrt(i))) + 1
```

Where:
- **Dimensions**: 10D (configurable)
- **Domain**: x_i ∈ [-10, 10] (reduced from typical [-600, 600] for clearer visualization)
- **Global minimum**: x* = (0, 0, ..., 0) with f(x*) = 0
- **Characteristics**: Many local minima, becomes more multimodal with increasing dimension

## Key Features

- **Scalable difficulty**: More challenging as dimensionality increases
- **Product term**: Cosine product creates intricate local minima structure
- **Dynamic slicing visualization**: Shows 2D slice through high-dimensional space
- **Dimension scalability**: Tests algorithm performance on higher dimensions

## Running the Example

```bash
uv run examples/griewank/run_ga.py
```

## Output Files

- `analysis_runs.csv`: Statistical summary of multiple runs
- `analysis_summary.json`: Detailed analysis results
- `output_YYYYMMDD-HHMMSS/`: Timestamped run directory containing:
  - Population data CSV files
  - 2D contour plots with dynamic slicing
  - Final optimization results

## Files

- `problem.py`: Griewank function implementation and problem class
- `plotting.py`: Dynamic slicing visualization utilities
- `run_ga.py`: Main execution script
- `README.md`: This documentation