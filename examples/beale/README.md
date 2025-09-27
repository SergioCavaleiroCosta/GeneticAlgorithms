# Beale Function Optimization Example

This example demonstrates genetic algorithm optimization on the Beale function, a classic 2D optimization benchmark.

## Problem Description

The Beale function is defined as:

```
f(x,y) = (1.5 - x + xy)² + (2.25 - x + xy²)² + (2.625 - x + xy³)²
```

Where:
- **Dimensions**: 2D
- **Domain**: x,y ∈ [-4.5, 4.5]
- **Global minimum**: (x*, y*) = (3, 0.5) with f(x*, y*) = 0
- **Characteristics**: Unimodal with a narrow curved valley leading to the minimum

## Key Features

- **2D contour visualization**: Direct plotting without dynamic slicing since problem is naturally 2D
- **Population evolution tracking**: Shows how population converges toward the global minimum
- **Valley navigation**: Demonstrates algorithm's ability to follow narrow curved valleys
- **Interactive plotting option**: Can enable interactive matplotlib plots

## Running the Example

```bash
uv run examples/beale/run_ga.py
```

## Output Files

- `analysis_runs.csv`: Statistical summary of multiple runs
- `analysis_summary.json`: Detailed analysis results
- `output_YYYYMMDD-HHMMSS/`: Timestamped run directory containing:
  - Population data CSV files
  - 2D contour plots showing population evolution
  - Final optimization results

## Files

- `problem.py`: Beale function implementation and problem class
- `plotting.py`: 2D contour visualization utilities
- `run_ga.py`: Main execution script
- `README.md`: This documentation