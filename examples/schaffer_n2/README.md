# Schaffer N.2 Function Optimization Example

This example demonstrates genetic algorithm optimization on the Schaffer N.2 function, a 2D benchmark with circular symmetry and multiple local minima.

## Problem Description

The Schaffer N.2 function is defined as:

```
f(x, y) = 0.5 + (sin²(x² - y²) - 0.5) / (1 + 0.001*(x² + y²))²
```

Where:
- **Dimensions**: 2D
- **Domain**: x,y ∈ [-100, 100]
- **Global minimum**: (x*, y*) = (0, 0) with f(x*, y*) = 0
- **Characteristics**: Circular symmetry with oscillatory behavior, multiple concentric rings of local minima

## Key Features

- **Circular symmetry**: Radially symmetric landscape
- **Oscillatory structure**: Sine function creates ripple patterns
- **Multiple local minima**: Concentric rings of local optima
- **Challenging convergence**: Fine-grained structure requires careful balance of exploration/exploitation

## Running the Example

```bash
uv run examples/schaffer_n2/run_ga.py
```

## Output Files

- `analysis_runs.csv`: Statistical summary of multiple runs
- `analysis_summary.json`: Detailed analysis results
- `output_YYYYMMDD-HHMMSS/`: Timestamped run directory containing:
  - Population data CSV files
  - 2D contour plots showing circular symmetry
  - Final optimization results

## Files

- `problem.py`: Schaffer N.2 function implementation and problem class
- `plotting.py`: 2D contour visualization utilities
- `run_ga.py`: Main execution script
- `README.md`: This documentation