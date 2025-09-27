# Schaffer N.4 Function Optimization Example

This example demonstrates genetic algorithm optimization on the Schaffer N.4 function, a variant of the Schaffer function with different trigonometric components.

## Problem Description

The Schaffer N.4 function is defined as:

```
f(x, y) = 0.5 + (cos²(sin(|x² - y²|)) - 0.5) / (1 + 0.001*(x² + y²))²
```

Where:
- **Dimensions**: 2D
- **Domain**: x,y ∈ [-100, 100]
- **Global minimum**: (x*, y*) = (0, 1.25313) with f(x*, y*) ≈ 0.292579
- **Characteristics**: Asymmetric landscape with complex trigonometric structure

## Key Features

- **Complex trigonometry**: Nested trigonometric functions create intricate patterns
- **Asymmetric global minimum**: Unlike Schaffer N.2, the optimum is not at origin
- **Fine-grained structure**: Multiple scales of variation
- **Challenging landscape**: Requires sophisticated search strategies

## Running the Example

```bash
uv run examples/schaffer_n4/run_ga.py
```

## Output Files

- `analysis_runs.csv`: Statistical summary of multiple runs
- `analysis_summary.json`: Detailed analysis results
- `output_YYYYMMDD-HHMMSS/`: Timestamped run directory containing:
  - Population data CSV files
  - 2D contour plots showing complex structure
  - Final optimization results

## Files

- `problem.py`: Schaffer N.4 function implementation and problem class
- `plotting.py`: 2D contour visualization utilities
- `run_ga.py`: Main execution script
- `README.md`: This documentation