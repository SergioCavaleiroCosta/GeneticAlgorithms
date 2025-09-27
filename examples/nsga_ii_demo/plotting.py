"""Plotting utilities for NSGA-II multi-objective optimization visualization."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence
import matplotlib.pyplot as plt
import numpy as np

from .problems import true_pareto_front_zdt1


def plot_pareto_front(objectives: list[Sequence[float]], output_dir: Path, iteration: int = -1) -> None:
    """Plot the current population and true Pareto front."""
    # Extract objective values
    f1_vals = [obj[0] for obj in objectives]
    f2_vals = [obj[1] for obj in objectives]
    
    # Create plot
    plt.figure(figsize=(10, 8))
    plt.scatter(f1_vals, f2_vals, alpha=0.6, s=30, label=f'NSGA-II Population (Gen {iteration})')
    
    # Plot true Pareto front
    true_front = true_pareto_front_zdt1(100)
    plt.plot(true_front[:, 0], true_front[:, 1], 'r-', linewidth=2, label='True Pareto Front')
    
    plt.xlabel('f1')
    plt.ylabel('f2')
    plt.title(f'NSGA-II on ZDT1 Problem - Generation {iteration}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save plot
    if iteration >= 0:
        filename = f"pareto_front_gen_{iteration:03d}.png"
    else:
        filename = "final_pareto_front.png"
    plt.savefig(output_dir / filename, dpi=150, bbox_inches='tight')
    plt.close()


class ParetoPlotter:
    """Event strategy for plotting Pareto fronts during optimization."""
    
    def __init__(self, output_dir: Path, plot_every: int = 10) -> None:
        self.output_dir = output_dir
        self.plot_every = plot_every
        self.figures_dir = output_dir / "figures"
        self.figures_dir.mkdir(parents=True, exist_ok=True)
    
    def execute(self, engine, stage) -> None:
        iteration = getattr(engine._convergence, 'iteration', 0)
        
        if iteration % self.plot_every == 0 or iteration == 0:
            objectives = engine.population.objectives
            plot_pareto_front(objectives, self.figures_dir, iteration)


class MultiObjectivePopulationLogger:
    """Logger for multi-objective population data."""
    
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.objectives_file = self.output_dir / "objectives.csv"
        self.solutions_file = self.output_dir / "solutions.csv"
        
        # Initialize files with headers
        with open(self.objectives_file, 'w') as f:
            f.write("iteration,individual,f1,f2\n")
        with open(self.solutions_file, 'w') as f:
            f.write("iteration,individual,solution\n")
    
    def execute(self, engine, stage) -> None:
        iteration = getattr(engine._convergence, 'iteration', 0)
        population = engine.population
        
        # Log objectives
        with open(self.objectives_file, 'a') as f:
            for i, objectives in enumerate(population.objectives):
                f.write(f"{iteration},{i},{objectives[0]},{objectives[1]}\n")
        
        # Log solutions (first 5 dimensions only to avoid huge files)
        with open(self.solutions_file, 'a') as f:
            for i, solution in enumerate(population.candidates):
                sol_str = ",".join(f"{x:.6f}" for x in solution[:5])
                f.write(f"{iteration},{i},{sol_str}\n")


__all__ = ["plot_pareto_front", "ParetoPlotter", "MultiObjectivePopulationLogger"]