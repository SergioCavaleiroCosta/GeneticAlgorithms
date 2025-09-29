"""Plotting utilities for NSGA-II Schaffer N.1 multi-objective optimization visualization."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence
import matplotlib.pyplot as plt
import numpy as np

# Make project src/ importable
import sys
ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nsga_ii.pareto_utils import non_dominated_sort
from .problems import SchafferN1Problem


def plot_pareto_front(objectives: list[Sequence[float]], output_dir: Path, iteration: int = -1) -> None:
    """Plot the current population and true Pareto front with color-coded ranks."""
    # Extract objective values
    f1_vals = [obj[0] for obj in objectives]
    f2_vals = [obj[1] for obj in objectives]
    
    # Perform non-dominated sorting to get ranks
    fronts = non_dominated_sort(objectives)
    
    # Create labels showing the generation number
    title = f'NSGA-II on Schaffer N.1 Problem - Generation {iteration}'
    
    # Define colors for the first 4 ranks
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Blue, Orange, Green, Red
    rank_labels = ['Rank 1 (Best)', 'Rank 2', 'Rank 3', 'Rank 4']
    
    # Create plot
    plt.figure(figsize=(10, 8))
    
    # Plot each rank with different colors
    for rank, front in enumerate(fronts[:4]):  # Only plot first 4 ranks
        if not front:  # Skip empty fronts
            continue
            
        # Extract coordinates for this rank
        rank_f1 = [f1_vals[i] for i in front]
        rank_f2 = [f2_vals[i] for i in front]
        
        plt.scatter(rank_f1, rank_f2, 
                   alpha=0.7, s=40, 
                   color=colors[rank], 
                   label=f'{rank_labels[rank]} ({len(front)} points)',
                   edgecolors='black', linewidth=0.5)
    
    # Plot remaining ranks (5+) in gray if they exist
    if len(fronts) > 4:
        remaining_indices = []
        for front in fronts[4:]:
            remaining_indices.extend(front)
        
        if remaining_indices:
            remaining_f1 = [f1_vals[i] for i in remaining_indices]
            remaining_f2 = [f2_vals[i] for i in remaining_indices]
            plt.scatter(remaining_f1, remaining_f2, 
                       alpha=0.4, s=20, 
                       color='gray', 
                       label=f'Ranks 5+ ({len(remaining_indices)} points)')
    
    # Plot true Pareto front for Schaffer N.1
    problem = SchafferN1Problem()
    true_f1, true_f2 = problem.get_true_pareto_front(100)
    plt.plot(true_f1, true_f2, 'r-', linewidth=2, label='True Pareto Front')
    
    plt.xlabel('f1', fontsize=16)
    plt.ylabel('f2', fontsize=16)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    if iteration >= 0:
        filename = f"pareto_front_gen_{iteration:03d}.png"
    else:
        filename = "final_pareto_front.png"
    plt.savefig(output_dir / filename, dpi=150, bbox_inches='tight')
    plt.close()


def plot_decision_space(population_candidates: list, output_dir: Path, iteration: int = -1) -> None:
    """Plot the population in decision space (x values) to show convergence."""
    # Extract x values (single decision variable)
    x_vals = [float(np.asarray(candidate).flatten()[0]) for candidate in population_candidates]
    
    # Create histogram of x values
    plt.figure(figsize=(10, 6))
    plt.hist(x_vals, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    
    # Mark the true Pareto optimal region [0, 2]
    plt.axvspan(0, 2, alpha=0.3, color='red', label='Pareto Optimal Region [0, 2]')
    
    plt.xlabel('Decision Variable x', fontsize=16)
    plt.ylabel('Frequency', fontsize=16)
    plt.title(f'Population Distribution in Decision Space - Generation {iteration}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    if iteration >= 0:
        filename = f"decision_space_gen_{iteration:03d}.png"
    else:
        filename = "final_decision_space.png"
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
            candidates = engine.population.candidates
            
            # Plot both objective space and decision space
            plot_pareto_front(objectives, self.figures_dir, iteration)
            plot_decision_space(candidates, self.figures_dir, iteration)


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
            f.write("iteration,individual,x\n")
    
    def execute(self, engine, stage) -> None:
        iteration = getattr(engine._convergence, 'iteration', 0)
        population = engine.population
        
        # Log objectives
        with open(self.objectives_file, 'a') as f:
            for i, objectives in enumerate(population.objectives):
                f.write(f"{iteration},{i},{objectives[0]},{objectives[1]}\n")
        
        # Log solutions (single variable)
        with open(self.solutions_file, 'a') as f:
            for i, solution in enumerate(population.candidates):
                x_val = float(np.asarray(solution).flatten()[0])
                f.write(f"{iteration},{i},{x_val:.6f}\n")


__all__ = ["plot_pareto_front", "plot_decision_space", "ParetoPlotter", "MultiObjectivePopulationLogger"]