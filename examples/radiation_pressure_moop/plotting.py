"""Visualize Pareto front and optimization results."""
from __future__ import annotations
import sys
from pathlib import Path
import json
import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def load_pareto_data(output_dir: Path) -> dict:
    """Load Pareto front data from output directory."""
    pareto_path = output_dir / "pareto_front.json"
    
    if not pareto_path.exists():
        raise FileNotFoundError(f"Pareto front file not found: {pareto_path}")
    
    with open(pareto_path, "r") as f:
        data = json.load(f)
    
    return data


def load_config(output_dir: Path) -> dict:
    """Load configuration from output directory."""
    config_path = output_dir / "config.json"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, "r") as f:
        config = json.load(f)
    
    return config


def plot_pareto_front(
    objectives: np.ndarray,
    output_path: Path | None = None,
    title: str = "",
) -> Figure:
    """Plot the Pareto front in objective space.
    
    Args:
        objectives: Array of shape (n_solutions, 2) with objectives [-η, ΔP]
        output_path: Path to save figure
        title: Plot title
        
    Returns:
        Matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Convert first objective back to positive (η)
    eta = -objectives[:, 0]
    pressure_drop = objectives[:, 1]
    
    # Plot Pareto front
    ax.scatter(eta, pressure_drop, c='red', s=100, alpha=0.7, 
               edgecolors='black', linewidths=1.5, zorder=5, label='Frente de Pareto')
    
    # Plot connecting line
    sorted_indices = np.argsort(eta)
    ax.plot(eta[sorted_indices], pressure_drop[sorted_indices], 
            'k--', alpha=0.3, linewidth=1, zorder=1)
    
    ax.set_xlabel(r'Eficiência radiativa ($\eta$)', fontsize=14)
    ax.set_ylabel(r'Queda de pressão ($\Delta P$, Pa)', fontsize=14)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=12)
    
    # Add some statistics
    stats_text = (
        f'Soluções de Pareto: {len(eta)}\n'
        f'$\\eta$ intervalo: [{eta.min():.4f}, {eta.max():.4f}]\n'
        f'$\\Delta P$ intervalo: [{pressure_drop.min():.1f}, {pressure_drop.max():.1f}] Pa'
    )
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if output_path is not None:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Gráfico da frente de Pareto guardado em: {output_path}")
    
    return fig


def plot_parameter_distributions(
    solutions: np.ndarray,
    output_path: Path | None = None,
) -> Figure:
    """Plot distributions of decision variables in Pareto set.
    
    Args:
        solutions: Array of shape (n_solutions, n_vars) with decision variables
        output_path: Path to save figure
        
    Returns:
        Matplotlib Figure object
    """
    param_names = [
        r'$\phi$', 
        r'$u_{\mathrm{avg}}$ (m/s)', 
        r'$ar$', 
        r'$lt_0$ (m)', 
        r'$lt_1$ (m)', 
        r'$\varepsilon_1$', 
        r'$a_1$ (1/m)'
    ]
    n_params = solutions.shape[1]
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for i in range(n_params):
        ax = axes[i]
        ax.hist(solutions[:, i], bins=20, edgecolor='black', alpha=0.7)
        ax.set_xlabel(param_names[i], fontsize=12)
        ax.set_ylabel('Frequência', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Add statistics
        mean_val = solutions[:, i].mean()
        std_val = solutions[:, i].std()
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'$\\mu$={mean_val:.3f}')
        ax.legend(fontsize=10)
    
    # Remove the last (8th) subplot
    fig.delaxes(axes[-1])
    
    plt.tight_layout()
    
    if output_path is not None:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Gráfico de distribuições de parâmetros guardado em: {output_path}")
    
    return fig


def plot_objective_tradeoff(
    objectives: np.ndarray,
    solutions: np.ndarray,
    param_idx: int = 0,
    output_path: Path | None = None,
) -> Figure:
    """Plot objective tradeoff colored by a decision variable.
    
    Args:
        objectives: Array of shape (n_solutions, 2) with objectives
        solutions: Array of shape (n_solutions, n_vars) with decision variables
        param_idx: Index of parameter to use for coloring
        output_path: Path to save figure
        
    Returns:
        Matplotlib Figure object
    """
    param_names = [
        r'$\phi$', 
        r'$u_{\mathrm{avg}}$ (m/s)', 
        r'$ar$', 
        r'$lt_0$ (m)', 
        r'$lt_1$ (m)', 
        r'$\varepsilon_1$', 
        r'$a_1$ (1/m)'
    ]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    eta = -objectives[:, 0]
    pressure_drop = objectives[:, 1]
    param_values = solutions[:, param_idx]
    
    scatter = ax.scatter(eta, pressure_drop, c=param_values, s=100, 
                        cmap='viridis', alpha=0.7, edgecolors='black', linewidths=1.5)
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label(param_names[param_idx], fontsize=14)
    
    ax.set_xlabel(r'Eficiência radiativa ($\eta$)', fontsize=14)
    ax.set_ylabel(r'Queda de pressão ($\Delta P$, Pa)', fontsize=14)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    if output_path is not None:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Gráfico de compromisso guardado em: {output_path}")
    
    return fig


def create_all_plots(output_dir: Path | str) -> None:
    """Create all visualization plots for a run.
    
    Args:
        output_dir: Path to the output directory containing results
    """
    output_dir = Path(output_dir)
    
    if not output_dir.exists():
        raise FileNotFoundError(f"Output directory not found: {output_dir}")
    
    print(f"Loading results from: {output_dir}")
    
    # Load data
    pareto_data = load_pareto_data(output_dir)
    config = load_config(output_dir)
    
    objectives = np.array(pareto_data["objectives"])
    solutions = np.array(pareto_data["solutions"])
    
    print(f"Carregadas {len(objectives)} soluções de Pareto")
    
    # Create figures directory
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)
    
    # Create plots
    print("\nA criar gráficos...")
    
    # 1. Pareto front
    plot_pareto_front(
        objectives,
        output_path=figures_dir / "pareto_front.png",
        title="",
    )
    
    # 2. Parameter distributions
    plot_parameter_distributions(
        solutions,
        output_path=figures_dir / "parameter_distributions.png",
    )
    
    # 3. Objective tradeoffs colored by different parameters
    for i, param_name in enumerate(['phi', 'u_avg', 'ar']):
        plot_objective_tradeoff(
            objectives,
            solutions,
            param_idx=i,
            output_path=figures_dir / f"tradeoff_by_{param_name}.png",
        )
    
    print(f"\nTodos os gráficos guardados em: {figures_dir}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Visualize NSGA-II optimization results"
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="Path to the output directory containing results",
    )
    
    args = parser.parse_args()
    
    create_all_plots(args.output_dir)


if __name__ == "__main__":
    main()
