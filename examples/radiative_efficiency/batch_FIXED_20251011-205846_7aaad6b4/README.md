# Radiative Efficiency Optimization - Batch Run Results

**Run Date:** October 11, 2025  
**Batch ID:** `batch_FIXED_20251011-205846_7aaad6b4`

---

## Problem Description

### Objective
Maximize the **radiative efficiency** of a fluidized bed combustion system by optimizing 7 operational and design parameters. The optimization uses symbolic regression models trained with PySR to predict:
1. **Alpha (α)**: Flame classifier (α > 0.5 indicates flame presence)
2. **Radiative Efficiency (η)**: Only computed when α > 0.5

### Mathematical Formulation

$$
\begin{align}
\text{Maximize:} \quad & \eta(\mathbf{x}) \\
\text{Subject to:} \quad & \alpha(\mathbf{x}) > 0.5 \quad \text{(flame condition)} \\
& \mathbf{x}_{\text{min}} \leq \mathbf{x} \leq \mathbf{x}_{\text{max}}
\end{align}
$$

Where $\mathbf{x} = [\phi, u_{\text{avg}}, ar, lt_0, lt_1, \varepsilon_1, a_1]^T$

---

## Optimization Parameters

### Decision Variables

| Parameter | Symbol | Description | Lower Bound | Upper Bound | Units |
|-----------|--------|-------------|-------------|-------------|-------|
| `phi` | $\phi$ | Equivalence ratio | 0.3 | 0.9 | - |
| `u_avg` | $u_{\text{avg}}$ | Average inlet velocity | 0.5 | 2.4 | m/s |
| `ar` | $ar$ | Aspect ratio | 1.1 | 5.0 | - |
| `lt_0` | $lt_0$ | First layer thickness | 0.005 | 0.015 | m |
| `lt_1` | $lt_1$ | Second layer thickness | 0.03 | 0.08 | m |
| `eps_1` | $\varepsilon_1$ | Second layer emissivity | 0.4 | 0.95 | - |
| `a_1` | $a_1$ | Second layer extinction coefficient | 400.0 | 800.0 | 1/m |

**Problem Dimensionality:** 7D  
**Search Space:** Continuous, bounded hypercube

### Physical Constraints

1. **Flame Condition:** $\alpha(\mathbf{x}) > 0.5$
   - No flame exists when α ≤ 0.5, resulting in zero radiative efficiency

2. **Box Constraints:** All parameters must remain within specified bounds

---

## Genetic Algorithm Configuration

### Algorithm: Real-Coded Genetic Algorithm (RCGA)

#### Population
- **Population Size:** 50 individuals
- **Initialization:** Uniform random sampling within parameter bounds
- **Encoding:** Real-valued vectors

#### Genetic Operators

##### Selection
- **Method:** Tournament Selection
- **Tournament Size:** k = 2

##### Crossover
- **Method:** Arithmetic Crossover
- **Probability:** 0.8
- **Mechanism:** $\mathbf{x}_{\text{child}} = \lambda \mathbf{x}_{\text{parent1}} + (1-\lambda) \mathbf{x}_{\text{parent2}}$, where λ is randomly sampled

##### Mutation
- **Method:** Gaussian Mutation
- **Application:** Per-gene (independent mutation for each parameter)
- **Probability:** 0.10 per gene
- **Mutation Strength:** σ = 0.10 (standard deviation in normalized space)
- **Expected Mutations:** ~0.7 genes per individual on average

##### Elitism
- **Strategy:** Best individual preserved across generations
- **Elite Count:** 1

#### Termination Criteria
- **Maximum Generations:** 100
- **Maximum Evaluations:** 500,000 function evaluations

---

## Symbolic Regression Models

The optimization relies on two data-driven models obtained through symbolic regression using PySR:

### Model 1: Flame Classifier
- **Purpose:** Predict flame presence/absence
- **Output:** α ∈ [0, 1]
- **Threshold:** α > 0.5 indicates stable flame
- **Model ID:** 20251010_155720_BXNBv9

### Model 2: Radiative Efficiency
- **Purpose:** Predict radiative efficiency when flame is present
- **Output:** η ∈ [0, 1]
- **Condition:** Only evaluated when α > 0.5
- **Model ID:** 20251010_162508_k1g9rP

Both models are algebraic expressions discovered from experimental data of fluidized bed combustion tests.

---

## Batch Run Configuration

- **Number of Independent Runs:** 100
- **Generations per Run:** 100
- **Total Function Evaluations:** ~500,000 (100 runs × 50 individuals × 100 generations)

---

## Results and Data Structure

Each run produces:

1. **Final Results** (`run_results.json`)
   - Best radiative efficiency achieved
   - Optimal parameter values
   - Number of generations and execution time

2. **Evolution Data** (`data/iter_X.csv`)
   - Complete population at each generation
   - Parameters in both normalized and real scales
   - Objective function values
   - Sorted by fitness (best individual first)

3. **Visualizations** (`figures/frame_XXXX.png`)
   - 2D contour plots: η(φ, u_avg) with other parameters fixed at best solution
   - Population distribution overlay
   - Generated at each generation

---

## Optimization Results

### Summary Statistics (100 Independent Runs)

**Success Rate:** 100% (all runs converged to valid solutions)

#### Radiative Efficiency
| Metric | Value |
|--------|-------|
| Best | 0.4492 |
| Worst | 0.4474 |
| Mean ± Std | 0.4487 ± 0.0003 |
| Median | 0.4487 |
| IQR | [0.4486, 0.4489] |

**Coefficient of Variation:** 0.077% (excellent consistency across runs)

#### Computational Performance
| Metric | Value |
|--------|-------|
| Mean Execution Time | 264.9 ± 34.3 s |
| Total Optimization Time | 7.36 hours |
| Function Evaluations per Run | 4,901 |
| Total Function Evaluations | 490,100 |

### Optimal Parameter Values

**Best Solution** (Run #3, η = 0.4492):

| Parameter | Optimal Value | Mean ± Std (all runs) | Range (min, max) |
|-----------|---------------|----------------------|------------------|
| φ | 0.728 | 0.726 ± 0.025 | [0.675, 0.790] |
| u_avg (m/s) | 0.500 | 0.502 ± 0.002 | [0.500, 0.509] |
| ar | 4.999 | 4.994 ± 0.005 | [4.970, 5.000] |
| lt_0 (m) | 0.00756 | 0.01009 ± 0.00214 | [0.00599, 0.01399] |
| lt_1 (m) | 0.0614 | 0.0592 ± 0.0086 | [0.0389, 0.0776] |
| ε_1 | 0.688 | 0.660 ± 0.111 | [0.426, 0.902] |
| a_1 (1/m) | 670.2 | 601.5 ± 84.8 | [402.0, 770.0] |

### Parameter Sensitivity Analysis

Based on coefficient of variation (CV = std/mean):

1. **Highly Constrained Parameters** (CV < 1%):
   - **u_avg**: CV = 0.31% → Optimal velocity narrowly defined near lower bound
   - **ar**: CV = 0.10% → Aspect ratio strongly converged to upper bound (~5.0)

2. **Moderately Constrained** (1% < CV < 10%):
   - **φ**: CV = 3.5% → Equivalence ratio shows some variability
   - **lt_1**: CV = 14.4% → Second layer thickness moderately constrained
   - **lt_0**: CV = 21.2% → First layer thickness less critical

3. **Weakly Constrained** (CV > 10%):
   - **ε_1**: CV = 16.8% → Emissivity has broader optimal region
   - **a_1**: CV = 14.1% → Extinction coefficient moderately flexible

**Key Findings:**
- **u_avg** and **ar** converge very tightly, indicating these are critical for optimal efficiency
- **ε_1** and **a_1** show higher variability, suggesting multiple near-optimal combinations exist
- All optimal solutions clustered in a small region of the search space (low CV in objective)

### Convergence Characteristics

- **Convergence Speed:** Fast (within expected 20-30 generations based on tight objective variance)
- **Solution Quality:** Highly consistent (std = 0.0003 in objective)
- **Robustness:** 100% success rate with narrow efficiency distribution
- **Optimum Type:** Strong indication of single global optimum in feasible region

---

## Discussion

### Achieved Performance

The optimization successfully achieved radiative efficiencies in the range **η = 0.4474 to 0.4492**, with remarkable consistency across all 100 independent runs (CV = 0.077%). This narrow distribution indicates:

1. **Well-defined optimum:** The search space contains a clear, reachable global optimum
2. **Algorithm effectiveness:** The genetic algorithm reliably converges to this region
3. **Model reliability:** The surrogate models provide consistent predictions

### Optimal Operating Conditions

The best configuration corresponds to:
- **Lean combustion** (φ ≈ 0.73, below stoichiometric)
- **Low inlet velocity** (u_avg ≈ 0.50 m/s, at lower bound)
- **High aspect ratio** (ar ≈ 5.0, at upper bound)
- **Thin first layer** (lt_0 ≈ 0.0076 m)
- **Moderate second layer** (lt_1 ≈ 0.061 m)
- **Moderate emissivity** (ε_1 ≈ 0.69)
- **High extinction coefficient** (a_1 ≈ 670 1/m)

### Physical Insights

1. **Velocity-Residence Time Trade-off:** Convergence to minimum velocity maximizes residence time for radiative heat transfer

2. **Aspect Ratio Effect:** Maximum aspect ratio (tall, narrow bed) likely promotes better temperature distribution and radiative exchange

3. **Layer Configuration:** Thin first layer with moderate second layer suggests optimal balance between thermal resistance and radiative absorption

4. **Optical Properties:** Moderate-to-high extinction coefficient indicates importance of particulate radiation in overall heat transfer

### Implications for Practice

- **Narrow Operational Window:** Critical parameters (u_avg, ar) must be tightly controlled
- **Design Priority:** Aspect ratio and velocity control are most critical for achieving optimal efficiency
- **Flexibility:** Emissivity and extinction coefficient (material properties) show broader acceptable ranges

---

## Performance Expectations (Initial)

- **Target Efficiency Range:** η ∈ [0.35, 0.50]
- **Convergence Behavior:** Significant improvement in first 20-30 generations
- **Solution Consistency:** Similar optimal regions across multiple runs indicate reliability

---

## References

### Symbolic Regression
- Cranmer, M. (2023). Interpretable Machine Learning for Science with PySR and SymbolicRegression.jl. arXiv:2305.01582

### Genetic Algorithms
- Goldberg, D. E. (1989). Genetic Algorithms in Search, Optimization, and Machine Learning. Addison-Wesley.
- Deb, K., & Agrawal, R. B. (1995). Simulated binary crossover for continuous search space. Complex Systems, 9(2), 115-148.

---

**Generated:** October 11, 2025
