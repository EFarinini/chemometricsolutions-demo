"""
GAPLSOPT - Randomization Test for GA Stop Criterion Selection
==============================================================

EXACT Python translation of Leardi's gaplsopt.m

This is the CRITICAL first step in Leardi's methodology:
1. Run GA on TRUE data vs SHUFFLED data
2. Plot "True vs Random" difference curve
3. User visually determines WHERE difference is maximum
4. That evaluation count becomes the stop criterion for production runs

Original paper:
- Leardi R. (2000). Invited lecture at Summer School in Chemometrics
  "Genetic Algorithms in Chemistry"

Usage:
------
>>> from ga_variable_selection import gaplsopt
>>> progr = gaplsopt(X, y, fitness_fn, test_type='optimization')
>>> # User looks at plot: "Plateau at 80 evals"
>>> # Then runs gaplssp with n_evals=80
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List, Optional, Callable
import warnings


class GAPLSOPT:
    """
    Randomization test for determining optimal GA stop criterion (gaplsopt.m).

    The key insight: Run GA on TRUE data vs SHUFFLED data (random baseline).
    The difference shows WHEN GA finds real patterns vs noise.

    Algorithm:
    - el=1: Randomization test (100 runs on true data, shows random baseline)
    - el=2: Optimization test (50 true + 50 shuffled, shows difference)

    The CRITICAL line (68-70 in MATLAB):
    ```matlab
    if el==1 | (el==2 & r>runs/2)
      k=randperm(o);
      y=y(k);  % SHUFFLE y for random baseline!
    end
    ```
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        fitness_fn: Callable,
        test_type: str = 'optimization',
        population_size: int = 30,
        cv_groups: int = 5,
        nvar_avg: int = 5,
        max_vars: int = 30,
        prob_mutation: float = 0.01,
        prob_crossover: float = 0.5,
        random_seed: Optional[int] = None
    ):
        """
        Initialize GAPLSOPT randomization test.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix (n_samples, n_features)
        y : np.ndarray
            Target variable (n_samples,)
        fitness_fn : callable
            Fitness function: fitness_fn(X_subset, y) -> float
            Should return R² × 100 (percentage variance explained)
        test_type : str
            'randomization' (el=1) or 'optimization' (el=2, recommended)
        population_size : int
            GA population size (cr=30 in Leardi)
        cv_groups : int
            CV deletion groups (ng=5 in Leardi)
        nvar_avg : int
            Average # variables per chromosome (Leardi: 5)
        max_vars : int
            Maximum variables allowed (Leardi: 30)
        prob_mutation : float
            Mutation probability (Leardi: 0.01)
        prob_crossover : float
            Crossover probability (Leardi: 0.5)
        random_seed : int, optional
            Random seed for reproducibility
        """

        self.X = X
        self.y_original = np.copy(y)  # Keep original
        self.fitness_fn = fitness_fn

        # Map test_type to el (MATLAB variable)
        self.el = 1 if test_type == 'randomization' else 2
        self.test_type = test_type

        # GA parameters (from MATLAB lines 33-38)
        self.cr = population_size  # cr=30
        self.ng = cv_groups  # ng=5
        self.probsel = nvar_avg / X.shape[1]  # probsel=5/v
        self.maxvar = max_vars  # maxvar=30
        self.probmut = prob_mutation  # probmut=0.01
        self.probcross = prob_crossover  # probcross=0.5

        # Runs and evaluations (MATLAB lines 39-45)
        self.runs = 100  # ALWAYS 100
        if self.el == 1:
            self.evaluat = 100  # Randomization: 100 evals
        else:
            self.evaluat = 200  # Optimization: 200 evals

        self.o, self.v = X.shape  # objects, variables

        # Progress matrix: runs × evaluations
        self.progr = []

        if random_seed is not None:
            np.random.seed(random_seed)

    def run(self, verbose: bool = True) -> Dict:
        """
        Execute randomization/optimization test.

        Returns
        -------
        results : dict
            'progr': progress matrix (runs × evaluations)
            'recommended_evals': optimal stop criterion (for el=2)
            'max_difference': maximum difference value (for el=2)
            'test_type': 'randomization' or 'optimization'
        """

        if verbose:
            print(f"\n{'='*70}")
            print(f"GAPLSOPT: {'Randomization' if self.el==1 else 'Optimization'} Test")
            print(f"{'='*70}")
            print(f"Runs: {self.runs}")
            print(f"Evaluations per run: {self.evaluat}")
            print(f"Data: {self.o} samples × {self.v} variables")
            if self.el == 2:
                print(f"Mode: First 50 TRUE data, Second 50 SHUFFLED (random baseline)")
            print()

        # Main loop: 100 runs
        for r in range(self.runs):
            if verbose and (r + 1) % 10 == 0:
                print(f"Run {r + 1}/{self.runs} ...", end=" ")

            # ═══════════════════════════════════════════════════════════
            # CRITICAL: Shuffle y for random baseline (MATLAB lines 68-70)
            # ═══════════════════════════════════════════════════════════
            y_run = np.copy(self.y_original)

            # el=1: All runs on true data (baseline reference)
            # el=2: Second half (r >= 50) use SHUFFLED data
            if self.el == 1 or (self.el == 2 and r >= self.runs // 2):
                y_run = np.random.permutation(y_run)
                if verbose and (r + 1) % 10 == 0:
                    print("[SHUFFLED]", end=" ")

            # ═══════════════════════════════════════════════════════════
            # Initialize population (MATLAB lines 72-112)
            # ═══════════════════════════════════════════════════════════
            population = []
            fitness_pop = []
            library = set()  # Already tested chromosomes
            cc = 0  # Evaluation counter

            # Create initial population (cr chromosomes)
            while cc < self.cr:
                # Generate chromosome
                sumvar = 0
                while sumvar == 0 or sumvar > self.maxvar:
                    a = np.random.random(self.v) < self.probsel
                    sumvar = np.sum(a)

                # Check if duplicate
                a_tuple = tuple(a)
                if a_tuple in library:
                    continue

                library.add(a_tuple)
                cc += 1

                # Evaluate fitness
                try:
                    var_indices = np.where(a)[0]
                    if len(var_indices) == 0:
                        continue

                    fit = self.fitness_fn(self.X[:, var_indices], y_run)
                    population.append(a)
                    fitness_pop.append(fit)
                except:
                    fitness_pop.append(0.0)
                    population.append(a)

            # Sort by fitness (MATLAB lines 114-119)
            if len(fitness_pop) > 0:
                sorted_idx = np.argsort(-np.array(fitness_pop))
                population = [population[i] for i in sorted_idx]
                fitness_pop = [fitness_pop[i] for i in sorted_idx]

            # Store progress after initial population (MATLAB line 121)
            run_progress = [fitness_pop[0] if fitness_pop else 0.0]

            # ═══════════════════════════════════════════════════════════
            # GA Evolution (MATLAB lines 126-202)
            # ═══════════════════════════════════════════════════════════
            while cc < self.evaluat:
                # ─────────────────────────────────────────────────────
                # Selection (roulette wheel) - MATLAB lines 128-153
                # ─────────────────────────────────────────────────────
                fitness_array = np.array(fitness_pop)
                cumsum_fitness = np.cumsum(fitness_array)

                if cumsum_fitness[-1] <= 0 or len(population) < 2:
                    break

                parents = []
                for _ in range(2):
                    r_val = np.random.random() * cumsum_fitness[-1]
                    parent_idx = np.searchsorted(cumsum_fitness, r_val)
                    parent_idx = min(parent_idx, len(population) - 1)
                    parents.append(population[parent_idx])

                # ─────────────────────────────────────────────────────
                # Crossover - MATLAB lines 156-161
                # ─────────────────────────────────────────────────────
                offspring = []
                diff_positions = np.where(parents[0] != parents[1])[0]

                for p_idx in range(2):
                    child = np.copy(parents[p_idx])

                    for pos in diff_positions:
                        if np.random.random() < self.probcross:
                            child[pos] = parents[1 if p_idx == 0 else 0][pos]

                    offspring.append(child)

                # ─────────────────────────────────────────────────────
                # Mutation - MATLAB lines 164-175
                # ─────────────────────────────────────────────────────
                for child in offspring:
                    mutations = np.random.random(self.v) < self.probmut
                    child[mutations] = 1 - child[mutations]

                # ─────────────────────────────────────────────────────
                # Evaluate offspring - MATLAB lines 178-201
                # ─────────────────────────────────────────────────────
                for child in offspring:
                    if cc >= self.evaluat:
                        break

                    sumvar = np.sum(child)

                    # Check constraints
                    if sumvar == 0 or sumvar > self.maxvar:
                        continue

                    child_tuple = tuple(child)
                    if child_tuple in library:
                        continue

                    library.add(child_tuple)
                    cc += 1

                    # Evaluate
                    try:
                        var_indices = np.where(child)[0]
                        fit = self.fitness_fn(self.X[:, var_indices], y_run)

                        # Replace worst if better (MATLAB line 196)
                        if fit > fitness_pop[-1]:
                            population[-1] = child
                            fitness_pop[-1] = fit

                            # Re-sort
                            sorted_idx = np.argsort(-np.array(fitness_pop))
                            population = [population[i] for i in sorted_idx]
                            fitness_pop = [fitness_pop[i] for i in sorted_idx]
                    except:
                        pass

                    # Store progress (MATLAB line 200)
                    run_progress.append(fitness_pop[0])

            # Store this run's progress
            self.progr.append(run_progress)

            if verbose and (r + 1) % 10 == 0:
                print(f"Best: {fitness_pop[0]:.2f}")

        if verbose:
            print(f"\n{'='*70}")
            print("GAPLSOPT Complete!")
            print(f"{'='*70}\n")

        # Format progress matrix
        progr_matrix = self._format_progress()

        # Prepare results
        results = {
            'progr': progr_matrix,
            'test_type': self.test_type,
            'n_runs': self.runs,
            'n_evals': self.evaluat
        }

        # Calculate recommendation for el=2 (MATLAB lines 233-235)
        if self.el == 2:
            n_half = self.runs // 2
            progr_true = progr_matrix[:n_half, :]
            progr_random = progr_matrix[n_half:, :]

            mean_true = np.mean(progr_true, axis=0)
            mean_random = np.mean(progr_random, axis=0)
            difference = mean_true - mean_random

            max_idx = np.argmax(difference)
            max_eval = self.cr + max_idx  # Add initial population size
            max_diff = difference[max_idx]

            results['mean_true'] = mean_true
            results['mean_random'] = mean_random
            results['difference'] = difference
            results['recommended_evals'] = int(max_eval)
            results['max_difference'] = float(max_diff)

            if verbose:
                print(f"🎯 RECOMMENDATION:")
                print(f"   Maximum difference: {max_diff:.4f}")
                print(f"   Occurs at evaluation: {int(max_eval)}")
                print(f"   → Use n_evals = {int(max_eval)} for production GAPLSSP runs")
                print()

        return results

    def _format_progress(self) -> np.ndarray:
        """Format progress matrix with consistent shape."""
        max_len = max(len(p) for p in self.progr)

        progr_matrix = np.zeros((len(self.progr), max_len))
        for i, p in enumerate(self.progr):
            progr_matrix[i, :len(p)] = p
            # Fill remaining with last value
            if len(p) < max_len:
                progr_matrix[i, len(p):] = p[-1]

        return progr_matrix

    def plot(self, figsize=(14, 10), save_path: Optional[str] = None):
        """
        Plot randomization test results.

        For el=1 (randomization): Shows all runs + mean
        For el=2 (optimization): Shows True vs Random + Difference
        """

        progr = self._format_progress()
        pp = np.arange(self.cr, self.cr + progr.shape[1])

        if self.el == 1:
            # ═══════════════════════════════════════════════════════
            # Randomization Test Plot (MATLAB lines 207-216)
            # ═══════════════════════════════════════════════════════
            fig, ax = plt.subplots(figsize=figsize)

            # All runs (yellow)
            for i in range(len(progr)):
                ax.plot(pp, progr[i, :], 'y-', alpha=0.3, linewidth=1)

            # Mean (red)
            mean_progr = np.mean(progr, axis=0)
            ax.plot(pp, mean_progr, 'r-', linewidth=2, label='Mean')

            ax.set_xlabel('Evaluations', fontsize=12)
            ax.set_ylabel('Fitness (% Variance)', fontsize=12)
            ax.set_title('Randomization Test: GA on True Data', fontsize=14)
            ax.legend()
            ax.grid(True, alpha=0.3)

            final_mean = mean_progr[-1]
            ax.text(0.95, 0.05, f'Final mean: {final_mean:.4f}',
                    transform=ax.transAxes, ha='right', va='bottom',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        else:
            # ═══════════════════════════════════════════════════════
            # Optimization Test Plot (MATLAB lines 219-239)
            # ═══════════════════════════════════════════════════════
            n_half = len(progr) // 2
            progr_true = progr[:n_half, :]
            progr_random = progr[n_half:, :]

            mean_true = np.mean(progr_true, axis=0)
            mean_random = np.mean(progr_random, axis=0)
            difference = mean_true - mean_random

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)

            # ─────────────────────────────────────────────────────
            # Subplot 1: All runs (MATLAB figure 1)
            # ─────────────────────────────────────────────────────
            for i in range(n_half):
                ax1.plot(pp, progr_true[i, :], 'y-', alpha=0.2, linewidth=0.5)
            for i in range(n_half):
                ax1.plot(pp, progr_random[i, :], 'r-', alpha=0.2, linewidth=0.5)

            ax1.plot(pp, mean_true, 'g-', linewidth=2.5, label='Mean (True data)')
            ax1.plot(pp, mean_random, 'c-', linewidth=2.5, label='Mean (Random)')

            ax1.set_xlabel('Evaluations', fontsize=12)
            ax1.set_ylabel('Fitness (% Variance)', fontsize=12)
            ax1.set_title('Optimization Test: True vs Random Baseline', fontsize=14)
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # ─────────────────────────────────────────────────────
            # Subplot 2: Difference (MATLAB figure 2, line 229)
            # ─────────────────────────────────────────────────────
            ax2.plot(pp, difference, 'w-', linewidth=2.5, label='True - Random')
            ax2.fill_between(pp, difference, alpha=0.3, color='blue')

            # Mark maximum (MATLAB line 233)
            max_idx = np.argmax(difference)
            max_eval = pp[max_idx]
            max_diff = difference[max_idx]

            ax2.plot(max_eval, max_diff, 'ro', markersize=12,
                     label=f'Max at {int(max_eval)} evals', zorder=10)
            ax2.axvline(max_eval, color='r', linestyle='--', alpha=0.7, linewidth=2)

            ax2.set_xlabel('Evaluations', fontsize=12)
            ax2.set_ylabel('Difference (True - Random)', fontsize=12)
            ax2.set_title("LEARDI'S STOPPING CRITERION: Where is the maximum?",
                          fontsize=14, fontweight='bold')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            # Recommendation text
            ax2.text(0.95, 0.95,
                     f'🎯 RECOMMENDATION:\nUse n_evals = {int(max_eval)}',
                     transform=ax2.transAxes, ha='right', va='top', fontsize=12,
                     bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Plot saved to: {save_path}")

        plt.show()

        return fig


def gaplsopt(
    X: np.ndarray,
    y: np.ndarray,
    fitness_fn: Callable,
    test_type: str = 'optimization',
    plot: bool = True,
    **kwargs
) -> Dict:
    """
    Wrapper function matching gaplsopt.m interface.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix (n_samples, n_features)
    y : np.ndarray
        Target variable (n_samples,)
    fitness_fn : callable
        Fitness function: fitness_fn(X_subset, y) -> float
        Should return R² × 100 (percentage variance)
    test_type : str
        'randomization' (el=1) or 'optimization' (el=2, recommended)
    plot : bool
        If True, display plot automatically
    **kwargs
        Additional GA parameters passed to GAPLSOPT

    Returns
    -------
    results : dict
        'progr': progress matrix (runs × evaluations)
        'recommended_evals': optimal stop criterion (for optimization mode)
        'max_difference': maximum difference value (for optimization mode)
        'test_type': 'randomization' or 'optimization'

    Example
    -------
    >>> from ga_variable_selection import gaplsopt, GAPLSSP
    >>>
    >>> # Step 1: Determine stop criterion
    >>> results = gaplsopt(X, y, fitness_fn, test_type='optimization')
    >>> # Plot shows: "Maximum at 80 evaluations"
    >>> n_evals = results['recommended_evals']
    >>>
    >>> # Step 2: Run production GA with that criterion
    >>> ga = GAPLSSP(X, y, fitness_fn, n_runs=100, n_evals=n_evals)
    >>> final_results = ga.run()
    """

    ga = GAPLSOPT(X, y, fitness_fn, test_type=test_type, **kwargs)
    results = ga.run(verbose=kwargs.get('verbose', True))

    # Plot if requested
    if plot:
        ga.plot()

    return results
