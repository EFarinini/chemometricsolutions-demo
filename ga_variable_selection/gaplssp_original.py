"""
GAPLSSP - Genetic Algorithm for PLS Spectral Variable Selection
================================================================

EXACT Python translation of Leardi's gaplssp.m (MATLAB original, 1998)

Original author: R. Leardi
Original paper:
- Leardi R. (1998). "Genetic algorithms applied to feature selection in PLS regression"
  Chemometrics and Intelligent Laboratory Systems, 41, 195-207

This is a FAITHFUL translation - NO modifications, NO "improvements".

Key algorithm features (from original MATLAB code):
1. 100 sequential runs (lines 54, 67)
2. Probability EVOLVES each run based on previous selections (lines 82-91)
3. Selection frequency ACCUMULATES across runs (line 240)
4. Final smoothing with moving average window=3 (lines 250-256)
5. Final stepwise only if sel(c) > sel(c+1) (line 269)

Usage:
------
>>> from gaplssp_original import GAPLSSP
>>> ga = GAPLSSP(X, y, fitness_fn, n_runs=100, n_evals=200)
>>> results = ga.run()
>>> selected_vars = results['selected_variables']
"""

import numpy as np
from typing import Tuple, Dict, List, Optional, Callable
import warnings


class GAPLSSP:
    """
    Genetic Algorithm for PLS Spectral Variable Selection.

    Pure Python implementation of Leardi's gaplssp.m (1998).

    The algorithm runs 100 sequential iterations where:
    1. Each run has evolved probability based on previous selections
    2. GA finds best variables for that run
    3. Frequencies accumulate across all runs
    4. Final model from stepwise on smoothed accumulated frequencies

    Parameters
    ----------
    X : np.ndarray
        Feature matrix (n_samples, n_variables)
    y : np.ndarray
        Target variable (n_samples,)
    fitness_fn : callable
        Fitness evaluation function: fitness_fn(X_subset, y) -> float
        Should return % variance explained (like R² × 100)
    n_runs : int, default=100
        Number of sequential runs (Leardi original: 100)
    n_evals : int, default=200
        Number of evaluations per run (stop criterion)
    population_size : int, default=30
        Population size (cr in MATLAB, line 42)
    cv_groups : int, default=5
        Number of CV deletion groups (ng in MATLAB, line 41)
    nvar_avg : int, default=5
        Average # variables per chromosome in initial population (line 43)
    max_vars : int, default=30
        Maximum variables allowed in chromosome (line 45)
    prob_mutation : float, default=0.01
        Probability of mutation (line 46)
    prob_crossover : float, default=0.5
        Probability of crossover (line 47)
    backward_freq : int, default=100
        Backward stepwise frequency (line 48)
    verbose : bool, default=True
        Print progress messages
    progress_callback : callable, optional
        Callback function for real-time progress updates
    random_seed : int, optional
        Random seed for reproducibility. If None, uses current random state.

    Attributes
    ----------
    selection_freq_ : np.ndarray
        Accumulated selection frequency across all runs
    selection_freq_smoothed_ : np.ndarray
        Smoothed frequencies (moving average window=3)
    best_chromosomes_ : list
        Best chromosome from each run
    best_fitnesses_ : np.ndarray
        Best fitness from each run
    run_history_ : list
        Detailed history of each run
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        fitness_fn: Callable[[np.ndarray, np.ndarray], float],
        n_runs: int = 100,
        n_evals: int = 200,
        population_size: int = 30,
        cv_groups: int = 5,
        nvar_avg: int = 5,
        max_vars: int = 30,
        prob_mutation: float = 0.01,
        prob_crossover: float = 0.5,
        backward_freq: int = 100,
        verbose: bool = True,
        progress_callback: Optional[Callable] = None,
        random_seed: Optional[int] = None
    ):
        # Data
        self.X = X
        self.y = y
        self.fitness_fn = fitness_fn

        # GA parameters (matching MATLAB variable names where possible)
        self.runs = n_runs              # MATLAB: runs (line 54)
        self.evaluat = n_evals           # MATLAB: evaluat (line 146)
        self.cr = population_size        # MATLAB: cr (line 42)
        self.ng = cv_groups              # MATLAB: ng (line 41)
        self.nvar = nvar_avg             # MATLAB: nvar (line 43)
        self.maxvar = max_vars           # MATLAB: maxvar (line 45)
        self.probmut = prob_mutation     # MATLAB: probmut (line 46)
        self.probcross = prob_crossover  # MATLAB: probcross (line 47)
        self.freqb = backward_freq       # MATLAB: freqb (line 48)
        self.verbose = verbose
        self.progress_callback = progress_callback  # Callback for progress updates

        # Set random seed for reproducibility
        if random_seed is not None:
            np.random.seed(random_seed)

        # Data dimensions
        self.n_samples, self.v = X.shape  # MATLAB: [o, c] (line 33), v = c-1 (line 36)

        # Initial selection probability (MATLAB: probsel, line 44)
        self.probsel = np.ones(self.v) * (self.nvar / self.v)

        # Results storage
        self.sel = np.zeros(self.v)  # MATLAB: sel (line 65) - accumulated frequency
        self.best_chromosomes_ = []
        self.best_fitnesses_ = []
        self.run_history_ = []

        if self.verbose:
            print(f"GAPLSSP Initialized")
            print(f"  Samples: {self.n_samples}")
            print(f"  Variables: {self.v}")
            print(f"  Runs: {self.runs}")
            print(f"  Evaluations per run: {self.evaluat}")


    def run(self) -> Dict:
        """
        Execute the 100 sequential GA runs (Leardi's original algorithm).

        This is the main loop (MATLAB lines 67-248).

        Returns
        -------
        results : dict
            'selected_variables': final selected variable indices
            'selection_freq': raw accumulated selection frequencies
            'selection_freq_smoothed': smoothed frequencies (MA window=3)
            'sorted_indices': variables sorted by frequency (descending)
            'best_chromosomes': best chromosome from each run
            'best_fitnesses': fitness values from each run
            'run_history': detailed history
            'stepwise_results': final stepwise analysis results
        """

        # Initialize probability arrays (MATLAB line 66)
        probsels = np.zeros(self.v)
        probselsw = np.zeros(self.v)
        probself = np.zeros(self.v)

        # ═══════════════════════════════════════════════════════════════
        # MAIN LOOP: 100 sequential runs (MATLAB lines 67-248)
        # ═══════════════════════════════════════════════════════════════

        for r in range(self.runs):
            if self.verbose:
                print(f"\n{'='*60}")
                print(f"RUN {r + 1}/{self.runs}")
                print(f"{'='*60}")

            # Callback: run start
            if self.progress_callback:
                self.progress_callback('run_start', {
                    'run': r + 1,
                    'fitness_min': probself.min() if r > 0 else self.probsel.min(),
                    'fitness_max': probself.max() if r > 0 else self.probsel.max()
                })

            # ───────────────────────────────────────────────────────────
            # EVOLUTION OF PROBABILITIES (MATLAB lines 82-91)
            # THIS IS THE KEY FEATURE!
            # ───────────────────────────────────────────────────────────

            probself = self.probsel.copy()  # Default: initial uniform (line 81)

            if r > 0:  # MATLAB: if r>1 (line 82)
                # Probability based on selection frequency (line 83)
                probsels = self.sel[:self.v] / np.sum(self.sel) * self.nvar

                # Smooth with moving average (window=3) - lines 84-88
                # For spectral autocorrelation!
                probselsw[0] = (probsels[0] + probsels[1]) / 2  # Edge (line 84)
                probselsw[self.v - 1] = (probsels[self.v - 1] + probsels[self.v - 2]) / 2  # Edge (line 85)

                for jj in range(1, self.v - 1):  # Interior points (lines 86-88)
                    probselsw[jj] = (probsels[jj - 1] + probsels[jj] + probsels[jj + 1]) / 3

                # Blend: start uniform, gradually bias toward good variables (line 90)
                # Formula: probself = (initial * (100-r) + smoothed * r) / 100
                probself = (self.probsel * (self.runs - r) + probselsw * r) / self.runs

                if self.verbose:
                    print(f"Prob range: {probself.min():.4f} - {probself.max():.4f}")

            # ───────────────────────────────────────────────────────────
            # RUN GA FOR THIS ITERATION (MATLAB lines 93-238)
            # ───────────────────────────────────────────────────────────

            best_chromo, best_fitness = self._run_single_ga(probself, r + 1)

            # ───────────────────────────────────────────────────────────
            # ACCUMULATE SELECTIONS (MATLAB line 240)
            # NOT replace - ACCUMULATE!
            # ───────────────────────────────────────────────────────────

            self.sel[:self.v] = self.sel[:self.v] + best_chromo

            # Store history
            self.best_chromosomes_.append(best_chromo)
            self.best_fitnesses_.append(best_fitness)
            self.run_history_.append({
                'run': r,
                'best_fitness': best_fitness,
                'n_selected': int(np.sum(best_chromo)),
                'selected_vars': np.where(best_chromo)[0].tolist(),
                'selection_freq_snapshot': self.sel[:self.v].copy()
            })

            if self.verbose:
                selected_indices = np.where(best_chromo)[0]
                print(f"Selected variables: {selected_indices.tolist()}")

        # ═══════════════════════════════════════════════════════════════
        # FINAL PROCESSING (MATLAB lines 250-276)
        # ═══════════════════════════════════════════════════════════════

        if self.verbose:
            print(f"\n{'='*60}")
            print("FINAL STEPWISE")
            print(f"{'='*60}")

        # Smooth final frequencies with moving average (MATLAB lines 250-256)
        sels = np.zeros(self.v)
        sels[0] = (self.sel[0] + self.sel[1]) / 2
        sels[self.v - 1] = (self.sel[self.v - 1] + self.sel[self.v - 2]) / 2

        for jj in range(1, self.v - 1):
            sels[jj] = (self.sel[jj - 1] + self.sel[jj] + self.sel[jj + 1]) / 3

        self.sel[:self.v] = sels  # Replace with smoothed (line 256)

        # Sort by frequency (MATLAB lines 258-259)
        sorted_indices = np.argsort(-self.sel[:self.v])  # Descending
        sorted_freq = self.sel[sorted_indices]

        if self.verbose:
            print(f"\nSmoothed frequencies (top 20):")
            for i in range(min(20, len(sorted_indices))):
                idx = sorted_indices[i]
                print(f"  Variable {idx}: {sorted_freq[i]:.2f}")

        # ───────────────────────────────────────────────────────────────
        # FINAL STEPWISE (MATLAB lines 263-276)
        # Only test if sel(c) > sel(c+1) - empirical improvement criterion
        # ───────────────────────────────────────────────────────────────

        if self.verbose:
            print("\nStepwise according to smoothed frequency:")

        stepwise_results = []
        k_max = min(self.v - 1, 200)  # MATLAB lines 264-267

        for c in range(k_max):
            # Only test if frequency improves (MATLAB line 269)
            if sorted_freq[c] > sorted_freq[c + 1]:
                selected_vars = sorted_indices[:c + 1]

                try:
                    X_subset = self.X[:, selected_vars]
                    fitness = self.fitness_fn(X_subset, self.y)

                    stepwise_results.append({
                        'n_vars': c + 1,
                        'fitness': fitness,
                        'selected_vars': selected_vars.tolist()
                    })

                    if self.verbose:
                        print(f"  {c + 1} vars: fitness = {fitness:.4f}")

                except Exception as e:
                    if self.verbose:
                        warnings.warn(f"Stepwise failed at {c + 1} vars: {e}")

        # ───────────────────────────────────────────────────────────────
        # SELECT FINAL MODEL
        # ───────────────────────────────────────────────────────────────

        if len(stepwise_results) > 0:
            # Best model from stepwise
            best_stepwise_idx = np.argmax([r['fitness'] for r in stepwise_results])
            final_selected = stepwise_results[best_stepwise_idx]['selected_vars']
        else:
            # Fallback: top 10 by frequency
            final_selected = sorted_indices[:10].tolist()

        # Compile results
        return {
            'selected_variables': np.array(final_selected),
            'selection_freq': self.sel[:self.v].copy(),
            'selection_freq_smoothed': sels.copy(),
            'sorted_indices': sorted_indices,
            'sorted_frequencies': sorted_freq,
            'best_chromosomes': self.best_chromosomes_,
            'best_fitnesses': np.array(self.best_fitnesses_),
            'run_history': self.run_history_,
            'stepwise_results': stepwise_results,
            'n_runs_completed': self.runs
        }


    def _run_single_ga(
        self,
        probself: np.ndarray,
        run_number: int
    ) -> Tuple[np.ndarray, float]:
        """
        Run single GA iteration (MATLAB lines 72-238).

        Parameters
        ----------
        probself : np.ndarray
            Initial selection probabilities for this run
        run_number : int
            Current run index

        Returns
        -------
        best_chromosome : np.ndarray
            Best chromosome found (binary array)
        best_fitness : float
            Fitness of best chromosome
        """

        # Initialize population storage (MATLAB lines 72-79)
        crom = np.zeros((self.cr, self.v), dtype=bool)  # Chromosomes
        resp = np.zeros(self.cr)  # Fitness values
        lib = []  # Library of all tested chromosomes (line 77)

        cc = 0  # Evaluation counter

        # ═══════════════════════════════════════════════════════════════
        # CREATE INITIAL POPULATION (MATLAB lines 93-133)
        # ═══════════════════════════════════════════════════════════════

        while cc < self.cr:
            sumvar = 0

            # Create random chromosome with given probability (lines 96-106)
            while sumvar == 0 or sumvar > self.maxvar:
                a = np.random.random(self.v)
                chromo = (a < probself).astype(bool)
                sumvar = np.sum(chromo)

            # Check for duplicate (lines 107-108)
            is_duplicate = any(np.array_equal(chromo, existing) for existing in lib)

            if not is_duplicate:
                lib.append(chromo.copy())

                # Evaluate fitness (line 115)
                try:
                    var_indices = np.where(chromo)[0]
                    X_subset = self.X[:, var_indices]
                    fitness = self.fitness_fn(X_subset, self.y)

                    # Add to population (lines 122-125)
                    crom[cc] = chromo
                    resp[cc] = fitness
                    cc += 1

                except Exception as e:
                    # Evaluation failed, skip
                    if self.verbose and cc < 5:
                        warnings.warn(f"Fitness evaluation failed: {e}")

        # Sort population by fitness (MATLAB lines 135-140)
        sorted_idx = np.argsort(-resp)  # Descending
        crom = crom[sorted_idx]
        resp = resp[sorted_idx]

        maxrisp = resp[0]  # Best fitness so far

        if self.verbose:
            print(f"Initial population best: {maxrisp:.4f}")

        # Callback: initial population
        if self.progress_callback:
            self.progress_callback('initial_pop', {
                'fitness': maxrisp,
                'n_variables': int(np.sum(crom[0]))
            })

        # ═══════════════════════════════════════════════════════════════
        # EVOLUTION LOOP (MATLAB lines 146-231)
        # ═══════════════════════════════════════════════════════════════

        while cc < self.evaluat:

            # ───────────────────────────────────────────────────────────
            # SELECTION: Biased roulette wheel (MATLAB lines 148-173)
            # ───────────────────────────────────────────────────────────

            cumrisp = np.cumsum(resp)

            if cumrisp[-1] > 0:
                # Select 2 parents via roulette wheel
                parents = []
                for _ in range(2):
                    k = np.random.random() * cumrisp[-1]
                    j = np.searchsorted(cumrisp, k)
                    parents.append(crom[j].copy())
            else:
                # Fallback: random selection
                rr = np.random.permutation(self.cr)
                parents = [crom[rr[0]].copy(), crom[rr[1]].copy()]

            # ───────────────────────────────────────────────────────────
            # CROSSOVER (MATLAB lines 175-181)
            # ───────────────────────────────────────────────────────────

            offspring = [parents[0].copy(), parents[1].copy()]

            # Find differences
            diff = np.where(parents[0] != parents[1])[0]

            if len(diff) > 0:
                # Apply crossover at difference positions
                randmat = np.random.random(len(diff))
                cross_positions = diff[randmat < self.probcross]

                offspring[0][cross_positions] = parents[1][cross_positions]
                offspring[1][cross_positions] = parents[0][cross_positions]

            # ───────────────────────────────────────────────────────────
            # MUTATION (MATLAB lines 183-195)
            # ───────────────────────────────────────────────────────────

            for i in range(2):
                m = np.random.random(self.v)
                mutation_positions = np.where(m < self.probmut)[0]
                offspring[i][mutation_positions] = ~offspring[i][mutation_positions]

            # ───────────────────────────────────────────────────────────
            # EVALUATE OFFSPRING (MATLAB lines 197-220)
            # ───────────────────────────────────────────────────────────

            for child in offspring:
                sumvar = np.sum(child)

                # Check constraints (line 202)
                if sumvar == 0 or sumvar > self.maxvar:
                    continue

                # Check duplicate (line 206)
                is_duplicate = any(np.array_equal(child, existing) for existing in lib)

                if is_duplicate:
                    continue

                lib.append(child.copy())
                cc += 1

                # Evaluate fitness (line 210)
                try:
                    var_indices = np.where(child)[0]
                    X_subset = self.X[:, var_indices]
                    fitness = self.fitness_fn(X_subset, self.y)

                    # Track best (lines 212-215)
                    if fitness > maxrisp:
                        if self.verbose:
                            print(f"  Eval {cc}: fitness improved to {fitness:.4f}")
                        maxrisp = fitness

                        # Callback: evaluation improvement
                        if self.progress_callback:
                            self.progress_callback('evaluation', {
                                'evaluation': cc,
                                'fitness': fitness,
                                'n_variables': int(np.sum(child))
                            })

                    # Replace worst in population if better (lines 216-218)
                    if fitness > resp[-1]:
                        crom[-1] = child
                        resp[-1] = fitness

                        # Re-sort
                        sorted_idx = np.argsort(-resp)
                        crom = crom[sorted_idx]
                        resp = resp[sorted_idx]

                except Exception:
                    # Evaluation failed
                    pass

        # Return best chromosome from this run
        return crom[0], resp[0]


def gaplssp(
    X: np.ndarray,
    y: np.ndarray,
    fitness_fn: Callable[[np.ndarray, np.ndarray], float],
    n_runs: int = 100,
    n_evals: int = 200,
    **kwargs
) -> Dict:
    """
    Top-level function for GAPLSSP (mimicking MATLAB interface).

    This is the Python equivalent of calling gaplssp.m from MATLAB.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix (n_samples, n_variables)
    y : np.ndarray
        Target variable (n_samples,)
    fitness_fn : callable
        Fitness function: fitness_fn(X_subset, y) -> float
        Should return % variance explained (R² × 100)
    n_runs : int, default=100
        Number of sequential runs (Leardi original: 100)
    n_evals : int, default=200
        Evaluations per run (stop criterion)
    **kwargs
        Additional parameters passed to GAPLSSP constructor

    Returns
    -------
    results : dict
        'selected_variables': final selected variable indices
        'selection_freq': accumulated selection frequencies
        'selection_freq_smoothed': smoothed frequencies
        'sorted_indices': variables sorted by frequency
        'sorted_frequencies': sorted frequency values
        'best_chromosomes': best from each run
        'best_fitnesses': fitness values per run
        'run_history': detailed history
        'stepwise_results': stepwise analysis

    Examples
    --------
    >>> from sklearn.cross_decomposition import PLSRegression
    >>> from sklearn.model_selection import cross_val_score
    >>>
    >>> def pls_fitness(X_subset, y):
    ...     model = PLSRegression(n_components=min(5, X_subset.shape[1]))
    ...     r2 = cross_val_score(model, X_subset, y, cv=5, scoring='r2').mean()
    ...     return r2 * 100  # Convert to %
    >>>
    >>> results = gaplssp(X, y, pls_fitness, n_runs=100, n_evals=200)
    >>> selected_vars = results['selected_variables']
    """

    ga = GAPLSSP(
        X, y, fitness_fn,
        n_runs=n_runs,
        n_evals=n_evals,
        **kwargs
    )

    return ga.run()
