"""
Example: Using GAPLSSP Original (Leardi's Algorithm)
====================================================

This example shows how to use the faithful Python translation
of Leardi's gaplssp.m

Compare with current GA engine to see differences.
"""

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_regression

# Import both engines
from ga_variable_selection.gaplssp_original import gaplssp, GAPLSSP
from ga_variable_selection.ga_engine import GeneticAlgorithm


def create_test_data():
    """Create synthetic spectral-like data."""
    np.random.seed(42)

    # Create data where only first 10 variables are important
    X, y = make_regression(
        n_samples=100,
        n_features=200,
        n_informative=10,
        noise=5.0,
        random_state=42
    )

    return X, y


def pls_fitness(X_subset, y):
    """
    Fitness function for PLS regression.

    Returns % variance explained (R² × 100) - Leardi's format.
    """
    try:
        n_components = min(5, X_subset.shape[1], X_subset.shape[0] - 1)

        if n_components < 1:
            return 0.0

        model = PLSRegression(n_components=n_components)
        r2 = cross_val_score(model, X_subset, y, cv=5, scoring='r2').mean()

        return max(0, r2 * 100)  # Convert to %
    except:
        return 0.0


def compare_engines():
    """Compare GAPLSSP original vs current GA engine."""

    print("="*70)
    print("COMPARISON: GAPLSSP Original vs Current GA Engine")
    print("="*70)

    # Create data
    X, y = create_test_data()
    print(f"\nData: {X.shape[0]} samples × {X.shape[1]} variables")

    # ══════════════════════════════════════════════════════════════
    # 1. GAPLSSP Original (Leardi's faithful implementation)
    # ══════════════════════════════════════════════════════════════

    print("\n" + "─"*70)
    print("1. GAPLSSP ORIGINAL (Leardi 1998)")
    print("─"*70)

    results_gaplssp = gaplssp(
        X, y, pls_fitness,
        n_runs=50,  # Reduced from 100 for speed (still sequential!)
        n_evals=100,
        population_size=30,
        cv_groups=5,
        nvar_avg=5,
        max_vars=30,
        verbose=False
    )

    print(f"\nResults:")
    print(f"  Selected variables: {len(results_gaplssp['selected_variables'])}")
    print(f"  Variables: {results_gaplssp['selected_variables'][:10]}...")
    print(f"  Best fitness (final): {max(results_gaplssp['best_fitnesses']):.2f}%")
    print(f"  Selection freq range: {results_gaplssp['selection_freq'].min():.0f} - {results_gaplssp['selection_freq'].max():.0f}")
    print(f"  Smoothed freq range: {results_gaplssp['selection_freq_smoothed'].min():.2f} - {results_gaplssp['selection_freq_smoothed'].max():.2f}")
    print(f"  Stepwise models tested: {len(results_gaplssp['stepwise_results'])}")

    # ══════════════════════════════════════════════════════════════
    # 2. Current GA Engine (independent runs)
    # ══════════════════════════════════════════════════════════════

    print("\n" + "─"*70)
    print("2. CURRENT GA ENGINE (Independent runs)")
    print("─"*70)

    # Fitness wrapper for current engine
    def fitness_wrapper(X_subset, y_vals, selected_indices):
        return pls_fitness(X_subset, y_vals)

    ga_current = GeneticAlgorithm(
        dataset=X,
        fitness_fn=fitness_wrapper,
        config={
            'runs': 10,  # Internal generations
            'population_size': 30,
            'evaluations': 100,
            'cv_groups': 5,
            'mutation_prob': 0.01,
            'crossover_prob': 0.5
        },
        y=y,
        random_seed=42
    )

    results_current = ga_current.run()

    print(f"\nResults:")
    print(f"  Selected variables: {len(results_current['selected_variables'])}")
    print(f"  Variables: {results_current['selected_variables'][:10]}...")
    print(f"  Best fitness: {results_current['best_fitness']:.2f}")
    print(f"  Selection freq range: {results_current['selection_frequency'].min():.0f} - {results_current['selection_frequency'].max():.0f}")
    print(f"  Runs completed: {results_current['n_runs_completed']}")

    # ══════════════════════════════════════════════════════════════
    # 3. COMPARISON SUMMARY
    # ══════════════════════════════════════════════════════════════

    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)

    print("\n┌─────────────────────────┬──────────────┬──────────────┐")
    print("│ Metric                  │ GAPLSSP      │ Current      │")
    print("├─────────────────────────┼──────────────┼──────────────┤")
    print(f"│ Selected variables      │ {len(results_gaplssp['selected_variables']):12d} │ {len(results_current['selected_variables']):12d} │")
    print(f"│ Best fitness            │ {max(results_gaplssp['best_fitnesses']):12.2f} │ {results_current['best_fitness']:12.2f} │")
    print(f"│ Runs completed          │ {results_gaplssp['n_runs_completed']:12d} │ {results_current['n_runs_completed']:12d} │")
    print(f"│ Smoothing applied       │ {'Yes (MA-3)':>12} │ {'No':>12} │")
    print(f"│ Probability evolution   │ {'Yes':>12} │ {'No':>12} │")
    print(f"│ Stepwise models         │ {len(results_gaplssp['stepwise_results']):12d} │ {'N/A':>12} │")
    print("└─────────────────────────┴──────────────┴──────────────┘")

    # Check overlap
    gaplssp_set = set(results_gaplssp['selected_variables'])
    current_set = set(results_current['selected_variables'])
    overlap = gaplssp_set & current_set

    print(f"\nVariable selection overlap: {len(overlap)} variables")
    print(f"  GAPLSSP only: {len(gaplssp_set - current_set)}")
    print(f"  Current only: {len(current_set - gaplssp_set)}")

    # Show top 10 by frequency from GAPLSSP
    print("\n" + "─"*70)
    print("GAPLSSP Top 10 Variables (by smoothed frequency)")
    print("─"*70)

    sorted_idx = results_gaplssp['sorted_indices']
    sorted_freq = results_gaplssp['sorted_frequencies']

    print("\n┌──────┬───────────┬──────────────┬──────────┐")
    print("│ Rank │ Variable  │ Smoothed Freq│ Selected │")
    print("├──────┼───────────┼──────────────┼──────────┤")

    for i in range(min(10, len(sorted_idx))):
        var_idx = sorted_idx[i]
        freq = sorted_freq[i]
        selected = "✓" if var_idx in results_gaplssp['selected_variables'] else ""

        print(f"│ {i+1:4d} │ {var_idx:9d} │ {freq:12.2f} │ {selected:^8} │")

    print("└──────┴───────────┴──────────────┴──────────┘")


def demonstrate_probability_evolution():
    """Show how probability evolves across runs (GAPLSSP feature)."""

    print("\n" + "="*70)
    print("DEMONSTRATION: Probability Evolution in GAPLSSP")
    print("="*70)

    X, y = create_test_data()

    # Use class interface to access internals
    ga = GAPLSSP(
        X, y, pls_fitness,
        n_runs=20,  # Just 20 for demo
        n_evals=50,
        verbose=False
    )

    print("\nTracking probability evolution across 20 runs...")
    print("\n┌──────┬───────────────┬───────────────┬──────────────┐")
    print("│ Run  │ Prob Min      │ Prob Max      │ Prob Range   │")
    print("├──────┼───────────────┼───────────────┼──────────────┤")

    # We'd need to modify GAPLSSP to expose probself at each run
    # For now, just run and show final results

    results = ga.run()

    # Show selection frequency evolution
    print("\n" + "─"*70)
    print("Selection Frequency Evolution (every 5 runs)")
    print("─"*70)

    for i, run_info in enumerate(results['run_history']):
        if (i + 1) % 5 == 0 or i == 0:
            freq_snap = run_info['selection_freq_snapshot']
            print(f"\nAfter run {run_info['run'] + 1:2d}:")
            print(f"  Freq range: {freq_snap.min():.0f} - {freq_snap.max():.0f}")
            print(f"  Variables selected ≥5 times: {np.sum(freq_snap >= 5)}")
            print(f"  Variables selected ≥10 times: {np.sum(freq_snap >= 10)}")


if __name__ == "__main__":
    print("\n" + "█"*70)
    print("█" + " "*68 + "█")
    print("█" + "  GAPLSSP vs Current GA Engine - Comparison Demo".center(68) + "█")
    print("█" + " "*68 + "█")
    print("█"*70)

    # Run comparison
    compare_engines()

    # Show probability evolution
    demonstrate_probability_evolution()

    print("\n" + "█"*70)
    print("█" + " "*68 + "█")
    print("█" + "  Demo Complete!".center(68) + "█")
    print("█" + " "*68 + "█")
    print("█"*70 + "\n")
