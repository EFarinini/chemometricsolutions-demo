"""
D-Optimal Design Algorithm
Based on MATLAB dopt.m implementation using exchange algorithm.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
import warnings


def calculate_inflation_factors(selected_matrix: np.ndarray) -> np.ndarray:
    """
    Calculate Variance Inflation Factors (VIF) from selected experiments.

    Formula (from MATLAB):
    VIF_j = sum(xc_j^2) * [inv(X' @ X)]_jj
    where xc = X - mean(X) (centered data)

    Args:
        selected_matrix: Selected design matrix (n_experiments, n_variables)

    Returns:
        Array of VIFs for each variable

    Interpretation:
        VIF < 4: acceptable
        VIF 4-8: moderate concern
        VIF > 8: high multicollinearity
    """
    X = selected_matrix.copy()
    n, p = X.shape

    # Center the matrix
    X_centered = X - X.mean(axis=0)

    # Compute X'X
    XtX = X.T @ X

    # Use pseudoinverse for robustness (handles near-singular matrices)
    XtX_inv = np.linalg.pinv(XtX)

    # Calculate VIF for each variable
    vif = np.zeros(p)
    for j in range(p):
        # Sum of squared centered values
        sum_sq = np.sum(X_centered[:, j] ** 2)
        # VIF = sum(xc_j^2) * [(X'X)^-1]_jj
        vif[j] = sum_sq * XtX_inv[j, j]

    return vif


def exchange_algorithm_iteration(candidate_set: np.ndarray,
                                 current_indices: np.ndarray,
                                 remaining_indices: np.ndarray) -> Tuple[np.ndarray, float, bool]:
    """
    [DEPRECATED - No longer used in main doptimal_design() function]

    Single iteration of the exchange algorithm.
    This function is kept for backward compatibility but is not used in the new
    R CAT-based implementation which embeds the exchange logic directly in the
    main optimization loop.

    Args:
        candidate_set: ALL available candidates (n_candidates, n_vars)
        current_indices: Indices currently selected
        remaining_indices: Indices not yet selected

    Returns:
        Tuple (new_indices, determinant_value, swapped)
        - new_indices: updated selection after one swap
        - determinant_value: determinant of X'X after swap
        - swapped: whether a swap was performed
    """
    # Extract current design matrix
    X_current = candidate_set[current_indices]

    # Calculate X'X and its determinant
    XtX = X_current.T @ X_current

    try:
        det_current = np.linalg.det(XtX)
    except:
        det_current = 0

    if det_current <= 0:
        # Singular or near-singular matrix
        return current_indices, det_current, False

    # Calculate pseudoinverse (more robust than inverse)
    XtX_inv = np.linalg.pinv(XtX)

    # Calculate leverage (hat matrix diagonals) for current points
    # h = diag(X @ (X'X)^-1 @ X')
    leverage_in = np.sum((X_current @ XtX_inv) * X_current, axis=1)

    # Calculate leverage for remaining points
    X_remaining = candidate_set[remaining_indices]
    leverage_out = np.sum((X_remaining @ XtX_inv) * X_remaining, axis=1)

    # Find point with MINIMUM leverage in current set
    idx_min = np.argmin(leverage_in)
    min_leverage = leverage_in[idx_min]

    # Find point with MAXIMUM leverage in remaining set
    idx_max = np.argmax(leverage_out)
    max_leverage = leverage_out[idx_max]

    # Check if swap would improve design (max leverage out > min leverage in)
    if max_leverage > min_leverage:
        # Perform swap
        new_indices = current_indices.copy()
        swap_out_global = current_indices[idx_min]
        swap_in_global = remaining_indices[idx_max]

        new_indices[idx_min] = swap_in_global

        # Calculate new determinant
        X_new = candidate_set[new_indices]
        XtX_new = X_new.T @ X_new
        try:
            det_new = np.linalg.det(XtX_new)
        except:
            det_new = det_current

        return new_indices, det_new, True
    else:
        # No beneficial swap found
        return current_indices, det_current, False


def doptimal_design(candidate_matrix: np.ndarray,
                   min_experiments: int,
                   max_experiments: int,
                   n_trials: int = 5,
                   n_variables: int = None,
                   verbose: bool = True) -> dict:
    """
    Main D-Optimal design function using exchange algorithm.
    Based on R CAT implementation (DOE_doptimal.r).

    ALGORITHM:
    - For each experiment size n from min to max:
        - For each trial (random restart):
            - Random permutation of candidates
            - WHILE (miss_count < 5):  ← KEY: Exchange until 5 no-improvement iterations
                - Calculate det(X'X)
                - If det improved: update best
                - Else: increment miss_count
                - Always perform exchange: swap min-leverage from IN with max-leverage from OUT
                - Recalculate leverage for swapped set

    Args:
        candidate_matrix: Candidate points matrix (n_candidates, n_variables)
        min_experiments: Minimum number of experiments
        max_experiments: Maximum number of experiments
        n_trials: Number of random starts for optimization
        n_variables: Number of model coefficients (for log(M) calculation). If None, uses candidate_matrix columns
        verbose: Print progress messages

    Returns:
        Dictionary with complete optimization results
    """
    # Input validation and preparation
    if isinstance(candidate_matrix, pd.DataFrame):
        if 'Experiment_ID' in candidate_matrix.columns:
            X = candidate_matrix.drop('Experiment_ID', axis=1).values
        else:
            X = candidate_matrix.values
    else:
        X = candidate_matrix.copy().astype(float)

    n_candidates, n_cols = X.shape

    # Use provided n_variables (number of model coefficients) or default to matrix columns
    if n_variables is None:
        n_variables = n_cols

    # Validation
    if min_experiments < n_variables:
        raise ValueError(f"min_experiments ({min_experiments}) must be >= n_variables ({n_variables})")
    if max_experiments > n_candidates - 1:
        raise ValueError(f"max_experiments ({max_experiments}) must be <= n_candidates - 1 ({n_candidates - 1})")
    if min_experiments > max_experiments:
        raise ValueError(f"min_experiments ({min_experiments}) must be <= max_experiments ({max_experiments})")

    results_by_size = {}

    # ═══════════════════════════════════════════════════════════════════════
    # MAIN LOOP: For each experiment size
    # ═══════════════════════════════════════════════════════════════════════

    for n in range(min_experiments, max_experiments + 1):
        if verbose:
            print(f"\n{'='*60}")
            print(f"Optimizing design size: {n} experiments")
            print(f"{'='*60}")

        best_det = -np.inf
        best_indices = None
        best_trial = 0

        # ═══════════════════════════════════════════════════════════════════
        # TRIAL LOOP: Multiple random starts
        # ═══════════════════════════════════════════════════════════════════

        for trial in range(n_trials):
            # Random permutation of ALL candidate indices
            all_indices = np.arange(n_candidates)
            np.random.shuffle(all_indices)

            # Split into IN (selected) and OUT (remaining)
            in_indices = sorted(all_indices[:n])
            out_indices = sorted(all_indices[n:])

            # Track the best det found during this trial
            trial_det = -np.inf

            # ═══════════════════════════════════════════════════════════════
            # EXCHANGE LOOP: WHILE (miss_count < 5)
            # KEY DIFFERENCE: This is a WHILE loop, not FOR
            # ═══════════════════════════════════════════════════════════════

            miss_count = 0
            iteration = 0
            max_iterations = 1000  # Safety limit to prevent infinite loops

            while miss_count < 5 and iteration < max_iterations:
                iteration += 1

                # Extract current and remaining design
                X_in = X[in_indices]
                X_out = X[out_indices]

                # Calculate determinant
                try:
                    XtX = X_in.T @ X_in
                    det_value = np.linalg.det(XtX)
                except:
                    det_value = 0

                # Track best determinant for this trial
                if det_value > trial_det:
                    trial_det = det_value
                    if det_value > best_det:
                        best_det = det_value
                        best_indices = np.array(in_indices, copy=True)
                        best_trial = trial + 1
                    miss_count = 0  # RESET: We found improvement
                else:
                    miss_count += 1  # INCREMENT: No improvement

                # ═══════════════════════════════════════════════════════════
                # EXCHANGE STEP: Always perform exchange, regardless of improvement
                # ═══════════════════════════════════════════════════════════

                # Calculate leverage (hat matrix diagonals) using pseudoinverse
                XtX_inv = np.linalg.pinv(XtX)

                # Leverage for IN points
                leverage_in = np.sum((X_in @ XtX_inv) * X_in, axis=1)

                # Leverage for OUT points
                leverage_out = np.sum((X_out @ XtX_inv) * X_out, axis=1)

                # Find indices with MIN leverage IN and MAX leverage OUT
                idx_min_in = np.argmin(leverage_in)
                idx_max_out = np.argmax(leverage_out)

                # Get actual candidate indices
                point_to_remove = in_indices[idx_min_in]
                point_to_add = out_indices[idx_max_out]

                # Perform SWAP
                in_indices[idx_min_in] = point_to_add
                out_indices[idx_max_out] = point_to_remove

                # Keep sorted for consistency
                in_indices = sorted(in_indices)
                out_indices = sorted(out_indices)

            # End of WHILE loop

            if verbose and (trial + 1) % max(1, n_trials // 5) == 0:
                print(f"  Trial {trial + 1}/{n_trials}: best det = {trial_det:.2e}")

        # End of TRIAL loop

        # ═══════════════════════════════════════════════════════════════════
        # Store results for this design size
        # ═══════════════════════════════════════════════════════════════════

        if best_indices is not None:
            X_selected = X[best_indices]
            XtX = X_selected.T @ X_selected

            try:
                det_value = np.linalg.det(XtX)
                log_det = np.log10(det_value) if det_value > 0 else -np.inf

                # Calculate log(M) normalized
                # Formula: log10(det(X'X) / n^p)
                # Where p = n_variables (number of model coefficients), n = number of experiments
                log_M = log_det - n_variables * np.log10(n)

                # Calculate inflation factors
                inflation_factors = calculate_inflation_factors(X_selected)

                results_by_size[n] = {
                    'det': det_value,
                    'log_det': log_det,
                    'log_M': log_M,
                    'inflation_factors': inflation_factors,
                    'max_vif': np.max(inflation_factors) if len(inflation_factors) > 0 else np.inf,
                    'selected_indices': sorted(best_indices.tolist()),
                    'best_trial': best_trial,
                    'n_experiments': n
                }

                if verbose:
                    print(f"✓ Design {n}: det={det_value:.2e}, log(M)={log_M:.2f}, max_vif={np.max(inflation_factors):.2f}")

            except Exception as e:
                warnings.warn(f"Could not calculate metrics for n={n}: {e}")
                results_by_size[n] = {
                    'det': best_det,
                    'log_det': np.log10(best_det) if best_det > 0 else -np.inf,
                    'log_M': -np.inf,
                    'inflation_factors': np.array([]),
                    'max_vif': np.inf,
                    'selected_indices': sorted(best_indices.tolist()) if best_indices is not None else [],
                    'best_trial': best_trial,
                    'n_experiments': n
                }

    # ═══════════════════════════════════════════════════════════════════════
    # Find best overall design (based on log_M)
    # ═══════════════════════════════════════════════════════════════════════

    if not results_by_size:
        raise RuntimeError("No valid designs found")

    best_size = max(
        results_by_size.keys(),
        key=lambda k: results_by_size[k]['log_M'] if np.isfinite(results_by_size[k]['log_M']) else -np.inf
    )

    best_design = results_by_size[best_size].copy()

    # Prepare output
    output = {
        'candidate_matrix': X,
        'results_by_size': results_by_size,
        'best_design': best_design,
        'selected_indices': best_design['selected_indices']
    }

    if verbose:
        print(f"\n{'='*60}")
        print(f"FINAL RESULT")
        print(f"{'='*60}")
        print(f"Best design size: {best_size} experiments")
        print(f"Determinant: {best_design['det']:.4e}")
        print(f"log(M): {best_design['log_M']:.4f}")
        print(f"Max VIF: {best_design['max_vif']:.2f}")
        print(f"Trial: {best_design['best_trial']}")
        print(f"Selected indices: {best_design['selected_indices']}")
        print(f"{'='*60}\n")

    return output


def format_doptimal_results(results: dict) -> pd.DataFrame:
    """
    Format D-optimal results into a summary DataFrame.

    Args:
        results: Output from doptimal_design()

    Returns:
        DataFrame with summary metrics for each design size
    """
    rows = []
    for n, data in results['results_by_size'].items():
        rows.append({
            'N_Experiments': n,
            'Determinant': data['det'],
            'log(Determinant)': data['log_det'],
            'log(M)': data['log_M'],
            'Max_VIF': data['max_vif'],
            'Best_Trial': data['best_trial']
        })

    df = pd.DataFrame(rows)
    df = df.sort_values('N_Experiments').reset_index(drop=True)

    return df
