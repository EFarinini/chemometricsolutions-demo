"""
D-Optimal Design by Sequential Addition
Extends existing experimental designs by adding optimal candidates.
Based on R CAT implementation (DOE_doptadd.r).
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
import warnings


def calculate_model_matrix(data: pd.DataFrame,
                          include_intercept: bool = True,
                          interactions_dict: Dict[str, bool] = None,
                          quadratic_dict: Dict[str, bool] = None) -> np.ndarray:
    """
    Build model matrix (design matrix) with specified terms.

    Args:
        data: DataFrame with variables (columns)
        include_intercept: Include intercept column (all 1s)
        interactions_dict: Dict mapping "i:j" -> True/False for X_i * X_j
        quadratic_dict: Dict mapping "i" -> True/False for X_i^2

    Returns:
        Model matrix as numpy array
    """
    if isinstance(data, pd.DataFrame):
        # Exclude Experiment_ID if present
        if 'Experiment_ID' in data.columns:
            X = data.drop('Experiment_ID', axis=1).values
            var_names = [c for c in data.columns if c != 'Experiment_ID']
        else:
            X = data.values
            var_names = list(data.columns)
    else:
        X = data.copy()
        var_names = [f"X{i}" for i in range(X.shape[1])]

    n_rows, n_vars = X.shape

    # Start with linear terms
    model_matrix = X.copy()

    # Add intercept as FIRST column if requested
    if include_intercept:
        model_matrix = np.column_stack([np.ones(n_rows), model_matrix])

    # Add interaction terms
    if interactions_dict:
        for key, include in interactions_dict.items():
            if include:
                i, j = map(int, key.split(':'))
                interaction = X[:, i] * X[:, j]
                model_matrix = np.column_stack([model_matrix, interaction])

    # Add quadratic terms
    if quadratic_dict:
        for key, include in quadratic_dict.items():
            if include:
                i = int(key)
                quadratic = X[:, i] ** 2
                model_matrix = np.column_stack([model_matrix, quadratic])

    return model_matrix


def calculate_inflation_factors_addition(X: np.ndarray) -> np.ndarray:
    """
    Calculate Variance Inflation Factors (VIF) for model matrix.

    Args:
        X: Model matrix (n_experiments, n_coefficients)

    Returns:
        Array of VIFs for each coefficient
    """
    n, p = X.shape

    # Center the matrix
    X_centered = X - X.mean(axis=0)

    # Compute X'X
    XtX = X.T @ X

    # Use pseudoinverse for robustness
    XtX_inv = np.linalg.pinv(XtX)

    # Calculate VIF for each variable
    vif = np.zeros(p)
    for j in range(p):
        sum_sq = np.sum(X_centered[:, j] ** 2)
        vif[j] = sum_sq * XtX_inv[j, j]

    return vif


def doptimal_by_addition(performed_experiments,
                        candidate_matrix,
                        min_to_add: int = 1,
                        max_to_add: int = 5,
                        n_trials: int = 10,
                        n_variables: int = None,
                        include_intercept: bool = True,
                        interactions_dict: Dict[str, bool] = None,
                        quadratic_dict: Dict[str, bool] = None,
                        verbose: bool = True) -> dict:
    """
    D-Optimal Design by Sequential Addition.

    **TWO MODES:**

    MODE 1: ENCODED MATRICES (Recommended - matches Tab 2 architecture)
        - performed_experiments and candidate_matrix are ALREADY ENCODED
        - Pass as numpy arrays (.values)
        - Must provide n_variables (number of coefficients)
        - Do NOT use interactions_dict/quadratic_dict (already in matrix)

    MODE 2: RAW DATA WITH MODEL BUILDING (Legacy)
        - performed_experiments and candidate_matrix are RAW
        - Pass as DataFrames
        - Provide interactions_dict and quadratic_dict
        - Function will build model matrix internally

    Algorithm (from R CAT DOE_doptadd.r):
    - For each n_add from min_to_add to max_to_add:
        - For each trial (random restart):
            - Random selection of n_add candidates
            - WHILE (miss_count < 5):  ← KEY: Exchange until 5 no-improvement iterations
                - X_combined = [performed; selected_candidates]
                - Calculate det(X'X)
                - If det improved: update best
                - Else: increment miss_count
                - Always perform exchange: swap min-leverage from IN with max-leverage from OUT
                - Recalculate leverage for swapped set

    Args:
        performed_experiments: Already performed experiments (numpy array or DataFrame)
        candidate_matrix: Available candidate points (numpy array or DataFrame)
        min_to_add: Minimum number of experiments to add
        max_to_add: Maximum number of experiments to add
        n_trials: Number of random starts for optimization
        n_variables: Number of model coefficients (REQUIRED for MODE 1)
        include_intercept: Include intercept in model (MODE 2 only)
        interactions_dict: Dict mapping "i:j" -> True/False (MODE 2 only)
        quadratic_dict: Dict mapping "i" -> True/False (MODE 2 only)
        verbose: Print progress messages

    Returns:
        Dictionary with complete optimization results
    """
    # ═══════════════════════════════════════════════════════════════════
    # STEP 1: Prepare data - DETECT MODE
    # ═══════════════════════════════════════════════════════════════════

    # MODE DETECTION: Check if n_variables was provided (MODE 1) or not (MODE 2)
    is_mode1 = (n_variables is not None)

    if is_mode1:
        # MODE 1: ENCODED MATRICES (matrices already have model terms)
        if verbose:
            print("MODE 1: Using pre-encoded design matrices")

        # Accept numpy arrays or DataFrames
        if isinstance(performed_experiments, pd.DataFrame):
            X_performed_model = performed_experiments.values
        else:
            X_performed_model = performed_experiments.copy()

        if isinstance(candidate_matrix, pd.DataFrame):
            X_candidates = candidate_matrix.values
        else:
            X_candidates = candidate_matrix.copy()

        n_performed, n_coefficients = X_performed_model.shape
        n_candidates, n_coefficients_cand = X_candidates.shape

        if n_coefficients != n_coefficients_cand:
            raise ValueError(f"Column mismatch: performed has {n_coefficients}, candidates have {n_coefficients_cand}")

        # In MODE 1, n_variables should match matrix columns
        if n_variables != n_coefficients:
            raise ValueError(f"n_variables ({n_variables}) != matrix columns ({n_coefficients})")

        n_vars = n_coefficients  # For compatibility with rest of code

    else:
        # MODE 2: RAW DATA WITH MODEL BUILDING (legacy behavior)
        if verbose:
            print("MODE 2: Building model matrices from raw data")

        # Convert to numpy arrays (exclude Experiment_ID if present)
        if isinstance(performed_experiments, pd.DataFrame):
            if 'Experiment_ID' in performed_experiments.columns:
                X_performed = performed_experiments.drop('Experiment_ID', axis=1).values
            else:
                X_performed = performed_experiments.values
        else:
            X_performed = performed_experiments.copy()

        if isinstance(candidate_matrix, pd.DataFrame):
            if 'Experiment_ID' in candidate_matrix.columns:
                X_candidates = candidate_matrix.drop('Experiment_ID', axis=1).values
            else:
                X_candidates = candidate_matrix.values
        else:
            X_candidates = candidate_matrix.copy()

        n_performed, n_vars = X_performed.shape
        n_candidates, n_vars_cand = X_candidates.shape

        if n_vars != n_vars_cand:
            raise ValueError(f"Variable mismatch: performed has {n_vars}, candidates have {n_vars_cand}")

        # Build model matrix for performed experiments
        X_performed_model = calculate_model_matrix(
            pd.DataFrame(X_performed),
            include_intercept=include_intercept,
            interactions_dict=interactions_dict,
            quadratic_dict=quadratic_dict
        )

        n_coefficients = X_performed_model.shape[1]

    # Common validation
    if max_to_add > n_candidates:
        raise ValueError(f"max_to_add ({max_to_add}) > n_candidates ({n_candidates})")
    if min_to_add < 0:
        raise ValueError(f"min_to_add must be >= 0")
    if min_to_add > max_to_add:
        raise ValueError(f"min_to_add ({min_to_add}) > max_to_add ({max_to_add})")

    if verbose:
        print(f"\n{'='*70}")
        print(f"D-OPTIMAL BY ADDITION")
        print(f"{'='*70}")
        print(f"Performed experiments: {n_performed}")
        print(f"Candidate experiments: {n_candidates}")
        print(f"Variables: {n_vars}")
        print(f"Model coefficients: {n_coefficients}")
        print(f"Range to add: {min_to_add} to {max_to_add}")
        print(f"{'='*70}\n")

    results_by_size = {}

    # ═══════════════════════════════════════════════════════════════════
    # MAIN LOOP: For each n_add
    # ═══════════════════════════════════════════════════════════════════

    for n_add in range(min_to_add, max_to_add + 1):
        if verbose:
            print(f"\n{'='*70}")
            print(f"Adding {n_add} experiments")
            print(f"{'='*70}")

        best_det = -np.inf
        best_indices = None
        best_trial = 0

        # ═══════════════════════════════════════════════════════════════
        # TRIAL LOOP: Multiple random starts
        # ═══════════════════════════════════════════════════════════════

        for trial in range(n_trials):
            # Random permutation of ALL candidate indices
            all_indices = np.arange(n_candidates)
            np.random.shuffle(all_indices)

            # Split into IN (selected to add) and OUT (remaining candidates)
            in_indices = sorted(all_indices[:n_add].tolist())
            out_indices = sorted(all_indices[n_add:].tolist())

            # Track the best det found during this trial
            trial_det = -np.inf

            # ═══════════════════════════════════════════════════════════
            # EXCHANGE LOOP: WHILE (miss_count < 5)
            # KEY DIFFERENCE: This is a WHILE loop, not FOR
            # ═══════════════════════════════════════════════════════════

            miss_count = 0
            iteration = 0
            max_iterations = 1000  # Safety limit

            while miss_count < 5 and iteration < max_iterations:
                iteration += 1

                # Extract selected candidates
                X_selected = X_candidates[in_indices]
                X_remaining = X_candidates[out_indices]

                # Build model matrices (MODE 1 vs MODE 2)
                if is_mode1:
                    # MODE 1: Already encoded - use directly
                    X_selected_model = X_selected
                else:
                    # MODE 2: Build model from raw data
                    X_selected_model = calculate_model_matrix(
                        pd.DataFrame(X_selected),
                        include_intercept=include_intercept,
                        interactions_dict=interactions_dict,
                        quadratic_dict=quadratic_dict
                    )

                # Combine with performed experiments
                X_combined = np.vstack([X_performed_model, X_selected_model])

                # Calculate determinant
                try:
                    XtX = X_combined.T @ X_combined
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

                # ═══════════════════════════════════════════════════════
                # EXCHANGE STEP: Always perform exchange
                # ═══════════════════════════════════════════════════════

                # Calculate leverage (hat matrix diagonals) using pseudoinverse
                XtX_inv = np.linalg.pinv(XtX)

                # Leverage for selected candidates (IN)
                leverage_in = np.sum((X_selected_model @ XtX_inv) * X_selected_model, axis=1)

                # Leverage for remaining candidates (OUT)
                if is_mode1:
                    # MODE 1: Already encoded - use directly
                    X_remaining_model = X_remaining
                else:
                    # MODE 2: Build model from raw data
                    X_remaining_model = calculate_model_matrix(
                        pd.DataFrame(X_remaining),
                        include_intercept=include_intercept,
                        interactions_dict=interactions_dict,
                        quadratic_dict=quadratic_dict
                    )
                leverage_out = np.sum((X_remaining_model @ XtX_inv) * X_remaining_model, axis=1)

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

        # ═══════════════════════════════════════════════════════════════
        # Store results for this n_add
        # ═══════════════════════════════════════════════════════════════

        if best_indices is not None:
            # Build final combined matrix
            X_best_selected = X_candidates[best_indices]

            if is_mode1:
                # MODE 1: Already encoded - use directly
                X_best_model = X_best_selected
            else:
                # MODE 2: Build model from raw data
                X_best_model = calculate_model_matrix(
                    pd.DataFrame(X_best_selected),
                    include_intercept=include_intercept,
                    interactions_dict=interactions_dict,
                    quadratic_dict=quadratic_dict
                )

            X_final = np.vstack([X_performed_model, X_best_model])

            XtX = X_final.T @ X_final

            try:
                det_value = np.linalg.det(XtX)
                log_det = np.log10(det_value) if det_value > 0 else -np.inf

                # Calculate log(M) normalized
                # Formula: log10(det(X'X) / n_total^p)
                # Where p = n_coefficients, n_total = n_performed + n_add
                n_total = n_performed + n_add
                log_M = log_det - n_coefficients * np.log10(n_total)

                # Calculate inflation factors
                inflation_factors = calculate_inflation_factors_addition(X_final)

                results_by_size[n_add] = {
                    'det': det_value,
                    'log_det': log_det,
                    'log_M': log_M,
                    'inflation_factors': inflation_factors,
                    'max_vif': np.max(inflation_factors) if len(inflation_factors) > 0 else np.inf,
                    'added_indices': sorted(best_indices.tolist()),
                    'best_trial': best_trial,
                    'n_add': n_add,
                    'n_total': n_total
                }

                if verbose:
                    print(f"✓ Add {n_add}: det={det_value:.2e}, log(M)={log_M:.2f}, max_vif={np.max(inflation_factors):.2f}")

            except Exception as e:
                warnings.warn(f"Could not calculate metrics for n_add={n_add}: {e}")
                results_by_size[n_add] = {
                    'det': best_det,
                    'log_det': np.log10(best_det) if best_det > 0 else -np.inf,
                    'log_M': -np.inf,
                    'inflation_factors': np.array([]),
                    'max_vif': np.inf,
                    'added_indices': sorted(best_indices.tolist()) if best_indices is not None else [],
                    'best_trial': best_trial,
                    'n_add': n_add,
                    'n_total': n_performed + n_add
                }

    # ═══════════════════════════════════════════════════════════════════
    # Find best overall n_add (based on log_M)
    # ═══════════════════════════════════════════════════════════════════

    if not results_by_size:
        raise RuntimeError("No valid designs found")

    best_n_to_add = max(
        results_by_size.keys(),
        key=lambda k: results_by_size[k]['log_M'] if np.isfinite(results_by_size[k]['log_M']) else -np.inf
    )

    best_design = results_by_size[best_n_to_add].copy()

    # Prepare output
    output = {
        'n_performed': n_performed,
        'n_candidates': n_candidates,
        'n_variables': n_vars,
        'n_coefficients': n_coefficients,
        'results_by_size': results_by_size,
        'best_n_to_add': best_n_to_add,
        'best_design': best_design
    }

    if verbose:
        print(f"\n{'='*70}")
        print(f"FINAL RESULT")
        print(f"{'='*70}")
        print(f"Best n_add: {best_n_to_add} experiments")
        print(f"Total experiments: {best_design['n_total']} (performed: {n_performed} + added: {best_n_to_add})")
        print(f"Determinant: {best_design['det']:.4e}")
        print(f"log(M): {best_design['log_M']:.4f}")
        print(f"Max VIF: {best_design['max_vif']:.2f}")
        print(f"Trial: {best_design['best_trial']}")
        print(f"Indices to add: {best_design['added_indices']}")
        print(f"{'='*70}\n")

    return output


def format_addition_results(results: dict) -> pd.DataFrame:
    """
    Format D-optimal by addition results into a summary DataFrame.

    Args:
        results: Output from doptimal_by_addition()

    Returns:
        DataFrame with summary metrics for each n_add
    """
    rows = []
    for n_add, data in results['results_by_size'].items():
        rows.append({
            'N_to_Add': n_add,
            'N_Total': data['n_total'],
            'Determinant': data['det'],
            'log(Determinant)': data['log_det'],
            'log(M)': data['log_M'],
            'Max_VIF': data['max_vif'],
            'Best_Trial': data['best_trial']
        })

    df = pd.DataFrame(rows)
    df = df.sort_values('N_to_Add').reset_index(drop=True)

    return df


def extract_added_experiments(candidate_matrix: pd.DataFrame,
                              added_indices: List[int]) -> pd.DataFrame:
    """
    Extract the experiments to add from candidate matrix.

    Args:
        candidate_matrix: Full candidate DataFrame
        added_indices: List of indices to extract

    Returns:
        DataFrame with selected experiments
    """
    if isinstance(candidate_matrix, pd.DataFrame):
        added_df = candidate_matrix.iloc[added_indices].copy()
    else:
        added_df = pd.DataFrame(candidate_matrix[added_indices])

    # Reset index and add Experiment_ID if not present
    added_df = added_df.reset_index(drop=True)

    if 'Experiment_ID' not in added_df.columns:
        added_df.insert(0, 'Experiment_ID', range(1, len(added_df) + 1))

    return added_df
