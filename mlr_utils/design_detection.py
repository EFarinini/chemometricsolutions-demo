"""
Design Detection Utilities for MLR/DoE
Shared functions for detecting replicates, central points, and pseudo-central points

This module contains shared detection functions used by:
- mlr_doe.py (single-response MLR/DOE)
- multi_doe_page.py (multi-response MLR/DOE)
- mlr_utils/model_computation.py
- mlr_utils/model_computation_multidoe.py
"""

import numpy as np
import pandas as pd


def detect_replicates(X_data, y_data, tolerance=1e-10):
    """
    Detect experimental replicates in the design matrix
    Returns replicate statistics or None if no replicates found

    Args:
        X_data: DataFrame with predictor variables
        y_data: Series with response variable
        tolerance: float, tolerance for considering points identical

    Returns:
        dict with replicate statistics or None if no replicates found
    """
    n_samples = len(X_data)
    X_values = X_data.values
    y_values = y_data.values

    replicate_groups = []
    used_indices = set()

    for i in range(n_samples):
        if i in used_indices:
            continue
        group = [i]
        used_indices.add(i)
        for j in range(i + 1, n_samples):
            if j in used_indices:
                continue
            if np.allclose(X_values[i], X_values[j], atol=tolerance):
                group.append(j)
                used_indices.add(j)
        if len(group) > 1:
            replicate_groups.append(group)

    if not replicate_groups:
        return None

    # Calculate pooled standard deviation
    variance_sum = 0
    dof_sum = 0
    group_stats = []

    for group in replicate_groups:
        y_group = y_values[group]
        n_reps = len(group)
        mean_y = np.mean(y_group)
        if n_reps > 1:
            var_y = np.var(y_group, ddof=1)
            std_y = np.sqrt(var_y)
            dof = n_reps - 1
            variance_sum += var_y * dof
            dof_sum += dof
            group_stats.append({
                'indices': group,
                'n_replicates': n_reps,
                'mean': mean_y,
                'std': std_y,
                'variance': var_y,
                'dof': dof
            })

    if dof_sum > 0:
        pooled_variance = variance_sum / dof_sum
        pooled_std = np.sqrt(pooled_variance)
    else:
        return None

    return {
        'n_replicate_groups': len(replicate_groups),
        'total_replicates': sum(len(g) for g in replicate_groups),
        'group_stats': group_stats,
        'pooled_std': pooled_std,
        'pooled_variance': pooled_variance,
        'pooled_dof': dof_sum
    }


def detect_central_points(X_data, tolerance=1e-6):
    """
    Detect central points in the design matrix
    A central point has ALL variables at their central value (coded: 0, natural: midpoint)
    Returns list of central point indices

    Args:
        X_data: DataFrame with predictor variables
        tolerance: float, tolerance for considering values at center

    Returns:
        list: Indices of central points
    """
    central_indices = []
    X_values = X_data.values

    for i in range(len(X_data)):
        # Method 1: Check for ALL zeros (coded variables)
        if np.allclose(X_values[i], 0, atol=tolerance):
            central_indices.append(i)
            continue

        # Method 2: Check if ALL values are at midpoint of their ranges
        is_central = True
        for j in range(X_data.shape[1]):
            col_values = X_values[:, j]
            unique_vals = np.unique(col_values)
            if len(unique_vals) <= 1:
                continue
            if set(unique_vals).issubset({-1, 0, 1}):
                if not np.isclose(X_values[i, j], 0, atol=tolerance):
                    is_central = False
                    break
            else:
                min_val = col_values.min()
                max_val = col_values.max()
                mid_val = (min_val + max_val) / 2
                if not np.isclose(X_values[i, j], mid_val, atol=tolerance):
                    is_central = False
                    break

        if is_central:
            central_indices.append(i)

    return central_indices


def detect_pseudo_central_points(X, design_analysis=None, tolerance=0.01):
    """
    Detect pseudo-central points based on PATTERN, not variable type.

    Pseudo-central points have:
    - Some (but not all) coordinates ≈ 0
    - Appear in replicates (same point repeated)

    Examples:
    - (0, 0, 1) in 3D design → pseudo-central (2 coords at 0)
    - (0, 0, 0) → TRUE central (excluded separately)
    - (1, -1, 0) → pseudo-central (1 coord at 0)

    Args:
        X: DataFrame with experimental design
        design_analysis: unused (kept for compatibility)
        tolerance: float, how close to 0 counts as "at center" (default 0.01)

    Returns:
        list: Indices of pseudo-central points
    """
    pseudo_central_indices = []

    if X.empty or len(X) < 2:
        return []

    # Identify true-central points (ALL coords ≈ 0)
    true_central_mask = np.all(np.abs(X.values) < tolerance, axis=1)

    # Identify points with SOME (but not all) coords ≈ 0
    coords_at_zero = np.abs(X.values) < tolerance
    num_zeros_per_row = np.sum(coords_at_zero, axis=1)

    # Pseudo-central = at least 1 zero, but not all zeros (not true-central)
    pseudo_central_mask = (num_zeros_per_row > 0) & (num_zeros_per_row < X.shape[1])
    pseudo_central_candidates = np.where(pseudo_central_mask)[0]

    # Further filter: keep only points that REPEAT (validation/variance points)
    if len(pseudo_central_candidates) > 0:
        for idx in pseudo_central_candidates:
            point = X.iloc[idx].values

            # Count how many points are identical (within tolerance)
            distances = np.linalg.norm(X.values - point, axis=1)
            num_replicates = np.sum(distances < tolerance)

            # If point appears 2+ times → it's pseudo-central
            if num_replicates >= 2:
                pseudo_central_indices.append(idx)

    # Remove duplicates and sort
    pseudo_central_indices = sorted(list(set(pseudo_central_indices)))

    return pseudo_central_indices
