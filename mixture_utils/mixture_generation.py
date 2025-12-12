"""
Mixture Design Generation Module
Functions for creating simplex centroid designs, applying constraints,
and pseudo-component transformations

Equivalent to R scripts: DOE_model_computation_mixt.r design generation
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from itertools import combinations
from scipy.optimize import minimize
import warnings


# ============================================================================
# COMPONENT NAMES AND CONSTANTS
# ============================================================================

COMPONENT_NAMES = ['X1', 'X2', 'X3', 'X4', 'X5']
MIN_COMPONENTS = 2
MAX_COMPONENTS = 5


# ============================================================================
# CORE DESIGN GENERATION FUNCTIONS
# ============================================================================

def generate_simplex_centroid_design(n_components, component_names=None):
    """
    Generate a simplex centroid design for mixture experiments

    Args:
        n_components: Number of mixture components (2-5)
        component_names: Optional list of component names

    Returns:
        pd.DataFrame with design matrix (2^n - 1 rows × n_components columns)

    Logic:
        For n components, generates 2^n - 1 experiments:
        - Pure components (vertices): n points
        - Binary mixtures: C(n,2) points at (0.5, 0.5, 0)
        - Ternary mixtures: C(n,3) points at (1/3, 1/3, 1/3, 0)
        - Quaternary: C(n,4) points at (0.25, 0.25, 0.25, 0.25, 0)
        - etc.

    Example for n=3:
        7 points: (1,0,0), (0,1,0), (0,0,1), (0.5,0.5,0),
                  (0.5,0,0.5), (0,0.5,0.5), (1/3,1/3,1/3)
    """
    if n_components < MIN_COMPONENTS or n_components > MAX_COMPONENTS:
        raise ValueError(f"n_components must be between {MIN_COMPONENTS} and {MAX_COMPONENTS}")

    # Use default names if not provided
    if component_names is None:
        component_names = COMPONENT_NAMES[:n_components]
    elif len(component_names) != n_components:
        raise ValueError(f"component_names must have length {n_components}")

    design_points = []

    # Generate all possible subsets of components (from 1 to n)
    # k=1: pure components
    # k=2: binary mixtures
    # k=3: ternary mixtures
    # etc.

    for k in range(1, n_components + 1):
        # Get all combinations of k components
        for subset in combinations(range(n_components), k):
            # Create a design point with equal fractions in selected components
            point = np.zeros(n_components)
            fraction = 1.0 / k  # Equal proportions

            for idx in subset:
                point[idx] = fraction

            design_points.append(point)

    # Create DataFrame
    design_matrix = pd.DataFrame(design_points, columns=component_names)

    # Verify design validity
    row_sums = design_matrix.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-10):
        warnings.warn("Design matrix rows do not sum to 1.0 exactly")

    return design_matrix


def apply_pseudo_components(design_matrix, component_ranges):
    """
    Transform pseudo-component design [0,1] to actual component values

    Args:
        design_matrix: pd.DataFrame with pseudo-component values [0-1]
        component_ranges: dict like {'X1': (0.2, 0.6), 'X2': (0.1, 0.45), 'X3': (0, 0.4)}
                          Each tuple is (min_value, max_value) in actual units

    Returns:
        tuple: (actual_design_df, transformation_metadata)

    Logic:
        Pseudo-component transformation maps constrained region to standard simplex

        For component i with range [L_i, U_i]:
        actual_i = pseudo_i * (U_i - L_i) + L_i

        IMPORTANT: The sum of lower bounds must be < 1, and sum of upper bounds must be > 1
        for the transformation to be valid

    Mathematical Background:
        When components have constraints like 0.2 ≤ X1 ≤ 0.6, we can't explore the full
        simplex. Pseudo-components map the constrained feasible region onto a standard
        simplex, making the design more efficient.
    """
    # Validate component ranges
    component_names = design_matrix.columns.tolist()

    if not all(comp in component_ranges for comp in component_names):
        raise ValueError("component_ranges must contain all components in design_matrix")

    # Check feasibility
    lower_sum = sum(component_ranges[comp][0] for comp in component_names)
    upper_sum = sum(component_ranges[comp][1] for comp in component_names)

    if lower_sum >= 1.0:
        raise ValueError(f"Sum of lower bounds ({lower_sum:.4f}) must be < 1.0 for valid mixture")

    if upper_sum <= 1.0:
        raise ValueError(f"Sum of upper bounds ({upper_sum:.4f}) must be > 1.0 for valid range")

    # Transform each component
    actual_design = design_matrix.copy()

    transformation_metadata = {}

    for comp in component_names:
        min_val, max_val = component_ranges[comp]

        if min_val >= max_val:
            raise ValueError(f"Invalid range for {comp}: min ({min_val}) >= max ({max_val})")

        # Linear transformation: actual = pseudo * (max - min) + min
        actual_design[comp] = design_matrix[comp] * (max_val - min_val) + min_val

        transformation_metadata[comp] = {
            'min': min_val,
            'max': max_val,
            'range': max_val - min_val,
            'transformation': f'{comp}_actual = {comp}_pseudo * {max_val - min_val:.4f} + {min_val:.4f}'
        }

    # Verify transformed values still sum to approximately 1
    actual_sums = actual_design.sum(axis=1)

    # Note: With pseudo-components, sums may not equal exactly 1 due to constraints
    # This is expected behavior

    transformation_metadata['_summary'] = {
        'original_component_names': component_names,
        'n_components': len(component_names),
        'pseudo_component_applied': True,
        'actual_sum_range': (actual_sums.min(), actual_sums.max()),
        'pseudo_component_design': design_matrix.copy()  # Store original for inversion
    }

    return actual_design, transformation_metadata


def apply_constraints(design_matrix, constraint_list):
    """
    Filter design matrix to satisfy component constraints

    Args:
        design_matrix: pd.DataFrame with mixture design
        constraint_list: list of constraint dicts:
            [
                {'type': 'independent_max', 'component': 'X3', 'value': 0.4},
                {'type': 'independent_min', 'component': 'X1', 'value': 0.2},
                {'type': 'independent_range', 'component': 'X2', 'min': 0.1, 'max': 0.5},
                {'type': 'relational', 'comp1': 'X1', 'comp2': 'X2', 'relation': '>='}
            ]

    Returns:
        tuple: (filtered_design_df, constraint_info_dict)

    Logic:
        - Filters rows to keep only those satisfying ALL constraints
        - Returns indices of valid and removed points
        - Warns if too many points removed (suggest D-optimal)
    """
    if not constraint_list:
        return design_matrix, {'n_removed': 0, 'n_kept': len(design_matrix), 'all_constraints_met': True}

    # Start with all points valid
    valid_mask = pd.Series([True] * len(design_matrix), index=design_matrix.index)

    constraint_info = {
        'constraints_applied': [],
        'points_removed_per_constraint': [],
        'removed_indices': []
    }

    # Apply each constraint
    for constraint in constraint_list:
        constraint_type = constraint['type']

        if constraint_type == 'independent_max':
            comp = constraint['component']
            max_val = constraint['value']

            # Keep only rows where component <= max_val
            mask = design_matrix[comp] <= max_val + 1e-10  # Small tolerance

            n_removed = (~mask & valid_mask).sum()
            valid_mask = valid_mask & mask

            constraint_info['constraints_applied'].append({
                'type': 'max',
                'component': comp,
                'value': max_val,
                'points_removed': n_removed
            })

        elif constraint_type == 'independent_min':
            comp = constraint['component']
            min_val = constraint['value']

            mask = design_matrix[comp] >= min_val - 1e-10

            n_removed = (~mask & valid_mask).sum()
            valid_mask = valid_mask & mask

            constraint_info['constraints_applied'].append({
                'type': 'min',
                'component': comp,
                'value': min_val,
                'points_removed': n_removed
            })

        elif constraint_type == 'independent_range':
            comp = constraint['component']
            min_val = constraint['min']
            max_val = constraint['max']

            mask = (design_matrix[comp] >= min_val - 1e-10) & (design_matrix[comp] <= max_val + 1e-10)

            n_removed = (~mask & valid_mask).sum()
            valid_mask = valid_mask & mask

            constraint_info['constraints_applied'].append({
                'type': 'range',
                'component': comp,
                'min': min_val,
                'max': max_val,
                'points_removed': n_removed
            })

        elif constraint_type == 'relational':
            comp1 = constraint['comp1']
            comp2 = constraint['comp2']
            relation = constraint['relation']

            if relation == '>=':
                mask = design_matrix[comp1] >= design_matrix[comp2] - 1e-10
            elif relation == '<=':
                mask = design_matrix[comp1] <= design_matrix[comp2] + 1e-10
            elif relation == '==':
                mask = np.isclose(design_matrix[comp1], design_matrix[comp2], atol=1e-10)
            else:
                raise ValueError(f"Unknown relation: {relation}. Use '>=', '<=', or '=='")

            n_removed = (~mask & valid_mask).sum()
            valid_mask = valid_mask & mask

            constraint_info['constraints_applied'].append({
                'type': 'relational',
                'comp1': comp1,
                'comp2': comp2,
                'relation': relation,
                'points_removed': n_removed
            })

    # Filter design matrix
    filtered_design = design_matrix[valid_mask].copy().reset_index(drop=True)

    # Store summary info
    constraint_info['n_original'] = len(design_matrix)
    constraint_info['n_removed'] = len(design_matrix) - len(filtered_design)
    constraint_info['n_kept'] = len(filtered_design)
    constraint_info['removal_fraction'] = constraint_info['n_removed'] / len(design_matrix)
    constraint_info['removed_indices'] = design_matrix.index[~valid_mask].tolist()

    # Warning if too many points removed
    if constraint_info['removal_fraction'] > 0.5:
        warnings.warn(
            f"⚠️ {constraint_info['removal_fraction']*100:.1f}% of design points removed by constraints. "
            "Consider using D-optimal design to generate more efficient design."
        )

    if len(filtered_design) == 0:
        raise ValueError("All design points removed by constraints! Constraints are too restrictive.")

    return filtered_design, constraint_info


def apply_d_optimal_design(candidate_points, n_experiments, model_formula='quadratic'):
    """
    Select optimal subset of candidate points using D-optimal criterion

    Args:
        candidate_points: pd.DataFrame of all candidate mixture points
        n_experiments: number of experiments to select
        model_formula: 'linear', 'quadratic', or 'cubic' (Scheffe polynomial degree)

    Returns:
        tuple: (selected_design_df, optimization_info_dict)

    Logic:
        D-optimal design maximizes det(X'X) where X is the model matrix
        This minimizes the generalized variance of parameter estimates

        Algorithm:
        1. Create model matrix from candidate points (Scheffe polynomial terms)
        2. Use exchange algorithm or greedy selection to maximize det(X'X)
        3. Return selected points with leverage values

    Mathematical Background:
        For mixture models, D-optimality is particularly important when constraints
        reduce the feasible region, making standard simplex centroids inefficient.
    """
    from mixture_utils.mixture_computation import create_scheffe_polynomial_matrix

    if n_experiments > len(candidate_points):
        raise ValueError(f"n_experiments ({n_experiments}) cannot exceed number of candidates ({len(candidate_points)})")

    if n_experiments < 3:
        raise ValueError("D-optimal design requires at least 3 experiments")

    # Create model matrix for all candidate points
    X_candidates = create_scheffe_polynomial_matrix(candidate_points, degree=model_formula)

    n_candidates = len(candidate_points)
    n_params = X_candidates.shape[1]

    if n_experiments < n_params:
        warnings.warn(
            f"n_experiments ({n_experiments}) < model parameters ({n_params}). "
            f"Model will be underspecified. Increase to at least {n_params}."
        )

    # D-optimal selection using greedy algorithm (Fedorov exchange algorithm simplified)
    # Start with random initial design
    np.random.seed(42)  # Reproducibility
    selected_indices = np.random.choice(n_candidates, size=min(n_params, n_experiments), replace=False).tolist()

    # Greedy forward selection to fill remaining slots
    while len(selected_indices) < n_experiments:
        X_current = X_candidates.iloc[selected_indices].values

        # Current D-optimality criterion
        try:
            current_det = np.linalg.det(X_current.T @ X_current)
        except np.linalg.LinAlgError:
            current_det = 0

        best_candidate = None
        best_det = current_det

        # Try adding each remaining candidate
        for cand_idx in range(n_candidates):
            if cand_idx in selected_indices:
                continue

            # Create trial design with this candidate
            trial_indices = selected_indices + [cand_idx]
            X_trial = X_candidates.iloc[trial_indices].values

            try:
                trial_det = np.linalg.det(X_trial.T @ X_trial)
            except np.linalg.LinAlgError:
                trial_det = 0

            if trial_det > best_det:
                best_det = trial_det
                best_candidate = cand_idx

        if best_candidate is not None:
            selected_indices.append(best_candidate)
        else:
            # No improvement found - add random remaining point
            remaining = [i for i in range(n_candidates) if i not in selected_indices]
            if remaining:
                selected_indices.append(np.random.choice(remaining))
            else:
                break

    # Extract selected design
    selected_design = candidate_points.iloc[selected_indices].copy().reset_index(drop=True)

    # Calculate leverage for selected points
    X_selected = X_candidates.iloc[selected_indices].values

    try:
        # Leverage = diagonal of X(X'X)^-1X'
        XtX_inv = np.linalg.inv(X_selected.T @ X_selected)
        leverage = np.diag(X_selected @ XtX_inv @ X_selected.T)
    except np.linalg.LinAlgError:
        leverage = np.full(len(selected_design), np.nan)
        warnings.warn("Could not compute leverage - singular matrix")

    # Optimization info
    try:
        final_det = np.linalg.det(X_selected.T @ X_selected)
        condition_number = np.linalg.cond(X_selected.T @ X_selected)
    except:
        final_det = np.nan
        condition_number = np.nan

    optimization_info = {
        'n_candidates': n_candidates,
        'n_selected': n_experiments,
        'n_parameters': n_params,
        'model_formula': model_formula,
        'd_criterion': final_det,
        'condition_number': condition_number,
        'leverage_values': leverage,
        'selected_indices': selected_indices,
        'average_leverage': np.nanmean(leverage) if not np.all(np.isnan(leverage)) else np.nan,
        'max_leverage': np.nanmax(leverage) if not np.all(np.isnan(leverage)) else np.nan
    }

    return selected_design, optimization_info


def validate_mixture_design(design_matrix, tolerance=1e-10):
    """
    Validate that a design matrix satisfies mixture constraints

    Args:
        design_matrix: pd.DataFrame or np.array with mixture design
        tolerance: tolerance for numerical checks

    Returns:
        dict with validation results:
        {
            'is_valid': bool,
            'errors': [list of error strings],
            'warnings': [list of warning strings],
            'stats': {dict of design statistics}
        }

    Logic:
        Checks:
        - Each row sums to 1 (within tolerance)
        - All values in [0, 1]
        - No duplicate rows
        - Presence of key design points (vertices, centroids)
    """
    if isinstance(design_matrix, pd.DataFrame):
        design_array = design_matrix.values
        component_names = design_matrix.columns.tolist()
    else:
        design_array = np.array(design_matrix)
        component_names = [f"X{i+1}" for i in range(design_array.shape[1])]

    n_rows, n_cols = design_array.shape

    errors = []
    warnings_list = []
    stats = {}

    # Check 1: Row sums equal 1
    row_sums = design_array.sum(axis=1)
    sum_check = np.allclose(row_sums, 1.0, atol=tolerance)

    if not sum_check:
        max_deviation = np.max(np.abs(row_sums - 1.0))
        errors.append(f"Rows do not sum to 1.0 (max deviation: {max_deviation:.6f})")
        stats['row_sum_deviation_max'] = max_deviation
    else:
        stats['row_sum_deviation_max'] = np.max(np.abs(row_sums - 1.0))

    # Check 2: All values in [0, 1]
    min_val = design_array.min()
    max_val = design_array.max()

    if min_val < -tolerance:
        errors.append(f"Negative values found (min: {min_val:.6f})")
    if max_val > 1.0 + tolerance:
        errors.append(f"Values > 1.0 found (max: {max_val:.6f})")

    stats['value_range'] = (min_val, max_val)

    # Check 3: Duplicate rows
    unique_rows = np.unique(design_array, axis=0)
    n_unique = len(unique_rows)

    if n_unique < n_rows:
        warnings_list.append(f"Duplicate rows detected ({n_rows - n_unique} duplicates)")
        stats['n_duplicates'] = n_rows - n_unique
    else:
        stats['n_duplicates'] = 0

    # Check 4: Presence of pure components (vertices)
    n_vertices = 0
    for i in range(n_cols):
        # Check if there's a row with component i = 1, all others = 0
        vertex = np.zeros(n_cols)
        vertex[i] = 1.0

        if any(np.allclose(row, vertex, atol=tolerance) for row in design_array):
            n_vertices += 1

    stats['n_vertices_present'] = n_vertices
    stats['n_vertices_expected'] = n_cols

    if n_vertices < n_cols:
        warnings_list.append(f"Not all pure components present ({n_vertices}/{n_cols} vertices)")

    # Check 5: Presence of overall centroid
    centroid = np.ones(n_cols) / n_cols
    centroid_present = any(np.allclose(row, centroid, atol=tolerance) for row in design_array)

    stats['centroid_present'] = centroid_present

    if not centroid_present:
        warnings_list.append("Overall centroid not present in design")

    # Overall validity
    is_valid = len(errors) == 0

    # Design statistics
    stats['n_experiments'] = n_rows
    stats['n_components'] = n_cols
    stats['n_unique_experiments'] = n_unique
    stats['component_names'] = component_names

    # Component-wise statistics
    component_stats = {}
    for i, comp_name in enumerate(component_names):
        comp_values = design_array[:, i]
        component_stats[comp_name] = {
            'min': comp_values.min(),
            'max': comp_values.max(),
            'mean': comp_values.mean(),
            'n_zero': np.sum(np.isclose(comp_values, 0, atol=tolerance)),
            'n_one': np.sum(np.isclose(comp_values, 1, atol=tolerance))
        }

    stats['component_statistics'] = component_stats

    return {
        'is_valid': is_valid,
        'errors': errors,
        'warnings': warnings_list,
        'stats': stats
    }


def plot_simplex_2d(design_matrix, response_values=None, constraints=None):
    """
    Plot mixture design on 2D simplex (ternary plot for 3 components)

    Args:
        design_matrix: pd.DataFrame with 3 components
        response_values: optional array of response values for color mapping
        constraints: optional constraint dict to show feasible region

    Returns:
        plotly figure

    Logic:
        For 3-component mixtures: Create ternary (triangular) plot
        For 2-component: Create 1D line plot
        For 4+ components: Return error (use slicing/projection)

    Note: Full implementation in mixture_ui_utils.py plot_ternary_design()
    """
    n_components = design_matrix.shape[1]

    if n_components != 3:
        raise ValueError(f"plot_simplex_2d requires exactly 3 components, got {n_components}")

    # Use plotly's scatterternary
    comp_names = design_matrix.columns.tolist()

    fig = go.Figure()

    # Prepare hover text
    hover_text = []
    for idx, row in design_matrix.iterrows():
        text = f"Point {idx+1}<br>"
        for comp in comp_names:
            text += f"{comp}: {row[comp]:.4f}<br>"
        if response_values is not None and idx < len(response_values):
            text += f"Response: {response_values[idx]:.4f}"
        hover_text.append(text)

    # Color by response if provided
    if response_values is not None:
        marker_color = response_values
        colorbar = dict(title="Response")
    else:
        marker_color = 'blue'
        colorbar = None

    fig.add_trace(go.Scatterternary(
        a=design_matrix.iloc[:, 0],
        b=design_matrix.iloc[:, 1],
        c=design_matrix.iloc[:, 2],
        mode='markers+text',
        marker=dict(
            size=12,
            color=marker_color,
            colorscale='Viridis' if response_values is not None else None,
            showscale=response_values is not None,
            colorbar=colorbar,
            line=dict(width=1, color='white')
        ),
        text=[f"{i+1}" for i in range(len(design_matrix))],
        textposition='top center',
        textfont=dict(size=10),
        hovertext=hover_text,
        hoverinfo='text'
    ))

    fig.update_layout(
        title='Mixture Design - Ternary Plot',
        ternary=dict(
            sum=1,
            aaxis=dict(title=comp_names[0], min=0, max=1),
            baxis=dict(title=comp_names[1], min=0, max=1),
            caxis=dict(title=comp_names[2], min=0, max=1)
        ),
        showlegend=False,
        height=600
    )

    return fig
