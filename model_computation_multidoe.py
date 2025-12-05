"""
Multi-DOE Model Computation Module
Core computation for multiple MLR models (one per Y variable)

This module provides functions to fit multiple MLR models simultaneously,
each with the same X predictors but different response variables.
"""

import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import plotly.graph_objects as go
import plotly.express as px

# Import existing single-model functions from model_computation.py
from .model_computation import (
    create_model_matrix,
    fit_mlr_model,
    statistical_summary
)


def fit_multidoe_model(X, y, terms=None, exclude_central=False, y_name="Y"):
    """
    Fit a single MLR model (wrapper around fit_mlr_model with y_name tracking)

    Args:
        X (DataFrame): Predictor variables
        y (Series): Response variable
        terms (list, optional): List of terms to include in model
        exclude_central (bool): Whether to exclude central points
        y_name (str): Name of response variable (for tracking)

    Returns:
        dict: Model result with added 'y_name' field
    """
    # Call existing fit_mlr_model from model_computation.py
    model_result = fit_mlr_model(X, y, terms=terms, exclude_central=exclude_central)

    # Add y_name to track which response this is
    model_result['y_name'] = y_name

    return model_result


def fit_all_multidoe_models(X, y_dict, terms=None, exclude_central=False):
    """
    Fit multiple MLR models, one per Y variable

    Args:
        X (DataFrame): Predictor variables (same for all models)
        y_dict (dict or DataFrame): Response variables
            - If dict: {y_name: Series, ...}
            - If DataFrame: use column names as y names
        terms (list, optional): List of terms to include in models
        exclude_central (bool): Whether to exclude central points

    Returns:
        tuple: (models_dict, y_order_list)
            - models_dict: {y_name: model_result_dict}
            - y_order_list: List of Y variable names in order they were processed
    """
    models_dict = {}
    y_order_list = []  # Track Y variable order

    # Handle both dict and DataFrame inputs
    if isinstance(y_dict, pd.DataFrame):
        y_items = [(col, y_dict[col]) for col in y_dict.columns]
    else:
        y_items = list(y_dict.items())

    for y_name, y_series in y_items:
        y_order_list.append(y_name)  # Capture order
        try:
            model = fit_multidoe_model(
                X, y_series,
                terms=terms,
                exclude_central=exclude_central,
                y_name=y_name
            )
            models_dict[y_name] = model
        except Exception as e:
            # Store error info instead of crashing
            models_dict[y_name] = {
                'error': str(e),
                'y_name': y_name,
                'status': 'Failed'
            }

    return models_dict, y_order_list


def statistical_summary_multidoe(models_dict):
    """
    Create summary table for all models

    Args:
        models_dict (dict): Dictionary of model results {y_name: model_result}

    Returns:
        DataFrame with columns:
        [Y_Variable, R²_adj, RMSE, DOF, Q²_adj, RMSECV, Status]
    """
    summary_rows = []

    for y_name, model_result in models_dict.items():
        if 'error' in model_result:
            status = f"❌ {model_result['error'][:30]}"
            r2_adj = np.nan
            rmse = np.nan
            dof = np.nan
            q2_adj = np.nan
            rmsecv = np.nan
        elif model_result.get('dof', 0) <= 0:
            status = "⚠️ Saturated"
            r2_adj = model_result.get('r_squared_adj', model_result.get('r_squared', np.nan))
            rmse = model_result.get('rmse', np.nan)
            dof = model_result.get('dof', np.nan)
            q2_adj = model_result.get('q2_adj', model_result.get('q2', np.nan))
            rmsecv = model_result.get('rmsecv', np.nan)
        else:
            status = "✅ OK"
            r2_adj = model_result.get('r_squared_adj', model_result.get('r_squared', np.nan))
            rmse = model_result.get('rmse', np.nan)
            dof = model_result.get('dof', np.nan)
            q2_adj = model_result.get('q2_adj', model_result.get('q2', np.nan))
            rmsecv = model_result.get('rmsecv', np.nan)

        summary_rows.append({
            'Y_Variable': y_name,
            'R²_adj': r2_adj,
            'RMSE': rmse,
            'DOF': dof,
            'Q²_adj': q2_adj,
            'RMSECV': rmsecv,
            'Status': status
        })

    return pd.DataFrame(summary_rows)


def extract_coefficients_comparison(models_dict, y_order=None):
    """
    Create DataFrame with all coefficients side-by-side

    Args:
        models_dict (dict): Dictionary of model results
        y_order (list, optional): List of Y variable names in desired column order.
                                  If None, preserves dict insertion order (Python 3.7+)

    Returns:
        DataFrame with coefficient names as index and Y variables as columns
    """
    # Get all unique coefficient names across all models
    all_coef_names = set()
    for model in models_dict.values():
        if 'coefficients' in model:
            all_coef_names.update(model['coefficients'].index)

    # Sort coefficient names for consistent ordering
    all_coef_names = sorted(all_coef_names)

    # Determine Y variable order
    if y_order is not None:
        # Use provided order (filter to only include Y variables in models_dict)
        y_names_ordered = [y for y in y_order if y in models_dict]
    else:
        # Preserve dict insertion order (Python 3.7+)
        y_names_ordered = list(models_dict.keys())

    # Build matrix with ordered columns
    coef_matrix = {}
    for y_name in y_names_ordered:
        model = models_dict[y_name]
        if 'coefficients' in model:
            coef_dict = model['coefficients'].to_dict()
        else:
            coef_dict = {}

        coef_matrix[y_name] = [coef_dict.get(name, np.nan) for name in all_coef_names]

    return pd.DataFrame(coef_matrix, index=all_coef_names)


def normalize_coefficients_for_comparison(coef_df):
    """
    Normalize coefficients by max absolute value per Y variable (column).
    This makes coefficients from different Y variables comparable on the same scale.

    Args:
        coef_df (DataFrame): Coefficients DataFrame (index=term names, columns=Y variables)

    Returns:
        tuple: (normalized_df, max_values_dict)
            - normalized_df: DataFrame with normalized coefficients (0-1 scale)
            - max_values_dict: {y_name: max_absolute_value} for reference
    """
    try:
        # Create copy to avoid modifying original
        normalized_df = coef_df.copy()
        max_values = {}

        # Exclude intercept from normalization (if exists)
        intercept_row = None
        intercept_index = None
        for idx in normalized_df.index:
            if 'intercept' in str(idx).lower():
                intercept_row = normalized_df.loc[idx].copy()
                intercept_index = idx
                normalized_df = normalized_df.drop(idx)
                break

        # Normalize each Y variable (column) independently
        for col in normalized_df.columns:
            col_values = normalized_df[col].dropna()

            if len(col_values) == 0:
                # All NaN - skip normalization
                max_values[col] = np.nan
                continue

            max_abs_value = col_values.abs().max()

            if max_abs_value == 0 or np.isnan(max_abs_value):
                # All zeros or invalid - skip normalization
                max_values[col] = 0
                continue

            # Normalize: divide by max absolute value
            normalized_df[col] = normalized_df[col] / max_abs_value
            max_values[col] = max_abs_value

            # Debug info (only if debug mode enabled)
            if st.session_state.get('debug_mode', False):
                st.write(f"**{col}**: max_abs = {max_abs_value:.4f}, normalized ✅")

        # Re-add intercept row if it existed (keep original values)
        if intercept_row is not None:
            normalized_df.loc[intercept_index] = intercept_row

        # Store max values in session state for later reference
        if 'streamlit' in str(type(st)):
            st.session_state['coef_max_values'] = max_values

        return normalized_df, max_values

    except Exception as e:
        # If normalization fails, return original DataFrame
        st.warning(f"⚠️ Normalization failed: {str(e)}. Using original coefficients.")
        return coef_df, {}


def get_coefficient_colors(coef_df):
    """
    Categorize coefficients by term type and assign colors.

    Term types:
    - Linear: Single variable name (e.g., 'X1', 'Temperature') → Red
    - Interaction: Contains '*' (e.g., 'X1*X2') → Green
    - Quadratic: Contains '^2' (e.g., 'X1^2') → Cyan

    Args:
        coef_df (DataFrame): Coefficients DataFrame with term names as index

    Returns:
        dict: {term_name: color_hex_value}
    """
    color_map = {}

    for term_name in coef_df.index:
        term_str = str(term_name).lower()

        # Skip intercept
        if 'intercept' in term_str:
            color_map[term_name] = '#808080'  # Gray for intercept
        # Quadratic terms (check first because they may also contain variable names)
        elif '^2' in term_str or '²' in term_str:
            color_map[term_name] = '#00CED1'  # Cyan (DarkTurquoise)
        # Interaction terms
        elif '*' in term_str or ':' in term_str:
            color_map[term_name] = '#32CD32'  # Green (LimeGreen)
        # Linear terms (single variable)
        else:
            color_map[term_name] = '#DC143C'  # Red (Crimson)

    return color_map


def categorize_coefficients_by_type(coef_df):
    """
    Categorize coefficient indices by term type for visualization separators.

    Args:
        coef_df (DataFrame): Coefficients DataFrame with term names as index

    Returns:
        dict: {
            'intercept': [list of intercept indices],
            'linear': [list of linear term indices],
            'interaction': [list of interaction term indices],
            'quadratic': [list of quadratic term indices]
        }

    Example output:
        {
            'intercept': [0],
            'linear': [1, 2, 3],
            'interaction': [4, 5],
            'quadratic': [6, 7, 8]
        }
    """
    categories = {
        'intercept': [],
        'linear': [],
        'interaction': [],
        'quadratic': []
    }

    for idx, term_name in enumerate(coef_df.index):
        term_str = str(term_name).lower()

        # Categorize by term type
        if 'intercept' in term_str:
            categories['intercept'].append(idx)
        elif '^2' in term_str or '²' in term_str:
            categories['quadratic'].append(idx)
        elif '*' in term_str or ':' in term_str:
            categories['interaction'].append(idx)
        else:
            categories['linear'].append(idx)

    return categories


def get_separator_positions(coef_df):
    """
    Calculate x-axis positions for vertical separators between term types.

    The separator is placed BETWEEN bars, not on them (hence the -0.5 offset).

    Args:
        coef_df (DataFrame): Coefficients DataFrame with terms as index

    Returns:
        dict: {
            'linear_end': float or None,
            'interaction_end': float or None,
            'quadratic_end': float or None
        }

    Example:
        If we have terms: [Intercept, X1, X2, X1*X2, X1^2]
        Returns: {'linear_end': 2.5, 'interaction_end': 3.5, 'quadratic_end': None}
    """
    categories = categorize_coefficients_by_type(coef_df)

    # Calculate cumulative positions
    n_intercept = len(categories['intercept'])
    n_linear = len(categories['linear'])
    n_interaction = len(categories['interaction'])
    n_quadratic = len(categories['quadratic'])

    positions = {
        'linear_end': None,
        'interaction_end': None,
        'quadratic_end': None
    }

    # Position after linear terms (if interaction or quadratic exists)
    if n_linear > 0 and (n_interaction > 0 or n_quadratic > 0):
        positions['linear_end'] = n_intercept + n_linear - 0.5

    # Position after interaction terms (if quadratic exists)
    if n_interaction > 0 and n_quadratic > 0:
        positions['interaction_end'] = n_intercept + n_linear + n_interaction - 0.5

    # No separator after quadratic (it's the last category)
    # positions['quadratic_end'] is left as None

    return positions


# ============================================================================
# SECTION: CI CALCULATION UTILITIES
# ============================================================================

def calculate_ci_at_point(model_result, X_point, x_vars, ci_type='Prediction',
                          s_model=None, dof_model=None, s_exp=None):
    """
    Calculate confidence interval at a single prediction point

    Args:
        model_result (dict): Model from fit_multidoe_model
        X_point (array): 1D array or DataFrame row with predictor values
        x_vars (list): X variable names
        ci_type (str): 'Prediction' or 'Experimental'
        s_model (float): Model std dev (if None, uses model RMSE)
        dof_model (int): Model DOF (if None, uses model dof)
        s_exp (float): Experimental/measurement std dev (only for Experimental)

    Returns:
        dict: {
            'ci_semiamplitude': float,
            'ci_lower': float,
            'ci_upper': float,
            'leverage': float,
            'y_pred': float
        }
    """
    from scipy import stats

    if s_model is None:
        s_model = model_result.get('rmse', 1.0)
    if dof_model is None:
        dof_model = model_result.get('dof', 1)

    # Get coefficient names and create model matrix for point
    coefficients = model_result['coefficients']
    coef_names = coefficients.index.tolist()

    # Import create_prediction_matrix from surface_analysis
    from .surface_analysis import create_prediction_matrix

    # Ensure X_point is 2D array (1, n_vars)
    if len(X_point.shape) == 1:
        X_point = X_point.reshape(1, -1)

    X_model = create_prediction_matrix(X_point, x_vars, coef_names)

    # Prediction
    y_pred = float(X_model @ coefficients.values)

    # Calculate leverage
    dispersion = model_result['XtX_inv']
    leverage = float((X_model @ dispersion @ X_model.T).diagonal()[0])

    # t-critical value (95% CI, two-sided)
    t_crit = stats.t.ppf(0.975, dof_model)

    # CI calculation
    if ci_type == 'Experimental' and s_exp is not None:
        # Experimental CI: √((CI_model)² + (CI_exp)²)
        # CI_model = t * s_model * √(leverage)
        # CI_exp = t * s_exp (for single measurement, or s_exp/√n for n replicates)
        ci_model = t_crit * s_model * np.sqrt(leverage)
        ci_exp = t_crit * s_exp
        ci_semi = np.sqrt(ci_model**2 + ci_exp**2)
    else:
        # Prediction CI (model uncertainty only)
        ci_semi = t_crit * s_model * np.sqrt(leverage)

    return {
        'y_pred': y_pred,
        'ci_semiamplitude': ci_semi,
        'ci_lower': y_pred - ci_semi,
        'ci_upper': y_pred + ci_semi,
        'leverage': leverage,
        's_model': s_model,
        'dof_model': dof_model,
        's_exp': s_exp
    }


def calculate_ci_at_point_experimental(model_result, X_point, x_vars,
                                       ci_type='Prediction',
                                       s_model=None, dof_model=None,
                                       n_replicates=1, s_exp=None, dof_exp=None):
    """
    Calculate confidence interval at a single prediction point with error propagation

    Supports both Prediction (model uncertainty only) and Experimental
    (model + measurement uncertainty with error propagation).

    Args:
        model_result (dict): Model from fit_multidoe_model
        X_point (array): 1D array with predictor values
        x_vars (list): X variable names
        ci_type (str): 'Prediction' or 'Experimental'
        s_model (float): Model std dev (if None, uses model RMSE)
        dof_model (int): Model DOF (if None, uses model dof)
        n_replicates (int): Number of replicate measurements (for Experimental)
        s_exp (float): Experimental std dev (for Experimental)
        dof_exp (int): Experimental DOF (for Experimental)

    Returns:
        dict with keys:
        - 'y_pred': float - predicted value
        - 'ci_semiamplitude': float - CI radius
        - 'ci_lower': float - lower CI bound
        - 'ci_upper': float - upper CI bound
        - 'leverage': float - leverage value
        - 'ci_type': str - 'Prediction' or 'Experimental'
        - 'ci_model': float - model component (for Experimental)
        - 'ci_exp': float - experimental component (for Experimental)
        - 's_model': float - model std dev used
        - 's_exp': float - experimental std dev used (if Experimental)
        - 'n_replicates': int - replicates used (if Experimental)
    """
    from scipy import stats

    # Set defaults
    if s_model is None:
        s_model = model_result.get('rmse', 1.0)
    if dof_model is None:
        dof_model = model_result.get('dof', 1)

    # Import create_prediction_matrix from surface_analysis
    from .surface_analysis import create_prediction_matrix

    # Create model matrix for point
    coef_names = model_result['coefficients'].index.tolist()

    # Ensure X_point is 2D
    if len(X_point.shape) == 1:
        X_point = X_point.reshape(1, -1)

    X_model = create_prediction_matrix(X_point, x_vars, coef_names)

    # Prediction
    y_pred = float(X_model @ model_result['coefficients'].values)

    # Calculate leverage
    leverage = float((X_model @ model_result['XtX_inv'] @ X_model.T).diagonal()[0])

    # t-critical value for model
    t_crit_model = stats.t.ppf(0.975, dof_model)

    result = {
        'y_pred': y_pred,
        'leverage': leverage,
        's_model': s_model,
        'dof_model': dof_model,
        'ci_type': ci_type
    }

    if ci_type == 'Experimental':
        # Set experimental defaults
        if s_exp is None:
            s_exp = s_model
        if dof_exp is None:
            dof_exp = dof_model

        # t-critical for experimental
        t_crit_exp = stats.t.ppf(0.975, dof_exp)

        # CI components
        ci_model = t_crit_model * s_model * np.sqrt(leverage)
        ci_exp = t_crit_exp * s_exp * np.sqrt(1.0 / n_replicates)

        # Total CI via error propagation
        ci_semi = np.sqrt(ci_model**2 + ci_exp**2)

        result.update({
            'ci_semiamplitude': ci_semi,
            'ci_lower': y_pred - ci_semi,
            'ci_upper': y_pred + ci_semi,
            'ci_model': ci_model,
            'ci_exp': ci_exp,
            's_exp': s_exp,
            'dof_exp': dof_exp,
            'n_replicates': n_replicates
        })

    else:  # Prediction
        ci_semi = t_crit_model * s_model * np.sqrt(leverage)

        result.update({
            'ci_semiamplitude': ci_semi,
            'ci_lower': y_pred - ci_semi,
            'ci_upper': y_pred + ci_semi
        })

    return result
