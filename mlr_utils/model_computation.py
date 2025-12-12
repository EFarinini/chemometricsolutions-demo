"""
MLR Model Computation - Core Functions and UI
Equivalent to DOE_model_computation.r
Complete model fitting workflow with term selection, diagnostics, and statistical tests

This module contains:
1. Core computation functions (create_model_matrix, fit_mlr_model, statistical_summary)
2. UI display functions (show_model_computation_ui)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy import stats


# ============================================================================
# CORE COMPUTATION FUNCTIONS
# ============================================================================

def analyze_design_structure(X):
    """
    Analyze experimental design structure.

    REGOLA 1 (2-LEVEL): All quantitative columns have exactly 2 levels
    → Recommend: Intercept + Linear + Interactions (quant only)

    REGOLA 2 (>2-LEVEL): At least one quantitative column has 3+ levels
    → Recommend: Intercept + Linear + Interactions + Quadratic (quant only)

    REGOLA 3 (QUALITATIVE): Column with only {0, 1} values
    → Is qualitative (one-hot from 3+ original levels)
    → For this variable: Linear only

    Args:
        X: DataFrame with experimental design

    Returns:
        dict with design analysis
    """
    warnings_list = []

    # STEP 1: Identify center points (all values ≈ 0)
    tolerance = 1e-10
    center_point_mask = np.all(np.abs(X.values) < tolerance, axis=1)
    center_point_indices = np.where(center_point_mask)[0].tolist()


    # STEP 2: Exclude center points for analysis
    X_non_center = X[~center_point_mask].copy()

    if X_non_center.empty:
        warnings_list.append("⚠️ All data points are at center (0,0,...,0)")
        return {
            'design_type': 'error',
            'n_levels_per_var': {},
            'center_points_indices': center_point_indices,
            'is_quantitative': {},
            'quantitative_vars': [],
            'qualitative_vars': [],
            'recommended_terms': {'intercept': True, 'linear': False, 'interactions': False, 'quadratic': False},
            'interpretation': "Cannot analyze - all points at center",
            'warnings': warnings_list
        }

    # STEP 3: Classify each column as QUANTITATIVE or QUALITATIVE
    n_levels_per_var = {}
    is_quantitative_var = {}
    quantitative_vars = []
    qualitative_vars = []

    for col_name in X.columns:
        unique_vals = X_non_center[col_name].unique()
        n_levels = len(unique_vals)
        n_levels_per_var[col_name] = n_levels

        # Classification rule:
        # If column has ONLY {0, 1} → QUALITATIVE (one-hot indicator)

        # Otherwise → QUANTITATIVE (can be -1,+1 or -1,0,+1 etc.)

        unique_set = set(np.round(unique_vals, 10))

        if unique_set == {0.0, 1.0} or unique_set == {0.0} or unique_set == {1.0}:
            # QUALITATIVE: one-hot encoded categorical
            is_quantitative_var[col_name] = False
            qualitative_vars.append(col_name)

        else:
            # QUANTITATIVE: regular design variable
            is_quantitative_var[col_name] = True
            quantitative_vars.append(col_name)


    # STEP 4: Apply the three rules based on QUANTITATIVE variables only

    if not quantitative_vars:
        # All variables are qualitative
        design_type = "qualitative_only"
        interpretation = "Pure categorical design"
        recommended_terms = {
            'intercept': True,
            'linear': True,
            'interactions': False,
            'quadratic': False
        }
        if qualitative_vars:
            warnings_list.append(f"ℹ️ Variables: {', '.join(qualitative_vars)} (all qualitative)")


    else:
        # We have quantitative variables - check their levels
        quant_levels = [n_levels_per_var[v] for v in quantitative_vars]
        max_levels = max(quant_levels)

        if max_levels == 2:
            # REGOLA 1: 2-LEVEL
            # All quantitative variables have exactly 2 levels
            design_type = "2-level"
            interpretation = "2-Level design"
            recommended_terms = {
                'intercept': True,
                'linear': True,
                'interactions': True,    # ✓ With quantitative variables
                'quadratic': False        # ✗ Can't fit with 2 levels
            }

        elif max_levels >= 3:
            # REGOLA 2: >2-LEVEL
            # At least one quantitative variable has 3+ levels
            design_type = ">2-level"  # Could be 3-level, 4-level, etc.
            interpretation = f"{max_levels}-Level design"
            recommended_terms = {
                'intercept': True,
                'linear': True,
                'interactions': True,    # ✓ With 2-level quantitative vars
                'quadratic': True         # ✓ With 2-level quantitative vars
            }

        else:
            design_type = "unknown"
            interpretation = "Unknown design"
            recommended_terms = {
                'intercept': True,
                'linear': True,
                'interactions': False,
                'quadratic': False
            }

    # STEP 5: Add warnings
    if qualitative_vars:
        warnings_list.append(f"ℹ️ Qualitative variables (one-hot): {', '.join(qualitative_vars)} → Linear only")

    if len(center_point_indices) > 0:
        warnings_list.append(f"ℹ️ Found {len(center_point_indices)} center point(s)")


    # Build interpretation message
    if quantitative_vars and qualitative_vars:
        interpretation += f" + {len(qualitative_vars)} qualitative"

    return {
        'design_type': design_type,
        'n_levels_per_var': n_levels_per_var,
        'center_points_indices': center_point_indices,
        'is_quantitative': is_quantitative_var,
        'quantitative_vars': quantitative_vars,
        'qualitative_vars': qualitative_vars,
        'recommended_terms': recommended_terms,
        'interpretation': interpretation,
        'warnings': warnings_list
    }


def create_model_matrix(X, terms_dict=None, include_intercept=True,
                       include_interactions=True, include_quadratic=True,
                       interaction_matrix=None):
    """
    Build design matrix (X) from selected terms with defensive checks

    GENERIC IMPLEMENTATION:
    - Works with any number of variables
    - Handles both full models and custom term selection
    - Defensive checks for matrix validity

    Args:
        X: DataFrame with predictor variables (n_samples × n_vars)
        terms_dict: dict with 'linear', 'interactions', 'quadratic' lists (optional)
        include_intercept: bool, include intercept term
        include_interactions: bool, include two-way interactions
        include_quadratic: bool, include quadratic terms
        interaction_matrix: DataFrame specifying which interactions to include (optional)

    Returns:
        tuple: (X_model DataFrame, term_names list)

    Raises:
        ValueError: if input validation fails
    """
    # Input validation
    if not isinstance(X, pd.DataFrame):
        raise ValueError("X must be a pandas DataFrame")

    if X.empty:
        raise ValueError("X DataFrame is empty")

    if X.isna().any().any():
        raise ValueError("X contains missing values - please remove or impute first")

    n_vars = X.shape[1]
    var_names = X.columns.tolist()


    # Start with linear terms
    model_matrix = X.copy()
    term_names = var_names.copy()


    # Track what we're adding
    added_interactions = []
    added_quadratics = []

    # Add interactions
    if include_interactions:
        for i in range(n_vars):
            for j in range(i+1, n_vars):
                # Check if this interaction should be included
                include_this = True

                if interaction_matrix is not None:
                    try:
                        include_this = bool(interaction_matrix.iloc[i, j])

                    except (IndexError, KeyError):
                        pass  # Include by default on error

                if include_this:
                    interaction = X.iloc[:, i] * X.iloc[:, j]
                    interaction_name = f"{var_names[i]}*{var_names[j]}"
                    model_matrix[interaction_name] = interaction
                    term_names.append(interaction_name)
                    added_interactions.append(interaction_name)


    # Add quadratic terms
    if include_quadratic:
        for i in range(n_vars):
            # Check if quadratic should be included
            include_this = True

            if interaction_matrix is not None:
                try:
                    include_this = bool(interaction_matrix.iloc[i, i])

                except (IndexError, KeyError):
                    pass  # Include by default on error

            if include_this:
                quadratic = X.iloc[:, i] ** 2
                quadratic_name = f"{var_names[i]}^2"
                model_matrix[quadratic_name] = quadratic
                term_names.append(quadratic_name)
                added_quadratics.append(quadratic_name)


    # Add intercept
    if include_intercept:
        model_matrix.insert(0, 'Intercept', 1.0)
        term_names.insert(0, 'Intercept')


    # Final validation
    if model_matrix.isna().any().any():
        raise ValueError("Model matrix contains NaN values after construction")


    # Check for constant columns (except intercept)
    for col in model_matrix.columns:
        if col != 'Intercept':
            if model_matrix[col].std() == 0:
                raise ValueError(f"Column '{col}' has zero variance - remove or check data")

    return model_matrix, term_names


def fit_mlr_model(X, y, terms=None, exclude_central=False, return_diagnostics=True):
    """
    Fit MLR model with defensive checks - GENERIC for any design

    HANDLES BOTH:
    - Designs WITH replicates (calculates pure error, lack of fit)
    - Designs WITHOUT replicates (only R², RMSE, VIF, Leverage)

    ALWAYS CALCULATES (independent of replicates):
    - VIF (multicollinearity)
    - Leverage (influential points)
    - Coefficients and predictions

    CONDITIONAL CALCULATIONS:
    - R², RMSE: only if DOF > 0
    - Statistical tests: only if DOF > 0
    - Pure error: only if replicates detected

    Args:
        X: model matrix DataFrame (n_samples × n_features)
        y: response variable Series (n_samples)
        terms: optional dict of selected terms
        exclude_central: bool, exclude central points (handled externally)
        return_diagnostics: bool, compute cross-validation

    Returns:
        dict with all available metrics (adapts to data structure)
    """
    # Input validation
    if not isinstance(X, pd.DataFrame):
        raise ValueError("X must be a pandas DataFrame")

    if not isinstance(y, (pd.Series, pd.DataFrame)):
        raise ValueError("y must be a pandas Series or DataFrame")

    if X.shape[0] != len(y):
        raise ValueError(f"X and y length mismatch: X has {X.shape[0]} rows, y has {len(y)} values")


    # Convert to numpy arrays
    X_mat = X.values
    y_vec = y.values if isinstance(y, pd.Series) else y.values.ravel()

    n_samples, n_features = X_mat.shape

    # Check rank
    rank = np.linalg.matrix_rank(X_mat)
    if rank < n_features:
        st.error(f"⚠️ Model matrix is rank deficient! Rank={rank}, Features={n_features}")
        return None

    # Degrees of freedom
    dof = n_samples - n_features
    # Initialize results dictionary
    results = {
        'n_samples': n_samples,
        'n_features': n_features,
        'dof': dof,
        'X': X,
        'y': y,
        'coefficients': None,
        'y_pred': None,
        'residuals': None
    }

    try:
        # ===== ALWAYS: Compute coefficients and predictions =====
        XtX = X_mat.T @ X_mat
        XtX_inv = np.linalg.inv(XtX)
        Xty = X_mat.T @ y_vec
        coefficients = XtX_inv @ Xty

        # Predictions
        y_pred = X_mat @ coefficients

        # Residuals
        residuals = y_vec - y_pred

        results.update({
            'coefficients': pd.Series(coefficients, index=X.columns),
            'y_pred': y_pred,
            'residuals': residuals,
            'XtX_inv': XtX_inv
        })

        # ===== CONDITIONAL: R², RMSE (only if DOF > 0) =====
        if dof > 0:
            # Variance of residuals
            rss = np.sum(residuals**2)
            var_res = rss / dof
            rmse = np.sqrt(var_res)


            # Variance of Y
            var_y = np.var(y_vec, ddof=1)


            # Adjusted R-squared: R²_adj = 1 - [RSS/(n-p)] / [TSS/(n-1)]
            # where p=n_features (number of parameters)
            tss = np.sum((y_vec - np.mean(y_vec))**2)
            r_squared_adj = 1 - (rss / dof) / (tss / (n_samples - 1))

            results.update({
                'rmse': rmse,
                'var_res': var_res,
                'var_y': var_y,
                'r_squared_adj': r_squared_adj,
                'r_squared': r_squared_adj  # Keep for backward compatibility
            })

            # ===== CONDITIONAL: Statistical tests (only if DOF > 0) =====
            # Standard errors of coefficients
            var_coef = var_res * np.diag(XtX_inv)
            se_coef = np.sqrt(var_coef)


            # t-statistics
            t_stats = coefficients / se_coef

            # p-values
            p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), dof))


            # Confidence intervals
            t_critical = stats.t.ppf(0.975, dof)
            ci_lower = coefficients - t_critical * se_coef
            ci_upper = coefficients + t_critical * se_coef

            results.update({
                'se_coef': pd.Series(se_coef, index=X.columns),
                't_stats': pd.Series(t_stats, index=X.columns),
                'p_values': pd.Series(p_values, index=X.columns),
                'ci_lower': pd.Series(ci_lower, index=X.columns),
                'ci_upper': pd.Series(ci_upper, index=X.columns)
            })
        else:            st.warning(f"⚠️ Saturated model (DOF={dof}): Cannot compute adjusted R², RMSE, or statistical tests")


        # ===== CONDITIONAL: Cross-validation (only if DOF > 0 and n ≤ 100) =====
        if return_diagnostics and dof > 0 and n_samples <= 100:
            try:
                cv_predictions = np.zeros(n_samples)

                for i in range(n_samples):
                    # Remove sample i
                    X_cv = np.delete(X_mat, i, axis=0)
                    y_cv = np.delete(y_vec, i)


                    # Fit model without sample i
                    XtX_cv = X_cv.T @ X_cv
                    XtX_cv_inv = np.linalg.inv(XtX_cv)
                    coef_cv = XtX_cv_inv @ (X_cv.T @ y_cv)


                    # Predict sample i
                    cv_predictions[i] = X_mat[i, :] @ coef_cv

                cv_residuals = y_vec - cv_predictions
                rss_cv = np.sum(cv_residuals**2)
                rmsecv = np.sqrt(rss_cv / n_samples)
                q2 = 1 - (rss_cv / (results.get('var_y', 1) * n_samples))

                results.update({
                    'cv_predictions': cv_predictions,
                    'cv_residuals': cv_residuals,
                    'rmsecv': rmsecv,
                    'q2': q2
                })
            except Exception as e:
                pass  # CV failed, skip

        # ===== ALWAYS: Leverage (independent of DOF) =====
        try:
            leverage = np.diag(X_mat @ XtX_inv @ X_mat.T)
            results['leverage'] = leverage
        except Exception as e:
            results['leverage'] = None

        # ===== ALWAYS: VIF (independent of DOF) =====
        if n_features > 1:
            try:
                vif = []

                # Center the X matrix (subtract column means)
                X_centered = X_mat - X_mat.mean(axis=0)

                for i in range(n_features):
                    if X.columns[i] == 'Intercept':
                        vif.append(np.nan)

                    else:
                        # Formula: sum(X_centered_i^2) * diag(XtX_inv)_i
                        ss_centered = np.sum(X_centered[:, i]**2)
                        vif_value = ss_centered * XtX_inv[i, i]
                        vif.append(vif_value)

                results['vif'] = pd.Series(vif, index=X.columns)
            except Exception as e:
                results['vif'] = None
        else:
            results['vif'] = None

    except np.linalg.LinAlgError as e:
        st.error(f"❌ Linear algebra error: {e}")
        return None
    except Exception as e:
        st.error(f"❌ Unexpected error in model fitting: {e}")
        import traceback
        traceback.print_exc()
        return None

    return results


def statistical_summary(model_results, X, y):
    """
    Generate statistical summary for ANY design (generic)

    ADAPTS TO AVAILABLE DATA:
    - Always shows: basic metrics, coefficients, VIF, leverage
    - Conditionally shows: R²/RMSE (if DOF>0), pure error (if replicates exist)

    Args:
        model_results: dict from fit_mlr_model()
        X: original predictor DataFrame
        y: original response Series

    Returns:
        dict with summary statistics (all available metrics)
    """
    summary = {
        'n_samples': model_results['n_samples'],
        'n_features': model_results['n_features'],
        'dof': model_results['dof']
    }

    # Add available metrics
    if 'r_squared' in model_results:
        summary['r_squared'] = model_results['r_squared']
    if 'rmse' in model_results:
        summary['rmse'] = model_results['rmse']
    if 'var_res' in model_results:
        summary['var_res'] = model_results['var_res']

    if 'var_y' in model_results:
        summary['var_y'] = model_results['var_y']

    # VIF summary
    if 'vif' in model_results and model_results['vif'] is not None:
        vif_clean = model_results['vif'].dropna()
        if not vif_clean.empty:
            summary['max_vif'] = vif_clean.max()
            summary['mean_vif'] = vif_clean.mean()

    # Leverage summary
    if 'leverage' in model_results and model_results['leverage'] is not None:
        summary['max_leverage'] = model_results['leverage'].max()
        summary['mean_leverage'] = model_results['leverage'].mean()

    # Cross-validation summary
    if 'q2' in model_results:
        summary['q2'] = model_results['q2']
        summary['rmsecv'] = model_results['rmsecv']

    return summary


# ============================================================================
# HELPER FUNCTIONS FOR UI
# ============================================================================


def create_term_selection_matrix(x_vars):
    """
    Create an interaction matrix for term selection

    Args:
        x_vars: list of X variable names

    Returns:
        DataFrame with shape (n_vars, n_vars) initialized to 1
    """
    n_vars = len(x_vars)
    matrix = pd.DataFrame(1, index=x_vars, columns=x_vars)
    return matrix


def display_term_selection_ui(x_vars, key_prefix="", design_analysis=None):
    """
    Display interactive term selection UI with intelligent disabling per design rules.

    Args:
        x_vars: list of X variable names
        key_prefix: prefix for streamlit keys
        design_analysis: dict from analyze_design_structure() with design type info

    Returns:
        tuple: (term_matrix DataFrame, selected_terms dict)
    """
    # Default design_analysis if not provided
    if design_analysis is None:
        design_analysis = {
            'design_type': 'unknown',
            'recommended_terms': {'interactions': True, 'quadratic': True},
            'qualitative_vars': []
        }

    n_vars = len(x_vars)
    term_matrix = pd.DataFrame(1, index=x_vars, columns=x_vars)


    # ════════════════════════════════════════════════════════════════════
    # APPLY RULES: Determine what's disabled based on design_type
    # ════════════════════════════════════════════════════════════════════

    # RULE 1 & 2: For 2-level, disable all quadratic
    disable_all_quadratic = (design_analysis['design_type'] == "2-level")


    # RULE 3: For qualitative variables, disable their interactions and quadratic
    qual_vars = design_analysis.get('qualitative_vars', [])


    # ════════════════════════════════════════════════════════════════════
    # QUADRATIC TERMS (diagonal)

    # ════════════════════════════════════════════════════════════════════

    st.markdown("**Quadratic Terms:**")
    quad_cols = st.columns(min(n_vars, 4))

    for i, var in enumerate(x_vars):
        with quad_cols[i % len(quad_cols)]:
            # Determine if this quadratic should be disabled
            should_disable = (
                disable_all_quadratic or  # Rule 1: 2-level disables all
                var in qual_vars          # Rule 3: qualitative disables its own
            )


            # Determine default value
            should_check = not should_disable

            selected = st.checkbox(
                f"{var}²",
                value=should_check,  # ← Pre-set based on rules
                disabled=should_disable,  # ← Disable based on rules
                key=f"{key_prefix}_quad_{i}"
            )

            term_matrix.iloc[i, i] = 1 if selected else 0

    # Add warning if quadratic disabled
    if disable_all_quadratic:
        st.caption("⚠️ Quadratic terms disabled (2-level design cannot fit)")
    if qual_vars:
        st.caption(f"⚠️ Qualitative variables {qual_vars}: no quadratic")


    # ════════════════════════════════════════════════════════════════════
    # INTERACTION TERMS (off-diagonal)

    # ════════════════════════════════════════════════════════════════════

    if n_vars > 1:
        st.markdown("**Interaction Terms:**")
        interactions = []

        for i in range(n_vars):
            for j in range(i+1, n_vars):
                interactions.append((i, j, f"{x_vars[i]}*{x_vars[j]}"))

        int_cols = st.columns(min(len(interactions), 3))

        for idx, (i, j, name) in enumerate(interactions):
            with int_cols[idx % len(int_cols)]:
                # Determine if this interaction should be disabled
                should_disable = (
                    x_vars[i] in qual_vars or  # Rule 3: can't interact with qualitative
                    x_vars[j] in qual_vars
                )


                # Default: enabled (interactions are OK unless qualitative involved)
                should_check = not should_disable

                selected = st.checkbox(
                    name,
                    value=should_check,  # ← Pre-set based on rules
                    disabled=should_disable,  # ← Disable if qualitative involved
                    key=f"{key_prefix}_int_{i}_{j}"
                )

                term_matrix.iloc[i, j] = 1 if selected else 0
                term_matrix.iloc[j, i] = 1 if selected else 0

        if qual_vars:
            st.caption(f"⚠️ Qualitative variables {qual_vars}: no interactions")


    # ════════════════════════════════════════════════════════════════════
    # Build selected_terms dict
    # ════════════════════════════════════════════════════════════════════

    selected_terms = {
        'linear': x_vars.copy(),
        'interactions': [],
        'quadratic': []
    }

    # Extract selected interactions
    for i in range(n_vars):
        for j in range(i+1, n_vars):
            if term_matrix.iloc[i, j] == 1:
                selected_terms['interactions'].append(f"{x_vars[i]}*{x_vars[j]}")


    # Extract selected quadratic
    for i in range(n_vars):
        if term_matrix.iloc[i, i] == 1:
            selected_terms['quadratic'].append(f"{x_vars[i]}^2")

    return term_matrix, selected_terms


def build_model_formula(y_var, selected_terms, include_intercept=True):
    """
    Build a readable model formula string with proper coefficient nomenclature.

    NOMENCLATURE RULES:
    - Linear terms: b1, b2, b3, ... (sequential subscripts based on variable order)
    - Interaction terms: bij where i,j are indices of variables in the interaction
      Example: b12 for X1*X2, b23 for X2*X3
    - Quadratic terms: bii where i is the variable index
      Example: b11 for X1^2, b22 for X2^2, b33 for X3^2

    Args:
        y_var: response variable name
        selected_terms: dict with 'linear', 'interactions', 'quadratic' lists
        include_intercept: bool, include intercept term

    Returns:
        str: model formula with proper coefficient notation
    """
    terms = []

    # Add intercept
    if include_intercept:
        terms.append("b₀")

    # Build mapping: variable name -> index (1-based)
    all_vars_ordered = []
    for var in selected_terms['linear']:
        if var not in all_vars_ordered:
            all_vars_ordered.append(var)

    var_to_index = {var: i+1 for i, var in enumerate(all_vars_ordered)}

    # LINEAR TERMS: b1·X1, b2·X2, b3·X3, ...
    for var in selected_terms['linear']:
        idx = var_to_index[var]
        terms.append(f"b{idx}·{var}")

    # INTERACTION TERMS: b12·X1*X2, b13·X1*X3, etc.
    # Parse interaction term to extract variable names
    for interaction_term in selected_terms['interactions']:
        # Interaction format: "X1*X2" or "Var1*Var2"
        parts = interaction_term.split('*')
        if len(parts) == 2:
            var1, var2 = parts[0].strip(), parts[1].strip()
            if var1 in var_to_index and var2 in var_to_index:
                idx1 = var_to_index[var1]
                idx2 = var_to_index[var2]
                # Ensure lower index first (b12, not b21)
                if idx1 > idx2:
                    idx1, idx2 = idx2, idx1
                terms.append(f"b{idx1}{idx2}·{interaction_term}")

    # QUADRATIC TERMS: b11·X1^2, b22·X2^2, etc.
    for quadratic_term in selected_terms['quadratic']:
        # Quadratic format: "X1^2" or "Var1^2"
        var_name = quadratic_term.replace('^2', '').strip()
        if var_name in var_to_index:
            idx = var_to_index[var_name]
            terms.append(f"b{idx}{idx}·{quadratic_term}")

    formula = f"{y_var} = {' + '.join(terms)}"
    return formula


def design_analysis(X_model, X_data, replicate_info):
    """
    Analyze design matrix without Y variable

    Args:
        X_model: model matrix DataFrame (with intercept and interactions)
        X_data: original X data
        replicate_info: dict from detect_replicates() or None

    Returns:
        dict with design analysis results
    """
    X_mat = X_model.values
    n_samples, n_features = X_mat.shape

    # Check rank
    rank = np.linalg.matrix_rank(X_mat)
    if rank < n_features:
        st.error(f"⚠️ Design matrix is rank deficient! Rank={rank}, Features={n_features}")
        return None

    # Degrees of freedom
    dof = n_samples - n_features

    # Compute dispersion matrix
    XtX = X_mat.T @ X_mat
    XtX_inv = np.linalg.inv(XtX)


    # Leverage
    leverage = np.diag(X_mat @ XtX_inv @ X_mat.T)


    # VIF
    vif = []
    X_centered = X_mat - X_mat.mean(axis=0)

    for i in range(n_features):
        if X_model.columns[i] == 'Intercept':
            vif.append(np.nan)

        else:
            ss_centered = np.sum(X_centered[:, i]**2)
            vif_value = ss_centered * XtX_inv[i, i]
            vif.append(vif_value)

    results = {
        'n_samples': n_samples,
        'n_features': n_features,
        'dof': dof,
        'X': X_model,
        'XtX_inv': XtX_inv,
        'leverage': leverage,
        'vif': pd.Series(vif, index=X_model.columns)
    }

    # Add experimental variance if replicates exist
    if replicate_info:
        results['experimental_std'] = replicate_info['pooled_std']
        results['experimental_dof'] = replicate_info['pooled_dof']

        # Calculate t-critical for predictions
        t_critical = stats.t.ppf(0.975, replicate_info['pooled_dof'])
        results['t_critical'] = t_critical

        # Prediction standard errors
        prediction_se = replicate_info['pooled_std'] * np.sqrt(leverage)
        results['prediction_se'] = prediction_se
    return results


# ============================================================================
# UI DISPLAY FUNCTIONS
# ============================================================================


def show_model_computation_ui(data, dataset_name):
    """
    Display the MLR Model Computation UI

    Args:
        data: DataFrame with experimental data
        dataset_name: name of the current dataset
    """
    # Import helper functions from parent module (avoid circular imports)

    # Note: create_model_matrix and fit_mlr_model are already in this module
    # We only need detect_replicates and detect_central_points from mlr_doe
    from mlr_doe import detect_replicates, detect_central_points

    st.markdown("## 🔧 MLR Model Computation")

    st.markdown("*Equivalent to DOE_model_computation.r*")


    # DATA PREVIEW SECTION
    st.markdown("### 👁️ Data Preview")
    with st.expander("Show current dataset", expanded=True):
        # Full scrollable dataframe
        st.dataframe(data, use_container_width=True, height=400)

        col_info1, col_info2, col_info3 = st.columns(3)
        with col_info1:
            st.metric("Total Samples", data.shape[0])
        with col_info2:
            st.metric("Total Variables", data.shape[1])
        with col_info3:
            numeric_cols_count = len(data.select_dtypes(include=[np.number]).columns)

            st.metric("Numeric Variables", numeric_cols_count)


    st.markdown("---")


    # Variable and sample selection
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📊 Variable Selection")

        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()

        if not numeric_columns:
            st.error("❌ No numeric columns found!")
            return

        # X variables
        x_vars = st.multiselect(
            "Select X variables (predictors):",
            numeric_columns,
            default=numeric_columns[:min(3, len(numeric_columns))],
            key="mlr_x_vars_widget"
        )


        # Y variable (OPTIONAL - for design analysis mode)
        remaining_cols = [col for col in numeric_columns if col not in x_vars]

        # Always show Y variable selector (even if no remaining cols - for Design Analysis Mode)
        y_options = ["(None - Design Analysis Only)"] + remaining_cols
        y_var_selected = st.selectbox(
            "Select Y variable (response - optional):",
            y_options,
            key="mlr_y_var_widget",
            help="Select '(None)' for design screening without response variable"
        )


        # Parse selection
        if y_var_selected == "(None - Design Analysis Only)":
            y_var = None
            st.info("**Design Analysis Mode**: No Y variable - will analyze design matrix only (VIF, Leverage, Dispersion)")

        else:
            y_var = y_var_selected

        # Show selected variables info
        if x_vars and y_var:
            x_vars_str = [str(var) for var in x_vars]
            st.info(f"Model: {y_var} ~ {' + '.join(x_vars_str)}")
        elif x_vars and y_var is None:
            x_vars_str = [str(var) for var in x_vars]
            st.info(f"Design Matrix: {' + '.join(x_vars_str)}")

    with col2:
        st.markdown("### 🎯 Sample Selection")


        # Sample selection options
        sample_selection_mode = st.radio(
            "Select samples:",
            ["Use all samples", "Select by index", "Select by range"],
            key="sample_selection_mode"
        )

        if sample_selection_mode == "Use all samples":
            selected_samples = data.index.tolist()

            st.success(f"Using all {len(selected_samples)} samples")

        elif sample_selection_mode == "Select by index":
            sample_input = st.text_input(
                "Enter sample indices (1-based, comma-separated or ranges):",
                value=f"1-{data.shape[0]}",
                help="Examples: 1,2,5-10,15 or 1-20"
            )

            try:
                selected_indices = []
                for part in sample_input.split(','):
                    part = part.strip()
                    if '-' in part:
                        start, end = map(int, part.split('-'))
                        selected_indices.extend(range(start-1, end))

                    else:
                        selected_indices.append(int(part)-1)

                selected_indices = sorted(list(set(selected_indices)))
                valid_indices = [i for i in selected_indices if 0 <= i < len(data)]
                selected_samples = data.index[valid_indices].tolist()


                st.success(f"Selected {len(selected_samples)} samples")


            except Exception as e:
                st.error(f"Invalid format: {e}")
                selected_samples = data.index.tolist()


        else:  # Select by range
            col_range1, col_range2 = st.columns(2)
            with col_range1:
                start_idx = st.number_input("From sample:", 1, len(data), 1)
            with col_range2:
                end_idx = st.number_input("To sample:", start_idx, len(data), len(data))

            selected_samples = data.index[start_idx-1:end_idx].tolist()

            st.success(f"Selected {len(selected_samples)} samples (rows {start_idx}-{end_idx})")


        # Show selected samples preview
        if len(selected_samples) < len(data):
            with st.expander("Preview selected samples"):
                st.dataframe(data.loc[selected_samples].head(10), use_container_width=True)


    st.markdown("---")


    # Interactive Term Selection UI
    if not x_vars:
        st.warning("Please select X variables first")
        return

    st.markdown("### 🎛️ Model Configuration")


    # ────────────────────────────────────────────────────────────────
    # NEW: Auto-analyze design structure
    # ────────────────────────────────────────────────────────────────

    st.markdown("#### 🔍 Design Structure Analysis")


    # Prepare X data for analysis
    X_for_analysis = data.loc[selected_samples, x_vars].copy()

    with st.spinner("Analyzing design structure..."):
        try:
            design_structure_info = analyze_design_structure(X_for_analysis)


            # Display design analysis results
            col_analysis1, col_analysis2 = st.columns([2, 1])

            with col_analysis1:
                st.info(design_structure_info['interpretation'])

            with col_analysis2:
                st.metric("Design Type", design_structure_info['design_type'])

                st.metric("Center Points", len(design_structure_info['center_points_indices']))


            # Show warnings if any
            if design_structure_info['warnings']:
                for warning_msg in design_structure_info['warnings']:
                    st.warning(warning_msg)


            # Display levels per variable
            st.markdown("**Levels per Variable (excluding center points)**")
            levels_df = pd.DataFrame([
                {
                    'Variable': var_name,
                    'Levels': n_levels,
                    'Type': 'Quantitative' if design_structure_info['is_quantitative'].get(var_name, True) else 'Categorical'
                }
                for var_name, n_levels in design_structure_info['n_levels_per_var'].items()
            ])

            st.dataframe(levels_df, use_container_width=True, hide_index=True)


        except Exception as e:
            st.warning(f"⚠️ Design analysis failed: {str(e)}")

            st.info("Using default configuration (intercept + linear terms)")

            # Fallback defaults
            design_structure_info = {
                'design_type': 'unknown',
                'recommended_terms': {
                    'intercept': True,
                    'linear': True,
                    'interactions': False,
                    'quadratic': False
                },
                'n_levels_per_var': {var: 2 for var in x_vars},
                'is_quantitative': {var: True for var in x_vars},
                'center_points_indices': [],
                'warnings': []
            }

    # ════════════════════════════════════════════════════════════════════
    # 🔧 MODEL CONFIGURATION (CAT-STYLE)

    # ════════════════════════════════════════════════════════════════════

    st.markdown("### 🔧 Model Configuration")


    # Show design analysis info as compact caption
    if design_structure_info['design_type'] == "2-level":
        st.caption("✅ 2-Level Design - Interactions OK, no quadratic")
    elif design_structure_info['design_type'] == ">2-level":
        st.caption("✅ >2-Level Design - All terms available")
    elif design_structure_info['design_type'] == "qualitative_only":
        st.caption("⚠️ Qualitative Only - Linear terms only")

    else:
        st.caption(f"ℹ️ Design: {design_structure_info['design_type']}")


    # ════════════════════════════════════════════════════════════════════
    # TOP CONTROLS (2 checkboxes CAT-style)

    # ════════════════════════════════════════════════════════════════════

    col_top1, col_top2 = st.columns(2)

    with col_top1:
        include_intercept = st.checkbox(
            "Include intercept",
            value=True,
            disabled=True,
            help="Always included in model"
        )

    with col_top2:
        # Disable for qualitative-only designs
        should_disable_higher_order = (design_structure_info['design_type'] == "qualitative_only")

        include_higher_order = st.checkbox(
            "Include higher-order terms",
            value=(design_structure_info['recommended_terms']['interactions'] or design_structure_info['recommended_terms']['quadratic']),
            disabled=should_disable_higher_order,
            help="Interactions and/or quadratic" if not should_disable_higher_order else "Not available for qualitative-only"
        )


    st.markdown("---")


    # ════════════════════════════════════════════════════════════════════
    # TERM SELECTION MATRIX (if higher-order enabled)

    # ════════════════════════════════════════════════════════════════════

    if include_higher_order and design_structure_info['design_type'] != "qualitative_only":

        st.markdown("### 📊 Select Model Terms")

        st.info("Use the matrix below to select interactions and quadratic terms")


        # Get the term selection matrix UI
        # Pass design_structure_info so function can apply rules intelligently
        term_matrix, selected_terms = display_term_selection_ui(
            x_vars,
            key_prefix="model_config",
            design_analysis=design_structure_info  # ← Pass this!
        )


        # Note: Rules are now applied inside display_term_selection_ui()

        # No need for manual disabling logic here

        # ════════════════════════════════════════════════════════════════
        # Display Summary
        # ════════════════════════════════════════════════════════════════

        st.markdown("#### Summary")

        col_sum1, col_sum2, col_sum3 = st.columns(3)

        with col_sum1:
            linear_count = len(selected_terms['linear'])

            st.metric("Linear Terms", linear_count)

        with col_sum2:
            interaction_count = len(selected_terms['interactions'])

            st.metric("Interactions", interaction_count)

        with col_sum3:
            quadratic_count = len(selected_terms['quadratic'])

            st.metric("Quadratic Terms", quadratic_count)


        # Saturation check
        n_total = 1 + linear_count + interaction_count + quadratic_count

        st.markdown("---")

        if n_total > len(X_for_analysis):
            st.error(f"❌ Model is saturated! {n_total} terms > {len(X_for_analysis)} observations")
        elif n_total >= len(X_for_analysis) * 0.8:
            st.warning(f"⚠️  Model is near saturation: {n_total} terms ≈ {len(X_for_analysis)} observations")

        else:
            st.success(f"✅ Model has {len(X_for_analysis) - n_total} degrees of freedom")


    else:
        # Higher-order disabled (qualitative-only or user unchecked)

        st.info("📊 **Select Model Terms** - Higher-order terms disabled")


        # Build simple selected_terms with linear only
        selected_terms = {
            'linear': x_vars.copy(),
            'interactions': [],
            'quadratic': []
        }

        # Create empty term_matrix (all zeros)
        term_matrix = create_term_selection_matrix(x_vars)
        for i in range(len(x_vars)):
            for j in range(len(x_vars)):
                term_matrix.iloc[i, j] = 0

    st.markdown("---")


    # Model Settings
    st.markdown("### ⚙️ Additional Model Settings")

    col_set1, col_set2 = st.columns(2)

    with col_set1:
        exclude_central_points = st.checkbox(
            "Exclude central points (0,0,0...)",
            value=False,
            help="Central points are typically used only for validation in factorial designs"
        )

    with col_set2:
        # Variance method selector
        variance_method = st.radio(
            "Variance estimation method:",
            ["Residuals", "Independent measurements"],
            help="Choose how to estimate model error variance"
        )

        run_cv = st.checkbox("Run cross-validation", value=True,
                            help="Leave-one-out CV (only for n≤100)")


    # Display model formula (only if Y variable is selected)

    st.markdown("---")
    if y_var:
        st.markdown("### 📐 Postulated Model Formula")

        try:
            formula = build_model_formula(y_var, selected_terms, include_intercept)

            st.code(formula, language="text")

        except Exception as e:
            st.warning(f"Could not generate formula display: {str(e)}")

            st.code(f"{y_var} = b0 + b1·X + ... (formula generation error)", language="text")

    else:
        st.markdown("### 📐 Design Structure")

        # Show design structure without Y variable
        terms_list = []
        if include_intercept:
            terms_list.append("Intercept")
        terms_list.extend(selected_terms['linear'])
        terms_list.extend(selected_terms['interactions'])
        terms_list.extend(selected_terms['quadratic'])


        st.code(f"Design Matrix Terms: {', '.join(str(t) for t in terms_list)}", language="text")


    # Summary of selected terms
    total_terms = len(selected_terms['linear']) + len(selected_terms['interactions']) + len(selected_terms['quadratic'])
    if include_intercept:
        total_terms += 1

    if y_var:
        st.info(f"""
        **Model Summary:**
        - Total parameters: {total_terms}
        - Response variable: {y_var}
        - Variance method: {variance_method}
        """)

    else:
        st.info(f"""
        **Design Analysis Summary:**
        - Total design terms: {total_terms}
        - Mode: Design screening (no response variable)
        - Analysis: Dispersion matrix, VIF, Leverage
        """)


    # Use term_matrix as interaction_matrix for backward compatibility
    interaction_matrix = term_matrix

    # Fit model or analyze design button
    button_text = "🚀 Fit MLR Model" if y_var else "🔍 Analyze Design"
    button_type = "primary"

    if st.button(button_text, type=button_type):
        try:
            # Prepare data with selected samples
            X_data = data.loc[selected_samples, x_vars].copy()


            # Handle Y variable (if present)
            if y_var:
                y_data = data.loc[selected_samples, y_var].copy()

                # Remove missing values
                valid_idx = ~(X_data.isnull().any(axis=1) | y_data.isnull())
                X_data = X_data[valid_idx]
                y_data = y_data[valid_idx]

                if len(X_data) < len(x_vars) + 1:
                    st.error("❌ Not enough samples for model fitting!")
                    return

                st.info(f"ℹ️ Using {len(X_data)} samples after removing missing values")

            else:
                # Design analysis mode - no Y variable
                # Remove rows with missing X values only
                valid_idx = ~X_data.isnull().any(axis=1)
                X_data = X_data[valid_idx]
                y_data = None

                if len(X_data) < len(x_vars):
                    st.error("❌ Not enough samples for design analysis!")
                    return

                st.info(f"ℹ️ Using {len(X_data)} samples for design analysis")


            # Detect and optionally exclude central points
            central_points = detect_central_points(X_data)

            if central_points:
                st.info(f"🎯 Detected {len(central_points)} central point(s) at indices: {[i+1 for i in central_points]}")

                if exclude_central_points:
                    # Store original indices before filtering
                    central_samples_original = X_data.index[central_points].tolist()


                    # Remove central points from modeling data
                    X_data = X_data.drop(X_data.index[central_points])
                    if y_data is not None:
                        y_data = y_data.drop(y_data.index[central_points])


                    st.warning(f"⚠️ Excluded {len(central_points)} central point(s) from analysis")

                    st.info(f"ℹ️ Using {len(X_data)} samples (excluding central points)")


                    # Store excluded central points for later validation (only if Y exists)
                    if y_var:
                        st.session_state.mlr_central_points = {
                            'X': data.loc[central_samples_original, x_vars],
                            'y': data.loc[central_samples_original, y_var],
                            'indices': central_samples_original
                        }
                else:
                    st.info("ℹ️ Central points included in the analysis")


            # Use term_matrix if user selected specific terms
            if 'term_matrix' in locals() and term_matrix is not None:
                interaction_matrix = term_matrix

            # Validate term_matrix
            if interaction_matrix is None:
                st.error("❌ Term selection matrix is None! Cannot create model.")

                st.info("This is a bug - please report with your data configuration.")
                return

            # Create model matrix
            with st.spinner("Creating model matrix..."):
                X_model, term_names = create_model_matrix(
                    X_data,
                    include_intercept=include_intercept,
                    include_interactions=True,  # Always True - term_matrix controls selection
                    include_quadratic=True,  # Always True - term_matrix controls selection
                    interaction_matrix=interaction_matrix
                )


            st.success(f"✅ Model matrix created: {X_model.shape[0]} × {X_model.shape[1]}")

            st.write(f"**Model terms:** {term_names}")


            # BRANCH: Model fitting vs Design analysis
            if y_var is not None:
                # ===== MODEL FITTING MODE (Y variable present) =====
                with st.spinner("Fitting MLR model..."):
                    model_results = fit_mlr_model(X_model, y_data, return_diagnostics=run_cv)

                if model_results is None:
                    return

                # Store results
                st.session_state.mlr_model = model_results
                st.session_state.mlr_y_var = y_var
                st.session_state.mlr_x_vars = x_vars

                st.success("✅ MLR model fitted successfully!")


                # Show model results (calling the display function)
                _display_model_results(
                    model_results, y_var, x_vars, data, selected_samples,
                    central_points, exclude_central_points, X_data, y_data
                )


            else:
                # ===== DESIGN ANALYSIS MODE (No Y variable) =====
                with st.spinner("Analyzing design matrix..."):
                    # In design analysis mode, we don't have Y values to detect replicates
                    # Pass None to skip experimental variance calculations
                    replicate_info = None

                    # Run design analysis
                    design_results = design_analysis(X_model, X_data, replicate_info)

                if design_results is None:
                    return

                st.success("✅ Design analysis completed successfully!")


                # Display design analysis results
                _display_design_analysis_results(design_results, x_vars, X_data)


        except Exception as e:
            st.error(f"❌ Error fitting model: {str(e)}")
            import traceback
            # Always show traceback for debugging
            with st.expander("🐛 Debug Info (click to expand)"):
                st.code(traceback.format_exc())


def _display_model_results(model_results, y_var, x_vars, data, selected_samples,
                           central_points, exclude_central_points, X_data, y_data):
    """
    Display complete model results with diagnostics and statistical tests

    GENERIC IMPLEMENTATION - Works with ANY dataset structure:
    - With or without replicates
    - With or without central points
    - Any number of samples and variables

    ALWAYS DISPLAYS:
    - R², RMSE (model quality)
    - VIF (multicollinearity)
    - Leverage (influential points)
    - Coefficients with significance tests
    - Cross-validation (if enabled)

    CONDITIONALLY DISPLAYS (if data structure allows):
    - Replicate analysis (if replicates exist)
    - Lack of fit test (if replicates exist)
    - Factor effects F-test (if replicates exist)
    - Central point validation (if central points excluded)
    """
    # Import helper function
    from mlr_doe import detect_replicates

    # DEBUG: Show what keys are in model_results
    with st.expander("🔍 Model Results Debug Info"):
        st.write("**Available keys in model_results:**")

        st.write(list(model_results.keys()))

        st.write("**Model results summary:**")
        for key, value in model_results.items():
            if isinstance(value, (int, float, np.integer, np.floating)):
                st.write(f"- {key}: {value}")
            elif isinstance(value, (pd.Series, pd.DataFrame)):
                st.write(f"- {key}: {type(value).__name__} with shape {value.shape}")
            elif isinstance(value, np.ndarray):
                st.write(f"- {key}: numpy array with shape {value.shape}")

            else:
                st.write(f"- {key}: {type(value)}")


    # Show number of experiments used for fitting
    st.info(f"📊 **Model fitted using {model_results['n_samples']} experiments** (after excluding central points if selected)")


    # ===== ALWAYS: Basic Model Quality =====
    st.markdown("### 📈 Model Quality Summary")

    summary_col1, summary_col2 = st.columns(2)

    with summary_col1:
        if 'r_squared' in model_results:
            var_explained_pct = model_results['r_squared'] * 100
            st.metric("% Explained Variance (adjusted R²)", f"{var_explained_pct:.2f}%")

    with summary_col2:
        if 'rmse' in model_results:
            st.metric("Std Dev of Residuals (RMSE)", f"{model_results['rmse']:.4f}")


    # ========== AUTOMATIC REPLICATE DETECTION ==========
    # ALWAYS use ALL original data (including central points) for experimental variability calculation
    all_X_data = data.loc[selected_samples, x_vars].copy()
    all_y_data = data.loc[selected_samples, y_var].copy()
    all_valid_idx = ~(all_X_data.isnull().any(axis=1) | all_y_data.isnull())
    all_X_data = all_X_data[all_valid_idx]
    all_y_data = all_y_data[all_valid_idx]

    replicate_info_full = detect_replicates(all_X_data, all_y_data)


    # ===== CONDITIONAL: Replicate Analysis (only if replicates exist) =====
    if replicate_info_full:
        _display_replicate_analysis(replicate_info_full, model_results, central_points,
                                    exclude_central_points, y_data, all_y_data)

    else:
        st.info("ℹ️ No replicates detected - pure experimental error cannot be estimated")


    # ===== CONDITIONAL: Central Points Validation (only if excluded) =====
    if central_points and exclude_central_points:
        _display_central_points_validation(central_points)


    # ===== CONDITIONAL: Model Data Replicates Check =====
    replicate_info = detect_replicates(X_data, y_data)
    if replicate_info:
        _display_model_data_replicates(replicate_info, replicate_info_full)


    # ===== ALWAYS: Statistical Analysis Summary (adapts to available data) =====
    _display_statistical_summary(model_results, all_y_data, y_data, central_points,
                                 exclude_central_points, replicate_info_full)


    # ===== ALWAYS: Dispersion Matrix, VIF, Leverage =====
    _display_model_summary(model_results)


    # ===== CONDITIONAL: Error Comparison (only if replicates exist) =====
    if replicate_info and 'rmse' in model_results:
        _display_error_comparison(model_results, replicate_info)


    # ===== ALWAYS: Coefficients Table =====
    _display_coefficients_table(model_results)


    # ===== ALWAYS: Coefficients Bar Plot =====
    _display_coefficients_barplot(model_results, y_var)


    # ===== ALWAYS: Cross-Validation Results (if CV was run) =====
    if 'q2' in model_results:
        st.markdown("### 🔄 Cross-Validation Results")

        cv_col1, cv_col2 = st.columns(2)
        with cv_col1:
            st.metric("RMSECV", f"{model_results['rmsecv']:.4f}")
        with cv_col2:
            st.metric("Q² (LOO-CV)", f"{model_results['q2']:.4f}")


def _display_replicate_analysis(replicate_info_full, model_results, central_points,
                                exclude_central_points, y_data, all_y_data):
    """Display experimental variability analysis from replicates"""
    st.markdown("### 🔬 Experimental Variability (Pure Error)")

    st.info("""
    **Pure experimental error** estimated from replicate measurements
    (including ALL points - central points always included for experimental error calculation).
    This represents the baseline measurement variability.
    """)

    rep_col1, rep_col2, rep_col3, rep_col4 = st.columns(4)

    with rep_col1:
        st.metric("Replicate Groups", replicate_info_full['n_replicate_groups'])
    with rep_col2:
        st.metric("Total Replicates", replicate_info_full['total_replicates'])
    with rep_col3:
        st.metric("Pooled Std Dev (s_exp)", f"{replicate_info_full['pooled_std']:.4f}")
    with rep_col4:
        st.metric("Pure Error DOF", replicate_info_full['pooled_dof'])

    with st.expander("📋 Replicate Groups Details"):
        rep_data = []
        for i, group in enumerate(replicate_info_full['group_stats'], 1):
            rep_data.append({
                'Group': i,
                'Samples': ', '.join([str(idx+1) for idx in group['indices']]),
                'N': group['n_replicates'],
                'Mean Y': f"{group['mean']:.4f}",
                'Std Dev': f"{group['std']:.4f}",
                'Variance': f"{group['variance']:.6f}",
                'DOF': group['dof']
            })

        rep_df = pd.DataFrame(rep_data)

        st.dataframe(rep_df, use_container_width=True)


        st.markdown(f"""
        **Pooled Standard Deviation Formula:**

        s_pooled = √[Σ(s²ᵢ × dfᵢ) / Σ(dfᵢ)]

        Where s²ᵢ is the variance of group i and dfᵢ is its degrees of freedom.

        **Result:** s_exp = {replicate_info_full['pooled_std']:.4f}
        (from {replicate_info_full['pooled_dof']} degrees of freedom)
        """)


    # Statistical tests
    _display_statistical_tests(model_results, replicate_info_full, central_points,
                               exclude_central_points, y_data, all_y_data)


def _display_statistical_tests(model_results, replicate_info_full, central_points,
                               exclude_central_points, y_data, all_y_data):
    """Display statistical tests for model quality"""
    st.markdown("---")

    st.markdown("### 📊 Statistical Analysis of Model Quality")


    # 1. DoE Factor Variability vs Experimental Variability
    st.markdown("#### 1️⃣ DoE Factor Variability vs Experimental Variability")

    if 'var_y' in model_results:
        # Determine which data to use for DoE variance
        if central_points and exclude_central_points:
            var_y_doe = np.var(y_data, ddof=1)
            dof_y_doe = len(y_data) - 1

            st.info(f"""
            **DoE Variability**: Calculated from {len(y_data)} DoE experimental points
            (central points excluded as they don't contribute to factor-induced variation).
            """)

        else:
            var_y_doe = model_results['var_y']
            dof_y_doe = len(all_y_data) - 1

            st.info("""
            **DoE Variability**: Calculated from all experimental points
            (central points included in model).
            """)


        # F-test: s²_DoE / s²_exp
        f_global = var_y_doe / replicate_info_full['pooled_variance']
        f_crit_global = stats.f.ppf(0.95, dof_y_doe, replicate_info_full['pooled_dof'])
        p_global = 1 - stats.f.cdf(f_global, dof_y_doe, replicate_info_full['pooled_dof'])

        test_col1, test_col2, test_col3 = st.columns(3)

        with test_col1:
            st.metric("DoE Variance (s²_DoE)", f"{var_y_doe:.6f}")

            st.metric("DOF", dof_y_doe)

        with test_col2:
            st.metric("Experimental Variance (s²_exp)", f"{replicate_info_full['pooled_variance']:.6f}")

            st.metric("DOF", replicate_info_full['pooled_dof'])

        with test_col3:
            st.metric("F-statistic", f"{f_global:.2f}")

            st.metric("p-value", f"{p_global:.4f}")

        if p_global < 0.05:
            st.success(f"✅ DoE factors induce significant variation in response (p={p_global:.4f})")

            st.info("The experimental factors have meaningful effects on the response variable.")

        else:
            st.warning(f"⚠️ DoE factor effects not significantly different from experimental noise (p={p_global:.4f})")

            st.info("The factors may have weak effects or the experimental error is too large.")


        # Show variance ratio
        variance_ratio = var_y_doe / replicate_info_full['pooled_variance']
        st.markdown(f"""
        **Variance Ratio**: s²_DoE / s²_exp = {variance_ratio:.2f}

        - Ratio > 4: Strong factor effects
        - Ratio 2-4: Moderate factor effects
        - Ratio < 2: Weak factor effects
        """)


    # 2. Lack of Fit test
    st.markdown("---")

    st.markdown("#### 2️⃣ Lack of Fit Test (Model Adequacy)")

    st.info("""
    **F-test**: Compares model residual variance vs pure experimental variance.
    - H₀: Model is adequate (s²_model = s²_exp)
    - H₁: Significant lack of fit (s²_model > s²_exp)
    """)

    if 'rmse' in model_results:
        lof_col1, lof_col2, lof_col3 = st.columns(3)

        with lof_col1:
            st.metric("Model RMSE", f"{model_results['rmse']:.4f}")

            st.caption(f"Variance: {model_results['var_res']:.6f}")

            st.caption(f"DOF: {model_results['dof']}")

        with lof_col2:
            st.metric("Experimental Std Dev", f"{replicate_info_full['pooled_std']:.4f}")

            st.caption(f"Variance: {replicate_info_full['pooled_variance']:.6f}")

            st.caption(f"DOF: {replicate_info_full['pooled_dof']}")

        with lof_col3:
            # F = variance_model / variance_exp
            f_lof = model_results['var_res'] / replicate_info_full['pooled_variance']
            f_crit = stats.f.ppf(0.95, model_results['dof'], replicate_info_full['pooled_dof'])
            p_lof = 1 - stats.f.cdf(f_lof, model_results['dof'], replicate_info_full['pooled_dof'])


            st.metric("F-statistic", f"{f_lof:.2f}")

            st.caption(f"F-crit (95%): {f_crit:.2f}")

            st.caption(f"p-value: {p_lof:.4f}")


        # Unified interpretation with ratio
        ratio = model_results['rmse'] / replicate_info_full['pooled_std']

        st.markdown("---")

        result_col1, result_col2 = st.columns([1, 3])

        with result_col1:
            st.metric("RMSE / s_exp", f"{ratio:.2f}")

        with result_col2:
            if p_lof > 0.05:
                st.success(f"✅ No significant Lack of Fit (p={p_lof:.4f})")
                if ratio < 1.2:
                    st.info("🎯 Model error ≈ experimental error - excellent fit")
                elif ratio < 2.0:
                    st.info("✅ Model error is reasonable")

                else:
                    st.warning("⚠️ Model error exceeds experimental error despite non-significant test")

            else:
                st.error(f"❌ Significant Lack of Fit detected (p={p_lof:.4f})")

                st.warning("""
                **Model inadequate!** Consider:
                - Adding missing interaction or quadratic terms
                - Checking for outliers or influential points
                - Data transformations (log, sqrt, etc.)
                - Verifying model assumptions
                """)

    else:
        st.warning("Insufficient data for Lack of Fit test")


def _display_central_points_validation(central_points):
    """Display central points validation section"""
    st.markdown("---")

    st.markdown("### 🎯 Central Points Validation")


    st.info(f"""
    **{len(central_points)} central point(s)** excluded from model fitting - reserved for validation.
    These points assess model adequacy and curvature effects at the center of the experimental domain.
    """)

    if 'mlr_central_points' in st.session_state:
        central_X = st.session_state.mlr_central_points['X']
        central_y = st.session_state.mlr_central_points['y']

        # Calculate central point statistics
        central_mean = central_y.mean()
        central_std = central_y.std(ddof=1) if len(central_y) > 1 else 0

        central_stats_col1, central_stats_col2, central_stats_col3 = st.columns(3)

        with central_stats_col1:
            st.metric("Central Points Count", len(central_y))
        with central_stats_col2:
            st.metric("Mean Response", f"{central_mean:.4f}")
        with central_stats_col3:
            if len(central_y) > 1:
                st.metric("Std Dev", f"{central_std:.4f}")

            else:
                st.metric("Std Dev", "N/A (single point)")

        with st.expander("📋 Central Points Details"):
            central_display = pd.DataFrame({
                'Sample': [str(idx) for idx in st.session_state.mlr_central_points['indices']],
                'Observed Y': central_y.values
            })

            for col in central_X.columns:
                central_display[col] = central_X[col].values

            st.dataframe(central_display, use_container_width=True)


        st.info("""
        **Central Point Validation**: Use these points for model validation in the Predictions tab.
        They help assess curvature and lack of fit at the experimental center.
        """)


def _display_model_data_replicates(replicate_info, replicate_info_full):
    """Display replicates found in the model data"""
    st.markdown("### 🔬 Experimental Replicates in Model Data")

    rep_col1, rep_col2, rep_col3, rep_col4 = st.columns(4)

    with rep_col1:
        st.metric("Replicate Groups", replicate_info['n_replicate_groups'])
    with rep_col2:
        st.metric("Total Replicates", replicate_info['total_replicates'])
    with rep_col3:
        st.metric("Pooled Std Dev", f"{replicate_info['pooled_std']:.4f}")
    with rep_col4:
        st.metric("Replicate DOF", replicate_info['pooled_dof'])

    with st.expander("📋 Model Data Replicate Groups Details"):
        rep_data = []
        for i, group in enumerate(replicate_info['group_stats'], 1):
            rep_data.append({
                'Group': i,
                'Samples': ', '.join([str(idx+1) for idx in group['indices']]),
                'N': group['n_replicates'],
                'Mean Y': f"{group['mean']:.4f}",
                'Std Dev': f"{group['std']:.4f}",
                'DOF': group['dof']
            })

        rep_df = pd.DataFrame(rep_data)

        st.dataframe(rep_df, use_container_width=True)


    st.info(f"""
    **Model Data Experimental Error** = {replicate_info['pooled_std']:.4f}
    (from {replicate_info['pooled_dof']} degrees of freedom)

    This represents the experimental error in the data actually used for modeling.
    """)


    # Compare model replicates vs full replicates
    if replicate_info['pooled_std'] != replicate_info_full['pooled_std']:
        st.warning(f"""
        **Note**: Model data experimental error ({replicate_info['pooled_std']:.4f}) differs from
        full dataset experimental error ({replicate_info_full['pooled_std']:.4f}).
        This occurs when central point replicates are excluded from modeling.
        """)


def _display_statistical_summary(model_results, all_y_data, y_data, central_points,
                                 exclude_central_points, replicate_info_full):
    """
    Display statistical analysis summary - FULLY GENERIC VERSION

    Dynamically builds summary based on available metrics.
    Never assumes any specific keys exist except those explicitly checked.

    ALWAYS SHOWS (if available):
    - R², RMSE, DOF, parameters
    - Coefficients, p-values (shown elsewhere)
    - VIF (shown in model summary)

    CONDITIONALLY SHOWS (only if keys exist):
    - Pure error and Lack of Fit (if replicates detected)
    - Central points validation (if excluded)
    - Cross-validation Q², RMSECV (if 'q2' in results)
    """
    st.markdown("---")

    st.markdown("### 📋 Statistical Analysis Summary")


    # Build summary text dynamically based on available data
    summary_parts = []

    # ===== ALWAYS: Data Structure =====
    summary_parts.append(f"""
    📊 **Data Structure:**
    - Total samples: {len(all_y_data)}
    - Model samples: {len(y_data)}
    - Central points: {len(central_points) if central_points else 0}""")

    if replicate_info_full:
        summary_parts.append(f"    - Replicate groups: {replicate_info_full['n_replicate_groups']}")

    else:
        summary_parts.append("    - Replicate groups: 0 (no replicates detected)")


    # ===== CONDITIONAL: Model Diagnostics (check each key) =====
    diagnostics_lines = ["", "    🎯 **Model Diagnostics:**"]

    if 'r_squared' in model_results:
        diagnostics_lines.append(f"    - Adjusted R² (explained variance): {model_results['r_squared']:.4f}")

    if 'rmse' in model_results:
        diagnostics_lines.append(f"    - RMSE (model error): {model_results['rmse']:.4f}")

    if 'dof' in model_results:
        diagnostics_lines.append(f"    - Degrees of freedom: {model_results['dof']}")

    if 'n_features' in model_results:
        diagnostics_lines.append(f"    - Number of parameters: {model_results['n_features']}")


    # Add diagnostics if at least one metric was found
    if len(diagnostics_lines) > 2:
        summary_parts.append("\n".join(diagnostics_lines))


    # ===== CONDITIONAL: Cross-Validation (only if keys exist) =====
    if 'q2' in model_results and 'rmsecv' in model_results:
        summary_parts.append(f"    - Q² (cross-validation): {model_results['q2']:.4f}")
        summary_parts.append(f"    - RMSECV: {model_results['rmsecv']:.4f}")


    # ===== CONDITIONAL: Experimental Error Analysis (only if replicates exist) =====
    if replicate_info_full and 'rmse' in model_results:
        error_ratio = model_results['rmse'] / replicate_info_full['pooled_std']
        summary_parts.append(f"""
    🔬 **Experimental Error (from replicates):**
    - Pure error: s_exp = {replicate_info_full['pooled_std']:.4f} (DOF = {replicate_info_full['pooled_dof']})
    - Error ratio: RMSE/s_exp = {error_ratio:.2f}""")


        # Interpret error ratio
        if error_ratio < 1.2:
            summary_parts.append("    - ✅ Excellent: Model error ≈ experimental error")
        elif error_ratio < 2.0:
            summary_parts.append("    - ✅ Good: Model error is reasonable")

        else:
            summary_parts.append("    - ⚠️ Warning: Model error exceeds experimental error")


        # ===== CONDITIONAL: Factor Effects F-test (only with replicates AND var_y) =====
        if 'var_y' in model_results or len(y_data) > 1:
            # Calculate DoE variance
            var_y_doe = model_results.get('var_y', 0)
            dof_y_doe = len(all_y_data) - 1

            # Recalculate if central points were excluded
            if central_points and exclude_central_points:
                var_y_doe = np.var(y_data, ddof=1)
                dof_y_doe = len(y_data) - 1

            if var_y_doe > 0:  # Only proceed if variance is valid
                f_global = var_y_doe / replicate_info_full['pooled_variance']
                p_global = 1 - stats.f.cdf(f_global, dof_y_doe, replicate_info_full['pooled_dof'])
                variance_ratio = var_y_doe / replicate_info_full['pooled_variance']

                summary_parts.append(f"""
    📈 **Factor Effects:**
    - DoE variance: s²_DoE = {var_y_doe:.6f}
    - F-test p-value: {p_global:.4f}
    - Variance amplification: {variance_ratio:.1f}×""")


                # Interpret variance ratio
                if variance_ratio > 4:
                    summary_parts.append("    - ✅ Strong factor effects")
                elif variance_ratio > 2:
                    summary_parts.append("    - ✅ Moderate factor effects")

                else:
                    summary_parts.append("    - ⚠️ Weak factor effects")

    elif replicate_info_full and 'rmse' not in model_results:
        # Replicates exist but RMSE is missing
        summary_parts.append("""
    🔬 **Experimental Error (from replicates):**
    - Pure error: Available from replicates
    - Error ratio: Cannot calculate (RMSE not available)""")


    else:
        # No replicates case
        summary_parts.append("""
    🔬 **Experimental Error:**
    - No replicates detected - pure error cannot be estimated
    - Model quality assessed using adjusted R², RMSE, and cross-validation only""")


    # ===== CONDITIONAL: Central Points Validation (only if excluded) =====
    if central_points and exclude_central_points:
        if 'mlr_central_points' in st.session_state:
            central_mean = st.session_state.mlr_central_points['y'].mean()
            summary_parts.append(f"""
    🎯 **Central Points:**
    - Excluded from model: {len(central_points)} points
    - Reserved for validation
    - Mean response: {central_mean:.4f}""")


    # Combine all parts and display
    summary_text = "\n".join(summary_parts)

    st.info(summary_text)


def _display_model_summary(model_results):
    """
    Display model summary: Dispersion Matrix, VIF, Leverage

    ALWAYS SHOWS (if available):
    - Dispersion Matrix (X'X)^-1
    - VIF (Variance Inflation Factors) - multicollinearity check
    - Leverage (hat values) - influential points
    """
    st.markdown("### 📋 Model Summary")


    # ===== CONDITIONAL: Dispersion Matrix =====
    if 'XtX_inv' in model_results and 'X' in model_results:
        st.markdown("#### Dispersion Matrix (X'X)^-1")
        try:
            dispersion_df = pd.DataFrame(
                model_results['XtX_inv'],
                index=model_results['X'].columns,
                columns=model_results['X'].columns
            )

            st.dataframe(dispersion_df.round(4), use_container_width=True)

            trace = np.trace(model_results['XtX_inv'])

            st.info(f"**Trace of Dispersion Matrix:** {trace:.4f}")

        except Exception as e:
            st.warning(f"⚠️ Could not display dispersion matrix: {str(e)}")


    # ===== CONDITIONAL: VIF (only if key exists) =====
    if 'vif' in model_results and model_results['vif'] is not None:
        st.markdown("#### Variance Inflation Factors (VIF)")
        try:
            vif_df = model_results['vif'].to_frame('VIF')
            vif_df_clean = vif_df[~vif_df.index.str.contains('Intercept', case=False, na=False)]
            vif_df_clean = vif_df_clean.dropna()

            if not vif_df_clean.empty:
                def interpret_vif(vif_val):
                    if vif_val <= 1:
                        return "✅ No covariance"
                    elif vif_val <= 2:
                        return "✅ OK"
                    elif vif_val <= 4:
                        return "⚠️ Good"
                    elif vif_val <= 8:
                        return "⚠️ Acceptable"
                    else:
                        return "❌ High multicollinearity"

                vif_df_clean['Interpretation'] = vif_df_clean['VIF'].apply(interpret_vif)

                st.dataframe(vif_df_clean.round(4), use_container_width=True)


                st.info("""
                **VIF Interpretation:**
                - VIF = 1: No covariance
                - VIF < 2: OK
                - VIF < 4: Good
                - VIF < 8: Acceptable
                - VIF > 8: High multicollinearity (problematic)
                """)

            else:
                st.info("VIF not applicable for this model")

        except Exception as e:
            st.warning(f"⚠️ Could not display VIF: {str(e)}")

    else:
        st.info("ℹ️ VIF not calculated for this model")


    # ===== CONDITIONAL: Leverage (only if key exists) =====
    if 'leverage' in model_results and model_results['leverage'] is not None:
        st.markdown("#### Leverage of Experimental Points")
        try:
            leverage_series = pd.Series(
                model_results['leverage'],
                index=range(1, len(model_results['leverage']) + 1)
            )

            st.dataframe(leverage_series.to_frame('Leverage').T.round(4), use_container_width=True)

            st.info(f"**Maximum Leverage:** {model_results['leverage'].max():.4f}")

        except Exception as e:
            st.warning(f"⚠️ Could not display leverage: {str(e)}")

    else:
        st.info("ℹ️ Leverage not calculated for this model")


def _display_error_comparison(model_results, replicate_info):
    """Display comparison between model error and experimental error"""
    st.markdown("#### 🎯 Model vs Experimental Error Comparison")

    comparison_col1, comparison_col2, comparison_col3 = st.columns(3)

    with comparison_col1:
        st.metric("Model RMSE", f"{model_results['rmse']:.4f}")

    with comparison_col2:
        st.metric("Experimental Std Dev", f"{replicate_info['pooled_std']:.4f}")

    with comparison_col3:
        ratio = model_results['rmse'] / replicate_info['pooled_std']
        st.metric("RMSE / Exp. Std Dev", f"{ratio:.2f}")

    if ratio < 1.2:
        st.success("✅ Model error is close to experimental error - excellent fit!")
    elif ratio < 2.0:
        st.info("ℹ️ Model error is reasonable compared to experimental error")

    else:
        st.warning("⚠️ Model error significantly exceeds experimental error")


def _display_coefficients_table(model_results):
    """Display coefficients table with statistics"""
    st.markdown("### 📊 Model Coefficients")

    try:
        # Validate that coefficients exist
        if 'coefficients' not in model_results or model_results['coefficients'] is None:
            st.error("❌ Coefficients data not available in model results")

        else:
            coef_df = pd.DataFrame({'Coefficient': model_results['coefficients']})


            # Check if ALL statistical keys exist
            has_statistics = (
                'se_coef' in model_results and model_results['se_coef'] is not None and
                't_stats' in model_results and model_results['t_stats'] is not None and
                'p_values' in model_results and model_results['p_values'] is not None and
                'ci_lower' in model_results and model_results['ci_lower'] is not None and
                'ci_upper' in model_results and model_results['ci_upper'] is not None
            )

            if has_statistics:
                # Add all statistical columns
                coef_df['Std. Error'] = model_results['se_coef']
                coef_df['t-statistic'] = model_results['t_stats']
                coef_df['p-value'] = model_results['p_values']
                coef_df['CI Lower'] = model_results['ci_lower']
                coef_df['CI Upper'] = model_results['ci_upper']

                def add_stars(p):
                    if p <= 0.001:
                        return '***'
                    elif p <= 0.01:
                        return '**'
                    elif p <= 0.05:
                        return '*'
                    else:
                        return ''

                coef_df['Sig.'] = coef_df['p-value'].apply(add_stars)


                st.dataframe(coef_df.round(4), use_container_width=True)

                st.info("Significance codes: *** p≤0.001, ** p≤0.01, * p≤0.05")

            else:
                # Fallback: Show only coefficients
                st.dataframe(coef_df.round(4), use_container_width=True)

                st.warning("⚠️ Statistical information (standard errors, p-values, confidence intervals) not available")

                st.info("This may occur when degrees of freedom ≤ 0 (not enough samples for the model complexity)")


    except Exception as e:
        st.error(f"❌ Error displaying coefficients: {str(e)}")
        import traceback
        with st.expander("🐛 Full error traceback"):
            st.code(traceback.format_exc())


def sort_coefficients_by_type(coef_names):
    """
    Sort coefficient names in standard order:
    1. Linear terms (no * or ^)
    2. Interaction terms (contains *)
    3. Quadratic terms (contains ^2)

    Args:
        coef_names: list of coefficient name strings

    Returns:
        sorted_names: list sorted by term type, maintaining alphabetical within each group
    """
    linear = []
    interactions = []
    quadratic = []

    for name in coef_names:
        if '^2' in name or ('^' in name and '2' in name):
            quadratic.append(name)
        elif '*' in name:
            interactions.append(name)
        else:
            linear.append(name)

    # Sort each group alphabetically for consistency
    linear.sort()
    interactions.sort()
    quadratic.sort()

    # Combine in correct order
    sorted_names = linear + interactions + quadratic
    return sorted_names


def sort_coefficients_correct_order(coef_names, variable_names):
    """
    Sort coefficients in CORRECT MODEL SEQUENCE based on variable order.

    Order: Intercept → Linear (b1,b2,b3...) → Interactions (b12,b13,b23...) → Quadratic (b11,b22,b33...)

    CRITICAL: Follows EXACT variable sequence from X matrix, NOT alphabetical!

    Args:
        coef_names (list): All coefficient names from model
        variable_names (list): X variable names in EXACT order as used in model
            Example: ['x1.Scavenger', 'x2.pH', 'x3.Form.start']

    Returns:
        list: Coefficient names in correct model sequence
            Example: ['Intercept', 'x1.Scavenger', 'x2.pH', 'x3.Form.start',
                      'x1.Scavenger*x2.pH', 'x1.Scavenger*x3.Form.start', 'x2.pH*x3.Form.start',
                      'x1.Scavenger^2', 'x2.pH^2', 'x3.Form.start^2']
    """

    if not variable_names:
        # Fallback to alphabetical if no variable names provided
        return sort_coefficients_by_type(coef_names)

    intercept = []
    linear = {}      # Use dict to maintain index order
    interactions = {}
    quadratic = {}
    unknown = []

    # Create mapping of variable names to indices
    var_to_idx = {var: idx for idx, var in enumerate(variable_names)}

    # Separate coefficients by type
    for coef_name in coef_names:
        coef_str = str(coef_name)

        # INTERCEPT
        if 'Intercept' in coef_str or coef_str.lower() == 'intercept':
            intercept.append(coef_name)

        # QUADRATIC (must check before interaction - contains ^2)
        elif '^2' in coef_str or ('^' in coef_str and '2' in coef_str):
            # Extract variable name (remove ^2)
            var_name = coef_str.replace('^2', '').replace('^', '').replace('2', '').strip()

            # Try to find matching variable
            var_idx = None
            for var in variable_names:
                if var in coef_str or var.replace('.', '') in coef_str.replace('.', ''):
                    var_idx = var_to_idx[var]
                    break

            if var_idx is not None:
                quadratic[var_idx] = coef_name
            else:
                unknown.append(coef_name)

        # INTERACTION (contains *)
        elif '*' in coef_str:
            # Parse interaction: extract variable names
            parts = coef_str.split('*')
            parts = [p.strip() for p in parts]

            if len(parts) == 2:
                var1, var2 = parts

                # Find indices for both variables
                idx1 = None
                idx2 = None

                for var in variable_names:
                    if var in var1 or var.replace('.', '') in var1.replace('.', ''):
                        idx1 = var_to_idx[var]
                    if var in var2 or var.replace('.', '') in var2.replace('.', ''):
                        idx2 = var_to_idx[var]

                if idx1 is not None and idx2 is not None:
                    # Ensure consistent ordering (smaller index first)
                    if idx1 > idx2:
                        idx1, idx2 = idx2, idx1

                    # Use tuple of indices as key
                    key = (idx1, idx2)
                    interactions[key] = coef_name
                else:
                    unknown.append(coef_name)
            else:
                unknown.append(coef_name)

        # LINEAR TERM
        else:
            # Single variable coefficient - find which variable it matches
            var_idx = None
            for var in variable_names:
                if var in coef_str or var.replace('.', '') in coef_str.replace('.', ''):
                    var_idx = var_to_idx[var]
                    break

            if var_idx is not None:
                linear[var_idx] = coef_name
            else:
                unknown.append(coef_name)

    # Build final order
    result = []

    # 1. Intercept first
    result.extend(intercept)

    # 2. Linear terms in variable order (b1, b2, b3, ...)
    for idx in sorted(linear.keys()):
        result.append(linear[idx])

    # 3. Interaction terms in variable pair order (b12, b13, b23, ...)
    for key in sorted(interactions.keys()):
        result.append(interactions[key])

    # 4. Quadratic terms in variable order (b11, b22, b33, ...)
    for idx in sorted(quadratic.keys()):
        result.append(quadratic[idx])

    # 5. Any unknown terms at the end
    result.extend(unknown)

    return result


def to_subscript(text):
    """
    Convert numbers to subscripts: 1→₁, 2→₂, etc.

    Args:
        text: Text containing numbers to convert

    Returns:
        str: Text with subscripted numbers
    """
    return str(text).translate(str.maketrans('0123456789', '₀₁₂₃₄₅₆₇₈₉'))


def to_superscript(text):
    """
    Convert numbers to superscripts: 1→¹, 2→², etc.

    Args:
        text: Text containing numbers to convert

    Returns:
        str: Text with superscripted numbers
    """
    return str(text).translate(str.maketrans('0123456789', '⁰¹²³⁴⁵⁶⁷⁸⁹'))


def generate_model_equation(
    coefficients,
    variable_names,
    y_variable_name="y",
    use_subscripts=True,
    show_coefficient_names=False,
    decimals=4,
    use_hat=True
):
    """
    Generate compact model equation with generic variable names and proper subscripts.

    Args:
        coefficients (dict or Series): Model coefficients
            Example: {'Intercept': 0.10, 'x1.Scavenger': -0.10, 'x2.pH': -0.03, ...}
        variable_names (list): Original variable names in correct order
            Example: ['x1.Scavenger', 'x2.pH', 'x3.Form.start']
            Maps to: x₁, x₂, x₃
        y_variable_name (str): Name of response variable (default 'y')
        use_subscripts (bool): If True, use subscripts (x₁); if False, use x1
        show_coefficient_names (bool): If True, show b₁, b₂; if False, show values
        decimals (int): Decimal places for coefficients
        use_hat (bool): If True, use ŷ; if False, use y

    Returns:
        str: Model equation
        Example: "ŷ = 0.1000 - 0.1000·x₁ - 0.0300·x₂ + 0.0600·x₃ - 0.0500·x₁·x₂ + 0.0300·x₁² - 0.0100·x₂²"
    """

    # Response variable name with hat
    y_name = "ŷ" if use_hat else y_variable_name

    # Convert coefficients to dict if needed
    if hasattr(coefficients, 'to_dict'):
        coef_dict = coefficients.to_dict()
    elif hasattr(coefficients, 'items'):
        coef_dict = dict(coefficients)
    else:
        coef_dict = coefficients

    terms = []
    n_vars = len(variable_names)

    # Create variable name mapping
    var_to_idx = {var: idx for idx, var in enumerate(variable_names)}

    # Helper to get variable display name
    def get_var_display(idx):
        if use_subscripts:
            return f"x{to_subscript(idx + 1)}"
        else:
            return f"x{idx + 1}"

    # Helper to get coefficient display name
    def get_coef_display(*indices):
        idx_str = ''.join(str(i+1) for i in indices)
        if use_subscripts:
            return f"b{to_subscript(idx_str)}"
        else:
            return f"b{idx_str}"

    # 1. INTERCEPT (b₀)
    intercept_val = None
    for key in ['Intercept', 'b0', 'const']:
        if key in coef_dict:
            intercept_val = coef_dict[key]
            break

    if intercept_val is not None:
        if show_coefficient_names:
            terms.append(get_coef_display())  # b₀
        else:
            # Don't use + sign for first term
            terms.append(f"{intercept_val:.{decimals}f}")

    # 2. LINEAR TERMS (b₁·x₁, b₂·x₂, b₃·x₃, ...)
    for i in range(n_vars):
        var_name = variable_names[i]

        # Try to find coefficient
        coef_val = None
        for key in coef_dict.keys():
            key_str = str(key)
            # Match if variable name is in the key and it's not an interaction or quadratic
            if var_name in key_str and '*' not in key_str and '^' not in key_str:
                coef_val = coef_dict[key]
                break

        if coef_val is not None:
            var_display = get_var_display(i)

            if show_coefficient_names:
                coef_display = get_coef_display(i)
                terms.append(f"{coef_display}·{var_display}")
            else:
                terms.append(f"{coef_val:+.{decimals}f}·{var_display}")

    # 3. INTERACTION TERMS (b₁₂·x₁·x₂, b₁₃·x₁·x₃, b₂₃·x₂·x₃, ...)
    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            var1_name = variable_names[i]
            var2_name = variable_names[j]

            # Try to find interaction coefficient
            coef_val = None
            for key in coef_dict.keys():
                key_str = str(key)
                # Check if both variables are in the key and it has *
                if '*' in key_str and var1_name in key_str and var2_name in key_str:
                    coef_val = coef_dict[key]
                    break

            if coef_val is not None:
                var1_display = get_var_display(i)
                var2_display = get_var_display(j)

                if show_coefficient_names:
                    coef_display = get_coef_display(i, j)
                    terms.append(f"{coef_display}·{var1_display}·{var2_display}")
                else:
                    terms.append(f"{coef_val:+.{decimals}f}·{var1_display}·{var2_display}")

    # 4. QUADRATIC TERMS (b₁₁·x₁², b₂₂·x₂², b₃₃·x₃², ...)
    for i in range(n_vars):
        var_name = variable_names[i]

        # Try to find quadratic coefficient
        coef_val = None
        for key in coef_dict.keys():
            key_str = str(key)
            # Check if variable is in key and it has ^2
            if '^2' in key_str and var_name in key_str:
                coef_val = coef_dict[key]
                break

        if coef_val is not None:
            var_display = get_var_display(i)

            if show_coefficient_names:
                coef_display = get_coef_display(i, i)
                terms.append(f"{coef_display}·{var_display}²")
            else:
                terms.append(f"{coef_val:+.{decimals}f}·{var_display}²")

    # Build final equation
    if not terms:
        return f"{y_name} = (no terms found)"

    # Join terms
    equation = f"{y_name} = {terms[0]}"
    for term in terms[1:]:
        if term.startswith('+') or term.startswith('-'):
            equation += f" {term}"
        else:
            equation += f" + {term}"

    # Clean up spacing
    equation = equation.replace("+ -", "- ").replace("  ", " ")

    return equation


def _display_coefficients_barplot(model_results, y_var):
    """Display coefficients bar plot with proper ordering: Linear, Interactions, Quadratic"""
    st.markdown("#### Coefficients Bar Plot")

    coefficients = model_results['coefficients']
    coef_no_intercept = coefficients[coefficients.index != 'Intercept']

    if len(coef_no_intercept) == 0:
        st.warning("No coefficients to plot (model contains only intercept)")
        return

    # ========================================================================
    # SORT COEFFICIENTS: Linear → Interactions → Quadratic
    # ========================================================================
    coef_names_raw = coef_no_intercept.index.tolist()
    coef_names_sorted = sort_coefficients_by_type(coef_names_raw)

    # Get values in sorted order
    coef_values = coef_no_intercept.loc[coef_names_sorted].values
    coef_names = coef_names_sorted  # Use sorted names

    # ========================================================================
    # DETERMINE COLORS based on term type
    # ========================================================================
    colors = []
    for name in coef_names:
        if '*' in name:
            n_asterisks = name.count('*')
            colors.append('cyan' if n_asterisks > 1 else 'green')
        elif '^2' in name or '^' in name:
            colors.append('cyan')
        else:
            colors.append('red')

    # ========================================================================
    # CREATE BAR CHART
    # ========================================================================
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=coef_names,
        y=coef_values,
        marker_color=colors,
        marker_line_color='black',
        marker_line_width=1,
        name='Coefficients',
        showlegend=False,
        hovertemplate='<b>%{x}</b><br>Value: %{y:.4f}<extra></extra>'
    ))

    # ========================================================================
    # ADD CONFIDENCE INTERVALS (if available)
    # ========================================================================
    if 'ci_lower' in model_results and 'ci_upper' in model_results:
        try:
            ci_lower = model_results['ci_lower'].loc[coef_names_sorted].values
            ci_upper = model_results['ci_upper'].loc[coef_names_sorted].values

            error_minus = coef_values - ci_lower
            error_plus = ci_upper - coef_values

            fig.update_traces(
                error_y=dict(
                    type='data',
                    symmetric=False,
                    array=error_plus,
                    arrayminus=error_minus,
                    color='black',
                    thickness=2,
                    width=4
                )
            )
        except Exception:
            pass

    # ========================================================================
    # ADD SIGNIFICANCE MARKERS (if available)
    # ========================================================================
    if 'p_values' in model_results:
        try:
            p_values = model_results['p_values'].loc[coef_names_sorted].values
            for i, (name, coef, p) in enumerate(zip(coef_names, coef_values, p_values)):
                y_pos = coef
                y_offset = max(abs(coef) * 0.05, 0.01) if coef >= 0 else -max(abs(coef) * 0.05, 0.01)

                if p <= 0.001:
                    sig_text = '***'
                elif p <= 0.01:
                    sig_text = '**'
                elif p <= 0.05:
                    sig_text = '*'
                else:
                    sig_text = None

                if sig_text:
                    fig.add_annotation(
                        x=name, y=y_pos + y_offset,
                        text=sig_text,
                        showarrow=False,
                        font=dict(size=14, color='black'),
                        yshift=10 if coef >= 0 else -10
                    )
        except Exception:
            pass

    # ========================================================================
    # UPDATE LAYOUT
    # ========================================================================
    fig.update_layout(
        title=f"Coefficients - {y_var}",
        xaxis_title="Term",
        yaxis_title="Coefficient Value",
        height=500,
        xaxis={'tickangle': 45},
        showlegend=False,
        yaxis=dict(zeroline=True, zerolinewidth=2, zerolinecolor='gray'),
        margin=dict(b=100)
    )

    st.plotly_chart(fig, use_container_width=True)

    # ========================================================================
    # DISPLAY COLOR LEGEND
    # ========================================================================
    col1, col2, col3 = st.columns(3)
    with col1:
        st.caption("🔴 Red = Linear terms")
    with col2:
        st.caption("🟢 Green = 2-term interactions")
    with col3:
        st.caption("🔵 Cyan = Quadratic terms")

    if 'p_values' in model_results:
        st.caption("Significance: *** p≤0.001, ** p≤0.01, * p≤0.05")

        # ============================================================
        # NEW: Display Generic Model Equation with Subscripts
        # ============================================================
        st.markdown("---")
        st.markdown("#### 📐 Generic Model Equation")
        st.info("Model equation with generic variable notation (x₁, x₂, x₃...) for easy interpretation")

        try:
            # Get x_vars from session state (needed for equation generation)
            x_vars = st.session_state.get('mlr_x_vars', [])

            if x_vars:
                # Generate numeric equation with subscripts
                equation_numeric = generate_model_equation(
                    coefficients=model_results['coefficients'],
                    variable_names=x_vars,
                    y_variable_name=y_var,
                    use_subscripts=True,
                    show_coefficient_names=False,
                    decimals=4,
                    use_hat=True
                )

                # Generate symbolic equation (with coefficient names)
                equation_symbolic = generate_model_equation(
                    coefficients=model_results['coefficients'],
                    variable_names=x_vars,
                    y_variable_name=y_var,
                    use_subscripts=True,
                    show_coefficient_names=True,
                    decimals=4,
                    use_hat=True
                )

                # Display both equations in expandable sections
                col_eq1, col_eq2 = st.columns([5, 1])

                with col_eq1:
                    st.markdown("**Numeric Equation:**")
                    st.code(equation_numeric, language="text")

                    with st.expander("Show symbolic form (coefficient names)"):
                        st.code(equation_symbolic, language="text")

                with col_eq2:
                    # Optional: Add copy button if pyperclip is available
                    if st.button("📋", key=f"copy_generic_eq_{y_var}", help="Copy numeric equation"):
                        try:
                            import pyperclip
                            pyperclip.copy(equation_numeric)
                            st.success("✅")
                        except ImportError:
                            st.info("Install pyperclip to enable copy")
            else:
                st.warning("⚠️ Cannot generate generic equation: x_vars not found in session state")

        except Exception as e:
            st.warning(f"Could not generate generic equation: {str(e)}")

        # ===== FITTED MODEL FORMULA WITH UNCERTAINTY-BASED DECIMALS =====
        st.markdown("---")
        st.markdown("#### Fitted Model Formula (with actual variable names)")

        import math

        # Get intercept
        intercept = model_results['coefficients'].get('Intercept', 0)

        # Helper function: determine decimals from uncertainty
        def get_decimals_from_uncertainty(coef_value, ci_lower, ci_upper):
            """
            Determine decimal places based on confidence interval width.

            Rule: Show coefficient decimals = where CI uncertainty starts

            Examples:
            - CI: [0.02, 0.07] → width 0.05 (1st decimal) → 1 decimal
            - CI: [0.039, 0.051] → width 0.012 (2nd decimal) → 2 decimals
            - CI: [0.0440, 0.0462] → width 0.0022 (3rd decimal) → 3 decimals
            """
            # Calculate uncertainty half-width
            uncertainty_width = ci_upper - ci_lower
            uncertainty_half_width = uncertainty_width / 2

            # If no uncertainty, use 3 decimals
            if uncertainty_half_width == 0:
                return 3

            # Find order of magnitude of uncertainty
            try:
                # Get the order of magnitude
                magnitude = math.floor(math.log10(abs(uncertainty_half_width)))

                # Convert magnitude to decimal places
                # magnitude = -1 (0.1) → 1 decimal
                # magnitude = -2 (0.01) → 2 decimals
                # magnitude = -3 (0.001) → 3 decimals
                decimals = -magnitude

                # Cap at range [1, 3]
                decimals = max(1, min(3, decimals))

            except (ValueError, OverflowError):
                # Fallback to 2 decimals if calculation fails
                decimals = 2

            return decimals

        # Build fitted formula with adaptive decimals
        formula_parts = []

        # Format intercept (1-3 decimals based on its uncertainty)
        if 'ci_lower' in model_results and 'ci_upper' in model_results:
            ci_lower_intercept = model_results['ci_lower'].get('Intercept', intercept)
            ci_upper_intercept = model_results['ci_upper'].get('Intercept', intercept)
            decimals_intercept = get_decimals_from_uncertainty(intercept, ci_lower_intercept, ci_upper_intercept)
        else:
            decimals_intercept = 2  # Default if no CI available

        # Format all coefficients with adaptive decimals
        coef_formatted = {}
        all_decimals = []

        for name in coef_names:
            coef_value = coef_no_intercept[name]

            # Get decimals from uncertainty if available
            if 'ci_lower' in model_results and 'ci_upper' in model_results:
                try:
                    ci_lower = model_results['ci_lower'][name]
                    ci_upper = model_results['ci_upper'][name]
                    decimals = get_decimals_from_uncertainty(coef_value, ci_lower, ci_upper)
                except (KeyError, TypeError):
                    decimals = 2  # Fallback
            else:
                decimals = 2  # Default if no CI available

            coef_formatted[name] = (coef_value, decimals)
            all_decimals.append(decimals)

        # Special rule: If many coefficients round to zero, increase precision
        # Try formatting with current decimals
        test_formatted = []
        for name in coef_names:
            coef_value, decimals = coef_formatted[name]
            test_str = f"{coef_value:.{decimals}f}"
            test_formatted.append(test_str)

        # Count how many are "0.0" or "-0.0" or "0.00" etc
        zero_count = sum(1 for s in test_formatted if float(s) == 0.0)

        # If more than 50% are zeros, add 1 decimal to all (but cap at 3)
        if len(test_formatted) > 0 and zero_count / len(test_formatted) > 0.5:
            for name in coef_names:
                coef_value, decimals = coef_formatted[name]
                # Increase decimals by 1, cap at 3
                coef_formatted[name] = (coef_value, min(3, decimals + 1))
            decimals_intercept = min(3, decimals_intercept + 1)

        # Build formula string
        intercept_str = f"{intercept:.{decimals_intercept}f}"
        formula_parts.append(f"{y_var} = {intercept_str}")

        for name in coef_names:
            coef_value, decimals = coef_formatted[name]
            coef_str = f"{coef_value:+.{decimals}f}"
            formula_parts.append(f"{coef_str}·{name}")

        fitted_formula = " ".join(formula_parts)

        # Display in code block
        st.code(fitted_formula, language="text")

        # Optional: Show precision info
        if 'ci_lower' in model_results:
            st.info("📊 **Coefficient precision:** Decimal places based on confidence interval width")

        # Copy to clipboard button
        col1, col2, col3 = st.columns([2, 1, 2])
        with col2:
            if st.button("📋 Copy to clipboard", key="copy_fitted_formula_tab1"):
                st.write("""
                <script>
                var text = `""" + fitted_formula.replace("`", "\\`") + """`
                navigator.clipboard.writeText(text);
                </script>
                """, unsafe_allow_html=True)
                st.success("✓ Formula copied!")


def _display_design_analysis_results(design_results, x_vars, X_data):
    """
    Display design analysis results (without Y variable)

    Shows:
    - Design matrix information
    - Dispersion Matrix (X'X)^-1
    - VIF (multicollinearity check)
    - Leverage (influential points)
    - Prediction confidence intervals (if replicates exist)

    Args:
        design_results: dict from design_analysis()
        x_vars: list of X variable names
        X_data: original X data (before model matrix expansion)
    """
    st.markdown("---")

    st.markdown("## 📊 Design Analysis Results")

    st.info("**Design Screening Mode**: Analyzing experimental design quality without response variable")


    # ===== DESIGN MATRIX INFO =====
    st.markdown("### 📐 Design Matrix Information")

    info_col1, info_col2, info_col3 = st.columns(3)

    with info_col1:
        st.metric("Experimental Points", design_results['n_samples'])

    with info_col2:
        st.metric("Model Terms", design_results['n_features'])

    with info_col3:
        st.metric("Degrees of Freedom", design_results['dof'])

    if design_results['dof'] <= 0:
        st.error(f"""
        ❌ **Insufficient degrees of freedom!**
        - You have {design_results['n_samples']} experimental points
        - The model requires {design_results['n_features']} parameters
        - Need at least {design_results['n_features'] + 1} points to fit a model
        """)

        st.warning("**Recommendation**: Add more experimental points or reduce model complexity")
    elif design_results['dof'] < 5:
        st.warning(f"""
        ⚠️ **Low degrees of freedom** (DOF = {design_results['dof']})
        - Model will have limited statistical power
        - Consider adding more experimental points for robust estimation
        """)

    else:
        st.success(f"✅ Adequate degrees of freedom (DOF = {design_results['dof']})")


    # ===== DISPERSION MATRIX =====
    st.markdown("---")

    st.markdown("### 📊 Dispersion Matrix (X'X)^-1")

    st.info("""
    The dispersion matrix shows the variance-covariance structure of model parameters.
    - **Diagonal elements**: Variance of coefficient estimates (smaller is better)
    - **Off-diagonal elements**: Correlation between coefficients
    """)

    try:
        dispersion_df = pd.DataFrame(
            design_results['XtX_inv'],
            index=design_results['X'].columns,
            columns=design_results['X'].columns
        )

        st.dataframe(dispersion_df.round(6), use_container_width=True)

        trace = np.trace(design_results['XtX_inv'])
        determinant = np.linalg.det(design_results['XtX_inv'])

        disp_metric_col1, disp_metric_col2 = st.columns(2)
        with disp_metric_col1:
            st.metric("Trace", f"{trace:.4f}", help="Sum of diagonal elements - measure of total variance")
        with disp_metric_col2:
            st.metric("Determinant", f"{determinant:.2e}", help="Measure of design efficiency")


    except Exception as e:
        st.error(f"❌ Could not display dispersion matrix: {str(e)}")


    # ===== VIF (Multicollinearity) =====
    st.markdown("---")

    st.markdown("### 🔍 Variance Inflation Factors (VIF)")

    st.info("""
    **VIF measures multicollinearity** between predictor variables:
    - VIF = 1: No covariance
    - VIF < 2: Excellent
    - VIF < 4: Good
    - VIF < 8: Acceptable
    - VIF > 8: **High multicollinearity** (problematic)
    """)

    if 'vif' in design_results and design_results['vif'] is not None:
        vif_df = design_results['vif'].to_frame('VIF')
        vif_df_clean = vif_df[~vif_df.index.str.contains('Intercept', case=False, na=False)]
        vif_df_clean = vif_df_clean.dropna()

        if not vif_df_clean.empty:
            def interpret_vif(vif_val):
                if vif_val <= 1:
                    return "✅ No covariance"
                elif vif_val <= 2:
                    return "✅ Excellent"
                elif vif_val <= 4:
                    return "✅ Good"
                elif vif_val <= 8:
                    return "⚠️ Acceptable"
                else:
                    return "❌ High multicollinearity"

            vif_df_clean['Interpretation'] = vif_df_clean['VIF'].apply(interpret_vif)

            st.dataframe(vif_df_clean.round(4), use_container_width=True)


            # Check for problematic VIF
            max_vif = vif_df_clean['VIF'].max()
            if max_vif > 8:
                st.error(f"""
                ❌ **High multicollinearity detected!** (Max VIF = {max_vif:.2f})
                Consider:
                - Removing correlated variables
                - Using centered/orthogonal coding
                - Reducing interaction/quadratic terms
                """)
            elif max_vif > 4:
                st.warning(f"⚠️ Moderate multicollinearity detected (Max VIF = {max_vif:.2f})")

            else:
                st.success(f"✅ Low multicollinearity (Max VIF = {max_vif:.2f})")

        else:
            st.info("VIF not applicable (single term model)")

    else:
        st.info("ℹ️ VIF not calculated")


    # ===== LEVERAGE =====
    st.markdown("---")

    st.markdown("### 📍 Leverage of Experimental Points")

    st.info("""
    **Leverage** measures how influential each experimental point is on model predictions:
    - Higher leverage = more influential point
    - Average leverage = p/n (where p = parameters, n = samples)
    - Points with leverage > 2×average may be influential
    """)

    if 'leverage' in design_results and design_results['leverage'] is not None:
        leverage_series = pd.Series(
            design_results['leverage'],
            index=range(1, len(design_results['leverage']) + 1),
            name='Leverage'
        )


        # Display as horizontal table (transposed)

        st.dataframe(leverage_series.to_frame().T.round(4), use_container_width=True)

        avg_leverage = design_results['n_features'] / design_results['n_samples']
        max_leverage = design_results['leverage'].max()
        max_leverage_idx = np.argmax(design_results['leverage']) + 1

        lev_col1, lev_col2, lev_col3 = st.columns(3)

        with lev_col1:
            st.metric("Average Leverage", f"{avg_leverage:.4f}")

        with lev_col2:
            st.metric("Max Leverage", f"{max_leverage:.4f}")

        with lev_col3:
            st.metric("Max at Point", max_leverage_idx)


        # Check for high leverage points
        high_leverage_threshold = 2 * avg_leverage
        high_leverage_points = np.where(design_results['leverage'] > high_leverage_threshold)[0] + 1

        if len(high_leverage_points) > 0:
            st.warning(f"""
            ⚠️ **{len(high_leverage_points)} point(s) with high leverage** (> {high_leverage_threshold:.4f}):
            Points: {', '.join(map(str, high_leverage_points))}

            High leverage points have strong influence on model predictions.
            """)

        else:
            st.success("✅ No unusually high leverage points detected")


    # ===== EXPERIMENTAL VARIANCE (if replicates exist) =====
    if 'experimental_std' in design_results:
        st.markdown("---")

        st.markdown("### 🔬 Experimental Variability")

        st.info("""
        **Pure experimental error** estimated from replicate measurements.
        This can be used to assess prediction uncertainty even without fitting a model.
        """)

        exp_col1, exp_col2, exp_col3 = st.columns(3)

        with exp_col1:
            st.metric("Experimental Std Dev (s_exp)", f"{design_results['experimental_std']:.4f}")

        with exp_col2:
            st.metric("Degrees of Freedom", design_results['experimental_dof'])

        with exp_col3:
            st.metric("t-critical (95%)", f"{design_results['t_critical']:.3f}")


        st.markdown("#### Prediction Standard Errors")

        st.info("Standard error for predictions at each experimental point (s_exp × √leverage)")

        se_pred_series = pd.Series(
            design_results['prediction_se'],
            index=range(1, len(design_results['prediction_se']) + 1),
            name='Prediction SE'
        )


        st.dataframe(se_pred_series.to_frame().T.round(4), use_container_width=True)


        st.success("""
        ✅ **Prediction confidence intervals can be computed** once a response variable is measured.
        The prediction uncertainty will be: ±{:.4f} × t-critical for each point.
        """.format(design_results['experimental_std']))


    else:
        st.markdown("---")

        st.info("""
        ℹ️ **No experimental replicates detected** in the design matrix.
        Prediction uncertainty cannot be estimated without replicate measurements or a fitted model.
        """)
