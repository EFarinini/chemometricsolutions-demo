"""
MLR/DoE Analysis Page - Refactored Clean Version
Complete Design of Experiments and Multiple Linear Regression suite
Equivalent to CAT DOE_* R scripts

This module imports core functions from mlr_utils submodules.
Only page UI logic and TAB1 workflow remain in this file.
"""

import streamlit as st
import numpy as np
import pandas as pd
from scipy import stats

# Import core MLR/DoE functions from mlr_utils
from mlr_utils.model_computation import (
    fit_mlr_model,
    statistical_summary,
    create_model_matrix
)
from mlr_utils.model_diagnostics import (
    calculate_vif,
    check_model_saturated,
    show_model_diagnostics_ui
)
from mlr_utils.response_surface import show_response_surface_ui
from mlr_utils.confidence_intervals import show_confidence_intervals_ui
from mlr_utils.predictions import show_predictions_ui
from mlr_utils.candidate_points import show_candidate_points_ui
from mlr_utils.export import show_export_ui


# ============================================================================
# HELPER FUNCTIONS (used in TAB1 only)
# ============================================================================

def detect_replicates(X_data, y_data, tolerance=1e-10):
    """
    Detect experimental replicates in the design matrix
    Returns replicate statistics or None if no replicates found
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


# ============================================================================
# MAIN PAGE FUNCTION
# ============================================================================

def show():
    """Display the MLR/DOE Analysis page"""

    st.markdown("# MLR & Design of Experiments")
    st.markdown("*Complete MLR/DoE analysis suite equivalent to CAT DOE_* R scripts*")

    # Check if data is loaded (but don't block the entire page)
    data_loaded = 'current_data' in st.session_state and st.session_state.current_data is not None
    data = st.session_state.current_data if data_loaded else None

    # Create 7 tabs (always show tabs, even without data)
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "Model Computation",
        "Model Diagnostics",
        "Response Surface",
        "Confidence Intervals",
        "Predictions",
        "Generate Matrix",
        "Extract & Export"
    ])

    # ========================================================================
    # TAB 1: MODEL COMPUTATION - Complete workflow with all statistical tests
    # ========================================================================
    with tab1:


        if not data_loaded:
            st.warning("⚠️ **No data loaded**")
            st.info("💡 **Two options:**\n"
                   "1. Go to **Data Handling** to load your dataset\n"
                   "2. Use **Generate Matrix** tab to create a DoE design")
        else:
            # Import the full tab1 content from the original (simplified for illustration)
            # In practice, you would import model_computation_ui from a separate module
            from mlr_utils.model_computation import show_model_computation_ui
            show_model_computation_ui(data, "current_data")

    # ========================================================================
    # TAB 2: MODEL DIAGNOSTICS
    # ========================================================================
    with tab2:
        if not data_loaded:
            st.warning("⚠️ **No data loaded**")
            st.info("💡 Load data in **Data Handling** or create a design in **Generate Matrix**")
        elif 'mlr_model' not in st.session_state:
            st.warning("⚠️ **No MLR model fitted**")
            st.info("💡 Go to **Model Computation** tab to fit a model first")
        else:
            show_model_diagnostics_ui(
                model_results=st.session_state.mlr_model,
                X=st.session_state.mlr_model['X'],
                y=st.session_state.mlr_model['y']
            )

    # ========================================================================
    # TAB 3: RESPONSE SURFACE
    # ========================================================================
    with tab3:
        if not data_loaded:
            st.warning("⚠️ **No data loaded**")
            st.info("💡 Load data in **Data Handling** or create a design in **Generate Matrix**")
        elif st.session_state.get('mlr_model'):
            show_response_surface_ui(
                st.session_state.mlr_model,
                st.session_state.mlr_x_vars,
                st.session_state.mlr_y_var
            )
        else:
            st.warning("⚠️ **No MLR model fitted**")
            st.info("💡 Go to **Model Computation** tab to fit a model first")

    # ========================================================================
    # TAB 4: CONFIDENCE INTERVALS
    # ========================================================================
    with tab4:
        if not data_loaded:
            st.warning("⚠️ **No data loaded**")
            st.info("💡 Load data in **Data Handling** or create a design in **Generate Matrix**")
        elif st.session_state.get('mlr_model'):
            show_confidence_intervals_ui(
                st.session_state.mlr_model,
                st.session_state.mlr_x_vars,
                st.session_state.mlr_y_var
            )
        else:
            st.warning("⚠️ **No MLR model fitted**")
            st.info("💡 Go to **Model Computation** tab to fit a model first")

    # ========================================================================
    # TAB 5: PREDICTIONS
    # ========================================================================
    with tab5:
        if not data_loaded:
            st.warning("⚠️ **No data loaded**")
            st.info("💡 Load data in **Data Handling** or create a design in **Generate Matrix**")
        elif st.session_state.get('mlr_model'):
            show_predictions_ui(
                st.session_state.mlr_model,
                st.session_state.mlr_x_vars,
                st.session_state.mlr_y_var,
                data
            )
        else:
            st.warning("⚠️ **No MLR model fitted**")
            st.info("💡 Go to **Model Computation** tab to fit a model first")

    # ========================================================================
    # TAB 6: GENERATE MATRIX - STANDALONE (no model dependency)
    # ========================================================================
    with tab6:
        st.markdown("## Experimental Design Matrix Generator")
        st.markdown("*Standalone tool - works independently of loaded data or fitted models*")
        st.info("Create custom experimental designs without needing to load data first")
        show_candidate_points_ui()

    # ========================================================================
    # TAB 7: EXPORT
    # ========================================================================
    with tab7:
        if not data_loaded:
            st.warning("⚠️ **No data loaded**")
            st.info("💡 Load data in **Data Handling** or create a design in **Generate Matrix**")
        elif 'mlr_model' not in st.session_state:
            st.warning("⚠️ **No MLR model fitted**")
            st.info("💡 Go to **Model Computation** tab to fit a model first")
        else:
            show_export_ui(
                st.session_state.mlr_model,
                st.session_state.mlr_y_var
            )
