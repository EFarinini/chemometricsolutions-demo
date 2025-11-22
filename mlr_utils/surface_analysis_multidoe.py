"""
Multi-DOE Surface Analysis Module - REVISED VERSION
Per-response optimization criteria with side-by-side contour visualization

This module provides enhanced surface analysis UI for Multi-DOE with:
- Per-response optimization objectives (Y1, Y2, Y3 with different criteria)
- Acceptability thresholds (min/max bounds per response)
- Target values with tolerances
- Side-by-side contour layouts (2 columns)
- CI-aware conservative estimates
- Interpretation and recommendations
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy import stats

# Import existing surface calculation functions
from .surface_analysis import (
    calculate_response_surface,
    calculate_ci_surface,
    calculate_ci_experimental_surface,
    create_prediction_matrix
)

# Import batch processing functions
from .response_surface import (
    apply_optimization_surface,
    extract_surface_bounds
)


# ============================================================================
# SECTION 1: OPTIMIZATION OBJECTIVE PANEL (PER-RESPONSE)
# ============================================================================

def show_optimization_objective_panel(y_vars):
    """
    Collect per-response optimization criteria

    Displays a table-like interface where each response can have:
    - Optimization type (None, Maximize, Minimize, Target, Threshold_Above, Threshold_Below)
    - Acceptability bounds (min/max)
    - Target value (with tolerance)

    Args:
        y_vars: list of response variable names

    Returns:
        dict: response_criteria = {
            y_var: {
                'optimization': str,
                'acceptability_min': float or None,
                'acceptability_max': float or None,
                'target': float or None,
                'target_tolerance': float or None
            }
        }
    """

    with st.expander("🎯 Optimization Objective (optional)", expanded=True):
        st.markdown("**Define optimization goals and acceptability thresholds for each response variable**")
        st.info("""
        Configure per-response criteria:
        - **None**: Standard analysis (no optimization)
        - **Maximize/Minimize**: Find best values with acceptability bounds
        - **Target**: Hit a specific value within tolerance
        - **Threshold_Above**: Ensure response reliably exceeds minimum
        - **Threshold_Below**: Ensure response reliably stays below maximum
        """)

        # Table header
        col_header1, col_header2, col_header3, col_header4 = st.columns([2, 2, 1.5, 1.5])
        with col_header1:
            st.markdown("**Response**")
        with col_header2:
            st.markdown("**Optimization**")
        with col_header3:
            st.markdown("**Acceptability**")
        with col_header4:
            st.markdown("**Target**")

        st.markdown("---")

        response_criteria = {}

        # For each response variable
        for y_var in y_vars:
            col1, col2, col3, col4 = st.columns([2, 2, 1.5, 1.5])

            with col1:
                st.markdown(f"**{y_var}**")

            with col2:
                # Optimization type for THIS response
                opt_type = st.selectbox(
                    f"Objective for {y_var}",
                    ["None", "Maximize", "Minimize", "Target", "Threshold_Above", "Threshold_Below"],
                    key=f"opt_type_{y_var}",
                    label_visibility="collapsed"
                )

            with col3:
                # Acceptability inputs (conditional based on opt_type)
                if opt_type == "None":
                    st.caption("—")
                    acceptability_min = None
                    acceptability_max = None

                elif opt_type in ["Maximize", "Minimize"]:
                    # Show min/max bounds
                    acceptability_min = st.number_input(
                        f"Min {y_var}",
                        value=-0.4,
                        step=0.1,
                        format="%.2f",
                        key=f"acc_min_{y_var}",
                        label_visibility="collapsed"
                    )
                    acceptability_max = st.number_input(
                        f"Max {y_var}",
                        value=0.4,
                        step=0.1,
                        format="%.2f",
                        key=f"acc_max_{y_var}",
                        label_visibility="collapsed"
                    )

                elif opt_type == "Threshold_Above":
                    # Only show threshold (becomes min)
                    threshold = st.number_input(
                        f"Threshold {y_var}",
                        value=0.0,
                        step=0.1,
                        format="%.2f",
                        key=f"threshold_{y_var}",
                        help="Minimum acceptable value",
                        label_visibility="collapsed"
                    )
                    acceptability_min = threshold
                    acceptability_max = None

                elif opt_type == "Threshold_Below":
                    # Only show threshold (becomes max)
                    threshold = st.number_input(
                        f"Threshold {y_var}",
                        value=0.0,
                        step=0.1,
                        format="%.2f",
                        key=f"threshold_{y_var}",
                        help="Maximum acceptable value",
                        label_visibility="collapsed"
                    )
                    acceptability_min = None
                    acceptability_max = threshold

                else:  # Target
                    st.caption("—")
                    acceptability_min = None
                    acceptability_max = None

            with col4:
                # Target value (only for Target objective)
                if opt_type == "Target":
                    target = st.number_input(
                        f"Target {y_var}",
                        value=0.25,
                        step=0.1,
                        format="%.2f",
                        key=f"target_{y_var}",
                        label_visibility="collapsed"
                    )
                    target_tol = st.number_input(
                        f"Tol {y_var}",
                        value=0.1,
                        step=0.05,
                        format="%.2f",
                        key=f"target_tol_{y_var}",
                        help="Target tolerance",
                        label_visibility="collapsed"
                    )
                else:
                    st.caption("—")
                    target = None
                    target_tol = None

            # Store criteria for this response
            response_criteria[y_var] = {
                'optimization': opt_type,
                'acceptability_min': acceptability_min,
                'acceptability_max': acceptability_max,
                'target': target,
                'target_tolerance': target_tol
            }

        st.markdown("---")

        return response_criteria


# ============================================================================
# SECTION 2: UNIFIED CONTROL PANEL (SHARED CONFIG)
# ============================================================================

def show_unified_control_panel_multidoe(models_dict, x_vars, y_vars):
    """
    Unified control panel for shared Multi-DOE surface analysis configuration

    Collects:
    - Variable selection (2 for surface, others fixed)
    - Fixed values
    - CI type
    - Model parameters
    - Experimental parameters (conditional)
    - Surface range settings

    Args:
        models_dict: dict {y_name: model_result}
        x_vars: list of X variable names
        y_vars: list of Y variable names

    Returns:
        dict with configuration parameters or None if invalid
    """

    # ========================================================================
    # SECTION A: VARIABLE SELECTION
    # ========================================================================
    st.markdown("### Variable Selection")
    st.info("Select two variables to create 2D response surfaces. Other variables will be held at fixed values.")

    col1, col2 = st.columns(2)

    with col1:
        var1 = st.selectbox(
            "Variable for X-axis:",
            x_vars,
            key="multidoe_surface_analysis_var1",
            help="First variable to plot on X-axis"
        )

    with col2:
        var2 = st.selectbox(
            "Variable for Y-axis:",
            [v for v in x_vars if v != var1],
            key="multidoe_surface_analysis_var2",
            help="Second variable to plot on Y-axis"
        )

    # Fixed values for other variables
    other_vars = [v for v in x_vars if v not in [var1, var2]]
    fixed_values = {}

    if len(other_vars) > 0:
        st.markdown("### Fixed Values for Other Variables")
        st.info("Set values for variables not shown in the surface (typically 0 for coded variables)")

        cols = st.columns(min(3, len(other_vars)))
        for i, var in enumerate(other_vars):
            with cols[i % len(cols)]:
                fixed_values[var] = st.number_input(
                    f"{var}:",
                    value=0.0,
                    step=0.1,
                    format="%.2f",
                    key=f"multidoe_surface_fixed_val_{var}",
                    help=f"Fixed value for {var}"
                )

    st.markdown("---")

    # ========================================================================
    # SECTION B: CI TYPE SELECTION
    # ========================================================================
    st.markdown("### Confidence Interval Type")
    st.info("""
    Choose what type of confidence interval to calculate:
    - **Prediction**: Shows model confidence in predictions (model uncertainty only)
    - **Experimental**: Combines model + measurement uncertainty (error propagation)
    """)

    ci_type = st.radio(
        "Select CI calculation mode:",
        ["Prediction (Model Uncertainty Only)", "Experimental (Model + Measurement Uncertainty)"],
        key="multidoe_surface_ci_type",
        help="Prediction CI shows model confidence; Experimental CI includes measurement noise"
    )

    st.markdown("---")

    # ========================================================================
    # SECTION C: MODEL PARAMETERS (for PREDICTION CI)
    # Always needed, regardless of CI type
    # ========================================================================
    st.markdown("### Model Parameters (for Prediction CI calculation)")
    st.info("""
    These parameters define the **model's prediction uncertainty**.
    They are used in BOTH Prediction and Experimental CI modes.
    """)

    # Get reference model (first valid one)
    reference_model = None
    for y_name, model in models_dict.items():
        if 'error' not in model:
            reference_model = model
            break

    if reference_model is None:
        st.error("❌ No valid models available for surface analysis")
        return None

    # Variance method selector
    variance_method_model = st.radio(
        "Model variance estimated from:",
        ["Model residuals (from fitting)", "Independent measurement"],
        key="multidoe_surface_model_variance_source",
        help="Source of variance for the model"
    )

    if variance_method_model == "Model residuals (from fitting)":
        s_model = reference_model.get('rmse')
        dof_model = reference_model.get('dof')

        if s_model is None or dof_model is None or dof_model <= 0:
            st.error("❌ Model does not have valid RMSE or degrees of freedom")
            st.info("This may occur when the model is saturated (too many parameters for the data)")
            return None

        col_model1, col_model2 = st.columns(2)
        with col_model1:
            st.metric("Model Std Dev (s_model)", f"{s_model:.6f}")
            st.caption("From model fitting residuals")
        with col_model2:
            st.metric("Model DOF", dof_model)
            st.caption("n - p (samples - parameters)")

    else:
        st.markdown("**Enter Independent Model Variance**")
        st.caption("(Only use if you have separate measurements to estimate model variance)")

        col_model_ind1, col_model_ind2 = st.columns(2)

        with col_model_ind1:
            s_model = st.number_input(
                "Model standard deviation (s_model):",
                value=reference_model.get('rmse', 1.0),
                min_value=0.0001,
                format="%.6f",
                step=0.001,
                key="multidoe_surface_s_model_independent",
                help="Std dev for model predictions"
            )

        with col_model_ind2:
            dof_model = st.number_input(
                "Model DOF:",
                value=reference_model.get('dof', 5),
                min_value=1,
                step=1,
                key="multidoe_surface_dof_model_independent",
                help="Degrees of freedom"
            )

    st.markdown("---")

    # ========================================================================
    # SECTION D: EXPERIMENTAL PARAMETERS (CONDITIONAL)
    # ========================================================================
    s_exp = None
    dof_exp = None
    n_replicates = 1

    if ci_type == "Experimental":
        st.markdown("### Experimental Measurement Parameters")
        st.info("""
        These parameters define the **experimental measurement uncertainty**.

        **Formula**: CI_total = √((CI_model)² + (CI_experimental)²)

        where:
        - CI_model = t_model × s_model × √(leverage)
        - CI_experimental = t_exp × s_exp × √(1/n_replicates)
        """)

        # Number of replicate measurements
        col_exp_rep, col_exp_space = st.columns([2, 1])

        with col_exp_rep:
            n_replicates = st.number_input(
                "Number of replicate measurements (n):",
                min_value=1,
                max_value=100,
                value=1,
                step=1,
                key="multidoe_surface_n_replicates",
                help="""
                How many replicate measurements per experimental point?
                • n=1: CI_exp = t_exp × s_exp
                • n=2: CI_exp = t_exp × s_exp / √2 ≈ 0.707 × t_exp × s_exp
                • n=4: CI_exp = t_exp × s_exp / 2 = 0.5 × t_exp × s_exp

                Doubling replicates reduces uncertainty by √2 ≈ 1.414
                """
            )

        # Experimental variance source
        variance_method_exp = st.radio(
            "Experimental variance estimated from:",
            ["Model residuals (from fitting)", "Independent measurement"],
            key="multidoe_surface_exp_variance_source",
            help="Source of variance for experimental measurements"
        )

        if variance_method_exp == "Model residuals (from fitting)":
            # Use same values as model
            s_exp = s_model
            dof_exp = dof_model

            col_exp_m1, col_exp_m2 = st.columns(2)
            with col_exp_m1:
                st.metric("Experimental Std Dev (s_exp)", f"{s_exp:.6f}")
                st.caption("From model fitting residuals")
            with col_exp_m2:
                st.metric("Experimental DOF", dof_exp)
                st.caption("Same as model fit")

        else:
            st.markdown("**Enter Independent Experimental Variance**")
            st.caption("""
            This should come from replicate measurements of the same point,
            reflecting your measurement instrument precision/variability
            """)

            col_exp_i1, col_exp_i2 = st.columns(2)

            with col_exp_i1:
                s_exp = st.number_input(
                    "Experimental std deviation (s_exp):",
                    value=0.02,
                    min_value=0.0001,
                    format="%.6f",
                    step=0.001,
                    key="multidoe_surface_s_exp_independent",
                    help="Measurement variability (from replicate experiments)"
                )

            with col_exp_i2:
                dof_exp = st.number_input(
                    "Experimental DOF:",
                    value=5,
                    min_value=1,
                    step=1,
                    key="multidoe_surface_dof_exp_independent",
                    help="Degrees of freedom from experimental replicates"
                )

    st.markdown("---")

    # ========================================================================
    # SECTION E: SURFACE RANGE SETTINGS
    # ========================================================================
    st.markdown("### Surface Range Configuration")

    col_range1, col_range2, col_range3 = st.columns(3)

    with col_range1:
        min_range = st.number_input(
            "Minimum value:",
            value=-1.0,
            step=0.1,
            format="%.2f",
            key="multidoe_surface_min_range"
        )

    with col_range2:
        max_range = st.number_input(
            "Maximum value:",
            value=1.0,
            step=0.1,
            format="%.2f",
            key="multidoe_surface_max_range"
        )

    with col_range3:
        n_steps = st.number_input(
            "Grid resolution:",
            min_value=10,
            max_value=100,
            value=30,
            step=5,
            key="multidoe_surface_n_steps",
            help="Number of grid points (higher = smoother but slower)"
        )

    # ========================================================================
    # RETURN CONFIGURATION DICTIONARY
    # ========================================================================
    return {
        # Variables and ranges
        'var1': var1,
        'var2': var2,
        'v1_idx': x_vars.index(var1),
        'v2_idx': x_vars.index(var2),
        'fixed_values': fixed_values,

        # Model parameters
        's_model': s_model,
        'dof_model': dof_model,

        # CI type
        'ci_type': ci_type,

        # Experimental parameters (only if experimental mode)
        's_exp': s_exp,
        'dof_exp': dof_exp,
        'n_replicates': n_replicates,

        # Range parameters
        'n_steps': n_steps,
        'min_range': min_range,
        'max_range': max_range
    }


# ============================================================================
# SECTION 3: SURFACE CALCULATION (WITH PER-RESPONSE OPTIMIZATION)
# ============================================================================

def calculate_surfaces_multidoe(models_dict, config, response_criteria, x_vars, y_vars):
    """
    Calculate response + CI surfaces with per-response optimization

    Args:
        models_dict: {y_var: model_result}
        config: unified config from show_unified_control_panel_multidoe()
        response_criteria: per-response optimization from show_optimization_objective_panel()
        x_vars: list of X variable names
        y_vars: list of Y variable names

    Returns:
        dict: {y_var: {
            'x_grid', 'y_grid', 'response_grid', 'ci_grid',
            'optimized_surface', 'bounds'
        }}
    """
    surfaces_dict = {}
    value_range = (config['min_range'], config['max_range'])

    for y_var in y_vars:
        model = models_dict.get(y_var)

        if model is None or 'error' in model:
            continue

        try:
            # Calculate raw response surface
            x_grid, y_grid, response_grid, _ = calculate_response_surface(
                model_results=model,
                x_vars=x_vars,
                v1_idx=config['v1_idx'],
                v2_idx=config['v2_idx'],
                fixed_values=config['fixed_values'],
                n_steps=config['n_steps'],
                value_range=value_range
            )

            # Calculate CI surface
            if config['ci_type'] == "Prediction":
                _, _, ci_grid, _ = calculate_ci_surface(
                    model_results=model,
                    x_vars=x_vars,
                    v1_idx=config['v1_idx'],
                    v2_idx=config['v2_idx'],
                    fixed_values=config['fixed_values'],
                    s=model.get('rmse'),
                    dof=model.get('dof'),
                    n_steps=config['n_steps'],
                    value_range=value_range
                )
            else:  # Experimental
                _, _, _, _, ci_grid, _ = calculate_ci_experimental_surface(
                    model_results=model,
                    x_vars=x_vars,
                    v1_idx=config['v1_idx'],
                    v2_idx=config['v2_idx'],
                    fixed_values=config['fixed_values'],
                    s_model=config['s_model'],
                    dof_model=config['dof_model'],
                    s_exp=config['s_exp'],
                    dof_exp=config['dof_exp'],
                    n_replicates=config['n_replicates'],
                    n_steps=config['n_steps'],
                    value_range=value_range
                )

            # Get per-response optimization
            opt_obj = response_criteria[y_var]['optimization']

            # Apply optimization transformation
            if opt_obj != "None":
                optimized = apply_optimization_surface(response_grid, ci_grid, opt_obj)
            else:
                optimized = response_grid.copy()

            # Extract bounds
            bounds = extract_surface_bounds(response_grid, ci_grid)

            surfaces_dict[y_var] = {
                'x_grid': x_grid,
                'y_grid': y_grid,
                'response_grid': response_grid,
                'ci_grid': ci_grid,
                'optimized_surface': optimized,
                'bounds': bounds
            }

        except Exception as e:
            st.warning(f"⚠️ Error calculating surface for {y_var}: {str(e)}")
            continue

    return surfaces_dict


# ============================================================================
# SECTION 4: PLOTTING (PER-RESPONSE CONTOUR)
# ============================================================================

def create_response_contour_multidoe(surface_data, var1_name, var2_name, y_var,
                                     fixed_values, optimization_objective, response_criteria):
    """
    Create contour plot for single response with optimization overlay AND threshold lines

    Args:
        surface_data: dict with x_grid, y_grid, response_grid, ci_grid, optimized_surface
        var1_name, var2_name: axis labels
        y_var: response variable name
        fixed_values: dict of fixed values
        optimization_objective: "None", "Maximize", "Minimize", "Target", "Threshold_Above", "Threshold_Below"
        response_criteria: {optimization, acceptability_min, acceptability_max, target, target_tolerance}

    Returns:
        plotly Figure with contours AND threshold lines
    """

    # ========================================================================
    # STEP 1: Select which surface to plot
    # ========================================================================
    if optimization_objective != "None":
        z_data = surface_data['optimized_surface']

        if optimization_objective == "Maximize":
            title_suffix = " (Conservative: z - CI)"
        elif optimization_objective == "Minimize":
            title_suffix = " (Conservative: z + CI)"
        elif optimization_objective == "Threshold_Above":
            title_suffix = " (Conservative: z - CI ≥ threshold)"
        elif optimization_objective == "Threshold_Below":
            title_suffix = " (Conservative: z + CI ≤ threshold)"
        elif optimization_objective == "Target":
            title_suffix = " (Target mode)"
        else:
            title_suffix = ""
    else:
        z_data = surface_data['response_grid']
        title_suffix = ""

    # ========================================================================
    # STEP 2: Create base contour plot
    # ========================================================================
    fig = go.Figure(data=[
        go.Contour(
            x=surface_data['x_grid'][0, :],
            y=surface_data['y_grid'][:, 0],
            z=z_data,
            colorscale='Viridis',
            contours=dict(
                coloring='heatmap',
                showlabels=True,
                labelfont=dict(size=10, color='white')
            ),
            colorbar=dict(title=y_var),
            hovertemplate=f'{var1_name}: %{{x:.3f}}<br>{var2_name}: %{{y:.3f}}<br>{y_var}: %{{z:.4f}}<extra></extra>',
            ncontours=15
        )
    ])

    # ========================================================================
    # STEP 3: ADD THRESHOLD LINES (RED SOLID)
    # ========================================================================
    # Add contour lines at acceptability boundaries

    if response_criteria.get('acceptability_min') is not None:
        # Add MIN acceptability contour line (red solid)
        threshold_min = response_criteria['acceptability_min']

        fig.add_trace(go.Contour(
            x=surface_data['x_grid'][0, :],
            y=surface_data['y_grid'][:, 0],
            z=z_data,
            contours=dict(
                start=threshold_min,
                end=threshold_min,
                size=1,
                coloring='none',
                showlabels=True,
                labelfont=dict(size=12, color='red')
            ),
            line=dict(color='red', width=3, dash='solid'),
            showscale=False,
            name=f'Min: {threshold_min:.3f}',
            hovertemplate=f'Min Threshold: {threshold_min:.3f}<extra></extra>'
        ))

    if response_criteria.get('acceptability_max') is not None:
        # Add MAX acceptability contour line (red solid)
        threshold_max = response_criteria['acceptability_max']

        fig.add_trace(go.Contour(
            x=surface_data['x_grid'][0, :],
            y=surface_data['y_grid'][:, 0],
            z=z_data,
            contours=dict(
                start=threshold_max,
                end=threshold_max,
                size=1,
                coloring='none',
                showlabels=True,
                labelfont=dict(size=12, color='red')
            ),
            line=dict(color='red', width=3, dash='solid'),
            showscale=False,
            name=f'Max: {threshold_max:.3f}',
            hovertemplate=f'Max Threshold: {threshold_max:.3f}<extra></extra>'
        ))

    # Special handling for Target mode: add target ± tolerance contours
    if optimization_objective == "Target" and response_criteria.get('target') is not None:
        target = response_criteria['target']
        tolerance = response_criteria.get('target_tolerance', 0.1)

        # Add lower tolerance contour (orange dotted)
        lower_band = target - tolerance
        fig.add_trace(go.Contour(
            x=surface_data['x_grid'][0, :],
            y=surface_data['y_grid'][:, 0],
            z=z_data,
            contours=dict(
                start=lower_band,
                end=lower_band,
                size=1,
                coloring='none',
                showlabels=True,
                labelfont=dict(size=10, color='orange')
            ),
            line=dict(color='orange', width=2, dash='dot'),
            showscale=False,
            name=f'Target - Tol: {lower_band:.3f}',
            hovertemplate=f'Lower Tolerance: {lower_band:.3f}<extra></extra>'
        ))

        # Add target contour (green solid)
        fig.add_trace(go.Contour(
            x=surface_data['x_grid'][0, :],
            y=surface_data['y_grid'][:, 0],
            z=z_data,
            contours=dict(
                start=target,
                end=target,
                size=1,
                coloring='none',
                showlabels=True,
                labelfont=dict(size=12, color='darkgreen', family='Arial Black')
            ),
            line=dict(color='darkgreen', width=4, dash='solid'),
            showscale=False,
            name=f'Target: {target:.3f}',
            hovertemplate=f'Target: {target:.3f}<extra></extra>'
        ))

        # Add upper tolerance contour (orange dotted)
        upper_band = target + tolerance
        fig.add_trace(go.Contour(
            x=surface_data['x_grid'][0, :],
            y=surface_data['y_grid'][:, 0],
            z=z_data,
            contours=dict(
                start=upper_band,
                end=upper_band,
                size=1,
                coloring='none',
                showlabels=True,
                labelfont=dict(size=10, color='orange')
            ),
            line=dict(color='orange', width=2, dash='dot'),
            showscale=False,
            name=f'Target + Tol: {upper_band:.3f}',
            hovertemplate=f'Upper Tolerance: {upper_band:.3f}<extra></extra>'
        ))

    # ========================================================================
    # STEP 4: Build title and update layout
    # ========================================================================
    fixed_str = ", ".join([f"{k}={v:.2f}" for k, v in fixed_values.items()]) if fixed_values else ""

    title_text = f"{y_var} Response Surface{title_suffix}"
    if fixed_str:
        title_text += f"<br><sub>{fixed_str}</sub>"

    fig.update_layout(
        title=title_text,
        xaxis_title=var1_name,
        yaxis_title=var2_name,
        height=500,
        yaxis=dict(scaleanchor="x", scaleratio=1),
        margin=dict(l=50, r=150, t=100, b=60),
        hovermode='closest',
        showlegend=True,
        legend=dict(
            x=0.01,
            y=1.12,
            xanchor='left',
            yanchor='top',
            bgcolor='rgba(255,255,255,0.95)',
            bordercolor='rgba(0,0,0,0.3)',
            borderwidth=1,
            orientation='h'
        )
    )

    return fig


# ============================================================================
# SECTION 5: INTERPRETATION & RECOMMENDATIONS (PER-RESPONSE)
# ============================================================================

def show_surface_interpretation_multidoe(surfaces_dict, models_dict, response_criteria, x_vars, y_vars, config):
    """
    Show interpretation panel with bounds, optimal regions, and recommendations

    Displays:
    1. Summary table (one row per response)
    2. Per-response optimization results
    3. Recommendations

    Args:
        surfaces_dict: dict from calculate_surfaces_multidoe
        models_dict: dict of model results
        response_criteria: per-response optimization criteria
        x_vars: list of X variable names
        y_vars: list of Y variable names
        config: configuration dict
    """
    st.markdown("### 📈 Interpretation & Recommendations")

    # ========================================================================
    # SUMMARY TABLE
    # ========================================================================
    st.markdown("#### Summary Statistics")

    table_data = []
    for y_var, surface in surfaces_dict.items():
        criteria = response_criteria[y_var]
        bounds = surface['bounds']

        # Calculate feasibility
        if criteria['optimization'] in ["Maximize", "Minimize"]:
            opt_surface = surface['optimized_surface']
            acc_min = criteria.get('acceptability_min')
            acc_max = criteria.get('acceptability_max')

            if acc_min is not None and acc_max is not None:
                feasible_mask = (opt_surface >= acc_min) & (opt_surface <= acc_max)
                feasible_pct = (feasible_mask.sum() / opt_surface.size) * 100
            else:
                feasible_pct = 100.0
        elif criteria['optimization'] == "Threshold_Above":
            opt_surface = surface['optimized_surface']
            threshold = criteria.get('acceptability_min')
            if threshold is not None:
                feasible_mask = opt_surface >= threshold
                feasible_pct = (feasible_mask.sum() / opt_surface.size) * 100
            else:
                feasible_pct = 100.0
        elif criteria['optimization'] == "Threshold_Below":
            opt_surface = surface['optimized_surface']
            threshold = criteria.get('acceptability_max')
            if threshold is not None:
                feasible_mask = opt_surface <= threshold
                feasible_pct = (feasible_mask.sum() / opt_surface.size) * 100
            else:
                feasible_pct = 100.0
        else:
            feasible_pct = 100.0

        table_data.append({
            'Response': y_var,
            'Optimization': criteria['optimization'],
            'Acc_Min': f"{criteria['acceptability_min']:.2f}" if criteria['acceptability_min'] is not None else "—",
            'Acc_Max': f"{criteria['acceptability_max']:.2f}" if criteria['acceptability_max'] is not None else "—",
            'Min_Observed': f"{bounds['min']:.4f}",
            'Max_Observed': f"{bounds['max']:.4f}",
            '%_Feasible': f"{feasible_pct:.1f}%"
        })

    summary_df = pd.DataFrame(table_data)
    st.dataframe(summary_df, use_container_width=True)

    st.markdown("---")

    # ========================================================================
    # PER-RESPONSE OPTIMIZATION RESULTS
    # ========================================================================
    st.markdown("#### Per-Response Optimization Results")

    for y_var, surface in surfaces_dict.items():
        criteria = response_criteria[y_var]
        opt_obj = criteria['optimization']

        if opt_obj == "None":
            continue

        st.markdown(f"##### {y_var}")

        opt_surface = surface['optimized_surface']

        if opt_obj == "Maximize":
            # Find maximum
            max_idx = np.argmax(opt_surface)
            max_i, max_j = np.unravel_index(max_idx, opt_surface.shape)

            opt_val = opt_surface[max_i, max_j]
            var1_val = surface['x_grid'][max_i, max_j]
            var2_val = surface['y_grid'][max_i, max_j]
            ci_val = surface['ci_grid'][max_i, max_j]

            st.success(f"""
            **Best conservative value: {opt_val:.4f}** (at {config['var1']}={var1_val:.3f}, {config['var2']}={var2_val:.3f})
            - CI at this point: ±{ci_val:.4f}
            - You'll exceed this value 95% of the time
            """)

        elif opt_obj == "Minimize":
            # Find minimum
            min_idx = np.argmin(opt_surface)
            min_i, min_j = np.unravel_index(min_idx, opt_surface.shape)

            opt_val = opt_surface[min_i, min_j]
            var1_val = surface['x_grid'][min_i, min_j]
            var2_val = surface['y_grid'][min_i, min_j]
            ci_val = surface['ci_grid'][min_i, min_j]

            st.success(f"""
            **Best conservative value: {opt_val:.4f}** (at {config['var1']}={var1_val:.3f}, {config['var2']}={var2_val:.3f})
            - CI at this point: ±{ci_val:.4f}
            - You'll stay below this value 95% of the time
            """)

        elif opt_obj == "Threshold_Above":
            threshold = criteria['acceptability_min']
            feasible_mask = opt_surface >= threshold
            feasible_pct = (feasible_mask.sum() / opt_surface.size) * 100

            if feasible_pct > 0:
                best_val = opt_surface[feasible_mask].max()
                st.success(f"""
                **Feasible region: {feasible_pct:.1f}%** of design space
                - Threshold: ≥ {threshold:.4f}
                - Best conservative value in feasible region: {best_val:.4f}
                """)
            else:
                st.error(f"""
                **No feasible region found**
                - Threshold: ≥ {threshold:.4f}
                - Maximum conservative value: {opt_surface.max():.4f}
                """)

        elif opt_obj == "Threshold_Below":
            threshold = criteria['acceptability_max']
            feasible_mask = opt_surface <= threshold
            feasible_pct = (feasible_mask.sum() / opt_surface.size) * 100

            if feasible_pct > 0:
                best_val = opt_surface[feasible_mask].min()
                st.success(f"""
                **Feasible region: {feasible_pct:.1f}%** of design space
                - Threshold: ≤ {threshold:.4f}
                - Best conservative value in feasible region: {best_val:.4f}
                """)
            else:
                st.error(f"""
                **No feasible region found**
                - Threshold: ≤ {threshold:.4f}
                - Minimum conservative value: {opt_surface.min():.4f}
                """)

        elif opt_obj == "Target":
            target = criteria['target']
            target_tol = criteria['target_tolerance']

            # Find closest to target
            distance = np.abs(opt_surface - target)
            min_dist_idx = np.argmin(distance)
            min_i, min_j = np.unravel_index(min_dist_idx, distance.shape)

            closest_val = opt_surface[min_i, min_j]
            var1_val = surface['x_grid'][min_i, min_j]
            var2_val = surface['y_grid'][min_i, min_j]
            deviation = closest_val - target

            if abs(deviation) <= target_tol:
                st.success(f"""
                **Closest to target: {closest_val:.4f}** (at {config['var1']}={var1_val:.3f}, {config['var2']}={var2_val:.3f})
                - Target: {target:.4f} ± {target_tol:.4f}
                - Deviation: {deviation:+.4f}
                - ✅ Within tolerance
                """)
            else:
                st.warning(f"""
                **Closest to target: {closest_val:.4f}** (at {config['var1']}={var1_val:.3f}, {config['var2']}={var2_val:.3f})
                - Target: {target:.4f} ± {target_tol:.4f}
                - Deviation: {deviation:+.4f}
                - ⚠️ Outside tolerance
                """)


# ============================================================================
# SECTION 6: MAIN UI FUNCTION (REORDERED)
# ============================================================================

def show_surface_analysis_ui_multidoe(models_dict, x_vars, y_vars):
    """
    Multi-response surface analysis UI with per-response optimization criteria

    NEW SECTION ORDER:
    1. Optimization Objective (per-response)
    2. Unified Control Panel (shared config)
    3. Generate Button
    4. Response Surface Visualization (2 columns)
    5. Interpretation & Recommendations

    Args:
        models_dict (dict): Dictionary {y_name: model_result}
        x_vars (list): List of X variable names
        y_vars (list): List of Y variable names
    """
    st.markdown("## 📊 Response Surface Analysis")
    st.info("**Per-response optimization with side-by-side contour visualization**")

    if len(y_vars) < 1:
        st.warning("No response variables available")
        return

    # ========================================================================
    # SECTION 1: OPTIMIZATION OBJECTIVE PANEL (PER-RESPONSE)
    # ========================================================================
    response_criteria = show_optimization_objective_panel(y_vars)

    st.markdown("---")

    # ========================================================================
    # SECTION 2: UNIFIED CONTROL PANEL (SHARED CONFIG)
    # ========================================================================
    config = show_unified_control_panel_multidoe(models_dict, x_vars, y_vars)

    if config is None:
        return  # Invalid configuration

    st.markdown("---")

    # ========================================================================
    # SECTION 3: GENERATE BUTTON
    # ========================================================================
    if st.button("🚀 Generate Surfaces", type="primary", key="multidoe_generate_surfaces_revised"):
        try:
            with st.spinner(f"Computing response surfaces for {len(y_vars)} responses..."):
                # Calculate all surfaces with per-response optimization
                surfaces_dict = calculate_surfaces_multidoe(
                    models_dict,
                    config,
                    response_criteria,
                    x_vars,
                    y_vars
                )

                # Store in session state
                st.session_state.multidoe_surfaces_data = surfaces_dict
                st.session_state.multidoe_surface_config = config
                st.session_state.multidoe_response_criteria = response_criteria

                # Store CI parameters for predictions module
                st.session_state.multidoe_ci_params = {
                    'ci_type': config['ci_type'],
                    's_model': config['s_model'],
                    'dof_model': config['dof_model'],
                    's_exp': config.get('s_exp'),
                    'dof_exp': config.get('dof_exp'),
                    'n_replicates': config.get('n_replicates', 1)
                }

            st.success(f"✅ Generated {len(surfaces_dict)} response surfaces ({(config['n_steps']+1)**2} points each)")

        except Exception as e:
            st.error(f"❌ Error generating surface analysis: {str(e)}")
            import traceback
            with st.expander("🐛 Error details"):
                st.code(traceback.format_exc())

    # ========================================================================
    # SECTION 4: RESPONSE SURFACE VISUALIZATION (2 COLUMNS)
    # ========================================================================
    if 'multidoe_surfaces_data' in st.session_state:
        surfaces_dict = st.session_state.multidoe_surfaces_data
        stored_criteria = st.session_state.get('multidoe_response_criteria', response_criteria)
        stored_config = st.session_state.get('multidoe_surface_config', config)

        st.markdown("---")
        st.markdown("### 📊 Contour Surfaces")

        y_vars_list = list(surfaces_dict.keys())

        # Display in 2-column grid
        for i in range(0, len(y_vars_list), 2):
            col1, col2 = st.columns(2)

            # First plot
            with col1:
                y_var = y_vars_list[i]
                fig = create_response_contour_multidoe(
                    surfaces_dict[y_var],
                    stored_config['var1'],
                    stored_config['var2'],
                    y_var,
                    stored_config['fixed_values'],
                    stored_criteria[y_var]['optimization'],
                    stored_criteria[y_var]
                )
                st.plotly_chart(fig, use_container_width=True)

                # Statistics
                col_stat1, col_stat2 = st.columns(2)
                with col_stat1:
                    st.metric(f"{y_var} Min", f"{surfaces_dict[y_var]['bounds']['min']:.4f}")
                with col_stat2:
                    st.metric(f"{y_var} Max", f"{surfaces_dict[y_var]['bounds']['max']:.4f}")

            # Second plot (if exists)
            if i + 1 < len(y_vars_list):
                with col2:
                    y_var = y_vars_list[i + 1]
                    fig = create_response_contour_multidoe(
                        surfaces_dict[y_var],
                        stored_config['var1'],
                        stored_config['var2'],
                        y_var,
                        stored_config['fixed_values'],
                        stored_criteria[y_var]['optimization'],
                        stored_criteria[y_var]
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Statistics
                    col_stat1, col_stat2 = st.columns(2)
                    with col_stat1:
                        st.metric(f"{y_var} Min", f"{surfaces_dict[y_var]['bounds']['min']:.4f}")
                    with col_stat2:
                        st.metric(f"{y_var} Max", f"{surfaces_dict[y_var]['bounds']['max']:.4f}")

        # ====================================================================
        # SECTION 5: INTERPRETATION & RECOMMENDATIONS
        # ====================================================================
        st.markdown("---")
        show_surface_interpretation_multidoe(
            surfaces_dict,
            models_dict,
            stored_criteria,
            x_vars,
            y_vars,
            stored_config
        )
