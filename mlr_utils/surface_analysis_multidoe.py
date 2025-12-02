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
# PLOT SETTINGS CONSTANTS
# ============================================================================
# Standardized plot dimensions and margins for consistent visual appearance
PLOT_HEIGHT = 550
PLOT_MARGINS = dict(l=60, r=50, t=80, b=60)


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
                'target_min': float or None,
                'target_max': float or None
            }
        }
    """

    with st.expander("🎯 Optimization Objective (optional)", expanded=True):
        st.markdown("**Define optimization goals and acceptability thresholds for each response variable**")
        st.info("""
        Configure per-response criteria:
        - **None**: Standard analysis (mean response surface)
        - **Maximize**: Conservative surface (z - CI) → find safe maximum
        - **Minimize**: Conservative surface (z + CI) → find safe minimum
        - **Target**: Define range [min, max] → shows feasibility map (green=safe, red=uncertain)
        - **Threshold_Above**: Response must exceed threshold (shows z - CI with line)
        - **Threshold_Below**: Response must stay below threshold (shows z + CI with line)

        **Note**: Each response uses its own model RMSE/DOF for CI calculation.
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
                    # Maximize/Minimize: NO acceptability inputs needed
                    # Surface shows conservative estimate only (lower CI for Max, upper CI for Min)
                    # Use Threshold_Above/Below if you need acceptability bounds with lines
                    st.caption("—")
                    acceptability_min = None
                    acceptability_max = None

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
                # Target range (only for Target objective)
                if opt_type == "Target":
                    target_min = st.number_input(
                        f"Min",
                        value=0.0,
                        step=0.1,
                        format="%.2f",
                        key=f"target_min_{y_var}",
                        label_visibility="collapsed",
                        help="Minimum acceptable value"
                    )
                    target_max = st.number_input(
                        f"Max",
                        value=1.0,
                        step=0.1,
                        format="%.2f",
                        key=f"target_max_{y_var}",
                        label_visibility="collapsed",
                        help="Maximum acceptable value"
                    )
                else:
                    st.caption("—")
                    target_min = None
                    target_max = None

            # Store criteria for this response
            response_criteria[y_var] = {
                'optimization': opt_type,
                'acceptability_min': acceptability_min,
                'acceptability_max': acceptability_max,
                'target_min': target_min,
                'target_max': target_max
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
    # ========================================================================
    st.markdown("### Model Parameters (for Prediction CI calculation)")
    st.info("""
    **Multi-DOE Note**: Each response model has its own RMSE and DOF.
    The CI for each response is calculated using **its own model parameters**.

    Formula: CI = t(α/2, DOF) × RMSE × √(leverage)
    """)

    # Show table of all model parameters
    model_params_data = []
    for y_name, model in models_dict.items():
        if 'error' not in model:
            model_params_data.append({
                'Response': y_name,
                'RMSE (s)': f"{model.get('rmse', np.nan):.6f}" if not np.isnan(model.get('rmse', np.nan)) else "—",
                'DOF': model.get('dof', '—'),
                'R²': f"{model.get('r_squared', np.nan):.4f}" if not np.isnan(model.get('r_squared', np.nan)) else "—",
                'Status': '✅ Valid' if model.get('dof', 0) > 0 else '⚠️ Saturated'
            })
        else:
            model_params_data.append({
                'Response': y_name,
                'RMSE (s)': "—",
                'DOF': "—",
                'R²': "—",
                'Status': f"❌ Error"
            })

    if model_params_data:
        params_df = pd.DataFrame(model_params_data)
        st.dataframe(params_df, use_container_width=True, hide_index=True)

    # Check if any model has valid DOF
    valid_models = [m for m in models_dict.values() if 'error' not in m and m.get('dof', 0) > 0]
    if not valid_models:
        st.error("❌ No valid models with DOF > 0 for CI calculation")
        return None

    # For Experimental CI mode, we need s_exp and dof_exp (shared across responses)
    # These are NOT per-model - they represent measurement uncertainty
    reference_model = valid_models[0]
    s_model = reference_model.get('rmse')
    dof_model = reference_model.get('dof')

    st.markdown("---")

    # ========================================================================
    # SECTION D: EXPERIMENTAL PARAMETERS (CONDITIONAL)
    # ========================================================================
    s_exp = None
    dof_exp = None
    n_replicates = 1

    if ci_type == "Experimental (Model + Measurement Uncertainty)":
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
            # IMPORTANT: Each response uses ITS OWN model RMSE and DOF
            # This is correct - each model has different fit quality
            model_rmse = model.get('rmse')
            model_dof = model.get('dof')

            if config['ci_type'] == "Prediction (Model Uncertainty Only)":
                _, _, ci_grid, _ = calculate_ci_surface(
                    model_results=model,
                    x_vars=x_vars,
                    v1_idx=config['v1_idx'],
                    v2_idx=config['v2_idx'],
                    fixed_values=config['fixed_values'],
                    s=model_rmse,      # Use THIS model's RMSE (not shared!)
                    dof=model_dof,     # Use THIS model's DOF (not shared!)
                    n_steps=config['n_steps'],
                    value_range=value_range
                )
            else:  # Experimental (Model + Measurement Uncertainty)
                # Model component: use THIS model's RMSE/DOF
                # Experimental component: use shared s_exp/dof_exp (measurement uncertainty)
                _, _, _, _, ci_grid, _ = calculate_ci_experimental_surface(
                    model_results=model,
                    x_vars=x_vars,
                    v1_idx=config['v1_idx'],
                    v2_idx=config['v2_idx'],
                    fixed_values=config['fixed_values'],
                    s_model=model_rmse,      # Use THIS model's RMSE
                    dof_model=model_dof,     # Use THIS model's DOF
                    s_exp=config['s_exp'],   # Shared experimental variance
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

            # Extract bounds from the optimized surface (what we actually display)
            bounds = {
                'min': float(np.nanmin(optimized)),
                'max': float(np.nanmax(optimized)),
                'mean': float(np.nanmean(optimized)),
                'std': float(np.nanstd(optimized))
            }

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
        response_criteria: {optimization, acceptability_min, acceptability_max, target_min, target_max}

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
            colorbar=dict(
                title=y_var,
                len=1.0,
                y=0.5,
                yanchor='middle'
            ),
            hovertemplate=f'{var1_name}: %{{x:.3f}}<br>{var2_name}: %{{y:.3f}}<br>{y_var}: %{{z:.4f}}<extra></extra>',
            ncontours=15
        )
    ])

    # ========================================================================
    # STEP 3: ADD THRESHOLD LINES (RED SOLID) - Only for Threshold modes
    # ========================================================================
    # Add contour lines at acceptability boundaries
    # Skip for Maximize/Minimize - they only show conservative surface without lines

    if optimization_objective not in ["Maximize", "Minimize"]:
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
        height=PLOT_HEIGHT,
        yaxis=dict(scaleanchor="x", scaleratio=1),
        margin=PLOT_MARGINS,
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


def create_target_feasibility_plots(surface_data, var1_name, var2_name, y_var,
                                    fixed_values, response_criteria):
    """
    Create TWO SEPARATE plots for Target mode:
    1. Response surface with target min/max lines
    2. Feasibility heatmap (green = safe, red = uncertain)

    Green = entire CI interval is within target range (FEASIBLE)
    Red = CI interval extends outside target range (INFEASIBLE)

    Args:
        surface_data: dict with x_grid, y_grid, response_grid, ci_grid
        var1_name, var2_name: axis labels
        y_var: response variable name
        fixed_values: dict of fixed values
        response_criteria: dict with target_min, target_max

    Returns:
        tuple: (fig_response, fig_feasibility, feasible_pct, feasible_mask)
    """
    # Extract data
    x_grid = surface_data['x_grid']
    y_grid = surface_data['y_grid']
    z_pred = surface_data['response_grid']  # Mean prediction
    ci_grid = surface_data['ci_grid']       # CI semiamplitude

    target_min = response_criteria['target_min']
    target_max = response_criteria['target_max']

    # Calculate feasibility mask
    # Point is FEASIBLE if: (z_pred - CI >= target_min) AND (z_pred + CI <= target_max)
    lower_bound = z_pred - ci_grid  # Conservative lower (95% sure we're above this)
    upper_bound = z_pred + ci_grid  # Conservative upper (95% sure we're below this)

    feasible_mask = (lower_bound >= target_min) & (upper_bound <= target_max)

    # Convert to numeric for plotting (1 = feasible, 0 = infeasible)
    feasibility_grid = feasible_mask.astype(float)

    # Calculate feasibility percentage
    feasible_pct = (feasible_mask.sum() / feasible_mask.size) * 100

    # Calculate PAIR (largest feasible rectangle) for QbD analysis
    pair_result = find_largest_feasible_rectangle(
        feasible_mask,
        x_grid[0, :],  # x coordinates
        y_grid[:, 0]   # y coordinates
    )

    # Fixed values string for subtitle
    fixed_str = ", ".join([f"{k}={v:.2f}" for k, v in fixed_values.items()]) if fixed_values else ""

    # ========================================================================
    # FIGURE 1: Response Surface with Target Lines
    # ========================================================================
    fig_response = go.Figure()

    # Main response surface contour
    fig_response.add_trace(
        go.Contour(
            x=x_grid[0, :],
            y=y_grid[:, 0],
            z=z_pred,
            colorscale='Viridis',
            contours=dict(
                coloring='heatmap',
                showlabels=True,
                labelfont=dict(size=10, color='white')
            ),
            colorbar=dict(
                title=y_var
            ),
            hovertemplate=f'{var1_name}: %{{x:.3f}}<br>{var2_name}: %{{y:.3f}}<br>{y_var}: %{{z:.4f}}<extra></extra>',
            ncontours=15
        )
    )

    # Add target min line (red dashed)
    fig_response.add_trace(
        go.Contour(
            x=x_grid[0, :],
            y=y_grid[:, 0],
            z=z_pred,
            contours=dict(
                start=target_min,
                end=target_min,
                size=1,
                coloring='none',
                showlabels=True,
                labelfont=dict(size=12, color='red')
            ),
            line=dict(color='red', width=3, dash='dash'),
            showscale=False,
            name=f'Target Min: {target_min:.2f}',
            hoverinfo='skip'
        )
    )

    # Add target max line (red dashed)
    fig_response.add_trace(
        go.Contour(
            x=x_grid[0, :],
            y=y_grid[:, 0],
            z=z_pred,
            contours=dict(
                start=target_max,
                end=target_max,
                size=1,
                coloring='none',
                showlabels=True,
                labelfont=dict(size=12, color='red')
            ),
            line=dict(color='red', width=3, dash='dash'),
            showscale=False,
            name=f'Target Max: {target_max:.2f}',
            hoverinfo='skip'
        )
    )

    title_response = f"{y_var} Response Surface (Target: [{target_min:.2f}, {target_max:.2f}])"
    if fixed_str:
        title_response += f"<br><sub>{fixed_str}</sub>"

    fig_response.update_layout(
        title=title_response,
        xaxis_title=var1_name,
        yaxis_title=var2_name,
        height=PLOT_HEIGHT,
        yaxis=dict(scaleanchor="x", scaleratio=1),
        margin=PLOT_MARGINS,
        showlegend=False
    )

    # ========================================================================
    # FIGURE 2: Feasibility Heatmap (Green/Red)
    # ========================================================================
    fig_feasibility = go.Figure()

    # Binary colorscale (sharp boundary)
    binary_colorscale = [
        [0.0, 'rgb(220, 60, 60)'],    # Red (infeasible)
        [0.49, 'rgb(220, 60, 60)'],   # Red
        [0.51, 'rgb(60, 180, 60)'],   # Green (feasible)
        [1.0, 'rgb(60, 180, 60)']     # Green
    ]

    fig_feasibility.add_trace(
        go.Heatmap(
            x=x_grid[0, :],
            y=y_grid[:, 0],
            z=feasibility_grid,
            colorscale=binary_colorscale,
            showscale=True,
            colorbar=dict(
                title='Feasible',
                tickvals=[0, 1],
                ticktext=['No (CI outside)', 'Yes (CI inside)']
            ),
            hovertemplate=f'{var1_name}: %{{x:.3f}}<br>{var2_name}: %{{y:.3f}}<br>Feasible: %{{z:.0f}}<extra></extra>',
            zmin=0,
            zmax=1
        )
    )

    # Add contour line showing feasibility boundary
    fig_feasibility.add_trace(
        go.Contour(
            x=x_grid[0, :],
            y=y_grid[:, 0],
            z=feasibility_grid,
            contours=dict(
                start=0.5,
                end=0.5,
                size=1,
                coloring='none'
            ),
            line=dict(color='black', width=2),
            showscale=False,
            name='Feasibility Boundary',
            hoverinfo='skip'
        )
    )

    # Note: PAIR rectangle is shown only in the NOR/PAR/PAIR comparison figure below
    # Feasibility map shows only green/red with black boundary for clarity

    title_feasibility = f"{y_var} Feasibility Map | {feasible_pct:.1f}% Feasible"
    if fixed_str:
        title_feasibility += f"<br><sub>{fixed_str}</sub>"

    fig_feasibility.update_layout(
        title=title_feasibility,
        xaxis_title=var1_name,
        yaxis_title=var2_name,
        height=PLOT_HEIGHT,
        yaxis=dict(scaleanchor="x", scaleratio=1),
        margin=PLOT_MARGINS,
        showlegend=False
    )

    return fig_response, fig_feasibility, feasible_pct, feasible_mask, pair_result


def create_nor_par_pair_figure(x_coords, y_coords, feasible_mask, pair_result,
                                var1_name, var2_name, fixed_values=None):
    """
    Create a figure showing NOR, PAR, and PAIR rectangles overlaid for comparison.

    Visually compares the three QbD operating range concepts:
    - NOR (Normal Operating Range) - full experimental domain - gray dashed
    - PAR (Proven Acceptable Range) - bounding box of feasible region - orange dotted
    - PAIR (Proven Acceptable Independent Range) - largest inscribed rectangle - blue solid

    Args:
        x_coords: 1D array of x coordinates
        y_coords: 1D array of y coordinates
        feasible_mask: 2D boolean array (True = feasible)
        pair_result: dict from find_largest_feasible_rectangle()
        var1_name, var2_name: axis labels
        fixed_values: dict of fixed values for subtitle

    Returns:
        plotly Figure with NOR/PAR/PAIR rectangles and legend
    """
    fig = go.Figure()

    # Get domain bounds (NOR)
    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()

    # Calculate PAR bounds (bounding box of feasible region)
    feasible_indices = np.where(feasible_mask)
    if len(feasible_indices[0]) > 0:
        par_y_min = y_coords[feasible_indices[0].min()]
        par_y_max = y_coords[feasible_indices[0].max()]
        par_x_min = x_coords[feasible_indices[1].min()]
        par_x_max = x_coords[feasible_indices[1].max()]
        par_found = True
    else:
        par_found = False

    # ========================================================================
    # 1. NOR Rectangle (Full Domain) - Gray dashed outline
    # ========================================================================
    fig.add_shape(
        type="rect",
        x0=x_min, y0=y_min,
        x1=x_max, y1=y_max,
        line=dict(color="gray", width=3, dash="dash"),
        fillcolor="rgba(128,128,128,0.1)",
        name="NOR"
    )

    # NOR label
    fig.add_annotation(
        x=x_max, y=y_max,
        text="NOR",
        showarrow=False,
        font=dict(size=14, color="gray", family="Arial Black"),
        xanchor="right", yanchor="bottom",
        xshift=-5, yshift=5
    )

    # ========================================================================
    # 2. PAR Rectangle (Bounding box of feasible region) - Orange dashed
    # ========================================================================
    if par_found:
        fig.add_shape(
            type="rect",
            x0=par_x_min, y0=par_y_min,
            x1=par_x_max, y1=par_y_max,
            line=dict(color="orange", width=3, dash="dot"),
            fillcolor="rgba(255,165,0,0.15)",
            name="PAR"
        )

        # PAR label
        fig.add_annotation(
            x=par_x_max, y=par_y_max,
            text="PAR",
            showarrow=False,
            font=dict(size=14, color="orange", family="Arial Black"),
            xanchor="right", yanchor="bottom",
            xshift=-5, yshift=5
        )

    # ========================================================================
    # 3. PAIR Rectangle (Largest inscribed rectangle) - Blue solid
    # ========================================================================
    if pair_result['found']:
        fig.add_shape(
            type="rect",
            x0=pair_result['x_min'], y0=pair_result['y_min'],
            x1=pair_result['x_max'], y1=pair_result['y_max'],
            line=dict(color="blue", width=4, dash="solid"),
            fillcolor="rgba(0,100,255,0.25)",
            name="PAIR"
        )

        # PAIR label (centered)
        fig.add_annotation(
            x=(pair_result['x_min'] + pair_result['x_max']) / 2,
            y=(pair_result['y_min'] + pair_result['y_max']) / 2,
            text=f"<b>PAIR</b><br>{pair_result['area_percentage']:.1f}%",
            showarrow=False,
            font=dict(size=16, color="blue"),
            align="center"
        )

    # ========================================================================
    # 4. Add feasible region contour (green outline)
    # ========================================================================
    # Show boundary of feasible region
    fig.add_trace(
        go.Contour(
            x=x_coords,
            y=y_coords,
            z=feasible_mask.astype(float),
            contours=dict(
                start=0.5,
                end=0.5,
                size=1,
                coloring='none'
            ),
            line=dict(color='green', width=2),
            showscale=False,
            name='Feasible Boundary',
            hoverinfo='skip'
        )
    )

    # ========================================================================
    # 5. Layout (no legend annotations - using simple caption below instead)
    # ========================================================================
    fixed_str = ", ".join([f"{k}={v:.2f}" for k, v in fixed_values.items()]) if fixed_values else ""

    title_text = "Operating Ranges Comparison (NOR / PAR / PAIR)"
    if fixed_str:
        title_text += f"<br><sub>{fixed_str}</sub>"

    fig.update_layout(
        title=title_text,
        xaxis_title=var1_name,
        yaxis_title=var2_name,
        height=PLOT_HEIGHT,
        yaxis=dict(scaleanchor="x", scaleratio=1),
        showlegend=False,
        margin=PLOT_MARGINS
    )

    return fig


# ============================================================================
# SECTION 4B: QbD DESIGN SPACE ANALYSIS (NOR/PAR/PAIR)
# ============================================================================

def largest_rectangle_in_histogram(heights, current_row):
    """
    Find largest rectangle in histogram using stack-based O(n) algorithm.

    Based on the maximal rectangle problem using monotonic stack.

    Args:
        heights: 1D array of bar heights (consecutive feasible rows)
        current_row: current row index (for tracking position)

    Returns:
        dict with area, height, col_start, col_end, row
    """
    stack = []  # Stack of (index, height)
    max_area = 0
    best = {'area': 0, 'height': 0, 'col_start': 0, 'col_end': 0, 'row': current_row}

    for i, h in enumerate(heights):
        start = i
        while stack and stack[-1][1] > h:
            idx, height = stack.pop()
            area = height * (i - idx)
            if area > max_area:
                max_area = area
                best = {
                    'area': area,
                    'height': height,
                    'col_start': idx,
                    'col_end': i - 1,
                    'row': current_row
                }
            start = idx
        stack.append((start, h))

    # Process remaining bars in stack
    for idx, height in stack:
        area = height * (len(heights) - idx)
        if area > max_area:
            max_area = area
            best = {
                'area': area,
                'height': height,
                'col_start': idx,
                'col_end': len(heights) - 1,
                'row': current_row
            }

    return best


def find_largest_feasible_rectangle(feasibility_mask, x_coords, y_coords):
    """
    Find the largest axis-aligned rectangle where all points are feasible.

    This gives the PAIR (Proven Acceptable Independent Range) according to
    ICH Q8 QbD guidelines. PAIR represents the largest hyper-rectangle where
    each factor can vary independently while maintaining response specifications.

    Algorithm: Uses dynamic programming with histogram approach - O(rows × cols) complexity.

    Args:
        feasibility_mask: 2D boolean array (True = feasible, False = infeasible)
        x_coords: 1D array of x coordinates (var1 values)
        y_coords: 1D array of y coordinates (var2 values)

    Returns:
        dict: {
            'found': bool,
            'x_min': float, 'x_max': float,
            'y_min': float, 'y_max': float,
            'width': float, 'height': float,
            'area': float,
            'area_percentage': float,
            'grid_points': int
        }
    """
    # Convert to int matrix (1 = feasible, 0 = not)
    matrix = feasibility_mask.astype(int)
    rows, cols = matrix.shape

    if rows == 0 or cols == 0:
        return {'found': False}

    # Height array for histogram approach
    heights = np.zeros(cols, dtype=int)

    max_area = 0
    best_rect = None  # (row_start, row_end, col_start, col_end)

    for i in range(rows):
        # Update heights: if current cell is 1, add to height; else reset to 0
        for j in range(cols):
            if matrix[i, j] == 1:
                heights[j] += 1
            else:
                heights[j] = 0

        # Find largest rectangle in this histogram row
        rect_info = largest_rectangle_in_histogram(heights, i)

        if rect_info['area'] > max_area:
            max_area = rect_info['area']
            best_rect = rect_info

    if best_rect is None or max_area == 0:
        return {'found': False}

    # Convert grid indices to actual coordinates
    row_end = best_rect['row']
    row_start = row_end - best_rect['height'] + 1
    col_start = best_rect['col_start']
    col_end = best_rect['col_end']

    # Get coordinate values
    x_min = x_coords[col_start]
    x_max = x_coords[col_end]
    y_min = y_coords[row_start]
    y_max = y_coords[row_end]

    # Calculate areas
    rect_area = (x_max - x_min) * (y_max - y_min)
    total_area = (x_coords[-1] - x_coords[0]) * (y_coords[-1] - y_coords[0])

    return {
        'found': True,
        'x_min': float(x_min),
        'x_max': float(x_max),
        'y_min': float(y_min),
        'y_max': float(y_max),
        'width': float(x_max - x_min),
        'height': float(y_max - y_min),
        'area': float(rect_area),
        'area_percentage': float(rect_area / total_area * 100) if total_area > 0 else 0,
        'grid_points': int(best_rect['area'])
    }


def show_nor_par_pair_summary(var1_name, var2_name, x_range, y_range,
                               feasible_mask, x_coords, y_coords, pair_result):
    """
    Display NOR, PAR, and PAIR summary for QbD compliance (ICH Q8).

    Args:
        var1_name: Name of first variable (X-axis)
        var2_name: Name of second variable (Y-axis)
        x_range: Tuple (min, max) for var1 domain
        y_range: Tuple (min, max) for var2 domain
        feasible_mask: 2D boolean array of feasibility
        x_coords: 1D array of x grid coordinates
        y_coords: 1D array of y grid coordinates
        pair_result: Result from find_largest_feasible_rectangle()
    """
    st.markdown("---")
    st.markdown("### 📐 Operating Ranges (QbD Analysis)")

    st.info("""
    **ICH Q8 Design Space Terminology:**
    - **NOR**: Normal Operating Range (full experimental domain)
    - **PAR**: Proven Acceptable Range (depends on other factors)
    - **PAIR**: Proven Acceptable Independent Range (factors vary independently)

    **PAIR is the gold standard for process control** - any combination within PAIR is guaranteed feasible.
    """)

    # Create summary table with 3 columns
    col1, col2, col3 = st.columns(3)

    # ========================================================================
    # NOR (Full Domain)
    # ========================================================================
    with col1:
        st.markdown("#### NOR (Full Domain)")
        st.write(f"**{var1_name}**: [{x_range[0]:.3f}, {x_range[1]:.3f}]")
        st.write(f"**{var2_name}**: [{y_range[0]:.3f}, {y_range[1]:.3f}]")
        nor_area = (x_range[1] - x_range[0]) * (y_range[1] - y_range[0])
        st.metric("NOR Area", f"{nor_area:.4f}", help="100% of experimental domain")

    # ========================================================================
    # PAR (Feasible Region - bounds may depend on other factors)
    # ========================================================================
    with col2:
        st.markdown("#### PAR (Feasible Region)")
        # Calculate PAR bounds (min/max of feasible region)
        feasible_indices = np.where(feasible_mask)
        if len(feasible_indices[0]) > 0:
            y_feasible = y_coords[feasible_indices[0]]
            x_feasible = x_coords[feasible_indices[1]]
            st.write(f"**{var1_name}**: [{x_feasible.min():.3f}, {x_feasible.max():.3f}]")
            st.write(f"**{var2_name}**: [{y_feasible.min():.3f}, {y_feasible.max():.3f}]")
            feasible_pct = feasible_mask.sum() / feasible_mask.size * 100
            st.metric("Feasible Area", f"{feasible_pct:.1f}%", help="% of domain that is feasible")
            st.caption("⚠️ Ranges are **dependent** - not all combinations are feasible")
        else:
            st.warning("No feasible region found")

    # ========================================================================
    # PAIR (Independent Range - largest rectangle)
    # ========================================================================
    with col3:
        st.markdown("#### PAIR (Independent Range)")
        if pair_result['found']:
            st.write(f"**{var1_name}**: [{pair_result['x_min']:.3f}, {pair_result['x_max']:.3f}]")
            st.write(f"**{var2_name}**: [{pair_result['y_min']:.3f}, {pair_result['y_max']:.3f}]")
            st.metric("PAIR Area", f"{pair_result['area_percentage']:.1f}%",
                     help="Largest rectangle where factors vary independently")

            # Show PAIR dimensions
            st.caption(f"✅ Width ({var1_name}): {pair_result['width']:.3f}")
            st.caption(f"✅ Height ({var2_name}): {pair_result['height']:.3f}")
        else:
            st.warning("No PAIR found (no fully feasible rectangle)")
            st.caption("All feasible points are isolated or non-rectangular")

    # ========================================================================
    # Interpretation & Recommendations
    # ========================================================================
    st.markdown("---")
    st.markdown("#### 📊 QbD Interpretation")

    if pair_result['found']:
        st.success(f"""
        **✅ PAIR Found!** You can operate with independent factor control:
        - **{var1_name}** anywhere in **[{pair_result['x_min']:.3f}, {pair_result['x_max']:.3f}]**
        - **{var2_name}** anywhere in **[{pair_result['y_min']:.3f}, {pair_result['y_max']:.3f}]**

        **Key benefit**: Any combination within PAIR is guaranteed feasible (95% CI-aware).
        This simplifies process control - no need to adjust {var2_name} based on {var1_name}.
        """)

        # Show PAIR vs PAR comparison
        if len(feasible_indices[0]) > 0:
            coverage = (pair_result['area_percentage'] / (feasible_mask.sum() / feasible_mask.size * 100)) * 100
            if coverage > 70:
                st.info(f"PAIR covers {coverage:.0f}% of PAR - excellent rectangular feasibility!")
            elif coverage > 40:
                st.info(f"PAIR covers {coverage:.0f}% of PAR - good rectangular coverage")
            else:
                st.warning(f"PAIR covers only {coverage:.0f}% of PAR - feasible region has complex shape")
    else:
        st.warning("""
        **⚠️ No PAIR found.** The feasible region has no rectangular subset.

        **Options**:
        1. **Use PAR** with factor dependencies (more complex process control)
        2. **Relax target specifications** to increase feasible region
        3. **Reduce CI tolerance** (improve model precision or increase replicates)
        4. **Add constraints** to reshape feasible region into more rectangular form
        """)

        # Suggest which variable has more restrictive range
        if len(feasible_indices[0]) > 0:
            x_feasible_range = x_feasible.max() - x_feasible.min()
            y_feasible_range = y_feasible.max() - y_feasible.min()
            x_nor_range = x_range[1] - x_range[0]
            y_nor_range = y_range[1] - y_range[0]

            x_restriction = 100 * (1 - x_feasible_range / x_nor_range)
            y_restriction = 100 * (1 - y_feasible_range / y_nor_range)

            if x_restriction > y_restriction:
                st.caption(f"💡 **{var1_name}** is more restricted ({x_restriction:.0f}% reduction) - focus optimization there")
            else:
                st.caption(f"💡 **{var2_name}** is more restricted ({y_restriction:.0f}% reduction) - focus optimization there")


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
            # Maximize/Minimize: no acceptability bounds, 100% feasible
            # The surface just shows conservative estimate
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
                **Feasible region: {feasible_pct:.1f}%** of experimental domain
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
                **Feasible region: {feasible_pct:.1f}%** of experimental domain
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
            # Use new format (target_min, target_max)
            target_min = criteria.get('target_min')
            target_max = criteria.get('target_max')

            if target_min is not None and target_max is not None:
                # Calculate feasibility
                z_pred = surface['response_grid']
                ci_grid = surface['ci_grid']
                lower_bound = z_pred - ci_grid
                upper_bound = z_pred + ci_grid
                feasible_mask = (lower_bound >= target_min) & (upper_bound <= target_max)
                feasible_pct = (feasible_mask.sum() / feasible_mask.size) * 100

                target_center = (target_min + target_max) / 2

                # Find point closest to target center in feasible region
                if feasible_pct > 0:
                    distance_to_center = np.abs(z_pred - target_center)
                    distance_to_center[~feasible_mask] = np.inf  # Exclude infeasible points
                    best_idx = np.argmin(distance_to_center)
                    best_i, best_j = np.unravel_index(best_idx, z_pred.shape)

                    best_val = z_pred[best_i, best_j]
                    var1_val = surface['x_grid'][best_i, best_j]
                    var2_val = surface['y_grid'][best_i, best_j]

                    st.success(f"""
                    **Feasible region: {feasible_pct:.1f}%** of experimental domain
                    - Target range: [{target_min:.4f}, {target_max:.4f}]
                    - Best feasible point: {best_val:.4f} at {config['var1']}={var1_val:.3f}, {config['var2']}={var2_val:.3f}
                    - ✅ CI-aware feasibility ensured
                    """)
                else:
                    st.error(f"""
                    **No feasible region found**
                    - Target range: [{target_min:.4f}, {target_max:.4f}]
                    - No points where entire CI fits within target range
                    - ⚠️ Consider widening target range or improving model precision
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

        # Display in 2-column grid - each response gets its own column
        for i in range(0, len(y_vars_list), 2):
            col1, col2 = st.columns(2)

            # First response (column 1)
            with col1:
                y_var = y_vars_list[i]
                opt_type = stored_criteria[y_var]['optimization']

                if opt_type == "Target":
                    # Target mode: Response surface + Feasibility map + NOR/PAR/PAIR + Summary
                    fig_response, fig_feasibility, feasible_pct, feasible_mask, pair_result = create_target_feasibility_plots(
                        surfaces_dict[y_var],
                        stored_config['var1'],
                        stored_config['var2'],
                        y_var,
                        stored_config['fixed_values'],
                        stored_criteria[y_var]
                    )

                    # Display response surface
                    st.plotly_chart(fig_response, use_container_width=True)

                    # Stats below response surface
                    col_s1, col_s2 = st.columns(2)
                    with col_s1:
                        st.metric("Target Min", f"{stored_criteria[y_var]['target_min']:.4f}")
                    with col_s2:
                        st.metric("Target Max", f"{stored_criteria[y_var]['target_max']:.4f}")

                    # Display feasibility map
                    st.plotly_chart(fig_feasibility, use_container_width=True)

                    # Feasibility stats
                    if feasible_pct > 50:
                        st.success(f"**Feasible Area: {feasible_pct:.1f}%**")
                    elif feasible_pct > 10:
                        st.warning(f"**Feasible Area: {feasible_pct:.1f}%**")
                    else:
                        st.error(f"**Feasible Area: {feasible_pct:.1f}%**")

                    # Interpretation
                    st.info(f"""
                    **Interpretation**:
                    - **Green areas**: CI within [{stored_criteria[y_var]['target_min']:.2f}, {stored_criteria[y_var]['target_max']:.2f}]
                    - **Red areas**: CI extends outside target range
                    """)

                else:
                    # Standard mode: Just the contour plot
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
                    col_s1, col_s2 = st.columns(2)
                    with col_s1:
                        st.metric(f"{y_var} Min", f"{surfaces_dict[y_var]['bounds']['min']:.4f}")
                    with col_s2:
                        st.metric(f"{y_var} Max", f"{surfaces_dict[y_var]['bounds']['max']:.4f}")

            # Second response (column 2) - if exists
            if i + 1 < len(y_vars_list):
                with col2:
                    y_var = y_vars_list[i + 1]
                    opt_type = stored_criteria[y_var]['optimization']

                    if opt_type == "Target":
                        # Target mode: Response surface + Feasibility map + NOR/PAR/PAIR + Summary
                        fig_response, fig_feasibility, feasible_pct, feasible_mask, pair_result = create_target_feasibility_plots(
                            surfaces_dict[y_var],
                            stored_config['var1'],
                            stored_config['var2'],
                            y_var,
                            stored_config['fixed_values'],
                            stored_criteria[y_var]
                        )

                        # Display response surface
                        st.plotly_chart(fig_response, use_container_width=True)

                        # Stats below response surface
                        col_s1, col_s2 = st.columns(2)
                        with col_s1:
                            st.metric("Target Min", f"{stored_criteria[y_var]['target_min']:.4f}")
                        with col_s2:
                            st.metric("Target Max", f"{stored_criteria[y_var]['target_max']:.4f}")

                        # Display feasibility map
                        st.plotly_chart(fig_feasibility, use_container_width=True)

                        # Feasibility stats
                        if feasible_pct > 50:
                            st.success(f"**Feasible Area: {feasible_pct:.1f}%**")
                        elif feasible_pct > 10:
                            st.warning(f"**Feasible Area: {feasible_pct:.1f}%**")
                        else:
                            st.error(f"**Feasible Area: {feasible_pct:.1f}%**")

                        # Interpretation
                        st.info(f"""
                        **Interpretation**:
                        - **Green areas**: CI within [{stored_criteria[y_var]['target_min']:.2f}, {stored_criteria[y_var]['target_max']:.2f}]
                        - **Red areas**: CI extends outside target range
                        """)

                    else:
                        # Standard mode: Just the contour plot
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
                        col_s1, col_s2 = st.columns(2)
                        with col_s1:
                            st.metric(f"{y_var} Min", f"{surfaces_dict[y_var]['bounds']['min']:.4f}")
                        with col_s2:
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
