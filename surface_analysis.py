"""
Surface Analysis - Unified Response Surface & Confidence Interval Visualization
Combines functionality from DOE_response_surface.r and DOE_CI_surface.r
Creates comparative visualizations of response surface and prediction uncertainty
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats


# ============================================================================
# SECTION 1: UTILITY FUNCTIONS
# ============================================================================

def create_prediction_matrix(X, x_vars, coef_names):
    """
    Create model matrix matching the coefficient structure

    Args:
        X: numpy array of base variables (n_samples × n_vars)
        x_vars: list of variable names
        coef_names: list of coefficient names from model

    Returns:
        numpy array of model matrix
    """
    n_samples, n_vars = X.shape
    X_model_list = []

    for coef_name in coef_names:
        if coef_name == 'Intercept':
            X_model_list.append(np.ones(n_samples))
        elif '*' in coef_name:
            # Interaction term
            vars_in_term = coef_name.split('*')
            idx1 = x_vars.index(vars_in_term[0])
            idx2 = x_vars.index(vars_in_term[1])
            X_model_list.append(X[:, idx1] * X[:, idx2])
        elif '^2' in coef_name:
            # Quadratic term
            var_name = coef_name.replace('^2', '')
            idx = x_vars.index(var_name)
            X_model_list.append(X[:, idx] ** 2)
        else:
            # Linear term
            idx = x_vars.index(coef_name)
            X_model_list.append(X[:, idx])

    X_model = np.column_stack(X_model_list)
    return X_model


# ============================================================================
# SECTION 2: UNIFIED CONTROL PANEL
# ============================================================================

def show_unified_control_panel(model_results, x_vars, y_var):
    """
    Unified control panel for surface analysis configuration

    Clearly separates:
    1. Model parameters (for prediction CI)
    2. CI type selection
    3. Experimental parameters (only if experimental CI selected)

    Returns:
        dict with configuration parameters
    """

    # ========================================================================
    # SECTION A: VARIABLE SELECTION
    # ========================================================================
    st.markdown("### Variable Selection")

    col1, col2 = st.columns(2)

    with col1:
        var1 = st.selectbox(
            "Variable for X-axis:",
            x_vars,
            key="surface_analysis_var1",
            help="First variable to plot"
        )

    with col2:
        var2 = st.selectbox(
            "Variable for Y-axis:",
            [v for v in x_vars if v != var1],
            key="surface_analysis_var2",
            help="Second variable to plot"
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
                    key=f"surface_fixed_val_{var}",
                    help=f"Fixed value for {var}"
                )

    st.markdown("---")

    # ========================================================================
    # SECTION B: CI TYPE SELECTION (moved up for clarity)
    # ========================================================================
    st.markdown("### Confidence Interval Type")
    st.info("""
    Choose what type of confidence interval to calculate:
    - **Prediction**: Shows model confidence in predictions (model uncertainty only)
    - **Experimental**: Combines model + measurement uncertainty (error propagation)
    """)

    ci_type = st.radio(
        "Select CI calculation mode:",
        ["Prediction (Model Uncertainty Only)",
         "Experimental (Model + Measurement Uncertainty)"],
        key="surface_ci_type",
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

    variance_method_model = st.radio(
        "Model variance estimated from:",
        ["Model residuals (from fitting)", "Independent measurement"],
        key="surface_model_variance_source",
        help="Source of variance for the model"
    )

    if variance_method_model == "Model residuals (from fitting)":
        s_model = model_results.get('rmse')
        dof_model = model_results.get('dof')

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
                value=model_results.get('rmse', 1.0),
                min_value=0.0001,
                format="%.6f",
                step=0.001,
                key="surface_s_model_independent",
                help="Std dev for model predictions"
            )

        with col_model_ind2:
            dof_model = st.number_input(
                "Model DOF:",
                value=model_results.get('dof', 5),
                min_value=1,
                step=1,
                key="surface_dof_model_independent",
                help="Degrees of freedom"
            )

    st.markdown("---")

    # ========================================================================
    # SECTION D: EXPERIMENTAL PARAMETERS (CONDITIONAL)
    # Only shown if ci_type == "Experimental"
    # ========================================================================

    s_exp = None
    dof_exp = None
    n_replicates = 1
    variance_method_exp = None

    if ci_type == "Experimental (Model + Measurement Uncertainty)":
        st.markdown("### Experimental Measurement Parameters")
        st.info("""
        These parameters define the **experimental measurement uncertainty**.

        **Formula**: CI_total = √((CI_model)² + (CI_experimental)²)

        where:
        - CI_model = t_model × s_model × √(leverage)
        - CI_experimental = t_exp × s_exp × √(1/n_replicates)
        """)

        # Number of replicates
        col_exp_rep, col_exp_space = st.columns([2, 1])

        with col_exp_rep:
            n_replicates = st.number_input(
                "Number of replicate measurements (n):",
                min_value=1,
                max_value=100,
                value=1,
                step=1,
                key="surface_n_replicates",
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
            key="surface_exp_variance_source",
            help="Source of variance for experimental measurements"
        )

        if variance_method_exp == "Model residuals (from fitting)":
            s_exp = model_results.get('rmse')
            dof_exp = model_results.get('dof')

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
                    key="surface_s_exp_independent",
                    help="Measurement variability (from replicate experiments)"
                )

            with col_exp_i2:
                dof_exp = st.number_input(
                    "Experimental DOF:",
                    value=5,
                    min_value=1,
                    step=1,
                    key="surface_dof_exp_independent",
                    help="Degrees of freedom from experimental replicates"
                )

    st.markdown("---")

    # ========================================================================
    # SECTION E: SURFACE RANGE SETTINGS
    # ========================================================================
    st.markdown("### Surface Range Configuration")

    # Auto-detect design space from data
    var1_data = st.session_state.get('X_data', pd.DataFrame()).get(var1)
    var2_data = st.session_state.get('X_data', pd.DataFrame()).get(var2)

    if var1_data is not None and var2_data is not None and len(var1_data) > 0:
        var1_min = var1_data.min()
        var1_max = var1_data.max()
        var2_min = var2_data.min()
        var2_max = var2_data.max()

        margin = 0.1
        design_min = min(var1_min, var2_min) - margin
        design_max = max(var1_max, var2_max) + margin
    else:
        design_min = -1.0
        design_max = 1.0

    col_range1, col_range2, col_range3 = st.columns(3)

    with col_range1:
        min_range = st.number_input(
            "Minimum value:",
            value=design_min,
            min_value=design_min - 0.5,
            max_value=design_max - 0.1,
            step=0.1,
            format="%.2f",
            key="surface_min_range"
        )

    with col_range2:
        max_range = st.number_input(
            "Maximum value:",
            value=design_max,
            min_value=design_min + 0.1,
            max_value=design_max + 0.5,
            step=0.1,
            format="%.2f",
            key="surface_max_range"
        )

    with col_range3:
        n_steps = st.number_input(
            "Grid resolution:",
            min_value=10,
            max_value=100,
            value=30,
            step=5,
            key="surface_n_steps",
            help="Number of grid points (higher = smoother but slower)"
        )

    if min_range < design_min or max_range > design_max:
        st.warning("⚠️ You are extrapolating beyond the design space (predictions may be unreliable)")

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

        # Model parameters (always present)
        's_model': s_model,
        'dof_model': dof_model,
        'variance_method_model': variance_method_model,

        # CI type
        'ci_type': ci_type,

        # Experimental parameters (only if experimental mode)
        's_exp': s_exp,
        'dof_exp': dof_exp,
        'n_replicates': n_replicates,
        'variance_method_exp': variance_method_exp,

        # Range parameters
        'n_steps': n_steps,
        'min_range': min_range,
        'max_range': max_range
    }


# ============================================================================
# SECTION 3: CALCULATION FUNCTIONS
# ============================================================================

def calculate_response_surface(model_results, x_vars, v1_idx, v2_idx,
                               fixed_values, n_steps, value_range):
    """
    Calculate response surface for two variables while holding others constant

    Args:
        model_results: dict from fit_mlr_model
        x_vars: list of X variable names
        v1_idx: index of first variable for surface
        v2_idx: index of second variable for surface
        fixed_values: dict with fixed values for other variables
        n_steps: number of grid points
        value_range: tuple (min, max) for the range

    Returns:
        tuple: (x_grid, y_grid, z_grid, grid_df)
    """
    n_vars = len(x_vars)
    coefficients = model_results['coefficients']
    coef_names = coefficients.index.tolist()

    # Create grid
    min_val, max_val = value_range
    step = (max_val - min_val) / n_steps
    grid_1d = np.arange(min_val, max_val + step, step)

    x_grid, y_grid = np.meshgrid(grid_1d, grid_1d)

    n_points = x_grid.size
    x_flat = x_grid.flatten()
    y_flat = y_grid.flatten()

    # Build full X matrix
    X_grid = np.zeros((n_points, n_vars))

    for i, var in enumerate(x_vars):
        X_grid[:, i] = fixed_values.get(var, 0)

    X_grid[:, v1_idx] = x_flat
    X_grid[:, v2_idx] = y_flat

    # Create model matrix
    X_model = create_prediction_matrix(X_grid, x_vars, coef_names)

    # Calculate predictions
    z_flat = X_model @ coefficients.values
    z_grid = z_flat.reshape(x_grid.shape)

    grid_df = pd.DataFrame({
        'x': x_flat,
        'y': y_flat,
        'z': z_flat
    })

    return x_grid, y_grid, z_grid, grid_df


def calculate_ci_surface(model_results, x_vars, v1_idx, v2_idx,
                         fixed_values, s, dof, n_steps, value_range):
    """
    Calculate confidence interval semiamplitude surface

    Args:
        model_results: dict from fit_mlr_model
        x_vars: list of X variable names
        v1_idx, v2_idx: indices of variables for surface
        fixed_values: dict with fixed values for other variables
        s: experimental standard deviation
        dof: degrees of freedom
        n_steps: grid resolution
        value_range: (min, max) for the range

    Returns:
        tuple: (x_grid, y_grid, ci_grid, grid_df)
    """
    n_vars = len(x_vars)
    coefficients = model_results['coefficients']
    dispersion = model_results['XtX_inv']
    coef_names = coefficients.index.tolist()

    if dof <= 0:
        raise ValueError("Degrees of freedom must be > 0 for confidence intervals")

    # Create grid
    min_val, max_val = value_range
    step = (max_val - min_val) / n_steps
    grid_1d = np.arange(min_val, max_val + step, step)

    x_grid, y_grid = np.meshgrid(grid_1d, grid_1d)

    n_points = x_grid.size
    x_flat = x_grid.flatten()
    y_flat = y_grid.flatten()

    # Build X matrix
    X_grid = np.zeros((n_points, n_vars))

    for i, var in enumerate(x_vars):
        X_grid[:, i] = fixed_values.get(var, 0)

    X_grid[:, v1_idx] = x_flat
    X_grid[:, v2_idx] = y_flat

    # Create model matrix
    X_model = create_prediction_matrix(X_grid, x_vars, coef_names)

    # Calculate leverage and CI
    t_critical = stats.t.ppf(0.975, dof)  # 95% confidence interval
    leverage = np.diag(X_model @ dispersion @ X_model.T)
    ci_flat = t_critical * s * np.sqrt(leverage)
    ci_grid = ci_flat.reshape(x_grid.shape)

    grid_df = pd.DataFrame({
        'x': x_flat,
        'y': y_flat,
        'ci': ci_flat,
        'leverage': leverage
    })

    return x_grid, y_grid, ci_grid, grid_df


def calculate_ci_experimental_surface(model_results, x_vars, v1_idx, v2_idx,
                                      fixed_values=None,
                                      s_model=None, dof_model=None,
                                      s_exp=None, dof_exp=None, n_replicates=1,
                                      n_steps=30, value_range=(-1, 1)):
    """
    Calculate experimental CI semiamplitude surface with uncertainty propagation

    Formula (from DOE_CI_surface_exp.r line 121):
        CI_exp = √((CI_model)² + (CI_experimental)²)

        where:
        CI_model = t_model * s_model * √(leverage)
        CI_experimental = t_exp * s_exp * √(1/n_replicates)

    Args:
        model_results: dict from fit_mlr_model
        x_vars: list of X variable names
        v1_idx, v2_idx: indices of variables for surface
        fixed_values: dict with fixed values for other variables
        s_model: model prediction standard deviation (model RMSE)
        dof_model: model degrees of freedom
        s_exp: experimental measurement standard deviation
        dof_exp: experimental degrees of freedom
        n_replicates: number of replicate measurements (1, 2, N, ...)
        n_steps: grid resolution
        value_range: (min, max) for the range

    Returns:
        tuple: (x_grid, y_grid, ci_model_grid, ci_exp_component, ci_total_grid, grid_df)
    """
    n_vars = len(x_vars)
    coefficients = model_results['coefficients']
    dispersion = model_results['XtX_inv']

    # Use model values if not provided
    if s_model is None:
        s_model = model_results.get('rmse', 1.0)
    if dof_model is None:
        dof_model = model_results.get('dof', 1)
    if s_exp is None:
        s_exp = s_model
    if dof_exp is None:
        dof_exp = dof_model

    if dof_model <= 0 or dof_exp <= 0:
        raise ValueError("Degrees of freedom must be > 0")

    # Fixed values
    if fixed_values is None:
        fixed_values = {var: 0 for var in x_vars}

    # Create grid
    min_val, max_val = value_range
    step = (max_val - min_val) / n_steps
    grid_1d = np.arange(min_val, max_val + step, step)

    x_grid, y_grid = np.meshgrid(grid_1d, grid_1d)
    n_points = x_grid.size
    x_flat = x_grid.flatten()
    y_flat = y_grid.flatten()

    # Build X matrix for grid
    X_grid = np.zeros((n_points, n_vars))
    for i, var in enumerate(x_vars):
        X_grid[:, i] = fixed_values.get(var, 0)
    X_grid[:, v1_idx] = x_flat
    X_grid[:, v2_idx] = y_flat

    # Create model matrix
    coef_names = coefficients.index.tolist()
    X_model = create_prediction_matrix(X_grid, x_vars, coef_names)

    # Calculate t-critical values (95% CI, two-sided)
    t_model = stats.t.ppf(0.975, dof_model)
    t_exp = stats.t.ppf(0.975, dof_exp)

    # Calculate leverage for model prediction CI
    leverage = np.diag(X_model @ dispersion @ X_model.T)

    # COMPONENT 1: Model Prediction CI
    # CI_model = t_model * s_model * √(leverage)
    ci_model_flat = t_model * s_model * np.sqrt(leverage)
    ci_model_grid = ci_model_flat.reshape(x_grid.shape)

    # COMPONENT 2: Experimental Measurement CI
    # CI_exp = t_exp * s_exp * √(1/n_replicates)
    # This is constant across the surface (doesn't depend on grid position)
    ci_exp_component = t_exp * s_exp * np.sqrt(1.0 / n_replicates)

    # TOTAL: Combine using error propagation
    # CI_total = √((CI_model)² + (CI_exp)²)
    ci_total_flat = np.sqrt(ci_model_flat**2 + ci_exp_component**2)
    ci_total_grid = ci_total_flat.reshape(x_grid.shape)

    grid_df = pd.DataFrame({
        'x': x_flat,
        'y': y_flat,
        'ci_model': ci_model_flat,
        'ci_exp': ci_exp_component,
        'ci_total': ci_total_flat,
        'leverage': leverage
    })

    return x_grid, y_grid, ci_model_grid, ci_exp_component, ci_total_grid, grid_df


# ============================================================================
# SECTION 4: PLOTTING FUNCTIONS
# ============================================================================

def plot_contour_comparison_2x2(x_grid, y_grid, z_grid_response, z_grid_ci_pred=None,
                                z_grid_ci_total=None, var1=None, var2=None, y_var=None,
                                fixed_values=None, ci_mode="prediction", title_suffix="", threshold_line=None):
    """
    Create separate contour plots for 2x2 grid layout

    Returns 2 or 3 figures depending on mode:
    - Prediction mode: (fig_response, fig_ci_pred)
    - Experimental mode: (fig_response, fig_ci_pred, fig_ci_total)

    Args:
        x_grid, y_grid: meshgrid arrays
        z_grid_response: response surface grid
        z_grid_ci_pred: prediction/model CI grid
        z_grid_ci_total: total CI grid (experimental mode only)
        var1, var2: variable names
        y_var: response variable name
        fixed_values: dict of fixed values for subtitle
        ci_mode: "prediction" or "experimental"
        title_suffix: additional text for response surface title
        threshold_line: dict with 'value' and 'label' to draw threshold line on response contour
                       Example: {'value': 70, 'label': 'Threshold: 70%'}

    Returns:
        tuple: (fig_response, fig_ci) or (fig_response, fig_ci_model, fig_ci_total)
    """
    # Build subtitle
    subtitle = ""
    if fixed_values:
        fixed_text = [f"{var}={val:.2f}" for var, val in fixed_values.items()]
        subtitle = ", ".join(fixed_text)

    # Figure 1: Response Surface (same for both modes)
    fig_response = go.Figure(data=[
        go.Contour(
            x=x_grid[0, :],
            y=y_grid[:, 0],
            z=z_grid_response,
            colorscale='Viridis',
            contours=dict(
                coloring='heatmap',
                showlabels=True,
                labelfont=dict(size=10, color='white')
            ),
            colorbar=dict(title=y_var),
            hovertemplate=f'{var1}: %{{x:.3f}}<br>{var2}: %{{y:.3f}}<br>{y_var}: %{{z:.3f}}<extra></extra>',
            ncontours=15
        )
    ])

    title_text = f"Response Surface - {y_var}{title_suffix}"
    if subtitle:
        title_text += f"<br><sub>{subtitle}</sub>"

    fig_response.update_layout(
        title=title_text,
        xaxis_title=var1,
        yaxis_title=var2,
        height=450,
        margin=dict(l=50, r=50, t=80, b=50),
        yaxis=dict(scaleanchor="x", scaleratio=1)
    )

    # Add threshold contour line if specified
    if threshold_line is not None:
        # Add a red contour line at the threshold Z value
        fig_response.add_trace(go.Contour(
            x=x_grid[0, :],
            y=y_grid[:, 0],
            z=z_grid_response,
            contours=dict(
                start=threshold_line['value'],
                end=threshold_line['value'],
                size=1,  # Single contour line
                coloring='none',  # No fill, just the line
                showlabels=True,
                labelfont=dict(size=12, color='red')
            ),
            line=dict(color='red', width=4),
            showscale=False,
            name=threshold_line['label'],
            hovertemplate=f"{threshold_line['label']}: {threshold_line['value']:.2f}<extra></extra>"
        ))

    # Figure 2: CI (Prediction or Model)
    if ci_mode == "prediction":
        ci_data = z_grid_ci_pred
        colorscale = 'YlGnBu'
        title_suffix = "Prediction CI"
    else:
        ci_data = z_grid_ci_pred
        colorscale = 'Oranges'
        title_suffix = "Model CI (Prediction component)"

    fig_ci = go.Figure(data=[
        go.Contour(
            x=x_grid[0, :],
            y=y_grid[:, 0],
            z=ci_data,
            colorscale=colorscale,
            contours=dict(
                coloring='heatmap',
                showlabels=True,
                labelfont=dict(size=10, color='white')
            ),
            colorbar=dict(title='CI'),
            hovertemplate=f'{var1}: %{{x:.3f}}<br>{var2}: %{{y:.3f}}<br>CI: %{{z:.4f}}<extra></extra>',
            ncontours=15
        )
    ])

    title_text_ci = f"{title_suffix} - {y_var}"
    if subtitle:
        title_text_ci += f"<br><sub>{subtitle}</sub>"

    fig_ci.update_layout(
        title=title_text_ci,
        xaxis_title=var1,
        yaxis_title=var2,
        height=450,
        margin=dict(l=50, r=50, t=80, b=50),
        yaxis=dict(scaleanchor="x", scaleratio=1)
    )

    # Figure 3: Total CI (only for experimental mode)
    if ci_mode == "experimental" and z_grid_ci_total is not None:
        fig_ci_total = go.Figure(data=[
            go.Contour(
                x=x_grid[0, :],
                y=y_grid[:, 0],
                z=z_grid_ci_total,
                colorscale='Reds',
                contours=dict(
                    coloring='heatmap',
                    showlabels=True,
                    labelfont=dict(size=10, color='white')
                ),
                colorbar=dict(title='Total CI'),
                hovertemplate=f'{var1}: %{{x:.3f}}<br>{var2}: %{{y:.3f}}<br>Total CI: %{{z:.4f}}<extra></extra>',
                ncontours=15
            )
        ])

        title_text_total = f"Total CI (Propagated) - {y_var}"
        if subtitle:
            title_text_total += f"<br><sub>{subtitle}</sub>"

        fig_ci_total.update_layout(
            title=title_text_total,
            xaxis_title=var1,
            yaxis_title=var2,
            height=450,
            margin=dict(l=50, r=50, t=80, b=50),
            yaxis=dict(scaleanchor="x", scaleratio=1)
        )

        return fig_response, fig_ci, fig_ci_total

    return fig_response, fig_ci


def plot_response_surface_3d(x_grid, y_grid, z_grid, var1, var2, y_var, fixed_values, title_suffix=""):
    """
    Create 3D response surface plot

    Returns:
        plotly Figure
    """
    fig = go.Figure(data=[
        go.Surface(
            x=x_grid,
            y=y_grid,
            z=z_grid,
            colorscale='Viridis',
            colorbar=dict(title=y_var),
            hovertemplate=f'{var1}: %{{x:.3f}}<br>{var2}: %{{y:.3f}}<br>{y_var}: %{{z:.3f}}<extra></extra>'
        )
    ])

    subtitle = ""
    if fixed_values:
        fixed_text = [f"{var}={val:.2f}" for var, val in fixed_values.items()]
        subtitle = ", ".join(fixed_text)

    title_text = f"Response Surface - {y_var}{title_suffix}"
    if subtitle:
        title_text += f"<br><sub>{subtitle}</sub>"

    fig.update_layout(
        title=title_text,
        scene=dict(
            xaxis_title=var1,
            yaxis_title=var2,
            zaxis_title=y_var,
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.3))
        ),
        height=500,
        margin=dict(l=50, r=50, t=60, b=60)
    )

    return fig


def plot_ci_surface_3d(x_grid, y_grid, ci_grid, var1, var2, y_var, fixed_values):
    """
    Create 3D CI semiamplitude surface plot

    Returns:
        plotly Figure
    """
    fig = go.Figure(data=[
        go.Surface(
            x=x_grid,
            y=y_grid,
            z=ci_grid,
            colorscale='YlGnBu',
            colorbar=dict(title='CI Semiampl.'),
            hovertemplate=f'{var1}: %{{x:.3f}}<br>{var2}: %{{y:.3f}}<br>CI: %{{z:.4f}}<extra></extra>'
        )
    ])

    subtitle = ""
    if fixed_values:
        fixed_text = [f"{var}={val:.2f}" for var, val in fixed_values.items()]
        subtitle = ", ".join(fixed_text)

    title_text = f"CI Semiamplitude - {y_var}"
    if subtitle:
        title_text += f"<br><sub>{subtitle}</sub>"

    fig.update_layout(
        title=title_text,
        scene=dict(
            xaxis_title=var1,
            yaxis_title=var2,
            zaxis_title='CI Semiamplitude',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.3))
        ),
        height=500,
        margin=dict(l=50, r=50, t=60, b=60)
    )

    return fig


# ============================================================================
# SECTION 5: MAIN UI FUNCTION
# ============================================================================

def show_surface_analysis_ui(model_results, x_vars, y_var):
    """
    Streamlit UI for unified surface analysis

    Args:
        model_results: dict from fit_mlr_model
        x_vars: list of X variable names
        y_var: Y variable name
    """
    st.markdown("## 🎨 Surface Analysis")
    st.markdown("*Unified Response Surface & Confidence Interval Analysis*")

    if model_results is None:
        st.warning("⚠️ No MLR model fitted. Please fit a model first.")
        return

    st.info("""
    **Comprehensive surface analysis** combining response surface and confidence intervals.

    **How to interpret CI:**
    - **Maximizing?** Focus on LOWER CI boundary (z - CI) for conservative estimate
    - **Minimizing?** Focus on UPPER CI boundary (z + CI) for conservative estimate
    - **Targeting?** Ensure response stays within z ± CI to hit target reliably

    Lower CI values indicate prediction confidence (narrower uncertainty band).
    """)

    # ========================================================================
    # OPTIMIZATION OBJECTIVE SECTION (OPTIONAL)
    # ========================================================================
    # Optional objective selection
    with st.expander("🎯 Optimization Objective (optional)", expanded=False):
        st.markdown("""
        **Choose if you want guided recommendations. Leave closed for standard analysis.**

        **Maximize/Minimize** vs **Threshold**:
        - **Maximize/Minimize**: You want the best possible value (with conservative estimates)
        - **Threshold**: You have a hard constraint (e.g., "response MUST be > 70%")
        """)

        col_obj1, col_obj2, col_obj3 = st.columns([2, 1, 1])

        with col_obj1:
            optimization_objective = st.radio(
                "Select optimization goal:",
                [
                    "None - Standard Analysis",
                    "Maximize Response",
                    "Minimize Response",
                    "Target Value",
                    "Threshold: Must be ABOVE (soglia minima)",
                    "Threshold: Must be BELOW (soglia massima)"
                ],
                horizontal=False,
                key="surface_optimization_objective",
                help="""
                Leave as 'None' for standard surface analysis without recommendations.

                Threshold options automatically show CONSERVATIVE bounds:
                - Must be ABOVE → shows z - CI (worst-case lower bound)
                - Must be BELOW → shows z + CI (worst-case upper bound)
                """
            )

        # Conditional: Target value input
        target_value = None
        target_tolerance = None
        threshold_value = None

        if optimization_objective == "Target Value":
            with col_obj2:
                target_value = st.number_input(
                    "Target value:",
                    value=0.0,
                    step=0.1,
                    format="%.2f",
                    key="surface_target_value",
                    help="Desired response value"
                )

            with col_obj3:
                target_tolerance = st.number_input(
                    "Tolerance (±):",
                    value=0.1,
                    min_value=0.001,
                    step=0.1,
                    format="%.2f",
                    key="surface_target_tolerance",
                    help="Acceptable deviation from target"
                )

        elif "Threshold" in optimization_objective:
            with col_obj2:
                threshold_value = st.number_input(
                    "Threshold value:",
                    value=0.0,
                    step=0.1,
                    format="%.2f",
                    key="surface_threshold_value",
                    help="Minimum/maximum acceptable response value"
                )

            with col_obj3:
                if "ABOVE" in optimization_objective:
                    st.caption("Surface shows: z - CI")
                    st.caption("(Conservative lower bound)")
                else:
                    st.caption("Surface shows: z + CI")
                    st.caption("(Conservative upper bound)")

    # If expander was not opened, set defaults
    if 'surface_optimization_objective' not in st.session_state:
        optimization_objective = "None - Standard Analysis"
        target_value = None
        target_tolerance = None
        threshold_value = None

    st.markdown("---")

    # Store in session for use in interpretation
    st.session_state.optimization_objective = optimization_objective
    st.session_state.target_value = target_value
    st.session_state.target_tolerance = target_tolerance
    st.session_state.threshold_value = threshold_value

    # Show unified control panel
    controls = show_unified_control_panel(model_results, x_vars, y_var)

    if controls is None:
        return  # Invalid variance parameters

    # Generate analysis button
    if st.button("🚀 Generate Surface Analysis", type="primary"):
        try:
            with st.spinner("Calculating response surface and confidence intervals..."):
                # Get variable indices
                v1_idx = x_vars.index(controls['var1'])
                v2_idx = x_vars.index(controls['var2'])
                value_range = (controls['min_range'], controls['max_range'])

                # Calculate response surface
                x_grid_rs, y_grid_rs, z_grid_rs, grid_df_rs = calculate_response_surface(
                    model_results=model_results,
                    x_vars=x_vars,
                    v1_idx=v1_idx,
                    v2_idx=v2_idx,
                    fixed_values=controls['fixed_values'],
                    n_steps=controls['n_steps'],
                    value_range=value_range
                )

                # Calculate CI surface based on type
                if controls['ci_type'] == "Prediction (Model Uncertainty Only)":
                    # Simple prediction CI (model uncertainty only)
                    x_grid_ci, y_grid_ci, z_grid_ci, grid_df_ci = calculate_ci_surface(
                        model_results=model_results,
                        x_vars=x_vars,
                        v1_idx=v1_idx,
                        v2_idx=v2_idx,
                        fixed_values=controls['fixed_values'],
                        s=controls['s_model'],
                        dof=controls['dof_model'],
                        n_steps=controls['n_steps'],
                        value_range=value_range
                    )
                    ci_mode = "prediction"
                    z_grid_ci_model = None
                    ci_exp_component = None

                else:  # Experimental (Model + Measurement Uncertainty)
                    # Experimental CI with uncertainty propagation
                    x_grid_ci, y_grid_ci, z_grid_ci_model, ci_exp_component, z_grid_ci, grid_df_ci = calculate_ci_experimental_surface(
                        model_results=model_results,
                        x_vars=x_vars,
                        v1_idx=v1_idx,
                        v2_idx=v2_idx,
                        fixed_values=controls['fixed_values'],
                        s_model=controls['s_model'],
                        dof_model=controls['dof_model'],
                        s_exp=controls['s_exp'],
                        dof_exp=controls['dof_exp'],
                        n_replicates=controls['n_replicates'],
                        n_steps=controls['n_steps'],
                        value_range=value_range
                    )
                    ci_mode = "experimental"

                # Verify grids match
                assert x_grid_rs.shape == x_grid_ci.shape, "Grid mismatch between response and CI surfaces"

                # ================================================================
                # STEP 2: Conservative Surface Transformation (based on objective)
                # ================================================================
                # Create z_grid_display that may be transformed based on optimization objective
                z_grid_display = z_grid_rs.copy()  # Default: show nominal response
                surface_title_suffix = ""
                surface_explanation = ""

                # Get optimization objective from session state
                opt_obj = st.session_state.get('optimization_objective', 'None - Standard Analysis')

                if opt_obj == "Maximize Response":
                    # Maximize: show conservative lower bound (z - CI)
                    z_grid_display = z_grid_rs - z_grid_ci
                    surface_title_suffix = " (Conservative: z - CI)"
                    surface_explanation = "💡 Surface shows **conservative lower bound** (z - CI) for maximization"

                elif opt_obj == "Minimize Response":
                    # Minimize: show conservative upper bound (z + CI)
                    z_grid_display = z_grid_rs + z_grid_ci
                    surface_title_suffix = " (Conservative: z + CI)"
                    surface_explanation = "💡 Surface shows **conservative upper bound** (z + CI) for minimization"

                elif "Threshold: Must be ABOVE" in opt_obj:
                    # Above threshold: show worst-case lower bound (z - CI)
                    z_grid_display = z_grid_rs - z_grid_ci
                    surface_title_suffix = " (Conservative: z - CI)"
                    surface_explanation = "💡 Surface shows **conservative lower bound** (z - CI) to ensure threshold is met"

                elif "Threshold: Must be BELOW" in opt_obj:
                    # Below threshold: show worst-case upper bound (z + CI)
                    z_grid_display = z_grid_rs + z_grid_ci
                    surface_title_suffix = " (Conservative: z + CI)"
                    surface_explanation = "💡 Surface shows **conservative upper bound** (z + CI) to ensure threshold is met"

                # For "Target Value" or "None", z_grid_display remains as z_grid_rs (nominal)

            st.success(f"✅ Surface analysis calculated ({(controls['n_steps']+1)**2} points)")

            # Display surface transformation info if applicable
            if surface_explanation:
                st.info(surface_explanation)

            # Display statistics
            col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
            with col_stat1:
                st.metric("Response Min", f"{z_grid_rs.min():.4f}")
            with col_stat2:
                st.metric("Response Max", f"{z_grid_rs.max():.4f}")
            with col_stat3:
                st.metric("CI Min", f"{z_grid_ci.min():.4f}")
            with col_stat4:
                st.metric("CI Max", f"{z_grid_ci.max():.4f}")

            # CI-specific statistics
            if ci_mode == "prediction":
                st.markdown("---")
                st.markdown("### 📊 Prediction CI Statistics")
                col_ci1, col_ci2, col_ci3 = st.columns(3)
                with col_ci1:
                    st.metric("Min CI", f"{z_grid_ci.min():.6f}")
                with col_ci2:
                    st.metric("Mean CI", f"{z_grid_ci.mean():.6f}")
                with col_ci3:
                    st.metric("Max CI", f"{z_grid_ci.max():.6f}")

            else:  # experimental mode
                st.markdown("---")
                st.markdown("### 📊 Experimental CI Statistics")
                col_ci_exp1, col_ci_exp2, col_ci_exp3, col_ci_exp4 = st.columns(4)

                with col_ci_exp1:
                    st.metric("Min Total CI", f"{z_grid_ci.min():.6f}")
                with col_ci_exp2:
                    st.metric("Mean Total CI", f"{z_grid_ci.mean():.6f}")
                with col_ci_exp3:
                    st.metric("Max Total CI", f"{z_grid_ci.max():.6f}")
                with col_ci_exp4:
                    st.metric("Experimental Component", f"{ci_exp_component:.6f}")

                # Breakdown interpretation
                st.info(f"""
                **Uncertainty Breakdown**:
                - **Experimental error**: ±{ci_exp_component:.6f} (constant across surface)
                - **Model prediction error**: Varies from {z_grid_ci_model.min():.6f} to {z_grid_ci_model.max():.6f}
                - **Total error**: Combined using error propagation (√(model² + exp²))
                """)

            st.markdown("---")

            # 2D Contour Comparison (2x2 grid layout)
            st.markdown("### 📊 2D Contour Comparison")

            # Prepare threshold line if applicable
            threshold_line = None
            opt_obj = st.session_state.get('optimization_objective', 'None - Standard Analysis')

            if "Threshold" in opt_obj:
                threshold_value = st.session_state.get('threshold_value')

                if threshold_value is not None:
                    if "ABOVE" in opt_obj:
                        threshold_line = {
                            'value': threshold_value,
                            'label': f'Threshold (must be above): {threshold_value:.1f}'
                        }
                    elif "BELOW" in opt_obj:
                        threshold_line = {
                            'value': threshold_value,
                            'label': f'Threshold (must be below): {threshold_value:.1f}'
                        }

            if ci_mode == "prediction":
                # Prediction mode: 2x1 layout (2 columns)
                fig_response, fig_ci = plot_contour_comparison_2x2(
                    x_grid_rs, y_grid_rs, z_grid_display,
                    z_grid_ci_pred=z_grid_ci,
                    var1=controls['var1'],
                    var2=controls['var2'],
                    y_var=y_var,
                    fixed_values=controls['fixed_values'],
                    ci_mode="prediction",
                    title_suffix=surface_title_suffix,
                    threshold_line=threshold_line
                )

                col_c1, col_c2 = st.columns(2)
                with col_c1:
                    st.plotly_chart(fig_response, use_container_width=True)
                with col_c2:
                    st.plotly_chart(fig_ci, use_container_width=True)

            else:  # experimental mode
                # Experimental mode: 2x2 grid layout
                fig_response, fig_ci_model, fig_ci_total = plot_contour_comparison_2x2(
                    x_grid_rs, y_grid_rs, z_grid_display,
                    z_grid_ci_pred=z_grid_ci_model,
                    z_grid_ci_total=z_grid_ci,
                    var1=controls['var1'],
                    var2=controls['var2'],
                    y_var=y_var,
                    fixed_values=controls['fixed_values'],
                    ci_mode="experimental",
                    title_suffix=surface_title_suffix,
                    threshold_line=threshold_line
                )

                # Top row: Response + Model CI
                col_c1, col_c2 = st.columns(2)
                with col_c1:
                    st.plotly_chart(fig_response, use_container_width=True)
                with col_c2:
                    st.plotly_chart(fig_ci_model, use_container_width=True)

                # Bottom row: Total CI + spacing
                col_c3, col_c4 = st.columns(2)
                with col_c3:
                    st.plotly_chart(fig_ci_total, use_container_width=True)
                with col_c4:
                    st.markdown("")  # Empty space

            # 3D Surface Comparison
            st.markdown("### 🎯 3D Surface Comparison")

            col_3d1, col_3d2 = st.columns(2)

            with col_3d1:
                st.markdown("**Response Surface**")
                fig_rs_3d = plot_response_surface_3d(
                    x_grid_rs, y_grid_rs, z_grid_display,
                    controls['var1'], controls['var2'], y_var,
                    controls['fixed_values'],
                    title_suffix=surface_title_suffix
                )
                st.plotly_chart(fig_rs_3d, use_container_width=True)

            with col_3d2:
                st.markdown("**CI Semiamplitude**")
                fig_ci_3d = plot_ci_surface_3d(
                    x_grid_ci, y_grid_ci, z_grid_ci,
                    controls['var1'], controls['var2'], y_var,
                    controls['fixed_values']
                )
                st.plotly_chart(fig_ci_3d, use_container_width=True)

            # Interpretation
            st.markdown("### 📈 Interpretation & Optimization")

            # Find optimum response
            max_idx_rs = np.argmax(z_grid_rs)
            max_i_rs, max_j_rs = np.unravel_index(max_idx_rs, z_grid_rs.shape)

            # Find most reliable prediction (lowest CI)
            min_idx_ci = np.argmin(z_grid_ci)
            min_i_ci, min_j_ci = np.unravel_index(min_idx_ci, z_grid_ci.shape)

            col_int1, col_int2 = st.columns(2)

            with col_int1:
                st.success(f"""
                **Maximum Response:** {z_grid_rs[max_i_rs, max_j_rs]:.4f}
                - {controls['var1']} = {x_grid_rs[max_i_rs, max_j_rs]:.3f}
                - {controls['var2']} = {y_grid_rs[max_i_rs, max_j_rs]:.3f}
                - CI at this point: ±{z_grid_ci[max_i_rs, max_j_rs]:.4f}
                """)

            with col_int2:
                st.info(f"""
                **Most Reliable Prediction** (Lowest CI):
                - {controls['var1']} = {x_grid_ci[min_i_ci, min_j_ci]:.3f}
                - {controls['var2']} = {y_grid_ci[min_i_ci, min_j_ci]:.3f}
                - Response at this point: {z_grid_rs[min_i_ci, min_j_ci]:.4f}
                - CI Semiampl.: ±{z_grid_ci[min_i_ci, min_j_ci]:.4f}
                """)

            # Additional insights
            st.markdown("---")
            st.markdown("**Key Insights:**")

            # Check if optimum is in reliable region
            ci_at_optimum = z_grid_ci[max_i_rs, max_j_rs]
            ci_median = np.median(z_grid_ci)

            if ci_at_optimum < ci_median:
                st.success("✅ The optimal response is located in a reliable prediction region (below median CI)")
            else:
                st.warning("⚠️ The optimal response is in a less reliable region (above median CI). Consider validation experiments.")

            # Store in session state
            st.session_state.surface_analysis_data = {
                'response_surface': {'x': x_grid_rs, 'y': y_grid_rs, 'z': z_grid_rs},
                'ci_surface': {'x': x_grid_ci, 'y': y_grid_ci, 'z': z_grid_ci},
                'controls': controls
            }

        except Exception as e:
            st.error(f"❌ Error generating surface analysis: {str(e)}")
            import traceback
            with st.expander("🐛 Error details"):
                st.code(traceback.format_exc())
