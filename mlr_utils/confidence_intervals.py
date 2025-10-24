"""
Confidence Interval Surface Visualization
Equivalent to DOE_CI_surface.r
Creates 3D and 2D plots of confidence interval semiamplitude
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy import stats


def calculate_ci_surface(model_results, x_vars, v1_idx, v2_idx,
                         fixed_values=None, s=None, dof=None,
                         n_steps=30, value_range=(-1, 1)):
    """
    Calculate confidence interval semiamplitude surface
    
    Args:
        model_results: dict from fit_mlr_model
        x_vars: list of X variable names
        v1_idx, v2_idx: indices of variables for surface
        fixed_values: dict with fixed values for other variables
        s: experimental standard deviation (if None, uses model RMSE)
        dof: degrees of freedom (if None, uses model dof)
        n_steps: grid resolution
        value_range: (min, max) for the range
    
    Returns:
        tuple: (x_grid, y_grid, ci_grid, grid_df)
    """
    n_vars = len(x_vars)
    coefficients = model_results['coefficients']
    dispersion = model_results['XtX_inv']
    
    # Use model values if not provided
    if s is None:
        s = model_results.get('rmse', 1.0)
    if dof is None:
        dof = model_results.get('dof', 1)
    
    if dof <= 0:
        raise ValueError("Degrees of freedom must be > 0 for confidence intervals")
    
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
    from .response_surface import create_prediction_matrix
    coef_names = coefficients.index.tolist()
    X_model = create_prediction_matrix(X_grid, x_vars, coef_names)
    
    # Calculate leverage for each point
    # leverage = diag(X * (X'X)^-1 * X')
    # CI semiamplitude = t_critical * s * sqrt(leverage)
    
    t_critical = stats.t.ppf(0.975, dof)  # 95% confidence interval
    
    # Calculate leverage for all points
    leverage = np.diag(X_model @ dispersion @ X_model.T)
    
    # CI semiamplitude
    ci_flat = t_critical * s * np.sqrt(leverage)
    ci_grid = ci_flat.reshape(x_grid.shape)
    
    grid_df = pd.DataFrame({
        'x': x_flat,
        'y': y_flat,
        'ci': ci_flat,
        'leverage': leverage
    })
    
    return x_grid, y_grid, ci_grid, grid_df


def plot_ci_surface_3d(x_grid, y_grid, ci_grid, var1_name, var2_name, y_var,
                       fixed_values=None):
    """
    Create 3D surface plot of confidence interval semiamplitude
    
    Args:
        x_grid, y_grid, ci_grid: meshgrid arrays
        var1_name, var2_name: variable names
        y_var: response variable name
        fixed_values: dict of fixed values
    
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
            hovertemplate=f'{var1_name}: %{{x:.3f}}<br>{var2_name}: %{{y:.3f}}<br>CI: %{{z:.4f}}<extra></extra>'
        )
    ])
    
    subtitle = ""
    if fixed_values:
        fixed_text = [f"{var}={val:.2f}" for var, val in fixed_values.items()]
        subtitle = ", ".join(fixed_text)
    
    title_text = f"Confidence Interval Semiamplitude - {y_var}"
    if subtitle:
        title_text += f"<br><sub>{subtitle}</sub>"
    
    fig.update_layout(
        title=title_text,
        scene=dict(
            xaxis_title=var1_name,
            yaxis_title=var2_name,
            zaxis_title='CI Semiamplitude',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.3)
            )
        ),
        height=600,
        width=800
    )
    
    return fig


def plot_ci_contour(x_grid, y_grid, ci_grid, var1_name, var2_name, y_var,
                    fixed_values=None, n_contours=15):
    """
    Create 2D contour plot of confidence interval semiamplitude
    
    Args:
        x_grid, y_grid, ci_grid: meshgrid arrays
        var1_name, var2_name: variable names
        y_var: response variable name
        fixed_values: dict of fixed values
        n_contours: number of contours
    
    Returns:
        plotly Figure
    """
    fig = go.Figure(data=[
        go.Contour(
            x=x_grid[0, :],
            y=y_grid[:, 0],
            z=ci_grid,
            colorscale='YlGnBu',
            contours=dict(
                coloring='heatmap',
                showlabels=True,
                labelfont=dict(size=10, color='white')
            ),
            colorbar=dict(title='CI Semiampl.'),
            hovertemplate=f'{var1_name}: %{{x:.3f}}<br>{var2_name}: %{{y:.3f}}<br>CI: %{{z:.4f}}<extra></extra>',
            ncontours=n_contours
        )
    ])
    
    subtitle = ""
    if fixed_values:
        fixed_text = [f"{var}={val:.2f}" for var, val in fixed_values.items()]
        subtitle = ", ".join(fixed_text)
    
    title_text = f"CI Semiamplitude Contour - {y_var}"
    if subtitle:
        title_text += f"<br><sub>{subtitle}</sub>"
    
    fig.update_layout(
        title=title_text,
        xaxis_title=var1_name,
        yaxis_title=var2_name,
        height=600,
        width=800,
        yaxis=dict(scaleanchor="x", scaleratio=1)
    )
    
    return fig


def show_confidence_intervals_ui(model_results, x_vars, y_var):
    """
    Streamlit UI for confidence interval surface visualization
    
    Args:
        model_results: dict from fit_mlr_model
        x_vars: list of X variable names
        y_var: Y variable name
    """
    st.markdown("## 🎨 Confidence Intervals")
    st.markdown("*Equivalent to DOE_CI_surface.r*")
    
    if model_results is None:
        st.warning("⚠️ No MLR model fitted. Please fit a model first.")
        return
    
    st.info("""
    Visualize the **semiamplitude of the 95% confidence interval** across the experimental domain.
    
    The confidence interval surface shows prediction uncertainty - lower values indicate more reliable predictions.
    """)
    
    # Variance estimation method
    variance_method = st.radio(
        "Experimental variance estimated from:",
        ["Residuals (from model)", "Independent measurements"],
        help="Choose the source of variance estimation"
    )
    
    # Get variance parameters
    if variance_method == "Residuals (from model)":
        s = model_results.get('rmse')
        dof = model_results.get('dof')
        
        if s is None or dof is None or dof <= 0:
            st.error("❌ Model does not have valid RMSE or degrees of freedom")
            st.info("This may occur when the model is saturated (too many parameters for the data)")
            return
        
        col_var1, col_var2 = st.columns(2)
        with col_var1:
            st.metric("Std. Deviation (s)", f"{s:.4f}")
        with col_var2:
            st.metric("Degrees of Freedom", dof)
    
    else:
        st.markdown("### Enter Independent Variance Estimate")
        col_var1, col_var2 = st.columns(2)
        with col_var1:
            s = st.number_input("Experimental standard deviation:", value=1.0, min_value=0.0001, 
                               format="%.4f", step=0.1)
        with col_var2:
            dof = st.number_input("Degrees of freedom:", value=5, min_value=1, step=1)
    
    # Variable selection
    st.markdown("### Variable Selection for Surface")
    col1, col2 = st.columns(2)
    
    with col1:
        var1 = st.selectbox("Variable for X-axis:", x_vars, key="ci_surface_var1")
    
    with col2:
        var2 = st.selectbox("Variable for Y-axis:", 
                           [v for v in x_vars if v != var1], 
                           key="ci_surface_var2")
    
    v1_idx = x_vars.index(var1)
    v2_idx = x_vars.index(var2)
    
    # Fixed values for other variables
    other_vars = [v for v in x_vars if v not in [var1, var2]]
    fixed_values = {}
    
    if len(other_vars) > 0:
        st.markdown("### Fixed Values for Other Variables")
        cols = st.columns(min(3, len(other_vars)))
        for i, var in enumerate(other_vars):
            with cols[i % len(cols)]:
                fixed_values[var] = st.number_input(
                    f"{var}:", value=0.0, step=0.1, format="%.2f",
                    key=f"ci_fixed_val_{var}"
                )
    
    # Range settings
    st.markdown("### Surface Range")
    col_range1, col_range2, col_range3 = st.columns(3)
    
    with col_range1:
        min_range = st.number_input("Minimum:", value=-1.0, step=0.1, format="%.2f", key="ci_min")
    with col_range2:
        max_range = st.number_input("Maximum:", value=1.0, step=0.1, format="%.2f", key="ci_max")
    with col_range3:
        n_steps = st.number_input("Resolution:", min_value=10, max_value=100, value=30, step=5, key="ci_steps")
    
    # Generate button
    if st.button("🚀 Generate CI Surface", type="primary"):
        try:
            with st.spinner("Calculating confidence interval surface..."):
                x_grid, y_grid, ci_grid, grid_df = calculate_ci_surface(
                    model_results=model_results,
                    x_vars=x_vars,
                    v1_idx=v1_idx,
                    v2_idx=v2_idx,
                    fixed_values=fixed_values,
                    s=s,
                    dof=dof,
                    n_steps=n_steps,
                    value_range=(min_range, max_range)
                )
            
            st.success(f"✅ CI surface calculated ({(n_steps+1)**2} points)")
            
            # Statistics
            col_stat1, col_stat2, col_stat3 = st.columns(3)
            with col_stat1:
                st.metric("Min CI Semiampl.", f"{ci_grid.min():.4f}")
            with col_stat2:
                st.metric("Average CI Semiampl.", f"{ci_grid.mean():.4f}")
            with col_stat3:
                st.metric("Max CI Semiampl.", f"{ci_grid.max():.4f}")
            
            # 3D Surface
            st.markdown("### 3D CI Semiamplitude Surface")
            fig_3d = plot_ci_surface_3d(x_grid, y_grid, ci_grid, var1, var2, y_var, fixed_values)
            st.plotly_chart(fig_3d, use_container_width=True)
            
            # 2D Contour
            st.markdown("### 2D CI Semiamplitude Contour")
            fig_contour = plot_ci_contour(x_grid, y_grid, ci_grid, var1, var2, y_var, fixed_values)
            st.plotly_chart(fig_contour, use_container_width=True)
            
            # Interpretation
            st.markdown("### Interpretation")
            
            # Find min/max CI regions
            min_idx = np.argmin(ci_grid)
            min_i, min_j = np.unravel_index(min_idx, ci_grid.shape)
            
            max_idx = np.argmax(ci_grid)
            max_i, max_j = np.unravel_index(max_idx, ci_grid.shape)
            
            col_int1, col_int2 = st.columns(2)
            
            with col_int1:
                st.success(f"""
                **Most Reliable Predictions** (lowest CI):
                - {var1} = {x_grid[min_i, min_j]:.3f}
                - {var2} = {y_grid[min_i, min_j]:.3f}
                - CI Semiampl. = ±{ci_grid[min_i, min_j]:.4f}
                """)
            
            with col_int2:
                st.warning(f"""
                **Least Reliable Predictions** (highest CI):
                - {var1} = {x_grid[max_i, max_j]:.3f}
                - {var2} = {y_grid[max_i, max_j]:.3f}
                - CI Semiampl. = ±{ci_grid[max_i, max_j]:.4f}
                """)
            
            st.info("""
            **Lower CI values** indicate regions where predictions are more reliable (closer to experimental points).
            
            **Higher CI values** indicate extrapolation regions where predictions are less certain.
            """)
            
            # Store in session state
            st.session_state.ci_surface_data = {
                'x_grid': x_grid,
                'y_grid': y_grid,
                'ci_grid': ci_grid,
                'var1': var1,
                'var2': var2,
                'y_var': y_var,
                'fixed_values': fixed_values,
                's': s,
                'dof': dof
            }
            
        except Exception as e:
            st.error(f"❌ Error generating CI surface: {str(e)}")
            import traceback
            with st.expander("🐛 Error details"):
                st.code(traceback.format_exc())
