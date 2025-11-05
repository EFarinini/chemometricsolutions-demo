"""
Response Surface Visualization
Equivalent to DOE_response_surface.r
Creates 3D wireframe and 2D contour plots of response surface
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def calculate_response_surface(model_results, x_vars, y_var, v1_idx, v2_idx, 
                               fixed_values=None, n_steps=30, value_range=(-1, 1)):
    """
    Calculate response surface for two variables while holding others constant
    
    Args:
        model_results: dict from fit_mlr_model
        x_vars: list of X variable names
        y_var: Y variable name
        v1_idx: index of first variable for surface (0-based)
        v2_idx: index of second variable for surface (0-based)
        fixed_values: dict with fixed values for other variables (default: all 0)
        n_steps: number of grid points (default: 30)
        value_range: tuple (min, max) for the range (default: -1, 1)
    
    Returns:
        tuple: (x_grid, y_grid, z_grid, grid_df)
    """
    n_vars = len(x_vars)
    coefficients = model_results['coefficients']
    
    # Parse coefficient names to understand model structure
    coef_names = coefficients.index.tolist()
    
    # Fixed values for other variables (default to 0 for coded variables)
    if fixed_values is None:
        fixed_values = {var: 0 for var in x_vars}
    
    # Create grid for the two variables
    min_val, max_val = value_range
    step = (max_val - min_val) / n_steps
    grid_1d = np.arange(min_val, max_val + step, step)
    
    # Create 2D mesh grid
    x_grid, y_grid = np.meshgrid(grid_1d, grid_1d)
    
    # Flatten for easier calculation
    n_points = x_grid.size
    x_flat = x_grid.flatten()
    y_flat = y_grid.flatten()
    
    # Build full X matrix for all grid points
    X_grid = np.zeros((n_points, n_vars))
    
    # Set fixed values for all variables first
    for i, var in enumerate(x_vars):
        X_grid[:, i] = fixed_values.get(var, 0)
    
    # Override with grid values for the two variables of interest
    X_grid[:, v1_idx] = x_flat
    X_grid[:, v2_idx] = y_flat
    
    # Create model matrix with interactions and quadratic terms
    X_model = create_prediction_matrix(X_grid, x_vars, coef_names)
    
    # Calculate predictions
    z_flat = X_model @ coefficients.values
    z_grid = z_flat.reshape(x_grid.shape)
    
    # Create DataFrame with grid data
    grid_df = pd.DataFrame({
        'x': x_flat,
        'y': y_flat,
        'z': z_flat
    })
    
    return x_grid, y_grid, z_grid, grid_df


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


def plot_response_surface_3d(x_grid, y_grid, z_grid, var1_name, var2_name, y_var, 
                             fixed_values=None, title_suffix=""):
    """
    Create 3D wireframe/surface plot
    
    Args:
        x_grid, y_grid, z_grid: meshgrid arrays
        var1_name, var2_name: variable names for axes
        y_var: response variable name
        fixed_values: dict of fixed values for subtitle
        title_suffix: additional text for title
    
    Returns:
        plotly Figure object
    """
    fig = go.Figure(data=[
        go.Surface(
            x=x_grid,
            y=y_grid,
            z=z_grid,
            colorscale='Viridis',
            colorbar=dict(title=y_var),
            hovertemplate=f'{var1_name}: %{{x:.3f}}<br>{var2_name}: %{{y:.3f}}<br>{y_var}: %{{z:.3f}}<extra></extra>'
        )
    ])
    
    # Build subtitle with fixed values
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
            xaxis_title=var1_name,
            yaxis_title=var2_name,
            zaxis_title=y_var,
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.3)
            )
        ),
        height=500,
        width=450,
        margin=dict(l=50, r=50, t=60, b=60),
    )
    
    return fig


def plot_response_contour(x_grid, y_grid, z_grid, var1_name, var2_name, y_var,
                          fixed_values=None, n_contours=15):
    """
    Create 2D contour plot
    
    Args:
        x_grid, y_grid, z_grid: meshgrid arrays
        var1_name, var2_name: variable names for axes
        y_var: response variable name
        fixed_values: dict of fixed values for subtitle
        n_contours: number of contour lines
    
    Returns:
        plotly Figure object
    """
    fig = go.Figure(data=[
        go.Contour(
            x=x_grid[0, :],
            y=y_grid[:, 0],
            z=z_grid,
            colorscale='Viridis',
            contours=dict(
                coloring='heatmap',
                showlabels=True,
                labelfont=dict(size=10, color='white')
            ),
            colorbar=dict(title=y_var),
            hovertemplate=f'{var1_name}: %{{x:.3f}}<br>{var2_name}: %{{y:.3f}}<br>{y_var}: %{{z:.3f}}<extra></extra>',
            ncontours=n_contours
        )
    ])
    
    # Build subtitle
    subtitle = ""
    if fixed_values:
        fixed_text = [f"{var}={val:.2f}" for var, val in fixed_values.items()]
        subtitle = ", ".join(fixed_text)
    
    title_text = f"Contour Plot - {y_var}"
    if subtitle:
        title_text += f"<br><sub>{subtitle}</sub>"
    
    fig.update_layout(
        title=title_text,
        xaxis_title=var1_name,
        yaxis_title=var2_name,
        height=500,
        width=450,
        margin=dict(l=50, r=50, t=60, b=60),
        font=dict(size=11),
        yaxis=dict(scaleanchor="x", scaleratio=1)
    )
    
    return fig


def show_response_surface_ui(model_results, x_vars, y_var):
    """
    Streamlit UI for response surface visualization
    
    Args:
        model_results: dict from fit_mlr_model
        x_vars: list of X variable names
        y_var: Y variable name
    """
    st.markdown("## 📈 Response Surface")
    st.markdown("*Equivalent to DOE_response_surface.r*")
    
    if model_results is None:
        st.warning("⚠️ No MLR model fitted. Please fit a model first.")
        return
    
    st.info("Visualize the response surface by selecting two variables")
    
    # Variable selection for surface
    col1, col2 = st.columns(2)
    
    with col1:
        var1 = st.selectbox(
            "Variable for X-axis:",
            x_vars,
            key="response_surface_var1",
            help="First variable to plot"
        )
    
    with col2:
        var2 = st.selectbox(
            "Variable for Y-axis:",
            [v for v in x_vars if v != var1],
            key="response_surface_var2",
            help="Second variable to plot"
        )
    
    # Get indices
    v1_idx = x_vars.index(var1)
    v2_idx = x_vars.index(var2)
    
    # Fixed values for other variables
    other_vars = [v for v in x_vars if v not in [var1, var2]]
    fixed_values = {}
    
    if len(other_vars) > 0:
        st.markdown("### Fixed Values for Other Variables")
        st.info("Set the values for variables not shown in the surface plot (typically 0 for coded variables)")
        
        cols = st.columns(min(3, len(other_vars)))
        for i, var in enumerate(other_vars):
            with cols[i % len(cols)]:
                fixed_values[var] = st.number_input(
                    f"{var}:",
                    value=0.0,
                    step=0.1,
                    format="%.2f",
                    key=f"fixed_val_{var}",
                    help=f"Fixed value for {var} (typically 0 for coded variables)"
                )
    
    # DESIGN SPACE LIMITS (auto-detect from data)
    var1_data = st.session_state.get('X_data', pd.DataFrame())[var1] if 'X_data' in st.session_state else None
    var2_data = st.session_state.get('X_data', pd.DataFrame())[var2] if 'X_data' in st.session_state else None

    # Set min/max based on actual data range (with small margin)
    if var1_data is not None and var2_data is not None:
        var1_min = var1_data.min()
        var1_max = var1_data.max()
        var2_min = var2_data.min()
        var2_max = var2_data.max()

        # Use data range with 10% margin
        margin = 0.1
        design_min = min(var1_min, var2_min) - margin
        design_max = max(var1_max, var2_max) + margin
    else:
        # Fallback to typical coded variable range
        design_min = -1.0
        design_max = 1.0

    # Range settings (constrained to design space)
    st.markdown("### Surface Range (within design space)")

    col_range1, col_range2, col_range3 = st.columns(3)

    with col_range1:
        min_range = st.number_input(
            "Minimum value:",
            value=design_min,
            min_value=design_min - 0.5,  # Allow slight extrapolation
            max_value=design_max - 0.1,
            step=0.1,
            format="%.2f"
        )

    with col_range2:
        max_range = st.number_input(
            "Maximum value:",
            value=design_max,
            min_value=design_min + 0.1,
            max_value=design_max + 0.5,  # Allow slight extrapolation
            step=0.1,
            format="%.2f"
        )

    with col_range3:
        n_steps = st.number_input(
            "Grid resolution:",
            min_value=10,
            max_value=100,
            value=30,
            step=5
        )

    # Warn if extrapolating
    if min_range < design_min or max_range > design_max:
        st.warning("⚠️ You are extrapolating beyond the design space (narrow colors indicate unreliable predictions)")
    
    # Generate surface button
    if st.button("🚀 Generate Response Surface", type="primary"):
        try:
            with st.spinner("Calculating response surface..."):
                # Calculate surface
                x_grid, y_grid, z_grid, grid_df = calculate_response_surface(
                    model_results=model_results,
                    x_vars=x_vars,
                    y_var=y_var,
                    v1_idx=v1_idx,
                    v2_idx=v2_idx,
                    fixed_values=fixed_values,
                    n_steps=n_steps,
                    value_range=(min_range, max_range)
                )
            
            st.success(f"✅ Response surface calculated ({(n_steps+1)**2} points)")
            
            # Check if surface is meaningful (not constant)
            z_range = z_grid.max() - z_grid.min()
            z_relative_range = abs(z_range / z_grid.max()) if z_grid.max() != 0 else 0
            
            if z_relative_range < 0.01:
                st.warning("⚠️ Response surface appears nearly constant. Check your model and variable ranges.")
            
            # Display statistics
            col_stat1, col_stat2, col_stat3 = st.columns(3)
            with col_stat1:
                st.metric("Min Response", f"{z_grid.min():.4f}")
            with col_stat2:
                st.metric("Max Response", f"{z_grid.max():.4f}")
            with col_stat3:
                st.metric("Range", f"{z_range:.4f}")
            
            # 3D Surface plot + 2D Contour plot (side-by-side)
            col_plot1, col_plot2 = st.columns(2)

            with col_plot1:
                st.markdown("### 3D Response Surface")
                fig_3d = plot_response_surface_3d(
                    x_grid, y_grid, z_grid,
                    var1, var2, y_var,
                    fixed_values=fixed_values
                )
                st.plotly_chart(fig_3d, use_container_width=True)

            with col_plot2:
                st.markdown("### 2D Contour Plot")
                fig_contour = plot_response_contour(
                    x_grid, y_grid, z_grid,
                    var1, var2, y_var,
                    fixed_values=fixed_values
                )
                st.plotly_chart(fig_contour, use_container_width=True)
            
            # Optimum finding
            st.markdown("### Response Optimization")
            col_opt1, col_opt2 = st.columns(2)
            
            with col_opt1:
                # Find maximum
                max_idx = np.argmax(z_grid)
                max_i, max_j = np.unravel_index(max_idx, z_grid.shape)
                st.success(f"""
                **Maximum Response:** {z_grid[max_i, max_j]:.4f}
                - {var1} = {x_grid[max_i, max_j]:.3f}
                - {var2} = {y_grid[max_i, max_j]:.3f}
                """)
            
            with col_opt2:
                # Find minimum
                min_idx = np.argmin(z_grid)
                min_i, min_j = np.unravel_index(min_idx, z_grid.shape)
                st.info(f"""
                **Minimum Response:** {z_grid[min_i, min_j]:.4f}
                - {var1} = {x_grid[min_i, min_j]:.3f}
                - {var2} = {y_grid[min_i, min_j]:.3f}
                """)
            
            # Store in session state for export
            st.session_state.response_surface_data = {
                'x_grid': x_grid,
                'y_grid': y_grid,
                'z_grid': z_grid,
                'var1': var1,
                'var2': var2,
                'y_var': y_var,
                'fixed_values': fixed_values
            }
            
        except Exception as e:
            st.error(f"❌ Error generating response surface: {str(e)}")
            import traceback
            with st.expander("🐛 Error details"):
                st.code(traceback.format_exc())
