"""
PCA Plotting Functions

Visualization functions for Principal Component Analysis (PCA) results.
Includes scores plots, loadings plots, scree plots, and diagnostic charts.
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List, Union
from scipy.spatial import ConvexHull

# Import color utilities
from color_utils import get_unified_color_schemes, create_categorical_color_map, is_quantitative_variable


def plot_scree(
    explained_variance_ratio: np.ndarray,
    is_varimax: bool = False,
    component_labels: Optional[List[str]] = None
) -> go.Figure:
    """
    Create scree plot showing variance explained by each component.

    Parameters
    ----------
    explained_variance_ratio : np.ndarray
        Array of variance explained ratios for each component (0-1 scale).
    is_varimax : bool, optional
        Whether this is for Varimax rotated factors. Default is False.
    component_labels : List[str], optional
        Custom labels for components. If None, uses PC1, PC2, etc.

    Returns
    -------
    go.Figure
        Plotly Figure object containing the scree plot.

    Examples
    --------
    >>> var_ratio = np.array([0.45, 0.25, 0.15, 0.10, 0.05])
    >>> fig = plot_scree(var_ratio)
    """
    title_suffix = " (Varimax Factors)" if is_varimax else " (Principal Components)"

    if component_labels is None:
        component_labels = (
            [f'Factor{i+1}' for i in range(len(explained_variance_ratio))]
            if is_varimax else
            [f'PC{i+1}' for i in range(len(explained_variance_ratio))]
        )

    fig = go.Figure()

    # Add bars
    fig.add_trace(go.Bar(
        x=component_labels,
        y=explained_variance_ratio * 100,
        name='Variance Explained',
        marker_color='lightgreen' if is_varimax else 'lightblue'
    ))

    # Add line
    fig.add_trace(go.Scatter(
        x=component_labels,
        y=explained_variance_ratio * 100,
        mode='lines+markers',
        name='Variance Line',
        line=dict(color='red', width=2),
        marker=dict(size=8)
    ))

    x_title = "Factor Number" if is_varimax else "Principal Component"

    fig.update_layout(
        title=f"Scree Plot - Variance Explained{title_suffix}",
        xaxis_title=x_title,
        yaxis_title="Variance Explained (%)",
        height=500
    )

    return fig


def plot_cumulative_variance(
    cumulative_variance: np.ndarray,
    is_varimax: bool = False,
    component_labels: Optional[List[str]] = None,
    reference_lines: Optional[List[float]] = None
) -> go.Figure:
    """
    Create cumulative variance plot.

    Parameters
    ----------
    cumulative_variance : np.ndarray
        Array of cumulative variance explained (0-1 scale).
    is_varimax : bool, optional
        Whether this is for Varimax factors. Default is False.
    component_labels : List[str], optional
        Custom labels for components. If None, auto-generated.
    reference_lines : List[float], optional
        Y-values for reference lines (e.g., [80, 95]). Default is [80, 95].

    Returns
    -------
    go.Figure
        Plotly Figure object containing the cumulative variance plot.

    Examples
    --------
    >>> cum_var = np.array([0.45, 0.70, 0.85, 0.95, 1.00])
    >>> fig = plot_cumulative_variance(cum_var)
    """
    title_suffix = " (Varimax)" if is_varimax else ""

    if component_labels is None:
        component_labels = (
            [f'Factor{i+1}' for i in range(len(cumulative_variance))]
            if is_varimax else
            [f'PC{i+1}' for i in range(len(cumulative_variance))]
        )

    if reference_lines is None:
        reference_lines = [80, 95]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=component_labels,
        y=cumulative_variance * 100,
        mode='lines+markers',
        name='Cumulative Variance',
        line=dict(color='green' if is_varimax else 'blue', width=3),
        marker=dict(size=10),
        fill='tonexty'
    ))

    # Add reference lines
    for threshold in reference_lines:
        fig.add_hline(
            y=threshold,
            line_dash="dash",
            line_color="red" if threshold == 80 else "orange",
            annotation_text=f"{threshold}%"
        )

    x_title = "Factor Number" if is_varimax else "Principal Component"

    fig.update_layout(
        title=f"Cumulative Variance Explained{title_suffix}",
        xaxis_title=x_title,
        yaxis_title="Cumulative Variance (%)",
        height=500
    )

    return fig


def plot_scores(
    scores: pd.DataFrame,
    pc_x: str,
    pc_y: str,
    explained_variance_ratio: np.ndarray,
    color_by: Optional[Union[pd.Series, str]] = None,
    text_labels: Optional[pd.Series] = None,
    is_varimax: bool = False,
    show_labels: bool = False,
    show_convex_hull: bool = False,
    hull_opacity: float = 0.7
) -> go.Figure:
    """
    Create scores scatter plot with optional color mapping and convex hulls.

    Parameters
    ----------
    scores : pd.DataFrame
        DataFrame containing PCA scores with PC columns.
    pc_x : str
        Column name for X-axis principal component (e.g., 'PC1').
    pc_y : str
        Column name for Y-axis principal component (e.g., 'PC2').
    explained_variance_ratio : np.ndarray
        Array of variance explained ratios.
    color_by : pd.Series or str, optional
        Data for coloring points. Can be categorical or quantitative.
    text_labels : pd.Series, optional
        Text labels for each point. If None, uses scores index.
    is_varimax : bool, optional
        Whether this is for Varimax factors. Default is False.
    show_labels : bool, optional
        Whether to display text labels on plot. Default is False.
    show_convex_hull : bool, optional
        Whether to add convex hulls for categorical groups. Default is False.
    hull_opacity : float, optional
        Opacity of convex hull lines (0-1). Default is 0.7.

    Returns
    -------
    go.Figure
        Plotly Figure object containing the scores plot.

    Examples
    --------
    >>> scores_df = pd.DataFrame({'PC1': [1, 2, 3], 'PC2': [4, 5, 6]})
    >>> var_ratio = np.array([0.4, 0.3, 0.2])
    >>> fig = plot_scores(scores_df, 'PC1', 'PC2', var_ratio)
    """
    title_suffix = " (Varimax Factors)" if is_varimax else ""

    # Get component indices
    pc_cols = scores.columns.tolist()
    pc_x_idx = pc_cols.index(pc_x)
    pc_y_idx = pc_cols.index(pc_y)
    var_x = explained_variance_ratio[pc_x_idx] * 100
    var_y = explained_variance_ratio[pc_y_idx] * 100
    var_total = var_x + var_y

    # Prepare text labels
    if text_labels is None:
        text_param = scores.index.astype(str)
    else:
        text_param = text_labels

    # Calculate axis range for equal aspect ratio
    x_range = [scores[pc_x].min(), scores[pc_x].max()]
    y_range = [scores[pc_y].min(), scores[pc_y].max()]
    max_abs_range = max(abs(min(x_range + y_range)), abs(max(x_range + y_range)))
    axis_range = [-max_abs_range * 1.1, max_abs_range * 1.1]

    # Create plot with smart color mapping
    color_discrete_map = None

    if color_by is None:
        # No coloring
        fig = px.scatter(
            x=scores[pc_x],
            y=scores[pc_y],
            text=text_param,
            labels={'x': f'{pc_x} ({var_x:.1f}%)', 'y': f'{pc_y} ({var_y:.1f}%)'}
        )
        title_text = f"Scores: {pc_x} vs {pc_y}{title_suffix}<br>Total Explained Variance: {var_total:.1f}%"
    else:
        # Determine if color data is quantitative or categorical
        color_data = color_by if isinstance(color_by, pd.Series) else pd.Series(color_by, index=scores.index)

        if is_quantitative_variable(color_data):
            # Quantitative: use blue-to-red continuous scale
            color_name = color_data.name if hasattr(color_data, 'name') and color_data.name else 'variable'
            fig = px.scatter(
                x=scores[pc_x],
                y=scores[pc_y],
                color=color_data,
                text=text_param,
                color_continuous_scale=[(0.0, 'rgb(0, 0, 255)'), (0.5, 'rgb(128, 0, 128)'), (1.0, 'rgb(255, 0, 0)')],
                labels={'x': f'{pc_x} ({var_x:.1f}%)', 'y': f'{pc_y} ({var_y:.1f}%)', 'color': color_name}
            )
            title_text = f"Scores: {pc_x} vs {pc_y} (colored by {color_name}){title_suffix}<br>Total Explained Variance: {var_total:.1f}%"
        else:
            # Categorical: use discrete color map
            color_name = color_data.name if hasattr(color_data, 'name') and color_data.name else 'group'
            unique_values = color_data.dropna().unique()
            color_discrete_map = create_categorical_color_map(unique_values)

            fig = px.scatter(
                x=scores[pc_x],
                y=scores[pc_y],
                color=color_data,
                text=text_param,
                color_discrete_map=color_discrete_map,
                labels={'x': f'{pc_x} ({var_x:.1f}%)', 'y': f'{pc_y} ({var_y:.1f}%)', 'color': color_name}
            )
            title_text = f"Scores: {pc_x} vs {pc_y} (colored by {color_name}){title_suffix}<br>Total Explained Variance: {var_total:.1f}%"

    # Add zero reference lines
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.7)
    fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.7)

    # Add convex hulls if requested (only for categorical data)
    if show_convex_hull and color_by is not None and color_discrete_map is not None:
        try:
            color_data = color_by if isinstance(color_by, pd.Series) else pd.Series(color_by, index=scores.index)
            fig = add_convex_hulls(fig, scores, pc_x, pc_y, color_data, color_discrete_map, hull_opacity)
        except Exception:
            pass  # Silently skip if convex hull fails

    # Configure text labels with optimized rendering
    if show_labels:
        # Convert labels to simple integers if they are numeric indices
        text_param_vals = [str(int(float(t))) if str(t).replace('.', '').isdigit() else str(t)
                          for t in text_param]

        # Update traces with optimized text display
        fig.update_traces(
            text=text_param_vals,
            textposition='middle center',      # Center on points
            textfont=dict(
                size=9,                        # Smaller, less intrusive
                color='rgba(0, 0, 0, 0.6)'    # Slightly transparent
            )
        )

    # Set equal aspect ratio with centered title and compact layout
    fig.update_layout(
        title=dict(
            text=title_text,
            x=0.5,                          # Center horizontally
            xanchor='center',
            font=dict(size=14, color='#333')  # Slightly smaller, darker
        ),
        height=600,
        width=900,                          # Wider for better use of space
        xaxis=dict(
            range=axis_range,
            scaleanchor="y",
            scaleratio=1,
            constrain="domain"
        ),
        yaxis=dict(
            range=axis_range,
            constrain="domain"
        ),
        # Compact margins
        margin=dict(l=60, r=60, t=80, b=60),
        # Legend inside plot, top-right corner - NO BORDER
        legend=dict(
            x=0.99,           # Very close to right edge
            y=0.99,           # Very close to top edge
            xanchor='right',
            yanchor='top',
            bgcolor='rgba(255, 255, 255, 0.9)',  # Slightly opaque white
            borderwidth=0,     # NO BORDER (removes border completely)
            font=dict(size=10)
        ),
        hovermode='closest',
        template='plotly_white'
    )

    return fig


def plot_loadings(
    loadings: pd.DataFrame,
    pc_x: str,
    pc_y: str,
    explained_variance_ratio: np.ndarray,
    is_varimax: bool = False,
    color_by_magnitude: bool = False
) -> go.Figure:
    """
    Create loadings scatter plot.

    Parameters
    ----------
    loadings : pd.DataFrame
        DataFrame containing PCA loadings with PC columns.
        Index should be variable names.
    pc_x : str
        Column name for X-axis principal component.
    pc_y : str
        Column name for Y-axis principal component.
    explained_variance_ratio : np.ndarray
        Array of variance explained ratios.
    is_varimax : bool, optional
        Whether this is for Varimax factors. Default is False.
    color_by_magnitude : bool, optional
        Whether to color points by loading magnitude. Default is False.

    Returns
    -------
    go.Figure
        Plotly Figure object containing the loadings plot.

    Examples
    --------
    >>> loadings_df = pd.DataFrame({'PC1': [0.8, 0.2], 'PC2': [0.1, 0.9]}, index=['var1', 'var2'])
    >>> var_ratio = np.array([0.6, 0.3])
    >>> fig = plot_loadings(loadings_df, 'PC1', 'PC2', var_ratio)
    """
    title_suffix = " (Varimax Factors)" if is_varimax else ""

    # Get component indices
    pc_cols = loadings.columns.tolist()
    pc_x_idx = pc_cols.index(pc_x)
    pc_y_idx = pc_cols.index(pc_y)
    var_x = explained_variance_ratio[pc_x_idx] * 100
    var_y = explained_variance_ratio[pc_y_idx] * 100
    var_total = var_x + var_y

    # Create scatter plot
    fig = px.scatter(
        x=loadings[pc_x],
        y=loadings[pc_y],
        text=loadings.index,
        title=f"Loadings Plot: {pc_x} vs {pc_y}{title_suffix}<br>Total Explained Variance: {var_total:.1f}%",
        labels={
            'x': f'{pc_x} Loadings ({var_x:.1f}%)',
            'y': f'{pc_y} Loadings ({var_y:.1f}%)'
        }
    )

    # Calculate symmetric axis range
    x_range = [loadings[pc_x].min(), loadings[pc_x].max()]
    y_range = [loadings[pc_y].min(), loadings[pc_y].max()]
    max_abs_range = max(abs(min(x_range + y_range)), abs(max(x_range + y_range)))
    axis_range = [-max_abs_range * 1.1, max_abs_range * 1.1]

    # Add zero reference lines
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.7)
    fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.7)

    # Color by magnitude if requested (useful for Varimax)
    if color_by_magnitude:
        magnitude = np.sqrt(loadings[pc_x]**2 + loadings[pc_y]**2)
        fig.update_traces(marker=dict(
            color=magnitude,
            colorscale='viridis',
            showscale=True,
            colorbar=dict(title="Loading Magnitude")
        ))

    # Show variable labels
    fig.update_traces(textposition="top center")

    # Set equal aspect ratio
    fig.update_layout(
        height=600,
        width=600,
        xaxis=dict(
            range=axis_range,
            scaleanchor="y",
            scaleratio=1,
            constrain="domain"
        ),
        yaxis=dict(
            range=axis_range,
            constrain="domain"
        )
    )

    return fig


def plot_loadings_line(
    loadings: pd.DataFrame,
    selected_components: List[str],
    is_varimax: bool = False
) -> go.Figure:
    """
    Create line plot of loadings across variables.

    Useful for visualizing loading patterns across many variables (e.g., spectral wavelengths).

    Parameters
    ----------
    loadings : pd.DataFrame
        DataFrame containing PCA loadings.
        Index should contain variable names.
    selected_components : List[str]
        List of component names to plot (e.g., ['PC1', 'PC2']).
    is_varimax : bool, optional
        Whether this is for Varimax factors. Default is False.

    Returns
    -------
    go.Figure
        Plotly Figure object containing the loadings line plot.

    Examples
    --------
    >>> loadings_df = pd.DataFrame(
    ...     {'PC1': [0.1, 0.2, 0.8], 'PC2': [0.9, 0.1, 0.1]},
    ...     index=['Var1', 'Var2', 'Var3']
    ... )
    >>> fig = plot_loadings_line(loadings_df, ['PC1', 'PC2'])
    """
    title_suffix = " (Varimax)" if is_varimax else ""

    fig = go.Figure()

    # Create x-axis with variable indices for positioning
    x_indices = list(range(len(loadings.index)))
    variable_names = list(loadings.index)

    for comp in selected_components:
        if comp in loadings.columns:
            fig.add_trace(go.Scatter(
                x=x_indices,
                y=loadings[comp],
                mode='lines+markers',
                name=comp,
                text=variable_names,
                hovertemplate='Variable: %{text}<br>Loading: %{y:.3f}<extra></extra>'
            ))

    fig.update_layout(
        title=f"Loading Line Plot{title_suffix}",
        xaxis_title="Variable Name",
        yaxis_title="Loading Value",
        height=500,
        hovermode='x unified'
    )

    # Replace x-axis tick labels with variable names
    # Show every nth label to avoid crowding (depends on number of variables)
    n_vars = len(loadings.index)
    if n_vars > 0:
        # Calculate tick spacing: show ~10-15 labels max
        tick_spacing = max(1, n_vars // 15) if n_vars > 15 else 1
        tick_positions = list(range(0, n_vars, tick_spacing))
        tick_labels = [variable_names[i] for i in tick_positions]

        fig.update_xaxes(
            tickmode='array',
            tickvals=tick_positions,
            ticktext=tick_labels,
            tickangle=-45
        )

    # Add zero reference line
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)

    return fig


def plot_loadings_line_antiderivative(
    loadings: pd.DataFrame,
    selected_components: List[str],
    derivative_order: int = 1,
    is_varimax: bool = False
) -> go.Figure:
    """
    Line plot of antiderivative loadings (recovered spectral shape from derivatives).

    Parameters
    ----------
    loadings : pd.DataFrame
        Loading matrix from derivative-preprocessed data.
        Index should contain variable names.
    selected_components : List[str]
        Component names to plot (e.g., ['PC1', 'PC2']).
    derivative_order : int, optional
        Order of derivative applied to original data (1 or 2). Default is 1.
    is_varimax : bool, optional
        Varimax rotated factors. Default is False.

    Returns
    -------
    go.Figure
        Plotly line plot with antiderivative loadings.
    """
    from pca_utils.pca_calculations import calculate_antiderivative_loadings

    title_suffix = " (Varimax)" if is_varimax else ""

    # Get component indices
    comp_indices = [list(loadings.columns).index(comp) for comp in selected_components]

    # Calculate antiderivatives using existing function
    antiderivatives = calculate_antiderivative_loadings(
        loadings.values, comp_indices, derivative_order=derivative_order
    )

    fig = go.Figure()

    # Original variable names from loadings index
    original_variable_names = list(loadings.index)

    # After n integrations, first n elements are lost
    # So create adjusted variable names list
    n_lost = derivative_order
    adjusted_variable_names = original_variable_names[n_lost:] if len(original_variable_names) > n_lost else original_variable_names

    # Add trace for each component
    for i, comp in enumerate(selected_components):
        antideriv_data = antiderivatives[f'antideriv_{comp_indices[i]}']
        x_indices = list(range(len(antideriv_data)))

        # Use adjusted variable names (accounting for integration)
        var_names_for_plot = adjusted_variable_names[:len(antideriv_data)]

        fig.add_trace(go.Scatter(
            x=x_indices,
            y=antideriv_data,
            mode='lines+markers',
            name=comp,
            text=var_names_for_plot,
            hovertemplate='Variable: %{text}<br>Antideriv: %{y:.3f}<extra></extra>'
        ))

    fig.update_layout(
        title=f"Loading Line Plot (Antiderivative - Order {derivative_order}){title_suffix}",
        xaxis_title="Variable Name",
        yaxis_title="Antiderivative Value",
        height=500,
        hovermode='x unified'
    )

    # Replace x-axis tick labels with variable names
    n_points = len(antideriv_data)
    if n_points > 0:
        # Calculate tick spacing: show ~10-15 labels max
        tick_spacing = max(1, n_points // 15) if n_points > 15 else 1
        tick_positions = list(range(0, n_points, tick_spacing))
        tick_labels = [adjusted_variable_names[i] if i < len(adjusted_variable_names) else f"Pt{i}"
                      for i in tick_positions]

        fig.update_xaxes(
            tickmode='array',
            tickvals=tick_positions,
            ticktext=tick_labels,
            tickangle=-45
        )

    # Add zero reference line
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)

    return fig


def plot_loadings_antiderivative(
    loadings: pd.DataFrame,
    pc_x: str,
    pc_y: str,
    explained_variance_ratio: np.ndarray,
    derivative_order: int = 1,
    is_varimax: bool = False,
    color_by_magnitude: bool = False
) -> go.Figure:
    """
    Create loadings scatter plot with antiderivative values.

    Recovers spectral shape from derivative loadings using trapezoidal integration.
    Displays antiderivative coordinates on 2D scatter plot.

    Parameters
    ----------
    loadings : pd.DataFrame
        Loading matrix from derivative-preprocessed data.
        Index should be variable names.
    pc_x : str
        Column name for X-axis component (e.g., 'PC1').
    pc_y : str
        Column name for Y-axis component (e.g., 'PC2').
    explained_variance_ratio : np.ndarray
        Array of variance explained ratios.
    derivative_order : int, optional
        Order of derivative applied to original data (1 or 2). Default is 1.
    is_varimax : bool, optional
        Whether this is for Varimax factors. Default is False.
    color_by_magnitude : bool, optional
        Whether to color points by magnitude. Default is False.

    Returns
    -------
    go.Figure
        Plotly scatter plot with antiderivative loadings.
    """
    from pca_utils.pca_calculations import calculate_antiderivative_loadings

    # Get component indices
    pc_cols = loadings.columns.tolist()
    pc_x_idx = pc_cols.index(pc_x)
    pc_y_idx = pc_cols.index(pc_y)

    # Calculate antiderivatives
    antiderivatives = calculate_antiderivative_loadings(
        loadings.values, [pc_x_idx, pc_y_idx], derivative_order=derivative_order
    )

    antideriv_x = antiderivatives[f'antideriv_{pc_x_idx}']
    antideriv_y = antiderivatives[f'antideriv_{pc_y_idx}']

    # Use minimum length (due to integration reducing dimensions)
    min_len = min(len(antideriv_x), len(antideriv_y))
    antideriv_x = antideriv_x[:min_len]
    antideriv_y = antideriv_y[:min_len]

    # Get variance info
    var_x = explained_variance_ratio[pc_x_idx] * 100
    var_y = explained_variance_ratio[pc_y_idx] * 100
    var_total = var_x + var_y

    title_suffix = " (Varimax Factors)" if is_varimax else ""
    deriv_note = f"(Antiderivative - Order {derivative_order})"

    # Prepare variable names (adjusted for integration loss)
    original_variable_names = list(loadings.index)
    n_lost = derivative_order
    adjusted_variable_names = original_variable_names[n_lost:] if len(original_variable_names) > n_lost else original_variable_names
    point_labels = adjusted_variable_names[:min_len]

    # CRITICAL: Synchronize lengths - trim antiderivatives if labels are shorter
    if len(point_labels) < len(antideriv_x):
        antideriv_x = antideriv_x[:len(point_labels)]
        antideriv_y = antideriv_y[:len(point_labels)]

    # Create scatter plot with labels
    fig = px.scatter(
        x=antideriv_x,
        y=antideriv_y,
        text=point_labels,
        title=f"Loadings Plot: {pc_x} vs {pc_y} {deriv_note}{title_suffix}<br>Total Explained Variance: {var_total:.1f}%",
        labels={
            'x': f'{pc_x} Antiderivative ({var_x:.1f}%)',
            'y': f'{pc_y} Antiderivative ({var_y:.1f}%)'
        }
    )

    # Calculate symmetric axis range
    x_range = [min(antideriv_x), max(antideriv_x)]
    y_range = [min(antideriv_y), max(antideriv_y)]
    max_abs_range = max(abs(min(x_range + y_range)), abs(max(x_range + y_range)))
    axis_range = [-max_abs_range * 1.1, max_abs_range * 1.1]

    # Add zero reference lines
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.7)
    fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.7)

    # Color by magnitude if requested
    if color_by_magnitude:
        magnitude = np.sqrt(antideriv_x**2 + antideriv_y**2)
        fig.update_traces(
            marker=dict(
                color=magnitude,
                colorscale='viridis',
                showscale=True,
                colorbar=dict(title="Antideriv Magnitude")
            ),
            textposition="top center"
        )
    else:
        fig.update_traces(marker=dict(size=8, opacity=0.6), textposition="top center")

    # Set equal aspect ratio
    fig.update_layout(
        height=600,
        width=600,
        xaxis=dict(
            range=axis_range,
            scaleanchor="y",
            scaleratio=1,
            constrain="domain"
        ),
        yaxis=dict(
            range=axis_range,
            constrain="domain"
        )
    )

    return fig


def plot_biplot(
    scores: pd.DataFrame,
    loadings: pd.DataFrame,
    pc_x: str,
    pc_y: str,
    explained_variance_ratio: np.ndarray,
    color_by: Optional[pd.Series] = None,
    loading_scale: float = 1.0,
    max_loadings: int = 20,
    is_varimax: bool = False
) -> go.Figure:
    """
    Create biplot combining scores and loadings.

    Parameters
    ----------
    scores : pd.DataFrame
        DataFrame containing PCA scores.
    loadings : pd.DataFrame
        DataFrame containing PCA loadings.
    pc_x : str
        Column name for X-axis principal component.
    pc_y : str
        Column name for Y-axis principal component.
    explained_variance_ratio : np.ndarray
        Array of variance explained ratios.
    color_by : pd.Series, optional
        Data for coloring score points. Default is None.
    loading_scale : float, optional
        Scaling factor for loading vectors. Default is 1.0.
    max_loadings : int, optional
        Maximum number of loading vectors to display. Default is 20.
    is_varimax : bool, optional
        Whether this is for Varimax factors. Default is False.

    Returns
    -------
    go.Figure
        Plotly Figure object containing the biplot.

    Notes
    -----
    Biplots show both samples (scores) and variables (loadings) simultaneously.

    Examples
    --------
    >>> scores_df = pd.DataFrame({'PC1': [1, 2], 'PC2': [3, 4]})
    >>> loadings_df = pd.DataFrame({'PC1': [0.8, 0.2], 'PC2': [0.1, 0.9]}, index=['var1', 'var2'])
    >>> var_ratio = np.array([0.6, 0.3])
    >>> fig = plot_biplot(scores_df, loadings_df, 'PC1', 'PC2', var_ratio)
    """
    # First create scores plot
    fig = plot_scores(scores, pc_x, pc_y, explained_variance_ratio, color_by=color_by, is_varimax=is_varimax)

    # Update title
    title_suffix = " (Varimax)" if is_varimax else ""
    pc_cols = scores.columns.tolist()
    pc_x_idx = pc_cols.index(pc_x)
    pc_y_idx = pc_cols.index(pc_y)
    var_x = explained_variance_ratio[pc_x_idx] * 100
    var_y = explained_variance_ratio[pc_y_idx] * 100
    var_total = var_x + var_y

    fig.update_layout(
        title=f"Biplot: {pc_x} vs {pc_y}{title_suffix}<br>Total Explained Variance: {var_total:.1f}%"
    )

    # Calculate loading magnitudes and select top contributors
    loading_magnitude = np.sqrt(loadings[pc_x]**2 + loadings[pc_y]**2)
    top_indices = loading_magnitude.nlargest(max_loadings).index

    # Add loading vectors
    for var_name in top_indices:
        x_load = loadings.loc[var_name, pc_x] * loading_scale
        y_load = loadings.loc[var_name, pc_y] * loading_scale

        # Add arrow (line)
        fig.add_trace(go.Scatter(
            x=[0, x_load],
            y=[0, y_load],
            mode='lines+text',
            line=dict(color='red', width=1.5),
            text=['', var_name],
            textposition='top center',
            textfont=dict(size=9, color='darkred'),
            showlegend=False,
            hoverinfo='text',
            hovertext=f'Variable: {var_name}<br>Loading {pc_x}: {loadings.loc[var_name, pc_x]:.3f}<br>Loading {pc_y}: {loadings.loc[var_name, pc_y]:.3f}'
        ))

        # Add arrowhead
        fig.add_annotation(
            x=x_load, y=y_load,
            ax=0, ay=0,
            xref='x', yref='y',
            axref='x', ayref='y',
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=1.5,
            arrowcolor='red',
            opacity=0.6
        )

    return fig


def add_convex_hulls(
    fig: go.Figure,
    scores: pd.DataFrame,
    pc_x: str,
    pc_y: str,
    color_data: Union[pd.Series, np.ndarray],
    color_discrete_map: Optional[Dict[Any, str]] = None,
    hull_opacity: float = 0.7
) -> go.Figure:
    """
    Add convex hulls for categorical groups to a scores plot.

    Parameters
    ----------
    fig : go.Figure
        Existing Plotly figure (typically a scores plot).
    scores : pd.DataFrame
        DataFrame containing PCA scores.
    pc_x : str
        Column name for X-axis principal component.
    pc_y : str
        Column name for Y-axis principal component.
    color_data : pd.Series or np.ndarray
        Categorical data defining groups.
    color_discrete_map : dict, optional
        Color mapping for groups. If None, uses default colors.
    hull_opacity : float, optional
        Opacity of hull lines (0-1). Default is 0.7.

    Returns
    -------
    go.Figure
        Modified Plotly Figure with convex hulls added.

    Notes
    -----
    Only works with categorical grouping variables.
    Requires at least 3 points per group to compute hull.

    Examples
    --------
    >>> fig = plot_scores(scores, 'PC1', 'PC2', var_ratio, color_by=groups)
    >>> fig = add_convex_hulls(fig, scores, 'PC1', 'PC2', groups, color_map)
    """
    try:
        if color_data is None:
            return fig

        # Convert color_data to Series
        if hasattr(color_data, 'index'):
            color_series = pd.Series(color_data, index=color_data.index)
        else:
            color_series = pd.Series(color_data, index=scores.index)

        color_series = color_series.reindex(scores.index)
        unique_groups = color_series.dropna().unique()

        if len(unique_groups) == 0:
            return fig

        # Use provided color map or create default
        if color_discrete_map is None:
            unique_values = color_series.dropna().unique()
            color_discrete_map = create_categorical_color_map(unique_values)

        # Calculate convex hull for each group
        for group in unique_groups:
            group_mask = color_series == group
            n_points = group_mask.sum()

            if n_points < 3:
                # Need at least 3 points for a hull
                continue

            # Extract coordinates
            group_scores_x = scores.loc[group_mask, pc_x].values
            group_scores_y = scores.loc[group_mask, pc_y].values
            group_points = np.column_stack([group_scores_x, group_scores_y])

            try:
                # Compute convex hull
                hull = ConvexHull(group_points)
                hull_vertices = hull.vertices
                hull_points = group_points[hull_vertices]

                # Close the polygon
                hull_x = np.append(hull_points[:, 0], hull_points[0, 0])
                hull_y = np.append(hull_points[:, 1], hull_points[0, 1])

                # Get color for this group
                group_color = color_discrete_map.get(group, 'gray')

                # Add hull trace
                fig.add_trace(go.Scatter(
                    x=hull_x,
                    y=hull_y,
                    mode='lines',
                    line=dict(color=group_color, width=1),
                    opacity=hull_opacity,
                    fill=None,
                    name=f'{group}_hull',
                    showlegend=False,
                    hoverinfo='skip'
                ))

            except Exception:
                # Skip this group if hull computation fails
                continue

    except Exception:
        # Return original figure if any error occurs
        pass

    return fig


def get_continuous_color_for_value(value: float, min_val: float, max_val: float) -> str:
    """
    Convert a numeric value to a color on the BLUE→RED gradient scale.

    Parameters
    ----------
    value : float
        The value to convert to color
    min_val : float
        Minimum value in the range (maps to BLUE)
    max_val : float
        Maximum value in the range (maps to RED)

    Returns
    -------
    str
        RGB color string in format 'rgb(R, G, B)'

    Examples
    --------
    >>> get_continuous_color_for_value(0, 0, 1)  # Minimum value
    'rgb(0, 0, 255)'
    >>> get_continuous_color_for_value(1, 0, 1)  # Maximum value
    'rgb(255, 0, 0)'
    >>> get_continuous_color_for_value(0.5, 0, 1)  # Middle value
    'rgb(128, 0, 128)'
    """
    # Normalize value to 0-1 range
    if max_val == min_val:
        normalized = 0.5  # If all values are the same, use middle color
    else:
        normalized = (value - min_val) / (max_val - min_val)
        normalized = max(0.0, min(1.0, normalized))  # Clamp to [0, 1]

    # Blue → Purple → Red gradient
    # Blue (0, 0, 255) at normalized=0
    # Purple (128, 0, 128) at normalized=0.5
    # Red (255, 0, 0) at normalized=1

    if normalized <= 0.5:
        # Blue to Purple
        t = normalized * 2  # Scale to 0-1
        r = int(0 + t * 128)
        g = 0
        b = int(255 - t * 127)
    else:
        # Purple to Red
        t = (normalized - 0.5) * 2  # Scale to 0-1
        r = int(128 + t * 127)
        g = 0
        b = int(128 - t * 128)

    return f'rgb({r}, {g}, {b})'


def add_sample_trajectory_lines(
    fig: go.Figure,
    scores: pd.DataFrame,
    pc_x: str,
    pc_y: str,
    line_strategy: str = "none",
    groupby_column: Optional[pd.Series] = None,
    line_width: int = 2,
    line_opacity: float = 0.6,
    color_by_index: bool = True,  # DEFAULT: Use sequential index (1→N)
    color_variable: Optional[str] = None,
    original_data: Optional[pd.DataFrame] = None,
    trajectory_color_vector: Optional[Dict[str, Any]] = None,
    color_discrete_map: Optional[Dict[Any, str]] = None,  # DEPRECATED: Not used
    metadata_column: Optional[pd.Series] = None,  # NEW: Metadata for categorical coloring
    use_category_colors: bool = False,  # NEW: Use category colors instead of sequential
    show_trajectory_arrow: bool = True  # NEW: Show directional arrow at end
) -> go.Figure:
    """
    Add sample trajectory lines to a score plot with gradient coloring.

    Visualizes the progression of samples through PCA space using different
    connection strategies and coloring options (sequential index or numeric variable).

    Parameters
    ----------
    fig : go.Figure
        Existing Plotly figure with scatter plot.
    scores : pd.DataFrame
        DataFrame containing PCA scores.
        Index represents sample order (0, 1, 2, ...).
        Columns contain PC scores (PC1, PC2, ...).
    pc_x : str
        Column name for X-axis principal component (e.g., 'PC1').
    pc_y : str
        Column name for Y-axis principal component (e.g., 'PC2').
    line_strategy : str, optional
        Line drawing strategy. Default is 'none'.
        Options:
        - 'none': No lines drawn (returns fig unchanged)
        - 'sequential': Single line connecting all points in dataset order
        - 'categorical': Separate trajectory lines per category (requires groupby_column)
    groupby_column : pd.Series, optional
        Categorical data for grouping (required if line_strategy='categorical').
        Index should align with scores.index.
    color_discrete_map : dict, optional
        Color mapping for categories {category: 'rgb(...)'}.
        Generated by create_categorical_color_map() if needed.
    line_width : int, optional
        Thickness of trajectory lines. Default is 2.
    line_opacity : float, optional
        Opacity of trajectory lines (0.0 to 1.0). Default is 0.6.
    color_by_index : bool, optional
        If True, color by sequential index (1, 2, 3, ..., N) per category. Default is False.
    color_variable : str, optional
        Name of numeric variable to color by (instead of index). Default is None.
    original_data : pd.DataFrame, optional
        Original data containing the color_variable column. Default is None.
    trajectory_color_vector : dict, optional
        Dict with keys: 'category', 'variable', 'values', 'min', 'max'
        When provided, applies custom coloring to specific batch. Default is None.
    metadata_column : pd.Series, optional
        Metadata column for category-based coloring (NEW). When provided with
        use_category_colors=True, each trajectory segment is colored by its category.
        Default is None.
    use_category_colors : bool, optional
        If True, use category colors from metadata_column instead of sequential coloring.
        Default is False.
    show_trajectory_arrow : bool, optional
        If True, add a vector arrow at the end of each trajectory line to indicate
        direction of progression. Default is True.

    Returns
    -------
    go.Figure
        Modified figure with trajectory lines added.

    Raises
    ------
    ValueError
        If line_strategy='categorical' but groupby_column is None.

    Notes
    -----
    - Lines are drawn as multiple segments to create gradient effects
    - Categories with fewer than 2 points are skipped
    - Default coloring: BLUE (start) → PURPLE (middle) → RED (end)
    - Trajectory direction is determined by sample order in scores DataFrame

    Examples
    --------
    >>> scores_df = pd.DataFrame(
    ...     {'PC1': [1, 2, 3], 'PC2': [4, 5, 6]},
    ...     index=[0, 1, 2]
    ... )
    >>> var_ratio = np.array([0.4, 0.3, 0.2])
    >>> fig = plot_scores(scores_df, 'PC1', 'PC2', var_ratio)
    >>> fig = add_sample_trajectory_lines(
    ...     fig, scores_df, 'PC1', 'PC2',
    ...     line_strategy='sequential',
    ...     color_by_index=True
    ... )
    """
    try:
        def add_trajectory_arrow(fig, start_x, start_y, end_x, end_y, color, size=15, width=2):
            """
            Add a simple classic arrow (vector style) at the end of trajectory.
            Two lines forming a V tip (like mathematical arrow).

            Parameters:
            - fig: Plotly figure
            - start_x, start_y: Arrow base (start point)
            - end_x, end_y: Arrow tip (end point)
            - color: Arrow color (RGB string)
            - size: Arrow tip size
            - width: Arrow line width

            Returns:
            - Modified figure with arrow added
            """
            import math

            # Calculate arrow direction
            dx = end_x - start_x
            dy = end_y - start_y
            distance = math.sqrt(dx**2 + dy**2)

            if distance < 0.01:  # Skip if start and end are too close
                return fig

            # Normalize direction vector
            dx_norm = dx / distance
            dy_norm = dy / distance

            # Perpendicular vector (for arrow branches)
            perp_x = -dy_norm
            perp_y = dx_norm

            # Arrow head size
            arrow_length = distance * 0.1  # 10% of trajectory length
            arrow_width = arrow_length * 0.4  # Width of the V

            # Calculate two points for the arrow V
            left_x = end_x - dx_norm * arrow_length + perp_x * arrow_width
            left_y = end_y - dy_norm * arrow_length + perp_y * arrow_width

            right_x = end_x - dx_norm * arrow_length - perp_x * arrow_width
            right_y = end_y - dy_norm * arrow_length - perp_y * arrow_width

            # === Add left line of arrow V ===
            fig.add_trace(go.Scatter(
                x=[left_x, end_x],
                y=[left_y, end_y],
                mode='lines',
                line=dict(color=color, width=width),
                showlegend=False,
                hoverinfo='skip'
            ))

            # === Add right line of arrow V ===
            fig.add_trace(go.Scatter(
                x=[right_x, end_x],
                y=[right_y, end_y],
                mode='lines',
                line=dict(color=color, width=width),
                showlegend=False,
                hoverinfo='skip'
            ))

            return fig

        # === CASE 0: No lines ===
        if line_strategy.lower() == "none":
            return fig

        # === CASE 1: Sequential line (all points in order) ===
        if line_strategy.lower() == "sequential":
            x_vals = scores[pc_x].values
            y_vals = scores[pc_y].values

            # Determine coloring strategy
            if color_by_index:
                # Use sequential index: 1 to N
                n = len(x_vals)
                indices = np.arange(1, n + 1)
                colors = [get_continuous_color_for_value(idx, 1, n) for idx in indices]
            elif color_variable is not None and original_data is not None and color_variable in original_data.columns:
                # Use variable values
                variable_vals = original_data.loc[scores.index, color_variable].values
                min_val = np.nanmin(variable_vals)
                max_val = np.nanmax(variable_vals)
                colors = [get_continuous_color_for_value(v, min_val, max_val) for v in variable_vals]
            else:
                # Default gray
                colors = ['rgba(128, 128, 128, 0.7)'] * len(x_vals)

            # Draw line segments with gradient
            for i in range(len(x_vals) - 1):
                line_trace = go.Scatter(
                    x=[x_vals[i], x_vals[i+1]],
                    y=[y_vals[i], y_vals[i+1]],
                    mode='lines',
                    line=dict(color=colors[i], width=line_width),
                    opacity=line_opacity,
                    name='Trajectory' if i == 0 else None,
                    showlegend=(i == 0),
                    hoverinfo='skip',
                    legendgroup='trajectory'
                )
                fig.add_trace(line_trace)

            # === Add arrow at the end of sequential trajectory ===
            if show_trajectory_arrow and len(x_vals) >= 2:
                # Get last two points to define arrow direction
                prev_x = x_vals[-2]
                prev_y = y_vals[-2]
                last_x = x_vals[-1]
                last_y = y_vals[-1]
                last_color = colors[-1]  # Use color of last segment

                fig = add_trajectory_arrow(
                    fig,
                    prev_x, prev_y,  # Start point
                    last_x, last_y,  # End point (arrow tip)
                    last_color,      # Arrow color
                    size=12,
                    width=line_width
                )

            return fig

        # === CASE 2: Categorical trajectories (one line per group) ===
        if line_strategy.lower() == "categorical":
            # Validation
            if groupby_column is None:
                raise ValueError(
                    "line_strategy='categorical' requires groupby_column parameter. "
                    "Please select a metadata column for grouping."
                )

            # Prepare groupby data
            if isinstance(groupby_column, pd.Series):
                group_series = groupby_column.copy()
            else:
                group_series = pd.Series(groupby_column, index=scores.index)

            # Align with scores index
            group_series = group_series.reindex(scores.index)

            # Get unique categories
            unique_categories = group_series.dropna().unique()

            if len(unique_categories) == 0:
                # No valid categories found
                return fig

            # === NEW: Prepare category color mapping if metadata provided ===
            category_color_map = None
            if use_category_colors and metadata_column is not None:
                # Prepare metadata series
                if isinstance(metadata_column, pd.Series):
                    meta_series = metadata_column.copy()
                else:
                    meta_series = pd.Series(metadata_column, index=scores.index)

                # Align with scores index
                meta_series = meta_series.reindex(scores.index)

                # Get unique values from metadata
                unique_meta_values = meta_series.dropna().unique()

                # Create categorical color map using color_utils
                category_color_map = create_categorical_color_map(unique_meta_values)

            # Draw line for each category
            for category in unique_categories:
                # Get mask for this category
                category_mask = group_series == category
                n_points_in_category = category_mask.sum()

                # Skip categories with fewer than 2 points (can't make a line)
                if n_points_in_category < 2:
                    continue

                # Get indices for this category
                category_indices = scores.index[category_mask]

                # Extract coordinates for this category
                x_vals = scores.loc[category_mask, pc_x].values
                y_vals = scores.loc[category_mask, pc_y].values

                # Handle any NaN values
                valid_mask = ~(np.isnan(x_vals) | np.isnan(y_vals))
                x_vals = x_vals[valid_mask]
                y_vals = y_vals[valid_mask]
                category_indices = category_indices[valid_mask]

                # Skip if not enough valid points
                if len(x_vals) < 2:
                    continue

                # [NEW] Determine if this is the selected batch
                is_selected_batch = (trajectory_color_vector and
                                    trajectory_color_vector.get('category') == category)

                # [NEW] Set opacity: Full for selected batch, dim for others
                # Selected batch: use user-controlled line_opacity (default 0.6)
                # Other batches: use fixed 0.15 (very dim, almost invisible)
                current_opacity = line_opacity if is_selected_batch else 0.15

                # === NEW: Determine coloring for this category ===
                if use_category_colors and category_color_map is not None and metadata_column is not None:
                    # Get metadata values for this category's points
                    meta_vals = meta_series.loc[category_indices].values

                    # Map each metadata value to its color
                    colors = [category_color_map.get(val, 'rgb(128, 128, 128)') for val in meta_vals]

                    # Use full opacity for category-colored trajectories
                    current_opacity = line_opacity

                elif is_selected_batch and trajectory_color_vector.get('mode') == 'numeric_color':
                    # Selected batch with numeric variable coloring (BRIGHT & PROMINENT)
                    color_vals = trajectory_color_vector['values'].reindex(category_indices).values
                    min_val = trajectory_color_vector['min']
                    max_val = trajectory_color_vector['max']
                    colors = [get_continuous_color_for_value(v, min_val, max_val) for v in color_vals]

                elif is_selected_batch and trajectory_color_vector.get('mode') == 'sequential_bright':
                    # [NEW] Selected batch with sequential index (BRIGHT)
                    # Use sequential coloring: 1→N within this batch
                    n = len(x_vals)
                    colors = [get_continuous_color_for_value(idx, 1, n) for idx in range(1, n + 1)]

                elif color_by_index:
                    # Other batches: Use gray background (DIM & BACKGROUND)
                    colors = ['rgb(200, 200, 200)'] * len(x_vals)

                elif color_variable is not None and original_data is not None and color_variable in original_data.columns:
                    # Other batches with variable coloring: Still use gray for consistency
                    colors = ['rgb(200, 200, 200)'] * len(x_vals)

                else:
                    # Fallback: use gray (dim background)
                    colors = ['rgb(200, 200, 200)'] * len(x_vals)

                # Draw line segments with appropriate opacity
                last_x = None
                last_y = None
                last_color = None

                for i in range(len(x_vals) - 1):
                    line_trace = go.Scatter(
                        x=[x_vals[i], x_vals[i+1]],
                        y=[y_vals[i], y_vals[i+1]],
                        mode='lines',
                        line=dict(color=colors[i], width=line_width),
                        opacity=current_opacity,  # ← [CHANGED] Dimmed or full opacity
                        name=str(category) if i == 0 else None,
                        showlegend=(i == 0),
                        hoverinfo='skip',
                        legendgroup=f'trajectory_{category}'
                    )
                    fig.add_trace(line_trace)

                    # Store last point for arrow
                    last_x = x_vals[i+1]
                    last_y = y_vals[i+1]
                    last_color = colors[i]

                # === NEW: Add arrow at the end of trajectory ===
                if show_trajectory_arrow and last_x is not None and last_y is not None and last_color is not None:
                    if len(x_vals) >= 2:
                        # Get second-to-last point to define arrow direction
                        prev_x = x_vals[-2]
                        prev_y = y_vals[-2]

                        fig = add_trajectory_arrow(
                            fig,
                            prev_x, prev_y,  # Start point
                            last_x, last_y,  # End point (arrow tip)
                            last_color,      # Arrow color
                            size=12,
                            width=line_width
                        )

            return fig

        # Unknown strategy
        return fig

    except Exception as e:
        # Log error and return original figure
        print(f"Error in add_sample_trajectory_lines: {str(e)}")
        return fig


def plot_varimax_component_selector(
    explained_variance_ratio: np.ndarray,
    cumulative_variance: np.ndarray
) -> go.Figure:
    """
    Create interactive screeplot for Varimax component selection.

    Shows both individual and cumulative variance to help user select optimal number of factors.

    Parameters
    ----------
    explained_variance_ratio : np.ndarray
        Array of variance explained ratios (0-1 scale).
    cumulative_variance : np.ndarray
        Array of cumulative variance (0-1 scale).

    Returns
    -------
    go.Figure
        Plotly Figure with interactive screeplot.
    """
    n_components = len(explained_variance_ratio)
    component_labels = [f'PC{i+1}' for i in range(n_components)]

    fig = go.Figure()

    # Add individual variance bars
    fig.add_trace(go.Bar(
        x=component_labels,
        y=explained_variance_ratio * 100,
        name='Individual Variance',
        marker_color='lightblue',
        yaxis='y'
    ))

    # Add cumulative variance line
    fig.add_trace(go.Scatter(
        x=component_labels,
        y=cumulative_variance * 100,
        mode='lines+markers',
        name='Cumulative Variance',
        line=dict(color='red', width=3),
        marker=dict(size=10),
        yaxis='y2'
    ))

    # Add 80% and 95% reference lines
    fig.add_hline(y=80, line_dash="dash", line_color="orange",
                  annotation_text="80%", yaxis='y2')
    fig.add_hline(y=95, line_dash="dash", line_color="green",
                  annotation_text="95%", yaxis='y2')

    fig.update_layout(
        title="Component Selection for Varimax Rotation<br><sub>Use slider below to select number of factors</sub>",
        xaxis_title="Principal Component",
        yaxis=dict(title="Individual Variance (%)", side='left'),
        yaxis2=dict(title="Cumulative Variance (%)", side='right', overlaying='y', range=[0, 105]),
        height=500,
        hovermode='x unified'
    )

    return fig


def plot_line_scores(
    scores: pd.DataFrame,
    pc_names: List[str],
    data: Optional[pd.DataFrame] = None,
    color_by: str = "None",
    encode_by: str = "None",
    show_labels: str = "None",
    label_source: Optional[pd.DataFrame] = None
) -> go.Figure:
    """
    Line plot of PCA scores with categorical coloring and segment separators.

    Draws line segments colored by category with gray dashed connectors between periods.
    Similar to MATLAB scoresprofiliperiodi.m

    Parameters
    ----------
    scores : pd.DataFrame
        PCA scores with PC columns
    pc_names : List[str]
        List of PC names to plot (e.g., ['PC1', 'PC2'])
    data : pd.DataFrame, optional
        Original data for color_by and encode_by columns
    color_by : str, optional
        Column for coloring segments. Default 'None'
    encode_by : str, optional
        Column for segment grouping (usually same as color_by). Default 'None'
    show_labels : str, optional
        Column to show as labels on points. Default 'None'
    label_source : pd.DataFrame, optional
        DataFrame with label data. Default None

    Returns
    -------
    go.Figure
        Plotly line plot with colored segments and optional labels
    """

    fig = go.Figure()
    sample_index = np.arange(1, len(scores) + 1)

    # Get color mapping from color_utils
    color_map = None
    if encode_by != 'None' and data is not None and encode_by in data.columns:
        unique_cats = data[encode_by].dropna().unique()
        color_map = create_categorical_color_map(unique_cats)

    # Plot each PC
    for pc_name in pc_names:
        pc_idx = int(pc_name.replace('PC', '')) - 1
        pc_scores = scores.iloc[:, pc_idx].values

        # Determine marker symbol based on PC index
        marker_symbol = ['circle', 'square', 'diamond'][pc_idx % 3]

        # If encode_by set: draw segments grouped by category
        if encode_by != 'None' and data is not None and encode_by in data.columns:
            encode_values = data[encode_by].values

            # Find segment boundaries
            segments = []
            current_start = 0

            for i in range(1, len(encode_values)):
                if encode_values[i] != encode_values[i-1]:
                    segments.append((current_start, i, encode_values[current_start]))
                    current_start = i
            segments.append((current_start, len(encode_values), encode_values[current_start]))

            # Draw each segment
            for start_idx, end_idx, segment_cat in segments:
                x_seg = sample_index[start_idx:end_idx]
                y_seg = pc_scores[start_idx:end_idx]

                # Get color from map
                seg_color = color_map.get(segment_cat, 'gray') if color_map else 'blue'

                # Prepare labels if requested
                text_labels = None
                if show_labels != 'None' and label_source is not None and show_labels in label_source.columns:
                    text_labels = label_source[show_labels].iloc[start_idx:end_idx].astype(str).values

                # Draw segment with optional labels
                fig.add_trace(go.Scatter(
                    x=x_seg,
                    y=y_seg,
                    name=f"{pc_name}: {segment_cat}",
                    mode='lines+markers+text' if text_labels is not None else 'lines+markers',
                    line=dict(color=seg_color, width=2, dash='solid'),
                    marker=dict(size=6, symbol=marker_symbol),
                    text=text_labels,
                    textposition="top center",
                    textfont=dict(size=8),
                    hovertemplate=f'{pc_name}<br>Sample: %{{x}}<br>Score: %{{y:.3f}}<br>{encode_by}: {segment_cat}<extra></extra>'
                ))

                # Add connector line (gray dashed) between segments
                if end_idx < len(sample_index):
                    connector_x = [sample_index[end_idx-1], sample_index[end_idx]]
                    connector_y = [pc_scores[end_idx-1], pc_scores[end_idx]]

                    fig.add_trace(go.Scatter(
                        x=connector_x,
                        y=connector_y,
                        mode='lines',
                        line=dict(color='gray', dash='dot', width=1),
                        showlegend=False,
                        hoverinfo='skip'
                    ))

        else:
            # No grouping: draw entire line
            if color_by == 'None':
                # Single color
                fig.add_trace(go.Scatter(
                    x=sample_index,
                    y=pc_scores,
                    name=pc_name,
                    mode='lines+markers',
                    line=dict(color='blue', width=2),
                    marker=dict(size=5, symbol=marker_symbol),
                    hovertemplate=f'{pc_name}<br>Sample: %{{x}}<br>Score: %{{y:.3f}}<extra></extra>'
                ))

            elif color_by == 'Index':
                # Gradient by index
                fig.add_trace(go.Scatter(
                    x=sample_index,
                    y=pc_scores,
                    name=pc_name,
                    mode='lines+markers',
                    line=dict(width=2),
                    marker=dict(
                        size=5,
                        symbol=marker_symbol,
                        color=sample_index,
                        colorscale='Blues',
                        showscale=True,
                        colorbar=dict(title="Index", x=1.02)
                    ),
                    hovertemplate=f'{pc_name}<br>Sample: %{{x}}<br>Score: %{{y:.3f}}<extra></extra>'
                ))

            elif color_by in data.columns:
                # Color by column
                if not is_quantitative_variable(data[color_by]):
                    # Categorical: use color_utils map
                    unique_cats = data[color_by].dropna().unique()
                    color_map_by = create_categorical_color_map(unique_cats)
                    color_vals = [color_map_by.get(cat, 'gray') for cat in data[color_by]]

                    fig.add_trace(go.Scatter(
                        x=sample_index,
                        y=pc_scores,
                        name=pc_name,
                        mode='lines+markers',
                        line=dict(width=2),
                        marker=dict(size=5, symbol=marker_symbol, color=color_vals),
                        hovertemplate=f'{pc_name}<br>Sample: %{{x}}<br>Score: %{{y:.3f}}<extra></extra>'
                    ))
                else:
                    # Quantitative: blue-to-red gradient
                    fig.add_trace(go.Scatter(
                        x=sample_index,
                        y=pc_scores,
                        name=pc_name,
                        mode='lines+markers',
                        line=dict(width=2),
                        marker=dict(
                            size=5,
                            symbol=marker_symbol,
                            color=data[color_by].values,
                            colorscale=[(0.0, 'rgb(0, 0, 255)'), (0.5, 'rgb(128, 0, 128)'), (1.0, 'rgb(255, 0, 0)')],
                            showscale=True,
                            colorbar=dict(title=color_by, x=1.02)
                        ),
                        hovertemplate=f'{pc_name}<br>Sample: %{{x}}<br>Score: %{{y:.3f}}<extra></extra>'
                    ))

    # Add zero line
    fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5)

    title_text = "PCA Scores Over Sample Sequence"
    if color_by != 'None' or encode_by != 'None':
        title_text += f"<br><sub>Color: {color_by} | Segments: {encode_by}</sub>"

    fig.update_layout(
        title=title_text,
        xaxis_title="Sample Index",
        yaxis_title="Score Value",
        hovermode='x unified',
        height=500,
        template='plotly_white'
    )

    return fig


def plot_loadings_scores_side_by_side(
    loadings: pd.DataFrame,
    scores: pd.DataFrame,
    pc_x: str,
    pc_y: str,
    explained_variance_ratio: np.ndarray,
    sample_colors: Optional[Dict[int, str]] = None,
    sample_labels: Optional[Dict[int, str]] = None,
    variable_annotations: Optional[Dict[str, str]] = None,
    arrow_scale: float = 1.0,
    # === NEW PARAMETERS ===
    color_data: Optional[pd.Series] = None,
    text_labels: Optional[Union[pd.Series, List[str]]] = None,
    show_convex_hull: bool = False,
    hull_opacity: float = 0.2,
    trajectory_strategy: str = "None",
    trajectory_groupby_column: Optional[pd.Series] = None,
    trajectory_metadata_for_coloring: Optional[pd.Series] = None,
    trajectory_color_vector: Optional[Dict] = None,
    trajectory_width: int = 2,
    trajectory_opacity: float = 0.6,
    trajectory_color_by_index: bool = True,
    trajectory_color_variable: Optional[str] = None,
    marker_size: int = 8
) -> go.Figure:
    """
    Create side-by-side subplots for joint loadings-scores interpretation.

    Parameters
    ----------
    loadings : pd.DataFrame
        Loading matrix with variables as rows, PCs as columns
    scores : pd.DataFrame
        Score matrix with samples as rows, PCs as columns
    pc_x : str
        X-axis principal component name (e.g., 'PC1')
    pc_y : str
        Y-axis principal component name (e.g., 'PC2')
    explained_variance_ratio : np.ndarray
        Array of variance explained ratios
    sample_colors : dict, optional
        Sample index to hex color mapping {sample_idx: '#RRGGBB'}
    sample_labels : dict, optional
        Sample index to label text mapping {sample_idx: 'label'}
    variable_annotations : dict, optional
        Variable name to interpretation notes mapping {var_name: 'notes'}
    arrow_scale : float, optional
        Scaling factor for loading arrow lengths (default: 1.0)

    Returns
    -------
    go.Figure
        Plotly Figure with 2 subplots (loadings + scores)

    Examples
    --------
    >>> loadings_df = pd.DataFrame({'PC1': [0.8, 0.2], 'PC2': [0.1, 0.9]}, index=['var1', 'var2'])
    >>> scores_df = pd.DataFrame({'PC1': [1, 2], 'PC2': [3, 4]}, index=[0, 1])
    >>> var_ratio = np.array([0.6, 0.3])
    >>> fig = plot_loadings_scores_side_by_side(loadings_df, scores_df, 'PC1', 'PC2', var_ratio)
    """
    from plotly.subplots import make_subplots

    # Get component indices and variance
    pc_cols = loadings.columns.tolist()
    pc_x_idx = pc_cols.index(pc_x)
    pc_y_idx = pc_cols.index(pc_y)
    var_x = explained_variance_ratio[pc_x_idx] * 100
    var_y = explained_variance_ratio[pc_y_idx] * 100

    # Create subplot figure
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            f"Loadings: {pc_x} vs {pc_y}",
            f"Scores: {pc_x} vs {pc_y}"
        ),
        horizontal_spacing=0.12
    )

    # ========== LEFT SUBPLOT: LOADINGS ==========

    # Add loading arrows
    for var_name in loadings.index:
        x_load = loadings.loc[var_name, pc_x] * arrow_scale
        y_load = loadings.loc[var_name, pc_y] * arrow_scale

        # Annotation text
        annotation_text = variable_annotations.get(var_name, 'N/A') if variable_annotations else 'N/A'

        # Arrow line (from origin to loading point) - ROSSO PURO
        fig.add_trace(
            go.Scatter(
                x=[0, x_load],
                y=[0, y_load],
                mode='lines',
                line=dict(color='rgb(220, 20, 60)', width=1),  # Rosso crimson, sottile
                showlegend=False,
                hoverinfo='text',
                hovertext=f"{var_name}<br>Loading {pc_x}: {loadings.loc[var_name, pc_x]:.3f}<br>Loading {pc_y}: {loadings.loc[var_name, pc_y]:.3f}<br>Annotation: {annotation_text}",
                name=var_name
            ),
            row=1, col=1
        )

        # Arrow head (Plotly annotation arrow) - ROSSO PURO
        fig.add_annotation(
            x=x_load,
            y=y_load,
            ax=0,
            ay=0,
            xref='x',
            yref='y',
            axref='x',
            ayref='y',
            showarrow=True,
            arrowhead=2,                      # Freccia standard Plotly
            arrowsize=1,
            arrowwidth=1,                     # Sottile
            arrowcolor='rgb(220, 20, 60)',    # Rosso crimson
            row=1, col=1
        )

        # Variable label at arrow endpoint
        fig.add_trace(
            go.Scatter(
                x=[x_load],
                y=[y_load],
                mode='text',
                text=[var_name],
                textposition='top center',
                textfont=dict(size=8, color='rgb(60, 60, 60)'),
                showlegend=False,
                hoverinfo='skip'
            ),
            row=1, col=1
        )

    # Add zero reference lines (loadings subplot)
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=1, col=1)
    fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5, row=1, col=1)

    # ========== RIGHT SUBPLOT: SCORES ==========

    # Prepare sample display data
    sample_indices = scores.index.tolist()
    score_x_vals = scores[pc_x].values
    score_y_vals = scores[pc_y].values

    # === NEW: Determine colors (priority: color_data > sample_colors > default) ===
    if color_data is not None:
        # Use new color_data parameter (flexible coloring)
        if is_quantitative_variable(color_data):
            # Quantitative: blue-to-red gradient
            color_values = color_data.reindex(scores.index).values
            point_colors = [get_continuous_color_for_value(v, color_data.min(), color_data.max())
                          for v in color_values]
        else:
            # Categorical: discrete color map
            unique_vals = color_data.dropna().unique()
            color_map = create_categorical_color_map(unique_vals)
            point_colors = [color_map.get(color_data.loc[idx], 'rgb(128, 128, 128)')
                          if idx in color_data.index else 'rgb(128, 128, 128)'
                          for idx in sample_indices]
    elif sample_colors:
        # Legacy parameter (backward compatibility)
        point_colors = [sample_colors.get(idx, 'rgb(100, 150, 200)') for idx in sample_indices]
    else:
        # Default: gradient coloring by index
        n_samples = len(sample_indices)
        point_colors = [get_continuous_color_for_value(i, 0, n_samples-1) for i in range(n_samples)]

    # === NEW: Determine text labels (priority: text_labels > sample_labels > default) ===
    if text_labels is not None:
        # Use new text_labels parameter
        if isinstance(text_labels, pd.Series):
            hover_labels = text_labels.reindex(scores.index).fillna('N/A').astype(str).tolist()
        else:
            hover_labels = [str(lbl) for lbl in text_labels]
    elif sample_labels:
        # Legacy parameter (backward compatibility)
        hover_labels = [sample_labels.get(idx, f"Sample {idx}") for idx in sample_indices]
    else:
        # Default: use sample indices
        hover_labels = [f"Sample {idx}" for idx in sample_indices]

    # Hover text
    hover_texts = [
        f"{label}<br>Score {pc_x}: {score_x_vals[i]:.3f}<br>Score {pc_y}: {score_y_vals[i]:.3f}"
        for i, label in enumerate(hover_labels)
    ]

    # Determine if we should show text labels on plot
    show_text_labels = text_labels is not None and (
        (isinstance(text_labels, pd.Series) and len(text_labels) > 0) or
        (isinstance(text_labels, list) and len(text_labels) > 0)
    )

    # Check if color_data is categorical
    is_categorical_color = color_data is not None and not is_quantitative_variable(color_data)

    if is_categorical_color:
        # === Scores scatter with LEGEND by category ===
        unique_categories = sorted(color_data.dropna().unique())
        color_map = create_categorical_color_map(unique_categories)

        for category in unique_categories:
            category_mask = (color_data == category)
            cat_indices = [i for i, idx in enumerate(sample_indices) if idx in scores.index[category_mask]]

            # Get data for this category
            cat_x = [score_x_vals[i] for i in cat_indices]
            cat_y = [score_y_vals[i] for i in cat_indices]
            cat_hover_texts = [hover_texts[i] for i in cat_indices]
            cat_hover_labels = [hover_labels[i] for i in cat_indices] if show_text_labels else None

            # Add trace for this category with legend
            fig.add_trace(
                go.Scatter(
                    x=cat_x,
                    y=cat_y,
                    mode='markers+text' if show_text_labels else 'markers',
                    marker=dict(
                        size=marker_size,
                        color=color_map.get(category, 'gray'),
                        opacity=0.7,
                        line=dict(color='white', width=1)
                    ),
                    text=cat_hover_labels if show_text_labels else None,
                    textposition='top center' if show_text_labels else None,
                    textfont=dict(
                        size=9,
                        color='rgba(40, 40, 40, 0.8)'
                    ) if show_text_labels else None,
                    hovertext=cat_hover_texts,
                    hoverinfo='text',
                    name=str(category),  # ← LEGENDA: Nome categoria
                    showlegend=True,      # ← LEGENDA: Mostra in legenda
                    legendgroup=str(category),
                ),
                row=1, col=2
            )
    else:
        # === Scores scatter WITHOUT legend (single color or quantitative) ===
        fig.add_trace(
            go.Scatter(
                x=score_x_vals,
                y=score_y_vals,
                mode='markers+text' if show_text_labels else 'markers',
                marker=dict(
                    size=marker_size,
                    color=point_colors,
                    opacity=0.7,
                    line=dict(color='white', width=1)
                ),
                text=hover_labels if show_text_labels else None,
                textposition='top center' if show_text_labels else None,
                textfont=dict(
                    size=9,
                    color='rgba(40, 40, 40, 0.8)'
                ) if show_text_labels else None,
                hovertext=hover_texts,
                hoverinfo='text',
                showlegend=False
            ),
            row=1, col=2
        )

    # === NEW: Add convex hulls if requested (only for categorical color_data) ===
    if show_convex_hull and color_data is not None and not is_quantitative_variable(color_data):
        try:
            # Get color map for convex hulls
            unique_vals = color_data.dropna().unique()
            color_map = create_categorical_color_map(unique_vals)

            # Add convex hulls for each category
            for category in unique_vals:
                category_mask = color_data == category
                n_points = category_mask.sum()

                if n_points < 3:
                    continue  # Need at least 3 points for a hull

                # Extract coordinates for this category
                cat_x = scores.loc[category_mask, pc_x].values
                cat_y = scores.loc[category_mask, pc_y].values
                cat_points = np.column_stack([cat_x, cat_y])

                try:
                    hull = ConvexHull(cat_points)
                    hull_vertices = hull.vertices
                    hull_points = cat_points[hull_vertices]

                    # Close the polygon
                    hull_x = np.append(hull_points[:, 0], hull_points[0, 0])
                    hull_y = np.append(hull_points[:, 1], hull_points[0, 1])

                    # Get color for this group
                    group_color = color_map.get(category, 'gray')

                    # Add hull trace to RIGHT subplot (row=1, col=2)
                    fig.add_trace(go.Scatter(
                        x=hull_x,
                        y=hull_y,
                        mode='lines',
                        line=dict(color=group_color, width=1),
                        opacity=hull_opacity,
                        fill=None,
                        name=f'{category}_hull',
                        showlegend=False,
                        hoverinfo='skip'
                    ), row=1, col=2)

                except Exception:
                    continue  # Skip if hull computation fails

        except Exception:
            pass  # Silently skip if convex hull fails

    # === NEW: Add trajectory lines if requested ===
    if trajectory_strategy.lower() != "none":
        try:
            # Create a temporary single-plot figure for trajectory computation
            temp_fig = go.Figure()

            # Add trajectory lines using the existing function
            temp_fig = add_sample_trajectory_lines(
                temp_fig,
                scores,
                pc_x,
                pc_y,
                line_strategy=trajectory_strategy,
                groupby_column=trajectory_groupby_column,
                line_width=trajectory_width,
                line_opacity=trajectory_opacity,
                color_by_index=trajectory_color_by_index,
                color_variable=trajectory_color_variable,
                trajectory_color_vector=trajectory_color_vector,
                metadata_column=trajectory_metadata_for_coloring,
                use_category_colors=(trajectory_metadata_for_coloring is not None)
            )

            # Transfer trajectory traces to RIGHT subplot (row=1, col=2)
            for trace in temp_fig.data:
                fig.add_trace(trace, row=1, col=2)

        except Exception:
            pass  # Silently skip if trajectory lines fail

    # Add zero reference lines (scores subplot)
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=1, col=2)
    fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5, row=1, col=2)

    # ========== LAYOUT CONFIGURATION ==========

    # Calculate symmetric axis ranges for loadings
    loading_x_range = [loadings[pc_x].min() * arrow_scale, loadings[pc_x].max() * arrow_scale]
    loading_y_range = [loadings[pc_y].min() * arrow_scale, loadings[pc_y].max() * arrow_scale]
    max_loading_abs = max(abs(min(loading_x_range + loading_y_range)), abs(max(loading_x_range + loading_y_range)))
    loading_axis_range = [-max_loading_abs * 1.2, max_loading_abs * 1.2]

    # Calculate symmetric axis ranges for scores
    score_x_range = [scores[pc_x].min(), scores[pc_x].max()]
    score_y_range = [scores[pc_y].min(), scores[pc_y].max()]
    max_score_abs = max(abs(min(score_x_range + score_y_range)), abs(max(score_x_range + score_y_range)))
    score_axis_range = [-max_score_abs * 1.2, max_score_abs * 1.2]

    # Update layout
    fig.update_layout(
        title=dict(
            text=f"Joint Loadings-Scores Analysis: {pc_x} vs {pc_y}",
            x=0.5,
            xanchor='center'
        ),
        height=500,
        showlegend=True,  # ← Enable legend for categories
        legend=dict(
            x=1.05,
            y=1.0,
            xanchor='left',
            yanchor='top',
            bgcolor='rgba(255, 255, 255, 0.9)',
            bordercolor='rgba(0, 0, 0, 0.2)',
            borderwidth=1
        ),
        template='plotly_white',
        hovermode='closest'
    )

    # Update axes for LEFT subplot (loadings)
    fig.update_xaxes(
        title=f"{pc_x} ({var_x:.1f}%)",
        range=loading_axis_range,
        scaleanchor="y",
        scaleratio=1,
        constrain="domain",
        row=1, col=1
    )
    fig.update_yaxes(
        title=f"{pc_y} ({var_y:.1f}%)",
        range=loading_axis_range,
        constrain="domain",
        row=1, col=1
    )

    # Update axes for RIGHT subplot (scores)
    fig.update_xaxes(
        title=f"{pc_x} ({var_x:.1f}%)",
        range=score_axis_range,
        scaleanchor="y2",
        scaleratio=1,
        constrain="domain",
        row=1, col=2
    )
    fig.update_yaxes(
        title=f"{pc_y} ({var_y:.1f}%)",
        range=score_axis_range,
        constrain="domain",
        row=1, col=2
    )

    return fig
