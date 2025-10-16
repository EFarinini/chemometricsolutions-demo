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
            title=f"Scores: {pc_x} vs {pc_y}{title_suffix}<br>Total Explained Variance: {var_total:.1f}%",
            labels={'x': f'{pc_x} ({var_x:.1f}%)', 'y': f'{pc_y} ({var_y:.1f}%)'}
        )
    else:
        # Determine if color data is quantitative or categorical
        color_data = color_by if isinstance(color_by, pd.Series) else pd.Series(color_by, index=scores.index)

        if is_quantitative_variable(color_data):
            # Quantitative: use blue-to-red continuous scale
            fig = px.scatter(
                x=scores[pc_x],
                y=scores[pc_y],
                color=color_data,
                text=text_param,
                title=f"Scores: {pc_x} vs {pc_y} (colored by variable){title_suffix}<br>Total Explained Variance: {var_total:.1f}%",
                labels={'x': f'{pc_x} ({var_x:.1f}%)', 'y': f'{pc_y} ({var_y:.1f}%)', 'color': 'Value'},
                color_continuous_scale=[(0, 'blue'), (1, 'red')]
            )
        else:
            # Categorical: use discrete color map
            unique_values = color_data.dropna().unique()
            color_discrete_map = create_categorical_color_map(unique_values)

            fig = px.scatter(
                x=scores[pc_x],
                y=scores[pc_y],
                color=color_data,
                text=text_param,
                title=f"Scores: {pc_x} vs {pc_y} (colored by category){title_suffix}<br>Total Explained Variance: {var_total:.1f}%",
                labels={'x': f'{pc_x} ({var_x:.1f}%)', 'y': f'{pc_y} ({var_y:.1f}%)', 'color': 'Category'},
                color_discrete_map=color_discrete_map
            )

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

    # Configure text labels
    if show_labels:
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
    >>> loadings_df = pd.DataFrame({'PC1': [0.1, 0.2, 0.8], 'PC2': [0.9, 0.1, 0.1]})
    >>> fig = plot_loadings_line(loadings_df, ['PC1', 'PC2'])
    """
    title_suffix = " (Varimax)" if is_varimax else ""

    fig = go.Figure()

    for comp in selected_components:
        if comp in loadings.columns:
            fig.add_trace(go.Scatter(
                x=list(range(len(loadings.index))),
                y=loadings[comp],
                mode='lines+markers',
                name=comp,
                text=loadings.index,
                hovertemplate='Variable: %{text}<br>Loading: %{y:.3f}<extra></extra>'
            ))

    fig.update_layout(
        title=f"Loading Line Plot{title_suffix}",
        xaxis_title="Variable Index",
        yaxis_title="Loading Value",
        height=500,
        hovermode='x unified'
    )

    # Add zero reference line
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)

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
