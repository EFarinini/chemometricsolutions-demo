"""
Bivariate Plotting Utilities
Interactive plotting functions for bivariate analysis using Plotly
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Optional, List, Dict
from color_utils import create_categorical_color_map, get_unified_color_schemes


def add_convex_hulls_to_figure(
    fig: go.Figure,
    data: pd.DataFrame,
    x_var: str,
    y_var: str,
    color_by: Optional[str] = None,
    color_map: Optional[dict] = None,
    hull_fill: bool = True,
    hull_opacity: float = 0.05,
    hull_line_style: str = 'dash',
    hull_line_width: int = 2
) -> go.Figure:
    """
    Add convex hulls to scatter plot for each categorical group.

    Parameters
    ----------
    fig : go.Figure
        Plotly figure with scatter plot
    data : pd.DataFrame
        Original data
    x_var : str
        X-axis variable name
    y_var : str
        Y-axis variable name
    color_by : str, optional
        Categorical column used for grouping
    color_map : dict, optional
        Mapping of categories to colors
    hull_fill : bool
        Fill the convex hull area (default True)
    hull_opacity : float
        Opacity of hull fill, 0-1 (default 0.05)
    hull_line_style : str
        Line style: 'solid', 'dash', 'dot', 'dashdot' (default 'dash')
    hull_line_width : int
        Width of hull boundary line (default 2)

    Returns
    -------
    go.Figure
        Figure with convex hulls added
    """
    from scipy.spatial import ConvexHull

    # If no color_by, can't create meaningful hulls
    if color_by is None:
        return fig

    # Get clean data
    plot_data = data[[x_var, y_var, color_by]].dropna()

    # Get unique categories
    categories = plot_data[color_by].unique()

    # Add hull for each category
    for category in categories:
        # Filter data for this category
        category_mask = plot_data[color_by] == category
        cat_data = plot_data[category_mask]

        if len(cat_data) < 3:
            # Need at least 3 points for convex hull
            continue

        # Extract X and Y coordinates
        points = cat_data[[x_var, y_var]].values

        try:
            # Compute convex hull
            hull = ConvexHull(points)
            hull_points = points[hull.vertices]

            # Close the hull by adding first point at end
            hull_points = np.vstack([hull_points, hull_points[0]])

            # Get color for this category
            if color_map and category in color_map:
                hull_color = color_map[category]
            else:
                hull_color = 'blue'

            # Add hull trace (lines connecting hull vertices)
            fig.add_trace(go.Scatter(
                x=hull_points[:, 0],
                y=hull_points[:, 1],
                mode='lines',
                name=f'{category} (hull)',
                line=dict(
                    color=hull_color,
                    width=hull_line_width,
                    dash=hull_line_style
                ),
                fill='toself' if hull_fill else 'none',
                fillcolor=hull_color,
                opacity=hull_opacity if hull_fill else 1,  # Always visible even without fill
                hovertemplate=None,
                showlegend=False,
                xaxis='x',
                yaxis='y'
            ))

        except Exception:
            # Convex hull computation failed (e.g., points are collinear)
            continue

    return fig


def create_scatter_plot(
    data: pd.DataFrame,
    x_var: str,
    y_var: str,
    color_by: Optional[str] = None,
    label_by: Optional[str] = None,
    custom_variables: Optional[dict] = None,
    point_size: int = 100,
    opacity: float = 0.7,
    title: Optional[str] = None,
    show_convex_hull: bool = False,
    hull_fill: bool = True,
    hull_opacity: float = 0.05,
    hull_line_style: str = 'dash',
    hull_line_width: int = 2
) -> go.Figure:
    """
    Create an interactive scatter plot with optional metadata coloring

    Parameters
    ----------
    data : pd.DataFrame
        Input dataset
    x_var : str
        Variable name for x-axis
    y_var : str
        Variable name for y-axis
    color_by : str, optional
        Metadata column for coloring points
    label_by : str, optional
        Metadata column for point labels
    custom_variables : dict, optional
        Dictionary of custom variables (metavariables) from session_state
    point_size : int
        Size of points (default 100)
    opacity : float
        Point opacity (0-1, default 0.7)
    title : str, optional
        Plot title
    show_convex_hull : bool
        Display convex hull boundaries around categorical groups (default False)
    hull_fill : bool
        Fill the convex hull area (default True)
    hull_opacity : float
        Opacity of hull fill, 0-1 (default 0.05)
    hull_line_style : str
        Line style: 'solid', 'dash', 'dot', 'dashdot' (default 'dash')
    hull_line_width : int
        Width of hull boundary line (default 2)

    Returns
    -------
    go.Figure
        Plotly figure object
    """
    # Clean data - remove NaN in x and y FIRST (before adding metadata)
    plot_data = data[[x_var, y_var]].copy()
    plot_data = plot_data.dropna(subset=[x_var, y_var])

    # Now that we have clean indices, add color_by
    if color_by:
        if color_by in data.columns:
            plot_data[color_by] = data.loc[plot_data.index, color_by]
        elif custom_variables and color_by in custom_variables:
            # Convert to Series with same index as original data
            color_array = custom_variables[color_by]

            # Create a Series with proper index matching
            if isinstance(color_array, pd.Series):
                # Already a Series, use as-is
                color_series = color_array
            elif hasattr(color_array, '__len__'):
                # Create Series with data index
                color_series = pd.Series(color_array, index=data.index)
            else:
                # Single value or unknown type
                color_by = None
                color_series = None

            if color_series is not None:
                # Only assign values for indices present in plot_data (after dropna)
                plot_data[color_by] = color_series.loc[plot_data.index]
        else:
            color_by = None  # Disable if not found

    # Add label_by (skip if "Index" - it will be handled later)
    if label_by and label_by != "Index":
        if label_by in data.columns:
            plot_data[label_by] = data.loc[plot_data.index, label_by]
        elif custom_variables and label_by in custom_variables:
            # Convert to Series with same index as original data
            label_array = custom_variables[label_by]

            # Create a Series with proper index matching
            if isinstance(label_array, pd.Series):
                # Already a Series, use as-is
                label_series = label_array
            elif hasattr(label_array, '__len__'):
                # Create Series with data index
                label_series = pd.Series(label_array, index=data.index)
            else:
                # Single value or unknown type
                label_by = None
                label_series = None

            if label_series is not None:
                # Only assign values for indices present in plot_data (after dropna)
                plot_data[label_by] = label_series.loc[plot_data.index]
        else:
            label_by = None  # Disable if not found

    if len(plot_data) == 0:
        # Return empty figure with message
        fig = go.Figure()
        fig.add_annotation(
            text="No data available after removing missing values",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig

    # === PREPARE TEXT LABELS FOR DISPLAY (PCA-style) ===
    # Determine if labels should be shown and what text to use
    show_text_labels = False
    text_labels_param = None

    if label_by is None or label_by == "None":
        # No labels
        show_text_labels = False
        text_labels_param = None

    elif label_by == "Index":
        # Show row indices
        show_text_labels = True
        text_labels_param = plot_data.index.astype(str)

    elif label_by in plot_data.columns:
        # Show column values
        show_text_labels = True
        text_labels_param = plot_data[label_by].astype(str)
    else:
        # Fallback: no labels
        show_text_labels = False
        text_labels_param = None

    # Hover labels always show full information (for tooltip)
    sample_indices = plot_data.index.tolist()
    hover_labels = []
    for i, idx in enumerate(sample_indices):
        if text_labels_param is not None:
            # Handle both Series and Index objects
            if isinstance(text_labels_param, pd.Series):
                label = text_labels_param.iloc[i]
            else:
                # It's an Index object, use direct indexing
                label = text_labels_param[i]
        else:
            label = str(idx)
        hover_labels.append(str(label) if label is not None else str(idx))

    # === BUILD HOVER TEXT ===
    hover_texts = []
    for i, label in enumerate(hover_labels):
        x_val = float(plot_data[x_var].iloc[i])
        y_val = float(plot_data[y_var].iloc[i])
        hover_text = f"{label}<br><b>{x_var}</b>: {x_val:.3f}<br><b>{y_var}</b>: {y_val:.3f}"
        hover_texts.append(hover_text)

    # Create figure
    if color_by and color_by in plot_data.columns:
        # Check if color variable is categorical or continuous
        unique_values = plot_data[color_by].dropna().unique()

        if len(unique_values) <= 20 or plot_data[color_by].dtype == 'object':
            # === CATEGORICAL COLORING ===
            color_map = create_categorical_color_map(unique_values)
            fig = go.Figure()

            # Build ONE trace per category with correct hovers and labels
            for category in sorted(unique_values):
                # Get indices for this category
                category_mask = (plot_data[color_by] == category)
                cat_indices = [i for i, idx in enumerate(plot_data.index) if idx in plot_data[category_mask].index]

                # Get data for this category
                cat_x = [plot_data[x_var].iloc[i] for i in cat_indices]
                cat_y = [plot_data[y_var].iloc[i] for i in cat_indices]
                cat_hover_texts = [hover_texts[i] for i in cat_indices]

                # Build text labels for this category (PCA-style)
                if show_text_labels and text_labels_param is not None:
                    # Extract labels for this category using same indices as data
                    cat_text_labels = [text_labels_param.iloc[i] if isinstance(text_labels_param, pd.Series)
                                       else text_labels_param[i]
                                       for i in cat_indices]
                    # Convert to simple int if numeric (like PCA)
                    cat_text_labels = [str(int(float(t))) if str(t).replace('.', '').replace('-', '').isdigit() else str(t)
                                       for t in cat_text_labels]
                else:
                    cat_text_labels = None

                # Add trace for this category
                fig.add_trace(go.Scatter(
                    x=cat_x,
                    y=cat_y,
                    mode='markers+text' if show_text_labels else 'markers',
                    marker=dict(
                        size=point_size / 10,
                        color=color_map.get(category, 'gray'),
                        opacity=opacity,
                        line=dict(color='white', width=0.5)
                    ),
                    text=cat_text_labels,  # Will be None if show_text_labels=False
                    textposition='top center',  # Above point (not overlapping)
                    textfont=dict(
                        size=9,
                        color='rgba(0, 0, 0, 0.6)',  # Semi-transparent black (like PCA)
                        family='Arial'
                    ),
                    hovertext=cat_hover_texts,
                    hoverinfo='text',
                    name=str(category),
                    showlegend=True,
                    legendgroup=color_by
                ))
        else:
            # === CONTINUOUS COLORING ===
            fig = px.scatter(
                plot_data,
                x=x_var,
                y=y_var,
                color=color_by,
                color_continuous_scale='Viridis',
                opacity=opacity
            )

            # Prepare text labels for continuous coloring (PCA-style conversion)
            if show_text_labels and text_labels_param is not None:
                if isinstance(text_labels_param, pd.Series):
                    text_vals = text_labels_param.values
                else:
                    text_vals = text_labels_param

                # Convert to simple int if numeric (like PCA lines 292-293)
                text_vals = [str(int(float(t))) if str(t).replace('.', '').replace('-', '').isdigit() else str(t)
                             for t in text_vals]
            else:
                text_vals = None

            # Update all traces
            fig.update_traces(
                mode='markers+text' if show_text_labels else 'markers',
                text=text_vals,  # Use converted text_vals
                textposition='top center',  # Above point (not overlapping)
                textfont=dict(
                    size=9,
                    color='rgba(0, 0, 0, 0.6)',  # Semi-transparent black
                    family='Arial'
                ),
                hovertext=hover_texts,
                hoverinfo='text',
                marker=dict(size=point_size / 10)
            )
    else:
        # === NO COLORING ===
        color_scheme = get_unified_color_schemes()
        fig = go.Figure()

        # Prepare text labels for no coloring (PCA-style conversion)
        if show_text_labels and text_labels_param is not None:
            if isinstance(text_labels_param, pd.Series):
                text_vals = text_labels_param.values
            else:
                text_vals = text_labels_param

            # Convert to simple int if numeric (like PCA)
            text_vals = [str(int(float(t))) if str(t).replace('.', '').replace('-', '').isdigit() else str(t)
                         for t in text_vals]
        else:
            text_vals = None

        fig.add_trace(go.Scatter(
            x=plot_data[x_var],
            y=plot_data[y_var],
            mode='markers+text' if show_text_labels else 'markers',
            marker=dict(
                size=point_size / 10,  # Scale down for plotly
                color=color_scheme['point_color'],
                opacity=opacity
            ),
            text=text_vals,  # Use converted text_vals
            textposition='top center',  # Above point (not overlapping)
            textfont=dict(
                size=9,
                color='rgba(0, 0, 0, 0.6)',  # Semi-transparent black
                family='Arial'
            ),
            hovertext=hover_texts,  # Use hovertext
            hoverinfo='text',  # Use 'text'
            name='Samples'
        ))

    # Update marker size if needed
    fig.update_traces(marker=dict(size=point_size / 10))

    # Set title
    if title is None:
        if color_by:
            # Explicit string conversion for color_by to avoid type issues
            title = f"{x_var} vs {y_var} | Colored by {str(color_by)}"
        else:
            title = f"{x_var} vs {y_var}"

    # CALCULATE AXIS RANGES WITH 5% PADDING
    x_data = plot_data[x_var].dropna()
    y_data = plot_data[y_var].dropna()

    x_min, x_max = x_data.min(), x_data.max()
    y_min, y_max = y_data.min(), y_data.max()

    # Calculate 5% padding
    x_padding = (x_max - x_min) * 0.05
    y_padding = (y_max - y_min) * 0.05

    # Set axis ranges with padding
    x_range = [x_min - x_padding, x_max + x_padding]
    y_range = [y_min - y_padding, y_max + y_padding]

    # ADD CONVEX HULL FIRST (IF REQUESTED)
    # This must be done BEFORE update_layout to prevent Plotly from recalculating axes
    if show_convex_hull and color_by and color_by in plot_data.columns:
        # Get color map for this plot
        unique_values = plot_data[color_by].dropna().unique()

        if len(unique_values) > 0:
            if len(unique_values) <= 20 or plot_data[color_by].dtype == 'object':
                # Categorical coloring
                color_map = create_categorical_color_map(unique_values)
            else:
                # For continuous, create a simple color map
                color_map = None

            # For convex hull, use plot_data (which has the custom variable if needed)
            fig = add_convex_hulls_to_figure(
                fig=fig,
                data=plot_data,  # Use plot_data instead of data
                x_var=x_var,
                y_var=y_var,
                color_by=color_by,
                color_map=color_map,
                hull_fill=hull_fill,
                hull_opacity=hull_opacity,
                hull_line_style=hull_line_style,
                hull_line_width=hull_line_width
            )

    # UPDATE LAYOUT LAST (AFTER CONVEX HULL)
    # This ensures axis ranges are the final settings and won't be overwritten
    color_scheme = get_unified_color_schemes()
    fig.update_layout(
        title=title,
        xaxis_title=x_var,
        yaxis_title=y_var,
        plot_bgcolor=color_scheme['background'],
        paper_bgcolor=color_scheme['paper'],
        font=dict(color=color_scheme['text']),
        hovermode='closest',
        width=700,
        height=700,  # Square size creates visually square plot
        xaxis=dict(
            showgrid=True,
            gridcolor=color_scheme['grid'],
            range=x_range,
            autorange=False  # Force use of explicit range
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor=color_scheme['grid'],
            range=y_range,
            autorange=False  # Force use of explicit range
        )
    )

    return fig


def create_pairs_plot(
    data: pd.DataFrame,
    variables: List[str],
    color_by: Optional[str] = None,
    opacity: float = 0.7
) -> go.Figure:
    """
    Create a pairs plot (scatter plot matrix) for multiple variables.
    Supports unlimited number of variables (will be large for >10 variables).

    Parameters
    ----------
    data : pd.DataFrame
        Input dataset
    variables : List[str]
        List of variable names to include
    color_by : str, optional
        Metadata column for coloring points
    opacity : float
        Point opacity (0-1, default 0.7)

    Returns
    -------
    go.Figure
        Plotly figure object with pairs plot
    """
    n_vars = len(variables)

    if n_vars < 2:
        raise ValueError("Need at least 2 variables for pairs plot")

    # No upper limit - Plotly will handle any number of subplots

    # Clean data
    plot_data = data[variables].copy()
    if color_by and color_by in data.columns:
        plot_data[color_by] = data[color_by]

    plot_data = plot_data.dropna()

    # Create color mapping if needed
    if color_by and color_by in plot_data.columns:
        unique_values = plot_data[color_by].unique()
        color_map = create_categorical_color_map(unique_values)
    else:
        color_map = None

    # Create subplots
    fig = make_subplots(
        rows=n_vars,
        cols=n_vars,
        shared_xaxes=False,  # Allow independent scaling per column
        shared_yaxes=False,  # Allow independent scaling per row
        vertical_spacing=0.02,
        horizontal_spacing=0.02
    )

    # Pre-calculate axis ranges for all variables (with 5% padding)
    axis_ranges = {}
    for var in variables:
        var_data = plot_data[var].dropna()
        if len(var_data) > 0:
            v_min, v_max = var_data.min(), var_data.max()
            padding = (v_max - v_min) * 0.05
            axis_ranges[var] = [v_min - padding, v_max + padding]
        else:
            axis_ranges[var] = None

    # Add scatter plots
    for i, var_y in enumerate(variables):
        for j, var_x in enumerate(variables):
            if i == j:
                # Diagonal - show histogram
                fig.add_trace(
                    go.Histogram(
                        x=plot_data[var_x],
                        name=var_x,
                        showlegend=False,
                        opacity=0.7
                    ),
                    row=i + 1,
                    col=j + 1
                )
            else:
                # Off-diagonal - scatter plot
                if color_by and color_map:
                    for category in unique_values:
                        mask = plot_data[color_by] == category
                        fig.add_trace(
                            go.Scatter(
                                x=plot_data.loc[mask, var_x],
                                y=plot_data.loc[mask, var_y],
                                mode='markers',
                                name=str(category),
                                marker=dict(
                                    color=color_map[category],
                                    size=5,
                                    opacity=opacity
                                ),
                                showlegend=(i == 0 and j == 1),
                                hovertemplate=f'{category}<br>{var_x}: %{{x:.3f}}<br>{var_y}: %{{y:.3f}}<extra></extra>'
                            ),
                            row=i + 1,
                            col=j + 1
                        )
                else:
                    color_scheme = get_unified_color_schemes()
                    fig.add_trace(
                        go.Scatter(
                            x=plot_data[var_x],
                            y=plot_data[var_y],
                            mode='markers',
                            marker=dict(
                                color=color_scheme['point_color'],
                                size=5,
                                opacity=opacity
                            ),
                            showlegend=False,
                            hovertemplate=f'{var_x}: %{{x:.3f}}<br>{var_y}: %{{y:.3f}}<extra></extra>'
                        ),
                        row=i + 1,
                        col=j + 1
                    )

            # === UPDATE AXES WITH AUTO-SCALED RANGES ===
            # Update X-axis
            if axis_ranges[var_x]:
                fig.update_xaxes(
                    range=axis_ranges[var_x],
                    row=i + 1,
                    col=j + 1
                )

            # Update Y-axis
            if axis_ranges[var_y]:
                fig.update_yaxes(
                    range=axis_ranges[var_y],
                    row=i + 1,
                    col=j + 1
                )

            # Update axes labels (only on edges)
            if j == 0:  # Left edge
                fig.update_yaxes(title_text=var_y, row=i + 1, col=j + 1)
            if i == n_vars - 1:  # Bottom edge
                fig.update_xaxes(title_text=var_x, row=i + 1, col=j + 1)

    # Update layout
    color_scheme = get_unified_color_schemes()
    fig.update_layout(
        title="Pairs Plot",
        plot_bgcolor=color_scheme['background'],
        paper_bgcolor=color_scheme['paper'],
        font=dict(color=color_scheme['text']),
        height=150 * n_vars,
        width=150 * n_vars
    )

    fig.update_xaxes(showgrid=True, gridcolor=color_scheme['grid'])
    fig.update_yaxes(showgrid=True, gridcolor=color_scheme['grid'])

    return fig


def create_correlation_heatmap(
    corr_matrix: pd.DataFrame,
    pval_matrix: Optional[pd.DataFrame] = None,
    significance_level: float = 0.05,
    title: str = "Correlation Matrix"
) -> go.Figure:
    """
    Create an interactive correlation heatmap

    Parameters
    ----------
    corr_matrix : pd.DataFrame
        Correlation matrix
    pval_matrix : pd.DataFrame, optional
        P-value matrix for significance marking
    significance_level : float
        Significance threshold (default 0.05)
    title : str
        Plot title

    Returns
    -------
    go.Figure
        Plotly figure object
    """
    # Prepare annotations
    annotations = []

    for i in range(len(corr_matrix.index)):
        for j in range(len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]

            # Format text
            text = f"{corr_val:.3f}"

            # Add significance marker if p-value available
            if pval_matrix is not None:
                pval = pval_matrix.iloc[i, j]
                if pval < significance_level:
                    text += "*"

            annotations.append(
                dict(
                    x=j,
                    y=i,
                    text=text,
                    showarrow=False,
                    font=dict(size=10, color='black' if abs(corr_val) < 0.5 else 'white')
                )
            )

    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        colorscale='RdBu_r',  # Red for negative, blue for positive
        zmid=0,
        zmin=-1,
        zmax=1,
        colorbar=dict(title="Correlation"),
        hovertemplate='%{y} vs %{x}<br>Correlation: %{z:.3f}<extra></extra>'
    ))

    fig.update_layout(
        title=title,
        annotations=annotations,
        xaxis=dict(side='bottom'),
        yaxis=dict(autorange='reversed'),
        width=max(500, 50 * len(corr_matrix.columns)),
        height=max(500, 50 * len(corr_matrix.index))
    )

    return fig
