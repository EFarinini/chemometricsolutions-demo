"""
Univariate Plots Module

Interactive Plotly visualizations for univariate analysis.
Integrates with color_utils.py for professional color palettes.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Optional, Dict, List, Union


def plot_histogram(
    data: Union[np.ndarray, pd.Series],
    column_name: str = "Column",
    n_bins: int = 30,
    color_palette: str = "viridis",
    title: Optional[str] = None,
    show_stats: bool = True
) -> go.Figure:
    """
    Interactive histogram with density curve overlay.

    Parameters
    ----------
    data : array-like
    column_name : str
        Name for labels
    n_bins : int
        Number of bins (default 30)
    color_palette : str
        Color palette name for integration with color_utils
    title : str, optional
    show_stats : bool
        Show mean/median lines

    Returns
    -------
    go.Figure
        Plotly figure object
    """
    data_clean = np.asarray(data).flatten()
    data_clean = data_clean[~np.isnan(data_clean)]

    if title is None:
        title = f"Histogram: {column_name}"

    fig = go.Figure()

    # Histogram
    fig.add_trace(go.Histogram(
        x=data_clean,
        nbinsx=n_bins,
        name='Data',
        marker_color='rgba(100, 150, 200, 0.7)',
        opacity=0.7
    ))

    # Add statistical lines
    if show_stats:
        mean_val = np.mean(data_clean)
        median_val = np.median(data_clean)

        fig.add_vline(
            x=mean_val,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Mean: {mean_val:.2f}",
            annotation_position="top right"
        )

        fig.add_vline(
            x=median_val,
            line_dash="dot",
            line_color="green",
            annotation_text=f"Median: {median_val:.2f}",
            annotation_position="top left"
        )

    fig.update_layout(
        title=title,
        xaxis_title=column_name,
        yaxis_title="Frequency",
        hovermode='x unified',
        template='plotly_white'
    )

    return fig


def plot_density(
    data: Union[np.ndarray, pd.Series],
    column_name: str = "Column",
    color: str = "blue",
    title: Optional[str] = None
) -> go.Figure:
    """
    Kernel density estimation plot.

    Parameters
    ----------
    data : array-like
    column_name : str
    color : str
        Plotly color name
    title : str, optional

    Returns
    -------
    go.Figure
    """
    data_clean = np.asarray(data).flatten()
    data_clean = data_clean[~np.isnan(data_clean)]

    if title is None:
        title = f"Density Plot: {column_name}"

    fig = px.density_contour(
        x=data_clean,
        title=title,
        labels={'x': column_name},
        marginal_x='histogram'
    )

    fig.update_layout(
        template='plotly_white',
        hovermode='x unified'
    )

    return fig


def plot_boxplot(
    data: Union[np.ndarray, pd.Series],
    column_name: str = "Column",
    color: str = "lightblue",
    title: Optional[str] = None,
    orientation: str = "v"
) -> go.Figure:
    """
    Boxplot with quartiles and outliers.

    Parameters
    ----------
    data : array-like
    column_name : str
    color : str
    title : str, optional
    orientation : str
        'v' for vertical, 'h' for horizontal

    Returns
    -------
    go.Figure
    """
    data_clean = np.asarray(data).flatten()
    data_clean = data_clean[~np.isnan(data_clean)]

    if title is None:
        title = f"Boxplot: {column_name}"

    if orientation == "v":
        fig = go.Figure(data=[go.Box(
            y=data_clean,
            name=column_name,
            marker_color=color,
            boxmean='sd'
        )])
        fig.update_layout(yaxis_title=column_name)
    else:
        fig = go.Figure(data=[go.Box(
            x=data_clean,
            name=column_name,
            marker_color=color,
            orientation='h',
            boxmean='sd'
        )])
        fig.update_layout(xaxis_title=column_name)

    fig.update_layout(
        title=title,
        template='plotly_white',
        hovermode='closest'
    )

    return fig


def plot_stripchart(
    data: Union[np.ndarray, pd.Series],
    column_name: str = "Column",
    color: str = "steelblue",
    title: Optional[str] = None,
    jitter: bool = True
) -> go.Figure:
    """
    Strip chart (scatter plot of all points).

    Parameters
    ----------
    data : array-like
    column_name : str
    color : str
    title : str, optional
    jitter : bool
        Add random jitter to x positions for visibility

    Returns
    -------
    go.Figure
    """
    data_clean = np.asarray(data).flatten()
    data_clean = data_clean[~np.isnan(data_clean)]

    if title is None:
        title = f"Strip Chart: {column_name}"

    x_pos = np.zeros_like(data_clean)
    if jitter:
        x_pos = np.random.normal(0, 0.04, len(data_clean))

    fig = go.Figure(data=[go.Scatter(
        x=x_pos,
        y=data_clean,
        mode='markers',
        marker=dict(
            size=8,
            color=color,
            opacity=0.6
        ),
        name=column_name,
        text=[f"{val:.3f}" for val in data_clean],
        hovertemplate='<b>%{text}</b><extra></extra>'
    )])

    fig.update_layout(
        title=title,
        yaxis_title=column_name,
        xaxis_title="",
        xaxis=dict(showticklabels=False),
        template='plotly_white',
        hovermode='closest',
        showlegend=False
    )

    return fig


def plot_eda_plot(
    data: Union[np.ndarray, pd.Series],
    column_name: str = "Column"
) -> go.Figure:
    """
    Exploratory Data Analysis plot (4-subplot: histogram, Q-Q, boxplot, density).

    Parameters
    ----------
    data : array-like
    column_name : str

    Returns
    -------
    go.Figure
    """
    data_clean = np.asarray(data).flatten()
    data_clean = data_clean[~np.isnan(data_clean)]

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Histogram", "Q-Q Plot", "Boxplot", "Density")
    )

    # Histogram
    fig.add_trace(
        go.Histogram(x=data_clean, name='Histogram', marker_color='steelblue'),
        row=1, col=1
    )

    # Q-Q plot approximation
    from scipy import stats
    sorted_data = np.sort(data_clean)
    theoretical_quantiles = stats.norm.ppf(
        np.linspace(0.01, 0.99, len(sorted_data))
    )
    fig.add_trace(
        go.Scatter(
            x=theoretical_quantiles,
            y=sorted_data,
            mode='markers',
            marker_color='green',
            name='Q-Q'
        ),
        row=1, col=2
    )

    # Boxplot
    fig.add_trace(
        go.Box(y=data_clean, name='Boxplot', marker_color='orange'),
        row=2, col=1
    )

    # Density (histogram as proxy)
    fig.add_trace(
        go.Histogram(x=data_clean, name='Density', nbinsx=20, marker_color='purple'),
        row=2, col=2
    )

    fig.update_xaxes(title_text=column_name, row=1, col=1)
    fig.update_xaxes(title_text="Theoretical", row=1, col=2)
    fig.update_xaxes(title_text="", row=2, col=1)
    fig.update_xaxes(title_text=column_name, row=2, col=2)

    fig.update_layout(
        title=f"EDA Plot: {column_name}",
        height=700,
        showlegend=False,
        template='plotly_white'
    )

    return fig


def plot_row_profiles(
    dataframe: pd.DataFrame,
    row_indices: Optional[List[int]] = None,
    title: str = "Analytical Profiles",
    color_per_sample: bool = False
) -> go.Figure:
    """
    Row profile plot (line plot of samples across columns).

    Parameters
    ----------
    dataframe : pd.DataFrame
    row_indices : list of int, optional
        Which rows to plot. If None, plot all rows.
    title : str
    color_per_sample : bool
        If True, each sample gets different color

    Returns
    -------
    go.Figure
    """
    if row_indices is None:
        row_indices = list(range(len(dataframe)))

    fig = go.Figure()

    for idx in row_indices:
        row_data = dataframe.iloc[idx].values

        if color_per_sample:
            color = px.colors.qualitative.Plotly[
                idx % len(px.colors.qualitative.Plotly)
            ]
        else:
            color = "steelblue"

        fig.add_trace(go.Scatter(
            x=dataframe.columns,
            y=row_data,
            mode='lines+markers',
            name=f"Sample {idx+1}",
            line=dict(color=color, width=2),
            hovertemplate='<b>%{x}</b><br>Value: %{y:.3f}<extra></extra>'
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Variables",
        yaxis_title="Values",
        template='plotly_white',
        hovermode='x unified',
        height=500,
        xaxis=dict(tickangle=-45)
    )

    return fig


def plot_row_profiles_colored(
    dataframe: pd.DataFrame,
    color_vector: Union[np.ndarray, pd.Series],
    row_indices: Optional[List[int]] = None,
    title: str = "Analytical Profiles (Colored)",
    colorscale: str = "Viridis"
) -> go.Figure:
    """
    Row profile plot with color gradient based on external variable.

    Parameters
    ----------
    dataframe : pd.DataFrame
    color_vector : array-like
        Values to map to colors (must be same length as dataframe)
    row_indices : list of int, optional
    title : str
    colorscale : str
        Plotly colorscale name

    Returns
    -------
    go.Figure
    """
    if row_indices is None:
        row_indices = list(range(len(dataframe)))

    color_vector = np.asarray(color_vector).flatten()

    # Normalize color vector to [0, 1]
    color_min = np.nanmin(color_vector)
    color_max = np.nanmax(color_vector)
    if color_max == color_min:
        normalized_colors = np.ones_like(color_vector) * 0.5
    else:
        normalized_colors = (color_vector - color_min) / (color_max - color_min)

    fig = go.Figure()

    for idx in row_indices:
        row_data = dataframe.iloc[idx].values
        norm_color = normalized_colors[idx]

        fig.add_trace(go.Scatter(
            x=dataframe.columns,
            y=row_data,
            mode='lines+markers',
            name=f"Sample {idx+1}",
            line=dict(
                color=f"rgba({int(255*norm_color)}, {int(100*(1-norm_color))}, 150, 0.8)",
                width=2
            ),
            hovertemplate=(
                f'<b>Sample {idx+1}</b><br>%{{x}}<br>Value: %{{y:.3f}}<br>'
                f'Color value: {color_vector[idx]:.3f}<extra></extra>'
            )
        ))

    # Add colorbar via dummy trace
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='markers',
        marker=dict(
            colorscale=colorscale,
            cmin=color_min,
            cmax=color_max,
            colorbar=dict(title="Color Value")
        ),
        hoverinfo='skip',
        showlegend=False
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Variables",
        yaxis_title="Values",
        template='plotly_white',
        hovermode='x unified',
        height=600,
        xaxis=dict(tickangle=-45)
    )

    return fig
