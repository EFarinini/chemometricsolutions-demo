"""
PLS Plotting Module

This module provides comprehensive visualizations for PLS regression models
using Plotly with color theme support from color_utils.

Key Features:
- RMSECV vs LV plot with optimal selection
- Predicted vs Observed plots (calibration and test)
- Residual plots with diagnostics
- Loading plots (bar and scatter)
- Regression coefficient plots
- Interactive Plotly charts with theme support

Author: ChemoMetric Solutions
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Any, Tuple
from scipy import stats
import color_utils
from .pls_calculations import calculate_metrics


def get_plot_colors(theme: str = 'light') -> Dict[str, str]:
    """
    Get color mapping for plots from unified color scheme.

    Parameters
    ----------
    theme : str
        Color theme (currently only 'light' supported)

    Returns
    -------
    Dict[str, str]
        Color mapping with keys: primary, primary_light, success, danger,
        warning, text, background, paper, grid, template
    """
    color_scheme = color_utils.get_unified_color_schemes()

    return {
        'primary': color_scheme['line_colors'][0],      # blue
        'primary_light': 'rgba(173, 216, 230, 0.3)',   # light blue with alpha
        'success': 'green',
        'danger': 'red',
        'warning': 'orange',
        'text': color_scheme['text'],                   # black
        'background': color_scheme['background'],       # white
        'paper': color_scheme['paper'],                 # white
        'grid': color_scheme['grid'],                   # #e6e6e6
        'template': 'plotly_white'
    }


def plot_rmsecv_vs_lv(cv_results: Dict[str, Any],
                      optimal_lv: int,
                      theme: str = 'light') -> go.Figure:
    """
    Plot RMSECV vs number of latent variables with optimal LV highlighted.

    Parameters
    ----------
    cv_results : Dict[str, Any]
        Results from repeated_kfold_cv()
    optimal_lv : int
        Optimal number of latent variables
    theme : str, optional
        Color theme ('light' or 'dark', default: 'light')

    Returns
    -------
    go.Figure
        Interactive Plotly figure

    Notes
    -----
    Shows:
    - RMSECV trend with error bars (if multiple repeats)
    - Optimal LV marked with vertical line and annotation
    - Elbow point visualization for LV selection
    """
    # Get data
    n_components = cv_results['n_components_range']
    rmsecv_mean = cv_results['RMSECV']
    rmsecv_std = cv_results['RMSECV_std']
    n_repeats = cv_results['n_repeats']

    # Calculate min/max from all repeats
    if 'RMSECV_all' in cv_results and cv_results['RMSECV_all'].ndim == 2:
        # RMSECV_all has shape (n_repeats, max_components)
        max_rmsecv = np.max(cv_results['RMSECV_all'], axis=0)
        min_rmsecv = np.min(cv_results['RMSECV_all'], axis=0)
    else:
        # Fallback to std dev if min/max not available
        max_rmsecv = rmsecv_mean + rmsecv_std
        min_rmsecv = rmsecv_mean - rmsecv_std

    # Get colors
    colors = get_plot_colors(theme)

    # Create figure
    fig = go.Figure()

    # Add min-max uncertainty band
    fig.add_trace(go.Scatter(
        x=np.concatenate([n_components, n_components[::-1]]),
        y=np.concatenate([max_rmsecv, min_rmsecv[::-1]]),
        fill='toself',
        fillcolor=colors['primary_light'],
        line=dict(color='rgba(255,255,255,0)'),
        customdata=np.column_stack([np.concatenate([min_rmsecv, max_rmsecv[::-1]]),
                                     np.concatenate([max_rmsecv, min_rmsecv[::-1]])]),
        hovertemplate='LV: %{x}<br>Min: %{customdata[0]:.4f}<br>Max: %{customdata[1]:.4f}<extra></extra>',
        showlegend=True,
        name='Min-Max Range'
    ))

    # Add mean RMSECV line
    fig.add_trace(go.Scatter(
        x=n_components,
        y=rmsecv_mean,
        mode='lines+markers',
        line=dict(color=colors['primary'], width=2),
        marker=dict(size=8, color=colors['primary']),
        name='Mean RMSECV',
        hovertemplate='LV: %{x}<br>RMSECV: %{y:.4f}<extra></extra>'
    ))

    # Highlight optimal LV
    optimal_idx = optimal_lv - 1
    if 0 <= optimal_idx < len(rmsecv_mean):
        fig.add_trace(go.Scatter(
            x=[optimal_lv],
            y=[rmsecv_mean[optimal_idx]],
            mode='markers',
            marker=dict(
                size=15,
                color=colors['success'],
                symbol='star',
                line=dict(color=colors['text'], width=2)
            ),
            name=f'Optimal (LV={optimal_lv})',
            hovertemplate=f'Optimal LV: {optimal_lv}<br>RMSECV: {rmsecv_mean[optimal_idx]:.4f}<extra></extra>'
        ))

        # Add vertical line at optimal LV
        fig.add_vline(
            x=optimal_lv,
            line_dash="dash",
            line_color=colors['success'],
            opacity=0.5,
            annotation_text=f"Optimal: {optimal_lv} LV",
            annotation_position="top"
        )

    # Update layout
    fig.update_layout(
        title=f"RMSECV vs Latent Variables (n={n_repeats} repeat{'s' if n_repeats > 1 else ''})",
        xaxis_title="Number of Latent Variables",
        yaxis_title="RMSECV",
        template=colors['template'],
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['paper'],
        font=dict(color=colors['text']),
        hovermode='closest',
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        )
    )

    # Add grid
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor=colors['grid'])
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor=colors['grid'])

    return fig


def plot_predictions_vs_observed(y_true: np.ndarray,
                                 y_pred: np.ndarray,
                                 title: str = "Predicted vs Observed",
                                 sample_names: Optional[List[str]] = None,
                                 theme: str = 'light') -> go.Figure:
    """
    Plot predicted vs observed values with 1:1 reference line.

    Parameters
    ----------
    y_true : np.ndarray
        True response values
    y_pred : np.ndarray
        Predicted response values
    title : str, optional
        Plot title
    sample_names : Optional[List[str]], optional
        Sample labels for hover info
    theme : str, optional
        Color theme ('light' or 'dark', default: 'light')

    Returns
    -------
    go.Figure
        Interactive Plotly figure

    Notes
    -----
    Shows:
    - Scatter plot of predictions vs observations
    - 1:1 reference line (perfect predictions)
    - R² and RMSE in annotation
    - Color-coded by residual magnitude (optional)
    """
    # Convert to numpy arrays
    y_true = np.asarray(y_true, dtype=np.float64).flatten()
    y_pred = np.asarray(y_pred, dtype=np.float64).flatten()

    # Calculate metrics
    metrics = calculate_metrics(y_true, y_pred)

    # Calculate residuals for coloring
    residuals = y_true - y_pred
    abs_residuals = np.abs(residuals)

    # Get colors
    colors = get_plot_colors(theme)

    # Generate sample names if not provided
    if sample_names is None:
        sample_names = [f"Sample {i+1}" for i in range(len(y_true))]

    # Create figure
    fig = go.Figure()

    # Add scatter points colored by residual magnitude
    fig.add_trace(go.Scatter(
        x=y_true,
        y=y_pred,
        mode='markers',
        marker=dict(
            size=8,
            color=abs_residuals,
            colorscale='Reds',
            showscale=True,
            colorbar=dict(
                title="Abs<br>Residual",
                x=1.15
            ),
            line=dict(color=colors['text'], width=0.5)
        ),
        name='Samples',
        text=sample_names,
        customdata=np.column_stack([residuals]),
        hovertemplate=(
            '<b>%{text}</b><br>' +
            'Observed: %{x:.3f}<br>' +
            'Predicted: %{y:.3f}<br>' +
            'Residual: %{customdata[0]:.3f}<br>' +
            '<extra></extra>'
        )
    ))

    # Add 1:1 reference line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    margin = (max_val - min_val) * 0.05

    fig.add_trace(go.Scatter(
        x=[min_val - margin, max_val + margin],
        y=[min_val - margin, max_val + margin],
        mode='lines',
        line=dict(color=colors['grid'], width=2, dash='dash'),
        name='1:1 Line',
        hoverinfo='skip'
    ))

    # Add regression line
    slope = metrics['slope']
    intercept = metrics['intercept']
    x_line = np.array([min_val - margin, max_val + margin])
    y_line = slope * x_line + intercept

    fig.add_trace(go.Scatter(
        x=x_line,
        y=y_line,
        mode='lines',
        line=dict(color=colors['primary'], width=2),
        name=f'Fit (slope={slope:.3f})',
        hoverinfo='skip'
    ))

    # Add metrics annotation
    annotation_text = (
        f"<b>Performance Metrics</b><br>"
        f"R² = {metrics['R2']:.4f}<br>"
        f"RMSE = {metrics['RMSE']:.4f}<br>"
        f"MAE = {metrics['MAE']:.4f}<br>"
        f"Bias = {metrics['Bias']:.4f}"
    )

    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.02, y=0.98,
        text=annotation_text,
        showarrow=False,
        bgcolor=colors['background'],
        bordercolor=colors['grid'],
        borderwidth=1,
        font=dict(size=10, color=colors['text']),
        align='left',
        xanchor='left',
        yanchor='top'
    )

    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title="Observed Values",
        yaxis_title="Predicted Values",
        template=colors['template'],
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['background'],
        font=dict(color=colors['text']),
        hovermode='closest',
        showlegend=True,
        legend=dict(
            yanchor="bottom",
            y=0.02,
            xanchor="right",
            x=0.98
        )
    )

    # Update axes
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor=colors['grid'],
        zeroline=True,
        zerolinecolor=colors['grid']
    )
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor=colors['grid'],
        zeroline=True,
        zerolinecolor=colors['grid'],
        scaleanchor="x",
        scaleratio=1
    )

    return fig


def plot_residuals(y_true: np.ndarray,
                  y_pred: np.ndarray,
                  sample_names: Optional[List[str]] = None,
                  theme: str = 'light') -> go.Figure:
    """
    Create residual diagnostic plots (multi-panel).

    Parameters
    ----------
    y_true : np.ndarray
        True response values
    y_pred : np.ndarray
        Predicted response values
    sample_names : Optional[List[str]], optional
        Sample labels for hover info
    theme : str, optional
        Color theme ('light' or 'dark', default: 'light')

    Returns
    -------
    go.Figure
        Interactive Plotly figure with subplots:
        1. Residuals vs Predicted
        2. Residuals vs Sample Index
        3. Histogram of Residuals
        4. Q-Q plot

    Notes
    -----
    Used to diagnose:
    - Heteroscedasticity (non-constant variance)
    - Systematic bias
    - Outliers
    - Normality of residuals
    """
    # Convert to numpy arrays
    y_true = np.asarray(y_true, dtype=np.float64).flatten()
    y_pred = np.asarray(y_pred, dtype=np.float64).flatten()

    # Calculate residuals
    residuals = y_true - y_pred
    n_samples = len(residuals)

    # Standardized residuals
    residual_std = np.std(residuals, ddof=1)
    std_residuals = residuals / residual_std if residual_std > 1e-10 else residuals

    # Get colors
    colors = get_plot_colors(theme)

    # Generate sample names if not provided
    if sample_names is None:
        sample_names = [f"Sample {i+1}" for i in range(n_samples)]

    # Create subplots (2x2 grid)
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Residuals vs Predicted',
            'Histogram of Residuals',
            'Residuals vs Sample Order',
            'Q-Q Plot'
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.10
    )

    # 1. Residuals vs Predicted
    fig.add_trace(
        go.Scatter(
            x=y_pred,
            y=residuals,
            mode='markers',
            marker=dict(
                size=6,
                color=colors['primary'],
                line=dict(color=colors['text'], width=0.5)
            ),
            text=sample_names,
            hovertemplate=(
                '<b>%{text}</b><br>' +
                'Predicted: %{x:.3f}<br>' +
                'Residual: %{y:.3f}<br>' +
                '<extra></extra>'
            ),
            showlegend=False
        ),
        row=1, col=1
    )

    # Add zero line
    fig.add_hline(
        y=0, line_dash="dash", line_color=colors['grid'],
        row=1, col=1
    )

    # Add ±2 std lines
    fig.add_hline(
        y=2*residual_std, line_dash="dot", line_color=colors['warning'],
        row=1, col=1, opacity=0.5
    )
    fig.add_hline(
        y=-2*residual_std, line_dash="dot", line_color=colors['warning'],
        row=1, col=1, opacity=0.5
    )

    # 2. Histogram of Residuals
    fig.add_trace(
        go.Histogram(
            x=residuals,
            nbinsx=20,
            marker=dict(
                color=colors['primary'],
                line=dict(color=colors['text'], width=0.5)
            ),
            showlegend=False,
            hovertemplate='Range: %{x}<br>Count: %{y}<extra></extra>'
        ),
        row=1, col=2
    )

    # Add normal distribution overlay
    x_norm = np.linspace(residuals.min(), residuals.max(), 100)
    y_norm = stats.norm.pdf(x_norm, loc=np.mean(residuals), scale=residual_std)
    # Scale to histogram
    bin_width = (residuals.max() - residuals.min()) / 20
    y_norm_scaled = y_norm * len(residuals) * bin_width

    fig.add_trace(
        go.Scatter(
            x=x_norm,
            y=y_norm_scaled,
            mode='lines',
            line=dict(color=colors['danger'], width=2, dash='dash'),
            name='Normal',
            showlegend=False,
            hoverinfo='skip'
        ),
        row=1, col=2
    )

    # 3. Residuals vs Sample Order
    sample_indices = np.arange(1, n_samples + 1)
    fig.add_trace(
        go.Scatter(
            x=sample_indices,
            y=residuals,
            mode='markers+lines',
            marker=dict(
                size=6,
                color=colors['primary'],
                line=dict(color=colors['text'], width=0.5)
            ),
            line=dict(color=colors['primary'], width=1, dash='dot'),
            text=sample_names,
            hovertemplate=(
                '<b>%{text}</b><br>' +
                'Index: %{x}<br>' +
                'Residual: %{y:.3f}<br>' +
                '<extra></extra>'
            ),
            showlegend=False
        ),
        row=2, col=1
    )

    # Add zero line
    fig.add_hline(
        y=0, line_dash="dash", line_color=colors['grid'],
        row=2, col=1
    )

    # 4. Q-Q Plot (Normal probability plot)
    # Calculate theoretical quantiles
    sorted_residuals = np.sort(std_residuals)
    theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(sorted_residuals)))

    fig.add_trace(
        go.Scatter(
            x=theoretical_quantiles,
            y=sorted_residuals,
            mode='markers',
            marker=dict(
                size=6,
                color=colors['primary'],
                line=dict(color=colors['text'], width=0.5)
            ),
            hovertemplate=(
                'Theoretical: %{x:.2f}<br>' +
                'Sample: %{y:.2f}<br>' +
                '<extra></extra>'
            ),
            showlegend=False
        ),
        row=2, col=2
    )

    # Add reference line (y=x)
    qq_range = [min(theoretical_quantiles.min(), sorted_residuals.min()),
                max(theoretical_quantiles.max(), sorted_residuals.max())]
    fig.add_trace(
        go.Scatter(
            x=qq_range,
            y=qq_range,
            mode='lines',
            line=dict(color=colors['danger'], width=2, dash='dash'),
            showlegend=False,
            hoverinfo='skip'
        ),
        row=2, col=2
    )

    # Update axes labels
    fig.update_xaxes(title_text="Predicted Values", row=1, col=1)
    fig.update_yaxes(title_text="Residuals", row=1, col=1)

    fig.update_xaxes(title_text="Residuals", row=1, col=2)
    fig.update_yaxes(title_text="Frequency", row=1, col=2)

    fig.update_xaxes(title_text="Sample Index", row=2, col=1)
    fig.update_yaxes(title_text="Residuals", row=2, col=1)

    fig.update_xaxes(title_text="Theoretical Quantiles", row=2, col=2)
    fig.update_yaxes(title_text="Sample Quantiles", row=2, col=2)

    # Update layout
    fig.update_layout(
        title="Residual Diagnostic Plots",
        template=colors['template'],
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['background'],
        font=dict(color=colors['text']),
        height=700,
        showlegend=False
    )

    # Update all axes with grid
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor=colors['grid'])
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor=colors['grid'])

    return fig


def plot_loadings(model: Dict[str, Any],
                 component_x: int,
                 component_y: int,
                 feature_names: Optional[List[str]] = None,
                 plot_type: str = 'scatter',
                 theme: str = 'light') -> go.Figure:
    """
    Plot PLS loadings for two components.

    Parameters
    ----------
    model : Dict[str, Any]
        Fitted PLS model from pls_nipals()
    component_x : int
        Component for x-axis (1-indexed)
    component_y : int
        Component for y-axis (1-indexed)
    feature_names : Optional[List[str]], optional
        Variable names for labeling
    plot_type : str, optional
        'scatter' or 'bar' (default: 'scatter')
    theme : str, optional
        Color theme ('light' or 'dark', default: 'light')

    Returns
    -------
    go.Figure
        Interactive Plotly figure

    Notes
    -----
    Loadings show how original variables contribute to latent variables.
    - Scatter: Loading plot (biplot style)
    - Bar: Variable importance per component
    """
    # TODO: Implement loading plots
    pass


def plot_regression_coefficients(model: Dict[str, Any],
                                 feature_names: Optional[List[str]] = None,
                                 plot_type: str = 'bar',
                                 sort_by_magnitude: bool = True,
                                 theme: str = 'light') -> go.Figure:
    """
    Plot PLS regression coefficients.

    Parameters
    ----------
    model : Dict[str, Any]
        Fitted PLS model from pls_nipals()
    feature_names : Optional[List[str]], optional
        Variable names for labeling
    plot_type : str, optional
        'bar' or 'waterfall' (default: 'bar')
    sort_by_magnitude : bool, optional
        Whether to sort coefficients by absolute value (default: True)
    theme : str, optional
        Color theme ('light' or 'dark', default: 'light')

    Returns
    -------
    go.Figure
        Interactive Plotly figure

    Notes
    -----
    Shows variable importance in final prediction equation.
    Color-coded by sign (positive/negative effect).
    """
    # Extract coefficients (key is 'B' in model dictionary)
    coefficients = model['B'].flatten()
    n_features = len(coefficients)

    # Generate feature names if not provided
    if feature_names is None:
        feature_names = [f"Var {i+1}" for i in range(n_features)]

    # Get colors
    colors = get_plot_colors(theme)

    # Sort by magnitude if requested
    if sort_by_magnitude:
        sort_idx = np.argsort(np.abs(coefficients))[::-1]
        coefficients = coefficients[sort_idx]
        feature_names = [feature_names[i] for i in sort_idx]

    # Color-code by sign
    bar_colors = [colors['success'] if c >= 0 else colors['danger'] for c in coefficients]

    # Create figure
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=coefficients,
        y=feature_names,
        orientation='h',
        marker=dict(
            color=bar_colors,
            line=dict(color=colors['text'], width=0.5)
        ),
        text=[f"{c:.4f}" for c in coefficients],
        textposition='auto',
        hovertemplate=(
            '<b>%{y}</b><br>' +
            'Coefficient: %{x:.4f}<br>' +
            '<extra></extra>'
        )
    ))

    # Add vertical line at zero
    fig.add_vline(
        x=0,
        line_dash="solid",
        line_color=colors['text'],
        line_width=1
    )

    # Update layout
    fig.update_layout(
        title=f"PLS Regression Coefficients (n={model['n_components']} LV)",
        xaxis_title="Coefficient Value",
        yaxis_title="Variable",
        template=colors['template'],
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['background'],
        font=dict(color=colors['text']),
        showlegend=False,
        height=max(400, n_features * 20)  # Dynamic height based on number of features
    )

    # Update axes
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor=colors['grid'],
        zeroline=True,
        zerolinecolor=colors['text']
    )
    fig.update_yaxes(
        showgrid=False,
        categoryorder='array',
        categoryarray=feature_names
    )

    return fig


def plot_scores(model: Dict[str, Any],
               component_x: int,
               component_y: int,
               sample_names: Optional[List[str]] = None,
               color_by: Optional[np.ndarray] = None,
               theme: str = 'light') -> go.Figure:
    """
    Plot PLS scores (T matrix).

    Parameters
    ----------
    model : Dict[str, Any]
        Fitted PLS model from pls_nipals()
    component_x : int
        Component for x-axis (1-indexed)
    component_y : int
        Component for y-axis (1-indexed)
    sample_names : Optional[List[str]], optional
        Sample labels for hover info
    color_by : Optional[np.ndarray], optional
        Values to color points by (e.g., response values)
    theme : str, optional
        Color theme ('light' or 'dark', default: 'light')

    Returns
    -------
    go.Figure
        Interactive Plotly figure

    Notes
    -----
    Scores show sample positions in latent variable space.
    Useful for detecting clusters, outliers, and patterns.
    """
    # TODO: Implement score plot
    pass


def plot_explained_variance(model: Dict[str, Any],
                           theme: str = 'light') -> go.Figure:
    """
    Plot explained variance by component (X and y).

    Parameters
    ----------
    model : Dict[str, Any]
        Fitted PLS model from pls_nipals()
    theme : str, optional
        Color theme ('light' or 'dark', default: 'light')

    Returns
    -------
    go.Figure
        Interactive Plotly figure with two series:
        - X variance explained per component
        - y variance explained per component

    Notes
    -----
    Shows how much information each component captures.
    Cumulative variance also displayed.
    """
    # TODO: Implement explained variance plot
    pass


def plot_cv_predictions_by_fold(cv_results: Dict[str, Any],
                               optimal_lv: int,
                               theme: str = 'light') -> go.Figure:
    """
    Plot CV predictions colored by fold.

    Parameters
    ----------
    cv_results : Dict[str, Any]
        Results from repeated_kfold_cv()
    optimal_lv : int
        Number of latent variables to use
    theme : str, optional
        Color theme ('light' or 'dark', default: 'light')

    Returns
    -------
    go.Figure
        Interactive Plotly figure

    Notes
    -----
    Each fold shown in different color to visualize CV strategy.
    Helps diagnose if certain folds perform poorly.
    """
    # TODO: Implement CV prediction plot
    pass


def apply_theme(fig: go.Figure, theme: str = 'light') -> go.Figure:
    """
    Apply color theme to Plotly figure using color_utils.

    Parameters
    ----------
    fig : go.Figure
        Plotly figure to style
    theme : str, optional
        Color theme ('light' or 'dark', default: 'light')

    Returns
    -------
    go.Figure
        Styled figure

    Notes
    -----
    Uses color_utils.get_plot_colors() to apply consistent theming.
    """
    # TODO: Implement theme application
    pass
