"""
PLS Regression Coefficients Plots - Smart Module

Features:
- Auto-detection: >100 variables = line plot (spectral), <100 = bar chart
- Multiple model comparison (different LV)
- Overfitting detection via coefficient overlay
- Interactive variable selection
- Professional visualization

Author: ChemoMetric Solutions
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Any, List, Optional


def plot_regression_coefficients_smart(
    model: Dict[str, Any],
    model_dict: Optional[Dict[int, Dict[str, Any]]] = None,
    overlay_models: Optional[List[int]] = None,
    feature_names: Optional[List[str]] = None
) -> go.Figure:
    """
    Smart regression coefficients plot with auto-detection of data type.

    Automatically chooses between:
    - Line plot for spectral data (>100 variables)
    - Bar chart for tabular data (<100 variables)

    Supports overlay of multiple models for overfitting diagnosis.

    Parameters
    ----------
    model : Dict[str, Any]
        Fitted PLS model with 'B' (coefficients)
    model_dict : Optional[Dict[int, Dict]]
        Dictionary of all fitted models: {n_lv: model}
        For comparing multiple LV models
    overlay_models : Optional[List[int]]
        List of LV numbers to overlay (e.g., [1, 3, 5, 8])
    feature_names : Optional[List[str]]
        Feature/variable names. If None, uses generic names

    Returns
    -------
    go.Figure
        Interactive plot (line or bar)

    Interpretation:
    - Smooth line = Good model (spectral data)
    - Jagged line = Overfitting (too many LV)
    - Overlaid plots diverging = Overfitting present
    """

    # Get coefficients
    B = model['B']
    n_features = len(B)

    # Feature names
    if feature_names is None:
        feature_names = [f"Var_{i+1}" for i in range(n_features)]
    elif isinstance(feature_names, str):
        # If single string, assume spectral and generate "1000_1100_1200" etc
        feature_names = feature_names

    # Determine plot type based on number of variables
    is_spectral = n_features > 100

    fig = go.Figure()

    # Define colors for overlay
    colors = [
        'rgba(255, 0, 0, 0.8)',      # Red - most LV
        'rgba(0, 0, 255, 0.8)',      # Blue
        'rgba(0, 128, 0, 0.8)',      # Green
        'rgba(255, 165, 0, 0.8)',    # Orange
        'rgba(128, 0, 128, 0.8)',    # Purple
        'rgba(255, 192, 203, 0.8)',  # Pink
        'rgba(0, 128, 128, 0.8)',    # Teal
        'rgba(128, 128, 0, 0.8)',    # Olive
    ]

    # If overlay_models provided, plot multiple models
    if overlay_models and model_dict:
        for idx, n_lv in enumerate(overlay_models):
            if n_lv in model_dict:
                model_lv = model_dict[n_lv]
                B_lv = model_lv['B']
                color = colors[idx % len(colors)]

                # Determine x-axis values (use feature names or 1-based indexing)
                if feature_names and len(feature_names) == n_features:
                    x_axis = feature_names
                else:
                    # 1-based indexing for variables
                    x_axis = [str(i+1) for i in range(n_features)]

                if is_spectral:
                    # Line plot for spectral
                    fig.add_trace(go.Scatter(
                        x=x_axis,
                        y=B_lv,
                        mode='lines',
                        name=f'{n_lv} LV',
                        line=dict(color=color, width=2),
                        hovertemplate='Var: %{x}<br>Coeff: %{y:.4f}<extra></extra>',
                        opacity=0.7
                    ))
                else:
                    # Bar plot for tabular (only main model, others as reference lines)
                    if idx == 0:
                        fig.add_trace(go.Bar(
                            x=x_axis,
                            y=B_lv,
                            name=f'{n_lv} LV',
                            marker=dict(
                                color=B_lv,
                                colorscale='RdBu',
                                cmid=0,
                                showscale=False,
                                line=dict(width=0.5, color='white')
                            ),
                            text=[f"{v:.4f}" for v in B_lv],
                            textposition='outside',
                            hovertemplate='Var: %{x}<br>Coeff: %{y:.4f}<extra></extra>'
                        ))
    else:
        # Single model plot
        # Determine x-axis values (use feature names or 1-based indexing)
        if feature_names and len(feature_names) == n_features:
            x_axis = feature_names
        else:
            # 1-based indexing for variables
            x_axis = [str(i+1) for i in range(n_features)]

        if is_spectral:
            # LINE PLOT for spectral data
            fig.add_trace(go.Scatter(
                x=x_axis,
                y=B,
                mode='lines+markers',
                name='Coefficients',
                line=dict(color='rgba(31, 119, 180, 0.8)', width=2),
                marker=dict(size=3, color='rgba(31, 119, 180, 0.8)'),
                fill='tozeroy',
                fillcolor='rgba(31, 119, 180, 0.2)',
                hovertemplate='Variable: %{x}<br>Coefficient: %{y:.4f}<extra></extra>'
            ))

            # Add zero line
            fig.add_hline(y=0, line_dash="dash", line_color="black", line_width=1)

        else:
            # BAR PLOT for tabular data
            fig.add_trace(go.Bar(
                x=x_axis,
                y=B,
                name='Coefficients',
                marker=dict(
                    color=B,
                    colorscale='RdBu',
                    cmid=0,
                    showscale=True,
                    colorbar=dict(title="Coefficient"),
                    line=dict(width=0.5, color='white')
                ),
                text=[f"{v:.4f}" for v in B],
                textposition='outside',
                hovertemplate='Variable: %{x}<br>Coefficient: %{y:.4f}<extra></extra>'
            ))

            # Add zero line
            fig.add_hline(y=0, line_dash="dash", line_color="black", line_width=1)

    # Update layout
    if is_spectral:
        plot_type = "Line Plot (Spectral Data)"
        height = 500
    else:
        plot_type = "Bar Chart (Tabular Data)"
        height = max(400, n_features * 8)

    n_lv_current = model.get('n_components', '?')

    fig.update_layout(
        title=f"PLS Regression Coefficients - {plot_type} (n={n_lv_current} LV)",
        xaxis_title="Variable Index" if is_spectral else "Variable",
        yaxis_title="Regression Coefficient (B)",
        hovermode='closest',
        width=1000,
        height=height,
        template='plotly_white',
        showlegend=True,
        legend=dict(x=0.02, y=0.98)
    )

    return fig


def create_coefficient_models_dict(
    models_by_lv: Dict[int, Dict[str, Any]]
) -> Dict[int, Dict[str, Any]]:
    """
    Create dictionary of models with different LV for comparison.

    Parameters
    ----------
    models_by_lv : Dict[int, Dict]
        Dictionary: {n_lv: model_dict}

    Returns
    -------
    Dict[int, Dict]
        Same structure, validated
    """
    validated = {}
    for n_lv, model in models_by_lv.items():
        if 'B' in model and 'n_components' in model:
            validated[n_lv] = model
    return validated


def analyze_overfitting(
    model_dict: Dict[int, Dict[str, Any]],
    feature_names: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Analyze coefficient stability across different LV counts.

    High variation = potential overfitting
    Stable coefficients = good model

    Parameters
    ----------
    model_dict : Dict[int, Dict]
        Models at different LV: {n_lv: model}
    feature_names : Optional[List[str]]
        Feature names

    Returns
    -------
    pd.DataFrame
        Analysis with columns:
        - Variable
        - Mean_Coeff
        - Std_Coeff
        - CV (Coefficient of Variation)
        - Stability (low CV = good)
    """
    if not model_dict or len(model_dict) < 2:
        return pd.DataFrame()

    n_features = len(next(iter(model_dict.values()))['B'])
    n_lv_values = sorted(model_dict.keys())

    # Collect all coefficients
    all_coeff = []
    for n_lv in n_lv_values:
        all_coeff.append(model_dict[n_lv]['B'])

    all_coeff = np.array(all_coeff).T  # Shape: (n_features, n_models)

    # Calculate statistics
    mean_coeff = np.mean(all_coeff, axis=1)
    std_coeff = np.std(all_coeff, axis=1)

    # Coefficient of Variation (CV)
    cv = np.zeros(n_features)
    non_zero_mask = np.abs(mean_coeff) > 1e-10
    cv[non_zero_mask] = std_coeff[non_zero_mask] / np.abs(mean_coeff[non_zero_mask])

    # Stability indicator (lower CV = more stable = better)
    stability = ['Good' if c < 0.1 else 'Moderate' if c < 0.3 else 'Poor'
                 for c in cv]

    if feature_names is None:
        feature_names = [f"Var_{i+1}" for i in range(n_features)]

    return pd.DataFrame({
        'Variable': feature_names,
        'Mean_Coeff': mean_coeff,
        'Std_Coeff': std_coeff,
        'CV': cv,
        'Stability': stability
    })


def plot_coefficient_comparison(
    model_dict: Dict[int, Dict[str, Any]],
    feature_names: Optional[List[str]] = None,
    top_n: int = 20
) -> go.Figure:
    """
    Create comparison plot of coefficients across different LV.

    Shows how coefficients change with model complexity.
    Large changes = potential overfitting.

    Parameters
    ----------
    model_dict : Dict[int, Dict]
        Models: {n_lv: model}
    feature_names : Optional[List[str]]
        Feature names
    top_n : int
        Show top N features by absolute coefficient value

    Returns
    -------
    go.Figure
        Heatmap showing coefficient evolution
    """
    if not model_dict:
        return go.Figure()

    n_lv_values = sorted(model_dict.keys())

    # Get coefficients
    coeff_matrix = []
    for n_lv in n_lv_values:
        coeff_matrix.append(model_dict[n_lv]['B'])

    coeff_matrix = np.array(coeff_matrix).T  # (n_features, n_models)
    n_features = coeff_matrix.shape[0]

    # Get top features
    if n_features > top_n:
        max_abs_idx = np.argsort(np.max(np.abs(coeff_matrix), axis=1))[-top_n:]
        coeff_matrix = coeff_matrix[max_abs_idx]
        if feature_names is None:
            feature_names = [f"Var_{i+1}" for i in max_abs_idx]
        else:
            feature_names = [feature_names[i] for i in max_abs_idx]
    else:
        if feature_names is None:
            feature_names = [f"Var_{i+1}" for i in range(n_features)]

    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=coeff_matrix,
        x=[f"{n_lv} LV" for n_lv in n_lv_values],
        y=feature_names,
        colorscale='RdBu',
        zmid=0,
        hovertemplate='Feature: %{y}<br>Model: %{x}<br>Coeff: %{z:.4f}<extra></extra>',
        colorbar=dict(title="Coefficient")
    ))

    fig.update_layout(
        title=f"Coefficient Evolution Across LV (Top {top_n} Features)",
        xaxis_title="Model Complexity (#LV)",
        yaxis_title="Variable",
        width=900,
        height=max(400, min(600, len(feature_names) * 15)),
        template='plotly_white'
    )

    return fig
