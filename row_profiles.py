"""
TAB 3: ROW PROFILES (ENHANCED COLORING)

Features:
1. Sample selection: All, Range, Specific
2. Color modes:
   - Uniform (all blue)
   - By row index (gradient)
   - By column value (numeric → blue-red)
   - By category (categorical → discrete colors)
3. Legend shows category/value, NOT sample number
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats
import streamlit as st

# Absolute import (color_utils is in root directory)
from color_utils import (
    is_quantitative_variable,
    create_categorical_color_map,
    get_continuous_color_for_value
)


def plot_row_profiles_enhanced(
    dataframe: pd.DataFrame,
    color_mode: str = "uniform",
    color_variable: str = None,
    row_indices: list = None,
    marker_size: int = 3
) -> go.Figure:
    """
    Row profile plot with 4 color modes.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Data (rows=samples, cols=variables)
    color_mode : str
        'uniform', 'row_index', 'column_value', 'category'
    color_variable : str
        Column name for coloring (when color_mode != 'uniform')
    row_indices : list
        Which rows to plot. If None, plot all.
    marker_size : int
        Size of data point markers (default: 3)

    Returns
    -------
    go.Figure
    """

    if row_indices is None:
        row_indices = list(range(len(dataframe)))

    fig = go.Figure()

    # ===== MODE 1: UNIFORM =====
    if color_mode == "uniform":
        for idx in row_indices:
            fig.add_trace(go.Scatter(
                x=dataframe.columns,
                y=dataframe.iloc[idx].values,
                mode='lines+markers',
                name=f"Sample {idx+1}",
                line=dict(color='steelblue', width=1.5),
                marker=dict(size=marker_size, color='steelblue'),
                showlegend=False,
                hovertemplate=(
                    f"<b>Sample {idx+1}</b><br>"
                    f"Variable: %{{x}}<br>"
                    f"Value: %{{y:.4f}}<extra></extra>"
                )
            ))

    # ===== MODE 2: BY ROW INDEX (gradient 0→N) =====
    elif color_mode == "row_index":
        n_rows = len(row_indices)
        for pos, idx in enumerate(row_indices):
            # Gradient from blue (0) to red (1)
            norm_val = pos / max(1, n_rows - 1)
            r = int(255 * norm_val)
            g = 0
            b = int(255 * (1 - norm_val))
            color = f'rgba({r}, {g}, {b}, 0.8)'

            fig.add_trace(go.Scatter(
                x=dataframe.columns,
                y=dataframe.iloc[idx].values,
                mode='lines+markers',
                name=f"Sample {idx+1}",
                line=dict(color=color, width=1.5),
                marker=dict(size=marker_size, color=color),
                showlegend=False,
                hovertemplate=(
                    f"<b>Sample {idx+1}</b><br>"
                    f"Variable: %{{x}}<br>"
                    f"Value: %{{y:.4f}}<extra></extra>"
                )
            ))

        # Add colorbar
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(
                colorscale=[[0, 'rgb(0,0,255)'], [1, 'rgb(255,0,0)']],
                cmin=0,
                cmax=n_rows,
                colorbar=dict(
                    title="Sample Index",
                    x=1.02,
                    len=0.7
                ),
                showscale=True
            ),
            showlegend=False,
            hoverinfo='skip'
        ))

    # ===== MODE 3: BY COLUMN VALUE =====
    elif color_mode == "column_value" and color_variable:
        if color_variable not in dataframe.columns:
            st.error(f"Column '{color_variable}' not found")
            return fig

        color_vals = dataframe[color_variable].values
        color_clean = pd.Series(color_vals).dropna()
        min_val = color_clean.min()
        max_val = color_clean.max()

        for idx in row_indices:
            if idx < len(color_vals) and pd.notna(color_vals[idx]):
                color = get_continuous_color_for_value(
                    color_vals[idx], min_val, max_val, 'blue_to_red'
                )
                label = f"Sample {idx+1} ({color_variable}={color_vals[idx]:.3f})"
            else:
                color = 'rgb(128, 128, 128)'
                label = f"Sample {idx+1}"

            fig.add_trace(go.Scatter(
                x=dataframe.columns,
                y=dataframe.iloc[idx].values,
                mode='lines+markers',
                name=label,
                line=dict(color=color, width=1.5),
                marker=dict(size=marker_size, color=color),
                showlegend=False,
                hovertemplate=(
                    f"<b>{label}</b><br>"
                    f"Variable: %{{x}}<br>"
                    f"Value: %{{y:.4f}}<extra></extra>"
                )
            ))

        # Add colorbar
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(
                colorscale=[[0, 'rgb(0,0,255)'], [1, 'rgb(255,0,0)']],
                cmin=min_val,
                cmax=max_val,
                colorbar=dict(
                    title=color_variable,
                    x=1.02,
                    len=0.7
                ),
                showscale=True
            ),
            showlegend=False,
            hoverinfo='skip'
        ))

    # ===== MODE 4: BY CATEGORY =====
    elif color_mode == "category" and color_variable:
        if color_variable not in dataframe.columns:
            st.error(f"Column '{color_variable}' not found")
            return fig

        cat_vals = dataframe[color_variable].values
        unique_cats = sorted(pd.Series(cat_vals).dropna().unique())
        color_map = create_categorical_color_map(unique_cats)

        # Track which categories already in legend
        cats_in_legend = set()

        for idx in row_indices:
            if idx < len(cat_vals) and pd.notna(cat_vals[idx]):
                cat = cat_vals[idx]
                color = color_map[cat]
                label = f"{color_variable}={cat}"
            else:
                color = 'rgb(128, 128, 128)'
                label = "Missing"

            # Show in legend only once per category
            show_legend = label not in cats_in_legend
            if show_legend:
                cats_in_legend.add(label)

            fig.add_trace(go.Scatter(
                x=dataframe.columns,
                y=dataframe.iloc[idx].values,
                mode='lines+markers',
                name=label,
                line=dict(color=color, width=1.5),
                marker=dict(size=marker_size, color=color),
                legendgroup=label,
                showlegend=show_legend,
                hovertemplate=(
                    f"<b>{label}</b><br>"
                    f"Sample: {idx+1}<br>"
                    f"Variable: %{{x}}<br>"
                    f"Value: %{{y:.4f}}<extra></extra>"
                )
            ))

    fig.update_layout(
        title=f"Row Profiles ({len(row_indices)} samples)",
        xaxis_title="Variables",
        yaxis_title="Values",
        template='plotly_white',
        hovermode='closest',
        height=600,
        xaxis=dict(tickangle=-45),
        legend=dict(
            x=1.05,
            y=1,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='gray',
            borderwidth=1
        )
    )

    return fig
