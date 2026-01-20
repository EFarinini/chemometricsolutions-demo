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


def _build_metadata_string(metadata_df: pd.DataFrame, sample_position: int) -> str:
    """
    Build metadata string for hover tooltip from metadata dataframe.

    Parameters
    ----------
    metadata_df : pd.DataFrame
        Metadata dataframe with metadata columns
    sample_position : int
        Position in the metadata_df (0-based index within selected samples)

    Returns
    -------
    str
        Formatted metadata string like "Block: A | Week: 2 | PC1-LAB: 1.5"
    """
    if metadata_df is None or sample_position >= len(metadata_df):
        return ""

    meta_parts = []
    for col in metadata_df.columns:
        val = metadata_df.iloc[sample_position][col]
        # Format numeric values with 2 decimals, keep strings as-is
        try:
            val_str = f"{float(val):.2f}"
        except (ValueError, TypeError):
            val_str = str(val)
        meta_parts.append(f"{col}: {val_str}")

    if meta_parts:
        return " | ".join(meta_parts)
    return ""


def plot_row_profiles_enhanced(
    dataframe: pd.DataFrame,
    color_mode: str = "uniform",
    color_variable: str = None,
    row_indices: list = None,
    marker_size: int = 3,
    custom_x_label: str = None,
    custom_y_label: str = None,
    custom_title: str = None,
    metadata_df: pd.DataFrame = None
) -> go.Figure:
    """
    Row profile plot with 4 color modes and optional metadata in hover.

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
    custom_x_label : str, optional
        Custom label for X-axis (default: "Variables")
    custom_y_label : str, optional
        Custom label for Y-axis (default: "Values")
    custom_title : str, optional
        Custom plot title (default: auto-generated)
    metadata_df : pd.DataFrame, optional
        Metadata for hover tooltips (same number of rows as selected samples)

    Returns
    -------
    go.Figure
    """

    if row_indices is None:
        row_indices = list(range(len(dataframe)))

    fig = go.Figure()

    # Determine which columns to plot (exclude color_variable if present)
    x_columns_original = [col for col in dataframe.columns if col != color_variable]
    # Convert to string to ensure proper display on X-axis (handles numeric column names)
    x_columns_display = [str(col) for col in x_columns_original]

    # ===== MODE 1: UNIFORM =====
    if color_mode == "uniform":
        for pos, original_idx in enumerate(row_indices):
            # Real row number (1-based) for display - use original_idx from row_indices
            row_number = original_idx + 1

            # Build metadata string for this sample
            meta_str = _build_metadata_string(metadata_df, pos)
            meta_line = f"<br>{meta_str}" if meta_str else ""

            fig.add_trace(go.Scatter(
                x=x_columns_display,  # Use display names for X-axis labels
                y=dataframe.iloc[pos][x_columns_original].values,  # Use original names for data access
                mode='lines+markers',
                name=f"Sample {row_number}",
                line=dict(color='steelblue', width=1.5),
                marker=dict(size=marker_size, color='steelblue'),
                showlegend=False,
                hovertemplate=(
                    f"<b>Sample {row_number}</b>{meta_line}<br>"
                    f"Variable: %{{x}}<br>"
                    f"Value: %{{y:.4f}}<extra></extra>"
                )
            ))

    # ===== MODE 2: BY ROW INDEX (discrete colors from color_utils) =====
    elif color_mode == "row_index":
        # Import color palette from color_utils
        try:
            from color_utils import get_unified_color_schemes
            color_scheme = get_unified_color_schemes()
            color_palette = color_scheme['categorical_colors']
            # Result: ['black', 'red', 'green', 'blue', 'orange', 'purple', 'brown', 'hotpink', ...]
        except ImportError:
            # Fallback: basic colors if color_utils not available
            color_palette = ['black', 'red', 'green', 'blue', 'orange', 'purple', 'brown', 'hotpink']

        for pos, original_idx in enumerate(row_indices):
            # Real row number (1-based) for display - use original_idx from row_indices
            row_number = original_idx + 1

            # Use discrete color from palette (cycles through colors)
            color = color_palette[pos % len(color_palette)]

            # Build metadata string for this sample
            meta_str = _build_metadata_string(metadata_df, pos)
            meta_line = f"<br>{meta_str}" if meta_str else ""

            fig.add_trace(go.Scatter(
                x=x_columns_display,  # Use display names for X-axis labels
                y=dataframe.iloc[pos][x_columns_original].values,  # Use original names for data access
                mode='lines+markers',
                name=f"Sample {row_number}",  # Legend shows "Sample 3", "Sample 5", "Sample 9"
                line=dict(color=color, width=1.5),
                marker=dict(size=marker_size, color=color),
                showlegend=True,  # Show legend for each sample
                hovertemplate=(
                    f"<b>Sample {row_number}</b>{meta_line}<br>"
                    f"Variable: %{{x}}<br>"
                    f"Value: %{{y:.4f}}<extra></extra>"
                )
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

        for pos, original_idx in enumerate(row_indices):
            # Real row number (1-based) for display - use original_idx from row_indices
            row_number = original_idx + 1

            if pos < len(color_vals) and pd.notna(color_vals[pos]):
                color = get_continuous_color_for_value(
                    color_vals[pos], min_val, max_val, 'blue_to_red'
                )
                label = f"Sample {row_number} ({color_variable}={color_vals[pos]:.3f})"
            else:
                color = 'rgb(128, 128, 128)'
                label = f"Sample {row_number}"

            # Build metadata string for this sample
            meta_str = _build_metadata_string(metadata_df, pos)
            meta_line = f"<br>{meta_str}" if meta_str else ""

            fig.add_trace(go.Scatter(
                x=x_columns_display,  # Use display names for X-axis labels
                y=dataframe.iloc[pos][x_columns_original].values,  # Use original names for data access
                mode='lines+markers',
                name=label,
                line=dict(color=color, width=1.5),
                marker=dict(size=marker_size, color=color),
                showlegend=False,
                hovertemplate=(
                    f"<b>{label}</b>{meta_line}<br>"
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

    # ===== MODE 4: BY CATEGORY (WITH color_utils) =====
    elif color_mode == "category" and color_variable:
        if color_variable not in dataframe.columns:
            st.error(f"Column '{color_variable}' not found")
            return fig

        cat_vals = dataframe[color_variable].values
        unique_cats = sorted(pd.Series(cat_vals).dropna().unique())

        # USE color_utils for professional categorical colors!
        color_map = create_categorical_color_map(unique_cats)

        # Track which categories already in legend
        cats_in_legend = set()

        for pos, original_idx in enumerate(row_indices):
            # Real row number (1-based) for display - use original_idx from row_indices
            row_number = original_idx + 1

            if pos < len(cat_vals) and pd.notna(cat_vals[pos]):
                cat = cat_vals[pos]
                color = color_map[cat]
                label = f"{color_variable}={cat}"
            else:
                color = 'rgb(128, 128, 128)'
                label = "Missing"

            # Show in legend only once per category
            show_legend = label not in cats_in_legend
            if show_legend:
                cats_in_legend.add(label)

            # Build metadata string for this sample
            meta_str = _build_metadata_string(metadata_df, pos)
            meta_line = f"<br>{meta_str}" if meta_str else ""

            fig.add_trace(go.Scatter(
                x=x_columns_display,  # Use display names for X-axis labels
                y=dataframe.iloc[pos][x_columns_original].values,  # Use original names for data access
                mode='lines+markers',
                name=label,
                line=dict(color=color, width=1.5),
                marker=dict(size=marker_size, color=color),
                legendgroup=label,
                showlegend=show_legend,
                hovertemplate=(
                    f"<b>{label}</b><br>"
                    f"Sample: {row_number}{meta_line}<br>"
                    f"Variable: %{{x}}<br>"
                    f"Value: %{{y:.4f}}<extra></extra>"
                )
            ))

    # ===== CALCULATE AUTO-SCALED Y-AXIS RANGE (X COLUMNS ONLY) =====
    # Use original column names for data access (excludes color_variable)
    all_values = []
    for pos in range(len(row_indices)):
        # Only use X columns, NOT the color_variable
        row_vals = dataframe.iloc[pos][x_columns_original].values
        all_values.extend(row_vals)

    # Convert to numeric, skipping non-numeric values (avoids TypeError)
    numeric_values = []
    for val in all_values:
        try:
            numeric_values.append(float(val))
        except (ValueError, TypeError):
            # Skip non-numeric (e.g., strings, categories)
            pass

    numeric_values = np.array(numeric_values)

    if len(numeric_values) > 0:
        y_min = np.min(numeric_values)
        y_max = np.max(numeric_values)
        # Add 10% padding above and below
        y_padding = (y_max - y_min) * 0.1
        y_range = [y_min - y_padding, y_max + y_padding]
    else:
        y_range = None

    # ===== APPLY LAYOUT WITH AUTO-SCALED Y-AXIS =====
    # Use custom labels if provided, otherwise use defaults
    x_label = custom_x_label if custom_x_label else "Variables"
    y_label = custom_y_label if custom_y_label else "Values"
    plot_title = custom_title if custom_title else f"Row Profiles ({len(row_indices)} samples)"

    fig.update_layout(
        title=plot_title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        template='plotly_white',
        hovermode='closest',
        height=600,
        xaxis=dict(tickangle=-45),
        yaxis=dict(range=y_range, autorange=False) if y_range else dict(autorange=True),
        legend=dict(
            x=1.05,
            y=1,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='gray',
            borderwidth=1
        )
    )

    return fig
