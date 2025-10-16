"""
Visualization functions for transformation comparisons
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from color_utils import (
    is_quantitative_variable,
    create_categorical_color_map,
    get_continuous_color_for_value
)


def plot_comparison(original_data, transformed_data, title_original, title_transformed,
                   color_data=None, color_variable=None):
    """
    Create two line plots for transformation comparison

    Parameters:
    -----------
    original_data : pd.DataFrame
        Original data before transformation
    transformed_data : pd.DataFrame
        Transformed data
    title_original : str
        Title for original data plot
    title_transformed : str
        Title for transformed data plot
    color_data : array-like, optional
        Data for coloring traces
    color_variable : str, optional
        Name of the coloring variable

    Returns:
    --------
    plotly.Figure : Comparison plot with two subplots
    """

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(title_original, title_transformed),
        vertical_spacing=0.15
    )

    if color_data is not None:
        # Convert color_data to list for easy access
        if hasattr(color_data, 'values'):
            color_values = color_data.values
        else:
            color_values = list(color_data)

        # Determine if variable is quantitative or categorical
        is_quantitative = is_quantitative_variable(color_data)

        if is_quantitative:
            # Quantitative variable: use blue-to-red scale
            color_data_series = pd.Series(color_values).dropna()
            min_val = color_data_series.min()
            max_val = color_data_series.max()

            # Plot original data
            for i, idx in enumerate(original_data.index):
                if i < len(color_values) and pd.notna(color_values[i]):
                    color = get_continuous_color_for_value(color_values[i], min_val, max_val, 'blue_to_red')
                    hover_text = f'Sample: {idx}<br>{color_variable}: {color_values[i]:.3f}<br>Value: %{{y:.3f}}<extra></extra>'
                else:
                    color = 'rgb(128, 128, 128)'
                    hover_text = f'Sample: {idx}<br>Value: %{{y:.3f}}<extra></extra>'

                fig.add_trace(
                    go.Scatter(
                        x=list(range(len(original_data.columns))),
                        y=original_data.iloc[i].values,
                        mode='lines',
                        line=dict(color=color, width=1.5),
                        showlegend=False,
                        hovertemplate=hover_text
                    ),
                    row=1, col=1
                )

            # Plot transformed data
            for i, idx in enumerate(transformed_data.index):
                if i < len(color_values) and pd.notna(color_values[i]):
                    color = get_continuous_color_for_value(color_values[i], min_val, max_val, 'blue_to_red')
                    hover_text = f'Sample: {idx}<br>{color_variable}: {color_values[i]:.3f}<br>Value: %{{y:.3f}}<extra></extra>'
                else:
                    color = 'rgb(128, 128, 128)'
                    hover_text = f'Sample: {idx}<br>Value: %{{y:.3f}}<extra></extra>'

                fig.add_trace(
                    go.Scatter(
                        x=list(range(len(transformed_data.columns))),
                        y=transformed_data.iloc[i].values,
                        mode='lines',
                        line=dict(color=color, width=1.5),
                        showlegend=False,
                        hovertemplate=hover_text
                    ),
                    row=2, col=1
                )

            # Add colorbar
            n_ticks = 6
            tick_vals = [min_val + i * (max_val - min_val) / (n_ticks - 1) for i in range(n_ticks)]

            fig.add_trace(
                go.Scatter(
                    x=[None], y=[None],
                    mode='markers',
                    marker=dict(
                        colorscale=[[0, 'rgb(0,0,255)'], [1, 'rgb(255,0,0)']],
                        cmin=min_val,
                        cmax=max_val,
                        colorbar=dict(
                            title=dict(
                                text=f"<b>{color_variable}</b>",
                                side="right",
                                font=dict(size=12)
                            ),
                            titleside="right",
                            x=1.02,
                            len=0.8,
                            y=0.5,
                            thickness=15,
                            tickmode="array",
                            tickvals=tick_vals,
                            ticktext=[f"{val:.2f}" for val in tick_vals],
                            tickfont=dict(size=10),
                            showticklabels=True,
                            ticks="outside",
                            ticklen=5
                        ),
                        showscale=True
                    ),
                    showlegend=False
                ),
                row=1, col=1
            )

        else:
            # Categorical variable: use discrete colors
            unique_values = pd.Series(color_values).dropna().unique()
            color_discrete_map = create_categorical_color_map(unique_values)

            for group in unique_values:
                group_indices = [i for i, val in enumerate(color_values) if val == group]
                first_idx = group_indices[0] if group_indices else None

                for i in group_indices:
                    is_first = bool(i == first_idx)
                    if i < len(original_data):
                        fig.add_trace(
                            go.Scatter(
                                x=list(range(len(original_data.columns))),
                                y=original_data.iloc[i].values,
                                mode='lines',
                                name=str(group),
                                line=dict(color=color_discrete_map[group], width=1),
                                showlegend=is_first,
                                legendgroup=str(group),
                                hovertemplate=f'Sample: {original_data.index[i]}<br>{color_variable}: {group}<br>Value: %{{y:.3f}}<extra></extra>'
                            ),
                            row=1, col=1
                        )

                    if i < len(transformed_data):
                        fig.add_trace(
                            go.Scatter(
                                x=list(range(len(transformed_data.columns))),
                                y=transformed_data.iloc[i].values,
                                mode='lines',
                                name=str(group),
                                line=dict(color=color_discrete_map[group], width=1),
                                showlegend=False,
                                legendgroup=str(group),
                                hovertemplate=f'Sample: {transformed_data.index[i]}<br>{color_variable}: {group}<br>Value: %{{y:.3f}}<extra></extra>'
                            ),
                            row=2, col=1
                        )
    else:
        # No coloring
        for i, idx in enumerate(original_data.index):
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(original_data.columns))),
                    y=original_data.iloc[i].values,
                    mode='lines',
                    line=dict(color='blue', width=1),
                    showlegend=False,
                    hovertemplate=f'Sample: {idx}<br>Value: %{{y:.3f}}<extra></extra>'
                ),
                row=1, col=1
            )

            if i < len(transformed_data):
                fig.add_trace(
                    go.Scatter(
                        x=list(range(len(transformed_data.columns))),
                        y=transformed_data.iloc[i].values,
                        mode='lines',
                        line=dict(color='blue', width=1),
                        showlegend=False,
                        hovertemplate=f'Sample: {idx}<br>Value: %{{y:.3f}}<extra></extra>'
                    ),
                    row=2, col=1
                )

    fig.update_xaxes(title_text="Variable Index", row=1, col=1, gridcolor='lightgray')
    fig.update_xaxes(title_text="Variable Index", row=2, col=1, gridcolor='lightgray')
    fig.update_yaxes(title_text="Value", row=1, col=1, gridcolor='lightgray')
    fig.update_yaxes(title_text="Value", row=2, col=1, gridcolor='lightgray')

    fig.update_layout(
        height=800,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='black'),
        hovermode='closest'
    )

    return fig
