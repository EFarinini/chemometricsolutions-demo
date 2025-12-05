"""
PCA Quality Control Page
Statistical Quality Control using PCA with T² and Q statistics
Using the same PCA computation as the PCA menu
"""

import streamlit as st
import pandas as pd
import numpy as np
import io
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import PCA computation (same as PCA menu)
from pca_utils.pca_calculations import compute_pca
# from pca_utils import plot_combined_monitoring_chart  # Not used - all plots defined locally

# Import pretreatment detection module (simplified - informational only)
from pca_utils.pca_pretreatments import PretreatmentInfo, detect_pretreatments, display_pretreatment_info, display_pretreatment_warning

# Import workspace utilities for dataset selection
from workspace_utils import display_workspace_dataset_selector


# ============================================================================
# PLOTTING FUNCTIONS FROM process_monitoring.py
# ============================================================================

def create_score_plot(test_scores, explained_variance, timestamps=None,
                      pca_params=None, start_sample_num=1,
                      show_trajectory=True, trajectory_style="simple",
                      trajectory_colors=None, color_data=None, labels_data=None):
    """
    Create PCA score plot with confidence ellipses and color support.

    Parameters:
    -----------
    test_scores : pd.DataFrame or np.ndarray
        DataFrame or array with PC1, PC2 columns/values
    explained_variance : array-like
        Explained variance percentages for each PC
    timestamps : list, optional
        Timestamps for samples
    pca_params : dict, optional
        Dictionary with 'n_samples_train', 'n_features'
    start_sample_num : int
        Starting sample number for labeling
    show_trajectory : bool
        Whether to connect points with lines (default True)
    trajectory_style : str
        "simple" for light gray uniform line (default)
        "gradient" for blue→red gradient with cyan star on last point
    trajectory_colors : list, optional
        DEPRECATED - now controlled by trajectory_style parameter
    color_data : array-like, optional
        Data for coloring points (categorical or quantitative)
    labels_data : list, optional
        Sample labels to display on points

    Returns:
    --------
    go.Figure
        Plotly figure with score plot
    """
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    import plotly.express as px

    # Try to import color utilities
    try:
        from color_utils import (
            get_unified_color_schemes,
            create_categorical_color_map,
            is_quantitative_variable,
            get_trajectory_colors,
            get_sample_order_colors
        )
        COLORS_AVAILABLE = True
    except ImportError:
        COLORS_AVAILABLE = False

    fig = go.Figure()

    # Correct sample numbering
    sample_numbers_correct = [f"{start_sample_num + i}" for i in range(len(test_scores))]

    # Use labels_data if provided, otherwise use sample numbers
    if labels_data is not None and len(labels_data) == len(test_scores):
        text_labels = labels_data
    else:
        text_labels = sample_numbers_correct

    # Create dynamic hover template
    if timestamps is not None and len(timestamps) == len(test_scores):
        time_strings = [ts.strftime('%Y-%m-%d %H:%M') if hasattr(ts, 'strftime') else str(ts) for ts in timestamps]
        hover_template = 'Sample: %{text}<br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<br>📅 %{customdata}<extra></extra>'
        custom_data = time_strings
    else:
        hover_template = 'Sample: %{text}<br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<extra></extra>'
        custom_data = None

    # Prepare data for plotting
    n_points = len(test_scores)

    # Extract x, y data
    if isinstance(test_scores, pd.DataFrame):
        x_data = test_scores.iloc[:, 0].values
        y_data = test_scores.iloc[:, 1].values
    else:
        x_data = test_scores[:, 0]
        y_data = test_scores[:, 1]

    # Create scatter plot with unified color support
    if color_data is not None and len(color_data) == len(test_scores) and COLORS_AVAILABLE:
        color_series = pd.Series(color_data)

        # Check if quantitative or categorical
        if is_quantitative_variable(color_series):
            # QUANTITATIVE: Use continuous blue-to-red scale
            mode = 'markers+text' if labels_data is not None else 'markers'

            fig_scatter = px.scatter(
                x=x_data,
                y=y_data,
                color=color_data,
                text=text_labels if labels_data is not None else None,
                labels={'x': 'PC1', 'y': 'PC2'},
                color_continuous_scale=[(0.0, 'rgb(0, 0, 255)'),
                                       (0.5, 'rgb(128, 0, 128)'),
                                       (1.0, 'rgb(255, 0, 0)')]
            )
            fig_scatter.update_traces(
                marker=dict(size=8),
                mode=mode,
                textposition='top center',
                hovertemplate=hover_template,
                customdata=custom_data
            )

            # Transfer traces to main fig
            for trace in fig_scatter.data:
                fig.add_trace(trace)
        else:
            # CATEGORICAL: Use discrete color map
            unique_values = color_series.dropna().unique()
            color_discrete_map = create_categorical_color_map(unique_values)
            mode = 'markers+text' if labels_data is not None else 'markers'

            fig_scatter = px.scatter(
                x=x_data,
                y=y_data,
                color=color_data,
                text=text_labels if labels_data is not None else None,
                labels={'x': 'PC1', 'y': 'PC2'},
                color_discrete_map=color_discrete_map
            )
            fig_scatter.update_traces(
                marker=dict(size=8),
                mode=mode,
                textposition='top center',
                hovertemplate=hover_template,
                customdata=custom_data
            )

            # Transfer traces to main fig
            for trace in fig_scatter.data:
                fig.add_trace(trace)

        # Add trajectory line if requested (for colored plots)
        if show_trajectory and n_points > 1:
            fig.add_trace(go.Scatter(
                x=x_data,
                y=y_data,
                mode='lines',
                name='Trajectory',
                line=dict(color='lightgray', width=1, dash='dot'),
                showlegend=True,
                hoverinfo='skip'
            ))
    else:
        # No color data - use default styling or trajectory visualization
        if not show_trajectory:
            # No trajectory - just markers with blue-to-red gradient
            if COLORS_AVAILABLE:
                point_colors = get_sample_order_colors(n_points, colorscale='blue_to_red')
            else:
                point_colors = 'steelblue'  # Fallback

            mode = 'markers+text' if labels_data else 'markers'
            fig.add_trace(go.Scatter(
                x=x_data,
                y=y_data,
                mode=mode,
                name='Samples',
                text=text_labels if labels_data else None,
                marker=dict(size=4, color=point_colors, opacity=0.7),
                textposition='top center',
                hovertemplate=hover_template,
                customdata=custom_data
            ))
        else:
            # Show trajectory - use new trajectory color system
            if n_points > 1 and COLORS_AVAILABLE:
                # Get trajectory colors based on style
                traj_colors = get_trajectory_colors(
                    n_points=n_points,
                    style=trajectory_style,
                    last_point_marker=(trajectory_style == 'gradient')
                )

                # FOR SIMPLE STYLE: First add all data points as small colored scatter
                if trajectory_style == 'simple':
                    # Generate blue-to-red gradient colors based on sample order
                    point_colors = get_sample_order_colors(n_points, colorscale='blue_to_red')

                    fig.add_trace(go.Scatter(
                        x=x_data,
                        y=y_data,
                        mode='markers',
                        name='Test Samples',
                        marker=dict(color=point_colors, size=4, opacity=0.7),
                        hovertemplate=hover_template,
                        customdata=custom_data,
                        text=text_labels,
                        showlegend=True
                    ))

                # Draw trajectory segments
                for i in range(n_points - 1):
                    line_color = traj_colors['line_colors'][i]
                    line_width = traj_colors['line_widths'][i]
                    marker_color = traj_colors['marker_colors'][i]
                    marker_size = traj_colors['marker_sizes'][i]

                    # For simple style: only draw lines (no markers on segments)
                    # For gradient style: draw lines with markers
                    if trajectory_style == 'simple':
                        mode = 'lines'
                        marker_dict = None
                    else:
                        mode = 'lines+markers'
                        marker_dict = dict(color=marker_color, size=marker_size, opacity=0.8)

                    fig.add_trace(go.Scatter(
                        x=x_data[i:i+2],
                        y=y_data[i:i+2],
                        mode=mode,
                        name='Test Trajectory' if i == 0 else '',
                        line=dict(color=line_color, width=line_width),
                        marker=marker_dict,
                        hoverinfo='skip',  # Skip hover for trajectory lines
                        showlegend=(i == 0),
                        legendgroup='trajectory'
                    ))

                # Add last point marker (only for gradient style)
                if trajectory_style == 'gradient' and traj_colors['last_point_style'] is not None:
                    last_style = traj_colors['last_point_style']
                    last_sample_num = start_sample_num + n_points - 1

                    if timestamps is not None and len(timestamps) == n_points:
                        last_time = timestamps[-1].strftime('%Y-%m-%d %H:%M') if hasattr(timestamps[-1], 'strftime') else str(timestamps[-1])
                        last_hover = f'Last Sample {last_sample_num}<br>PC1: %{{x:.2f}}<br>PC2: %{{y:.2f}}<br>📅 {last_time}<extra></extra>'
                    else:
                        last_hover = f'Last Sample {last_sample_num}<br>PC1: %{{x:.2f}}<br>PC2: %{{y:.2f}}<extra></extra>'

                    fig.add_trace(go.Scatter(
                        x=[x_data[-1]],
                        y=[y_data[-1]],
                        mode='markers',
                        name='Last Point',
                        marker=dict(
                            color=last_style['color'],
                            size=last_style['size'],
                            symbol=last_style['symbol']
                        ),
                        hovertemplate=last_hover
                    ))

            elif n_points == 1:
                # Single point - use blue color
                if COLORS_AVAILABLE:
                    point_colors = get_sample_order_colors(1, colorscale='blue_to_red')
                    marker_color = point_colors[0]
                else:
                    marker_color = 'steelblue'

                fig.add_trace(go.Scatter(
                    x=x_data,
                    y=y_data,
                    mode='markers',
                    name='Test Sample',
                    marker=dict(color=marker_color, size=4),
                    hovertemplate=hover_template,
                    customdata=custom_data,
                    text=text_labels
                ))
            else:
                # Fallback if COLORS_AVAILABLE is False
                mode = 'markers+lines+text' if labels_data else 'markers+lines'
                fig.add_trace(go.Scatter(
                    x=x_data,
                    y=y_data,
                    mode=mode,
                    name='Samples',
                    text=text_labels if labels_data else None,
                    marker=dict(size=8, color='steelblue'),
                    line=dict(color='lightgray', width=1),
                    textposition='top center',
                    hovertemplate=hover_template,
                    customdata=custom_data
                ))

    # Add confidence ellipses if pca_params provided
    if pca_params is not None:
        if isinstance(pca_params, dict):
            n_train = pca_params.get('n_samples_train', 100)
            n_variables = pca_params.get('n_features', 10)

            # Convert explained_variance to array
            try:
                if isinstance(explained_variance, np.ndarray):
                    exp_var_array = explained_variance.flatten()
                elif isinstance(explained_variance, (list, tuple)):
                    exp_var_array = np.array(explained_variance)
                else:
                    exp_var_array = np.array([explained_variance[0], explained_variance[1]])
            except:
                exp_var_array = np.array([25.0, 15.0])  # Fallback

            if len(exp_var_array) < 2:
                exp_var_array = np.array([25.0, 15.0])

            # Variance explained PC1 and PC2 (as fraction, not percentage)
            var_pc1 = float(exp_var_array[0]) / 100.0
            var_pc2 = float(exp_var_array[1]) / 100.0

            # Confidence levels and F quantiles
            confidence_data = [
                (0.95, 2.996, 'green', 'solid', '95%'),
                (0.99, 4.605, 'orange', 'dash', '99%'),
                (0.999, 6.908, 'red', 'dot', '99.9%')
            ]

            theta = np.linspace(0, 2*np.pi, 100)

            for conf_level, f_value, color, dash, label in confidence_data:
                # Hotelling T² distribution formula
                correction_factor = np.sqrt(2 * (n_train**2 - 1) / (n_train * (n_train - 2)) * f_value)

                # Ellipse radii (R/MATLAB formulas)
                rad1 = np.sqrt(var_pc1 * ((n_train - 1) / n_train) * n_variables) * correction_factor
                rad2 = np.sqrt(var_pc2 * ((n_train - 1) / n_train) * n_variables) * correction_factor

                # Ellipse coordinates
                x_ellipse = rad1 * np.cos(theta)
                y_ellipse = rad2 * np.sin(theta)

                # Add ellipse to plot
                fig.add_trace(go.Scatter(
                    x=x_ellipse,
                    y=y_ellipse,
                    mode='lines',
                    name=f'{label} T² Ellipse',
                    line=dict(color=color, dash=dash, width=2),
                    showlegend=True,
                    hoverinfo='skip'
                ))

    # Layout
    pc1_var = float(exp_var_array[0]) if 'exp_var_array' in locals() else 25.0
    pc2_var = float(exp_var_array[1]) if 'exp_var_array' in locals() else 15.0

    fig.update_layout(
        title=f'PC1 vs PC2 ({pc1_var:.1f}% + {pc2_var:.1f}% = {pc1_var + pc2_var:.1f}% of total variance)',
        xaxis_title=f'PC1 ({pc1_var:.1f}% variance)',
        yaxis_title=f'PC2 ({pc2_var:.1f}% variance)',
        width=700,
        height=500,
        template='plotly_white'
    )

    # Equal axis scaling (important for score plots!)
    fig.update_yaxes(scaleanchor="x", scaleratio=1)

    return fig


def create_t2_q_plot(t2_values, q_values, t2_limits, q_limits, timestamps=None, start_sample_num=1, show_trajectory=True,
                     trajectory_style="simple", color_data=None, labels_data=None):
    """
    Create T²-Q influence plot with control limits and color support.

    Parameters:
    -----------
    t2_values : array-like
        T² statistic values
    q_values : array-like
        Q residual values
    t2_limits : list
        T² control limits [95%, 99%, 99.9%]
    q_limits : list
        Q control limits [95%, 99%, 99.9%]
    timestamps : list, optional
        Timestamps for samples
    start_sample_num : int
        Starting sample number for labeling
    show_trajectory : bool
        Whether to connect points with lines (default True)
    trajectory_style : str
        "simple" for light gray uniform line (default)
        "gradient" for blue→red gradient with cyan star on last point
    color_data : array-like, optional
        Data for coloring points (categorical or quantitative)
    labels_data : list, optional
        Sample labels to display on points

    Returns:
    --------
    go.Figure
        Plotly figure with T²-Q plot
    """
    import numpy as np
    import pandas as pd
    import plotly.express as px

    # Try to import color utilities
    try:
        from color_utils import (
            get_unified_color_schemes,
            create_categorical_color_map,
            is_quantitative_variable,
            get_trajectory_colors,
            get_sample_order_colors
        )
        COLORS_AVAILABLE = True
    except ImportError:
        COLORS_AVAILABLE = False

    fig = go.Figure()

    # Correct sample numbering
    sample_numbers_correct = [f"{start_sample_num + i}" for i in range(len(t2_values))]

    # Use labels_data if provided, otherwise use sample numbers
    if labels_data is not None and len(labels_data) == len(t2_values):
        text_labels = labels_data
    else:
        text_labels = sample_numbers_correct

    # Create dynamic hover template
    if timestamps is not None and len(timestamps) == len(t2_values):
        time_strings = [ts.strftime('%Y-%m-%d %H:%M') if hasattr(ts, 'strftime') else str(ts) for ts in timestamps]
        hover_template = 'Sample: %{text}<br>T²: %{x:.2f}<br>Q: %{y:.2f}<br>📅 %{customdata}<extra></extra>'
        custom_data = time_strings
    else:
        hover_template = 'Sample: %{text}<br>T²: %{x:.2f}<br>Q: %{y:.2f}<extra></extra>'
        custom_data = None

    # Prepare data for plotting
    n_points = len(t2_values)

    # Create scatter with unified color support
    if color_data is not None and len(color_data) == len(t2_values) and COLORS_AVAILABLE:
        color_series = pd.Series(color_data)

        if is_quantitative_variable(color_series):
            # Quantitative: Use blue-to-red continuous scale
            mode = 'markers+text' if labels_data is not None else 'markers'

            fig_scatter = px.scatter(
                x=t2_values,
                y=q_values,
                color=color_data,
                text=text_labels if labels_data is not None else None,
                labels={'x': 'T² Statistic', 'y': 'Q Residual'},
                color_continuous_scale=[(0.0, 'rgb(0, 0, 255)'),
                                       (0.5, 'rgb(128, 0, 128)'),
                                       (1.0, 'rgb(255, 0, 0)')]
            )
            fig_scatter.update_traces(
                marker=dict(size=8),
                mode=mode,
                textposition='top center',
                hovertemplate=hover_template,
                customdata=custom_data
            )

            # Transfer traces to main fig
            for trace in fig_scatter.data:
                fig.add_trace(trace)
        else:
            # Categorical: Use discrete color map
            unique_values = color_series.dropna().unique()
            color_discrete_map = create_categorical_color_map(unique_values)
            mode = 'markers+text' if labels_data is not None else 'markers'

            fig_scatter = px.scatter(
                x=t2_values,
                y=q_values,
                color=color_data,
                text=text_labels if labels_data is not None else None,
                labels={'x': 'T² Statistic', 'y': 'Q Residual'},
                color_discrete_map=color_discrete_map
            )
            fig_scatter.update_traces(
                marker=dict(size=8),
                mode=mode,
                textposition='top center',
                hovertemplate=hover_template,
                customdata=custom_data
            )

            # Transfer traces to main fig
            for trace in fig_scatter.data:
                fig.add_trace(trace)

        # Add trajectory if needed
        if show_trajectory and n_points > 1:
            fig.add_trace(go.Scatter(
                x=t2_values,
                y=q_values,
                mode='lines',
                name='Trajectory',
                line=dict(color='lightgray', width=1, dash='dot'),
                showlegend=True,
                hoverinfo='skip'
            ))
    else:
        # No color data - use default styling or trajectory visualization
        if not show_trajectory:
            # No trajectory - just markers with blue-to-red gradient
            if COLORS_AVAILABLE:
                point_colors = get_sample_order_colors(n_points, colorscale='blue_to_red')
            else:
                point_colors = 'steelblue'  # Fallback

            mode = 'markers+text' if labels_data else 'markers'
            fig.add_trace(go.Scatter(
                x=t2_values,
                y=q_values,
                mode=mode,
                name='T²-Q Points',
                text=text_labels if labels_data else None,
                marker=dict(size=4, color=point_colors, opacity=0.7),
                textposition='top center',
                hovertemplate=hover_template,
                customdata=custom_data
            ))
        else:
            # Show trajectory - use new trajectory color system
            if n_points > 1 and COLORS_AVAILABLE:
                # Get trajectory colors based on style
                traj_colors = get_trajectory_colors(
                    n_points=n_points,
                    style=trajectory_style,
                    last_point_marker=(trajectory_style == 'gradient')
                )

                # FOR SIMPLE STYLE: First add all data points as small colored scatter
                if trajectory_style == 'simple':
                    # Generate blue-to-red gradient colors based on sample order
                    point_colors = get_sample_order_colors(n_points, colorscale='blue_to_red')

                    fig.add_trace(go.Scatter(
                        x=t2_values,
                        y=q_values,
                        mode='markers',
                        name='T²-Q Points',
                        marker=dict(color=point_colors, size=4, opacity=0.7),
                        hovertemplate=hover_template,
                        customdata=custom_data,
                        text=text_labels,
                        showlegend=True
                    ))

                # Draw trajectory segments
                for i in range(n_points - 1):
                    line_color = traj_colors['line_colors'][i]
                    line_width = traj_colors['line_widths'][i]
                    marker_color = traj_colors['marker_colors'][i]
                    marker_size = traj_colors['marker_sizes'][i]

                    # For simple style: only draw lines (no markers on segments)
                    # For gradient style: draw lines with markers
                    if trajectory_style == 'simple':
                        mode = 'lines'
                        marker_dict = None
                    else:
                        mode = 'lines+markers'
                        marker_dict = dict(color=marker_color, size=marker_size, opacity=0.8)

                    fig.add_trace(go.Scatter(
                        x=t2_values[i:i+2],
                        y=q_values[i:i+2],
                        mode=mode,
                        name='T²-Q Trajectory' if i == 0 else '',
                        line=dict(color=line_color, width=line_width),
                        marker=marker_dict,
                        hoverinfo='skip',  # Skip hover for trajectory lines
                        showlegend=(i == 0),
                        legendgroup='trajectory_t2q'
                    ))

                # Add last point marker (only for gradient style)
                if trajectory_style == 'gradient' and traj_colors['last_point_style'] is not None:
                    last_style = traj_colors['last_point_style']
                    last_sample_num = start_sample_num + n_points - 1

                    if timestamps is not None and len(timestamps) == n_points:
                        last_time = timestamps[-1].strftime('%Y-%m-%d %H:%M') if hasattr(timestamps[-1], 'strftime') else str(timestamps[-1])
                        last_hover = f'Last Sample {last_sample_num}<br>T²: %{{x:.2f}}<br>Q: %{{y:.2f}}<br>📅 {last_time}<extra></extra>'
                    else:
                        last_hover = f'Last Sample {last_sample_num}<br>T²: %{{x:.2f}}<br>Q: %{{y:.2f}}<extra></extra>'

                    fig.add_trace(go.Scatter(
                        x=[t2_values[-1]],
                        y=[q_values[-1]],
                        mode='markers',
                        name='Last Point',
                        marker=dict(
                            color=last_style['color'],
                            size=last_style['size'],
                            symbol=last_style['symbol']
                        ),
                        hovertemplate=last_hover
                    ))

            elif n_points == 1:
                # Single point - use blue color
                if COLORS_AVAILABLE:
                    point_colors = get_sample_order_colors(1, colorscale='blue_to_red')
                    marker_color = point_colors[0]
                else:
                    marker_color = 'steelblue'

                fig.add_trace(go.Scatter(
                    x=t2_values,
                    y=q_values,
                    mode='markers',
                    name='T²-Q Point',
                    marker=dict(color=marker_color, size=4),
                    hovertemplate=hover_template,
                    customdata=custom_data,
                    text=text_labels
                ))
            else:
                # Fallback if COLORS_AVAILABLE is False
                mode = 'markers+lines+text' if labels_data else 'markers+lines'
                fig.add_trace(go.Scatter(
                    x=t2_values,
                    y=q_values,
                    mode=mode,
                    name='Samples',
                    text=text_labels if labels_data else None,
                    marker=dict(size=8, color='steelblue'),
                    line=dict(color='lightgray', width=1),
                    textposition='top center',
                    hovertemplate=hover_template,
                    customdata=custom_data
                ))

    # Calculate adaptive range
    if len(t2_values) > 0 and len(q_values) > 0:
        data_max_t2 = max(t2_values)
        data_max_q = max(q_values)

        green_limit_t2 = t2_limits[0]
        green_limit_q = q_limits[0]

        max_t2 = max(data_max_t2, green_limit_t2) * 1.15
        max_q = max(data_max_q, green_limit_q) * 1.15

        min_t2_range = green_limit_t2 * 1.2
        min_q_range = green_limit_q * 1.2

        max_t2 = max(max_t2, min_t2_range)
        max_q = max(max_q, min_q_range)
    else:
        max_t2 = t2_limits[0] * 1.3
        max_q = q_limits[0] * 1.3

    # Add control limits
    confidence_levels = ['97.5%', '99.5%', '99.95%']
    colors = ['green', 'orange', 'red']
    dash_styles = ['solid', 'dash', 'dot']

    for i, (conf, color, dash) in enumerate(zip(confidence_levels, colors, dash_styles)):
        # T² limits
        fig.add_shape(
            type="line",
            x0=t2_limits[i], y0=0,
            x1=t2_limits[i], y1=max_q,
            line=dict(color=color, width=2, dash=dash),
        )

        # Q limits
        fig.add_shape(
            type="line",
            x0=0, y0=q_limits[i],
            x1=max_t2, y1=q_limits[i],
            line=dict(color=color, width=2, dash=dash),
        )

        # Annotations only for 95%
        if i == 0:
            fig.add_annotation(
                x=t2_limits[i], y=max_q * 0.9,
                text=f"T² {conf} = {t2_limits[i]:.2f}",
                showarrow=True,
                arrowhead=2
            )

            fig.add_annotation(
                x=max_t2 * 0.8, y=q_limits[i],
                text=f"Q {conf} = {q_limits[i]:.2f}",
                showarrow=True,
                arrowhead=2
            )

    fig.update_layout(
        title='Boxes define acceptancy regions at 97.5%, 99.5%, 99.95% limits',
        xaxis_title='T² Statistic',
        yaxis_title='Q Statistic',
        width=700,
        height=500,
        template='plotly_white',
        xaxis=dict(range=[0, max_t2]),
        yaxis=dict(range=[0, max_q])
    )

    return fig


def calculate_t2_statistic_process(scores, explained_variance_pct, n_samples_train, n_variables):
    """
    DEPRECATED: Use calculate_hotelling_t2 from pca_utils.pca_statistics instead.

    This function is kept for backward compatibility but delegates to the correct implementation.
    """
    # Import correct function
    from pca_utils.pca_statistics import calculate_hotelling_t2

    # Need to reconstruct eigenvalues from explained variance
    # eigenvalue = explained_variance * (n-1)
    total_var = n_variables  # For scaled data, total variance = n_variables
    eigenvalues = (explained_variance_pct / 100.0) * total_var * (n_samples_train - 1)

    # Use correct T² calculation
    t2_values, _ = calculate_hotelling_t2(scores, eigenvalues, alpha=0.95)

    return t2_values


def calculate_q_statistic_process(test_scaled, scores, loadings):
    """
    DEPRECATED: Use calculate_q_residuals from pca_utils.pca_statistics instead.

    This function is kept for backward compatibility but delegates to the correct implementation.
    """
    # Import correct function
    from pca_utils.pca_statistics import calculate_q_residuals

    # Use correct Q calculation
    q_values, _ = calculate_q_residuals(test_scaled, scores, loadings, alpha=0.95)

    return q_values


def create_time_control_charts(t2_values, q_values, timestamps, t2_limits, q_limits, start_sample_num=1):
    """
    Create T² and Q control charts over time (from process_monitoring.py).
    """
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('T² Control Chart', 'Q Control Chart'),
        vertical_spacing=0.12,
        shared_xaxes=True
    )

    sample_numbers_correct = [f"{start_sample_num + i}" for i in range(len(t2_values))]

    # T² plot
    fig.add_trace(
        go.Scatter(
            x=timestamps if timestamps is not None else [start_sample_num + i for i in range(len(t2_values))],
            y=t2_values,
            mode='lines+markers',
            name='T² Values',
            line=dict(color='blue', width=1.5),
            marker=dict(size=3),
            text=sample_numbers_correct,
            hovertemplate='Sample: %{text}<br>T²: %{y:.2f}<br>Time: %{x}<extra></extra>' if timestamps is not None
                         else 'Sample: %{text}<br>T²: %{y:.2f}<extra></extra>'
        ),
        row=1, col=1
    )

    # T² limits
    confidence_levels = ['97.5%', '99.5%', '99.95%']
    colors = ['green', 'orange', 'red']
    dash_styles = ['solid', 'dash', 'dot']

    for i, (conf, color, dash) in enumerate(zip(confidence_levels, colors, dash_styles)):
        fig.add_hline(
            y=t2_limits[i],
            line_dash=dash,
            line_color=color,
            annotation_text=f"T² {conf} = {t2_limits[i]:.2f}" if i == 0 else f"{conf}",
            row=1, col=1
        )

    # Q plot
    fig.add_trace(
        go.Scatter(
            x=timestamps if timestamps is not None else [start_sample_num + i for i in range(len(q_values))],
            y=q_values,
            mode='lines+markers',
            name='Q Values',
            line=dict(color='green', width=1.5),
            marker=dict(size=3),
            text=sample_numbers_correct,
            hovertemplate='Sample: %{text}<br>Q: %{y:.2f}<br>Time: %{x}<extra></extra>' if timestamps is not None
                         else 'Sample: %{text}<br>Q: %{y:.2f}<extra></extra>'
        ),
        row=2, col=1
    )

    # Q limits
    for i, (conf, color, dash) in enumerate(zip(confidence_levels, colors, dash_styles)):
        fig.add_hline(
            y=q_limits[i],
            line_dash=dash,
            line_color=color,
            annotation_text=f"Q {conf} = {q_limits[i]:.2f}" if i == 0 else f"{conf}",
            row=2, col=1
        )

    # Mark outliers
    outliers_t2_95 = np.where(t2_values > t2_limits[0])[0]
    outliers_q_95 = np.where(q_values > q_limits[0])[0]

    if len(outliers_t2_95) > 0:
        outlier_sample_numbers_correct = [f"{start_sample_num + i}" for i in outliers_t2_95]
        fig.add_trace(
            go.Scatter(
                x=[timestamps[i] if timestamps is not None else start_sample_num + i for i in outliers_t2_95],
                y=t2_values[outliers_t2_95],
                mode='markers',
                name='T² Outliers (97.5%)',
                marker=dict(color='red', size=8, symbol='x'),
                showlegend=True,
                text=outlier_sample_numbers_correct,
                hovertemplate='🚨 T² OUTLIER<br>Sample: %{text}<br>T²: %{y:.2f}<br>Time: %{x}<extra></extra>' if timestamps is not None
                             else '🚨 T² OUTLIER<br>Sample: %{text}<br>T²: %{y:.2f}<extra></extra>'
            ),
            row=1, col=1
        )

    if len(outliers_q_95) > 0:
        outlier_sample_numbers_correct = [f"{start_sample_num + i}" for i in outliers_q_95]
        fig.add_trace(
            go.Scatter(
                x=[timestamps[i] if timestamps is not None else start_sample_num + i for i in outliers_q_95],
                y=q_values[outliers_q_95],
                mode='markers',
                name='Q Outliers (97.5%)',
                marker=dict(color='red', size=8, symbol='x'),
                showlegend=True,
                text=outlier_sample_numbers_correct,
                hovertemplate='🚨 Q OUTLIER<br>Sample: %{text}<br>Q: %{y:.2f}<br>Time: %{x}<extra></extra>' if timestamps is not None
                             else '🚨 Q OUTLIER<br>Sample: %{text}<br>Q: %{y:.2f}<extra></extra>'
            ),
            row=2, col=1
        )

    fig.update_layout(
        title='T² and Q Control Charts Over Time (97.5%, 99.5%, 99.95% limits)',
        height=600,
        template='plotly_white',
        hovermode='x unified'
    )

    fig.update_xaxes(title_text="Time" if timestamps is not None else "Sample Number", row=2, col=1)
    fig.update_yaxes(title_text="T² Statistic", row=1, col=1)
    fig.update_yaxes(title_text="Q Statistic", row=2, col=1)

    return fig


def calculate_all_contributions(test_scaled, scores, loadings, pca_params, debug=False):
    """
    Calculate Q and T² contributions for all samples (from process_monitoring.py).

    Parameters:
    -----------
    test_scaled : array-like, shape (n_samples, n_variables)
        Preprocessed test data (centered/scaled)
    scores : array-like, shape (n_samples, n_components)
        PCA scores for test data (USE THIS, don't recalculate!)
    loadings : array-like, shape (n_variables, n_components)
        PCA loadings matrix P
    pca_params : dict
        Dictionary with 's' key containing training scores
    debug : bool
        If True, print diagnostic information

    Returns:
    --------
    q_contrib : array-like, shape (n_samples, n_variables)
        Q (SPE) contributions per variable
    t2_contrib : array-like, shape (n_samples, n_variables)
        T² contributions per variable
    """
    n_samples, n_variables = test_scaled.shape
    n_components = scores.shape[1]

    # Get loadings in correct format (variables × components)
    if loadings.shape[0] == n_variables:
        P = loadings
    else:
        P = loadings.T

    # Get training scores standard deviations
    s_train = pca_params['s']
    Ls = np.std(s_train, axis=0)  # Should give values around 1-3, not huge numbers

    if debug:
        print("\n=== CONTRIBUTION CALCULATION DEBUG ===")
        print(f"Test data shape: {test_scaled.shape}")
        print(f"Scores shape: {scores.shape}")
        print(f"Loadings shape: {P.shape}")
        print(f"Training scores std (Ls): {Ls}")
        print(f"Ls range: [{Ls.min():.4f}, {Ls.max():.4f}]")

    # USE PASSED SCORES (don't recalculate!)
    # Old buggy code: scores_calc = test_scaled @ P
    # This was recalculating scores instead of using the passed parameter!

    # Reconstruction using PASSED scores
    reconstruction = scores @ P.T

    # Q contributions: sign(residuals) * (residuals^2)
    residuals = test_scaled - reconstruction
    q_contrib = np.sign(residuals) * (residuals ** 2)

    # T² contributions: test_scaled @ P @ diag(1/Ls) @ P.T
    # This is equivalent to: test_scaled @ MT, where MT = P @ diag(1/Ls) @ P.T
    MT = P @ np.diag(1.0 / Ls) @ P.T
    t2_contrib = test_scaled @ MT

    if debug:
        print(f"\n=== FIRST SAMPLE CONTRIBUTIONS (BEFORE NORMALIZATION) ===")
        print(f"Q contributions range: [{q_contrib[0].min():.6f}, {q_contrib[0].max():.6f}]")
        print(f"T² contributions range: [{t2_contrib[0].min():.6f}, {t2_contrib[0].max():.6f}]")
        print(f"Q contrib sample: {q_contrib[0, :3]}")  # First 3 variables
        print(f"T² contrib sample: {t2_contrib[0, :3]}")  # First 3 variables

        # Manual calculation check for first variable, first sample
        print(f"\n=== MANUAL CHECK (Sample 0, Variable 0) ===")
        print(f"Residual: {residuals[0, 0]:.6f}")
        print(f"Q contrib formula: sign({residuals[0, 0]:.6f}) * ({residuals[0, 0]:.6f})^2 = {q_contrib[0, 0]:.6f}")
        print(f"T² contrib: {t2_contrib[0, 0]:.6f}")

    return q_contrib, t2_contrib


def create_contribution_plot_all_vars(contrib_values, variable_names, statistic='Q'):
    """
    Create contribution bar plot showing ALL variables in ORIGINAL order.
    Red bars: |contrib|>1, Blue bars: |contrib|<1
    Threshold line at ±1 (normalized by 95th percentile of training set)
    """
    # Keep original order - NO SORTING
    # Convert variable names for display: if numeric (0,1,2...), show as 1-based (1,2,3...)
    display_vars = []
    for var in variable_names:
        try:
            # Try to convert to int - if it works and equals the float version, it's a numeric index
            var_int = int(var)
            var_float = float(var)
            if var_int == var_float:  # It's a numeric index like 0, 1, 2...
                display_vars.append(str(var_int + 1))  # Convert to 1-based: 0→1, 1→2, etc.
            else:
                display_vars.append(str(var))  # Keep as-is
        except (ValueError, TypeError):
            # Not numeric, keep as-is
            display_vars.append(str(var))

    # Color based on |contrib|>1: red if exceeds threshold, blue otherwise
    colors = ['red' if abs(val) > 1.0 else 'blue' for val in contrib_values]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=display_vars,  # Use display names (1-based if numeric) in ORIGINAL order
        y=contrib_values,  # Original order
        marker_color=colors,
        name=f'{statistic} Contributions',
        hovertemplate='%{x}<br>Contribution: %{y:.3f}<extra></extra>'
    ))

    # Threshold lines at ±1 (95th percentile of training set)
    fig.add_hline(y=1.0, line_dash="dash", line_color="black", annotation_text="Threshold +1 (95th pct)")
    fig.add_hline(y=-1.0, line_dash="dash", line_color="black", annotation_text="Threshold -1 (95th pct)")
    fig.add_hline(y=0.0, line_dash="solid", line_color="grey", line_width=1)

    fig.update_layout(
        title=f'{statistic} Contributions - All Variables (original order)',
        xaxis_title='Variable',
        yaxis_title=f'{statistic} Contribution (normalized by 95th percentile)',
        height=600,
        template='plotly_white',
        showlegend=False,
        xaxis=dict(
            tickangle=-90,
            tickmode='linear'
        )
    )

    return fig


def create_correlation_scatter(X_train, X_test, X_sample, var1_idx, var2_idx,
                               var1_name, var2_name, correlation_val, sample_idx):
    """
    Create correlation scatter plot showing training (grey), test (blue), sample (red star).
    """
    fig = go.Figure()

    # Training set (grey, sampled for performance)
    n_sample = min(1000, X_train.shape[0])
    sample_indices = np.random.choice(X_train.shape[0], n_sample, replace=False)

    fig.add_trace(go.Scatter(
        x=X_train[sample_indices, var1_idx],
        y=X_train[sample_indices, var2_idx],
        mode='markers',
        name='Training',
        marker=dict(color='lightgrey', size=4, opacity=0.5),
        hovertemplate=f'{var1_name}: %{{x:.2f}}<br>{var2_name}: %{{y:.2f}}<extra></extra>'
    ))

    # Test set (blue)
    fig.add_trace(go.Scatter(
        x=X_test[:, var1_idx],
        y=X_test[:, var2_idx],
        mode='markers',
        name='Test',
        marker=dict(color='blue', size=5, opacity=0.6),
        hovertemplate=f'{var1_name}: %{{x:.2f}}<br>{var2_name}: %{{y:.2f}}<extra></extra>'
    ))

    # Selected sample (red star)
    fig.add_trace(go.Scatter(
        x=[X_sample[var1_idx]],
        y=[X_sample[var2_idx]],
        mode='markers',
        name=f'Sample {sample_idx+1}',
        marker=dict(color='red', size=15, symbol='star'),
        hovertemplate=f'{var1_name}: %{{x:.2f}}<br>{var2_name}: %{{y:.2f}}<br>Sample {sample_idx+1}<extra></extra>'
    ))

    fig.update_layout(
        title=f'{var1_name} vs {var2_name}<br><sub>Correlation (training): r = {correlation_val:.3f}</sub>',
        xaxis_title=var1_name,
        yaxis_title=var2_name,
        height=400,
        template='plotly_white',
        showlegend=True
    )

    return fig


# ============================================================================
# STREAMLIT PAGE
# ============================================================================

def show():
    """Display the PCA Quality Control page"""

    st.markdown("# 📊 PCA Quality Control")
    st.markdown("*Statistical Quality Control using PCA (same computation as PCA menu)*")

    # Introduction
    with st.expander("ℹ️ About PCA Quality Control", expanded=False):
        st.markdown("""
        **PCA-based Statistical Quality Control (MSPC)** enables real-time monitoring of multivariate processes.

        **Key Features:**
        - **Same PCA computation as PCA menu** - Uses `compute_pca` from pca_utils
        - **T² Statistic (Hotelling)**: Detects unusual patterns within the model space
        - **Q Statistic (SPE)**: Detects deviations from the model structure
        - **Multiple Control Limits**: 97.5%, 99.5%, 99.95% confidence levels
        - **Score Plots**: With T² confidence ellipses
        - **Influence Plots**: T² vs Q for fault classification
        - **Model Persistence**: Save and load trained models
        """)

    # Main tabs - ADD SCORE PLOTS TAB
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔧 Model Training",
        "📊 Score Plots & Diagnostics",  # NEW TAB
        "🔍 Testing & Monitoring",
        "💾 Model Management"
    ])

    # ===== TAB 1: MODEL TRAINING =====
    with tab1:
        st.markdown("## 🔧 Train Monitoring Model")
        st.markdown("*Build PCA model using the same computation as PCA menu*")

        # Data source selection using workspace selector
        st.markdown("### 📊 Select Training Data")

        # Use workspace selector
        result = display_workspace_dataset_selector(
            label="Select training dataset from workspace:",
            key="qc_training_data_selector",
            help_text="Choose a dataset to train the quality control model",
            show_info=True
        )

        train_data = None
        train_dataset_name = None

        if result is not None:
            train_dataset_name, train_data = result
            st.success(f"✅ Selected: **{train_dataset_name}**")

        # Show data preview and variable selection
        if train_data is not None:
            st.markdown("### 🎯 Variable Selection")

            # Identify numeric columns
            numeric_cols = train_data.select_dtypes(include=[np.number]).columns.tolist()
            non_numeric_cols = train_data.select_dtypes(exclude=[np.number]).columns.tolist()
            all_columns = train_data.columns.tolist()

            # Dataset info (matching pca.py style)
            if len(numeric_cols) > 0:
                first_numeric_pos = all_columns.index(numeric_cols[0]) + 1
                last_numeric_pos = all_columns.index(numeric_cols[-1]) + 1
            else:
                first_numeric_pos = 1
                last_numeric_pos = 1

            st.info(f"Dataset: {len(all_columns)} total columns, {len(numeric_cols)} numeric (positions {first_numeric_pos}-{last_numeric_pos})")

            # === COLUMN SELECTION (Variables) ===
            st.markdown("#### 📊 Column Selection (Variables)")

            col1, col2 = st.columns(2)

            with col1:
                first_var = st.number_input(
                    "First column (1-based):",
                    min_value=1,
                    max_value=len(all_columns),
                    value=first_numeric_pos,
                    key="monitor_first_col"
                )

            with col2:
                last_var = st.number_input(
                    "Last column (1-based):",
                    min_value=first_var,
                    max_value=len(all_columns),
                    value=last_numeric_pos,
                    key="monitor_last_col"
                )

            # Get selected columns
            selected_cols = all_columns[first_var-1:last_var]
            # Filter only numeric columns
            selected_vars = [col for col in selected_cols if col in numeric_cols]
            n_selected_vars = len(selected_vars)

            st.info(f"Will analyze {n_selected_vars} variables (from column {first_var} to {last_var})")

            # === ROW SELECTION (Objects/Samples) ===
            st.markdown("#### 🎯 Row Selection (Objects/Samples)")

            n_samples = len(train_data)

            col3, col4 = st.columns(2)

            with col3:
                first_sample = st.number_input(
                    "First sample (1-based):",
                    min_value=1,
                    max_value=n_samples,
                    value=1,
                    key="monitor_first_sample"
                )

            with col4:
                last_sample = st.number_input(
                    "Last sample (1-based):",
                    min_value=first_sample,
                    max_value=n_samples,
                    value=n_samples,
                    key="monitor_last_sample"
                )

            # Get selected samples
            selected_sample_indices = list(range(first_sample-1, last_sample))
            n_selected_samples = len(selected_sample_indices)

            st.info(f"Will analyze {n_selected_samples} samples (from sample {first_sample} to {last_sample})")

            if len(selected_vars) == 0:
                st.warning("⚠️ Please select at least one variable (check your column range)")
            else:
                # Prepare training matrix (use selected rows and columns)
                X_train = train_data.iloc[selected_sample_indices][selected_vars]

                # Data preview
                with st.expander("👁️ Preview Training Data"):
                    st.dataframe(X_train.head(10), use_container_width=True)

                    st.markdown("**Basic Statistics:**")
                    stats_df = X_train.describe()
                    st.dataframe(stats_df, use_container_width=True)

                st.markdown("### ⚙️ Model Configuration")

                config_col1, config_col2, config_col3 = st.columns(3)

                with config_col1:
                    n_components = st.number_input(
                        "Number of components:",
                        min_value=1,
                        max_value=min(X_train.shape[0]-1, X_train.shape[1]),
                        value=min(5, X_train.shape[1]),
                        help="Number of principal components to retain"
                    )

                with config_col2:
                    scaling_method = st.selectbox(
                        "Data preprocessing:",
                        ["Center only", "Center + Scale (Auto)"],
                        help="Center only = mean centering, Auto = standardization"
                    )

                    center = True
                    scale = (scaling_method == "Center + Scale (Auto)")

                with config_col3:
                    st.markdown("**Control Limits:**")
                    st.info("97.5%, 99.5%, 99.95%")

                alpha_levels = [0.975, 0.995, 0.9995]

                # ===== PRETREATMENT DETECTION =====
                st.markdown("---")
                st.markdown("### 🔬 Pretreatment Detection")

                # Detect pretreatments from transformation history
                pretreat_info = None

                if train_dataset_name:
                    # Detect pretreatments for selected dataset
                    pretreat_info = detect_pretreatments(
                        train_dataset_name,
                        st.session_state.get('transformation_history', {})
                    )

                    if pretreat_info:
                        display_pretreatment_info(pretreat_info, context="training")
                    else:
                        st.info("📊 No pretreatments detected - using raw data for model training")

                # Train button
                st.markdown("---")

                if st.button("🚀 Train Monitoring Model", type="primary", use_container_width=True):
                    with st.spinner("Training PCA model (same as PCA menu)..."):
                        try:
                            # Use compute_pca from pca_utils (same as PCA menu)
                            pca_results = compute_pca(
                                X_train,
                                n_components=n_components,
                                center=center,
                                scale=scale
                            )

                            # Extract results
                            scores_train = pca_results['scores'].values
                            loadings = pca_results['loadings'].values
                            explained_variance = pca_results['explained_variance']
                            explained_variance_ratio = pca_results['explained_variance_ratio']

                            # Calculate explained variance as percentage
                            explained_variance_pct = explained_variance_ratio * 100

                            # Store in session state (including training data and pretreatment info)
                            st.session_state.pca_monitor_model = pca_results
                            st.session_state.pca_monitor_vars = selected_vars
                            st.session_state.pca_monitor_n_components = n_components
                            st.session_state.pca_monitor_center = center
                            st.session_state.pca_monitor_scale = scale
                            st.session_state.pca_monitor_trained = True
                            st.session_state.pca_monitor_training_data = X_train.copy()  # Store training data
                            st.session_state.pca_monitor_explained_variance_pct = explained_variance_pct  # For scree plot
                            st.session_state.pca_monitor_pretreat_info = pretreat_info  # Store pretreatment info for display

                            # Success message
                            st.success("✅ **Model trained successfully using PCA menu computation!**")

                            if pretreat_info:
                                st.info("📊 **Pretreatment detected** - remember to apply the same transformation to test data!")

                            # Display results
                            st.markdown("### 📊 Model Summary")

                            sum_col1, sum_col2, sum_col3, sum_col4 = st.columns(4)

                            with sum_col1:
                                st.metric("Components", n_components)
                            with sum_col2:
                                st.metric("Variables", len(selected_vars))
                            with sum_col3:
                                st.metric("Training Samples", X_train.shape[0])
                            with sum_col4:
                                st.metric("Variance Explained", f"{explained_variance_ratio.sum()*100:.1f}%")

                            # Variance per component
                            st.markdown("**Variance Explained per Component:**")
                            var_df = pd.DataFrame({
                                'Component': [f'PC{i+1}' for i in range(n_components)],
                                'Variance (%)': explained_variance_pct,
                                'Cumulative (%)': np.cumsum(explained_variance_pct)
                            })
                            st.dataframe(var_df, use_container_width=True)

                        except Exception as e:
                            st.error(f"❌ Error training model: {str(e)}")
                            import traceback
                            st.code(traceback.format_exc())

        # Show scree plot if model is trained (OUTSIDE button callback to avoid reset)
        if st.session_state.get('pca_monitor_trained', False):
            st.markdown("---")
            st.markdown("### 📊 Select Components for Monitoring (Scree Plot)")

            n_components = st.session_state.pca_monitor_n_components
            explained_variance_pct = st.session_state.pca_monitor_explained_variance_pct

            # Create scree plot
            fig_scree = go.Figure()

            # Bar plot of variance per component
            fig_scree.add_trace(go.Bar(
                x=[f'PC{i+1}' for i in range(n_components)],
                y=explained_variance_pct,
                name='Individual Variance',
                marker_color='steelblue',
                text=[f'{v:.1f}%' for v in explained_variance_pct],
                textposition='outside'
            ))

            # Line plot of cumulative variance
            cumulative_var = np.cumsum(explained_variance_pct)
            fig_scree.add_trace(go.Scatter(
                x=[f'PC{i+1}' for i in range(n_components)],
                y=cumulative_var,
                name='Cumulative Variance',
                mode='lines+markers',
                line=dict(color='red', width=2),
                marker=dict(size=8),
                yaxis='y2'
            ))

            # Dual y-axes
            fig_scree.update_layout(
                title='Scree Plot - Select Components for Monitoring',
                xaxis_title='Principal Component',
                yaxis=dict(title='Individual Variance (%)', side='left'),
                yaxis2=dict(
                    title='Cumulative Variance (%)',
                    overlaying='y',
                    side='right',
                    range=[0, 105]
                ),
                height=400,
                template='plotly_white',
                showlegend=True
            )

            st.plotly_chart(fig_scree, use_container_width=True)

            # Component selection slider (persistent)
            default_n = st.session_state.get('pca_monitor_n_components_selected', min(3, n_components))
            n_components_selected = st.slider(
                "Select number of components for monitoring:",
                min_value=2,
                max_value=n_components,
                value=default_n,
                help="Select how many principal components to use for T² and Q calculations",
                key="monitor_n_components_slider"
            )

            # Store selected N in session state
            st.session_state.pca_monitor_n_components_selected = n_components_selected

            # Show cumulative variance for selected components
            cumulative_selected = cumulative_var[n_components_selected - 1]
            st.info(f"✅ Using **{n_components_selected} components** → Cumulative variance: **{cumulative_selected:.1f}%**")

    # ===== TAB 2: SCORE PLOTS & DIAGNOSTICS =====
    with tab2:
        st.markdown("## 📊 Score Plots & Diagnostics")
        st.markdown("*Visualize PCA scores with T² ellipses and influence plots*")

        # Check if model is trained
        if 'pca_monitor_trained' not in st.session_state or not st.session_state.pca_monitor_trained:
            st.warning("⚠️ **No model trained yet.** Please train a model in the **Model Training** tab first.")
        else:
            pca_results = st.session_state.pca_monitor_model
            model_vars = st.session_state.pca_monitor_vars
            n_components = st.session_state.pca_monitor_n_components

            # Get selected N components (if user selected in scree plot)
            if 'pca_monitor_n_components_selected' in st.session_state:
                n_components_use = st.session_state.pca_monitor_n_components_selected
                st.success(f"✅ **Model loaded** ({len(model_vars)} variables, using **{n_components_use}/{n_components}** selected components)")
            else:
                n_components_use = n_components
                st.success(f"✅ **Model loaded** ({len(model_vars)} variables, {n_components} components)")
                st.info("💡 Tip: In Model Training tab, use scree plot to select N components for monitoring")

            # Automatically use training data (no data source selection)
            if 'pca_monitor_training_data' in st.session_state:
                X_plot = st.session_state.pca_monitor_training_data

                st.info(f"📊 Displaying training data: {X_plot.shape[0]} samples × {X_plot.shape[1]} variables")

                try:
                    # Preprocess data (same way as training)
                    X_plot_array = X_plot.values

                    if st.session_state.pca_monitor_scale:
                        # Use NIPALS means and stds
                        means = pca_results['means']
                        stds = pca_results['stds']
                        X_plot_scaled = (X_plot_array - means) / stds
                    elif st.session_state.pca_monitor_center:
                        # Use NIPALS means
                        means = pca_results['means']
                        X_plot_scaled = X_plot_array - means
                    else:
                        X_plot_scaled = X_plot_array

                    # Project to PCA space using manual calculation: T = X @ P
                    loadings_full = pca_results['loadings'].values
                    scores_plot_full = X_plot_scaled @ loadings_full

                    # Use only N selected components
                    scores_plot = scores_plot_full[:, :n_components_use]

                    # Get parameters (only for N selected components)
                    loadings_full = pca_results['loadings'].values
                    loadings = loadings_full[:, :n_components_use]

                    explained_variance_pct_full = pca_results['explained_variance_ratio'] * 100
                    explained_variance_pct = explained_variance_pct_full[:n_components_use]

                    n_samples_train = X_plot.shape[0]  # Use training data size for params
                    n_variables = len(model_vars)

                    # ===== IMPORT CORRECT FUNCTIONS =====
                    from pca_utils.pca_statistics import calculate_hotelling_t2, calculate_q_residuals
                    from scipy.stats import f, chi2

                    # Get eigenvalues for the selected components
                    eigenvalues_diag = pca_results['eigenvalues'][:n_components_use]

                    # ===== CALCULATE T² FOR ALL SAMPLES (Using Corrected Function) =====
                    # Use the already-corrected function for consistency
                    # This automatically applies the correct formula: T² = Σ(score_k² × (n-1) / λ_k)
                    # where λ_k = t't (sum of squared scores from NIPALS)
                    t2_values, _ = calculate_hotelling_t2(scores_plot, eigenvalues_diag, alpha=0.95)

                    # ===== CALCULATE Q FOR ALL SAMPLES (Using Corrected Function) =====
                    # Use the already-corrected function for consistency
                    q_values, _ = calculate_q_residuals(X_plot_scaled, scores_plot, loadings, alpha=0.95)

                    # ===== CALCULATE CONTROL LIMITS (3 levels) =====
                    # For Independent approach (97.5%, 99.5%, 99.95%)
                    alpha_levels = [0.975, 0.995, 0.9995]
                    confidence_labels = ['97.5%', '99.5%', '99.95%']

                    t2_limits = []
                    q_limits = []

                    for alpha in alpha_levels:
                        # Get T² and Q limits from corrected pca_utils functions
                        # This ensures consistency with all other calculations
                        _, t2_lim = calculate_hotelling_t2(scores_plot, eigenvalues_diag, alpha=alpha)
                        t2_limits.append(t2_lim)

                        _, q_lim = calculate_q_residuals(X_plot_scaled, scores_plot, loadings, alpha=alpha)
                        q_limits.append(q_lim)

                    # Prepare params for plotting
                    pca_params_plot = {
                        'n_samples_train': n_samples_train,
                        'n_features': n_variables
                    }

                    # Check for timestamps
                    timestamps = None
                    if 'timestamp' in X_plot.columns or 'Timestamp' in X_plot.columns:
                        timestamp_col = 'timestamp' if 'timestamp' in X_plot.columns else 'Timestamp'
                        timestamps = X_plot[timestamp_col].tolist()

                    # Create plots side by side (automatically, no button)
                    st.markdown("### 📊 Score Plot & T²-Q Influence Plot")

                    # Trajectory style controls
                    st.markdown("**Trajectory Visualization Options**")
                    traj_col1, traj_col2 = st.columns(2)

                    with traj_col1:
                        show_trajectory_score = st.checkbox("Show Trajectory", value=True, key="show_trajectory_score_tab2")
                        if show_trajectory_score:
                            trajectory_style_score = st.radio(
                                "Score Plot Trajectory Style",
                                options=['simple', 'gradient'],
                                format_func=lambda x: {
                                    'simple': 'Simple (Light Gray)',
                                    'gradient': 'Gradient (Blue→Red with Star)'
                                }[x],
                                index=0,
                                horizontal=True,
                                key="trajectory_style_score_tab2"
                            )
                        else:
                            trajectory_style_score = 'simple'

                    with traj_col2:
                        show_trajectory_t2q = st.checkbox("Show Trajectory", value=True, key="show_trajectory_t2q_tab2")
                        if show_trajectory_t2q:
                            trajectory_style_t2q = st.radio(
                                "T²-Q Plot Trajectory Style",
                                options=['simple', 'gradient'],
                                format_func=lambda x: {
                                    'simple': 'Simple (Light Gray)',
                                    'gradient': 'Gradient (Blue→Red with Star)'
                                }[x],
                                index=0,
                                horizontal=True,
                                key="trajectory_style_t2q_tab2"
                            )
                        else:
                            trajectory_style_t2q = 'simple'

                    plot_col1, plot_col2 = st.columns(2)

                    with plot_col1:
                        st.markdown("**PCA Score Plot (PC1 vs PC2)**")
                        fig_score = create_score_plot(
                            scores_plot,
                            explained_variance_pct,
                            timestamps=timestamps,
                            pca_params=pca_params_plot,
                            start_sample_num=1,
                            show_trajectory=show_trajectory_score,
                            trajectory_style=trajectory_style_score
                        )
                        st.plotly_chart(fig_score, use_container_width=True)

                    with plot_col2:
                        st.markdown(f"**T²-Q Influence Plot** (Calculated with {n_components_use} components)")
                        fig_t2q = create_t2_q_plot(
                            t2_values,
                            q_values,
                            t2_limits,
                            q_limits,
                            timestamps=timestamps,
                            start_sample_num=1,
                            show_trajectory=show_trajectory_t2q,
                            trajectory_style=trajectory_style_t2q
                        )
                        st.plotly_chart(fig_t2q, use_container_width=True)

                    # Statistics summary
                    st.markdown("### 📈 Statistics Summary")

                    stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)

                    with stat_col1:
                        st.metric("Max T²", f"{np.max(t2_values):.2f}")
                    with stat_col2:
                        st.metric("Max Q", f"{np.max(q_values):.2f}")
                    with stat_col3:
                        n_t2_outliers = (t2_values > t2_limits[0]).sum()
                        st.metric("T² Outliers", f"{n_t2_outliers} ({n_t2_outliers/len(t2_values)*100:.1f}%)")
                    with stat_col4:
                        n_q_outliers = (q_values > q_limits[0]).sum()
                        st.metric("Q Outliers", f"{n_q_outliers} ({n_q_outliers/len(q_values)*100:.1f}%)")

                    # ===== CONTROL CHARTS =====
                    st.markdown("---")
                    st.markdown("### 📊 Control Charts Over Time")

                    fig_control = create_time_control_charts(
                        t2_values,
                        q_values,
                        timestamps,
                        t2_limits,
                        q_limits,
                        start_sample_num=1
                    )
                    st.plotly_chart(fig_control, use_container_width=True)

                    # ===== CONTRIBUTION ANALYSIS =====
                    st.markdown("---")
                    st.markdown("### 🔬 Contribution Analysis")
                    st.markdown("*Analyze samples exceeding T²/Q control limits*")

                    # Find samples exceeding limits (97.5%)
                    t2_outliers = np.where(t2_values > t2_limits[0])[0]
                    q_outliers = np.where(q_values > q_limits[0])[0]
                    outlier_samples = np.unique(np.concatenate([t2_outliers, q_outliers]))

                    if len(outlier_samples) == 0:
                        st.success("✅ **No samples exceed control limits.** All samples are within normal operating conditions.")
                    else:
                        st.warning(f"⚠️ **{len(outlier_samples)} samples exceed control limits** (T²>limit OR Q>limit)")

                        # Calculate contributions (normalized by training set 95th percentile)
                        pca_params_contrib = {
                            's': pca_results['scores'].values[:, :n_components_use]
                        }

                        q_contrib, t2_contrib = calculate_all_contributions(
                            X_plot_scaled,
                            scores_plot,
                            loadings,
                            pca_params_contrib
                        )

                        # Normalize contributions by 95th percentile of training set
                        q_contrib_95th = np.percentile(np.abs(q_contrib), 95, axis=0)
                        t2_contrib_95th = np.percentile(np.abs(t2_contrib), 95, axis=0)

                        # Avoid division by zero
                        q_contrib_95th[q_contrib_95th == 0] = 1.0
                        t2_contrib_95th[t2_contrib_95th == 0] = 1.0

                        # Select sample from outliers only
                        sample_select_col, _ = st.columns([1, 1])
                        with sample_select_col:
                            sample_idx = st.selectbox(
                                "Select outlier sample for contribution analysis:",
                                options=outlier_samples,
                                format_func=lambda x: f"Sample {x+1} (T²={t2_values[x]:.2f}, Q={q_values[x]:.2f})",
                                key="train_contrib_sample"
                            )

                        # Get contributions for selected sample
                        q_contrib_sample = q_contrib[sample_idx, :]
                        t2_contrib_sample = t2_contrib[sample_idx, :]

                        # Normalize
                        q_contrib_norm = q_contrib_sample / q_contrib_95th
                        t2_contrib_norm = t2_contrib_sample / t2_contrib_95th

                        # Bar plots side by side (ALL variables, red if |contrib|>1, blue otherwise)
                        contrib_col1, contrib_col2 = st.columns(2)

                        with contrib_col1:
                            st.markdown(f"**T² Contributions - Sample {sample_idx+1}**")
                            fig_t2_contrib = create_contribution_plot_all_vars(
                                t2_contrib_norm,
                                model_vars,
                                statistic='T²'
                            )
                            st.plotly_chart(fig_t2_contrib, use_container_width=True)

                        with contrib_col2:
                            st.markdown(f"**Q Contributions - Sample {sample_idx+1}**")
                            fig_q_contrib = create_contribution_plot_all_vars(
                                q_contrib_norm,
                                model_vars,
                                statistic='Q'
                            )
                            st.plotly_chart(fig_q_contrib, use_container_width=True)

                        # Table: Variables where |contrib|>1 with real values vs training mean
                        st.markdown("### 🏆 Top Contributing Variables")
                        st.markdown("*Variables exceeding 95th percentile threshold (|contribution| > 1)*")

                        # Get training mean for comparison
                        training_mean = X_plot.mean()

                        # Get real values for selected sample
                        sample_values = X_plot.iloc[sample_idx]

                        # Filter variables where |contrib|>1 for either T² or Q
                        high_contrib_t2 = np.abs(t2_contrib_norm) > 1.0
                        high_contrib_q = np.abs(q_contrib_norm) > 1.0
                        high_contrib = high_contrib_t2 | high_contrib_q

                        if high_contrib.sum() > 0:
                            contrib_table_data = []
                            for i, var in enumerate(model_vars):
                                if high_contrib[i]:
                                    real_val = sample_values[var]
                                    mean_val = training_mean[var]
                                    diff = real_val - mean_val
                                    direction = "Higher ↑" if diff > 0 else "Lower ↓"

                                    contrib_table_data.append({
                                        'Variable': var,
                                        'Real Value': f"{real_val:.3f}",
                                        'Training Mean': f"{mean_val:.3f}",
                                        'Difference': f"{diff:.3f}",
                                        'Direction': direction,
                                        '|T² Contrib|': f"{abs(t2_contrib_norm[i]):.2f}",
                                        '|Q Contrib|': f"{abs(q_contrib_norm[i]):.2f}"
                                    })

                            contrib_table = pd.DataFrame(contrib_table_data)
                            # Sort by max absolute contribution
                            contrib_table['Max_Contrib'] = contrib_table.apply(
                                lambda row: max(float(row['|T² Contrib|']), float(row['|Q Contrib|'])),
                                axis=1
                            )
                            contrib_table = contrib_table.sort_values('Max_Contrib', ascending=False).drop('Max_Contrib', axis=1)

                            st.dataframe(contrib_table, use_container_width=True)
                        else:
                            st.info("No variables exceed the 95th percentile threshold.")

                        # Correlation scatter: training (grey), test (blue), sample (red star)
                        st.markdown("### 📈 Correlation Analysis - Top Q Contributor")
                        st.markdown("*Select from top Q contributors to see correlation with most correlated variable*")

                        # Get top Q contributors (variables with highest |Q contribution|)
                        q_contrib_abs = np.abs(q_contrib_norm)
                        top_q_indices = np.argsort(q_contrib_abs)[::-1][:5]
                        top_q_contributors = [model_vars[i] for i in top_q_indices]

                        # Dropdown to select from top Q contributors
                        corr_col1, corr_col2 = st.columns([2, 1])

                        with corr_col1:
                            selected_q_var = st.selectbox(
                                "Select from top Q contributors:",
                                options=top_q_contributors,
                                key="train_top_q_var"
                            )

                        # Calculate correlations for selected variable (from training data)
                        var1_idx = model_vars.index(selected_q_var)
                        correlations = {}
                        for i, var in enumerate(model_vars):
                            if var != selected_q_var:
                                corr = np.corrcoef(X_plot_array[:, var1_idx], X_plot_array[:, i])[0, 1]
                                correlations[var] = (corr, i)

                        # Find most correlated variable
                        most_corr_var = max(correlations, key=lambda k: abs(correlations[k][0]))
                        corr_coef, var2_idx = correlations[most_corr_var]

                        with corr_col2:
                            st.metric("Correlation (training)", f"{corr_coef:.4f}")

                        # Create 3-layer scatter plot (training=grey, test=blue, sample=red star)
                        # In Tab 2, training and test are the same, but we still show the structure
                        fig_corr_scatter = create_correlation_scatter(
                            X_train=X_plot_array,
                            X_test=X_plot_array,  # Same as training in Tab 2
                            X_sample=X_plot_array[sample_idx, :],
                            var1_idx=var1_idx,
                            var2_idx=var2_idx,
                            var1_name=selected_q_var,
                            var2_name=most_corr_var,
                            correlation_val=corr_coef,
                            sample_idx=sample_idx
                        )

                        st.plotly_chart(fig_corr_scatter, use_container_width=True)

                    # Store for other tabs
                    st.session_state.pca_monitor_plot_results = {
                        'scores': scores_plot,
                        't2': t2_values,
                        'q': q_values,
                        't2_limits': t2_limits,
                        'q_limits': q_limits,
                        'X_scaled': X_plot_scaled
                    }

                    # ========== EXPORT SECTION - TRAINING DATA ==========
                    st.markdown("---")
                    st.markdown("## 📥 Export Training Data Diagnostics")
                    st.markdown("*Export T², Q values, limits, and outlier diagnostics for training dataset*")

                    export_col1, export_col2, export_col3 = st.columns(3)

                    with export_col1:
                        st.markdown("### 📊 Basic Data Export")

                        # Prepare basic data export for training data
                        basic_export_df = pd.DataFrame({
                            'Sample': [f"Train_Sample_{i+1}" for i in range(len(t2_values))],
                            'T2': t2_values,
                            'Q': q_values,
                            'T2_Limit_97.5%': [t2_limits[0]] * len(t2_values),
                            'T2_Limit_99.5%': [t2_limits[1]] * len(t2_values),
                            'T2_Limit_99.95%': [t2_limits[2]] * len(t2_values),
                            'Q_Limit_97.5%': [q_limits[0]] * len(q_values),
                            'Q_Limit_99.5%': [q_limits[1]] * len(q_values),
                            'Q_Limit_99.95%': [q_limits[2]] * len(q_values)
                        })

                        # Convert to CSV
                        csv_basic = basic_export_df.to_csv(index=False)

                        st.download_button(
                            label="📥 Download T²-Q Values (CSV)",
                            data=csv_basic,
                            file_name=f"training_t2_q_values_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            help="Download training data T² and Q values with all critical limits"
                        )

                    with export_col2:
                        st.markdown("### 🔍 Independent Diagnostics")
                        st.markdown("*T² and Q evaluated independently*")

                        # Import monitoring functions
                        from pca_utils.pca_monitoring import export_monitoring_data_to_excel

                        # Prepare sample labels for training data
                        sample_labels_export = [f"Train_Sample_{i+1}" for i in range(len(t2_values))]

                        # Create Excel file with independent diagnostics
                        if st.button("📊 Generate Independent Report", key="btn_independent_tab2"):
                            with st.spinner("Generating independent diagnostics report..."):
                                excel_independent = export_monitoring_data_to_excel(
                                    t2_values=t2_values,
                                    q_values=q_values,
                                    t2_limits=t2_limits,
                                    q_limits=q_limits,
                                    sample_labels=sample_labels_export,
                                    approach='independent'
                                )

                                st.download_button(
                                    label="📥 Download Independent Report (Excel)",
                                    data=excel_independent,
                                    file_name=f"training_monitoring_independent_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                    help="Excel file with training data T² and Q outliers classified independently",
                                    key="download_independent_tab2"
                                )

                    with export_col3:
                        st.markdown("### 🎯 Joint Diagnostics")
                        st.markdown("*Hierarchical outlier classification*")

                        # Create Excel file with joint diagnostics
                        if st.button("📊 Generate Joint Report", key="btn_joint_tab2"):
                            with st.spinner("Generating joint diagnostics report..."):
                                excel_joint = export_monitoring_data_to_excel(
                                    t2_values=t2_values,
                                    q_values=q_values,
                                    t2_limits=t2_limits,
                                    q_limits=q_limits,
                                    sample_labels=sample_labels_export,
                                    approach='joint'
                                )

                                st.download_button(
                                    label="📥 Download Joint Report (Excel)",
                                    data=excel_joint,
                                    file_name=f"training_monitoring_joint_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                    help="Excel file with hierarchical outlier classification"
                                )

                    # Complete report with both approaches
                    st.markdown("---")
                    complete_col1, complete_col2 = st.columns([2, 1])

                    with complete_col1:
                        st.markdown("### 📦 Complete Training Data Report")
                        st.markdown("*Includes both independent and joint diagnostics for training data*")

                    with complete_col2:
                        if st.button("📊 Generate Complete Report", key="btn_complete_tab2", type="primary"):
                            with st.spinner("Generating complete monitoring report..."):
                                excel_complete = export_monitoring_data_to_excel(
                                    t2_values=t2_values,
                                    q_values=q_values,
                                    t2_limits=t2_limits,
                                    q_limits=q_limits,
                                    sample_labels=sample_labels_export,
                                    approach='both'
                                )

                                st.download_button(
                                    label="📥 Download Complete Report (Excel)",
                                    data=excel_complete,
                                    file_name=f"training_monitoring_complete_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                    help="Comprehensive Excel file with all training data diagnostics",
                                    key="download_complete_tab2"
                                )

                    # Info about report contents
                    with st.expander("📋 What's included in the reports?", expanded=False):
                        st.markdown("""
                        **Basic Data Export (CSV)**:
                        - T² and Q values for all samples
                        - Critical limits at all confidence levels (97.5%, 99.5%, 99.95%)

                        **Independent Diagnostics Report (Excel)**:
                        - Sheet 1: T² and Q values with limits
                        - Sheet 2: Occurrences summary (count of outliers per threshold)
                        - Sheet 3: T² outliers (samples exceeding T² limits)
                        - Sheet 4: Q outliers (samples exceeding Q limits)
                        - Sheet 5: Combined outliers (samples exceeding T² OR Q)

                        **Joint Diagnostics Report (Excel)**:
                        - Sheet 1: T² and Q values with limits
                        - Sheet 2: Occurrences summary (count per threshold level)
                        - Sheet 3: Hierarchical outlier classification
                          - *** (99.95%): Most severe outliers
                          - ** (99.5%): Moderate outliers
                          - * (97.5%): Mild outliers

                        **Complete Report (Excel)**:
                        - All sheets from both Independent and Joint diagnostics
                        - 7 sheets total with comprehensive outlier analysis

                        **Diagnostic Approaches**:
                        - **Independent**: T² and Q evaluated separately. A sample can be counted as outlier in both.
                        - **Joint**: Hierarchical classification. Each sample assigned to highest exceeded threshold only.
                        """)

                except Exception as e:
                    st.error(f"❌ Error generating plots: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
            else:
                st.warning("⚠️ **Training data not available.** Please retrain the model.")

    # ===== TAB 3: TESTING & MONITORING =====
    with tab3:
        st.markdown("## 🔍 Testing & Monitoring")
        st.markdown("*Project test data onto training model and detect faults*")

        # Check if model is trained
        if 'pca_monitor_trained' not in st.session_state or not st.session_state.pca_monitor_trained:
            st.warning("⚠️ **No model trained yet.** Please train a model in the **Model Training** tab first.")
        else:
            pca_results = st.session_state.pca_monitor_model
            model_vars = st.session_state.pca_monitor_vars
            n_components = st.session_state.pca_monitor_n_components

            # Get selected N components (if user selected in scree plot)
            if 'pca_monitor_n_components_selected' in st.session_state:
                n_components_use = st.session_state.pca_monitor_n_components_selected
                st.success(f"✅ **Model loaded** ({len(model_vars)} variables, using **{n_components_use}/{n_components}** selected components)")
            else:
                n_components_use = n_components
                st.success(f"✅ **Model loaded** ({len(model_vars)} variables, {n_components} components)")
                st.info("💡 Tip: In Model Training tab, use scree plot to select N components for monitoring")

            # Test data source - workspace selector
            st.markdown("### 📊 Select Test Data")

            # Use workspace selector
            test_result = display_workspace_dataset_selector(
                label="Select test dataset from workspace:",
                key="qc_test_data_selector",
                help_text="Choose a dataset to project onto the training model",
                show_info=True
            )

            test_data = None
            selected_dataset_name = None

            if test_result is not None:
                selected_dataset_name, test_data = test_result

                # Check if dataset has the required variables
                missing_vars = [v for v in model_vars if v not in test_data.columns]

                if len(missing_vars) > 0:
                    st.error(f"❌ **Dimension mismatch!** Missing variables: {missing_vars}")
                    st.warning(f"Training model requires: {model_vars}")
                    st.warning(f"Selected dataset has: {list(test_data.columns)}")
                    test_data = None  # Don't proceed with incompatible data
                else:
                    # Check if dataset has the same number of variables
                    test_vars_count = len([v for v in model_vars if v in test_data.columns])
                    st.success(f"✅ Dimension check passed: {test_vars_count} variables match")

            # Test the data
            if test_data is not None:
                # Check if required variables are present
                missing_vars = [v for v in model_vars if v not in test_data.columns]

                if len(missing_vars) > 0:
                    st.error(f"❌ **Missing variables in test data**: {missing_vars}")
                else:
                    st.info(f"📊 Full test data: {len(test_data)} samples × {len(model_vars)} variables")

                    # ===== ROW SELECTION (OBJECTS/SAMPLES) =====
                    st.markdown("### 🎯 Row Selection (Objects/Samples)")
                    st.info("Select a subset of samples from test data for analysis")

                    # Get total samples in test data
                    n_test_samples = len(test_data)

                    col1, col2 = st.columns(2)

                    with col1:
                        first_sample = st.number_input(
                            "First sample (1-based):",
                            min_value=1,
                            max_value=n_test_samples,
                            value=1,
                            key="test_first_sample"
                        )

                    with col2:
                        last_sample = st.number_input(
                            "Last sample (1-based):",
                            min_value=first_sample,
                            max_value=n_test_samples,
                            value=n_test_samples,  # Default to all samples
                            key="test_last_sample"
                        )

                    # Show range info
                    n_selected = last_sample - first_sample + 1
                    st.info(f"✅ Will analyze {n_selected} samples (from sample {first_sample} to {last_sample})")

                    # SELECT subset (convert 1-based to 0-based indexing)
                    test_data_subset = test_data.iloc[first_sample-1:last_sample]
                    X_test = test_data_subset[model_vars]

                    st.divider()

                    # ===== PRETREATMENT WARNING =====
                    # Check if pretreatment info exists from training
                    training_pretreat_info = st.session_state.get('pca_monitor_pretreat_info', None)

                    if training_pretreat_info is not None and training_pretreat_info.pretreatments:
                        # Display pretreatment comparison and warnings
                        display_pretreatment_warning(training_pretreat_info, selected_dataset_name)
                    else:
                        # No pretreatments on training data
                        st.markdown("---")
                        st.info("📊 No pretreatments detected on training data - ensure test data is also untransformed")

                    # Test button
                    st.markdown("---")
                    if st.button("🔍 Test Data on Model", type="primary", use_container_width=True):
                        with st.spinner("Testing data on model..."):
                            try:
                                # Preprocess test data (use training means/stds)
                                X_test_array = X_test.values

                                if st.session_state.pca_monitor_scale:
                                    # Use NIPALS means and stds from training
                                    means = pca_results['means']
                                    stds = pca_results['stds']
                                    X_test_scaled = (X_test_array - means) / stds
                                elif st.session_state.pca_monitor_center:
                                    # Use NIPALS means from training
                                    means = pca_results['means']
                                    X_test_scaled = X_test_array - means
                                else:
                                    X_test_scaled = X_test_array

                                # Project test data to PCA space using manual calculation: T = X @ P
                                loadings_full = pca_results['loadings'].values
                                scores_test_full = X_test_scaled @ loadings_full

                                # Use only N selected components
                                scores_test = scores_test_full[:, :n_components_use]

                                # Get parameters (only for N selected components)
                                loadings_full = pca_results['loadings'].values
                                loadings = loadings_full[:, :n_components_use]

                                explained_variance_pct_full = pca_results['explained_variance_ratio'] * 100
                                explained_variance_pct = explained_variance_pct_full[:n_components_use]

                                # Use TRAINING parameters for limits (not test data size!)
                                n_samples_train = pca_results['scores'].shape[0]  # Training set size
                                n_variables = len(model_vars)

                                # ===== IMPORT CORRECT FUNCTIONS =====
                                from pca_utils.pca_statistics import calculate_hotelling_t2, calculate_q_residuals
                                from scipy.stats import f, chi2
                                from pca_utils.pca_statistics import calculate_hotelling_t2_matricial

                                # Get eigenvalues for the selected components
                                eigenvalues_diag = pca_results['eigenvalues'][:n_components_use]

                                # ===== CALCULATE T² FOR TEST SAMPLES =====
                                # Use R/CAT matricial formula for test data projection
                                # This correctly handles centering/scaling with training statistics
                                X_train_mean = pca_results['means']
                                X_train_std = pca_results['stds'] if st.session_state.pca_monitor_scale else None

                                # Calculate T² using matricial method (matches R/CAT exactly)
                                t2_values, _ = calculate_hotelling_t2_matricial(
                                    X_test=X_test.values,
                                    loadings=loadings,
                                    eigenvalues=eigenvalues_diag,
                                    X_train_mean=X_train_mean,
                                    X_train_std=X_train_std,
                                    n_train=n_samples_train
                                )

                                # ===== CALCULATE Q FOR TEST SAMPLES =====
                                # Correct formula: Q = ||residuals||² = ||X - X_reconstructed||²
                                X_test_reconstructed = scores_test @ loadings.T
                                test_residuals = X_test_scaled - X_test_reconstructed
                                q_values = np.sum(test_residuals ** 2, axis=1)

                                # ===== CALCULATE CONTROL LIMITS BASED ON TRAINING DATA =====
                                # Limits must be calculated from TRAINING set, not test set
                                alpha_levels = [0.975, 0.995, 0.9995]

                                t2_limits = []
                                q_limits = []

                                # Get training scores for limit calculation
                                training_scores = pca_results['scores'].values[:, :n_components_use]

                                # Reconstruct training data preprocessing
                                X_train_for_limit = st.session_state.pca_monitor_training_data.values
                                if st.session_state.pca_monitor_scale:
                                    X_train_scaled_for_limit = (X_train_for_limit - pca_results['means']) / pca_results['stds']
                                elif st.session_state.pca_monitor_center:
                                    X_train_scaled_for_limit = X_train_for_limit - pca_results['means']
                                else:
                                    X_train_scaled_for_limit = X_train_for_limit

                                # Calculate training Q values for limit estimation
                                X_train_reconstructed = training_scores @ loadings.T
                                train_residuals = X_train_scaled_for_limit - X_train_reconstructed
                                q_values_train = np.sum(train_residuals ** 2, axis=1)

                                for alpha in alpha_levels:
                                    # T² limit using F-distribution (based on training set size)
                                    df1 = n_components_use
                                    df2 = n_samples_train - n_components_use
                                    f_value = f.ppf(alpha, df1, df2)
                                    t2_lim = ((n_samples_train - 1) * n_components_use / (n_samples_train - n_components_use)) * f_value
                                    t2_limits.append(t2_lim)

                                    # Q limit using chi-square approximation (based on training Q distribution)
                                    q_mean_train = np.mean(q_values_train)
                                    q_var_train = np.var(q_values_train, ddof=1)

                                    if q_var_train > 0 and q_mean_train > 0:
                                        g = q_var_train / (2 * q_mean_train)
                                        h = (2 * q_mean_train ** 2) / q_var_train
                                        q_lim = g * chi2.ppf(alpha, h)
                                    else:
                                        # Fallback to percentile if variance is zero
                                        q_lim = np.percentile(q_values_train, alpha * 100)
                                    q_limits.append(q_lim)

                                # Prepare params for plotting (use TRAINING params!)
                                pca_params_test = {
                                    'n_samples_train': n_samples_train,
                                    'n_features': n_variables
                                }

                                # Check for timestamps
                                timestamps = None
                                if 'timestamp' in test_data.columns or 'Timestamp' in test_data.columns:
                                    timestamp_col = 'timestamp' if 'timestamp' in test_data.columns else 'Timestamp'
                                    timestamps = test_data[timestamp_col].tolist()

                                # Count faults (calculate here for session state)
                                t2_faults = t2_values > t2_limits[0]
                                q_faults = q_values > q_limits[0]
                                total_faults = np.logical_or(t2_faults, q_faults)

                                # Store test results in session state for plots and contribution analysis
                                st.session_state.pca_monitor_test_results = {
                                    't2_values': t2_values,
                                    'q_values': q_values,
                                    't2_limits': t2_limits,
                                    'q_limits': q_limits,
                                    'X_test': X_test.copy(),
                                    'X_test_scaled': X_test_scaled,
                                    'scores_test': scores_test,
                                    'loadings': loadings,
                                    'n_components_use': n_components_use,
                                    'model_vars': model_vars,
                                    't2_faults': t2_faults,
                                    'q_faults': q_faults,
                                    'total_faults': total_faults,
                                    # Additional data for plots (keep plots visible)
                                    'timestamps': timestamps,
                                    'pca_params_test': pca_params_test,
                                    'explained_variance_pct': explained_variance_pct,
                                    'n_samples_train': n_samples_train
                                }

                                # Detailed fault information
                                with st.expander("📋 Fault Details"):
                                    fault_df = pd.DataFrame({
                                        'Sample': range(1, len(t2_values) + 1),
                                        'T² Statistic': t2_values,
                                        'Q Statistic': q_values,
                                        'T² Limit (97.5%)': t2_limits[0],
                                        'Q Limit (97.5%)': q_limits[0],
                                        'T² Fault': t2_faults,
                                        'Q Fault': q_faults,
                                        'Any Fault': total_faults
                                    })

                                    # Show only faulty samples by default
                                    faulty_samples = fault_df[fault_df['Any Fault']]
                                    if len(faulty_samples) > 0:
                                        st.markdown(f"**{len(faulty_samples)} faulty samples detected:**")
                                        st.dataframe(faulty_samples, use_container_width=True)
                                    else:
                                        st.success("✅ No faults detected in test data!")

                                    # Option to show all samples
                                    if st.checkbox("Show all test samples", value=False, key="show_all_test_diagnostics"):
                                        st.dataframe(fault_df, use_container_width=True)

                            except Exception as e:
                                st.error(f"❌ Error testing data: {str(e)}")
                                import traceback
                                st.code(traceback.format_exc())

                    # ===== PLOTS & FAULT SUMMARY (OUTSIDE BUTTON - KEEP VISIBLE) =====
                    # Display plots and fault summary outside button to keep them visible when dropdown changes
                    if 'pca_monitor_test_results' in st.session_state:
                        # Retrieve test results from session state
                        test_results = st.session_state.pca_monitor_test_results
                        t2_values_plot = test_results['t2_values']
                        q_values_plot = test_results['q_values']
                        t2_limits_plot = test_results['t2_limits']
                        q_limits_plot = test_results['q_limits']
                        scores_test_plot = test_results['scores_test']
                        timestamps_plot = test_results['timestamps']
                        pca_params_test_plot = test_results['pca_params_test']
                        explained_variance_pct_plot = test_results['explained_variance_pct']
                        t2_faults_plot = test_results['t2_faults']
                        q_faults_plot = test_results['q_faults']
                        total_faults_plot = test_results['total_faults']
                        X_test_plot = test_results['X_test']

                        # Create plots side by side (ONLY TEST SAMPLES)
                        st.markdown("### 📊 Test Data Projection on Training Model")
                        st.info(f"Displaying **{X_test_plot.shape[0]} test samples** projected onto training model using **{test_results['n_components_use']} components**")

                        # Trajectory style controls for test data
                        st.markdown("**Trajectory Visualization Options**")
                        test_traj_col1, test_traj_col2 = st.columns(2)

                        with test_traj_col1:
                            show_trajectory_score_test = st.checkbox("Show Trajectory", value=True, key="show_trajectory_score_test")
                            if show_trajectory_score_test:
                                trajectory_style_score_test = st.radio(
                                    "Score Plot Trajectory Style",
                                    options=['simple', 'gradient'],
                                    format_func=lambda x: {
                                        'simple': 'Simple (Light Gray)',
                                        'gradient': 'Gradient (Blue→Red with Star)'
                                    }[x],
                                    index=0,
                                    horizontal=True,
                                    key="trajectory_style_score_test"
                                )
                            else:
                                trajectory_style_score_test = 'simple'

                        with test_traj_col2:
                            show_trajectory_t2q_test = st.checkbox("Show Trajectory", value=True, key="show_trajectory_t2q_test")
                            if show_trajectory_t2q_test:
                                trajectory_style_t2q_test = st.radio(
                                    "T²-Q Plot Trajectory Style",
                                    options=['simple', 'gradient'],
                                    format_func=lambda x: {
                                        'simple': 'Simple (Light Gray)',
                                        'gradient': 'Gradient (Blue→Red with Star)'
                                    }[x],
                                    index=0,
                                    horizontal=True,
                                    key="trajectory_style_t2q_test"
                                )
                            else:
                                trajectory_style_t2q_test = 'simple'

                        test_plot_col1, test_plot_col2 = st.columns(2)

                        with test_plot_col1:
                            st.markdown("**Test Samples - Score Plot (PC1 vs PC2)**")
                            fig_score_test = create_score_plot(
                                scores_test_plot,
                                explained_variance_pct_plot,
                                timestamps=timestamps_plot,
                                pca_params=pca_params_test_plot,
                                start_sample_num=1,
                                show_trajectory=show_trajectory_score_test,
                                trajectory_style=trajectory_style_score_test
                            )
                            st.plotly_chart(fig_score_test, use_container_width=True)

                        with test_plot_col2:
                            st.markdown("**Test Samples - T²-Q Influence Plot**")
                            fig_t2q_test = create_t2_q_plot(
                                t2_values_plot,
                                q_values_plot,
                                t2_limits_plot,
                                q_limits_plot,
                                timestamps=timestamps_plot,
                                start_sample_num=1,
                                show_trajectory=show_trajectory_t2q_test,
                                trajectory_style=trajectory_style_t2q_test
                            )
                            st.plotly_chart(fig_t2q_test, use_container_width=True)

                        # Fault detection summary
                        st.markdown("### 📈 Fault Detection Summary")

                        fault_col1, fault_col2, fault_col3, fault_col4 = st.columns(4)

                        with fault_col1:
                            st.metric("Total Samples", len(t2_values_plot))
                        with fault_col2:
                            n_total_faults = total_faults_plot.sum()
                            st.metric("Total Faults", f"{n_total_faults} ({n_total_faults/len(t2_values_plot)*100:.1f}%)")
                        with fault_col3:
                            n_t2_faults = t2_faults_plot.sum()
                            st.metric("T² Faults", f"{n_t2_faults} ({n_t2_faults/len(t2_values_plot)*100:.1f}%)")
                        with fault_col4:
                            n_q_faults = q_faults_plot.sum()
                            st.metric("Q Faults", f"{n_q_faults} ({n_q_faults/len(q_values_plot)*100:.1f}%)")

                        # ===== CONTROL CHARTS =====
                        st.markdown("---")
                        st.markdown("### 📊 Control Charts Over Time")

                        fig_control_test = create_time_control_charts(
                            t2_values_plot,
                            q_values_plot,
                            timestamps_plot,
                            t2_limits_plot,
                            q_limits_plot,
                            start_sample_num=1
                        )
                        st.plotly_chart(fig_control_test, use_container_width=True)

                        # Detailed fault information
                        with st.expander("📋 Fault Details"):
                            fault_df = pd.DataFrame({
                                'Sample': range(1, len(t2_values_plot) + 1),
                                'T² Statistic': t2_values_plot,
                                'Q Statistic': q_values_plot,
                                'T² Limit (97.5%)': t2_limits_plot[0],
                                'Q Limit (97.5%)': q_limits_plot[0],
                                'T² Fault': t2_faults_plot,
                                'Q Fault': q_faults_plot,
                                'Any Fault': total_faults_plot
                            })

                            # Show only faulty samples by default
                            faulty_samples = fault_df[fault_df['Any Fault']]
                            if len(faulty_samples) > 0:
                                st.markdown(f"**{len(faulty_samples)} faulty samples detected:**")
                                st.dataframe(faulty_samples, use_container_width=True)
                            else:
                                st.success("✅ No faults detected in test data!")

                            # Option to show all samples
                            if st.checkbox("Show all test samples", value=False, key="show_all_test_monitoring"):
                                st.dataframe(fault_df, use_container_width=True)

                    # ===== CONTRIBUTION ANALYSIS (OUTSIDE BUTTON) =====
                    # Check if test results exist in session state
                    if 'pca_monitor_test_results' in st.session_state and 'pca_monitor_training_data' in st.session_state:
                        # Retrieve test results from session state
                        test_results = st.session_state.pca_monitor_test_results
                        t2_values = test_results['t2_values']
                        q_values = test_results['q_values']
                        t2_limits = test_results['t2_limits']
                        q_limits = test_results['q_limits']
                        X_test = test_results['X_test']
                        X_test_scaled = test_results['X_test_scaled']
                        scores_test = test_results['scores_test']
                        loadings = test_results['loadings']
                        n_components_use = test_results['n_components_use']
                        model_vars = test_results['model_vars']

                        # Find test samples exceeding limits (97.5%)
                        t2_test_outliers = np.where(t2_values > t2_limits[0])[0]
                        q_test_outliers = np.where(q_values > q_limits[0])[0]
                        test_outlier_samples = np.unique(np.concatenate([t2_test_outliers, q_test_outliers]))

                        if len(test_outlier_samples) == 0:
                            st.success("✅ **No samples exceed control limits.** All test samples are within normal operating conditions.")
                        else:
                            st.markdown("---")
                            st.markdown("### 🔬 Contribution Analysis")
                            st.markdown("*Analyze samples exceeding T²/Q control limits*")

                            st.warning(f"⚠️ **{len(test_outlier_samples)} test samples exceed control limits** (T²>limit OR Q>limit)")

                            # Get training data from session state
                            X_train = st.session_state.pca_monitor_training_data
                            X_train_array = X_train.values

                            # Scale training data the same way
                            pca_results = st.session_state.pca_monitor_model
                            if st.session_state.pca_monitor_scale:
                                # Use NIPALS means and stds
                                means = pca_results['means']
                                stds = pca_results['stds']
                                X_train_scaled = (X_train_array - means) / stds
                            elif st.session_state.pca_monitor_center:
                                # Use NIPALS means
                                means = pca_results['means']
                                X_train_scaled = X_train_array - means
                            else:
                                X_train_scaled = X_train_array

                            # Calculate contributions for TRAINING set to get normalization factors
                            pca_params_train = {
                                's': pca_results['scores'].values[:, :n_components_use]
                            }

                            # Project training data using manual calculation: T = X @ P
                            loadings_train = pca_results['loadings'].values[:, :n_components_use]
                            scores_train_calc = X_train_scaled @ loadings_train

                            q_contrib_train, t2_contrib_train = calculate_all_contributions(
                                X_train_scaled,
                                scores_train_calc,
                                loadings,
                                pca_params_train
                            )

                            # Normalize contributions by 95th percentile of TRAINING set
                            q_contrib_95th_train = np.percentile(np.abs(q_contrib_train), 95, axis=0)
                            t2_contrib_95th_train = np.percentile(np.abs(t2_contrib_train), 95, axis=0)

                            # Avoid division by zero
                            q_contrib_95th_train[q_contrib_95th_train == 0] = 1.0
                            t2_contrib_95th_train[t2_contrib_95th_train == 0] = 1.0

                            # Calculate contributions for TEST set
                            pca_params_test_contrib = {
                                's': pca_results['scores'].values[:, :n_components_use]
                            }

                            X_test_array = X_test.values

                            q_contrib_test, t2_contrib_test = calculate_all_contributions(
                                X_test_scaled,
                                scores_test,
                                loadings,
                                pca_params_test_contrib
                            )

                            # Select sample from outliers only
                            test_sample_select_col, _ = st.columns([1, 1])
                            with test_sample_select_col:
                                # Use session state to persist dropdown selection
                                if 'test_contrib_sample_idx' not in st.session_state:
                                    st.session_state.test_contrib_sample_idx = test_outlier_samples[0]

                                test_sample_idx = st.selectbox(
                                    "Select outlier test sample for contribution analysis:",
                                    options=test_outlier_samples,
                                    format_func=lambda x: f"Test Sample {x+1} (T²={t2_values[x]:.2f}, Q={q_values[x]:.2f})",
                                    key="test_contrib_sample",
                                    index=int(np.where(test_outlier_samples == st.session_state.test_contrib_sample_idx)[0][0]) if st.session_state.test_contrib_sample_idx in test_outlier_samples else 0
                                )

                                # Update session state
                                st.session_state.test_contrib_sample_idx = test_sample_idx

                            # Get contributions for selected test sample
                            q_contrib_test_sample = q_contrib_test[test_sample_idx, :]
                            t2_contrib_test_sample = t2_contrib_test[test_sample_idx, :]

                            # Normalize by TRAINING set 95th percentile
                            q_contrib_test_norm = q_contrib_test_sample / q_contrib_95th_train
                            t2_contrib_test_norm = t2_contrib_test_sample / t2_contrib_95th_train

                            # Bar plots side by side (ALL variables, red if |contrib|>1, blue otherwise)
                            test_contrib_col1, test_contrib_col2 = st.columns(2)

                            with test_contrib_col1:
                                st.markdown(f"**T² Contributions - Test Sample {test_sample_idx+1}**")
                                fig_t2_contrib_test = create_contribution_plot_all_vars(
                                    t2_contrib_test_norm,
                                    model_vars,
                                    statistic='T²'
                                )
                                st.plotly_chart(fig_t2_contrib_test, use_container_width=True)

                            with test_contrib_col2:
                                st.markdown(f"**Q Contributions - Test Sample {test_sample_idx+1}**")
                                fig_q_contrib_test = create_contribution_plot_all_vars(
                                    q_contrib_test_norm,
                                    model_vars,
                                    statistic='Q'
                                )
                                st.plotly_chart(fig_q_contrib_test, use_container_width=True)

                            # Table: Variables where |contrib|>1 with real values vs training mean
                            st.markdown("### 🏆 Top Contributing Variables")
                            st.markdown("*Variables exceeding 95th percentile threshold (|contribution| > 1)*")

                            # Get training mean for comparison
                            training_mean = X_train.mean()

                            # Get real values for selected test sample
                            test_sample_values = X_test.iloc[test_sample_idx]

                            # Filter variables where |contrib|>1 for either T² or Q
                            high_contrib_t2 = np.abs(t2_contrib_test_norm) > 1.0
                            high_contrib_q = np.abs(q_contrib_test_norm) > 1.0
                            high_contrib = high_contrib_t2 | high_contrib_q

                            if high_contrib.sum() > 0:
                                contrib_table_data = []
                                for i, var in enumerate(model_vars):
                                    if high_contrib[i]:
                                        real_val = test_sample_values[var]
                                        mean_val = training_mean[var]
                                        diff = real_val - mean_val
                                        direction = "Higher ↑" if diff > 0 else "Lower ↓"

                                        contrib_table_data.append({
                                            'Variable': var,
                                            'Real Value': f"{real_val:.3f}",
                                            'Training Mean': f"{mean_val:.3f}",
                                            'Difference': f"{diff:.3f}",
                                            'Direction': direction,
                                            '|T² Contrib|': f"{abs(t2_contrib_test_norm[i]):.2f}",
                                            '|Q Contrib|': f"{abs(q_contrib_test_norm[i]):.2f}"
                                        })

                                contrib_table = pd.DataFrame(contrib_table_data)
                                # Sort by max absolute contribution
                                contrib_table['Max_Contrib'] = contrib_table.apply(
                                    lambda row: max(float(row['|T² Contrib|']), float(row['|Q Contrib|'])),
                                    axis=1
                                )
                                contrib_table = contrib_table.sort_values('Max_Contrib', ascending=False).drop('Max_Contrib', axis=1)

                                st.dataframe(contrib_table, use_container_width=True)
                            else:
                                st.info("No variables exceed the 95th percentile threshold.")

                            # Correlation scatter: training (grey), test (blue), sample (red star)
                            st.markdown("### 📈 Correlation Analysis - Top Q Contributor")
                            st.markdown("*Select from top Q contributors to see correlation with most correlated variable*")

                            # Get top Q contributors (variables with highest |Q contribution|)
                            q_contrib_abs = np.abs(q_contrib_test_norm)
                            top_q_indices = np.argsort(q_contrib_abs)[::-1][:5]
                            top_q_contributors_test = [model_vars[i] for i in top_q_indices]

                            # Dropdown to select from top Q contributors
                            test_corr_col1, test_corr_col2 = st.columns([2, 1])

                            with test_corr_col1:
                                selected_q_var_test = st.selectbox(
                                    "Select from top Q contributors:",
                                    options=top_q_contributors_test,
                                    key="test_top_q_var"
                                )

                            # Calculate correlations for selected variable (using TRAINING data)
                            var1_idx_test = model_vars.index(selected_q_var_test)
                            correlations_test = {}
                            for i, var in enumerate(model_vars):
                                if var != selected_q_var_test:
                                    # Use training data for correlation calculation
                                    corr = np.corrcoef(X_train_array[:, var1_idx_test], X_train_array[:, i])[0, 1]
                                    correlations_test[var] = (corr, i)

                            # Find most correlated variable
                            most_corr_var_test = max(correlations_test, key=lambda k: abs(correlations_test[k][0]))
                            corr_coef_test, var2_idx_test = correlations_test[most_corr_var_test]

                            with test_corr_col2:
                                st.metric("Correlation (training)", f"{corr_coef_test:.4f}")

                            # Create 3-layer scatter plot (training=grey, test=blue, sample=red star)
                            fig_corr_scatter_test = create_correlation_scatter(
                                X_train=X_train_array,
                                X_test=X_test_array,
                                X_sample=X_test_array[test_sample_idx, :],
                                var1_idx=var1_idx_test,
                                var2_idx=var2_idx_test,
                                var1_name=selected_q_var_test,
                                var2_name=most_corr_var_test,
                                correlation_val=corr_coef_test,
                                sample_idx=test_sample_idx
                            )

                            st.plotly_chart(fig_corr_scatter_test, use_container_width=True)

                            # ========== EXPORT SECTION - TEST DATA ==========
                            st.markdown("---")
                            st.markdown("## 📥 Export Test Data Monitoring Results")
                            st.markdown("*Export T², Q values, limits, and outlier diagnostics for test dataset*")

                            export_test_col1, export_test_col2, export_test_col3 = st.columns(3)

                            with export_test_col1:
                                st.markdown("### 📊 Basic Data Export")

                                # Prepare basic data export for test
                                basic_export_df_test = pd.DataFrame({
                                    'Sample': [f"Test_Sample_{i+1}" for i in range(len(t2_values))],
                                    'T2': t2_values,
                                    'Q': q_values,
                                    'T2_Limit_97.5%': [t2_limits[0]] * len(t2_values),
                                    'T2_Limit_99.5%': [t2_limits[1]] * len(t2_values),
                                    'T2_Limit_99.95%': [t2_limits[2]] * len(t2_values),
                                    'Q_Limit_97.5%': [q_limits[0]] * len(q_values),
                                    'Q_Limit_99.5%': [q_limits[1]] * len(q_values),
                                    'Q_Limit_99.95%': [q_limits[2]] * len(q_values)
                                })

                                # Add timestamps if available
                                if timestamps is not None and len(timestamps) == len(t2_values):
                                    basic_export_df_test.insert(1, 'Timestamp', timestamps)

                                # Convert to CSV
                                csv_basic_test = basic_export_df_test.to_csv(index=False)

                                st.download_button(
                                    label="📥 Download T²-Q Values (CSV)",
                                    data=csv_basic_test,
                                    file_name=f"test_t2_q_values_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                    mime="text/csv",
                                    help="Download test data T² and Q values with all critical limits"
                                )

                            with export_test_col2:
                                st.markdown("### 🔍 Independent Diagnostics")
                                st.markdown("*T² and Q evaluated independently*")

                                # Prepare sample labels for test
                                sample_labels_test_export = [f"Test_Sample_{i+1}" for i in range(len(t2_values))]

                                # Create Excel file with independent diagnostics
                                if st.button("📊 Generate Independent Report", key="btn_independent_test"):
                                    with st.spinner("Generating independent diagnostics report for test data..."):
                                        try:
                                            from pca_utils.pca_monitoring import export_monitoring_data_to_excel

                                            excel_independent_test = export_monitoring_data_to_excel(
                                                t2_values=t2_values,
                                                q_values=q_values,
                                                t2_limits=t2_limits,
                                                q_limits=q_limits,
                                                sample_labels=sample_labels_test_export,
                                                approach='independent'
                                            )

                                            st.download_button(
                                                label="📥 Download Independent Report (Excel)",
                                                data=excel_independent_test,
                                                file_name=f"test_monitoring_independent_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                                help="Excel file with test data T² and Q outliers classified independently",
                                                key="download_independent_test"
                                            )
                                            st.success("✅ Independent report generated!")
                                        except Exception as e:
                                            st.error(f"❌ Error: {str(e)}")

                            with export_test_col3:
                                st.markdown("### 🎯 Joint Diagnostics")
                                st.markdown("*Hierarchical outlier classification*")

                                # Create Excel file with joint diagnostics
                                if st.button("📊 Generate Joint Report", key="btn_joint_test"):
                                    with st.spinner("Generating joint diagnostics report for test data..."):
                                        try:
                                            from pca_utils.pca_monitoring import export_monitoring_data_to_excel

                                            excel_joint_test = export_monitoring_data_to_excel(
                                                t2_values=t2_values,
                                                q_values=q_values,
                                                t2_limits=t2_limits,
                                                q_limits=q_limits,
                                                sample_labels=sample_labels_test_export,
                                                approach='joint'
                                            )

                                            st.download_button(
                                                label="📥 Download Joint Report (Excel)",
                                                data=excel_joint_test,
                                                file_name=f"test_monitoring_joint_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                                help="Excel file with hierarchical outlier classification for test data",
                                                key="download_joint_test"
                                            )
                                            st.success("✅ Joint report generated!")
                                        except Exception as e:
                                            st.error(f"❌ Error: {str(e)}")

                            # Complete report with both approaches
                            st.markdown("---")
                            complete_test_col1, complete_test_col2 = st.columns([2, 1])

                            with complete_test_col1:
                                st.markdown("### 📦 Complete Test Monitoring Report")
                                st.markdown("*Includes both independent and joint diagnostics for test data*")

                            with complete_test_col2:
                                if st.button("📊 Generate Complete Report", key="btn_complete_test", type="primary"):
                                    with st.spinner("Generating complete monitoring report for test data..."):
                                        try:
                                            from pca_utils.pca_monitoring import export_monitoring_data_to_excel

                                            excel_complete_test = export_monitoring_data_to_excel(
                                                t2_values=t2_values,
                                                q_values=q_values,
                                                t2_limits=t2_limits,
                                                q_limits=q_limits,
                                                sample_labels=sample_labels_test_export,
                                                approach='both'
                                            )

                                            st.download_button(
                                                label="📥 Download Complete Report (Excel)",
                                                data=excel_complete_test,
                                                file_name=f"test_monitoring_complete_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                                help="Comprehensive Excel file with all diagnostics for test data",
                                                key="download_complete_test"
                                            )
                                            st.success("✅ Complete report generated!")
                                        except Exception as e:
                                            st.error(f"❌ Error: {str(e)}")

                            # Info about report contents
                            with st.expander("📋 What's included in the test reports?", expanded=False):
                                st.markdown("""
                                **Basic Data Export (CSV)**:
                                - T² and Q values for all test samples
                                - Critical limits at all confidence levels (97.5%, 99.5%, 99.95%)
                                - Timestamps (if available)

                                **Independent Diagnostics Report (Excel)**:
                                - Sheet 1: T² and Q values with limits
                                - Sheet 2: Occurrences summary (count of outliers per threshold)
                                - Sheet 3: T² outliers (samples exceeding T² limits)
                                - Sheet 4: Q outliers (samples exceeding Q limits)
                                - Sheet 5: Combined outliers (samples exceeding T² OR Q)

                                **Joint Diagnostics Report (Excel)**:
                                - Sheet 1: T² and Q values with limits
                                - Sheet 2: Occurrences summary (count per threshold level)
                                - Sheet 3: Hierarchical outlier classification
                                  - *** (99.95%): Most severe outliers
                                  - ** (99.5%): Moderate outliers
                                  - * (97.5%): Mild outliers

                                **Complete Report (Excel)**:
                                - All sheets from both Independent and Joint diagnostics
                                - 7 sheets total with comprehensive outlier analysis
                                """)

    # ===== TAB 4: MODEL MANAGEMENT & COMPREHENSIVE EXPORT =====
    with tab4:
        st.markdown("## 💾 Model Management & Comprehensive Export")
        st.markdown("*Export complete monitoring analysis and manage models*")

        # Check if model is trained
        if 'pca_monitor_trained' not in st.session_state or not st.session_state.pca_monitor_trained:
            st.warning("⚠️ **No model trained yet.** Please train a model in the **Model Training** tab first.")
        else:
            st.success("✅ **Model loaded and ready for comprehensive export**")

            # Section 1: Training Data Export
            st.markdown("### 📊 Training Data Export")

            if 'pca_monitor_plot_results' in st.session_state:
                training_results = st.session_state.pca_monitor_plot_results

                train_export_col1, train_export_col2 = st.columns(2)

                with train_export_col1:
                    st.markdown("**Training Dataset Diagnostics**")

                    # Basic CSV
                    basic_train_df = pd.DataFrame({
                        'Sample': [f"Train_Sample_{i+1}" for i in range(len(training_results['t2']))],
                        'T2': training_results['t2'],
                        'Q': training_results['q'],
                        'T2_Limit_97.5%': [training_results['t2_limits'][0]] * len(training_results['t2']),
                        'T2_Limit_99.5%': [training_results['t2_limits'][1]] * len(training_results['t2']),
                        'T2_Limit_99.95%': [training_results['t2_limits'][2]] * len(training_results['t2']),
                        'Q_Limit_97.5%': [training_results['q_limits'][0]] * len(training_results['q']),
                        'Q_Limit_99.5%': [training_results['q_limits'][1]] * len(training_results['q']),
                        'Q_Limit_99.95%': [training_results['q_limits'][2]] * len(training_results['q'])
                    })

                    csv_train = basic_train_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Training Data (CSV)",
                        data=csv_train,
                        file_name=f"training_data_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )

                with train_export_col2:
                    if st.button("📊 Generate Training Report (Excel)", key="btn_train_report_tab4"):
                        with st.spinner("Generating training report..."):
                            try:
                                from pca_utils.pca_monitoring import export_monitoring_data_to_excel

                                sample_labels_train = [f"Train_Sample_{i+1}" for i in range(len(training_results['t2']))]

                                excel_train = export_monitoring_data_to_excel(
                                    t2_values=training_results['t2'],
                                    q_values=training_results['q'],
                                    t2_limits=training_results['t2_limits'],
                                    q_limits=training_results['q_limits'],
                                    sample_labels=sample_labels_train,
                                    approach='both'
                                )

                                st.download_button(
                                    label="📥 Download Training Report (Excel)",
                                    data=excel_train,
                                    file_name=f"training_complete_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                    key="download_train_report_tab4"
                                )
                                st.success("✅ Training report generated!")
                            except Exception as e:
                                st.error(f"❌ Error: {str(e)}")

            st.markdown("---")

            # Section 2: Test Data Export (if available)
            st.markdown("### 🔬 Test Data Export")

            if 'pca_monitor_test_results' in st.session_state:
                test_results = st.session_state.pca_monitor_test_results

                test_export_col1, test_export_col2 = st.columns(2)

                with test_export_col1:
                    st.markdown("**Test Dataset Monitoring Results**")

                    # Basic CSV
                    basic_test_df = pd.DataFrame({
                        'Sample': [f"Test_Sample_{i+1}" for i in range(len(test_results['t2_values']))],
                        'T2': test_results['t2_values'],
                        'Q': test_results['q_values'],
                        'T2_Limit_97.5%': [test_results['t2_limits'][0]] * len(test_results['t2_values']),
                        'T2_Limit_99.5%': [test_results['t2_limits'][1]] * len(test_results['t2_values']),
                        'T2_Limit_99.95%': [test_results['t2_limits'][2]] * len(test_results['t2_values']),
                        'Q_Limit_97.5%': [test_results['q_limits'][0]] * len(test_results['q_values']),
                        'Q_Limit_99.5%': [test_results['q_limits'][1]] * len(test_results['q_values']),
                        'Q_Limit_99.95%': [test_results['q_limits'][2]] * len(test_results['q_values'])
                    })

                    csv_test = basic_test_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Test Data (CSV)",
                        data=csv_test,
                        file_name=f"test_data_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )

                with test_export_col2:
                    if st.button("📊 Generate Test Report (Excel)", key="btn_test_report_tab4"):
                        with st.spinner("Generating test report..."):
                            try:
                                from pca_utils.pca_monitoring import export_monitoring_data_to_excel

                                sample_labels_test = [f"Test_Sample_{i+1}" for i in range(len(test_results['t2_values']))]

                                excel_test = export_monitoring_data_to_excel(
                                    t2_values=test_results['t2_values'],
                                    q_values=test_results['q_values'],
                                    t2_limits=test_results['t2_limits'],
                                    q_limits=test_results['q_limits'],
                                    sample_labels=sample_labels_test,
                                    approach='both'
                                )

                                st.download_button(
                                    label="📥 Download Test Report (Excel)",
                                    data=excel_test,
                                    file_name=f"test_complete_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                    key="download_test_report_tab4"
                                )
                                st.success("✅ Test report generated!")
                            except Exception as e:
                                st.error(f"❌ Error: {str(e)}")
            else:
                st.info("ℹ️ No test data available. Run monitoring in **Testing & Monitoring** tab first.")

            st.markdown("---")

            # Section 3: Model Save/Load (placeholder)
            st.markdown("### 💾 Model Save/Load")
            st.info("🚧 Model save/load functionality to be implemented")


if __name__ == "__main__":
    show()
