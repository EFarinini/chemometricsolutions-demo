"""
Pareto Front Multi-Criteria Optimization
Equivalent to DOE_Pareto.r
Identifies optimal compromise solutions for multiple objectives
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from io import BytesIO
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment


# ============================================================================
# DATA CODING/DECODING UTILITIES
# ============================================================================

def code_value(real_value, real_min, real_max):
    """
    Code real value to [-1, +1] range.

    Formula: coded = (real - center) / range
    where center = (max + min) / 2
          range = (max - min) / 2

    Args:
        real_value: Value in real/natural units
        real_min: Minimum real value (→ coded -1)
        real_max: Maximum real value (→ coded +1)

    Returns:
        Coded value in [-1, +1]

    Example:
        >>> code_value(5, 5, 15)   # min → -1
        -1.0
        >>> code_value(10, 5, 15)  # center → 0
        0.0
        >>> code_value(15, 5, 15)  # max → +1
        1.0
    """
    real_center = (real_max + real_min) / 2
    real_range = (real_max - real_min) / 2

    if real_range == 0:
        return 0.0

    coded = (real_value - real_center) / real_range
    return coded


def decode_value(coded_value, real_min, real_max):
    """
    Decode coded value [-1, +1] back to real units.

    Formula: real = coded × range + center

    Args:
        coded_value: Value in coded units [-1, +1]
        real_min: Minimum real value
        real_max: Maximum real value

    Returns:
        Value in real/natural units

    Example:
        >>> decode_value(-1, 5, 15)  # -1 → min
        5.0
        >>> decode_value(0, 5, 15)   # 0 → center
        10.0
        >>> decode_value(1, 5, 15)   # +1 → max
        15.0
    """
    real_center = (real_max + real_min) / 2
    real_range = (real_max - real_min) / 2

    real_value = coded_value * real_range + real_center
    return real_value


def create_coding_dict(x_vars, grid_config):
    """
    Create coding dictionary from grid configuration.

    Args:
        x_vars: List of variable names
        grid_config: Dict with 'real_min', 'real_max' for each variable
            Example: {'Coal_load_t_h': {'real_min': 5, 'real_max': 15, 'steps': 15}}

    Returns:
        Dict with coding parameters for each variable

    Example output:
        {
            'Coal_load_t_h': {
                'real_min': 5,
                'real_max': 15,
                'real_center': 10,
                'real_range': 5,
                'coded_min': -1.0,
                'coded_max': 1.0
            }
        }
    """
    coding_dict = {}

    for var in x_vars:
        real_min = grid_config[var]['real_min']
        real_max = grid_config[var]['real_max']

        coding_dict[var] = {
            'real_min': real_min,
            'real_max': real_max,
            'real_center': (real_max + real_min) / 2,
            'real_range': (real_max - real_min) / 2,
            'coded_min': -1.0,
            'coded_max': 1.0
        }

    return coding_dict


# ============================================================================
# CONFIDENCE INTERVALS FOR PARETO PREDICTIONS
# ============================================================================

def calculate_confidence_intervals_for_pareto(candidate_df, model_results, x_vars, y_var, confidence_level=0.95):
    """
    Calculate confidence intervals for predicted Y values in Pareto analysis.

    Similar to confidence_intervals.py from Surface Analysis tab, but applied
    to Pareto candidate grid.

    Args:
        candidate_df: DataFrame with candidate points (must have coded X columns)
        model_results: dict from fit_mlr_model (contains XtX_inv, rmse, dof)
        x_vars: List of X variable names (coded)
        y_var: Y variable name
        confidence_level: Confidence level (default 0.95 for 95% CI)

    Returns:
        DataFrame with added columns:
        - {y_var}_predicted_lower: Lower CI bound
        - {y_var}_predicted_upper: Upper CI bound
        - {y_var}_ci_semiwidth: CI semiamplitude
    """
    from scipy import stats

    # Get model parameters
    coefficients = model_results['coefficients']
    coef_names = coefficients.index.tolist()
    dispersion = model_results['XtX_inv']
    rmse = model_results.get('rmse', 1.0)
    dof = model_results.get('dof', 1)

    if dof <= 0:
        raise ValueError("Degrees of freedom must be > 0 for confidence intervals")

    # Build model matrix for predictions
    from mlr_utils.response_surface import create_prediction_matrix

    X_coded = candidate_df[x_vars].values
    X_model = create_prediction_matrix(X_coded, x_vars, coef_names)

    # Calculate predictions (should match existing predictions)
    predictions = X_model @ coefficients.values

    # Calculate leverage for each point
    # leverage = diag(X * (X'X)^-1 * X')
    leverage = np.diag(X_model @ dispersion @ X_model.T)

    # Calculate CI semiamplitude
    # CI = t_critical * rmse * sqrt(leverage)
    alpha = 1 - confidence_level
    t_critical = stats.t.ppf(1 - alpha/2, dof)

    ci_semiwidth = t_critical * rmse * np.sqrt(leverage)

    # Calculate bounds
    lower_bound = predictions - ci_semiwidth
    upper_bound = predictions + ci_semiwidth

    # Add to dataframe
    result_df = candidate_df.copy()
    result_df[f'{y_var}_predicted_lower'] = lower_bound
    result_df[f'{y_var}_predicted_upper'] = upper_bound
    result_df[f'{y_var}_ci_semiwidth'] = ci_semiwidth

    return result_df


# ============================================================================
# PARETO FRONT CALCULATION
# ============================================================================

def calculate_pareto_front(df, objectives_dict, n_fronts=3):
    """
    Calculate Pareto front ranking for multi-objective optimization.

    Args:
        df: DataFrame with candidate points and predicted values
        objectives_dict: Dictionary defining objectives
            Example: {
                'Y_predicted': 'maximize',  # Response variable
                'X1': 'minimize',  # Predictor constraint
                'X2': ('target', 50.0),  # Target value
            }
        n_fronts: Number of Pareto fronts to identify (default: 3)

    Returns:
        DataFrame with added columns: 'pareto_rank', 'is_dominated',
        'dominance_count', 'crowding_distance'

    Algorithm:
    1. Transform all objectives to maximization format
    2. Iteratively identify non-dominated points (Pareto ranking)
    3. Calculate crowding distance for diversity
    4. Return ranked DataFrame
    """
    df_result = df.copy()

    # Transform objectives to maximization format
    objectives_matrix, transform_info = _transform_objectives(df, objectives_dict)

    # Store original objective values
    objective_cols = list(objectives_dict.keys())
    for col in objective_cols:
        if col in df.columns:
            df_result[f'{col}_original'] = df[col]

    # Initialize ranking columns
    df_result['pareto_rank'] = 0
    df_result['is_dominated'] = False
    df_result['dominance_count'] = 0
    df_result['crowding_distance'] = 0.0

    # Iteratively identify Pareto fronts
    remaining_indices = np.arange(len(df))
    current_rank = 1

    while len(remaining_indices) > 0 and current_rank <= n_fronts:
        # Get objectives for remaining points
        remaining_objectives = objectives_matrix[remaining_indices]

        # Identify non-dominated points in remaining set
        is_front = _identify_pareto_front(remaining_objectives)
        front_indices = remaining_indices[is_front]

        # Assign rank
        df_result.loc[front_indices, 'pareto_rank'] = current_rank

        # Calculate crowding distance for this front
        if len(front_indices) > 2:
            front_objectives = objectives_matrix[front_indices]
            crowding = _calculate_crowding_distance(front_objectives)
            df_result.loc[front_indices, 'crowding_distance'] = crowding
        else:
            # For small fronts, assign max crowding
            df_result.loc[front_indices, 'crowding_distance'] = np.inf

        # Mark dominated points
        df_result.loc[remaining_indices[~is_front], 'is_dominated'] = True

        # Remove front from remaining
        remaining_indices = remaining_indices[~is_front]
        current_rank += 1

    # Mark all remaining points as dominated
    if len(remaining_indices) > 0:
        df_result.loc[remaining_indices, 'is_dominated'] = True
        df_result.loc[remaining_indices, 'pareto_rank'] = n_fronts + 1

    # Calculate dominance count (how many points dominate each point)
    for idx in df_result.index:
        point = objectives_matrix[idx]
        dominated_by = 0
        for other_idx in df_result.index:
            if idx != other_idx:
                other_point = objectives_matrix[other_idx]
                if _dominates(other_point, point):
                    dominated_by += 1
        df_result.loc[idx, 'dominance_count'] = dominated_by

    return df_result


def _transform_objectives(df, objectives_dict):
    """
    Transform all objectives to maximization format.

    Supported objective types:
    - 'maximize': keep as is
    - 'minimize': multiply by -1
    - ('target', value): use -|x - target|
    - 'maximize_ci': use lower CI bound (conservative)
    - 'minimize_ci': use upper CI bound (conservative)
    - ('target_ci', value): use -|x - target| - CI (penalize uncertainty)
    - ('threshold_above', value): penalize if below threshold
    - ('threshold_below', value): penalize if above threshold
    - ('threshold_above_ci', value): use lower CI bound
    - ('threshold_below_ci', value): use upper CI bound

    Args:
        df: DataFrame with objective columns
        objectives_dict: Dictionary of objectives

    Returns:
        objectives_matrix: numpy array (n_points × n_objectives) in maximization format
        transform_info: Dictionary with transformation details
    """
    n_points = len(df)
    n_objectives = len(objectives_dict)
    objectives_matrix = np.zeros((n_points, n_objectives))
    transform_info = {}

    for i, (obj_name, obj_type) in enumerate(objectives_dict.items()):
        # Determine column name
        if obj_name not in df.columns:
            raise ValueError(f"Objective '{obj_name}' not found in DataFrame")

        values = df[obj_name].values

        # Handle different objective types
        if obj_type == 'maximize':
            objectives_matrix[:, i] = values
            transform_info[obj_name] = {'type': 'maximize', 'factor': 1.0}

        elif obj_type == 'minimize':
            objectives_matrix[:, i] = -values
            transform_info[obj_name] = {'type': 'minimize', 'factor': -1.0}

        elif obj_type == 'maximize_ci':
            # Conservative: maximize LOWER CI bound
            lower_col = obj_name.replace('_predicted', '_predicted_lower')
            if lower_col in df.columns:
                objectives_matrix[:, i] = df[lower_col].values
                transform_info[obj_name] = {'type': 'maximize_ci', 'uses_lower_bound': True}
            else:
                objectives_matrix[:, i] = values
                transform_info[obj_name] = {'type': 'maximize', 'warning': 'CI not available'}

        elif obj_type == 'minimize_ci':
            # Conservative: minimize UPPER CI bound
            upper_col = obj_name.replace('_predicted', '_predicted_upper')
            if upper_col in df.columns:
                objectives_matrix[:, i] = -df[upper_col].values
                transform_info[obj_name] = {'type': 'minimize_ci', 'uses_upper_bound': True}
            else:
                objectives_matrix[:, i] = -values
                transform_info[obj_name] = {'type': 'minimize', 'warning': 'CI not available'}

        elif isinstance(obj_type, tuple):
            if obj_type[0] == 'target':
                target_value = obj_type[1]
                objectives_matrix[:, i] = -np.abs(values - target_value)
                transform_info[obj_name] = {'type': 'target', 'target': target_value}

            elif obj_type[0] == 'target_ci':
                # Target with CI penalty for uncertainty
                target_value = obj_type[1]
                ci_col = obj_name.replace('_predicted', '_ci_semiwidth')
                if ci_col in df.columns:
                    ci_width = df[ci_col].values
                    distance = np.abs(values - target_value)
                    # Penalize: distance + uncertainty
                    objectives_matrix[:, i] = -(distance + ci_width)
                    transform_info[obj_name] = {'type': 'target_ci', 'target': target_value}
                else:
                    objectives_matrix[:, i] = -np.abs(values - target_value)
                    transform_info[obj_name] = {'type': 'target', 'warning': 'CI not available'}

            elif obj_type[0] == 'threshold_above':
                threshold = obj_type[1]
                # Penalty if below threshold
                penalty = np.where(values >= threshold, 0, threshold - values)
                objectives_matrix[:, i] = -penalty
                transform_info[obj_name] = {'type': 'threshold_above', 'threshold': threshold}

            elif obj_type[0] == 'threshold_above_ci':
                threshold = obj_type[1]
                lower_col = obj_name.replace('_predicted', '_predicted_lower')
                if lower_col in df.columns:
                    lower_values = df[lower_col].values
                    penalty = np.where(lower_values >= threshold, 0, threshold - lower_values)
                    objectives_matrix[:, i] = -penalty
                    transform_info[obj_name] = {'type': 'threshold_above_ci', 'threshold': threshold}
                else:
                    penalty = np.where(values >= threshold, 0, threshold - values)
                    objectives_matrix[:, i] = -penalty
                    transform_info[obj_name] = {'type': 'threshold_above', 'warning': 'CI not available'}

            elif obj_type[0] == 'threshold_below':
                threshold = obj_type[1]
                # Penalty if above threshold
                penalty = np.where(values <= threshold, 0, values - threshold)
                objectives_matrix[:, i] = -penalty
                transform_info[obj_name] = {'type': 'threshold_below', 'threshold': threshold}

            elif obj_type[0] == 'threshold_below_ci':
                threshold = obj_type[1]
                upper_col = obj_name.replace('_predicted', '_predicted_upper')
                if upper_col in df.columns:
                    upper_values = df[upper_col].values
                    penalty = np.where(upper_values <= threshold, 0, upper_values - threshold)
                    objectives_matrix[:, i] = -penalty
                    transform_info[obj_name] = {'type': 'threshold_below_ci', 'threshold': threshold}
                else:
                    penalty = np.where(values <= threshold, 0, values - threshold)
                    objectives_matrix[:, i] = -penalty
                    transform_info[obj_name] = {'type': 'threshold_below', 'warning': 'CI not available'}

            else:
                raise ValueError(f"Unknown objective tuple type: {obj_type}")

        else:
            raise ValueError(f"Invalid objective type for '{obj_name}': {obj_type}")

    return objectives_matrix, transform_info


def _dominates(point_a, point_b):
    """
    Check if point_a dominates point_b (Pareto dominance).

    Point A dominates point B if:
    1. A is >= B in ALL objectives (no worse in any)
    2. A is > B in AT LEAST ONE objective (better in at least one)

    Args:
        point_a, point_b: numpy arrays of objective values (maximization format)

    Returns:
        bool: True if point_a dominates point_b
    """
    better_or_equal_all = np.all(point_a >= point_b)
    strictly_better_at_least_one = np.any(point_a > point_b)
    return better_or_equal_all and strictly_better_at_least_one


def _identify_pareto_front(objectives_matrix):
    """
    Identify non-dominated points in objectives matrix.

    Args:
        objectives_matrix: numpy array (n_points × n_objectives)
        All objectives assumed to be in maximization format

    Returns:
        boolean array: True for non-dominated points (Pareto front)
    """
    n_points = objectives_matrix.shape[0]
    is_pareto = np.ones(n_points, dtype=bool)

    for i in range(n_points):
        if not is_pareto[i]:
            continue

        # Check if point i is dominated by any other point
        for j in range(n_points):
            if i != j and is_pareto[j]:
                if _dominates(objectives_matrix[j], objectives_matrix[i]):
                    is_pareto[i] = False
                    break

    return is_pareto


def _calculate_crowding_distance(objectives_matrix):
    """
    Calculate crowding distance for diversity in Pareto front.

    Higher values = more isolated = more diverse = better representation of front.
    Used to select diverse representatives from Pareto front.

    Args:
        objectives_matrix: numpy array of objectives for Pareto front points

    Returns:
        crowding_distances: array of crowding distances
    """
    n_points, n_objectives = objectives_matrix.shape
    crowding = np.zeros(n_points)

    # Handle edge cases
    if n_points <= 2:
        return np.full(n_points, np.inf)

    # For each objective
    for obj_idx in range(n_objectives):
        # Sort points by this objective
        sorted_indices = np.argsort(objectives_matrix[:, obj_idx])

        # Boundary points get infinite distance
        crowding[sorted_indices[0]] = np.inf
        crowding[sorted_indices[-1]] = np.inf

        # Get objective range
        obj_range = (objectives_matrix[sorted_indices[-1], obj_idx] -
                     objectives_matrix[sorted_indices[0], obj_idx])

        if obj_range == 0:
            continue

        # Calculate distance for interior points
        for i in range(1, n_points - 1):
            idx = sorted_indices[i]
            idx_prev = sorted_indices[i - 1]
            idx_next = sorted_indices[i + 1]

            distance = (objectives_matrix[idx_next, obj_idx] -
                       objectives_matrix[idx_prev, obj_idx]) / obj_range

            crowding[idx] += distance

    return crowding


def plot_pareto_2d(df, obj1, obj2, objectives_dict, color_by='pareto_rank',
                   show_arrows=True):
    """
    Create 2D scatter plot showing Pareto front.

    Args:
        df: DataFrame with Pareto analysis results
        obj1, obj2: Column names for X and Y axes
        objectives_dict: Dictionary of objectives
        color_by: Column to color points by ('pareto_rank' or 'crowding_distance')
        show_arrows: Whether to show optimization direction arrows

    Returns:
        plotly Figure
    """
    fig = go.Figure()

    # Get unique ranks
    ranks = sorted(df['pareto_rank'].unique())

    # Color scheme
    colors = ['#00CC96', '#FFA15A', '#EF553B', '#AB63FA', '#636EFA']

    for rank in ranks[:5]:  # Show up to 5 fronts
        df_rank = df[df['pareto_rank'] == rank]

        if rank == 1:
            marker_symbol = 'star'
            marker_size = 12
            marker_line_width = 2
            name = f'Front {rank} (Optimal)'
        else:
            marker_symbol = 'circle'
            marker_size = 8
            marker_line_width = 1
            name = f'Front {rank}'

        # Color by rank or crowding distance
        if color_by == 'crowding_distance' and rank == 1:
            marker_color = df_rank['crowding_distance'].replace([np.inf], df_rank['crowding_distance'][np.isfinite(df_rank['crowding_distance'])].max() * 2)
            colorscale = 'Viridis'
            showscale = True
        else:
            marker_color = colors[rank - 1] if rank <= len(colors) else colors[-1]
            colorscale = None
            showscale = False

        fig.add_trace(go.Scatter(
            x=df_rank[obj1],
            y=df_rank[obj2],
            mode='markers',
            name=name,
            marker=dict(
                symbol=marker_symbol,
                size=marker_size,
                color=marker_color,
                colorscale=colorscale,
                showscale=showscale,
                line=dict(width=marker_line_width, color='white'),
                colorbar=dict(title='Crowding<br>Distance') if showscale else None
            ),
            hovertemplate=f'<b>{name}</b><br>' +
                         f'{obj1}: %{{x:.4f}}<br>' +
                         f'{obj2}: %{{y:.4f}}<br>' +
                         'Crowding: %{customdata:.3f}<extra></extra>',
            customdata=df_rank['crowding_distance']
        ))

    # Add optimization direction arrows
    if show_arrows:
        # Determine arrow directions based on objectives
        obj1_type = objectives_dict.get(obj1, 'maximize')
        obj2_type = objectives_dict.get(obj2, 'maximize')

        x_arrow_dir = 1 if (obj1_type == 'maximize') else -1
        y_arrow_dir = 1 if (obj2_type == 'maximize') else -1

        # Arrow positions (top-right corner)
        x_range = df[obj1].max() - df[obj1].min()
        y_range = df[obj2].max() - df[obj2].min()

        arrow_x_start = df[obj1].max() - 0.15 * x_range
        arrow_y_start = df[obj2].max() - 0.15 * y_range

        # Add annotations with arrows
        fig.add_annotation(
            x=arrow_x_start + 0.1 * x_range * x_arrow_dir,
            y=arrow_y_start,
            ax=arrow_x_start,
            ay=arrow_y_start,
            xref='x', yref='y',
            axref='x', ayref='y',
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor='gray',
            text=f"{'Maximize' if x_arrow_dir > 0 else 'Minimize'}<br>{obj1}",
            font=dict(size=10, color='gray'),
            bgcolor='rgba(255,255,255,0.8)',
            borderpad=4
        )

        fig.add_annotation(
            x=arrow_x_start,
            y=arrow_y_start + 0.1 * y_range * y_arrow_dir,
            ax=arrow_x_start,
            ay=arrow_y_start,
            xref='x', yref='y',
            axref='x', ayref='y',
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor='gray',
            text=f"{'Maximize' if y_arrow_dir > 0 else 'Minimize'}<br>{obj2}",
            font=dict(size=10, color='gray'),
            bgcolor='rgba(255,255,255,0.8)',
            borderpad=4
        )

    fig.update_layout(
        title=f'Pareto Front: {obj1} vs {obj2}',
        xaxis_title=obj1,
        yaxis_title=obj2,
        height=500,
        hovermode='closest',
        showlegend=True,
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.8)')
    )

    return fig


def plot_pareto_parallel_coordinates(df, objective_cols, n_points=50):
    """
    Create parallel coordinates plot for multi-objective visualization.

    Args:
        df: DataFrame with Pareto results (should be Front 1 only)
        objective_cols: List of objective column names
        n_points: Maximum number of points to display

    Returns:
        plotly Figure
    """
    # Take top n_points by crowding distance
    df_plot = df.nlargest(n_points, 'crowding_distance')

    # Prepare dimensions for parallel coordinates
    dimensions = []

    for col in objective_cols:
        dimensions.append(dict(
            label=col,
            values=df_plot[col],
            range=[df[col].min(), df[col].max()]
        ))

    # Color by crowding distance
    crowding_finite = df_plot['crowding_distance'].replace([np.inf],
                        df_plot['crowding_distance'][np.isfinite(df_plot['crowding_distance'])].max() * 2)

    fig = go.Figure(data=
        go.Parcoords(
            line=dict(
                color=crowding_finite,
                colorscale='Viridis',
                showscale=True,
                cmin=crowding_finite.min(),
                cmax=crowding_finite.max(),
                colorbar=dict(title='Crowding<br>Distance')
            ),
            dimensions=dimensions
        )
    )

    fig.update_layout(
        title=f'Parallel Coordinates: Pareto Front 1 (Top {len(df_plot)} points)',
        height=400,
        margin=dict(l=100, r=100, t=80, b=50)
    )

    return fig


def plot_pareto_tradeoff_matrix(df, objectives_dict):
    """
    Create tradeoff matrix showing pairwise objective conflicts.

    Args:
        df: DataFrame with first Pareto front points
        objectives_dict: Dictionary of objectives

    Returns:
        plotly Figure with subplots (scatter matrix)
    """
    objective_cols = list(objectives_dict.keys())
    n_objectives = len(objective_cols)

    if n_objectives < 2:
        st.warning("Need at least 2 objectives for tradeoff matrix")
        return None

    # Create subplot grid
    fig = make_subplots(
        rows=n_objectives,
        cols=n_objectives,
        subplot_titles=[f'{obj1} vs {obj2}'
                       for obj1 in objective_cols
                       for obj2 in objective_cols],
        vertical_spacing=0.05,
        horizontal_spacing=0.05
    )

    # Fill subplots
    for i, obj1 in enumerate(objective_cols):
        for j, obj2 in enumerate(objective_cols):
            if i == j:
                # Diagonal: histogram
                fig.add_trace(
                    go.Histogram(
                        x=df[obj1],
                        name=obj1,
                        showlegend=False,
                        marker_color='lightblue'
                    ),
                    row=i+1, col=j+1
                )
            else:
                # Off-diagonal: scatter
                fig.add_trace(
                    go.Scatter(
                        x=df[obj2],
                        y=df[obj1],
                        mode='markers',
                        marker=dict(
                            size=6,
                            color=df['crowding_distance'].replace([np.inf], df['crowding_distance'][np.isfinite(df['crowding_distance'])].max() * 2),
                            colorscale='Viridis',
                            showscale=(i == 0 and j == n_objectives - 1),
                            colorbar=dict(title='Crowding') if (i == 0 and j == n_objectives - 1) else None
                        ),
                        showlegend=False,
                        hovertemplate=f'{obj2}: %{{x:.3f}}<br>{obj1}: %{{y:.3f}}<extra></extra>'
                    ),
                    row=i+1, col=j+1
                )

            # Update axes labels
            if j == 0:
                fig.update_yaxes(title_text=obj1, row=i+1, col=j+1)
            if i == n_objectives - 1:
                fig.update_xaxes(title_text=obj2, row=i+1, col=j+1)

    fig.update_layout(
        title='Objective Tradeoff Matrix (Pareto Front 1)',
        height=200 * n_objectives,
        showlegend=False
    )

    return fig


def export_pareto_results(df, objectives_dict, filename='Pareto_Analysis.xlsx'):
    """
    Export Pareto analysis results to Excel with multiple sheets.

    Sheets:
    - Summary: Objectives and statistics
    - Front_1: First Pareto front (best compromises)
    - Front_2: Second Pareto front
    - Front_3: Third Pareto front
    - All_Points: All evaluated points with rankings

    Returns:
        BytesIO buffer for download
    """
    excel_buffer = BytesIO()

    with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
        # Sheet 1: Summary
        summary_data = {
            'Objective': [],
            'Type': [],
            'Target_Value': []
        }

        for obj_name, obj_type in objectives_dict.items():
            summary_data['Objective'].append(obj_name)
            if isinstance(obj_type, tuple):
                summary_data['Type'].append(obj_type[0])
                summary_data['Target_Value'].append(obj_type[1])
            else:
                summary_data['Type'].append(obj_type)
                summary_data['Target_Value'].append('N/A')

        summary_df = pd.DataFrame(summary_data)

        # Add statistics
        stats_df = pd.DataFrame({
            'Metric': [
                'Total_Candidates',
                'Pareto_Front_1_Size',
                'Pareto_Front_2_Size',
                'Pareto_Front_3_Size',
                'Dominated_Points'
            ],
            'Value': [
                len(df),
                (df['pareto_rank'] == 1).sum(),
                (df['pareto_rank'] == 2).sum(),
                (df['pareto_rank'] == 3).sum(),
                df['is_dominated'].sum()
            ]
        })

        # Combine and export
        summary_df.to_excel(writer, sheet_name='Objectives', index=False)
        stats_df.to_excel(writer, sheet_name='Statistics', index=False)

        # Sheet 2-4: Pareto Fronts
        for rank in [1, 2, 3]:
            df_front = df[df['pareto_rank'] == rank].copy()

            if len(df_front) > 0:
                # Sort by crowding distance
                df_front = df_front.sort_values('crowding_distance', ascending=False)
                df_front.to_excel(writer, sheet_name=f'Front_{rank}', index=True)

        # Sheet 5: All points
        df_all = df.sort_values(['pareto_rank', 'crowding_distance'], ascending=[True, False])
        df_all.to_excel(writer, sheet_name='All_Points', index=True)

    excel_buffer.seek(0)
    return excel_buffer
