"""
Unified Color Mapping System for PCA Analysis
Light theme only - simplified color system
"""

import numpy as np
import pandas as pd

def get_unified_color_schemes():
    """
    Unified color schemes for light theme only
    
    Returns:
        dict: Complete color scheme with categorical colors and plot styling
    """
    
    # Colori per tema chiaro (sfondo bianco)
    light_theme_colors = [
        'black', 'red', 'green', 'blue', 'orange', 'purple', 'brown', 'hotpink', 
        'gray', 'olive', 'cyan', 'magenta', 'gold', 'navy', 'darkgreen', 'darkred', 
        'indigo', 'coral', 'teal', 'chocolate', 'crimson', 'darkviolet', 'darkorange', 
        'darkslategray', 'royalblue', 'saddlebrown'
    ]
    
    return {
        # Plot styling colors
        'background': 'white',
        'paper': 'white', 
        'text': 'black',
        'grid': '#e6e6e6',
        'control_colors': ['green', 'orange', 'red'],  # Standard colors for light theme
        'point_color': 'blue',
        'line_colors': ['blue', 'red'],
        
        # Categorical colors for data points
        'categorical_colors': light_theme_colors,
        
        # Color mapping dictionary (for backward compatibility)
        'color_map': {chr(65+i): color for i, color in enumerate(light_theme_colors)},
        
        # Theme identifier
        'theme': 'light'
    }


def create_categorical_color_map(unique_values):
    """
    Create a color mapping for categorical variables
    
    Args:
        unique_values (list): List of unique categorical values
    
    Returns:
        dict: Mapping of values to colors
    """
    color_scheme = get_unified_color_schemes()
    colors = color_scheme['categorical_colors']
    
    # Create mapping for unique values
    color_discrete_map = {}
    
    for i, val in enumerate(sorted(unique_values)):
        if i < len(colors):
            color_discrete_map[val] = colors[i]
        else:
            # Generate additional colors using HSL
            color_discrete_map[val] = f'hsl({(i*137) % 360}, 70%, 50%)'
    
    return color_discrete_map


def create_quantitative_color_map(values, colorscale='blue_to_red'):
    """
    Create a color mapping for quantitative variables
    Genera una scala cromatica continua dal blu puro al rosso puro
    
    Args:
        values (array-like): Array of quantitative values
        colorscale (str): Type of color scale ('blue_to_red', 'viridis', etc.)
    
    Returns:
        dict: Mapping of values to RGB colors
    """
    values = pd.Series(values).dropna()
    
    if len(values) == 0:
        return {}
    
    # Normalizza i valori tra 0 e 1
    min_val = values.min()
    max_val = values.max()
    
    if min_val == max_val:
        # Tutti i valori sono uguali, usa un colore singolo
        return {val: 'rgb(128, 0, 128)' for val in values.unique()}
    
    normalized_values = (values - min_val) / (max_val - min_val)
    
    color_map = {}
    
    for i, val in enumerate(values):
        if pd.isna(val):
            color_map[val] = 'rgb(128, 128, 128)'  # Grigio per valori mancanti
            continue
            
        norm_val = normalized_values.iloc[i]
        
        if colorscale == 'blue_to_red':
            # Scala dal blu puro (0) al rosso puro (1)
            r = int(255 * norm_val)
            g = 0
            b = int(255 * (1 - norm_val))
            color_map[val] = f'rgb({r},{g},{b})'
        
        elif colorscale == 'viridis':
            # Scala viridis approssimata
            if norm_val < 0.25:
                r, g, b = int(68 + norm_val * 4 * (85-68)), int(1 + norm_val * 4 * (104-1)), int(84 + norm_val * 4 * (109-84))
            elif norm_val < 0.5:
                r, g, b = int(85 + (norm_val-0.25) * 4 * (59-85)), int(104 + (norm_val-0.25) * 4 * (142-104)), int(109 + (norm_val-0.25) * 4 * (140-109))
            elif norm_val < 0.75:
                r, g, b = int(59 + (norm_val-0.5) * 4 * (94-59)), int(142 + (norm_val-0.5) * 4 * (201-142)), int(140 + (norm_val-0.5) * 4 * (98-140))
            else:
                r, g, b = int(94 + (norm_val-0.75) * 4 * (253-94)), int(201 + (norm_val-0.75) * 4 * (231-201)), int(98 + (norm_val-0.75) * 4 * (37-98))
            
            color_map[val] = f'rgb({r},{g},{b})'
        
        else:
            # Default: blue to red
            r = int(255 * norm_val)
            g = 0
            b = int(255 * (1 - norm_val))
            color_map[val] = f'rgb({r},{g},{b})'
    
    return color_map


def get_continuous_color_for_value(value, min_val, max_val, colorscale='blue_to_red'):
    """
    Get a single color for a specific value in a continuous range

    Args:
        value (float): The value to get color for
        min_val (float): Minimum value in the range
        max_val (float): Maximum value in the range
        colorscale (str): Color scale type

    Returns:
        str: RGB color string
    """
    if pd.isna(value):
        return 'rgb(128, 128, 128)'

    if min_val == max_val:
        return 'rgb(128, 0, 128)'

    # Normalizza il valore
    norm_val = (value - min_val) / (max_val - min_val)
    norm_val = max(0, min(1, norm_val))  # Clamp tra 0 e 1

    if colorscale == 'blue_to_red':
        # Scala dal blu puro al rosso puro
        r = int(255 * norm_val)
        g = 0
        b = int(255 * (1 - norm_val))
        return f'rgb({r},{g},{b})'

    # Default
    r = int(255 * norm_val)
    g = 0
    b = int(255 * (1 - norm_val))
    return f'rgb({r},{g},{b})'


def get_trajectory_colors(n_points, style='gradient', last_point_marker=True):
    """
    Generate colors for trajectory visualization in monitoring plots

    Args:
        n_points (int): Number of points in trajectory
        style (str): 'simple' for light gray line, 'gradient' for blue-to-red progression
        last_point_marker (bool): Whether to include special marker for last point

    Returns:
        dict: Dictionary containing:
            - 'line_colors': List of RGB color strings for each segment
            - 'line_widths': List of line widths for each segment
            - 'marker_colors': List of marker colors for each point
            - 'marker_sizes': List of marker sizes for each point
            - 'last_point_style': Dictionary with last point marker style (if enabled)
    """
    trajectory_colors = {
        'line_colors': [],
        'line_widths': [],
        'marker_colors': [],
        'marker_sizes': [],
        'last_point_style': None
    }

    if style == 'simple':
        # Light gray uniform trajectory
        trajectory_colors['line_colors'] = ['lightgray'] * max(1, n_points - 1)
        trajectory_colors['line_widths'] = [1.5] * max(1, n_points - 1)
        trajectory_colors['marker_colors'] = ['lightgray'] * n_points
        trajectory_colors['marker_sizes'] = [4] * n_points

        if last_point_marker and n_points > 0:
            trajectory_colors['last_point_style'] = {
                'color': 'lightgray',
                'size': 6,
                'symbol': 'circle'
            }

    elif style == 'gradient':
        # Blue-to-red gradient with increasing thickness
        for i in range(max(1, n_points - 1)):
            ratio = i / max(1, n_points - 2)

            # Blue to red gradient
            r = int(255 * ratio)
            g = 0
            b = int(255 * (1 - ratio))
            color = f'rgb({r},{g},{b})'

            trajectory_colors['line_colors'].append(color)
            trajectory_colors['line_widths'].append(1.5 + (ratio * 1.5))
            trajectory_colors['marker_colors'].append(color)
            trajectory_colors['marker_sizes'].append(3 + (ratio * 3))

        # Add color for last marker if needed
        if n_points > 0:
            trajectory_colors['marker_colors'].append('rgb(255,0,0)')
            trajectory_colors['marker_sizes'].append(6)

        if last_point_marker and n_points > 0:
            trajectory_colors['last_point_style'] = {
                'color': 'cyan',
                'size': 10,
                'symbol': 'star'
            }

    return trajectory_colors


def get_sample_order_colors(n_points, colorscale='blue_to_red'):
    """
    Generate color gradient for data points based on their order in sequence
    Used for coloring points independently from trajectory visualization

    Args:
        n_points (int): Number of data points
        colorscale (str): Color scale type ('blue_to_red' default)

    Returns:
        list: List of RGB color strings, one for each point
    """
    colors = []

    for i in range(n_points):
        if n_points == 1:
            # Single point: use blue
            colors.append('rgb(0, 0, 255)')
        else:
            # Multiple points: gradient from blue to red
            ratio = i / (n_points - 1)

            if colorscale == 'blue_to_red':
                r = int(255 * ratio)
                g = 0
                b = int(255 * (1 - ratio))
                colors.append(f'rgb({r},{g},{b})')
            else:
                # Default to blue_to_red
                r = int(255 * ratio)
                g = 0
                b = int(255 * (1 - ratio))
                colors.append(f'rgb({r},{g},{b})')

    return colors


def is_quantitative_variable(data):
    """
    Determine if a variable is quantitative (numeric and continuous)
    
    Args:
        data (pd.Series or array-like): Data to check
    
    Returns:
        bool: True if quantitative, False if categorical
    """
    if not hasattr(data, 'dtype'):
        data = pd.Series(data)
    
    # Check if numeric
    if not pd.api.types.is_numeric_dtype(data):
        return False
    
    # Check if too many unique values suggest continuous data
    n_unique = data.nunique()
    n_total = len(data.dropna())
    
    if n_total == 0:
        return False
    
    # If more than 50% unique values, consider it continuous
    # Or if more than 20 unique values
    return (n_unique / n_total > 0.5) or (n_unique > 20)


def get_custom_color_map():
    """
    Mappa colori personalizzata per variabili categoriche - VERSIONE UNIFICATA
    """
    return get_unified_color_schemes()['color_map']