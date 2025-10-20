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