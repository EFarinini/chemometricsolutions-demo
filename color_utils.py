"""
Unified Color Mapping System for PCA Analysis
Consistent color schemes for both pca.py and pca_diagnostics_complete.py
"""

def get_unified_color_schemes(dark_mode=False):
    """
    Unified color schemes for consistent theming across PCA modules
    Returns both categorical colors and plot styling colors
    
    Args:
        dark_mode (bool): If True, returns dark mode colors, else light mode
    
    Returns:
        dict: Complete color scheme with categorical colors and plot styling
    """
    
    # Colori per tema chiaro (sfondo bianco) - da PCA.py
    light_theme_colors = [
        'black', 'red', 'green', 'blue', 'orange', 'purple', 'brown', 'hotpink', 
        'gray', 'olive', 'cyan', 'magenta', 'gold', 'navy', 'darkgreen', 'darkred', 
        'indigo', 'coral', 'teal', 'chocolate', 'crimson', 'darkviolet', 'darkorange', 
        'darkslategray', 'royalblue', 'saddlebrown'
    ]
    
    # Colori per tema scuro (sfondo nero/grigio scuro) - da PCA.py
    dark_theme_colors = [
        'white', 'lightcoral', 'lightgreen', 'lightblue', 'orange', 'violet', 'tan', 
        'hotpink', 'lightgray', 'yellowgreen', 'cyan', 'magenta', 'gold', 'cornflowerblue', 
        'lime', 'tomato', 'mediumpurple', 'coral', 'turquoise', 'sandybrown', 'crimson', 
        'plum', 'darkorange', 'lightsteelblue', 'royalblue', 'wheat'
    ]
    
    if dark_mode:
        return {
            # Plot styling colors for diagnostics
            'background': 'rgba(0,0,0,0)',  # Transparent - let Streamlit handle
            'paper': 'rgba(0,0,0,0)', 
            'text': '#ffffff',
            'grid': '#404040',
            'control_colors': ['lime', 'orange', 'red'],  # Bright colors for dark theme
            'point_color': 'lightblue',
            'line_colors': ['lightblue', 'lightcoral'],
            
            # Categorical colors for data points
            'categorical_colors': dark_theme_colors,
            
            # Color mapping dictionary (for backward compatibility)
            'color_map': {chr(65+i): color for i, color in enumerate(dark_theme_colors)},
            
            # Theme identifier
            'theme': 'dark'
        }
    else:
        return {
            # Plot styling colors for diagnostics
            'background': 'rgba(0,0,0,0)',  # Transparent - let Streamlit handle
            'paper': 'rgba(0,0,0,0)', 
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


def create_categorical_color_map(unique_values, dark_mode=False):
    """
    Create a color mapping for categorical variables
    
    Args:
        unique_values (list): List of unique categorical values
        dark_mode (bool): Whether to use dark mode colors
    
    Returns:
        dict: Mapping of values to colors
    """
    color_scheme = get_unified_color_schemes(dark_mode)
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


# Updated function for pca_diagnostics_complete.py
def get_color_schemes(dark_mode=False):
    """
    UPDATED VERSION - now uses unified color system
    Get color schemes for light and dark modes - using exact PCA.py colors
    """
    return get_unified_color_schemes(dark_mode)


# Updated function for pca.py
def get_custom_color_map(theme='auto'):
    """
    UPDATED VERSION - now uses unified color system
    Mappa colori personalizzata per variabili categoriche
    theme: 'light', 'dark', o 'auto'
    """
    if theme == 'light':
        return get_unified_color_schemes(dark_mode=False)['color_map']
    elif theme == 'dark':
        return get_unified_color_schemes(dark_mode=True)['color_map']
    else:
        # Auto: usa tema scuro come default (più comune)
        return get_unified_color_schemes(dark_mode=True)['color_map']


# Helper function for creating plotly scatter plots with consistent colors
def create_scatter_with_unified_colors(x, y, color_data=None, dark_mode=False, **kwargs):
    """
    Create a plotly scatter plot with unified color scheme
    
    Args:
        x, y: Data for scatter plot
        color_data: Categorical data for coloring points
        dark_mode: Whether to use dark mode colors
        **kwargs: Additional arguments for px.scatter
    
    Returns:
        plotly figure with consistent colors
    """
    import plotly.express as px
    import pandas as pd
    
    if color_data is not None:
        unique_values = pd.Series(color_data).dropna().unique()
        color_discrete_map = create_categorical_color_map(unique_values, dark_mode)
        
        return px.scatter(
            x=x, y=y, color=color_data,
            color_discrete_map=color_discrete_map,
            **kwargs
        )
    else:
        color_scheme = get_unified_color_schemes(dark_mode)
        return px.scatter(
            x=x, y=y,
            color_discrete_sequence=[color_scheme['point_color']],
            **kwargs
        )


# Example usage and integration instructions
def integration_example():
    """
    Example of how to integrate this unified system
    """
    
    # For pca.py - replace existing get_custom_color_map function
    # def get_custom_color_map(theme='auto'):
    #     return get_custom_color_map(theme)  # Use the updated version above
    
    # For pca_diagnostics_complete.py - replace existing get_color_schemes function  
    # def get_color_schemes(dark_mode=False):
    #     return get_color_schemes(dark_mode)  # Use the updated version above
    
    # Example usage in both files:
    dark_mode = True  # or False
    color_scheme = get_unified_color_schemes(dark_mode)
    
    # For categorical data coloring
    unique_categories = ['A', 'B', 'C', 'D']
    color_map = create_categorical_color_map(unique_categories, dark_mode)
    
    # For plot styling
    plot_colors = color_scheme['control_colors']  # For control lines
    point_color = color_scheme['point_color']     # For single-color points
    
    print(f"Theme: {color_scheme['theme']}")
    print(f"Categorical colors available: {len(color_scheme['categorical_colors'])}")
    print(f"Color map for categories: {color_map}")
    
    return color_scheme, color_map


# Test the unified system
if __name__ == "__main__":
    print("=== Unified Color System Test ===")
    
    # Test light mode
    print("\nLight Mode:")
    light_scheme, light_map = integration_example()
    
    # Test dark mode  
    print("\nDark Mode:")
    dark_mode = True
    dark_scheme = get_unified_color_schemes(dark_mode)
    dark_map = create_categorical_color_map(['Group1', 'Group2', 'Group3'], dark_mode)
    print(f"Theme: {dark_scheme['theme']}")
    print(f"Color map: {dark_map}")
    
    print("\n=== Integration Complete ===")