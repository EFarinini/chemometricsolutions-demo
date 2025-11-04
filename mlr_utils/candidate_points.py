"""
Candidate Points Generation & Design of Experiments
Equivalent to DOE_candidate_points.r
Comprehensive standalone design generator - no data/model required

Features:
- Multiple design types: Full Factorial, Plackett-Burman, Central Composite Design
- Automatic coding: Quantitative → [-1, +1], Qualitative → dummy variables
- Constraint builder for excluding combinations
- 2D and 3D interactive visualization with rotation/zoom
- Point classification: corner, edge, center, axial points
- CSV/Excel export with coded matrices
"""

import streamlit as st
import pandas as pd
import itertools
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import encoding functions from data_handling
# Note: These functions are optional and used only in the coded matrix export section
# from data_handling import encode_quantitative, encode_full_matrix


def is_numeric_list(values_str):
    """Check if comma-separated string contains only numeric values"""
    try:
        [float(x.strip()) for x in values_str.split(',') if x.strip()]
        return True
    except ValueError:
        return False


def parse_levels(levels_input, is_numeric=True):
    """
    Parse levels from user input

    Args:
        levels_input: comma-separated string
        is_numeric: if True, try numeric parsing

    Returns:
        tuple: (levels_list, is_numeric_actual)
    """
    if not levels_input or not levels_input.strip():
        return None, False

    # Try numeric first
    try:
        levels = [float(x.strip()) for x in levels_input.split(',')]
        return levels, True
    except ValueError:
        # Non-numeric
        levels = [x.strip() for x in levels_input.split(',') if x.strip()]
        return levels, False


def generate_full_factorial(levels_dict):
    """
    Generate full factorial design (all combinations)

    Args:
        levels_dict: dict with variable names and their levels

    Returns:
        DataFrame with all combinations
    """
    keys = list(levels_dict.keys())
    values = list(levels_dict.values())

    combinations = list(itertools.product(*values))
    design = pd.DataFrame(combinations, columns=keys)

    return design


# ============================================================================
# VERIFIED PLACKETT-BURMANN DESIGNS (Hardcoded from EXC_PB.xlsx)
# ============================================================================

PB_DESIGNS = {
    8: [
        [1, -1, -1, 1, -1, 1, 1],
        [1, 1, -1, -1, 1, -1, 1],
        [1, 1, 1, -1, -1, 1, -1],
        [-1, 1, 1, 1, -1, -1, 1],
        [1, -1, 1, 1, 1, -1, -1],
        [-1, 1, -1, 1, 1, 1, -1],
        [-1, -1, 1, -1, 1, 1, 1],
        [-1, -1, -1, -1, -1, -1, -1],
    ],
    12: [
        [1, 1, -1, 1, 1, 1, -1, -1, -1, 1, -1],
        [-1, 1, 1, -1, 1, 1, 1, -1, -1, -1, 1],
        [1, -1, 1, 1, -1, 1, 1, 1, -1, -1, -1],
        [-1, 1, -1, 1, 1, -1, 1, 1, 1, -1, -1],
        [-1, -1, 1, -1, 1, 1, -1, 1, 1, 1, -1],
        [-1, -1, -1, 1, -1, 1, 1, -1, 1, 1, 1],
        [1, -1, -1, -1, 1, -1, 1, 1, -1, 1, 1],
        [1, 1, -1, -1, -1, 1, -1, 1, 1, -1, 1],
        [1, 1, 1, -1, -1, -1, 1, -1, 1, 1, -1],
        [-1, 1, 1, 1, -1, -1, -1, 1, -1, 1, 1],
        [1, -1, 1, 1, 1, -1, -1, -1, 1, -1, 1],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    ],
    16: [
        [1, 1, 1, 1, -1, 1, -1, 1, 1, -1, -1, 1, -1, -1, -1],
        [-1, 1, 1, 1, 1, -1, 1, -1, 1, 1, -1, -1, 1, -1, -1],
        [-1, -1, 1, 1, 1, 1, -1, 1, -1, 1, 1, -1, -1, 1, -1],
        [-1, -1, -1, 1, 1, 1, 1, -1, 1, -1, 1, 1, -1, -1, 1],
        [1, -1, -1, -1, 1, 1, 1, 1, -1, 1, -1, 1, 1, -1, -1],
        [-1, 1, -1, -1, -1, 1, 1, 1, 1, -1, 1, -1, 1, 1, -1],
        [-1, -1, 1, -1, -1, -1, 1, 1, 1, 1, -1, 1, -1, 1, 1],
        [1, -1, -1, 1, -1, -1, -1, 1, 1, 1, 1, -1, 1, -1, 1],
        [1, 1, -1, -1, 1, -1, -1, -1, 1, 1, 1, 1, -1, 1, -1],
        [-1, 1, 1, -1, -1, 1, -1, -1, -1, 1, 1, 1, 1, -1, 1],
        [1, -1, 1, 1, -1, -1, 1, -1, -1, -1, 1, 1, 1, 1, -1],
        [-1, 1, -1, 1, 1, -1, -1, 1, -1, -1, -1, 1, 1, 1, 1],
        [1, -1, 1, -1, 1, 1, -1, -1, 1, -1, -1, -1, 1, 1, 1],
        [1, 1, -1, 1, -1, 1, 1, -1, -1, 1, -1, -1, -1, 1, 1],
        [1, 1, 1, -1, 1, -1, 1, 1, -1, -1, 1, -1, -1, -1, 1],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    ],
}


# ============================================================================
# FRACTIONAL FACTORIAL DESIGNS (Hardcoded)
# From EXC_FFFF.xlsx
# ============================================================================

FF_DESIGNS = {
    # 2^4-1: 8 runs (resolution IV, 4 factors)
    "2^4-1": {
        "runs": 8,
        "factors": 4,
        "resolution": "IV",
        "matrix": [
            [-1, -1, -1, -1],
            [1, -1, -1, 1],
            [-1, 1, -1, 1],
            [1, 1, -1, -1],
            [-1, -1, 1, 1],
            [1, -1, 1, -1],
            [-1, 1, 1, -1],
            [1, 1, 1, 1],
        ]
    },
    # 2^5-1: 16 runs (resolution V, 5 factors)
    "2^5-1": {
        "runs": 16,
        "factors": 5,
        "resolution": "V",
        "matrix": [
            [-1, -1, -1, -1, 1],
            [1, -1, -1, -1, -1],
            [-1, 1, -1, -1, -1],
            [1, 1, -1, -1, 1],
            [-1, -1, 1, -1, -1],
            [1, -1, 1, -1, 1],
            [-1, 1, 1, -1, 1],
            [1, 1, 1, -1, -1],
            [-1, -1, -1, 1, -1],
            [1, -1, -1, 1, 1],
            [-1, 1, -1, 1, 1],
            [1, 1, -1, 1, -1],
            [-1, -1, 1, 1, 1],
            [1, -1, 1, 1, -1],
            [-1, 1, 1, 1, -1],
            [1, 1, 1, 1, 1],
        ]
    }
}


def generate_plackett_burman(k, n_replicates=0):
    """
    Generate Plackett-Burmann design from verified hardcoded matrices

    Formula: N = first multiple of 4 ≥ (k+1)
    Examples: k=7 → N=8 | k=11 → N=12 | k=15 → N=16

    Args:
        k: number of factors (variables)
        n_replicates: number of additional replicates
                     (0 = no replicates, 1 = design repeated twice, etc.)

    Returns:
        DataFrame with coded design matrix [-1, +1]
        Columns: [X1, X2, ..., Xk]
        Shape: (N × (1 + n_replicates), k)
    """
    import math

    # Calculate required N (first multiple of 4 ≥ k+1)
    N = math.ceil((k + 1) / 4) * 4

    # Check if N is available in verified designs
    if N not in PB_DESIGNS:
        st.warning(f"""
        ⚠️ **N={N} not available in verified designs**
        Available: N ∈ {{8, 12, 16}} (supports k ≤ 15)

        **Fallback:** Using Full Factorial 2^{k} = {2**k} experiments instead
        """)

        # Fallback to Full Factorial 2^k
        fallback_dict = {f"X{i+1}": [-1, 1] for i in range(k)}
        fallback_matrix = generate_full_factorial(fallback_dict)

        if n_replicates > 0:
            fallback_matrix = pd.concat([fallback_matrix] * (n_replicates + 1), ignore_index=True)

        return fallback_matrix

    # Get verified PB matrix
    pb_matrix = pd.DataFrame(
        PB_DESIGNS[N],
        columns=[f"X{i+1}" for i in range(N-1)]
    )

    # Use only first k columns
    pb_matrix = pb_matrix.iloc[:, :k]

    # Apply replicates if requested
    if n_replicates > 0:
        pb_matrix = pd.concat([pb_matrix] * (n_replicates + 1), ignore_index=True)

    # Show info message
    st.info(f"""
    ✅ **Plackett-Burmann Design (Verified Matrix)**
    - Factors (k): {k}
    - Design size (N): {N}
    - Total runs: {len(pb_matrix)} = {N} × (1 + {n_replicates} replicates)
    - Source: **EXC_PB.xlsx (hardcoded - verified correct)**
    """)

    return pb_matrix


def generate_fractional_factorial(design_name, k, n_replicates=0):
    """
    Generate Fractional Factorial design (2^k-1)

    Formula:
    - 2^4-1: 8 runs, 4 factors, Resolution IV
    - 2^5-1: 16 runs, 5 factors, Resolution V

    Args:
        design_name: "2^4-1" or "2^5-1"
        k: number of factors (must match design)
        n_replicates: number of replicates

    Returns:
        DataFrame with FF design (±1 coded)
    """

    if design_name not in FF_DESIGNS:
        st.warning(f"⚠️ Design {design_name} not available")
        return None

    design_spec = FF_DESIGNS[design_name]
    N = design_spec['runs']
    n_factors = design_spec['factors']
    resolution = design_spec['resolution']

    # Check if k matches design
    if k > n_factors:
        st.warning(f"⚠️ {design_name} supports max {n_factors} factors, but {k} requested")
        return None

    # Get matrix
    ff_matrix_list = design_spec['matrix']

    # Convert to DataFrame
    ff_matrix = pd.DataFrame(
        ff_matrix_list,
        columns=[f"X{i+1}" for i in range(n_factors)]
    )

    # Use only first k columns
    ff_matrix = ff_matrix.iloc[:, :k]

    # Apply replicates if requested
    if n_replicates > 0:
        ff_matrix = pd.concat([ff_matrix] * (n_replicates + 1), ignore_index=True)

    # Show info
    st.info(f"""
    ✅ **Fractional Factorial Design {design_name} (Resolution {resolution})**
    - Factors (k): {k}
    - Design runs (N): {N}
    - Total runs: {len(ff_matrix)} = {N} × (1 + {n_replicates} replicates)
    - Source: **Verified Hardcoded Matrices**
    """)

    return ff_matrix


def generate_central_composite_design(n_variables, alpha='orthogonal'):
    """
    Generate Central Composite Design (CCD) for response surface methodology

    Args:
        n_variables: number of factors
        alpha: 'orthogonal', 'rotatable', or numeric value for axial distance

    Returns:
        DataFrame with coded design matrix
    """
    # Factorial points (corners of cube): 2^k
    factorial_dict = {f"X{i+1}": [-1, 1] for i in range(n_variables)}
    factorial_points = generate_full_factorial(factorial_dict)

    # Calculate alpha (axial distance)
    if alpha == 'orthogonal':
        n_f = 2 ** n_variables
        alpha_val = np.sqrt((np.sqrt(n_f * (n_f + 2)) - n_f) / 2)
    elif alpha == 'rotatable':
        alpha_val = (2 ** n_variables) ** 0.25
    else:
        alpha_val = float(alpha)

    # Axial points (star points): 2*k
    axial_points = []
    for i in range(n_variables):
        # +alpha point
        point_plus = [0] * n_variables
        point_plus[i] = alpha_val
        axial_points.append(point_plus)
        # -alpha point
        point_minus = [0] * n_variables
        point_minus[i] = -alpha_val
        axial_points.append(point_minus)

    axial_df = pd.DataFrame(axial_points, columns=[f"X{i+1}" for i in range(n_variables)])

    # Center points (typically 3-5 replicates)
    n_center = max(3, n_variables)
    center_points = pd.DataFrame(
        [[0] * n_variables] * n_center,
        columns=[f"X{i+1}" for i in range(n_variables)]
    )

    # Combine all points
    design = pd.concat([factorial_points, axial_df, center_points], ignore_index=True)

    return design


def code_quantitative_variable(values, min_val, max_val):
    """
    Code quantitative variable to [-1, +1] range

    Args:
        values: array of real values
        min_val: minimum value (maps to -1)
        max_val: maximum value (maps to +1)

    Returns:
        array of coded values
    """
    return 2 * (values - min_val) / (max_val - min_val) - 1


def decode_quantitative_variable(coded_values, min_val, max_val):
    """
    Decode from [-1, +1] to real values

    Args:
        coded_values: array of coded values
        min_val: minimum value
        max_val: maximum value

    Returns:
        array of real values
    """
    return min_val + (coded_values + 1) * (max_val - min_val) / 2


def encode_full_matrix(design_real, variables_info):
    """
    Encode complete design matrix from real values to coded/dummy format

    - Quantitative variables: real [min, max] → coded [-1, +1]
    - Qualitative variables (k>2): expanded to k-1 dummy variables (reference = 0)
    - Qualitative variables (k=2): mapped to single dummy (0/1)

    Args:
        design_real: DataFrame with real values (natural scale)
        variables_info: list of variable configuration dicts

    Returns:
        design_coded: DataFrame with coded/dummy values ready for MLR
    """
    design_coded = pd.DataFrame()

    for var_info in variables_info:
        var_name = var_info['name']
        var_type = var_info['type']

        if var_type == 'Quantitative':
            # Encode quantitative: [min, max] → [-1, +1]
            real_col = design_real[var_name].values
            min_val = var_info['min']
            max_val = var_info['max']

            coded_col = 2 * (real_col - min_val) / (max_val - min_val) - 1
            design_coded[var_name] = np.round(coded_col, 3)

        elif var_type == 'Qualitative':
            # Encode qualitative: dummy coding (k-1 for k>2, 1 for k=2)
            categories = var_info['categories']
            n_categories = len(categories)
            reference_idx = var_info.get('reference_level', 0)
            reference_cat = categories[reference_idx]

            if n_categories > 2:
                # Create k-1 dummy variables (skip reference)
                for i, category in enumerate(categories):
                    if i != reference_idx:
                        dummy_col_name = f"{var_name}_{category}"
                        design_coded[dummy_col_name] = (design_real[var_name] == category).astype(int)

            elif n_categories == 2:
                # Binary: create single dummy (second category = 1)
                non_ref_cat = categories[1] if reference_idx == 0 else categories[0]
                dummy_col_name = f"{var_name}_{non_ref_cat}"
                design_coded[dummy_col_name] = (design_real[var_name] == non_ref_cat).astype(int)

    return design_coded


def create_dummy_variables(values, categories):
    """
    Create dummy variables for qualitative factors

    Args:
        values: array of categorical values
        categories: list of unique categories

    Returns:
        DataFrame with dummy columns (one-hot encoded)
    """
    df = pd.DataFrame({'value': values})
    dummies = pd.get_dummies(df['value'], prefix='', prefix_sep='')
    return dummies


def dummy_coding(data_column, var_name, categories):
    """
    Create dummy coded matrix for qualitative variables (n-1 coding)

    Args:
        data_column: Series with categorical values
        var_name: name of the variable
        categories: list of category levels

    Returns:
        DataFrame with dummy columns (n-1 coding, first category is reference)

    Example:
        Input: ["Low","Medium","High"]
        Output: [[1,0],[0,1],[0,0]]  (Low=reference)
    """
    coded_df = pd.DataFrame()

    # Use first category as reference (all zeros)
    for i, category in enumerate(categories[:-1]):  # Skip last category
        dummy_col_name = f"{var_name}_{category}"
        # 1 if matches this category, 0 otherwise
        coded_df[dummy_col_name] = (data_column == category).astype(int)

    return coded_df


def apply_constraints(design, constraints):
    """
    Apply constraint filters to design matrix

    Args:
        design: DataFrame with design points
        constraints: list of constraint functions

    Returns:
        Filtered DataFrame
    """
    if not constraints:
        return design

    mask = pd.Series([True] * len(design))
    for constraint_func in constraints:
        try:
            mask &= constraint_func(design)
        except Exception as e:
            st.warning(f"Constraint error: {str(e)}")

    return design[mask].reset_index(drop=True)


def plot_design_2d(design, x_var, y_var, color_var=None):
    """
    Create 2D scatter plot of design points

    Args:
        design: DataFrame with design matrix
        x_var: column name for x-axis
        y_var: column name for y-axis
        color_var: optional column for color coding

    Returns:
        plotly figure
    """
    fig = go.Figure()

    if color_var and color_var in design.columns:
        # Color by variable
        unique_vals = design[color_var].unique()
        for val in unique_vals:
            subset = design[design[color_var] == val]
            fig.add_trace(go.Scatter(
                x=subset[x_var],
                y=subset[y_var],
                mode='markers',
                marker=dict(size=10),
                name=f"{color_var}={val}",
                text=[f"Point {i+1}" for i in subset.index],
                hovertemplate=f'<b>Point %{{text}}</b><br>{x_var}: %{{x}}<br>{y_var}: %{{y}}<extra></extra>'
            ))
    else:
        # Single color
        fig.add_trace(go.Scatter(
            x=design[x_var],
            y=design[y_var],
            mode='markers',
            marker=dict(size=10, color='blue'),
            text=[f"Point {i+1}" for i in range(len(design))],
            hovertemplate=f'<b>Point %{{text}}</b><br>{x_var}: %{{x}}<br>{y_var}: %{{y}}<extra></extra>'
        ))

    fig.update_layout(
        title=f"Design Space: {x_var} vs {y_var}",
        xaxis_title=x_var,
        yaxis_title=y_var,
        height=500,
        showlegend=bool(color_var)
    )

    return fig


def classify_design_points(design, coded_design=None):
    """
    Classify design points as corner, edge, center, or axial points

    Args:
        design: DataFrame with design matrix (real values)
        coded_design: Optional DataFrame with coded values (for better classification)

    Returns:
        list of point types: 'corner', 'edge', 'center', 'axial', 'interior'
    """
    if coded_design is None:
        # Try to detect from real values
        coded_design = design.copy()
        for col in design.columns:
            if design[col].dtype in [np.float64, np.int64]:
                col_min = design[col].min()
                col_max = design[col].max()
                if col_max != col_min:
                    coded_design[col] = 2 * (design[col] - col_min) / (col_max - col_min) - 1

    point_types = []
    n_vars = len(coded_design.columns)

    for idx in range(len(coded_design)):
        row = coded_design.iloc[idx].values

        # Check if all values are 0 (center point)
        if np.allclose(row, 0, atol=0.1):
            point_types.append('center')
            continue

        # Count how many values are at extremes (-1 or +1)
        n_at_extremes = np.sum(np.abs(np.abs(row) - 1) < 0.1)

        # Count how many values are at center (0)
        n_at_center = np.sum(np.abs(row) < 0.1)

        # Classify based on position
        if n_at_extremes == n_vars:
            # All at extremes: corner point (factorial point)
            point_types.append('corner')
        elif n_at_extremes == 1 and n_at_center == n_vars - 1:
            # One at extreme, rest at center: axial/star point
            point_types.append('axial')
        elif n_at_extremes >= 1 and n_at_center >= 1:
            # Mix of extremes and centers: edge point
            point_types.append('edge')
        else:
            # Interior point (not at specific design position)
            point_types.append('interior')

    return point_types


def create_3d_scatter(design, x_var, y_var, z_var, color_var=None, point_types=None):
    """
    Create 3D scatter plot of design points with rotation and zoom

    Args:
        design: DataFrame with design matrix
        x_var: column name for x-axis
        y_var: column name for y-axis
        z_var: column name for z-axis
        color_var: optional column for color coding or 'experiment_number' or 'point_type'
        point_types: optional list of point classifications ('corner', 'edge', 'center', etc.)

    Returns:
        plotly figure
    """
    fig = go.Figure()

    # Add experiment number column
    design_with_exp = design.copy()
    design_with_exp['experiment_number'] = range(1, len(design) + 1)

    # Prepare color mapping
    if color_var == 'experiment_number':
        # Color by experiment number (continuous scale)
        fig.add_trace(go.Scatter3d(
            x=design[x_var],
            y=design[y_var],
            z=design[z_var],
            mode='markers',
            marker=dict(
                size=8,
                color=design_with_exp['experiment_number'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Exp. #"),
                line=dict(color='black', width=1)
            ),
            text=[f"Exp {i}" for i in design_with_exp['experiment_number']],
            hovertemplate='<b>%{text}</b><br>' +
                         f'{x_var}: %{{x}}<br>{y_var}: %{{y}}<br>{z_var}: %{{z}}<extra></extra>',
            name='Design Points'
        ))

    elif color_var == 'point_type' and point_types is not None:
        # Color by point type
        point_type_colors = {
            'corner': 'red',
            'edge': 'orange',
            'center': 'green',
            'axial': 'blue',
            'interior': 'gray'
        }

        unique_types = sorted(set(point_types))
        for pt_type in unique_types:
            mask = [pt == pt_type for pt in point_types]
            subset = design[mask]
            exp_nums = design_with_exp[mask]['experiment_number']

            fig.add_trace(go.Scatter3d(
                x=subset[x_var],
                y=subset[y_var],
                z=subset[z_var],
                mode='markers',
                marker=dict(
                    size=8,
                    color=point_type_colors.get(pt_type, 'gray'),
                    line=dict(color='black', width=1)
                ),
                name=pt_type.capitalize(),
                text=[f"Exp {i} ({pt_type})" for i in exp_nums],
                hovertemplate='<b>%{text}</b><br>' +
                             f'{x_var}: %{{x}}<br>{y_var}: %{{y}}<br>{z_var}: %{{z}}<extra></extra>'
            ))

    elif color_var and color_var in design.columns:
        # Color by selected variable
        unique_vals = sorted(design[color_var].unique())

        for val in unique_vals:
            subset = design[design[color_var] == val]
            exp_nums = design_with_exp[design[color_var] == val]['experiment_number']

            fig.add_trace(go.Scatter3d(
                x=subset[x_var],
                y=subset[y_var],
                z=subset[z_var],
                mode='markers',
                marker=dict(size=8, line=dict(color='black', width=1)),
                name=f"{color_var}={val}",
                text=[f"Exp {i}" for i in exp_nums],
                hovertemplate='<b>%{text}</b><br>' +
                             f'{x_var}: %{{x}}<br>{y_var}: %{{y}}<br>{z_var}: %{{z}}<extra></extra>'
            ))

    else:
        # No specific coloring - single color
        fig.add_trace(go.Scatter3d(
            x=design[x_var],
            y=design[y_var],
            z=design[z_var],
            mode='markers',
            marker=dict(
                size=8,
                color='blue',
                line=dict(color='black', width=1)
            ),
            text=[f"Exp {i}" for i in design_with_exp['experiment_number']],
            hovertemplate='<b>%{text}</b><br>' +
                         f'{x_var}: %{{x}}<br>{y_var}: %{{y}}<br>{z_var}: %{{z}}<extra></extra>',
            name='Design Points'
        ))

    # Update layout for 3D
    fig.update_layout(
        title=f"3D Design Space: {x_var} × {y_var} × {z_var}",
        scene=dict(
            xaxis_title=x_var,
            yaxis_title=y_var,
            zaxis_title=z_var,
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.3)
            )
        ),
        height=700,
        showlegend=True
    )

    return fig


def validate_for_design_type(design_type, variables_info):
    """
    Validate and auto-adjust variables for design type

    Returns: (is_valid, error_msg, constraints, adjusted_vars)
    """
    import math
    import copy

    k = len(variables_info)
    adjusted_vars = copy.deepcopy(variables_info)
    adjustments_made = []

    if "All Combinations" in design_type:
        # FLEXIBLE: any type, any levels
        n_comb = 1
        for var in adjusted_vars:
            if var['type'] == 'Quantitative':
                n_comb *= var['n_levels']
            else:
                n_comb *= len(var['categories'])

        return (True, None, {
            'constraint': 'No constraints',
            'n_experiments': n_comb,
            'coding': 'Flexible levels, dummy coding for qualitative',
            'description': 'Any variable types, any number of levels',
            'adjustments': []
        }, adjusted_vars)

    elif "Full Factorial 2^k" in design_type:
        # Auto-adjust to 2 levels (min, max)
        for var in adjusted_vars:
            if var['type'] == 'Quantitative':
                if var['n_levels'] != 2:
                    adjustments_made.append(f"{var['name']}: {var['n_levels']} levels → 2 levels (min/max)")
                    var['n_levels'] = 2  # FORCE 2
                    # Keep only 2 levels for manual mode
                    if var.get('levels_mode') == 'manual' and 'levels' in var:
                        if len(var['levels']) > 2:
                            var['levels'] = [var['levels'][0], var['levels'][-1]]  # first and last
            else:  # Qualitative
                if len(var['categories']) != 2:
                    adjustments_made.append(f"{var['name']}: {len(var['categories'])} categories → 2 categories")
                    # Keep first 2 categories
                    var['categories'] = var['categories'][:2]

        return (True, None, {
            'constraint': 'Auto-adjusted to 2 levels (min/max extremes)',
            'n_experiments': 2 ** k,
            'coding': 'All variables coded as [-1, +1]',
            'description': f'2^{k} = {2**k} experiments',
            'adjustments': adjustments_made
        }, adjusted_vars)

    elif "2^4-1" in design_type or "2^5-1" in design_type:
        # Fractional Factorial - Auto-adjust to 2 levels
        ff_name = "2^4-1" if "2^4-1" in design_type else "2^5-1"
        max_factors = 4 if "2^4-1" in design_type else 5
        n_runs = 8 if "2^4-1" in design_type else 16
        resolution = "IV" if "2^4-1" in design_type else "V"

        # Check factor limit
        if k > max_factors:
            return (False,
                f"❌ **{ff_name}** supports maximum {max_factors} factors, but {k} requested",
                None, None)

        # Auto-adjust to 2 levels
        for var in adjusted_vars:
            if var['type'] == 'Quantitative':
                if var['n_levels'] != 2:
                    adjustments_made.append(f"{var['name']}: {var['n_levels']} levels → 2 levels (min/max)")
                    var['n_levels'] = 2
                    if var.get('levels_mode') == 'manual' and 'levels' in var:
                        if len(var['levels']) > 2:
                            var['levels'] = [var['levels'][0], var['levels'][-1]]
            else:
                if len(var['categories']) != 2:
                    adjustments_made.append(f"{var['name']}: {len(var['categories'])} categories → 2 categories")
                    var['categories'] = var['categories'][:2]

        return (True, None, {
            'constraint': f'Auto-adjusted to 2 levels (Fractional Factorial {ff_name})',
            'n_experiments': n_runs,
            'coding': 'All variables coded as [-1, +1]',
            'description': f'{ff_name}: {n_runs} experiments, Resolution {resolution} (efficient screening)',
            'adjustments': adjustments_made
        }, adjusted_vars)

    elif "Plackett-Burmann" in design_type:
        # Auto-adjust to 2 levels
        for var in adjusted_vars:
            if var['type'] == 'Quantitative':
                if var['n_levels'] != 2:
                    adjustments_made.append(f"{var['name']}: {var['n_levels']} levels → 2 levels (min/max)")
                    var['n_levels'] = 2  # FORCE 2
                    # Keep only 2 levels for manual mode
                    if var.get('levels_mode') == 'manual' and 'levels' in var:
                        if len(var['levels']) > 2:
                            var['levels'] = [var['levels'][0], var['levels'][-1]]
            else:
                if len(var['categories']) != 2:
                    adjustments_made.append(f"{var['name']}: {len(var['categories'])} categories → 2 categories")
                    var['categories'] = var['categories'][:2]

        n_pb = math.ceil((k + 1) / 4) * 4
        return (True, None, {
            'constraint': 'Auto-adjusted to 2 levels (min/max extremes)',
            'n_experiments': n_pb,
            'coding': 'All variables coded as [-1, +1]',
            'description': f'N = {n_pb} experiments (screening design)',
            'adjustments': adjustments_made
        }, adjusted_vars)

    elif "Central Composite" in design_type:
        # MUST be quantitative + 3 levels
        for var in adjusted_vars:
            if var['type'] != 'Quantitative':
                return (False,
                    f"❌ **{var['name']}**: Central Composite requires QUANTITATIVE variables only",
                    None, None)
            if var['n_levels'] != 3:
                adjustments_made.append(f"{var['name']}: {var['n_levels']} levels → 3 levels")
                var['n_levels'] = 3  # FORCE 3

        n_factorial = 2 ** k
        n_axial = 2 * k
        n_center = max(3, k)
        n_total = n_factorial + n_axial + n_center
        return (True, None, {
            'constraint': 'Auto-adjusted to 3 levels [-1, 0, +1]',
            'n_experiments': n_total,
            'coding': 'All quantitative at [-1, 0, +1]',
            'description': f'CCD: {n_total} experiments (response surface)',
            'adjustments': adjustments_made
        }, adjusted_vars)

    return (False, "Unknown design type", None, None)


def apply_dummy_coding(design_df, variables_info, return_both=False):
    """
    Apply dummy coding for qualitative variables with > 2 levels

    For qualitative with k categories: create k-1 dummy columns
    Reference category coded as 0 in all dummies
    User-selected reference level is used (stored in variables_info)

    Args:
        design_df: DataFrame with real values
        variables_info: list of variable configuration dicts
        return_both: if True, returns (real+dummy, dummy_only), else just dummy_only

    Returns:
        If return_both=False: DataFrame with dummies expanded (original categorical column removed)
        If return_both=True: tuple of (real_with_dummies, dummies_only)
    """
    design_dummy = design_df.copy()
    dummy_info = []  # Track which dummies were created

    for var_info in variables_info:
        if var_info['type'] == 'Qualitative':
            n_categories = len(var_info['categories'])

            if n_categories > 2:
                # Create k-1 dummy variables using user-selected reference
                var_name = var_info['name']
                categories = var_info['categories']
                reference_idx = var_info.get('reference_level', 0)

                # Get reference category
                reference_cat = categories[reference_idx]

                # Create dummy columns manually to control reference level
                for i, category in enumerate(categories):
                    if i != reference_idx:  # Skip reference level
                        dummy_col_name = f"{var_name}_{category}"
                        # 1 if matches this category, 0 otherwise
                        design_dummy[dummy_col_name] = (design_df[var_name] == category).astype(int)
                        dummy_info.append({
                            'variable': var_name,
                            'dummy_column': dummy_col_name,
                            'category': category,
                            'reference': reference_cat
                        })

                # Store dummy coding info in design metadata
                if not hasattr(design_dummy, '_dummy_coding_info'):
                    design_dummy._dummy_coding_info = []
                design_dummy._dummy_coding_info.append({
                    'variable': var_name,
                    'categories': categories,
                    'reference': reference_cat,
                    'reference_idx': reference_idx,
                    'n_dummies': n_categories - 1
                })

    if return_both:
        # Return both: design with real categories + dummies, and just the dummies
        design_with_dummies = design_dummy.copy()  # Keep categorical columns
        design_only_dummies = design_dummy.copy()

        # In dummy-only version, drop original categorical columns
        for var_info in variables_info:
            if var_info['type'] == 'Qualitative' and len(var_info['categories']) > 2:
                if var_info['name'] in design_only_dummies.columns:
                    design_only_dummies = design_only_dummies.drop(columns=[var_info['name']])

        return design_with_dummies, design_only_dummies, dummy_info
    else:
        # Original behavior: remove categorical columns, keep only dummies
        for var_info in variables_info:
            if var_info['type'] == 'Qualitative' and len(var_info['categories']) > 2:
                if var_info['name'] in design_dummy.columns:
                    design_dummy = design_dummy.drop(columns=[var_info['name']])

        return design_dummy


def show_candidate_points_ui():
    """
    Comprehensive standalone design generator UI
    No data or model required - completely independent

    Features:
    - Multiple design types (Full Factorial, Plackett-Burman, CCD)
    - Quantitative and Qualitative variables
    - Automatic coding ([-1, +1] or dummy variables)
    - Constraint builder
    - 2D/3D interactive visualization with rotation and zoom
    - Point classification (corner, edge, center, axial)
    - Design space coverage analysis
    - CSV/Excel export (real and coded matrices)
    """
    st.markdown("## 🎯 Standalone Design Generator")
    st.markdown("*Generate DoE matrices without existing data or models*")

    st.info("""
    **Comprehensive Design Generator** - Create experimental designs from scratch:
    - **Full Factorial**: All combinations of factor levels
    - **Plackett-Burman**: Efficient screening designs (two-level)
    - **Central Composite**: Response surface methodology
    - **Custom Constraints**: Exclude specific combinations
    """)

    # ===== STEP 1: Variable Configuration =====
    st.markdown("### 📝 Step 1: Variable Configuration")

    n_variables = st.number_input(
        "Number of variables:",
        min_value=2,
        max_value=10,
        value=3,
        help="Number of factors in the experimental design"
    )

    variables_info = []

    st.markdown("#### Configure each variable:")

    # Use tabs instead of expanders to avoid nesting issues
    tab_labels = [f"Variable {i+1}" for i in range(n_variables)]
    var_tabs = st.tabs(tab_labels)

    for i, tab in enumerate(var_tabs):
        with tab:
            col1, col2 = st.columns(2)

            with col1:
                var_name = st.text_input(
                    "Variable name:",
                    value=f"X{i+1}",
                    key=f"var_name_{i}",
                    help="Name for this variable"
                )

            with col2:
                var_type = st.selectbox(
                    "Variable type:",
                    ["Quantitative", "Qualitative"],
                    key=f"var_type_{i}",
                    help="Quantitative: numeric ranges (min-max) | Qualitative: categories (A,B,C). Type will auto-switch if manual input doesn't match."
                )

            if var_type == "Quantitative":
                # Add toggle for auto-generate vs manual definition
                st.markdown("**Level Definition:**")
                levels_mode = st.radio(
                    "Select mode:",
                    ["Auto-generate (min-max-levels)", "Define manually (comma-separated)"],
                    key=f"levels_mode_{i}",
                    horizontal=True,
                    help="Choose how to define variable levels"
                )

                if levels_mode == "Auto-generate (min-max-levels)":
                    # Original auto-generate mode
                    col_q1, col_q2, col_q3 = st.columns(3)
                    with col_q1:
                        min_val = st.number_input(
                            "Min value:",
                            value=0.0,
                            key=f"var_min_{i}",
                            help="Minimum value (coded to -1)"
                        )
                    with col_q2:
                        max_val = st.number_input(
                            "Max value:",
                            value=10.0,
                            key=f"var_max_{i}",
                            help="Maximum value (coded to +1)"
                        )
                    with col_q3:
                        n_levels = st.number_input(
                            "Number of levels:",
                            min_value=2,
                            max_value=10,
                            value=3,
                            key=f"var_nlevels_{i}",
                            help="How many levels to test"
                        )

                    variables_info.append({
                        'name': var_name,
                        'type': 'Quantitative',
                        'levels_mode': 'auto',
                        'min': min_val,
                        'max': max_val,
                        'n_levels': n_levels
                    })

                else:  # Define manually
                    # Manual level definition mode
                    levels_input = st.text_input(
                        "Enter levels (comma-separated):",
                        value="20, 40, 60",
                        key=f"var_levels_manual_{i}",
                        help="Enter exact values separated by commas (numeric: 20, 40, 60 | categorical: A,B,C)"
                    )

                    # Parse and validate input - AUTO-DETECT numeric vs non-numeric
                    levels_list, is_numeric = parse_levels(levels_input)

                    # AUTO-SWITCH Variable Type if needed
                    if levels_list and not is_numeric:
                        # Non-numeric input → must be Qualitative
                        if var_type == "Quantitative":
                            st.warning(f"⚠️ Non-numeric levels detected. Auto-switching to **Qualitative**")
                            var_type = "Qualitative"

                        # Show categories info
                        st.info(f"✅ Categories: {' | '.join([str(x) for x in levels_list])}")

                        # Store as qualitative
                        # Standardize to A, B, C, D, ... format
                        std_labels = [chr(65 + j) for j in range(len(levels_list))]  # A=65 in ASCII
                        mapping_text = " | ".join([f"{levels_list[j]} → {std_labels[j]}" for j in range(len(levels_list))])
                        st.caption(f"**Standardized labels:** {mapping_text}")

                        # Reference level selection (for dummy coding with >2 levels)
                        reference_level_idx = 0  # default
                        if len(levels_list) > 2:
                            st.markdown("**Dummy Coding Setup (k-1 encoding):**")
                            reference_level_idx = st.radio(
                                "Select reference level (base = all zeros):",
                                options=range(len(levels_list)),
                                format_func=lambda idx: f"{std_labels[idx]} ({levels_list[idx]})",
                                key=f"var_reference_manual_{i}",
                                horizontal=True,
                                help="The reference level will be coded as 0 in all dummy variables"
                            )
                            st.caption(f"📌 **{std_labels[reference_level_idx]}** will be the reference: coded as (0, 0, ..., 0) in dummy variables")

                        variables_info.append({
                            'name': var_name,
                            'type': 'Qualitative',
                            'categories': std_labels,
                            'original_categories': [str(x) for x in levels_list],
                            'reference_level': reference_level_idx
                        })

                    elif levels_list and is_numeric:
                        # Numeric input - proceed as Quantitative
                        if len(levels_list) < 2:
                            st.warning("⚠️ Please enter at least 2 levels")
                            levels_list = [0.0, 10.0]  # Fallback

                        # Calculate min/max from manual levels
                        min_val = min(levels_list)
                        max_val = max(levels_list)

                        # Display parsed levels
                        st.caption(f"Parsed levels: {levels_list} (min={min_val}, max={max_val})")

                        variables_info.append({
                            'name': var_name,
                            'type': 'Quantitative',
                            'levels_mode': 'manual',
                            'levels': levels_list,
                            'min': min_val,
                            'max': max_val,
                            'n_levels': len(levels_list)
                        })

                    else:
                        # Invalid or empty input - use fallback
                        st.warning("⚠️ Invalid input. Using default values.")
                        variables_info.append({
                            'name': var_name,
                            'type': 'Quantitative',
                            'levels_mode': 'auto',
                            'min': 0.0,
                            'max': 10.0,
                            'n_levels': 3
                        })

            else:  # Qualitative
                categories_input = st.text_input(
                    "Categories (comma-separated):",
                    value="Low,Medium,High",
                    key=f"var_categories_{i}",
                    help="List of categories (e.g., Low,High or A,B,C)"
                )

                # Parse user input categories
                user_categories = [cat.strip() for cat in categories_input.split(',')]
                n_categories = len(user_categories)

                # Auto-rename to A, B, C, D, ... for standardization
                std_labels = [chr(65 + j) for j in range(n_categories)]  # A=65 in ASCII

                # Show mapping info
                mapping_text = " | ".join([f"{user_categories[j]} → {std_labels[j]}" for j in range(n_categories)])
                st.caption(f"**Standardized labels:** {mapping_text}")

                # Reference level selection (for dummy coding with >2 levels)
                reference_level_idx = 0  # default
                if n_categories > 2:
                    st.markdown("**Dummy Coding Setup (k-1 encoding):**")
                    reference_level_idx = st.radio(
                        "Select reference level (base = all zeros):",
                        options=range(n_categories),
                        format_func=lambda idx: f"{std_labels[idx]} ({user_categories[idx]})",
                        key=f"var_reference_{i}",
                        horizontal=True,
                        help="The reference level will be coded as 0 in all dummy variables"
                    )
                    st.caption(f"📌 **{std_labels[reference_level_idx]}** will be the reference: coded as (0, 0, ..., 0) in dummy variables")

                variables_info.append({
                    'name': var_name,
                    'type': 'Qualitative',
                    'categories': std_labels,  # Use standardized A, B, C, D
                    'original_categories': user_categories,  # Keep original for mapping
                    'reference_level': reference_level_idx  # Store reference index
                })

    # ===== STEP 2: Design Type Selection =====
    st.markdown("---")
    st.markdown("### 🎨 Step 2: Design Type")

    design_type = st.selectbox(
        "Select design type:",
        [
            "All Combinations",
            "Full Factorial 2^k",
            "Fractional Factorial 2^4-1",
            "Fractional Factorial 2^5-1",
            "Plackett-Burmann (screening)",
            "Central Composite Design (Face Centered)"
        ],
        help="Choose the type of experimental design"
    )

    # Validate and show auto-adjustments
    if variables_info:
        is_valid, error_msg, constraints, adjusted_vars = validate_for_design_type(design_type, variables_info)

        # Show constraints and auto-adjustments
        if constraints:
            with st.expander("📋 Design Requirements", expanded=False):
                st.write(f"**Description:** {constraints['description']}")
                st.write(f"**Constraint:** {constraints['constraint']}")
                st.write(f"**Experiments:** {constraints['n_experiments']}")
                st.write(f"**Coding:** {constraints['coding']}")

        if error_msg:
            st.error(error_msg)
            st.stop()

        # Show auto-adjustments if any
        if constraints and constraints.get('adjustments'):
            st.warning("⚙️ **Auto-adjusted to meet design requirements:**")
            for adjustment in constraints['adjustments']:
                st.write(f"  • {adjustment}")

            if "2^k" in design_type or "Plackett-Burmann" in design_type:
                st.info("""
                **2-Level Design:** Using MIN and MAX values only
                - Quantitative: min/max extremes coded as [-1, +1]
                - Qualitative: first 2 categories coded as [-1, +1]
                """)
        else:
            # No adjustments needed
            st.success(f"✅ Configuration valid: {constraints['n_experiments']} experiments will be generated")

        # Update variables_info with adjustments
        variables_info = adjusted_vars

    # Replicates option (for Plackett-Burmann, FF, and other designs)
    n_replicates = 0
    if ("Plackett-Burmann" in design_type or "Central Composite" in design_type or
        "2^4-1" in design_type or "2^5-1" in design_type):
        st.markdown("#### Design Options")
        n_replicates = st.number_input(
            "Number of additional replicates:",
            min_value=0,
            max_value=5,
            value=0,
            help="Repeat the entire design matrix (0 = no replicates, 1 = design repeated twice, etc.)"
        )
        if n_replicates > 0:
            st.info(f"Design will be repeated {n_replicates + 1} times (1 original + {n_replicates} replicates)")

    # ===== STEP 3: Constraint Builder (Optional) =====
    st.markdown("---")
    st.markdown("### 🚫 Step 3: Constraints (Optional)")

    use_constraints = st.checkbox(
        "Add constraints to exclude certain combinations",
        help="Filter out invalid or unwanted experimental points"
    )

    constraints = []
    if use_constraints:
        st.info("Define constraints to exclude points. Examples: X1 > 5, X1 + X2 < 10")
        n_constraints = st.number_input("Number of constraints:", min_value=1, max_value=5, value=1)

        for j in range(n_constraints):
            constraint_text = st.text_input(
                f"Constraint {j+1}:",
                key=f"constraint_{j}",
                help="Use variable names. Example: X1 > 0 & X2 < 5"
            )
            if constraint_text:
                # Store as text for later evaluation
                constraints.append(constraint_text)

    # ===== STEP 4: Generate Design =====
    st.markdown("---")
    if st.button("🚀 Generate Design Matrix", type="primary"):
        try:
            # Build design based on type
            if "2^4-1" in design_type or "2^5-1" in design_type:
                # Fractional Factorial
                import re
                match = re.search(r'(2\^[45]-1)', design_type)
                if match:
                    ff_name = match.group(1)
                    design_coded = generate_fractional_factorial(ff_name, n_variables, n_replicates)
                else:
                    st.error("❌ Could not parse FF design name")
                    design_coded = None

                if design_coded is None:
                    st.stop()

                # Decode to real values (same logic as PB)
                design_real = pd.DataFrame()

                for i, var_info in enumerate(variables_info):
                    var_name = var_info['name']
                    col_name = f"X{i+1}"
                    coded_values = design_coded[col_name].values

                    if var_info['type'] == 'Quantitative':
                        # Decode coded (-1, +1) to real values
                        real_values = decode_quantitative_variable(
                            coded_values,
                            var_info['min'],
                            var_info['max']
                        )
                        design_real[var_name] = real_values
                    else:
                        # For qualitative: map -1 to first, +1 to last category
                        categories = var_info['categories']
                        qual_values = pd.Series(coded_values).map({
                            -1: categories[0],
                            1: categories[-1]
                        }).values
                        design_real[var_name] = qual_values

            elif "Plackett-Burmann" in design_type:
                # Generate coded PB design
                design_coded = generate_plackett_burman(n_variables, n_replicates)

                # Decode to real values (no mixed columns)
                design_real = pd.DataFrame()

                for i, var_info in enumerate(variables_info):
                    var_name = var_info['name']
                    col_name = f"X{i+1}"  # NEW: PB function returns X1, X2, ... not Factor_1, Factor_2, ...
                    coded_values = design_coded[col_name].values

                    if var_info['type'] == 'Quantitative':
                        # Decode coded (-1, +1) to real values
                        real_values = decode_quantitative_variable(
                            coded_values,
                            var_info['min'],
                            var_info['max']
                        )
                        design_real[var_name] = real_values
                    else:
                        # For qualitative: map -1 to first, +1 to last category
                        categories = var_info['categories']
                        qual_values = pd.Series(coded_values).map({
                            -1: categories[0],
                            1: categories[-1]
                        }).values
                        design_real[var_name] = qual_values

            elif "Central Composite" in design_type:
                # Generate coded CCD
                design_coded = generate_central_composite_design(n_variables)

                # Decode to real values
                design_real = design_coded.copy()
                for i, var_info in enumerate(variables_info):
                    var_name = var_info['name']
                    col_name = f"X{i+1}"
                    if var_info['type'] == 'Quantitative':
                        design_real[var_name] = decode_quantitative_variable(
                            design_coded[col_name].values,
                            var_info['min'],
                            var_info['max']
                        )

                design_real = design_real[[var_info['name'] for var_info in variables_info]]

            else:  # Full Factorial: All Combinations or 2^k
                levels_dict = {}
                for var_info in variables_info:
                    if var_info['type'] == 'Quantitative':
                        if "Full Factorial 2^k" in design_type:
                            # Binary only: -1, +1
                            levels_dict[var_info['name']] = [-1, 1]
                        else:
                            # All Combinations: use specified levels
                            if var_info.get('levels_mode') == 'manual' and 'levels' in var_info:
                                levels_dict[var_info['name']] = var_info['levels']
                            else:
                                levels = np.linspace(var_info['min'], var_info['max'], var_info['n_levels'])
                                levels_dict[var_info['name']] = levels.tolist()
                    else:
                        levels_dict[var_info['name']] = var_info['categories']

                design_real = generate_full_factorial(levels_dict)

                # For 2^k: decode back to real values
                if "Full Factorial 2^k" in design_type:
                    for i, var_info in enumerate(variables_info):
                        if var_info['type'] == 'Quantitative':
                            design_real[var_info['name']] = decode_quantitative_variable(
                                design_real[var_info['name']].values,
                                var_info['min'],
                                var_info['max']
                            )

            # Apply constraints if any
            if constraints:
                st.info(f"Applying {len(constraints)} constraint(s)...")
                original_size = len(design_real)

                # Parse and apply constraints
                for constraint in constraints:
                    try:
                        # Replace variable names with column references
                        constraint_eval = constraint
                        for var_info in variables_info:
                            constraint_eval = constraint_eval.replace(
                                var_info['name'],
                                f"design_real['{var_info['name']}']"
                            )
                        # Evaluate constraint
                        mask = eval(constraint_eval)
                        design_real = design_real[mask].reset_index(drop=True)
                    except Exception as e:
                        st.warning(f"Could not apply constraint '{constraint}': {str(e)}")

                filtered_size = len(design_real)
                st.success(f"Filtered: {original_size} → {filtered_size} points ({original_size - filtered_size} excluded)")

            # ===== STANDARDIZE 2-LEVEL VARIABLES TO [-1, +1] =====
            # CRITICAL: All 2-level factors (quantitative or qualitative) must be coded as -1, +1
            # This ensures consistent encoding for MLR analysis
            for var_info in variables_info:
                var_name = var_info['name']

                # Check if variable has exactly 2 levels
                is_two_level = False
                if var_info['type'] == 'Quantitative':
                    # Check if only 2 unique values exist in the column
                    unique_vals = design_real[var_name].nunique()
                    if unique_vals == 2:
                        is_two_level = True
                elif var_info['type'] == 'Qualitative':
                    # Check if exactly 2 categories
                    if len(var_info['categories']) == 2:
                        is_two_level = True

                # Force -1, +1 encoding for all 2-level variables
                if is_two_level:
                    unique_values = sorted(design_real[var_name].unique())

                    # Create mapping: first value → -1, second value → +1
                    if len(unique_values) == 2:
                        mapping = {unique_values[0]: -1, unique_values[1]: +1}
                        design_real[var_name] = design_real[var_name].map(mapping)

                        # Update variables_info to reflect binary coding
                        var_info['is_binary_coded'] = True
                        var_info['binary_mapping'] = {
                            '-1': str(unique_values[0]),
                            '+1': str(unique_values[1])
                        }

            # Store in session state
            st.session_state.candidate_points = design_real

            # ===== DISPLAY RESULTS =====
            st.markdown("---")
            st.success(f"✅ Design generated: {len(design_real)} experiments × {len([v for v in variables_info])} factors")

            # Metrics
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            with col_m1:
                st.metric("Total Points", len(design_real))
            with col_m2:
                st.metric("Variables", len(variables_info))
            with col_m3:
                st.metric("Design Type", design_type.split()[0])
            with col_m4:
                st.metric("Size (KB)", f"{design_real.memory_usage(deep=True).sum() / 1024:.1f}")

            st.markdown("### 📊 Design Matrix")

            # Remove coded columns if they exist (from Plackett-Burmann)
            design_real_clean = design_real.copy()
            coded_cols = [col for col in design_real_clean.columns if col.endswith('_coded')]
            if coded_cols:
                design_real_clean = design_real_clean.drop(columns=coded_cols)

            # TAB: Real vs Real+Dummy vs Coded (THREE TABS)
            # Check if there are qualitative variables with >2 levels (need dummy coding)
            has_qualitative_multicat = any(
                v['type'] == 'Qualitative' and len(v['categories']) > 2
                for v in variables_info
            )

            if has_qualitative_multicat:
                # THREE TABS: Real | Real+Dummy | Coded
                tab_real, tab_dummy, tab_coded = st.tabs([
                    "🔬 Real Values (Natural Scale)",
                    "🔀 Real Values + Dummy Coded",
                    "🔢 Coded Values [-1, +1]"
                ])
            else:
                # TWO TABS: Real | Coded (no dummy coding needed)
                tab_real, tab_coded = st.tabs([
                    "🔬 Real Values (Natural Scale)",
                    "🔢 Coded Values [-1, +1]"
                ])
                tab_dummy = None  # Not needed

            # ===== TAB 1: Real Values (Natural Scale) =====
            with tab_real:
                st.dataframe(design_real_clean, use_container_width=True)

                col_exp1, col_exp2, col_exp3 = st.columns(3)
                with col_exp1:
                    csv_real = design_real_clean.to_csv(index=False)
                    st.download_button(
                        "📥 Download Real (CSV)",
                        csv_real,
                        f"doe_{design_type.replace(' ', '_').lower()}_real_n{len(design_real_clean)}.csv",
                        "text/csv"
                    )

            # ===== TAB 2: Real Values + Dummy Coded (if applicable) =====
            if tab_dummy is not None:
                with tab_dummy:
                    # Apply dummy coding with return_both=True
                    try:
                        design_with_dummies, design_only_dummies, dummy_info = apply_dummy_coding(
                            design_real_clean,
                            variables_info,
                            return_both=True
                        )

                        st.info("**Dummy Coding Applied:** Qualitative variables removed, replaced with k-1 dummy variables")

                        # Show dummy coding scheme in expander
                        with st.expander("📘 Dummy Coding Scheme Explanation"):
                            st.markdown("### How Dummy Coding Works (k-1 encoding)")
                            st.markdown("""
                            For a qualitative variable with **k categories**, we create **k-1 dummy variables**.
                            One category is chosen as the **reference level** (coded as 0 in all dummies).
                            **Original qualitative columns are REMOVED and replaced with dummy columns.**
                            """)

                            # Show scheme for each qualitative variable
                            for var_info in variables_info:
                                if var_info['type'] == 'Qualitative' and len(var_info['categories']) > 2:
                                    var_name = var_info['name']
                                    categories = var_info['categories']
                                    orig_cats = var_info.get('original_categories', categories)
                                    ref_idx = var_info.get('reference_level', 0)
                                    ref_cat = categories[ref_idx]
                                    orig_ref = orig_cats[ref_idx]

                                    st.markdown(f"**Variable: {var_name}**")
                                    st.write(f"- Original categories: {', '.join(orig_cats)}")
                                    st.write(f"- Standardized labels: {', '.join(categories)}")
                                    st.write(f"- Reference level: **{ref_cat}** ({orig_ref}) → coded as (0, 0, ..., 0)")

                                    # Show example coding table
                                    coding_table = []
                                    for i, cat in enumerate(categories):
                                        row = {'Category': f"{cat} ({orig_cats[i]})"}
                                        for j, dummy_cat in enumerate(categories):
                                            if j != ref_idx:
                                                row[f"{var_name}_{dummy_cat}"] = 1 if i == j else 0
                                        coding_table.append(row)

                                    coding_df = pd.DataFrame(coding_table)
                                    st.dataframe(coding_df, use_container_width=True, hide_index=True)
                                    st.caption(f"Reference level **{ref_cat}** has all zeros")

                        # Display design with ONLY dummy variables (original qualitative columns removed)
                        st.dataframe(design_only_dummies, use_container_width=True)

                        col_dummy1, col_dummy2 = st.columns(2)
                        with col_dummy1:
                            st.metric("Total Columns", len(design_only_dummies.columns))
                        with col_dummy2:
                            n_dummy_cols = len(dummy_info)
                            st.metric("Dummy Variables", n_dummy_cols)

                        # Download button
                        csv_dummy = design_only_dummies.to_csv(index=False)
                        st.download_button(
                            "📥 Download Real+Dummy (CSV)",
                            csv_dummy,
                            f"doe_{design_type.replace(' ', '_').lower()}_dummy_n{len(design_only_dummies)}.csv",
                            "text/csv"
                        )

                    except Exception as e:
                        st.error(f"Error applying dummy coding: {str(e)}")
                        st.dataframe(design_real_clean, use_container_width=True)

            # ===== TAB 3 (or 2): Coded Values [-1, +1] =====
            with tab_coded:
                # Create coded matrix for display
                design_coded_display = pd.DataFrame()
                for i, var_info in enumerate(variables_info):
                    var_name = var_info['name']

                    # Check if this is a binary-coded variable (2 levels)
                    if var_info.get('is_binary_coded', False):
                        # Binary variables: already coded as -1, +1 in design_real_clean
                        design_coded_display[var_name] = design_real_clean[var_name].astype(int)
                    elif var_info['type'] == 'Quantitative':
                        # Multi-level quantitative: encode real → coded [-1, ..., +1]
                        real_col = design_real_clean[var_name].values
                        coded_col = 2 * (real_col - var_info['min']) / (var_info['max'] - var_info['min']) - 1
                        design_coded_display[var_name] = np.round(coded_col, 3)
                    else:
                        # Multi-level categorical: show as is (categorical labels)
                        design_coded_display[var_name] = design_real_clean[var_name]

                # Add Replicate_ID if present
                if 'Replicate_ID' in design_real_clean.columns:
                    design_coded_display['Replicate_ID'] = design_real_clean['Replicate_ID']

                # Show binary coding info if applicable
                binary_vars = [v for v in variables_info if v.get('is_binary_coded', False)]
                if binary_vars:
                    st.info("**Binary Coding Applied:** All 2-level variables standardized to [-1, +1]")
                    with st.expander("📘 Binary Coding Mappings"):
                        for var_info in binary_vars:
                            mapping = var_info.get('binary_mapping', {})
                            st.write(f"**{var_info['name']}:** {mapping['-1']} → -1 | {mapping['+1']} → +1")

                st.dataframe(design_coded_display, use_container_width=True)

                col_exp1, col_exp2, col_exp3 = st.columns(3)
                with col_exp1:
                    csv_coded = design_coded_display.to_csv(index=False)
                    st.download_button(
                        "📥 Download Coded (CSV)",
                        csv_coded,
                        f"doe_{design_type.replace(' ', '_').lower()}_coded_n{len(design_coded_display)}.csv",
                        "text/csv"
                    )

            # ===== 2D/3D VISUALIZATION =====
            st.markdown("---")
            st.markdown("### 📊 Design Visualization")

            if len(variables_info) >= 2:
                # Initialize variables at top level to prevent NameError
                x_var = None
                y_var = None
                z_var = None
                color_var = None
                color_var_3d = None
                point_types = None

                # View mode selector
                view_options = ["2D", "3D"] if len(variables_info) >= 3 else ["2D"]
                view_mode = st.radio("View:", view_options, horizontal=True, key="design_view_mode")

                # ===== 2D MODE =====
                if view_mode == "2D":
                    col_viz1, col_viz2, col_viz3 = st.columns(3)
                    with col_viz1:
                        x_var = st.selectbox("X-axis:", [v['name'] for v in variables_info], index=0, key="viz_x_2d")
                    with col_viz2:
                        y_var = st.selectbox("Y-axis:", [v['name'] for v in variables_info], index=min(1, len(variables_info)-1), key="viz_y_2d")
                    with col_viz3:
                        color_options_2d = ["None"] + [v['name'] for v in variables_info]
                        color_var = st.selectbox("Color by:", color_options_2d, key="viz_color_2d")
                        if color_var == "None":
                            color_var = None

                # ===== 3D MODE =====
                elif view_mode == "3D":
                    if len(variables_info) >= 3:
                        st.info("**3D Mode**: Interactive plot with rotation and zoom. Drag to rotate, scroll to zoom.")

                        col_3d1, col_3d2, col_3d3, col_3d4 = st.columns(4)
                        with col_3d1:
                            x_var = st.selectbox("X-axis:", [v['name'] for v in variables_info], index=0, key="viz_x_3d")
                        with col_3d2:
                            y_var = st.selectbox("Y-axis:", [v['name'] for v in variables_info], index=min(1, len(variables_info)-1), key="viz_y_3d")
                        with col_3d3:
                            z_var = st.selectbox("Z-axis:", [v['name'] for v in variables_info], index=min(2, len(variables_info)-1), key="viz_z_3d")
                        with col_3d4:
                            color_options_3d = ["None", "Experiment Number", "Point Type"] + [v['name'] for v in variables_info]
                            color_choice_3d = st.selectbox("Color by:", color_options_3d, key="viz_color_3d")

                            if color_choice_3d == "Experiment Number":
                                color_var_3d = 'experiment_number'
                            elif color_choice_3d == "Point Type":
                                color_var_3d = 'point_type'
                                # Classify design points
                                try:
                                    point_types = classify_design_points(design_real_clean)
                                    type_counts = pd.Series(point_types).value_counts()
                                    st.caption("Point types: " + " | ".join([f"{k}: {v}" for k, v in type_counts.items()]))
                                except:
                                    st.warning("Could not classify point types")
                                    color_var_3d = None
                            elif color_choice_3d != "None":
                                color_var_3d = color_choice_3d
                            else:
                                color_var_3d = None

                    else:
                        # Fallback to 2D when <3 variables
                        st.warning("⚠️ 3D visualization requires at least 3 variables. Falling back to 2D...")
                        view_mode = "2D"  # Force 2D rendering

                        col_viz1, col_viz2, col_viz3 = st.columns(3)
                        with col_viz1:
                            x_var = st.selectbox("X-axis:", [v['name'] for v in variables_info], index=0, key="viz_x_fallback")
                        with col_viz2:
                            y_var = st.selectbox("Y-axis:", [v['name'] for v in variables_info], index=min(1, len(variables_info)-1), key="viz_y_fallback")
                        with col_viz3:
                            color_options_fallback = ["None"] + [v['name'] for v in variables_info]
                            color_var = st.selectbox("Color by:", color_options_fallback, key="viz_color_fallback")
                            if color_var == "None":
                                color_var = None

                # ===== PLOT GENERATION =====
                if x_var and y_var:  # Only plot if axes defined
                    try:
                        if view_mode == "2D":
                            fig = plot_design_2d(design_real_clean, x_var, y_var, color_var)
                            st.plotly_chart(fig, use_container_width=True, key="design_2d_plot")

                            # Multiple pairwise plots if many variables
                            if len(variables_info) > 2:
                                with st.expander("Show all pairwise plots"):
                                    n_vars = len(variables_info)
                                    for i in range(min(3, n_vars-1)):
                                        for j in range(i+1, min(i+3, n_vars)):
                                            fig_pair = plot_design_2d(
                                                design_real_clean,
                                                variables_info[i]['name'],
                                                variables_info[j]['name']
                                            )
                                            st.plotly_chart(fig_pair, use_container_width=True, key=f"design_pair_{i}_{j}")

                        elif view_mode == "3D" and z_var:
                            fig_3d = create_3d_scatter(design_real_clean, x_var, y_var, z_var, color_var_3d, point_types)
                            st.plotly_chart(fig_3d, use_container_width=True, key="design_3d_plot")

                            # Design space coverage info
                            with st.expander("📐 Design Space Coverage Analysis"):
                                st.markdown("**3D Design Space Metrics:**")

                                # Calculate bounding box volume
                                x_range = design_real_clean[x_var].max() - design_real_clean[x_var].min()
                                y_range = design_real_clean[y_var].max() - design_real_clean[y_var].min()
                                z_range = design_real_clean[z_var].max() - design_real_clean[z_var].min()

                                col_cov1, col_cov2, col_cov3 = st.columns(3)
                                with col_cov1:
                                    st.metric(f"{x_var} Range", f"{x_range:.3f}")
                                with col_cov2:
                                    st.metric(f"{y_var} Range", f"{y_range:.3f}")
                                with col_cov3:
                                    st.metric(f"{z_var} Range", f"{z_range:.3f}")

                                # Point distribution
                                if point_types:
                                    st.markdown("**Point Distribution:**")
                                    type_counts = pd.Series(point_types).value_counts()
                                    for pt_type, count in type_counts.items():
                                        st.write(f"- **{pt_type.capitalize()}**: {count} points ({count/len(point_types)*100:.1f}%)")

                    except Exception as e:
                        st.error(f"❌ Visualization error: {str(e)}")
                else:
                    st.warning("⚠️ Select X and Y axes to display visualization")

            # ===== EXPORT OPTIONS =====
            st.markdown("---")
            st.markdown("### 💾 Export Design")

            col_exp1, col_exp2, col_exp3 = st.columns(3)

            with col_exp1:
                # CSV export
                csv_data = design_real.to_csv(index=False)
                st.download_button(
                    "📥 Download CSV",
                    csv_data,
                    f"doe_design_{design_type.split()[0].lower()}.csv",
                    "text/csv",
                    help="Download design matrix as CSV file"
                )

            with col_exp2:
                # Excel export with openpyxl
                from io import BytesIO
                excel_buffer = BytesIO()
                try:
                    with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                        design_real.to_excel(writer, sheet_name='Design', index=False)
                    excel_buffer.seek(0)
                    st.download_button(
                        "📥 Download Excel",
                        excel_buffer.getvalue(),
                        f"doe_design_{design_type.split()[0].lower()}.xlsx",
                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        help="Download design matrix as Excel file"
                    )
                except:
                    st.info("Excel export requires openpyxl")

            with col_exp3:
                # Send to workspace using workspace utilities
                if st.button("📤 Send to Workspace"):
                    # Import workspace utilities
                    try:
                        import sys
                        import os
                        # Add parent directory to path if needed
                        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                        if parent_dir not in sys.path:
                            sys.path.insert(0, parent_dir)

                        from workspace_utils import save_to_workspace

                        # Generate clean dataset name
                        # Replace spaces and special chars for clean workspace names
                        design_type_clean = design_type.replace(" ", "_").replace("(", "").replace(")", "").lower()
                        dataset_name = f"doe_{design_type_clean}_n{len(design_real)}"

                        # Prepare detailed metadata
                        metadata = {
                            'design_type': design_type,
                            'n_factors': len(variables_info),
                            'n_experiments': len(design_real),
                            'factors': [v['name'] for v in variables_info]
                        }

                        # Save to workspace
                        success, message = save_to_workspace(
                            dataset=design_real,
                            dataset_name=dataset_name,
                            design_type=design_type,
                            metadata=metadata
                        )

                        if success:
                            st.success(message)
                            st.info(f"""
                            **Access your design:**
                            1. Go to **Data Handling** (sidebar)
                            2. Scroll to **📊 Workspace Selector**
                            3. Find **"{dataset_name}"** in the list
                            4. Click to load → Ready for MLR analysis

                            **Design Summary:**
                            - Type: {design_type}
                            - Experiments: {len(design_real)} runs
                            - Factors: {len(variables_info)}
                            """)

                            # Show quick preview
                            with st.expander("📋 Preview (first 5 rows)"):
                                st.dataframe(design_real.head(), use_container_width=True)
                                st.caption(f"{len(design_real)} rows × {len(design_real.columns)} columns")
                        else:
                            st.error(message)

                    except ImportError as e:
                        st.error(f"❌ Could not import workspace utilities: {e}")
                        st.info("Falling back to basic session state storage...")

                        # Fallback: basic session state storage
                        dataset_name = f"DoE_{design_type.split()[0]}_{len(design_real)}pts"
                        st.session_state.current_data = design_real.copy()
                        st.session_state.current_dataset = dataset_name
                        st.success(f"✅ Design saved as: {dataset_name}")

            # ===== CODED MATRIX EXPORT =====
            st.markdown("---")
            st.markdown("#### 🔢 Coded Matrix (for MLR)")
            st.info("Quantitative → [-1, +1] coding | Qualitative → Dummy variables (n-1)")

            try:
                # Use encode_full_matrix from data_handling
                design_coded = encode_full_matrix(design_real, variables_info)

                # Show coded matrix preview
                with st.expander("👁️ Preview Coded Matrix"):
                    st.dataframe(design_coded.head(10), use_container_width=True)

                    col_code1, col_code2, col_code3 = st.columns(3)
                    with col_code1:
                        st.metric("Coded Samples", len(design_coded))
                    with col_code2:
                        st.metric("Coded Variables", len(design_coded.columns))
                    with col_code3:
                        # Check for qualitative dummy expansion
                        n_original = len(variables_info)
                        n_coded = len(design_coded.columns)
                        if n_coded > n_original:
                            st.metric("Dummy Expansion", f"+{n_coded - n_original}")
                        else:
                            st.metric("Variables", n_coded)

                # Download coded matrix
                col_coded1, col_coded2 = st.columns(2)

                with col_coded1:
                    csv_coded = design_coded.to_csv(index=False)
                    st.download_button(
                        "📥 Download Coded CSV",
                        csv_coded,
                        f"doe_design_{design_type.split()[0].lower()}_coded.csv",
                        "text/csv",
                        help="Download coded matrix for MLR analysis"
                    )

                with col_coded2:
                    # Coded Excel
                    from io import BytesIO
                    excel_coded_buffer = BytesIO()
                    try:
                        with pd.ExcelWriter(excel_coded_buffer, engine='openpyxl') as writer:
                            design_coded.to_excel(writer, sheet_name='Coded_Design', index=False)
                            design_real.to_excel(writer, sheet_name='Real_Values', index=False)
                        excel_coded_buffer.seek(0)
                        st.download_button(
                            "📥 Download Coded Excel",
                            excel_coded_buffer.getvalue(),
                            f"doe_design_{design_type.split()[0].lower()}_coded.xlsx",
                            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            help="Excel with both coded and real values"
                        )
                    except:
                        st.info("Excel export requires openpyxl")

            except Exception as e:
                st.warning(f"Could not generate coded matrix: {str(e)}")

            # ===== DESIGN INFORMATION =====
            with st.expander("📋 Design Information"):
                st.write("**Design Type:**", design_type)
                st.write(f"**Number of factors:** {len(variables_info)}")
                st.write("**Variable details:**")
                for var_info in variables_info:
                    if var_info['type'] == 'Quantitative':
                        st.write(f"  - {var_info['name']}: Quantitative [{var_info['min']}, {var_info['max']}], {var_info['n_levels']} levels")
                    else:
                        st.write(f"  - {var_info['name']}: Qualitative {var_info['categories']}")
                st.write(f"**Total experimental points:** {len(design_real)}")

                if constraints:
                    st.write("**Constraints applied:**")
                    for constraint in constraints:
                        st.write(f"  - {constraint}")

        except Exception as e:
            st.error(f"❌ Error generating design: {str(e)}")
            import traceback
            with st.expander("🐛 Error details"):
                st.code(traceback.format_exc())


# Keep original function for backward compatibility
def generate_candidate_points(variables_config):
    """
    Generate candidate points for experimental design
    Equivalent to expand.grid() in R

    Args:
        variables_config: dict with variable names as keys and levels as values

    Returns:
        DataFrame with all combinations of factor levels
    """
    levels_dict = {}

    for var_name, levels in variables_config.items():
        if isinstance(levels, str):
            try:
                levels_dict[var_name] = [float(x.strip()) for x in levels.split(',')]
            except ValueError:
                levels_dict[var_name] = [x.strip() for x in levels.split(',')]
        else:
            levels_dict[var_name] = levels

    return generate_full_factorial(levels_dict)
