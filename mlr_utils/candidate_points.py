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
from data_handling import encode_quantitative, encode_full_matrix


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


def generate_plackett_burman(k, n_replicates=0):
    """
    Generate Plackett-Burman screening design
    Two-level fractional factorial for efficient screening of k factors

    Standard PB sizes: N = 8, 12, 16, 20, 24
    Construction: N-1 cyclic shifts of generator row + 1 fold-over row (all -1)

    Args:
        k: number of factors (variables)
        n_replicates: number of additional replicates (0 = no replicates, 1 = design repeated twice, etc.)

    Returns:
        DataFrame with coded design matrix [-1, +1]
        Shape: (N × (1 + n_replicates), k) where N is calculated from k
    """
    import math

    # Step 1: Calculate N from k
    # N must be multiple of 4 and >= k+1
    N = math.ceil((k + 1) / 4) * 4

    # Step 2: Validate k and get base design (generator row)
    # These are the standard Plackett-Burman generators from literature
    PB_GENERATORS = {
        8: [1, -1, -1, 1, -1, 1, 1],  # length 7
        12: [1, 1, -1, 1, 1, 1, -1, -1, -1, 1, -1],  # length 11
        16: [1, 1, 1, 1, -1, 1, -1, 1, 1, -1, -1, 1, -1, -1, -1],  # length 15
        20: [1, 1, -1, -1, 1, 1, 1, 1, -1, 1, -1, 1, -1, -1, -1, -1, 1, 1, -1],  # length 19
        24: [1, 1, 1, 1, 1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, -1, 1, -1, -1, -1, -1]  # length 23
    }

    # Validate N
    if N not in PB_GENERATORS:
        st.error(f"❌ Cannot generate Plackett-Burman design for k={k} factors (requires N={N})")
        st.warning(f"Supported sizes: N ∈ {list(PB_GENERATORS.keys())} (for k up to N-1 factors)")
        # Fallback: use full factorial
        design_dict = {f"X{i+1}": [-1, 1] for i in range(k)}
        return generate_full_factorial(design_dict)

    if k > N - 1:
        st.error(f"❌ k={k} factors exceeds maximum for N={N} design (max k={N-1})")
        st.warning(f"Reduce number of factors or use a larger design")
        # Fallback
        design_dict = {f"X{i+1}": [-1, 1] for i in range(k)}
        return generate_full_factorial(design_dict)

    # Step 3: Get generator row for this N
    first_row = PB_GENERATORS[N]

    # Step 4: Generate design matrix using cyclic shifts
    matrix = []

    # Rows 1 to N-1: cyclic right shifts of first_row
    for i in range(N - 1):
        # Cyclic right shift by i positions
        shifted_row = first_row[-(i):] + first_row[:-(i)] if i > 0 else first_row.copy()
        matrix.append(shifted_row)

    # Row N (last row): all -1 (fold-over point)
    matrix.append([-1] * len(first_row))

    # Step 5: Select only first k columns (we have N-1 columns from generator)
    # Convert to numpy for easier slicing
    matrix_np = np.array(matrix)
    design_matrix = matrix_np[:, :k]

    # Step 6: Apply replicates if requested
    if n_replicates > 0:
        # Repeat the design n_replicates times
        replicated_matrix = design_matrix.copy()
        for _ in range(n_replicates):
            replicated_matrix = np.vstack([replicated_matrix, design_matrix])
        design_matrix = replicated_matrix

    # Step 7: Create DataFrame
    var_names = [f"X{i+1}" for i in range(k)]
    design = pd.DataFrame(design_matrix, columns=var_names)

    # Add design info
    st.info(f"""
    **Plackett-Burman Design Generated:**
    - Factors (k): {k}
    - Design size (N): {N}
    - Total runs: {len(design)} = {N} × (1 + {n_replicates} replicates)
    - Generator: {first_row[:10]}{'...' if len(first_row) > 10 else ''}
    """)

    return design


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
                    help="Quantitative=numeric (continuous), Qualitative=categorical"
                )

            if var_type == "Quantitative":
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
                    'min': min_val,
                    'max': max_val,
                    'n_levels': n_levels
                })

            else:  # Qualitative
                categories_input = st.text_input(
                    "Categories (comma-separated):",
                    value="Low,Medium,High",
                    key=f"var_categories_{i}",
                    help="List of categories (e.g., Low,High or A,B,C)"
                )

                categories = [cat.strip() for cat in categories_input.split(',')]

                variables_info.append({
                    'name': var_name,
                    'type': 'Qualitative',
                    'categories': categories
                })

    # ===== STEP 2: Design Type Selection =====
    st.markdown("---")
    st.markdown("### 🎨 Step 2: Design Type")

    design_type = st.selectbox(
        "Select design type:",
        [
            "Full Factorial (all combinations)",
            "Plackett-Burman (screening)",
            "Central Composite Design (response surface)",
            "Custom (specify levels)"
        ],
        help="Choose the type of experimental design"
    )

    # Replicates option (for Plackett-Burman and other designs)
    n_replicates = 0
    if "Plackett-Burman" in design_type or "Central Composite" in design_type:
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
            if "Plackett-Burman" in design_type:
                # Generate coded PB design with new signature (k, n_replicates)
                design_coded = generate_plackett_burman(n_variables, n_replicates)

                # Decode to real values for quantitative variables
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
                    else:
                        # For qualitative: map -1 to first category, +1 to second
                        categories = var_info['categories']
                        design_real[var_name] = design_coded[col_name].map({
                            -1: categories[0],
                            1: categories[-1]
                        })

                # Remove original coded columns
                design_real = design_real[[var_info['name'] for var_info in variables_info]]

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

            else:  # Full Factorial or Custom
                # Build levels dict
                levels_dict = {}
                for var_info in variables_info:
                    if var_info['type'] == 'Quantitative':
                        levels = np.linspace(var_info['min'], var_info['max'], var_info['n_levels'])
                        levels_dict[var_info['name']] = levels.tolist()
                    else:
                        levels_dict[var_info['name']] = var_info['categories']

                design_real = generate_full_factorial(levels_dict)

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

            # Store in session state
            st.session_state.candidate_points = design_real

            # ===== DISPLAY RESULTS =====
            st.markdown("---")
            st.markdown("### ✅ Generated Design Matrix")

            # Summary metrics
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            with col_m1:
                st.metric("Total Points", len(design_real))
            with col_m2:
                st.metric("Variables", len(variables_info))
            with col_m3:
                st.metric("Design Type", design_type.split()[0])
            with col_m4:
                memory_kb = design_real.memory_usage(deep=True).sum() / 1024
                st.metric("Size (KB)", f"{memory_kb:.1f}")

            # Design table
            st.dataframe(design_real, use_container_width=True, height=400)

            # ===== 2D/3D VISUALIZATION =====
            st.markdown("---")
            st.markdown("### 📊 Design Visualization")

            if len(variables_info) >= 2:
                # View mode radio (stable - no reset)
                view_options = ["2D", "3D"] if len(variables_info) >= 3 else ["2D"]
                view_mode = st.radio("View:", view_options, horizontal=True, key="design_view_mode")

                # ===== AXIS SELECTORS (SHARED) =====
                # Always show selectors based on view mode to prevent resets
                if view_mode == "2D":
                    # 2D axis selectors
                    col_viz1, col_viz2, col_viz3 = st.columns(3)
                    with col_viz1:
                        x_var = st.selectbox("X-axis:", [v['name'] for v in variables_info], index=0, key="viz_x")
                    with col_viz2:
                        y_var = st.selectbox("Y-axis:", [v['name'] for v in variables_info], index=min(1, len(variables_info)-1), key="viz_y")
                    with col_viz3:
                        color_var = st.selectbox("Color by:", ["None"] + [v['name'] for v in variables_info], key="viz_color")

                    if color_var == "None":
                        color_var = None

                else:  # 3D mode
                    if len(variables_info) >= 3:
                        st.info("**3D Mode**: Interactive plot with rotation and zoom. Drag to rotate, scroll to zoom.")

                        # 3D axis selectors
                        col_3d1, col_3d2, col_3d3, col_3d4 = st.columns(4)
                        with col_3d1:
                            x_var = st.selectbox("X-axis:", [v['name'] for v in variables_info], index=0, key="viz_x")
                        with col_3d2:
                            y_var = st.selectbox("Y-axis:", [v['name'] for v in variables_info], index=min(1, len(variables_info)-1), key="viz_y")
                        with col_3d3:
                            z_var = st.selectbox("Z-axis:", [v['name'] for v in variables_info], index=min(2, len(variables_info)-1), key="viz_z")
                        with col_3d4:
                            color_options = ["None", "Experiment Number", "Point Type"] + [v['name'] for v in variables_info]
                            color_choice = st.selectbox("Color by:", color_options, key="viz_color")

                        # Map color choice to internal format
                        color_var_3d = None
                        point_types = None

                        if color_choice == "Experiment Number":
                            color_var_3d = 'experiment_number'
                        elif color_choice == "Point Type":
                            color_var_3d = 'point_type'
                            # Classify design points
                            try:
                                point_types = classify_design_points(design_real)
                                # Show point type distribution
                                type_counts = pd.Series(point_types).value_counts()
                                st.caption(f"Point types: " + " | ".join([f"{k}: {v}" for k, v in type_counts.items()]))
                            except:
                                st.warning("Could not classify point types")
                                color_var_3d = None
                        elif color_choice != "None":
                            color_var_3d = color_choice
                    else:
                        st.warning("⚠️ 3D visualization requires at least 3 variables. Showing 2D view instead.")
                        view_mode = "2D"  # Force 2D mode

                        # Fallback to 2D selectors
                        col_viz1, col_viz2, col_viz3 = st.columns(3)
                        with col_viz1:
                            x_var = st.selectbox("X-axis:", [v['name'] for v in variables_info], index=0, key="viz_x")
                        with col_viz2:
                            y_var = st.selectbox("Y-axis:", [v['name'] for v in variables_info], index=min(1, len(variables_info)-1), key="viz_y")
                        with col_viz3:
                            color_var = st.selectbox("Color by:", ["None"] + [v['name'] for v in variables_info], key="viz_color")

                        if color_var == "None":
                            color_var = None

                # ===== CREATE PLOTS =====
                # Generate the selected plot
                if view_mode == "2D":
                    fig = plot_design_2d(design_real, x_var, y_var, color_var)
                    st.plotly_chart(fig, use_container_width=True, key="design_2d_plot")

                    # Multiple pairwise plots if many variables
                    if len(variables_info) > 2:
                        with st.expander("Show all pairwise plots"):
                            n_vars = len(variables_info)
                            for i in range(min(3, n_vars-1)):
                                for j in range(i+1, min(i+3, n_vars)):
                                    fig_pair = plot_design_2d(
                                        design_real,
                                        variables_info[i]['name'],
                                        variables_info[j]['name']
                                    )
                                    st.plotly_chart(fig_pair, use_container_width=True, key=f"design_pair_{i}_{j}")

                else:  # 3D mode
                    fig_3d = create_3d_scatter(design_real, x_var, y_var, z_var, color_var_3d, point_types)
                    st.plotly_chart(fig_3d, use_container_width=True, key="design_3d_plot")

                    # Design space coverage info
                    with st.expander("📐 Design Space Coverage Analysis"):
                        st.markdown("**3D Design Space Metrics:**")

                        # Calculate bounding box volume
                        x_range = design_real[x_var].max() - design_real[x_var].min()
                        y_range = design_real[y_var].max() - design_real[y_var].min()
                        z_range = design_real[z_var].max() - design_real[z_var].min()

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
                # Copy to data handling
                if st.button("📤 Send to Data Handling"):
                    import datetime

                    # Generate dataset name
                    dataset_name = f"DoE_Design_{design_type.split()[0]}"

                    # Store in multiple session state locations for compatibility
                    st.session_state.current_data = design_real.copy()
                    st.session_state.current_dataset = dataset_name
                    st.session_state['design_matrix'] = design_real.copy()

                    # Initialize transformation_history if it doesn't exist
                    if 'transformation_history' not in st.session_state:
                        st.session_state.transformation_history = {}

                    # Add to transformation history for workspace integration
                    st.session_state.transformation_history[dataset_name] = {
                        'data': design_real.copy(),
                        'transform': f'{design_type}',
                        'params': {
                            'n_variables': len(variables_info),
                            'n_points': len(design_real),
                            'design_type': design_type
                        },
                        'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'original_dataset': 'DoE Generator'
                    }

                    # Success message with navigation guidance
                    st.success("✅ Design matrix saved successfully!")
                    st.info(f"""
                    **Next Steps:**
                    1. Navigate to **Data Handling** page (use sidebar)
                    2. Select **"{dataset_name}"** from workspace
                    3. Your design matrix is ready to use with {len(design_real)} experimental points

                    The design is now available in:
                    - Data Handling workspace
                    - MLR/DoE analysis tabs
                    """)

                    # Show quick preview of what was saved
                    with st.expander("📋 Preview of saved design"):
                        st.dataframe(design_real.head(), use_container_width=True)
                        st.caption(f"Design: {len(design_real)} rows × {len(design_real.columns)} columns")

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
