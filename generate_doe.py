"""
Generate DoE Module
Provides experimental design generation and D-optimal optimization.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Optional

# Import utilities
from generatedoe_utils.doe_designs import (
    generate_full_factorial,
    generate_plackett_burman,
    generate_central_composite,
    generate_custom_design
)
from generatedoe_utils.candidate_generator import (
    create_candidate_matrix,
    apply_constraints,
    validate_candidate_matrix
)
from generatedoe_utils.doptimal_algorithm import (
    doptimal_design,
    calculate_inflation_factors,
    format_doptimal_results
)
from generatedoe_utils.doptimal_by_addition import (
    doptimal_by_addition,
    format_addition_results,
    extract_added_experiments,
    calculate_model_matrix
)
from workspace_utils import save_to_workspace, get_workspace_datasets

# Import column transforms - try DoE-specific coding first, fallback to auto_encode
try:
    from transforms.column_transforms import column_doe_coding, column_auto_encode, detect_column_type
    USE_DOE_CODING = True
except ImportError:
    from transforms.column_transforms import column_auto_encode
    USE_DOE_CODING = False

try:
    from color_utils import get_color_scheme
    USE_COLOR_UTILS = True
except ImportError:
    USE_COLOR_UTILS = False


def initialize_session_state():
    """Initialize session state variables for Generate DoE."""
    if 'generated_design' not in st.session_state:
        st.session_state.generated_design = None
    if 'candidate_matrix' not in st.session_state:
        st.session_state.candidate_matrix = None
    if 'doptimal_results' not in st.session_state:
        st.session_state.doptimal_results = None
    if 'variables_config' not in st.session_state:
        st.session_state.variables_config = {}
    # New session state variables for encoding workflow
    if 'doe_candidate_matrix_raw' not in st.session_state:
        st.session_state.doe_candidate_matrix_raw = None
    if 'doe_candidate_matrix_encoded' not in st.session_state:
        st.session_state.doe_candidate_matrix_encoded = None
    if 'doe_encoding_metadata' not in st.session_state:
        st.session_state.doe_encoding_metadata = None
    if 'doe_generated_matrix' not in st.session_state:
        st.session_state.doe_generated_matrix = None
    if 'doe_candidate_matrix_constrained' not in st.session_state:
        st.session_state.doe_candidate_matrix_constrained = None
    if 'doe_constraints_list' not in st.session_state:
        st.session_state.doe_constraints_list = []
    if 'doe_generated_design_tab3' not in st.session_state:
        st.session_state.doe_generated_design_tab3 = None
    # Model specification for Tab 2
    if 'doe_design_matrix' not in st.session_state:
        st.session_state.doe_design_matrix = None
    if 'doe_model_spec' not in st.session_state:
        st.session_state.doe_model_spec = {}
    if 'doe_model_terms' not in st.session_state:
        st.session_state.doe_model_terms = []
    if 'doe_n_coefficients' not in st.session_state:
        st.session_state.doe_n_coefficients = 0
    if 'doe_selected_interactions' not in st.session_state:
        st.session_state.doe_selected_interactions = {}
    if 'doe_selected_quadratic' not in st.session_state:
        st.session_state.doe_selected_quadratic = {}
    if 'doe_include_intercept' not in st.session_state:
        st.session_state.doe_include_intercept = True
    # Tab 4 - D-Optimal by Addition session state
    if 'tab4_candidate_matrix' not in st.session_state:
        st.session_state.tab4_candidate_matrix = None
    if 'tab4_results' not in st.session_state:
        st.session_state.tab4_results = None


def render_variable_configuration(n_variables: int) -> Dict:
    """
    Render UI for configuring variables.

    Args:
        n_variables: Number of variables to configure

    Returns:
        Dictionary with variable configurations
    """
    variables_config = {}

    st.markdown("### 📋 Variable Configuration")

    for i in range(n_variables):
        with st.expander(f"**Variable {i+1} Configuration**", expanded=(i < 2)):
            col1, col2 = st.columns([2, 1])

            with col1:
                var_name = st.text_input(
                    "Variable Name",
                    value=f"Var_{i+1}",
                    key=f"var_name_{i}"
                )

            with col2:
                var_type = st.radio(
                    "Type",
                    ["Quantitative", "Categorical"],
                    key=f"var_type_{i}",
                    horizontal=True
                )

            if var_type == "Quantitative":
                col1, col2 = st.columns(2)
                with col1:
                    min_val = st.number_input(
                        "Minimum Value",
                        value=0.0,
                        key=f"var_min_{i}"
                    )
                with col2:
                    max_val = st.number_input(
                        "Maximum Value",
                        value=100.0,
                        key=f"var_max_{i}"
                    )

                level_mode = st.radio(
                    "Level Definition",
                    ["Auto-generate", "Define manually"],
                    key=f"level_mode_{i}",
                    horizontal=True
                )

                if level_mode == "Auto-generate":
                    n_levels = st.slider(
                        "Number of Levels",
                        min_value=2,
                        max_value=10,
                        value=3,
                        key=f"n_levels_{i}"
                    )
                    levels = list(np.linspace(min_val, max_val, n_levels))
                else:
                    levels_str = st.text_area(
                        "Levels (comma-separated)",
                        value=f"{min_val}, {(min_val+max_val)/2}, {max_val}",
                        key=f"levels_str_{i}"
                    )
                    try:
                        levels = [float(x.strip()) for x in levels_str.split(',')]
                    except:
                        st.error("Invalid levels format. Using default.")
                        levels = [min_val, max_val]

                # Validation
                if min_val >= max_val:
                    st.error("❌ Minimum must be < Maximum")
                elif len(levels) < 2:
                    st.error("❌ At least 2 levels required")
                else:
                    st.success(f"✓ {len(levels)} levels: {levels}")

                variables_config[var_name] = {
                    'min': min_val,
                    'max': max_val,
                    'levels': sorted(levels)
                }

            else:  # Categorical
                levels_str = st.text_area(
                    "Categories (comma-separated)",
                    value="Low, Medium, High",
                    key=f"cat_levels_{i}"
                )
                levels = [x.strip() for x in levels_str.split(',')]

                if len(levels) < 2:
                    st.error("❌ At least 2 categories required")
                else:
                    st.success(f"✓ {len(levels)} categories: {levels}")

                variables_config[var_name] = {
                    'min': 0,
                    'max': len(levels) - 1,
                    'levels': list(range(len(levels))),
                    'labels': levels
                }

    return variables_config


def render_design_type_selection() -> tuple:
    """
    Render UI for design type selection with ALL 5 options and info cards.

    Returns:
        Tuple (design_type, alpha_type)
    """
    st.markdown("### 🎯 Design Type Selection")

    # Design type selector - ALL 5 OPTIONS
    design_type = st.radio(
        "Select Design Type:",
        [
            "Full Factorial 2^k",
            "Fractional Factorial 2^(k-1) Resolution IV",
            "Fractional Factorial 2^(k-2) Resolution V",
            "Plackett-Burmann (screening)",
            "Central Composite Design (Face Centered)"
        ],
        key="doe_design_type_selector",
        horizontal=False
    )

    st.markdown("---")

    # ═════════════════════════════════════════════════════════════════════════
    # INFO CARDS FOR EACH DESIGN TYPE
    # ═════════════════════════════════════════════════════════════════════════

    alpha_type = None
    n_vars_for_calc = st.session_state.get('doe_n_variables', 3)

    if design_type == "Full Factorial 2^k":
        n_exp = 2 ** n_vars_for_calc

        st.markdown("### 📋 Design Requirements")

        col_d1, col_d2, col_d3 = st.columns(3)
        with col_d1:
            st.metric("Description", "2^k experiments")
        with col_d2:
            st.metric("Experiments", n_exp)
        with col_d3:
            st.metric("Resolution", "Complete")

        st.info(f"""
**Design:** Full Factorial 2^{n_vars_for_calc}

**Auto-adjusted to meet design requirements:**
• All variables: 3+ levels → 2 levels (min/max)

**2-Level Design:** Using MIN and MAX values only
* Quantitative: min/max extremes coded as [-1, +1]
* Qualitative: first 2 categories coded as [-1, +1]
        """)

    elif design_type == "Fractional Factorial 2^(k-1) Resolution IV":
        n_exp = 2 ** (n_vars_for_calc - 1)

        st.markdown("### 📋 Design Requirements")

        col_d1, col_d2, col_d3 = st.columns(3)
        with col_d1:
            st.metric("Description", "2^(k-1) efficient")
        with col_d2:
            st.metric("Experiments", n_exp)
        with col_d3:
            st.metric("Resolution", "IV")

        st.info(f"""
**Design:** Fractional Factorial 2^({n_vars_for_calc}-1): {n_exp} experiments ✅ VERIFIED

**Auto-adjusted to meet design requirements:**
• All variables: 3+ levels → 2 levels (min/max)

**Resolution IV:** Main effects clear, some interactions confounded
* Quantitative: min/max extremes coded as [-1, +1]
* Qualitative: first 2 categories coded as [-1, +1]
* More efficient than full factorial (1/2 experiments)
        """)

    elif design_type == "Fractional Factorial 2^(k-2) Resolution V":
        if n_vars_for_calc < 3:
            st.error("❌ Fractional Factorial 2^(k-2) requires at least 3 variables")
            return None, None

        n_exp = 2 ** (n_vars_for_calc - 2)

        st.markdown("### 📋 Design Requirements")

        col_d1, col_d2, col_d3 = st.columns(3)
        with col_d1:
            st.metric("Description", "2^(k-2) very efficient")
        with col_d2:
            st.metric("Experiments", n_exp)
        with col_d3:
            st.metric("Resolution", "V")

        st.info(f"""
**Design:** Fractional Factorial 2^({n_vars_for_calc}-2): {n_exp} experiments ✅ VERIFIED

**Auto-adjusted to meet design requirements:**
• All variables: 3+ levels → 2 levels (min/max)

**Resolution V:** Main effects and 2-factor interactions clear
* Quantitative: min/max extremes coded as [-1, +1]
* Qualitative: first 2 categories coded as [-1, +1]
* Very efficient (1/4 experiments vs full factorial)
        """)

    elif design_type == "Plackett-Burmann (screening)":
        # Calculate nearest valid PB size
        valid_sizes = [4, 8, 12, 16, 20, 24, 32]
        n_exp = next((s for s in valid_sizes if s >= n_vars_for_calc + 1), 32)

        st.markdown("### 📋 Design Requirements")

        col_d1, col_d2, col_d3 = st.columns(3)
        with col_d1:
            st.metric("Description", "PB screening")
        with col_d2:
            st.metric("Experiments", n_exp)
        with col_d3:
            st.metric("Resolution", "Screening")

        st.info(f"""
**Design:** Plackett-Burmann N={n_exp} ✅ VERIFIED

**Auto-adjusted to meet design requirements:**
• All variables: 3+ levels → 2 levels (min/max)
• Design size: Rounded to valid PB multiple

**Screening Design:** Very rapid initial factor screening
* Quantitative: min/max extremes coded as [-1, +1]
* Qualitative: first 2 categories coded as [-1, +1]
* Orthogonal design (all factors uncorrelated)
        """)

        st.warning("⚠️ Plackett-Burman requires all variables to have exactly 2 levels")

    else:  # Central Composite Design
        n_fac = 2 ** n_vars_for_calc
        n_axial = 2 * n_vars_for_calc
        n_center = 5
        n_exp = n_fac + n_axial + n_center

        st.markdown("### 📋 Design Requirements")

        col_d1, col_d2, col_d3 = st.columns(3)
        with col_d1:
            st.metric("Description", "RSM response surface")
        with col_d2:
            st.metric("Experiments", n_exp)
        with col_d3:
            st.metric("Resolution", "Quadratic")

        st.info(f"""
**Design:** Central Composite Design (Face-Centered): {n_exp} experiments
  • Factorial points: {n_fac}
  • Axial points: {n_axial}
  • Center points: {n_center}

**Auto-adjusted to 3 levels:**
• All variables: 3+ levels → 3 levels (min, center, max)

**Response Surface Methodology (RSM):**
* Can estimate quadratic models
* Provides design curvature information
* Quantitative: min/center/max coded as [-1, 0, +1]
* Qualitative: first 3 categories coded
        """)

        alpha_type = st.selectbox(
            "Alpha Type:",
            ["orthogonal", "rotatable", "face"],
            help="Orthogonal: Minimizes correlation | "
                 "Rotatable: Equal prediction variance | "
                 "Face: Axial points at ±1"
        )

    return design_type, alpha_type


def render_constraints_section() -> List:
    """
    Render UI for adding constraints.

    Returns:
        List of constraint dictionaries
    """
    st.markdown("### 🚫 Constraints (Optional)")

    add_constraints = st.checkbox("Add constraints?", value=False)
    constraints = []

    if add_constraints:
        st.info("Define constraints to exclude certain candidate points")

        constraint_type = st.selectbox(
            "Constraint Type",
            ["exclude_rows", "custom_expression", "range", "comparison"]
        )

        if constraint_type == "exclude_rows":
            indices_str = st.text_input(
                "Row indices to exclude (comma-separated)",
                help="e.g., 1, 5, 12, 18"
            )
            if indices_str:
                try:
                    indices = [int(x.strip()) for x in indices_str.split(',')]
                    constraints.append({'type': 'exclude_rows', 'indices': indices})
                    st.success(f"✓ Will exclude {len(indices)} rows")
                except:
                    st.error("Invalid indices format")

        elif constraint_type == "custom_expression":
            st.warning("Custom expressions require Python knowledge")
            expr_str = st.text_area(
                "Lambda expression",
                value="lambda row: row['Var_1'] < 50",
                help="Use variable names as shown in configuration"
            )

        # Add more constraint types as needed

    return constraints


def generate_design_matrix(variables_config: Dict, design_type: str, alpha_type: Optional[str] = None) -> pd.DataFrame:
    """
    Generate design matrix based on configuration.

    Args:
        variables_config: Variable configurations
        design_type: Type of design
        alpha_type: Alpha type for CCD

    Returns:
        Generated design DataFrame
    """
    if design_type == "Full Factorial 2^k":
        return generate_full_factorial(variables_config)

    elif design_type == "Fractional Factorial 2^(k-1) Resolution IV":
        from generatedoe_utils.doe_designs import generate_fractional_factorial_IV
        return generate_fractional_factorial_IV(variables_config)

    elif design_type == "Fractional Factorial 2^(k-2) Resolution V":
        from generatedoe_utils.doe_designs import generate_fractional_factorial_V
        return generate_fractional_factorial_V(variables_config)

    elif design_type == "Plackett-Burmann (screening)":
        return generate_plackett_burman(variables_config)

    elif design_type == "Central Composite Design (Face Centered)":
        return generate_central_composite(variables_config, alpha_type=alpha_type)

    else:
        raise ValueError(f"Unknown design type: {design_type}")


def tab1_design_generator():
    """
    TAB 1: Generate ALL Combinations of Variables
    Simplified workflow: Configure → Generate → Encode → Apply Constraints → Save
    """
    st.markdown("## 📊 Step 1: Configure Variables")
    st.markdown("Define each variable with REAL values (natural units)")

    col_nvar1, col_nvar2 = st.columns([2, 1])
    with col_nvar1:
        n_variables = st.slider(
            "Number of variables:",
            min_value=1,
            max_value=30,
            value=3,
            key="doe_n_variables"
        )
    with col_nvar2:
        st.markdown("")
        st.info(f"**Variables:** {n_variables}")

    st.markdown("---")

    # ═══════════════════════════════════════════════════════════════════════
    # Configure each variable
    # ═══════════════════════════════════════════════════════════════════════

    variables_config_raw = {}

    for i in range(n_variables):
        with st.expander(f"**Variable {i+1}**", expanded=(i < 2)):

            col_name, col_type = st.columns([2, 1])

            with col_name:
                var_name = st.text_input(
                    "Name:",
                    value=f"X{i+1}",
                    key=f"doe_var_name_{i}"
                )

            with col_type:
                var_type = st.radio(
                    "Type:",
                    ["Quantitative", "Categorical"],
                    key=f"doe_var_type_{i}",
                    horizontal=True
                )

            if var_type == "Quantitative":
                col_min, col_max = st.columns(2)

                with col_min:
                    min_val = st.number_input(
                        "Min:",
                        value=0.0,
                        key=f"doe_min_{i}"
                    )

                with col_max:
                    max_val = st.number_input(
                        "Max:",
                        value=100.0,
                        key=f"doe_max_{i}"
                    )

                step_size = st.number_input(
                    "Step:",
                    value=10.0,
                    min_value=0.01,
                    key=f"doe_step_{i}"
                )

                # Generate levels
                if step_size > 0 and min_val < max_val:
                    levels = list(np.arange(min_val, max_val + step_size/2, step_size))
                    levels = [round(x, 6) for x in levels]
                    st.success(f"✓ {len(levels)} levels: {[f'{x:.4g}' for x in levels]}")
                else:
                    levels = [min_val, max_val]
                    st.warning("⚠️ Using min and max")

                variables_config_raw[var_name] = {
                    'type': 'Quantitative',
                    'min': min_val,
                    'max': max_val,
                    'step': step_size,
                    'levels': levels
                }

            else:  # Categorical
                levels_input = st.text_area(
                    "Levels (comma-separated):",
                    value="A,B,C",
                    height=68,
                    key=f"doe_categories_{i}"
                )

                try:
                    levels = [x.strip() for x in levels_input.split(',') if x.strip()]
                    if len(levels) < 2:
                        st.error("❌ Need at least 2 levels")
                        levels = []
                    else:
                        st.success(f"✓ {len(levels)} levels: {levels}")
                except:
                    st.error("❌ Invalid format")
                    levels = []

                variables_config_raw[var_name] = {
                    'type': 'Categorical',
                    'levels': levels
                }

    st.markdown("---")

    # ═══════════════════════════════════════════════════════════════════════
    # Step 2: Show design statistics
    # ═══════════════════════════════════════════════════════════════════════

    st.markdown("## 📊 Step 2: Design Statistics")

    total_combinations = 1
    for var_name, config in variables_config_raw.items():
        total_combinations *= len(config['levels'])

    st.info(f"""
**All Combinations Full Factorial:**
- Variables: {len(variables_config_raw)}
- Total experiments: **{total_combinations}**
""")

    if total_combinations > 1000:
        st.warning(f"⚠️ **{total_combinations} experiments is VERY LARGE!**")
        st.markdown("Consider: Using **Tab 3** for Plackett-Burman or Central Composite designs, or applying **Constraints** below")

    st.markdown("---")

    # ═══════════════════════════════════════════════════════════════════════
    # Step 3: Generate candidate matrix
    # ═══════════════════════════════════════════════════════════════════════

    st.markdown("## 🔄 Step 3: Generate Matrix")

    if st.button(
        "🔄 GENERATE ALL COMBINATIONS",
        type="secondary",
        use_container_width=True,
        key="doe_gen_combinations"
    ):
        with st.spinner("Generating..."):
            try:
                if not variables_config_raw:
                    st.error("❌ Configure at least one variable")
                    return

                candidate_matrix_raw = create_candidate_matrix(variables_config_raw)
                st.session_state.doe_candidate_matrix_raw = candidate_matrix_raw

                st.success(f"✓ Generated: {candidate_matrix_raw.shape[0]} rows × {candidate_matrix_raw.shape[1]} columns")
                st.dataframe(candidate_matrix_raw.head(10), use_container_width=True)
                st.caption(f"*Showing first 10 of {len(candidate_matrix_raw)} rows*")

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

    st.markdown("---")

    # ═══════════════════════════════════════════════════════════════════════
    # Step 4: Apply constraints (optional)
    # ═══════════════════════════════════════════════════════════════════════

    if st.session_state.get('doe_candidate_matrix_raw') is not None:
        st.markdown("## 🚫 Step 4: Apply Constraints (Optional)")
        st.markdown("Define conditions to exclude experimental regions")

        use_constraints = st.checkbox(
            "Apply constraints?",
            value=False,
            key="doe_use_constraints"
        )

        constraints_list = []

        if use_constraints:
            st.markdown("### Add Constraint Rules")

            # Initialize constraint list in session state
            if 'doe_constraints_list' not in st.session_state:
                st.session_state.doe_constraints_list = []

            # UI for adding constraints
            col_const1, col_const2 = st.columns([2, 1])

            with col_const1:
                constraint_type = st.selectbox(
                    "Type:",
                    ["Range Constraint", "Logical Expression"],
                    key="doe_constraint_type"
                )

            if constraint_type == "Range Constraint":
                col_var, col_min, col_max = st.columns(3)

                var_names = list(variables_config_raw.keys())
                with col_var:
                    selected_var = st.selectbox("Variable:", var_names, key="doe_const_var")

                with col_min:
                    const_min = st.number_input("Min:", key="doe_const_min")

                with col_max:
                    const_max = st.number_input("Max:", value=100.0, key="doe_const_max")

                if st.button("➕ Add Range Constraint", key="doe_add_range"):
                    st.session_state.doe_constraints_list.append({
                        'type': 'range',
                        'variable': selected_var,
                        'min': const_min,
                        'max': const_max
                    })
                    st.success(f"✓ Added: {selected_var} in [{const_min}, {const_max}]")

            else:  # Logical Expression
                expr_help = """
Examples:
- X1 > 50
- X1 + X2 < 100
- X1 * X2 > 1000
- (X1 > 20) and (X2 < 80)
                """
                st.markdown(expr_help)

                expr_str = st.text_input(
                    "Expression (using variable names):",
                    value="",
                    key="doe_const_expr"
                )

                if st.button("➕ Add Expression", key="doe_add_expr"):
                    st.session_state.doe_constraints_list.append({
                        'type': 'expression',
                        'expression': expr_str
                    })
                    st.success(f"✓ Added: {expr_str}")

            # Display constraints list
            if st.session_state.doe_constraints_list:
                st.markdown("### Active Constraints")
                for idx, const in enumerate(st.session_state.doe_constraints_list):
                    col_display, col_remove = st.columns([4, 1])

                    with col_display:
                        if const['type'] == 'range':
                            st.write(f"{idx+1}. **{const['variable']}** ∈ [{const['min']}, {const['max']}]")
                        else:
                            st.write(f"{idx+1}. **{const['expression']}**")

                    with col_remove:
                        if st.button("🗑️", key=f"doe_remove_const_{idx}"):
                            st.session_state.doe_constraints_list.pop(idx)
                            st.rerun()

                constraints_list = st.session_state.doe_constraints_list

        st.markdown("---")

    # ═══════════════════════════════════════════════════════════════════════
    # Step 5: Encode matrix using DoE-specific coding
    # ═══════════════════════════════════════════════════════════════════════

    if st.session_state.get('doe_candidate_matrix_raw') is not None:
        st.markdown("## 🔐 Step 5: Encode Matrix")
        st.markdown("""
Coding transforms real values to DoE standard format:
- **2-level (numeric/categorical):** → [-1, +1]
- **3+ levels (numeric):** → [-1, 0, ..., +1] range
- **3+ levels (categorical):** → One-hot (k-1) encoding
        """)

        if st.button(
            "🔐 ENCODE TO STANDARD DOE FORMAT",
            type="primary",
            use_container_width=True,
            key="doe_encode"
        ):
            with st.spinner("Encoding matrix..."):
                try:
                    from transforms.column_transforms import column_doe_coding

                    candidate_raw = st.session_state.doe_candidate_matrix_raw

                    # Apply constraints BEFORE encoding
                    if 'doe_constraints_list' in st.session_state and st.session_state.doe_constraints_list:
                        constraints_list = st.session_state.doe_constraints_list
                        candidate_raw = apply_constraints(candidate_raw, constraints_list)
                        st.success(f"✓ Constraints applied: {candidate_raw.shape[0]} rows remaining")

                    # Encode using column_doe_coding (specialized for DoE)
                    # col_range = (0, len(candidate_raw.columns)) encodes ALL columns
                    candidate_encoded, encoding_metadata, multiclass_info = column_doe_coding(
                        candidate_raw,
                        col_range=(0, len(candidate_raw.columns))
                    )

                    # Store results
                    st.session_state.doe_candidate_matrix_encoded = candidate_encoded
                    st.session_state.doe_encoding_metadata = encoding_metadata
                    st.session_state.doe_candidate_matrix_constrained = candidate_raw

                    st.success("✓ Encoded successfully!")
                    st.write(f"**Result:** {candidate_encoded.shape[0]} rows × {candidate_encoded.shape[1]} columns")

                    st.rerun()

                except ImportError:
                    st.error("❌ column_doe_coding not found. Using fallback column_auto_encode...")
                    try:
                        from transforms.column_transforms import column_auto_encode

                        candidate_raw = st.session_state.doe_candidate_matrix_raw

                        if 'doe_constraints_list' in st.session_state and st.session_state.doe_constraints_list:
                            constraints_list = st.session_state.doe_constraints_list
                            candidate_raw = apply_constraints(candidate_raw, constraints_list)

                        candidate_encoded, encoding_metadata = column_auto_encode(
                            candidate_raw,
                            col_range=(0, len(candidate_raw.columns)),
                            exclude_cols=[]
                        )

                        st.session_state.doe_candidate_matrix_encoded = candidate_encoded
                        st.session_state.doe_encoding_metadata = encoding_metadata
                        st.session_state.doe_candidate_matrix_constrained = candidate_raw

                        st.warning("⚠️ Using fallback encoder (limited functionality)")
                        st.rerun()
                    except Exception as e2:
                        st.error(f"❌ Fallback also failed: {str(e2)}")

                except Exception as e:
                    st.error(f"❌ Encoding error: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())

        st.markdown("---")

    # ═══════════════════════════════════════════════════════════════════════
    # Step 6: Display results and next steps
    # ═══════════════════════════════════════════════════════════════════════

    if st.session_state.get('doe_candidate_matrix_encoded') is not None:
        st.markdown("## ✅ Results")

        candidate_encoded = st.session_state.doe_candidate_matrix_encoded
        encoding_meta = st.session_state.doe_encoding_metadata

        st.markdown("### Coded Matrix")
        st.dataframe(candidate_encoded.head(10), use_container_width=True)
        st.caption(f"*{len(candidate_encoded)} rows total*")

        st.markdown("### Encoding Rules")
        for var_name, meta in encoding_meta.items():
            with st.expander(f"**{var_name}:** {meta['encoding_rule']}", expanded=False):
                if 'original_unique' in meta:
                    st.write(f"Original values: {meta['original_unique']}")
                if 'encoding_map' in meta:
                    st.json(meta['encoding_map'])

        st.markdown("---")

        st.markdown("## 🚀 Next Steps")

        col_action1, col_action2, col_action3 = st.columns(3)

        with col_action1:
            if st.button("💾 Save Full Design", key="doe_save_full"):
                design_name = f"FullCombo_{candidate_encoded.shape[0]}exp"
                success, message = save_to_workspace(
                    candidate_encoded,
                    design_name,
                    metadata={
                        'design_type': "All Combinations",
                        'n_experiments': candidate_encoded.shape[0],
                        'n_variables': candidate_encoded.shape[1],
                        'encoding_metadata': encoding_meta
                    }
                )
                if success:
                    st.success(message)
                else:
                    st.error(message)

        with col_action2:
            if st.button("🎯 Optimize (Tab 2)", key="doe_goto_tab2"):
                st.info("👉 Switch to **Tab 2: D-Optimal** to optimize")

        with col_action3:
            csv_data = candidate_encoded.to_csv(index=False)
            st.download_button(
                "📥 Download CSV",
                csv_data,
                "doe_combinations_encoded.csv",
                "text/csv",
                key="doe_download"
            )


def tab2_doptimal_design():
    """
    TAB 2: D-Optimal Design - Complete R CAT Matching Implementation

    WORKFLOW:
    1. Load candidate matrix (encoded + raw)
    2. Model configuration (intercept + higher-order)
    3. Select model terms (quadratic + interactions)
    4. Build design matrix
    5. D-optimal parameters (min, max, step, trials)
    6. Run optimization
    7. View plots (log_M and VIF)
    8. Manual selection (no automation)
    9. Solution details (inflation factors per coefficient)
    10. Subset extraction (real + coded paired)
    11. Export
    """
    st.markdown("## 🎯 D-Optimal Design Optimization")
    st.markdown("Select optimal subset from candidate matrix using D-optimal algorithm")

    # ═══════════════════════════════════════════════════════════════════════
    # SELECTOR: Choose candidate matrix source
    # ═══════════════════════════════════════════════════════════════════════

    st.markdown("### Step 0: 📦 Select Candidate Matrix Source")

    col_source1, col_source2 = st.columns(2)

    with col_source1:
        matrix_source = st.radio(
            "Candidate matrix source:",
            ["Tab 1 (Generated)", "Workspace (Data Handling)"],
            key="dopt_matrix_source",
            horizontal=False
        )

    # ═══════════════════════════════════════════════════════════════════════
    # LOAD MATRIX FROM SELECTED SOURCE
    # ═══════════════════════════════════════════════════════════════════════

    candidate_encoded = None
    candidate_raw = None

    if matrix_source == "Tab 1 (Generated)":
        # Option 1: From Tab 1
        with col_source2:
            if 'doe_candidate_matrix_encoded' in st.session_state and \
               st.session_state.doe_candidate_matrix_encoded is not None:

                candidate_encoded = st.session_state.doe_candidate_matrix_encoded
                candidate_raw = st.session_state.get('doe_candidate_matrix_raw', None)

                st.success("✓ Loaded from Tab 1")
                st.metric("Rows", candidate_encoded.shape[0])
                st.metric("Columns", candidate_encoded.shape[1])
            else:
                st.warning("⚠️ No matrix from Tab 1")
                st.info("💡 Go to **Tab 1** to generate candidate matrix first")
                st.markdown("---")
                st.markdown("**OR** select **Workspace** option below to use existing data")
                return

    else:
        # Option 2: From Workspace
        st.markdown("---")
        st.markdown("### Select from Workspace")

        from workspace_utils import get_workspace_datasets

        available_datasets = get_workspace_datasets()

        if len(available_datasets) == 0:
            with col_source2:
                st.error("❌ No datasets in workspace")
            st.info("💡 Go to **Data Handling** to load or create datasets first")
            return

        # Selectbox for workspace dataset
        selected_dataset_name = st.selectbox(
            "Choose dataset:",
            options=list(available_datasets.keys()),
            key="dopt_workspace_dataset"
        )

        if selected_dataset_name:
            candidate_encoded = available_datasets[selected_dataset_name].copy()
            candidate_raw = candidate_encoded.copy()  # Assume already encoded or use as-is

            # Show info
            col_info1, col_info2, col_info3 = st.columns(3)
            with col_info1:
                st.metric("Rows (Candidates)", candidate_encoded.shape[0])
            with col_info2:
                st.metric("Columns (Variables)", candidate_encoded.shape[1])
            with col_info3:
                st.metric("Total Cells", candidate_encoded.shape[0] * candidate_encoded.shape[1])
        else:
            return

    # ═══════════════════════════════════════════════════════════════════════
    # VALIDATION: Ensure we have a matrix
    # ═══════════════════════════════════════════════════════════════════════

    if candidate_encoded is None:
        st.error("❌ Could not load candidate matrix")
        return

    st.markdown("---")

    # ═══════════════════════════════════════════════════════════════════════
    # Step 1: Show loaded matrix preview
    # ═══════════════════════════════════════════════════════════════════════

    st.markdown("### Step 1: Candidate Matrix Preview")

    with st.expander("📋 View first 10 rows", expanded=False):
        st.dataframe(candidate_encoded.head(10), use_container_width=True)
        st.write(f"**Full shape:** {candidate_encoded.shape[0]} rows × {candidate_encoded.shape[1]} columns")

    if candidate_raw is not None:
        with st.expander("📋 Original Values (Real)"):
            st.dataframe(candidate_raw.head(10), use_container_width=True)

    st.markdown("---")

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 2-3: Model Specification (mlr_doe.py style - CONSISTENT!)
    # ═══════════════════════════════════════════════════════════════════════

    st.markdown("## Step 2-3: 🎯 Model Specification")
    st.markdown("**Select model terms to include in the design matrix**")

    st.info("""
D-optimal will optimize the DESIGN MATRIX (with selected terms), not just candidate matrix.
The design matrix will include: Main effects + Interactions + Quadratic + Intercept
    """)

    # ═══════════════════════════════════════════════════════════════════════
    # ANALYSIS: Detect variable types (2-level vs 3+ level categorical)
    # ═══════════════════════════════════════════════════════════════════════

    from transforms.column_transforms import detect_column_type

    var_info = {}
    multiclass_vars = set()  # Variables with 3+ categorical levels (NO interactions)
    binary_vars = set()      # 2-level variables (can have interactions)

    # Only analyze original variables (not encoded dummy columns)
    if candidate_raw is not None:
        for var_name in candidate_raw.columns:
            col_analysis = detect_column_type(candidate_raw[var_name])
            var_info[var_name] = col_analysis

            if col_analysis['dtype_detected'] == 'multiclass_cat':
                multiclass_vars.add(var_name)
            else:
                binary_vars.add(var_name)
    else:
        # If no raw data, treat all encoded columns as binary
        binary_vars = set(candidate_encoded.columns)

    # Show variable analysis if multiclass detected
    if multiclass_vars:
        st.warning(f"""
⚠️ **Categorical Variables with 3+ Levels:**
{', '.join(sorted(multiclass_vars))}

**Note:** Interactions involving these variables are disabled.
(One-hot encoding creates multiple binary columns → direct interactions not applicable)
        """)

    st.markdown("---")

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 1: Basic Model Options
    # ═══════════════════════════════════════════════════════════════════════

    st.markdown("### Model Structure")

    col_opt1, col_opt2 = st.columns(2)

    with col_opt1:
        include_intercept = st.checkbox(
            "✓ Include intercept (constant term)",
            value=True,
            help="Add intercept/constant to model",
            key="dopt_intercept"
        )

    with col_opt2:
        include_higher_terms = st.checkbox(
            "✓ Include higher-order terms",
            value=False,
            help="Add interactions and/or quadratic terms",
            key="dopt_higher_terms"
        )

    st.markdown("---")

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 2: Term Selection (mlr_doe.py STYLE - checkbox in columns)
    # ═══════════════════════════════════════════════════════════════════════

    selected_interactions = {}
    selected_quadratic = {}

    if include_higher_terms:
        st.markdown("### Select Model Terms (✓ = include)")

        var_names = list(candidate_encoded.columns)
        n_vars = len(var_names)

        # ═════════════════════════════════════════════════════════════════
        # QUADRATIC TERMS (3 columns - like mlr_doe.py)
        # ═════════════════════════════════════════════════════════════════

        st.markdown("#### Quadratic Terms:")

        # Detect which variables are binary (2 levels only)
        binary_vars = set()
        for var_name in var_names:
            n_unique = candidate_encoded[var_name].nunique()
            if n_unique == 2:
                binary_vars.add(var_name)

        # Show info about binary variables
        if binary_vars:
            st.info(f"""
⚠️ **Binary Variables Detected:** {', '.join(sorted(binary_vars))}

These variables have only 2 levels, so quadratic terms are disabled.
(Quadratic terms would be constant and provide no information)
            """)

        cols_quad = st.columns(3)
        for i, var_name in enumerate(var_names):
            with cols_quad[i % 3]:
                key_quad = f"dopt_quad_{var_name}"

                if var_name in binary_vars:
                    # Disabled for binary variables
                    st.checkbox(
                        f"☑ {var_name}²",
                        value=False,
                        disabled=True,
                        help="Binary variables cannot have quadratic terms",
                        key=key_quad
                    )
                    selected_quadratic[var_name] = False
                else:
                    # Enabled for multi-level variables
                    is_selected = st.checkbox(
                        f"☑ {var_name}²",
                        value=True,
                        key=key_quad
                    )
                    selected_quadratic[var_name] = is_selected

        st.markdown("")

        # ═════════════════════════════════════════════════════════════════
        # INTERACTION TERMS (3 columns - like mlr_doe.py)
        # ═════════════════════════════════════════════════════════════════

        st.markdown("#### Interaction Terms:")

        # Generate all possible interactions (upper triangle only: i < j)
        interaction_pairs = []
        for i in range(n_vars):
            for j in range(i+1, n_vars):
                interaction_pairs.append((var_names[i], var_names[j]))

        cols_inter = st.columns(3)

        for idx, (var_i, var_j) in enumerate(interaction_pairs):
            with cols_inter[idx % 3]:

                # Check if interaction is ALLOWED
                is_multiclass_involved = (var_i in multiclass_vars) or (var_j in multiclass_vars)

                key_inter = f"dopt_inter_{var_i}_{var_j}"

                if is_multiclass_involved:
                    # DISABLED: Show grayed out with explanation
                    st.checkbox(
                        f"☑ {var_i}×{var_j}",
                        value=False,
                        disabled=True,
                        help="Disabled: One or both variables are multi-level categorical",
                        key=key_inter
                    )
                    selected_interactions[f"{var_i}×{var_j}"] = False
                else:
                    # ENABLED: Allow selection
                    is_selected = st.checkbox(
                        f"☑ {var_i}×{var_j}",
                        value=True,
                        key=key_inter
                    )
                    selected_interactions[f"{var_i}×{var_j}"] = is_selected

        st.markdown("")

        # ═════════════════════════════════════════════════════════════════
        # SUMMARY (like mlr_doe.py)
        # ═════════════════════════════════════════════════════════════════

        st.markdown("#### Summary")

        n_linear = n_vars
        n_interactions_selected = sum(1 for v in selected_interactions.values() if v)
        n_quadratic_selected = sum(1 for v in selected_quadratic.values() if v)
        n_intercept = 1 if include_intercept else 0

        total_coefficients = n_linear + n_interactions_selected + n_quadratic_selected + n_intercept

        col_s1, col_s2, col_s3 = st.columns(3)

        with col_s1:
            st.metric("Linear Terms", n_linear)
        with col_s2:
            st.metric("Interactions", n_interactions_selected)
        with col_s3:
            st.metric("Quadratic Terms", n_quadratic_selected)

        st.markdown(f"**Total Coefficients (incl. intercept): {total_coefficients}**")

    else:
        # NO higher terms selected
        st.info("**Model will include:** Main Effects + Intercept")

        n_linear = len(candidate_encoded.columns)
        n_intercept = 1 if include_intercept else 0
        total_coefficients = n_linear + n_intercept

        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.metric("Linear Terms", n_linear)
        with col_s2:
            st.metric("Total Coefficients (incl. intercept)", total_coefficients)

    st.markdown("---")

    # ═══════════════════════════════════════════════════════════════════════
    # BUILD DESIGN MATRIX
    # ═══════════════════════════════════════════════════════════════════════

    if st.button("🔨 BUILD DESIGN MATRIX FROM MODEL", type="primary", use_container_width=True):
        with st.spinner("Building design matrix..."):
            try:
                # Start with encoded candidate matrix
                X = candidate_encoded.copy()
                model_terms = list(X.columns)  # Main effects in order

                # Add selected interactions
                if include_higher_terms:
                    for inter_name, is_selected in selected_interactions.items():
                        if is_selected:
                            # Parse interaction name to get variable names
                            var_i, var_j = inter_name.split('×')
                            if var_i in X.columns and var_j in X.columns:
                                X[inter_name] = X[var_i] * X[var_j]
                                model_terms.append(inter_name)

                # Add selected quadratic terms
                if include_higher_terms:
                    for var_name, is_selected in selected_quadratic.items():
                        if is_selected and var_name in X.columns:
                            quad_col_name = f"{var_name}²"
                            X[quad_col_name] = X[var_name] ** 2
                            model_terms.append(quad_col_name)

                # Add intercept (always first if included)
                if include_intercept:
                    X.insert(0, 'Intercept', 1.0)
                    model_terms.insert(0, 'Intercept')

                n_coefficients = X.shape[1]

                # Store in session state
                st.session_state.doe_design_matrix = X
                st.session_state.doe_model_terms = model_terms
                st.session_state.doe_n_coefficients = n_coefficients
                st.session_state.doe_selected_interactions = {k:v for k,v in selected_interactions.items() if v}
                st.session_state.doe_selected_quadratic = {k:v for k,v in selected_quadratic.items() if v}
                st.session_state.doe_include_intercept = include_intercept

                st.success(f"✓ Design matrix built: {n_coefficients} coefficients")

                # Show preview
                with st.expander("📋 Design Matrix Preview (first 10 rows)", expanded=True):
                    st.dataframe(X.head(10), use_container_width=True)
                    st.write(f"**Full dimensions:** {X.shape[0]} rows × {X.shape[1]} columns")

                # Show model terms list
                with st.expander("📋 Model Terms (in order)"):
                    st.code(
                        "[\n  " + ",\n  ".join(f"'{t}'" for t in model_terms) + "\n]",
                        language="python"
                    )

                st.rerun()

            except Exception as e:
                st.error(f"❌ Error building design matrix: {str(e)}")
                import traceback
                st.code(traceback.format_exc())

    st.markdown("---")

    # ═══════════════════════════════════════════════════════════════════════
    # Step 3: D-Optimal Parameters (R CAT matching defaults)
    # ═══════════════════════════════════════════════════════════════════════

    if st.session_state.get('doe_design_matrix') is not None:
        design_matrix = st.session_state.doe_design_matrix
        st.markdown("## Step 3: D-Optimal Parameters")

        n_candidates = design_matrix.shape[0]  # r in R code
        n_coefficients = design_matrix.shape[1]  # co in R code

        # R CAT defaults: (co, r-1, 1, 10)
        default_min = n_coefficients
        default_max = n_candidates - 1  # KEY: NOT n_candidates!
        default_step = 1
        default_trials = 10

        st.info(f"""
**Design Matrix Info:**
- Candidates: {n_candidates}
- Coefficients: {n_coefficients}

**R CAT Defaults Applied:** Min={default_min}, Max={default_max}, Step={default_step}, Trials={default_trials}
        """)

        col_p1, col_p2, col_p3, col_p4 = st.columns(4)

        with col_p1:
            min_exp = st.number_input(
                "Min experiments:",
                min_value=n_coefficients,
                max_value=n_candidates-2,
                value=default_min,  # Auto-set to n_coefficients
                key="dopt_min",
                help=f"Must be ≥ {n_coefficients} (number of coefficients)"
            )

        with col_p2:
            max_exp = st.number_input(
                "Max experiments:",
                min_value=min_exp+1,
                max_value=n_candidates-1,  # KEY: capped at n_candidates-1
                value=default_max,  # Auto-set to n_candidates-1
                key="dopt_max",
                help=f"Must be ≤ {n_candidates-1} (need at least 1 point in candidate pool)"
            )

        with col_p3:
            step_exp = st.number_input(
                "Incremental step:",
                min_value=1,
                max_value=max(1, max_exp-min_exp),
                value=default_step,  # Default to 1
                key="dopt_step",
                help="Sequence: min, min+step, min+2*step, ..., max"
            )

        with col_p4:
            n_trials = st.slider(
                "Trials:",
                min_value=1,
                max_value=20,
                value=default_trials,  # Default to 10
                key="dopt_trials",
                help="Random restarts per design size"
            )

        # Generate design sequence
        design_sequence = list(range(min_exp, max_exp + 1, step_exp))
        if max_exp not in design_sequence:
            design_sequence.append(max_exp)

        st.markdown("### Design Sequence")

        col_seq1, col_seq2, col_seq3 = st.columns(3)

        with col_seq1:
            st.metric("Sizes to test", len(design_sequence))

        with col_seq2:
            # Show sequence (truncate if too long)
            if len(design_sequence) <= 10:
                st.code(str(design_sequence), language="python")
            else:
                preview = design_sequence[:10]
                st.code(str(preview) + f"\n... +{len(design_sequence)-10} more", language="python")

        with col_seq3:
            total_runs = len(design_sequence) * n_trials
            st.info(f"**Runs:** {len(design_sequence)} × {n_trials} = **{total_runs}**")

        # Warning for large designs
        if len(design_sequence) > 100:
            st.warning(f"""
⚠️ **Large Design Warning:**
You are about to test {len(design_sequence)} design sizes with {n_trials} trials each = **{total_runs} optimizations**

**Estimated time:** {total_runs // 10}-{total_runs // 5} minutes

**Recommendation:** Increase step size to reduce computation time.
- Current step: {step_exp}
- Suggested step: {max(5, (max_exp - min_exp) // 50)} (→ ~{(max_exp - min_exp) // max(5, (max_exp - min_exp) // 50)} sizes)
            """)
        elif len(design_sequence) > 50:
            st.info(f"""
💡 **Medium-sized design:** {total_runs} optimizations
**Estimated time:** {total_runs // 20}-{total_runs // 10} minutes
            """)

        st.markdown("---")

        # ═══════════════════════════════════════════════════════════════════
        # Step 4: Run D-Optimal
        # ═══════════════════════════════════════════════════════════════════

        st.markdown("## Step 4: Run D-Optimal")

        if st.button("🚀 RUN D-OPTIMAL", type="primary", use_container_width=True, key="doe_run"):
            with st.spinner("Running optimization on design matrix..."):
                try:
                    # Use DESIGN MATRIX (with model terms), not candidate matrix!
                    # Pass n_variables (n_coefficients) for correct log(M) calculation
                    results = doptimal_design(
                        design_matrix,
                        min_exp,
                        max_exp,
                        n_trials=n_trials,
                        n_variables=n_coefficients,
                        verbose=True
                    )
                    st.session_state.doptimal_results = results
                    st.success("✓ Complete!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ {str(e)}")

        st.markdown("---")

        # ═══════════════════════════════════════════════════════════════════
        # Step 5: Display Plots
        # ═══════════════════════════════════════════════════════════════════

        if st.session_state.get('doptimal_results') is not None:
            results = st.session_state.doptimal_results
            results_by_size = results.get('results_by_size', {})

            st.markdown("## Step 5: Inspect Plots")
            st.markdown("**Choose best design from efficiency (log M) and multicollinearity (VIF)**")

            col_plot1, col_plot2 = st.columns(2)

            # PLOT 1: log(M)
            with col_plot1:
                st.markdown("### 📊 PLOT 1: Efficiency")
                st.markdown("*Higher log(M) = better*")

                sizes = sorted(results_by_size.keys())
                log_m_values = [results_by_size[s]['log_M'] for s in sizes]

                fig1 = go.Figure()
                fig1.add_trace(go.Scatter(
                    x=sizes,
                    y=log_m_values,
                    mode='lines+markers',
                    name='log(M)',
                    line=dict(color='#E63946', width=3),
                    marker=dict(size=12),
                    hovertemplate='<b>%{x} exp</b><br>log(M)=%{y:.4f}<extra></extra>'
                ))

                fig1.update_layout(
                    title="log(M) vs Experiments",
                    xaxis_title="Number of Experiments",
                    yaxis_title="log(M)",
                    height=500,
                    template='plotly_white',
                    hovermode='x unified'
                )

                st.plotly_chart(fig1, use_container_width=True)

            # PLOT 2: VIF
            with col_plot2:
                st.markdown("### 📊 PLOT 2: Multicollinearity")
                st.markdown("*Lower VIF = better*")

                max_vif_values = [results_by_size[s]['max_vif'] for s in sizes]

                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(
                    x=sizes,
                    y=max_vif_values,
                    mode='lines+markers',
                    name='Max VIF',
                    line=dict(color='#E63946', width=3),
                    marker=dict(size=12),
                    hovertemplate='<b>%{x} exp</b><br>VIF=%{y:.2f}<extra></extra>'
                ))

                fig2.add_hline(y=4, line_dash="solid", line_color="green",
                              annotation_text="✓ Excellent (VIF<4)", annotation_position="right")
                fig2.add_hline(y=8, line_dash="dash", line_color="orange",
                              annotation_text="⚠️ Concerning (VIF>8)", annotation_position="right")

                fig2.update_layout(
                    title="Max VIF vs Experiments",
                    xaxis_title="Number of Experiments",
                    yaxis_title="Maximum VIF",
                    height=500,
                    template='plotly_white',
                    hovermode='x unified'
                )

                st.plotly_chart(fig2, use_container_width=True)

            st.markdown("---")

            # ═══════════════════════════════════════════════════════════════
            # Step 6: Selection
            # ═══════════════════════════════════════════════════════════════

            st.markdown("## Step 6: Select Best Design")
            st.markdown("**Inspect the two plots below to choose the best design**")
            st.markdown("**Select size from plot using the slider below:**")

            # Manual selection only - slider
            selected_size = st.select_slider(
                "Size from plot:",
                options=sizes,
                value=sizes[len(sizes)//2],
                key="doe_slider"
            )

            st.markdown("---")

            # ═══════════════════════════════════════════════════════════════
            # Step 7: Display Selected
            # ═══════════════════════════════════════════════════════════════

            st.markdown(f"## ✅ Selected: {selected_size} Experiments")

            selected_result = results_by_size[selected_size]
            selected_indices = selected_result['selected_indices']
            selected_design = candidate_encoded.iloc[selected_indices].reset_index(drop=True)

            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            with col_m1:
                st.metric("N Exp", selected_size)
            with col_m2:
                st.metric("log(M)", f"{selected_result['log_M']:.3f}")
            with col_m3:
                st.metric("Det", f"{selected_result['det']:.2e}")
            with col_m4:
                vif = selected_result['max_vif']
                status = "✓" if vif < 4 else "⚠️" if vif < 8 else "❌"
                st.metric(f"VIF {status}", f"{vif:.2f}")

            st.markdown("---")

            # ═══════════════════════════════════════════════════════════════
            # Step 8: Subset Extraction (Real + Coded)
            # ═══════════════════════════════════════════════════════════════

            st.markdown(f"## Step 8: Subset Extraction ({selected_size} of {candidate_encoded.shape[0]})")
            st.markdown("---")
            st.markdown("### 📊 Design Matrix: Real Values | Coded Values")

            # Get selected data
            selected_encoded = candidate_encoded.iloc[selected_indices].reset_index(drop=True)

            # Get variable names
            var_names = candidate_encoded.columns.tolist()

            # Create LEFT block: Real values
            if candidate_raw is not None:
                selected_raw = candidate_raw.iloc[selected_indices].reset_index(drop=True)

                # Build REAL matrix
                real_data = {}
                real_data['Exp_ID'] = [selected_indices[i] + 1 for i in range(len(selected_indices))]

                for var_name in var_names:
                    if var_name in selected_raw.columns:
                        real_data[var_name] = selected_raw[var_name].round(4)

                real_matrix = pd.DataFrame(real_data)
            else:
                real_matrix = None

            # Build CODED matrix
            coded_data = {}
            coded_data['Exp_ID'] = [selected_indices[i] + 1 for i in range(len(selected_indices))]

            for var_name in var_names:
                coded_data[var_name] = selected_encoded[var_name].round(4)

            coded_matrix = pd.DataFrame(coded_data)

            # Display as TWO SIDE-BY-SIDE BLOCKS
            col_real, col_coded = st.columns(2, gap="small")

            # LEFT BLOCK: Real Values
            with col_real:
                st.markdown("#### Real Values")
                if real_matrix is not None:
                    st.dataframe(real_matrix, use_container_width=True, height=600)
                else:
                    st.info("No raw data available - only coded values shown on right")

            # RIGHT BLOCK: Coded Values
            with col_coded:
                st.markdown("#### Coded Values")
                st.dataframe(coded_matrix, use_container_width=True, height=600)

            st.markdown("---")

            # Encoding reference (optional expandable)
            if real_matrix is not None:
                with st.expander("🔑 Encoding Reference (Real → Coded Mapping)"):
                    encoding_list = []

                    for var_name in var_names:
                        if var_name in selected_raw.columns:
                            # Get unique mappings from selected data
                            unique_pairs = list(zip(
                                sorted(selected_raw[var_name].unique()),
                                sorted(selected_encoded[var_name].unique())
                            ))

                            for real_val, coded_val in unique_pairs:
                                encoding_list.append({
                                    'Variable': var_name,
                                    'Real': round(real_val, 4),
                                    'Coded': round(coded_val, 4)
                                })

                    if encoding_list:
                        enc_df = pd.DataFrame(encoding_list)
                        st.dataframe(enc_df, use_container_width=True, hide_index=True)

            st.markdown("---")

            # ═══════════════════════════════════════════════════════════════
            # Step 9: Export
            # ═══════════════════════════════════════════════════════════════

            st.markdown("## Step 9: Export")

            col_exp1, col_exp2, col_exp3 = st.columns(3)

            with col_exp1:
                csv = selected_design.to_csv(index=False)
                st.download_button(
                    "📥 Download CSV (Coded)",
                    csv,
                    f"doptimal_{selected_size}exp.csv",
                    "text/csv",
                    key="doe_dl"
                )

            with col_exp2:
                if st.button("💾 Save to Workspace", key="doe_save"):
                    name = f"D-Optimal_{selected_size}exp"
                    success, msg = save_to_workspace(
                        selected_design,
                        name,
                        metadata={
                            'design_type': "D-Optimal",
                            'n_experiments': selected_size,
                            'log_M': float(selected_result['log_M']),
                            'determinant': float(selected_result['det']),
                            'max_vif': float(selected_result['max_vif']),
                            'indices': selected_indices
                        }
                    )
                    if success:
                        st.success(msg)
                    else:
                        st.error(msg)

            with col_exp3:
                if st.button("📊 All Results", key="doe_table"):
                    st.dataframe(
                        pd.DataFrame([
                            {
                                'Size': s,
                                'log(M)': f"{results_by_size[s]['log_M']:.3f}",
                                'Det': f"{results_by_size[s]['det']:.2e}",
                                'VIF': f"{results_by_size[s]['max_vif']:.2f}"
                            }
                            for s in sizes
                        ]),
                        use_container_width=True
                    )




def tab3_design_selection():
    """
    TAB 3: Design Selection and Generation
    Implements Full Factorial, Plackett-Burman, Central Composite designs
    """
    st.markdown("## 🎨 Design Selection and Generation")
    st.markdown("Generate classical experimental designs for specific objectives")

    st.markdown("---")

    # ═══════════════════════════════════════════════════════════════════════
    # Step 1: Select design type
    # ═══════════════════════════════════════════════════════════════════════

    st.markdown("## Step 1: Select Design Type")

    design_options = [
        ("Full Factorial 2^k", "full_factorial",
         "All combinations: 2 levels per variable (min/max)"),

        ("Fractional Factorial 2^4-1", "fractional_factorial_IV",
         "Fixed design: 8 experiments, 4 factors, Resolution IV ✅ VERIFIED"),

        ("Fractional Factorial 2^5-1", "fractional_factorial_V",
         "Fixed design: 16 experiments, 5 factors, Resolution V ✅ VERIFIED"),

        ("Plackett-Burman (N=8,12,16)", "plackett_burman",
         "Auto-size: Rapid screening, any number of factors ✅ VERIFIED"),

        ("Central Composite Design", "central_composite",
         "Response surface: 3 levels per variable (min/center/max)")
    ]

    design_type_idx = st.radio(
        "Choose design type:",
        range(len(design_options)),
        format_func=lambda x: design_options[x][0],
        key="doe_design_type_tab3"
    )

    design_type = design_options[design_type_idx]
    design_type_key = design_type[1]
    design_desc = design_type[2]

    st.info(design_desc)

    st.markdown("---")

    # ═══════════════════════════════════════════════════════════════════════
    # Step 2: Configure variables
    # ═══════════════════════════════════════════════════════════════════════

    st.markdown("## Step 2: Configure Variables")

    n_variables = st.slider(
        "Number of variables:",
        min_value=2,
        max_value=50,  # ← INCREASED: Support for larger screening designs
        value=3,
        key="doe_n_vars_tab3"
    )

    # ═══════════════════════════════════════════════════════════════════════════
    # Determine design type level requirements
    # ═══════════════════════════════════════════════════════════════════════════

    is_two_level_design = design_type_key in [
        "full_factorial",
        "fractional_factorial_IV",
        "fractional_factorial_V",
        "plackett_burman"
    ]
    is_ccd_design = design_type_key == "central_composite"

    st.markdown("---")

    variables_config = {}

    for i in range(n_variables):
        with st.expander(f"**Variable {i+1}**", expanded=(i < 2)):

            col_name, col_type = st.columns([2, 1])

            with col_name:
                var_name = st.text_input(
                    "Name:",
                    value=f"X{i+1}",
                    key=f"doe_tab3_var_name_{i}"
                )

            with col_type:
                var_type = st.radio(
                    "Type:",
                    ["Quantitative"],  # Only quantitative for standard designs
                    key=f"doe_tab3_var_type_{i}",
                    horizontal=True
                )

            col_min, col_max = st.columns(2)

            with col_min:
                min_val = st.number_input(
                    "Min (real):",
                    value=0.0,
                    key=f"doe_tab3_min_{i}"
                )

            with col_max:
                max_val = st.number_input(
                    "Max (real):",
                    value=100.0,
                    key=f"doe_tab3_max_{i}"
                )

            # ═══════════════════════════════════════════════════════════════════
            # CONDITIONAL: Show appropriate level inputs based on design type
            # ═══════════════════════════════════════════════════════════════════

            if is_two_level_design:
                # 2-level designs: use min/max only
                st.success("✓ 2 levels: min/max")
                levels = [min_val, max_val]

            elif is_ccd_design:
                # CCD: show center point input (3 levels)
                center_val = st.number_input(
                    "Center point (optional, default = middle):",
                    value=(min_val + max_val) / 2,
                    key=f"doe_tab3_center_{i}",
                    help="Center point for response surface"
                )
                st.success("✓ 3 levels: min, center, max")
                levels = [min_val, center_val, max_val]

            else:
                # Fallback (shouldn't happen)
                levels = [min_val, max_val]
                st.caption("Using min/max")

            variables_config[var_name] = {
                'type': 'Quantitative',
                'min': min_val,
                'max': max_val,
                'levels': levels
            }

    st.markdown("---")

    # ═══════════════════════════════════════════════════════════════════════════
    # Display design level info
    # ═══════════════════════════════════════════════════════════════════════════

    if is_two_level_design:
        st.info(f"""
✓ **2-Level Design Selected**
• Each variable uses: **MIN** and **MAX** values only
• Total levels per variable: **2**
• Example with {n_variables} variables: 2^{n_variables} = {2**n_variables} experiments (for Full Factorial)
        """)

    elif is_ccd_design:
        n_factorial = 2**n_variables
        n_axial = 2 * n_variables
        n_center = 5
        n_total = n_factorial + n_axial + n_center
        st.info(f"""
✓ **Central Composite Design (3-Level) Selected**
• Each variable uses: **MIN**, **CENTER**, and **MAX** values
• Total levels per variable: **3**
• Total experiments: 2^k (factorial) + 2k (axial) + 5 (center)
• Example with {n_variables} variables: {n_factorial} + {n_axial} + {n_center} = {n_total} experiments
        """)

    st.markdown("---")

    # ═══════════════════════════════════════════════════════════════════════
    # Step 3: Design-specific options
    # ═══════════════════════════════════════════════════════════════════════

    st.markdown("## Step 3: Design Options")

    if design_type_key == "central_composite":
        alpha_type = st.selectbox(
            "Alpha type (scaling):",
            ["orthogonal", "rotatable", "face"],
            key="doe_alpha_type",
            help="""
            - Orthogonal: Independent factors
            - Rotatable: Uniform variance sphere
            - Face: Uses factorial points as axial
            """
        )
    else:
        alpha_type = None

    st.markdown("---")

    # ═══════════════════════════════════════════════════════════════════════
    # Step 4: Generate
    # ═══════════════════════════════════════════════════════════════════════

    st.markdown("## Step 4: Generate Design")

    if st.button(
        "🚀 GENERATE DESIGN",
        type="primary",
        use_container_width=True,
        key="doe_gen_design_tab3"
    ):
        with st.spinner("Generating..."):
            try:
                # Validate
                if not variables_config:
                    st.error("❌ Configure variables")
                    return

                # Check 2-level requirement for PB
                if design_type_key == "plackett_burman":
                    for var_name, config in variables_config.items():
                        if len(config['levels']) != 2:
                            st.error(f"❌ Plackett-Burman needs 2 levels. {var_name} has {len(config['levels'])}")
                            return

                # ═══════════════════════════════════════════════════════════════════
                # Validate factor count for FIXED fractional factorial designs
                # ═══════════════════════════════════════════════════════════════════

                if design_type_key == "fractional_factorial_IV":
                    if n_variables != 4:
                        st.error(f"""
❌ **2^4-1 Design requires exactly 4 factors**

You selected: {n_variables} variables

2^4-1 (Fractional Factorial) is a FIXED design:
  • 8 experiments (runs)
  • 4 factors (variables) - REQUIRED!
  • Resolution IV

**Please choose one of:**
  • Adjust number of variables to 4
  • Select "Full Factorial 2^k" (scalable to any k)
  • Select "Plackett-Burman" (scalable to any number of factors)
                        """)
                        return

                if design_type_key == "fractional_factorial_V":
                    if n_variables != 5:
                        st.error(f"""
❌ **2^5-1 Design requires exactly 5 factors**

You selected: {n_variables} variables

2^5-1 (Fractional Factorial) is a FIXED design:
  • 16 experiments (runs)
  • 5 factors (variables) - REQUIRED!
  • Resolution V

**Please choose one of:**
  • Adjust number of variables to 5
  • Select "Full Factorial 2^k" (scalable to any k)
  • Select "Plackett-Burman" (scalable to any number of factors)
                        """)
                        return

                # Generate
                if design_type_key == "full_factorial":
                    design = generate_full_factorial(variables_config)
                elif design_type_key == "fractional_factorial_IV":
                    from generatedoe_utils.doe_designs import generate_fractional_factorial_IV
                    design = generate_fractional_factorial_IV(variables_config)
                elif design_type_key == "fractional_factorial_V":
                    from generatedoe_utils.doe_designs import generate_fractional_factorial_V
                    design = generate_fractional_factorial_V(variables_config)
                elif design_type_key == "plackett_burman":
                    design = generate_plackett_burman(variables_config)
                elif design_type_key == "central_composite":
                    design = generate_central_composite(variables_config, alpha_type=alpha_type)

                st.session_state.doe_generated_design_tab3 = design
                st.success(f"✓ Generated: {len(design)} experiments")
                st.rerun()

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                import traceback
                st.code(traceback.format_exc())

    st.markdown("---")

    # ═══════════════════════════════════════════════════════════════════════
    # Step 5: Display results
    # ═══════════════════════════════════════════════════════════════════════

    if st.session_state.get('doe_generated_design_tab3') is not None:
        st.markdown("## ✅ Generated Design")

        design = st.session_state.doe_generated_design_tab3

        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.metric("Experiments", len(design))
        with col_m2:
            n_cols = len([c for c in design.columns if c != 'Experiment_ID'])
            st.metric("Variables", n_cols)
        with col_m3:
            st.metric("Design Type", design_type[0])

        st.markdown("---")

        # ═══════════════════════════════════════════════════════════════════════
        # DOWNLOAD SECTION - SIMPLE CSV FORMAT
        # ═══════════════════════════════════════════════════════════════════════

        st.markdown("### 📥 Download Design Matrix")

        try:
            # Import the coding function
            from generatedoe_utils.doe_designs import (
                create_real_and_coded_matrices,
                create_combined_csv_side_by_side
            )

            # Create coded matrices
            df_real = design
            df_coded = create_real_and_coded_matrices(df_real, variables_config)

            # Generate filename based on design type
            design_type_key = design.attrs.get('design_type', 'design')
            n_vars = len([col for col in design.columns if col != 'Experiment_ID'])

            # Create filename mapping
            filename_map = {
                'full_factorial': f'FullFactorial_2pow{n_vars}',
                'fractional_factorial_IV': 'FractionalFactorial_2pow4_minus1',
                'fractional_factorial_V': 'FractionalFactorial_2pow5_minus1',
                'plackett_burman': f'PlackettBurman_N{design.attrs.get("pb_size", 8)}',
                'central_composite': f'CentralComposite_k{n_vars}'
            }

            filename_base = filename_map.get(design_type_key, 'ExperimentalDesign')

            # ═══════════════════════════════════════════════════════════════════════
            # DISPLAY: Real and Coded SIDE-BY-SIDE
            # ═══════════════════════════════════════════════════════════════════════

            st.markdown("#### Real Values (Left) | Coded Values (Right)")

            # Create display dataframe with alternating Real/Coded columns
            import pandas as pd
            display_df = pd.DataFrame()
            display_df['Experiment_ID'] = df_real['Experiment_ID']

            var_names = [col for col in df_real.columns if col != 'Experiment_ID']

            for var_name in var_names:
                display_df[var_name + '_Real'] = df_real[var_name].astype(int)
                display_df[var_name + '_Coded'] = df_coded[var_name].astype(int)

            st.dataframe(display_df, use_container_width=True)

            # ═══════════════════════════════════════════════════════════════════════
            # DOWNLOAD OPTIONS
            # ═══════════════════════════════════════════════════════════════════════

            st.markdown("---")
            st.markdown("#### 📥 Download Options")

            col_real, col_coded, col_both = st.columns(3)

            # Download: REAL ONLY
            with col_real:
                csv_real = df_real.to_csv(index=False)
                st.download_button(
                    label="📄 Real Values Only",
                    data=csv_real,
                    file_name=f"{filename_base}_Real.csv",
                    mime="text/csv",
                    key="doe_download_real_tab3"
                )
                st.caption("Original scale values")

            # Download: CODED ONLY
            with col_coded:
                csv_coded = df_coded.to_csv(index=False)
                st.download_button(
                    label="🔢 Coded Values Only",
                    data=csv_coded,
                    file_name=f"{filename_base}_Coded.csv",
                    mime="text/csv",
                    key="doe_download_coded_tab3"
                )
                st.caption("Standardized [-1, +1]")

            # Download: REAL + CODED SIDE-BY-SIDE
            with col_both:
                combined_csv = create_combined_csv_side_by_side(df_real, df_coded)
                st.download_button(
                    label="📊 Real + Coded (Combined)",
                    data=combined_csv,
                    file_name=f"{filename_base}_RealAndCoded.csv",
                    mime="text/csv",
                    key="doe_download_combined_tab3"
                )
                st.caption("Side-by-side in one file")

            # ═══════════════════════════════════════════════════════════════════════
            # ENCODING KEY
            # ═══════════════════════════════════════════════════════════════════════

            with st.expander("📖 Encoding Key - Real to Coded Mapping"):
                st.markdown("**How values are encoded:**")

                for var_name in var_names:
                    config = variables_config.get(var_name, {})

                    if 'levels' in config and len(config['levels']) > 0:
                        levels = config['levels']
                        min_val = min(levels)
                        max_val = max(levels)
                        if len(levels) >= 3:
                            center_val = levels[len(levels)//2]
                        else:
                            center_val = (min_val + max_val) / 2
                    else:
                        min_val = config.get('min', 0)
                        max_val = config.get('max', 100)
                        center_val = (min_val + max_val) / 2

                    n_levels = design.attrs.get('n_levels', 2)

                    if n_levels == 2:
                        st.write(f"**{var_name}:** {int(min_val)} → **-1**, {int(max_val)} → **+1**")
                    else:
                        st.write(f"**{var_name}:** {int(min_val)} → **-1**, {int(center_val)} → **0**, {int(max_val)} → **+1**")

        except Exception as e:
            st.error(f"❌ Download preparation failed: {str(e)}")
            import traceback
            with st.expander("🐛 Error details"):
                st.code(traceback.format_exc())

        st.markdown("---")

        st.markdown("## 🚀 Next Steps")

        col_next1, col_next2, col_next3 = st.columns(3)

        with col_next1:
            if st.button("🔐 Encode & Save", key="doe_encode_tab3"):
                with st.spinner("Encoding..."):
                    try:
                        from transforms.column_transforms import column_doe_coding

                        # Find where Experiment_ID column is (if present)
                        if 'Experiment_ID' in design.columns:
                            col_start = 1  # Skip Experiment_ID
                        else:
                            col_start = 0

                        # Encode using column_doe_coding (specialized for DoE)
                        # Returns 3-tuple: (encoded_df, metadata, multiclass_info)
                        design_encoded, meta, multiclass_info = column_doe_coding(
                            design,
                            col_range=(col_start, len(design.columns))
                        )

                        design_name = f"{design_type[0].replace(' ', '_')}_{len(design)}exp"
                        success, message = save_to_workspace(
                            design_encoded,
                            design_name,
                            metadata={
                                'design_type': design_type[0],
                                'n_experiments': len(design),
                                'n_variables': n_cols,
                                'encoding_metadata': meta
                            }
                        )

                        if success:
                            st.success(message)
                        else:
                            st.error(message)

                    except ImportError:
                        st.error("❌ column_doe_coding not found. Using fallback column_auto_encode...")
                        try:
                            from transforms.column_transforms import column_auto_encode

                            # Exclude Experiment_ID from encoding
                            data_cols = [c for c in design.columns if c != 'Experiment_ID']

                            design_encoded, meta = column_auto_encode(
                                design[data_cols],
                                col_range=(0, len(data_cols)),
                                exclude_cols=[]
                            )

                            design_name = f"{design_type[0].replace(' ', '_')}_{len(design)}exp"
                            success, message = save_to_workspace(
                                design_encoded,
                                design_name,
                                metadata={
                                    'design_type': design_type[0],
                                    'n_experiments': len(design),
                                    'n_variables': n_cols,
                                    'encoding_metadata': meta
                                }
                            )

                            if success:
                                st.success(message)
                            else:
                                st.error(message)

                            st.warning("⚠️ Using fallback encoder (limited functionality)")

                        except Exception as e2:
                            st.error(f"❌ Fallback also failed: {str(e2)}")
                            import traceback
                            st.code(traceback.format_exc())

                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())

        with col_next2:
            if st.button("🎯 D-Optimal (Tab 2)", key="doe_goto_dopt_tab3"):
                st.info("Use coded design in Tab 2 for optimization")

        with col_next3:
            csv = design.to_csv(index=False)
            st.download_button(
                "📥 Download CSV",
                csv,
                f"{design_type[0].lower().replace(' ', '_')}_design.csv",
                "text/csv",
                key="doe_download_tab3"
            )


# ════════════════════════════════════════════════════════════════════════════
# TAB 4: D-OPTIMAL BY ADDITION (WITH MATRIX DETECTION & MODEL DIALOG)
# ════════════════════════════════════════════════════════════════════════════

def detect_if_encoded(df: pd.DataFrame) -> tuple:
    """
    Detect if a DataFrame contains encoded/coded data.

    Returns:
        (is_encoded: bool, coding_type: str, reason: str)

    Coding patterns:
    - Dummy coding: columns have only 0, 1 values
    - Orthogonal coding: columns have -1, +1 values
    - Raw: continuous values (floats), wide range
    """
    if df.empty:
        return False, None, "Empty DataFrame"

    # Exclude metadata columns
    exclude_cols = {'Experiment_ID', 'Intercept'}
    data_cols = [c for c in df.columns if c not in exclude_cols]

    if not data_cols:
        return False, None, "No data columns"

    # Check each column
    unique_values_per_col = {}
    for col in data_cols:
        unique_vals = set(df[col].dropna().unique())
        unique_values_per_col[col] = unique_vals

    # Pattern 1: Dummy coding (0, 1 only)
    is_dummy = all(
        vals.issubset({0, 1, 0.0, 1.0})
        for vals in unique_values_per_col.values()
        if len(vals) > 0
    )

    if is_dummy:
        return True, "Dummy", "One-hot/dummy coding detected (0, 1 values)"

    # Pattern 2: Orthogonal coding (-1, +1 only)
    is_orthogonal = all(
        vals.issubset({-1, 1, -1.0, 1.0})
        for vals in unique_values_per_col.values()
        if len(vals) > 0
    )

    if is_orthogonal:
        return True, "Orthogonal", "Orthogonal coding detected (-1, +1 values)"

    # Pattern 3: Check for very few unique values (likely coded)
    unique_counts = [len(vals) for vals in unique_values_per_col.values()]
    if all(u <= 5 for u in unique_counts):  # All columns have ≤5 unique values
        return True, "Categorical", "Few unique values per column (likely coded)"

    # Otherwise: Raw
    return False, None, "Raw data detected (many unique continuous values)"


def tab4_doptimal_by_addition():
    """
    D-Optimal Design by Sequential Addition
    WITH MATRIX DETECTION & MODEL DIALOG (like mlr_doe.py in R)
    """
    st.markdown("## 🔧 D-Optimal Design by Sequential Addition")
    st.markdown("**Add optimal experiments to existing designs**")

    workspace_data = get_workspace_datasets()
    if not workspace_data:
        st.warning("⚠️ No datasets in workspace. Load data first from **Data Handling** tab")
        return

    data_names = list(workspace_data.keys())

    # ═══════════════════════════════════════════════════════════════════════
    # WORKSPACE VIEWER
    # ═══════════════════════════════════════════════════════════════════════

    st.markdown("### 📊 Available Datasets in Workspace")

    # Create simple info table
    datasets_info = []
    for name in data_names:
        df = workspace_data[name]
        datasets_info.append({
            'Dataset': name,
            'Experiments': df.shape[0],
            'Columns': df.shape[1]
        })

    st.dataframe(
        pd.DataFrame(datasets_info),
        use_container_width=True,
        hide_index=True,
        key="tab4_workspace_list"
    )

    st.markdown(f"**Total: {len(data_names)} datasets available**")
    st.markdown("---")

    # ─────────────────────────────────────────────────────────────────────
    # STEP 1: MATRIX DETECTION & SELECTION
    # ─────────────────────────────────────────────────────────────────────

    st.markdown("### Step 1️⃣ Select Experimental Matrices")
    st.markdown("Choose your performed experiments and candidate points (both must be ENCODED)")

    col_select1, col_select2 = st.columns(2)

    # Perform encoding detection for all datasets
    matrix_info = {}
    for name in data_names:
        df = workspace_data[name]
        is_coded, code_type, reason = detect_if_encoded(df)
        matrix_info[name] = {
            'is_coded': is_coded,
            'code_type': code_type,
            'reason': reason,
            'shape': df.shape
        }

    # PERFORMED EXPERIMENTS
    with col_select1:
        st.markdown("**Performed Experiments**")

        selected_performed = st.selectbox(
            "Select performed experiments (ENCODED)",
            data_names,
            key="tab4_performed_select",
            help="Must be an encoded design matrix"
        )

        if selected_performed:
            perf_df = workspace_data[selected_performed]
            perf_info = matrix_info[selected_performed]

            col_p1, col_p2 = st.columns(2)
            with col_p1:
                st.metric("Experiments", perf_info['shape'][0])
            with col_p2:
                st.metric("Columns", perf_info['shape'][1])

            # Encoding status
            if perf_info['is_coded']:
                st.success(f"✅ {perf_info['code_type']} coding detected")
            else:
                st.warning(f"⚠️ {perf_info['reason']}")

            with st.expander("📋 Preview"):
                st.dataframe(perf_df.head(6), use_container_width=True, height=200)

    # ═══════════════════════════════════════════════════════════════════════
    # WORKSPACE VIEWER (between selections)
    # ═══════════════════════════════════════════════════════════════════════

    st.markdown("---")
    st.markdown("### 📊 Available Datasets in Workspace")

    # Create simple info table
    datasets_info_mid = []
    for name in data_names:
        df = workspace_data[name]
        datasets_info_mid.append({
            'Dataset': name,
            'Experiments': df.shape[0],
            'Columns': df.shape[1]
        })

    st.dataframe(
        pd.DataFrame(datasets_info_mid),
        use_container_width=True,
        hide_index=True,
        key="tab4_workspace_list_mid"
    )

    st.markdown(f"**Total: {len(data_names)} datasets available**")
    st.markdown("---")

    # CANDIDATE POINTS (SAME AS PERFORMED - ALL DATASETS)
    with col_select2:
        st.markdown("**Candidate Points**")

        selected_candidates = st.selectbox(
            "Select candidates (ENCODED)",
            data_names,
            key="tab4_candidates_select",
            help="Must be an encoded design matrix (same encoding as performed)"
        )

        if selected_candidates:
            cand_df = workspace_data[selected_candidates]
            cand_info = matrix_info[selected_candidates]

            col_c1, col_c2 = st.columns(2)
            with col_c1:
                st.metric("Candidates", cand_info['shape'][0])
            with col_c2:
                st.metric("Columns", cand_info['shape'][1])

            # Encoding status
            if cand_info['is_coded']:
                st.success(f"✅ {cand_info['code_type']} coding detected")
            else:
                st.warning(f"⚠️ {cand_info['reason']}")

            with st.expander("📋 Preview"):
                st.dataframe(cand_df.head(6), use_container_width=True, height=200)

    st.markdown("---")

    # ─────────────────────────────────────────────────────────────────────
    # STEP 2: SHAPE VALIDATION
    # ─────────────────────────────────────────────────────────────────────

    if selected_performed and selected_candidates:
        perf_df = workspace_data[selected_performed]
        cand_df = workspace_data[selected_candidates]

        if perf_df.shape[1] != cand_df.shape[1]:
            st.error(f"""
❌ **Column count mismatch!**
- Performed: {perf_df.shape[1]} columns
- Candidates: {cand_df.shape[1]} columns

Both matrices must have **same columns** (same encoding).
Use encoding output from same Tab 1/2 design process.
            """)
            return

        if perf_df.shape[0] >= cand_df.shape[0]:
            st.warning(f"""
⚠️ **Few candidates available**
- Performed: {perf_df.shape[0]} experiments
- Candidates: {cand_df.shape[0]} points

Need more candidates than performed experiments for addition to work.
            """)

        st.success("✓ Matrices validated (same structure)")

        # ─────────────────────────────────────────────────────────────────
        # STEP 3: MODEL SPECIFICATION DIALOG (like R mlr_doe)
        # ─────────────────────────────────────────────────────────────────

        st.markdown("### Step 2️⃣ Model Specification")
        st.markdown("Define which terms to include in your model")

        col_model1, col_model2 = st.columns(2)

        with col_model1:
            include_intercept = st.checkbox(
                "✓ Include Intercept (constant term)",
                value=True,
                key="tab4_intercept",
                help="Add intercept/constant to model"
            )

        with col_model2:
            include_higher_terms = st.checkbox(
                "✓ Include Higher-Order Terms",
                value=False,
                key="tab4_higher_terms",
                help="Add interactions and/or quadratic terms"
            )

        st.markdown("")

        # Get column names (excluding Intercept and Experiment_ID)
        exclude_cols = {'Intercept', 'Experiment_ID'}
        var_names = [c for c in perf_df.columns if c not in exclude_cols]
        n_vars = len(var_names)

        selected_interactions = {}
        selected_quadratic = {}

        if include_higher_terms and n_vars > 0:
            # ═════════════════════════════════════════════════════════════
            # QUADRATIC TERMS (if applicable)
            # ═════════════════════════════════════════════════════════════

            st.markdown("#### Quadratic Terms")

            # Only show quadratic if columns look like continuous variables
            has_quadratic_candidates = any(
                c not in ['Intercept', 'Experiment_ID'] and '×' not in c
                for c in perf_df.columns
            )

            if has_quadratic_candidates:
                cols_quad = st.columns(3)
                for i, var_name in enumerate(var_names):
                    with cols_quad[i % 3]:
                        selected_quadratic[var_name] = st.checkbox(
                            f"☑ {var_name}²",
                            value=True,
                            key=f"tab4_quad_{var_name}"
                        )
                st.markdown("")

            # ═════════════════════════════════════════════════════════════
            # INTERACTION TERMS
            # ═════════════════════════════════════════════════════════════

            st.markdown("**Interaction Terms:**")

            # STEP 1: Detect which variables are dummy variables (only {0, 1} values)
            # This is the same logic as in model_computation.py analyze_design_structure()
            dummy_vars = set()

            for col_name in var_names:
                unique_vals = perf_df[col_name].dropna().unique()
                unique_set = set(np.round(unique_vals, 10))

                # If column contains ONLY {0, 1} → it's a dummy variable (qualitative)
                if unique_set == {0.0, 1.0} or unique_set == {0.0} or unique_set == {1.0}:
                    dummy_vars.add(col_name)

            # STEP 2: Group dummy variables by prefix
            # Example: X1_A, X1_B, X1_C → all belong to group "X1"
            dummy_groups = {}
            for dummy_col in dummy_vars:
                if '_' in dummy_col:
                    prefix = dummy_col.rsplit('_', 1)[0]
                    if prefix not in dummy_groups:
                        dummy_groups[prefix] = []
                    dummy_groups[prefix].append(dummy_col)

            # STEP 3: Identify multi-level categorical groups (3+ dummies)
            multi_level_categorical_groups = {
                prefix: cols for prefix, cols in dummy_groups.items()
                if len(cols) >= 3
            }

            # STEP 4: Show warning if multi-level categoricals detected
            if multi_level_categorical_groups:
                multi_level_names = ', '.join(sorted(multi_level_categorical_groups.keys()))
                st.warning(f"""
⚠️ **Categorical Variables with 3+ Levels:** {multi_level_names}

Interactions involving these variables are **DISABLED**.
*(One-hot encoding → cannot use direct interactions)*
                """)

            # STEP 5: Build valid interactions only
            # Valid = interactions that DON'T involve dummy variables from multi-level categoricals
            valid_interactions = []

            for i in range(len(var_names)):
                for j in range(i + 1, len(var_names)):
                    var_i = var_names[i]
                    var_j = var_names[j]

                    # Check if either variable belongs to a multi-level categorical
                    var_i_in_multiclass = any(var_i in dummy_groups.get(prefix, [])
                                             for prefix in multi_level_categorical_groups.keys())
                    var_j_in_multiclass = any(var_j in dummy_groups.get(prefix, [])
                                             for prefix in multi_level_categorical_groups.keys())

                    # ✅ ALLOW only if NEITHER belongs to multi-level categorical
                    if not (var_i_in_multiclass or var_j_in_multiclass):
                        valid_interactions.append(f"{var_i}×{var_j}")

            # STEP 6: Display valid interactions
            if len(valid_interactions) > 0:
                interaction_cols = st.columns(3)

                selected_interactions = {}
                for idx, interaction in enumerate(valid_interactions):
                    col_idx = idx % 3
                    with interaction_cols[col_idx]:
                        default_checked = True
                        selected_interactions[interaction] = st.checkbox(
                            f"✓ {interaction}",
                            value=default_checked,
                            key=f"tab4_interaction_{interaction}"
                        )

                st.session_state.selected_interactions = [k for k, v in selected_interactions.items() if v]
            else:
                st.info("ℹ️ No valid interactions available")
                st.session_state.selected_interactions = []

            st.markdown("")

        # ═════════════════════════════════════════════════════════════════
        # SUMMARY (like mlr_doe.py)
        # ═════════════════════════════════════════════════════════════════

        st.markdown("#### Model Summary")

        n_linear = n_vars
        n_interactions = sum(1 for v in selected_interactions.values() if v)
        n_quadratic = sum(1 for v in selected_quadratic.values() if v)
        n_intercept = 1 if include_intercept else 0

        total_coefficients = n_linear + n_interactions + n_quadratic + n_intercept

        col_s1, col_s2, col_s3, col_s4 = st.columns(4)

        with col_s1:
            st.metric("Linear", n_linear)
        with col_s2:
            st.metric("Interactions", n_interactions)
        with col_s3:
            st.metric("Quadratic", n_quadratic)
        with col_s4:
            st.metric("Total Coefficients", total_coefficients)

        st.markdown("---")

        # ─────────────────────────────────────────────────────────────────
        # STEP 4: OPTIMIZATION PARAMETERS
        # ─────────────────────────────────────────────────────────────────

        st.markdown("### Step 3️⃣ Optimization Parameters")

        n_performed = perf_df.shape[0]
        n_candidates = cand_df.shape[0]
        n_coefficients = total_coefficients

        col_opt1, col_opt2, col_opt3, col_opt4 = st.columns(4)

        with col_opt1:
            min_to_add = st.number_input(
                "Min to add",
                value=1,
                min_value=0,
                max_value=20,
                key="tab4_min_add"
            )

        with col_opt2:
            max_to_add = st.number_input(
                "Max to add",
                value=min(5, n_candidates - 1),
                min_value=max(1, min_to_add),
                max_value=n_candidates - 1,
                key="tab4_max_add"
            )

        with col_opt3:
            n_trials = st.number_input(
                "N trials",
                value=10,
                min_value=1,
                max_value=50,
                key="tab4_n_trials"
            )

        with col_opt4:
            verbose = st.checkbox(
                "Verbose",
                value=True,
                key="tab4_verbose"
            )

        st.markdown(f"""
**Setup:**
- Performed: {n_performed} exp
- Candidates: {n_candidates} points
- Model coefficients: {n_coefficients}
- Estimated time: ~{max(2, (max_to_add-min_to_add+1)*n_trials//10)}s
        """)

        st.markdown("---")

        # ─────────────────────────────────────────────────────────────────
        # STEP 5: RUN OPTIMIZATION
        # ─────────────────────────────────────────────────────────────────

        st.markdown("### Step 4️⃣ Run Optimization")

        if st.button("🚀 RUN D-OPTIMAL BY ADDITION", type="primary", use_container_width=True, key="tab4_run"):
            with st.spinner(f"Running optimization ({(max_to_add-min_to_add+1)*n_trials} runs)..."):
                try:
                    # Build interaction and quadratic dicts in proper format
                    interactions_dict = {}
                    inter_idx = 0
                    for i in range(n_vars):
                        for j in range(i + 1, n_vars):
                            key = f"{i}:{j}"
                            inter_name = f"{var_names[i]}×{var_names[j]}"
                            interactions_dict[key] = selected_interactions.get(inter_name, False)
                            inter_idx += 1

                    quadratic_dict = {
                        str(i): selected_quadratic.get(var_names[i], False)
                        for i in range(n_vars)
                    }

                    # Run optimization (MODE 2 - with model building)
                    results = doptimal_by_addition(
                        performed_experiments=perf_df,
                        candidate_matrix=cand_df,
                        min_to_add=min_to_add,
                        max_to_add=max_to_add,
                        n_trials=n_trials,
                        include_intercept=include_intercept,
                        interactions_dict=interactions_dict,
                        quadratic_dict=quadratic_dict,
                        verbose=verbose
                    )

                    st.session_state['tab4_results'] = results
                    st.session_state['tab4_perf_df'] = perf_df
                    st.session_state['tab4_cand_df'] = cand_df
                    st.session_state['tab4_var_names'] = var_names
                    st.session_state['tab4_min_add_value'] = min_to_add
                    st.session_state['tab4_max_add_value'] = max_to_add
                    st.success("✓ Optimization complete!")
                    st.rerun()

                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())

    # ─────────────────────────────────────────────────────────────────────
    # STEP 6: Display results
    # ─────────────────────────────────────────────────────────────────────

    # ═════════════════════════════════════════════════════════════════════
    # STEP 6: RESULTS WITH USER SELECTION
    # ═════════════════════════════════════════════════════════════════════

    if st.session_state.get('tab4_results') is not None:
        st.markdown("---")
        st.markdown("## ✅ Results")

        results = st.session_state['tab4_results']
        perf_df = st.session_state['tab4_perf_df']
        cand_df = st.session_state['tab4_cand_df']
        min_to_add_value = st.session_state.get('tab4_min_add_value', 1)
        max_to_add_value = st.session_state.get('tab4_max_add_value', 5)

        col_m1, col_m2, col_m3, col_m4 = st.columns(4)

        with col_m1:
            st.metric("Performed", results['n_performed'])
        with col_m2:
            st.metric("Candidates", results['n_candidates'])
        with col_m3:
            st.metric("Tested Sizes", len(results['results_by_size']))
        with col_m4:
            st.metric("Model Coefficients", results['n_coefficients'])

        st.markdown("---")

        # Results table
        results_df = format_addition_results(results)
        st.dataframe(results_df, use_container_width=True)

        st.markdown("---")

        # Charts
        col_chart1, col_chart2 = st.columns(2)

        with col_chart1:
            st.markdown("### log(M) - Efficiency")
            fig_logm = go.Figure()
            fig_logm.add_trace(go.Scatter(
                x=results_df['N_to_Add'],
                y=results_df['log(M)'],
                mode='lines+markers',
                name='Efficiency',
                line=dict(color='#FF6B6B', width=2),
                marker=dict(size=8),
                hovertemplate='N_add=%{x}<br>log(M)=%{y:.4f}<extra></extra>'
            ))
            fig_logm.update_layout(
                height=350,
                margin=dict(l=40, r=20, t=30, b=40),
                hovermode='x unified'
            )
            st.plotly_chart(fig_logm, use_container_width=True)

        with col_chart2:
            st.markdown("### Max VIF - Quality ⚠️")
            fig_vif = go.Figure()
            fig_vif.add_trace(go.Scatter(
                x=results_df['N_to_Add'],
                y=results_df['Max_VIF'],
                mode='lines+markers',
                name='Max VIF',
                line=dict(color='#4ECDC4', width=2),
                marker=dict(size=8),
                hovertemplate='N_add=%{x}<br>VIF=%{y:.2f}<extra></extra>'
            ))
            fig_vif.add_hline(y=4, line_dash="dash", line_color="orange", annotation_text="VIF=4 (Good)")
            fig_vif.add_hline(y=8, line_dash="dash", line_color="red", annotation_text="VIF=8 (Warning)")
            fig_vif.update_layout(
                height=350,
                margin=dict(l=40, r=20, t=30, b=40),
                hovermode='x unified'
            )
            st.plotly_chart(fig_vif, use_container_width=True)

        st.markdown("---")

        # ═════════════════════════════════════════════════════════════════
        # USER SELECTION: How many experiments to add?
        # ═════════════════════════════════════════════════════════════════

        st.markdown("### 🎯 Select Number of Experiments to Add")
        st.markdown("Choose how many new experiments you want to add to your design")

        col_slider1, col_slider2 = st.columns([3, 1])

        with col_slider1:
            # Handle case where min equals max (only one option)
            if min_to_add_value == max_to_add_value:
                n_to_add_selected = min_to_add_value
                st.info(f"Only one configuration available: **{n_to_add_selected} experiments**")
            else:
                n_to_add_selected = st.slider(
                    "Number of experiments to add:",
                    min_value=min_to_add_value,
                    max_value=max_to_add_value,
                    value=min_to_add_value,
                    step=1,
                    key="tab4_n_to_add_slider",
                    help=f"Range: {min_to_add_value} to {max_to_add_value}"
                )

        with col_slider2:
            st.metric("Total Experiments", results['n_performed'] + n_to_add_selected)

        st.markdown("---")

        # Find results for selected number
        results_by_size = results['results_by_size']

        if n_to_add_selected in results_by_size:
            selected_design = results_by_size[n_to_add_selected]

            # ═════════════════════════════════════════════════════════════
            # CONFIGURATION METRICS
            # ═════════════════════════════════════════════════════════════

            st.markdown("### 📊 Configuration for Your Selection")

            col_config1, col_config2, col_config3 = st.columns(3)

            with col_config1:
                st.metric("log(M)", f"{selected_design['log_M']:.4f}")
            with col_config2:
                st.metric("Determinant", f"{selected_design['det']:.2e}")
            with col_config3:
                st.metric("Max VIF", f"{selected_design['max_vif']:.2f}")

            st.markdown(f"""
**Your Selection:**
- **Add {n_to_add_selected} experiments** to your {results['n_performed']} performed
- **Total: {results['n_performed'] + n_to_add_selected} experiments**
- **Candidate indices to add:** {selected_design['added_indices']}
            """)

            st.markdown("---")

            # ═════════════════════════════════════════════════════════════
            # SIDE-BY-SIDE COMPARISON
            # ═════════════════════════════════════════════════════════════

            st.markdown("### 🔀 Performed vs. To Add (Comparison)")

            col_perf, col_add = st.columns(2)

            with col_perf:
                st.markdown(f"**📋 Performed Experiments** ({len(perf_df)})")
                st.dataframe(perf_df, use_container_width=True, height=300)

            with col_add:
                st.markdown(f"**➕ Experiments to Add** ({n_to_add_selected})")
                added_expts = cand_df.iloc[selected_design['added_indices']].copy()
                st.dataframe(added_expts, use_container_width=True, height=300)

            st.markdown("---")

            # ═════════════════════════════════════════════════════════════
            # MERGED DESIGN
            # ═════════════════════════════════════════════════════════════

            st.markdown("### ✨ Merged Design (Combined)")

            # Create merged design
            merged_df = pd.concat([perf_df, added_expts], ignore_index=True)

            st.dataframe(merged_df, use_container_width=True, height=400)

            st.markdown(f"""
**Merged Design Summary:**
- Original: {len(perf_df)} experiments
- Added: {n_to_add_selected} experiments
- **Total: {len(merged_df)} experiments**
            """)

            st.markdown("---")

            # ═════════════════════════════════════════════════════════════
            # SAVE OPTIONS
            # ═════════════════════════════════════════════════════════════

            st.markdown("### 💾 Save")

            col_save1, col_save2, col_save3 = st.columns(3)

            with col_save1:
                if st.button("💾 Save Added Experiments", key="tab4_save1", use_container_width=True):
                    try:
                        success, msg = save_to_workspace(
                            added_expts,
                            f"AddedExp_{n_to_add_selected}exp",
                            metadata={
                                'source': 'D-Optimal by Addition',
                                'n_added': n_to_add_selected,
                                'n_total': len(merged_df),
                                'log_M': float(selected_design['log_M']),
                                'max_vif': float(selected_design['max_vif'])
                            }
                        )
                        st.success(msg) if success else st.error(msg)
                    except Exception as e:
                        st.error(f"Error saving: {str(e)}")

            with col_save2:
                if st.button("💾 Save Merged Design", key="tab4_save_merged", use_container_width=True):
                    try:
                        success, msg = save_to_workspace(
                            merged_df,
                            f"Merged_Design_{len(merged_df)}exp",
                            metadata={
                                'source': 'D-Optimal by Addition',
                                'n_performed': len(perf_df),
                                'n_added': n_to_add_selected,
                                'n_total': len(merged_df),
                                'log_M': float(selected_design['log_M']),
                                'max_vif': float(selected_design['max_vif'])
                            }
                        )
                        st.success(msg) if success else st.error(msg)
                    except Exception as e:
                        st.error(f"Error saving: {str(e)}")

            with col_save3:
                if st.button("💾 Save All Results", key="tab4_save_results", use_container_width=True):
                    try:
                        success, msg = save_to_workspace(
                            results_df,
                            "DOE_Addition_Results_Full",
                            metadata={
                                'source': 'D-Optimal by Addition',
                                'n_performed': len(perf_df),
                                'n_candidates': len(cand_df),
                                'selected_n': n_to_add_selected
                            }
                        )
                        st.success(msg) if success else st.error(msg)
                    except Exception as e:
                        st.error(f"Error saving: {str(e)}")

            st.markdown("---")

            # ═════════════════════════════════════════════════════════════
            # DOWNLOAD OPTIONS
            # ═════════════════════════════════════════════════════════════

            st.markdown("### 📥 Download")

            col_dl1, col_dl2, col_dl3 = st.columns(3)

            with col_dl1:
                csv_added = added_expts.to_csv(index=False)
                st.download_button(
                    "📥 Added Experiments CSV",
                    csv_added,
                    f"added_exp_{n_to_add_selected}exp.csv",
                    "text/csv",
                    use_container_width=True
                )

            with col_dl2:
                csv_merged = merged_df.to_csv(index=False)
                st.download_button(
                    "📥 Merged Design CSV",
                    csv_merged,
                    f"merged_design_{len(merged_df)}exp.csv",
                    "text/csv",
                    use_container_width=True
                )

            with col_dl3:
                csv_results = results_df.to_csv(index=False)
                st.download_button(
                    "📥 Results Table CSV",
                    csv_results,
                    f"doe_addition_results.csv",
                    "text/csv",
                    use_container_width=True
                )

        else:
            st.error(f"❌ No results found for {n_to_add_selected} experiments. This shouldn't happen!")


def show():
    """Main function to display the Generate DoE page."""
    initialize_session_state()

    st.title("⚡ Generate DoE")
    st.markdown("**Experimental Design Generation and D-Optimal Optimization**")

    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 All Combinations Generator",
        "🎯 D-Optimal Design",
        "🎨 Design Selection (FF, PB, CCD)",
        "🔧 D-Optimal by Addition"
    ])

    with tab1:
        tab1_design_generator()

    with tab2:
        tab2_doptimal_design()

    with tab3:
        tab3_design_selection()

    with tab4:
        tab4_doptimal_by_addition()


if __name__ == "__main__":
    # For standalone testing
    st.set_page_config(page_title="Generate DoE", layout="wide")
    show()
