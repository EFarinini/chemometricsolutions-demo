"""
Bayesian Optimization Page for Streamlit
=========================================

Complete UI for Bayesian Optimization experimental design including:
- Factor and target selection
- GP model configuration
- Optimization execution
- Results visualization
- Iterative refinement
- Validation mode (train/test split)

Author: ChemoMetric Solutions
License: MIT
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
from typing import Optional
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment
from io import BytesIO

# Import BO modules
from bayesian_optimization_doe import BayesianOptimizationDesigner, display_bo_summary
from bayesian_utils import (
    validate_bounds,
    format_results_display,
    compute_acquisition_ei,
    compute_acquisition_lcb,
    detect_encoding,
    infer_factor_bounds,
    inverse_transform_predictions
)


def show_bayesian_optimization_page():
    """
    Main function to display Bayesian Optimization page.

    Sections:
    1. Header with metrics
    2. Factor selection and bounds
    3. Target selection
    4. GP configuration
    5. Optimization execution
    6. Results display
    7. Visualizations
    8. Iterative refinement
    9. Validation mode (train/test split)
    """

    # ========================================================================
    # SECTION 1 - HEADER (50 lines)
    # ========================================================================

    st.title("🎯 Bayesian Optimization for Experimental Design")

    st.markdown("""
    ## 🔄 Bayesian Optimization Workflow

    **Follow this 3-round strategy:**

    **✅ ROUND 1: SCREENING (Plackett-Burman)**
    - Input: 6+ factors, no prior knowledge
    - Method: 12-run Plackett-Burman
    - Output: Identify 3-5 significant factors (p < 0.05)

    **✅ ROUND 2: CHARACTERIZATION (Factorial/CCD)**
    - Input: 4-5 significant factors from Round 1
    - Method: Full Factorial/Central Composite Design (CCD)
    - Output: 16-30 data points + validated MLR model

    **🔄 ROUND 3: BAYESIAN OPTIMIZATION (This Tab)**
    - Input: 30+ data points + significant factors
    - Method: Gaussian Process + Expected Improvement
    - Output: 5-10 optimal suggestions per iteration

    **📊 Result:** ~40-50 experiments vs 60-80 (classical DoE) = **30-40% reduction**
    """)

    st.divider()

    # Check for data availability
    if 'current_data' not in st.session_state or st.session_state.current_data is None:
        st.warning("⚠️ No data loaded. Please load a dataset from the Data Management page.")
        st.stop()

    current_data = st.session_state.current_data

    # ========================================================================
    # DATA PREVIEW & STATISTICS SECTION
    # ========================================================================

    with st.expander("📊 Data Preview & Statistics", expanded=False):
        st.markdown("### Dataset Overview")

        # Get basic statistics
        n_rows = len(current_data)
        n_cols = len(current_data.columns)
        numeric_cols = current_data.select_dtypes(include=[np.number]).columns.tolist()
        n_numeric = len(numeric_cols)

        # Display statistics row
        col_stat1, col_stat2, col_stat3 = st.columns(3)
        with col_stat1:
            st.metric("📋 Observations", n_rows)
        with col_stat2:
            st.metric("📊 Total Variables", n_cols)
        with col_stat3:
            st.metric("🔢 Numeric Variables", n_numeric)

        st.divider()

        # Display first 10 rows
        st.markdown("**First 10 rows of dataset:**")
        display_df = current_data.head(10).copy()

        # Round numeric columns for better display
        for col in display_df.select_dtypes(include=[np.number]).columns:
            display_df[col] = display_df[col].round(4)

        st.dataframe(display_df, use_container_width=True, hide_index=False)

        if n_rows > 10:
            st.caption(f"Showing 10 of {n_rows} total observations")

        st.divider()

        # Auto-detect if data is coded
        st.markdown("### 🔍 Data Encoding Detection")

        def detect_coded_data(df, numeric_columns):
            """
            Detect if data is coded (DoE-style [-1, 0, +1]) or natural units.

            Returns:
            --------
            is_coded : bool
                True if data appears to be coded
            confidence : str
                'high', 'medium', 'low', or 'uncertain'
            details : dict
                Diagnostic information
            """
            if not numeric_columns:
                return False, 'uncertain', {'message': 'No numeric columns found'}

            # Check each numeric column
            coded_count = 0
            natural_count = 0
            uncertain_count = 0

            column_analysis = []

            for col in numeric_columns:
                values = df[col].dropna()
                if len(values) == 0:
                    continue

                # Get unique values
                unique_vals = values.unique()
                n_unique = len(unique_vals)

                # Check if values are close to -1, 0, +1 pattern
                # Tolerance for coded values (allow some rounding errors)
                tol = 0.15

                # Count how many values are near -1, 0, or +1
                near_coded = np.sum(
                    (np.abs(values + 1) < tol) |  # Near -1
                    (np.abs(values) < tol) |       # Near 0
                    (np.abs(values - 1) < tol)     # Near +1
                )

                pct_near_coded = near_coded / len(values)

                # Also check if range is approximately [-1, 1]
                val_min, val_max = values.min(), values.max()
                in_coded_range = (val_min >= -1.2 and val_max <= 1.2)

                # Decision logic
                if pct_near_coded > 0.7 and in_coded_range and n_unique <= 5:
                    # Strong evidence for coded
                    coded_count += 1
                    col_type = 'Coded'
                    confidence = 'High'
                elif pct_near_coded > 0.5 and in_coded_range:
                    # Moderate evidence for coded
                    coded_count += 1
                    col_type = 'Coded'
                    confidence = 'Medium'
                elif n_unique > 10 and (val_max - val_min) > 2.5:
                    # Strong evidence for natural units
                    natural_count += 1
                    col_type = 'Natural'
                    confidence = 'High'
                else:
                    # Uncertain
                    uncertain_count += 1
                    col_type = 'Uncertain'
                    confidence = 'Low'

                column_analysis.append({
                    'Column': col,
                    'Type': col_type,
                    'Confidence': confidence,
                    'Range': f"[{val_min:.4f}, {val_max:.4f}]",
                    'Unique Values': n_unique,
                    '% Near DoE Levels': f"{pct_near_coded*100:.1f}%"
                })

            # Overall decision
            total_analyzed = coded_count + natural_count + uncertain_count
            if total_analyzed == 0:
                return False, 'uncertain', {'column_analysis': column_analysis}

            pct_coded = coded_count / total_analyzed
            pct_natural = natural_count / total_analyzed

            if pct_coded > 0.7:
                overall_is_coded = True
                overall_confidence = 'high'
            elif pct_coded > 0.5:
                overall_is_coded = True
                overall_confidence = 'medium'
            elif pct_natural > 0.7:
                overall_is_coded = False
                overall_confidence = 'high'
            elif pct_natural > 0.5:
                overall_is_coded = False
                overall_confidence = 'medium'
            else:
                overall_is_coded = False
                overall_confidence = 'uncertain'

            details = {
                'column_analysis': column_analysis,
                'coded_count': coded_count,
                'natural_count': natural_count,
                'uncertain_count': uncertain_count,
                'pct_coded': pct_coded * 100,
                'pct_natural': pct_natural * 100
            }

            return overall_is_coded, overall_confidence, details

        # Run detection
        is_coded, confidence, details = detect_coded_data(current_data, numeric_cols)

        # Display detection results
        col_det1, col_det2 = st.columns([1, 2])

        with col_det1:
            if is_coded:
                st.success("**Detection: CODED DATA** ✓")
                st.write(f"**Confidence:** {confidence.upper()}")
            else:
                st.info("**Detection: NATURAL UNITS** 📏")
                st.write(f"**Confidence:** {confidence.upper()}")

        with col_det2:
            if 'column_analysis' in details and details['column_analysis']:
                st.markdown("**Detection Summary:**")
                if is_coded:
                    st.write(f"- {details['coded_count']}/{len(details['column_analysis'])} columns appear coded (DoE-style)")
                    st.write(f"- Values predominantly near -1, 0, or +1")
                    st.write(f"- Likely from factorial/CCD design matrix")
                else:
                    st.write(f"- {details['natural_count']}/{len(details['column_analysis'])} columns in natural units")
                    st.write(f"- Wide value ranges detected")
                    st.write(f"- Physical/engineering units (e.g., °C, bar, %)")

        # Detailed column analysis
        if 'column_analysis' in details and details['column_analysis']:
            st.divider()
            st.markdown("**Per-Column Analysis:**")

            analysis_df = pd.DataFrame(details['column_analysis'])

            # Color-code by type
            def highlight_type(row):
                if row['Type'] == 'Coded':
                    return ['background-color: #c8e6c9'] * len(row)
                elif row['Type'] == 'Natural':
                    return ['background-color: #bbdefb'] * len(row)
                else:
                    return ['background-color: #fff9c4'] * len(row)

            # Display with styling (note: Streamlit may not support all styling)
            st.dataframe(analysis_df, use_container_width=True, hide_index=True)

            st.caption("""
            **Legend:**
            - **Coded**: DoE-style encoded values (typically -1, 0, +1)
            - **Natural**: Physical/engineering units (temperature, pressure, concentration, etc.)
            - **Uncertain**: Cannot determine with confidence
            """)

        st.divider()

        # Manual override checkbox
        st.markdown("### ⚙️ Manual Override")

        user_override = st.checkbox(
            "Data is already coded (DoE-style)",
            value=is_coded,
            help="Check this if your data is already coded as -1, 0, +1 from a DoE software"
        )

        if user_override != is_coded:
            if user_override:
                st.warning("""
                ⚠️ **You've indicated data IS coded**, but auto-detection suggests otherwise.

                If your data comes from DoE software (e.g., Design-Expert, JMP) and uses
                coded values (-1, 0, +1), this is correct. Otherwise, leave unchecked.
                """)
            else:
                st.info("""
                ℹ️ **You've indicated data is NOT coded**, but auto-detection suggests it might be.

                If your data uses natural units (e.g., 50°C, 2.5 bar, 10%), this is correct.
                """)

        # Store detection result in session state
        st.session_state.data_is_coded = user_override

        st.divider()

        # Factor ranges summary
        st.markdown("### 📏 Factor Ranges (Numeric Variables)")

        if numeric_cols:
            ranges_data = []
            for col in numeric_cols:
                col_data = current_data[col].dropna()
                if len(col_data) > 0:
                    ranges_data.append({
                        'Variable': col,
                        'Min': f"{col_data.min():.4f}",
                        'Max': f"{col_data.max():.4f}",
                        'Mean': f"{col_data.mean():.4f}",
                        'Std Dev': f"{col_data.std():.4f}",
                        'Range': f"{col_data.max() - col_data.min():.4f}"
                    })

            if ranges_data:
                ranges_df = pd.DataFrame(ranges_data)
                st.dataframe(ranges_df, use_container_width=True, hide_index=True)
            else:
                st.warning("No numeric data with valid ranges found")
        else:
            st.warning("No numeric columns detected in dataset")

        # Usage tip
        st.info("""
        **💡 Tip for Bayesian Optimization:**
        - If data is **coded**: You're likely at Round 2/3 of the workflow (ready for BO)
        - If data is **natural units**: Perfect! BO will work directly with physical values
        - **Recommended data size**: 20-30+ observations for reliable GP modeling
        """, icon="💡")

    # ========================================================================
    # CANDIDATE GRID BUILDER (REQUIRED FOR BAYESIAN OPTIMIZATION)
    # ========================================================================
    st.header("📊 Candidate Grid for Bayesian Optimization")
    st.markdown("**Define factor ranges & steps from original data.** BO will rank all candidates from this grid.")

    numeric_cols_grid = current_data.select_dtypes(include=[np.number]).columns.tolist()

    if numeric_cols_grid:
        grid_factors = st.multiselect(
            "Select factors for BO grid:",
            options=numeric_cols_grid,
            default=numeric_cols_grid[:min(3, len(numeric_cols_grid))],
            key="bo_grid_factors_select"
        )

        if grid_factors:
            st.markdown("**Configure ranges & step sizes:**")

            grid_config = {}
            grid_sizes = []

            # Compact UI: 4 columns per row
            for factor in grid_factors:
                col1, col2, col3, col4 = st.columns([2, 1.2, 1.2, 1.2])

                data_min = float(current_data[factor].min())
                data_max = float(current_data[factor].max())
                data_range = data_max - data_min

                # Smart default step
                if data_range > 100:
                    default_step = 10.0
                elif data_range > 10:
                    default_step = 1.0
                elif data_range > 1:
                    default_step = 0.1
                else:
                    default_step = 0.01

                with col1:
                    st.write(f"**{factor}**")

                with col2:
                    min_val = st.number_input(
                        "Min", value=data_min, format="%.4f",
                        key=f"bo_grid_min_{factor}",
                        label_visibility="collapsed"
                    )

                with col3:
                    max_val = st.number_input(
                        "Max", value=data_max, format="%.4f",
                        key=f"bo_grid_max_{factor}",
                        label_visibility="collapsed"
                    )

                with col4:
                    step_val = st.number_input(
                        "Step", value=default_step, min_value=0.0001,
                        format="%.4f",
                        key=f"bo_grid_step_{factor}",
                        label_visibility="collapsed"
                    )

                if step_val > 0 and max_val > min_val:
                    n_levels = int(np.ceil((max_val - min_val) / step_val)) + 1
                    grid_sizes.append(n_levels)
                    grid_config[factor] = {
                        'min': min_val,
                        'max': max_val,
                        'step': step_val,
                        'n_levels': n_levels
                    }

            # Show grid summary
            if grid_config and grid_sizes:
                st.divider()

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Candidates", f"{int(np.prod(grid_sizes)):,}")
                with col2:
                    st.metric("Factors", len(grid_factors))
                with col3:
                    calc_str = " × ".join([str(s) for s in grid_sizes])
                    st.caption(f"Formula: {calc_str}")

                # Store in session for later use
                st.session_state.bo_grid_config = grid_config
                st.session_state.bo_grid_factors = grid_factors
                st.session_state.bo_grid_sizes = grid_sizes
                st.session_state.bo_total_candidates = int(np.prod(grid_sizes))

                st.success(f"✅ Grid configured: {int(np.prod(grid_sizes)):,} candidate points")

    st.divider()

    # ========================================================================
    # DATA TRANSFORMATION FOR BAYESIAN OPTIMIZATION
    # ========================================================================

    st.header("🔄 Data Transformation for Bayesian Optimization")

    st.markdown("""
    **⚠️ MANDATORY:** Bayesian Optimization requires normalized **factors** [-1, +1].
    **Responses stay in original units** for model fitting.
    """)

    # Step 1: Identify response variables
    st.markdown("### 🎯 Step 1: Specify Response Variables")
    st.markdown("Select which variables are **responses** (NOT to be transformed):")

    all_numeric = current_data.select_dtypes(include=[np.number]).columns.tolist()

    response_vars = st.multiselect(
        "Response variables (Y - outcome to optimize):",
        options=all_numeric,
        default=['Conversion_A_', 'DE_HYDROXY_A_'] if 'Conversion_A_' in all_numeric else [],
        key="response_vars_select"
    )

    # Factor variables = all numeric EXCEPT responses
    factor_vars = [col for col in all_numeric if col not in response_vars]

    st.write(f"✅ **Factors to normalize:** {len(factor_vars)} variables")
    st.write(f"✅ **Responses (original units):** {len(response_vars)} variables")

    if not factor_vars:
        st.error("❌ No factors left for BO! Check response variable selection.")
        st.stop()

    st.divider()

    # Step 2: Show transformation
    st.markdown("### 📊 Step 2: Transformation Preview")

    transform_info = []
    for col in factor_vars:
        orig_min = current_data[col].min()
        orig_max = current_data[col].max()
        orig_range = orig_max - orig_min

        transform_info.append({
            'Factor': col,
            'Original Range': f"[{orig_min:.4f}, {orig_max:.4f}]",
            'Range Width': f"{orig_range:.4f}",
            'Coded Range': "[-1.0000, +1.0000]"
        })

    transform_df = pd.DataFrame(transform_info)
    st.dataframe(transform_df, use_container_width=True, hide_index=True)

    st.latex(r"X_{coded} = 2 \times \frac{X_{original} - X_{min}}{X_{max} - X_{min}} - 1")

    st.divider()

    # Step 3: Apply transformation
    st.markdown("### ✅ Step 3: Applying Transformation")

    transformed_data = current_data.copy()

    # Transform ONLY factors
    for col in factor_vars:
        col_min = current_data[col].min()
        col_max = current_data[col].max()
        transformed_data[col] = 2 * (current_data[col] - col_min) / (col_max - col_min) - 1

    # Store metadata
    st.session_state.data_transformed = True
    st.session_state.factor_variables = factor_vars
    st.session_state.response_variables = response_vars
    st.session_state.transform_metadata = {
        'factors': factor_vars,
        'ranges': {col: (float(current_data[col].min()), float(current_data[col].max()))
                  for col in factor_vars}
    }
    st.session_state.transformed_data = transformed_data
    st.session_state.transformation_applied = True

    # Also store metadata in the expected format for inverse transform
    transformation_metadata = {}
    for col in factor_vars:
        orig_min = float(current_data[col].min())
        orig_max = float(current_data[col].max())
        transformation_metadata[col] = {
            'type': 'continuous',
            'method': 'range_11',
            'original_min': orig_min,
            'original_max': orig_max,
            'transformed_min': -1.0,
            'transformed_max': 1.0,
            'inverse_formula': f'x_original = (x_coded + 1) * ({orig_max} - {orig_min}) / 2 + {orig_min}'
        }
    st.session_state.transformation_metadata = transformation_metadata

    st.success("✅ Data transformation applied (factors coded to [-1, +1], responses unchanged)")

    with st.expander("📋 Preview (first 5 rows)"):
        preview_df = transformed_data[factor_vars + response_vars].head(5).copy()

        # Format
        for col in factor_vars:
            preview_df[col] = preview_df[col].round(4)
        for col in response_vars:
            preview_df[col] = preview_df[col].round(4)

        st.dataframe(preview_df, use_container_width=True, hide_index=True)

        st.caption("✓ Factors normalized | Response variables in original units")

    st.info("ℹ️ Using transformed factors for BO. Responses stay original for model calibration.")
    st.divider()

    # Use transformed data for subsequent sections if available
    if 'transformed_data' in st.session_state and st.session_state.get('transformation_applied', False):
        current_data = st.session_state.transformed_data
        st.info("""
        ℹ️ **Using transformed data** for factor selection and optimization below.
        """)

    # ========================================================================
    # SECTION 2 - FACTOR SELECTION (100 lines)
    # ========================================================================

    st.header("1️⃣ Factor Selection & Bounds")

    st.markdown("""
    Select which variables to use as experimental factors and define their valid ranges.
    """)

    # Initialize BO designer for factor detection
    try:
        bo_designer = BayesianOptimizationDesigner(current_data)
        is_valid, validation_msg = bo_designer.validate_workspace_data()

        if not is_valid:
            st.error(f"❌ Data validation failed: {validation_msg}")
            st.stop()
        else:
            st.success(f"✅ {validation_msg}")

        # Auto-detect factors
        detected_factors = bo_designer.detect_experimental_factors()

        with st.expander("🔍 Auto-detected Factors", expanded=True):
            if detected_factors:
                st.write(f"Found **{len(detected_factors)}** potential factors:")
                for i, factor in enumerate(detected_factors, 1):
                    data_min = current_data[factor].min()
                    data_max = current_data[factor].max()
                    data_mean = current_data[factor].mean()
                    st.write(f"{i}. **{factor}**: Range [{data_min:.4f}, {data_max:.4f}], Mean: {data_mean:.4f}")
            else:
                st.warning("No suitable factors detected. Need at least 3 unique values per column.")

        # Manual factor selection
        st.subheader("Select Factors")

        numeric_columns = current_data.select_dtypes(include=[np.number]).columns.tolist()

        if not numeric_columns:
            st.error("❌ No numeric columns found in dataset.")
            st.stop()

        selected_factors = st.multiselect(
            "Choose experimental factors:",
            options=numeric_columns,
            default=detected_factors[:min(3, len(detected_factors))] if detected_factors else [],
            help="Select variables you want to optimize"
        )

        if not selected_factors:
            st.warning("⚠️ Please select at least one factor to proceed.")
            st.stop()

        # SCREENING WARNING for composite/material applications
        if len(selected_factors) > 5:
            st.warning("""
            ⚠️ **You have many factors (>5)**

            **Recommendation:** Use Plackett-Burman screening FIRST:
            1. Go to **MLR/DoE → "Generate Matrix"** tab
            2. Create **12-run Plackett-Burman design**
            3. Run those 12 experiments
            4. Identify **significant factors** (p < 0.05)
            5. Return here with **3-4 significant factors** only

            **Why?** BO works better with fewer factors and sufficient data.
            """)

            st.info(f"""
            **Current status:**
            - Factors: **{len(selected_factors)}**
            - Data points: **{len(current_data)}**
            - Ratio: **{len(current_data)/len(selected_factors):.1f}x** (ideal: 3-5x factors)
            """)

        # Configure bounds for each factor
        st.subheader("Configure Factor Bounds")

        bounds_dict = {}
        bounds_display_data = []

        for factor in selected_factors:
            with st.expander(f"⚙️ {factor}", expanded=False):
                col1, col2, col3 = st.columns([2, 2, 1])

                # Get data statistics for suggestions
                data_min = float(current_data[factor].min())
                data_max = float(current_data[factor].max())
                data_range = data_max - data_min

                # Suggest bounds with some margin
                suggested_lower = data_min - 0.1 * data_range
                suggested_upper = data_max + 0.1 * data_range

                with col1:
                    lower_bound = st.number_input(
                        f"Lower Bound",
                        value=float(suggested_lower),
                        key=f"lower_{factor}",
                        format="%.4f"
                    )

                with col2:
                    upper_bound = st.number_input(
                        f"Upper Bound",
                        value=float(suggested_upper),
                        key=f"upper_{factor}",
                        format="%.4f"
                    )

                with col3:
                    factor_type = st.selectbox(
                        "Type",
                        options=["continuous", "discrete"],
                        key=f"type_{factor}"
                    )

                # Step for discrete
                step_value = None
                if factor_type == "discrete":
                    step_value = st.number_input(
                        "Step size",
                        value=1.0,
                        min_value=0.001,
                        key=f"step_{factor}",
                        format="%.4f"
                    )

                # Store bounds
                bounds_dict[factor] = {
                    'lower': lower_bound,
                    'upper': upper_bound,
                    'type': factor_type,
                    'step': step_value
                }

                # For display table
                bounds_display_data.append({
                    'Factor': factor,
                    'Lower': f"{lower_bound:.4f}",
                    'Upper': f"{upper_bound:.4f}",
                    'Type': factor_type,
                    'Range': f"{upper_bound - lower_bound:.4f}"
                })

        # Display bounds summary
        st.write("**Bounds Summary:**")
        bounds_df = pd.DataFrame(bounds_display_data)
        st.dataframe(bounds_df, use_container_width=True, hide_index=True)

        # Validate bounds
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("✅ Validate Bounds", type="secondary"):
                is_valid, msg = validate_bounds(bounds_dict, current_data)
                if is_valid:
                    st.success("✓ All bounds valid!")
                    if "warning" in msg.lower():
                        st.warning(msg)
                else:
                    st.error(msg)

        # Store in session state
        st.session_state.bo_bounds = bounds_dict
        st.session_state.bo_factors = selected_factors

        # Preview Candidate Grid
        st.divider()

        if st.button("🔍 Preview Candidate Grid", use_container_width=True):
            st.markdown("### 📊 Candidate Grid Preview")

            # Prepare bounds data for grid generation
            factor_names = list(bounds_dict.keys())
            natural_grids = []

            for factor in factor_names:
                bound_info = bounds_dict[factor]
                lower = bound_info['lower']
                upper = bound_info['upper']
                factor_type = bound_info['type']
                step = bound_info.get('step')

                # Generate grid points based on type
                if factor_type == 'discrete' and step is not None and step > 0:
                    # Discrete: use step
                    grid_points = np.arange(lower, upper + step/2, step)
                    grid_points = np.clip(grid_points, lower, upper)
                else:
                    # Continuous: use 10 points as default
                    grid_points = np.linspace(lower, upper, 10)

                natural_grids.append(grid_points)

            # Create meshgrid
            grids = np.meshgrid(*natural_grids, indexing='ij')
            candidates_natural = np.column_stack([g.ravel() for g in grids])

            n_total = len(candidates_natural)

            # Display metrics
            col_info1, col_info2, col_info3 = st.columns(3)
            with col_info1:
                st.metric("Total Candidates", f"{n_total:,}")
            with col_info2:
                st.metric("Factors", len(bounds_dict))
            with col_info3:
                st.metric("To Suggest", 5)

            # Preview first 20 candidates
            st.markdown("**First 20 candidate combinations (preview):**")
            preview_df = pd.DataFrame(candidates_natural[:20], columns=factor_names)
            # Round for display
            for col in preview_df.columns:
                preview_df[col] = preview_df[col].round(4)
            st.dataframe(preview_df, use_container_width=True, hide_index=True)

            if n_total > 20:
                st.info(f"Showing 20 of {n_total:,} total candidates...")

            # Store grid in session state
            st.session_state.candidate_grid_natural = candidates_natural
            st.session_state.candidate_factor_names = factor_names
            st.session_state.n_candidates = n_total

            st.success(f"✅ Grid ready! Will rank all {n_total:,} candidates in BO.")

    except Exception as e:
        st.error(f"❌ Error in factor selection: {str(e)}")
        st.stop()

    # ========================================================================
    # SECTION 3 - TARGET SELECTION (50 lines)
    # ========================================================================

    st.header("2️⃣ Target Response Selection")

    st.markdown("""
    Select the response variable you want to optimize.
    """)

    # Get numeric columns excluding factors
    available_targets = [col for col in numeric_columns if col not in selected_factors]

    if not available_targets:
        st.error("❌ No available target columns (all numeric columns are selected as factors)")
        st.stop()

    col1, col2 = st.columns(2)

    with col1:
        target_column = st.selectbox(
            "Select target response:",
            options=available_targets,
            help="The variable you want to maximize or minimize"
        )

    with col2:
        optimization_direction = st.radio(
            "Optimization goal:",
            options=["Maximize", "Minimize"],
            horizontal=True,
            help="Whether to find maximum or minimum response"
        )

    maximize = (optimization_direction == "Maximize")

    # Display target statistics
    target_stats = current_data[target_column].describe()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Mean", f"{target_stats['mean']:.4f}")
    with col2:
        st.metric("Std Dev", f"{target_stats['std']:.4f}")
    with col3:
        st.metric("Min", f"{target_stats['min']:.4f}")
    with col4:
        st.metric("Max", f"{target_stats['max']:.4f}")

    # Store in session state
    st.session_state.bo_target = target_column
    st.session_state.bo_maximize = maximize

    # ========================================================================
    # SECTION 4 - GP CONFIG (60 lines)
    # ========================================================================

    st.header("3️⃣ Gaussian Process Configuration")

    with st.expander("⚙️ GP Model Settings", expanded=True):
        st.markdown("""
        Configure the Gaussian Process surrogate model and acquisition function.
        """)

        col1, col2 = st.columns(2)

        with col1:
            kernel_type = st.selectbox(
                "Kernel Type:",
                options=["Matern52", "Matern32", "RBF"],
                help="""
                - **Matern52**: Smooth, twice differentiable (recommended)
                - **Matern32**: Less smooth, once differentiable
                - **RBF**: Infinitely smooth (may overfit)
                """
            )

        with col2:
            acquisition_type = st.selectbox(
                "Acquisition Function:",
                options=["EI", "LCB"],
                help="""
                - **EI**: Expected Improvement (balanced)
                - **LCB**: Lower Confidence Bound (pessimistic)
                """
            )

        # Acquisition parameters
        st.subheader("Acquisition Parameters")

        if acquisition_type == "EI":
            jitter = st.slider(
                "Jitter (ξ):",
                min_value=0.0,
                max_value=0.1,
                value=0.01,
                step=0.001,
                format="%.3f",
                help="Exploration parameter. Higher values encourage more exploration."
            )
            weight = None
            st.session_state.bo_jitter = jitter
        else:  # LCB
            weight = st.slider(
                "Weight (κ):",
                min_value=0.5,
                max_value=5.0,
                value=2.0,
                step=0.1,
                format="%.1f",
                help="Exploration-exploitation trade-off. Higher values → more exploration."
            )
            jitter = None
            st.session_state.bo_weight = weight

        # Store config
        st.session_state.bo_kernel = kernel_type
        st.session_state.bo_acquisition = acquisition_type

    st.info("""
    **💡 Tips:**
    - Start with **Matern52** kernel and **EI** acquisition
    - Use higher jitter/weight if optimization seems stuck in local optima
    - Lower jitter/weight for fine-tuning near known good regions
    """)

    # ========================================================================
    # SECTION 5 - EXECUTION (80 lines)
    # ========================================================================

    st.header("4️⃣ Run Optimization")

    st.markdown("""
    **Grid-based Bayesian Optimization** ranks all candidate points from your custom grid.
    This approach is **more robust** and produces **diverse suggestions** compared to gradient-based methods.
    """)

    st.info("""
    **✅ Recommended Strategy:**
    1. Define candidate grid (ranges + steps) in section above
    2. Click "Run BO on Feasible Grid" below
    3. BO will rank ALL candidates using acquisition function
    4. Get top N diverse experimental suggestions
    """, icon="💡")

    col1, col2 = st.columns(2)

    with col1:
        batch_size = st.slider(
            "Batch Size:",
            min_value=1,
            max_value=20,
            value=5,
            help="Number of new experimental points to suggest"
        )

    with col2:
        st.metric("Selected Factors", len(selected_factors))

    st.divider()

    # Check if candidate grid exists
    if 'candidate_grid_natural' not in st.session_state or st.session_state.candidate_grid_natural is None:
        st.warning("""
        ⚠️ **Candidate Grid Required**

        Please generate a candidate grid first:
        1. Scroll to **"1️⃣ Factor Selection & Bounds"** section above
        2. Click **"Preview Candidate Grid"** button
        3. Grid will be generated and stored
        4. Return here to run optimization
        """, icon="⚠️")
        st.stop()

    # Grid is available - proceed
    st.success(f"""
    ✅ **Candidate Grid Ready**

    Grid contains **{st.session_state.n_candidates:,} candidate points**.
    BO will rank all candidates and suggest the top {batch_size} experiments.
    """)

    if st.button("🚀 Run Bayesian Optimization", type="primary", use_container_width=True):
        with st.spinner(f"🔄 Evaluating {st.session_state.n_candidates:,} candidates..."):
            try:
                # Initialize BO Designer
                bo_manager = BayesianOptimizationDesigner(current_data)

                # Set factor bounds
                for factor, bounds in bounds_dict.items():
                    bo_manager.set_factor_bounds(
                        factor,
                        bounds['lower'],
                        bounds['upper'],
                        bounds['type'],
                        bounds.get('step')
                    )

                # Fit Gaussian Process
                st.info(f"📊 Fitting GP model with {kernel_type} kernel...")
                success = bo_manager.fit_gaussian_process_from_data(
                    X_cols=selected_factors,
                    y_col=target_column,
                    kernel_type=kernel_type
                )

                if not success:
                    st.error("❌ GP fitting failed")
                    st.stop()

                st.success(f"✅ GP model fitted (R² = {bo_manager.gp_score:.4f})")

                # Run BO on pre-computed grid
                st.info(f"🎯 Ranking {st.session_state.n_candidates:,} candidates with {acquisition_type} acquisition...")

                suggested_points = bo_manager.run_bayesian_optimization_on_grid(
                    candidate_grid=st.session_state.candidate_grid_natural,
                    batch_size=batch_size,
                    acquisition=acquisition_type,
                    maximize=maximize,
                    jitter=jitter if acquisition_type == "EI" else weight
                )

                # Store results in session state
                st.session_state.bo_manager = bo_manager
                st.session_state.bo_suggested = suggested_points

                st.success(f"""
                ✅ **Grid-based Optimization Complete!**

                Evaluated **{st.session_state.n_candidates:,}** candidates.
                Suggested **{len(suggested_points)}** best experimental points.
                Scroll down to view results and visualizations.
                """)

                # Check for BO warnings
                if hasattr(bo_manager, 'bo_warning') and bo_manager.bo_warning is not None:
                    warning_info = bo_manager.bo_warning
                    st.warning(f"⚠️ {warning_info['message']}", icon="⚠️")

                    with st.expander("🔍 Why is this happening?"):
                        st.markdown("**Possible causes:**")
                        for cause in warning_info['causes']:
                            st.write(f"- {cause}")

                        st.markdown("**Data diagnostics:**")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Training Samples", warning_info['n_training_samples'])
                        with col2:
                            st.metric("N Factors", warning_info['n_factors'])
                        with col3:
                            st.metric("Candidates Evaluated", warning_info.get('n_candidates_evaluated', 'N/A'))

                        st.markdown("**Recommendations:**")
                        for rec in warning_info['recommendations']:
                            st.write(f"✓ {rec}")

            except Exception as e:
                st.error(f"❌ Grid-based optimization failed: {str(e)}")
                import traceback
                with st.expander("🔍 Error Details"):
                    st.code(traceback.format_exc())

    # ========================================================================
    # SECTION 6 - RESULTS (80 lines)
    # ========================================================================

    if 'bo_suggested' not in st.session_state or st.session_state.bo_suggested is None:
        st.info("👆 Configure parameters above and click **Run Bayesian Optimization** to see results.")
        return

    st.header("5️⃣ Optimization Results")

    suggested_df = st.session_state.bo_suggested
    bo_manager = st.session_state.bo_manager

    # Format results for display
    try:
        formatted_df = format_results_display(
            suggested_df.copy(),
            factor_names=selected_factors
        )
    except:
        formatted_df = suggested_df.copy()
        # Round numeric columns
        numeric_cols = formatted_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            formatted_df[col] = formatted_df[col].round(4)

    # Display results
    st.dataframe(
        formatted_df,
        use_container_width=True,
        hide_index=True
    )

    # Summary metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Points Suggested", len(suggested_df))

    with col2:
        if 'Expected_Target' in suggested_df.columns:
            best_target = suggested_df['Expected_Target'].max() if maximize else suggested_df['Expected_Target'].min()
            st.metric(
                "Best Expected Target",
                f"{best_target:.4f}",
                delta=None
            )
        elif 'Predicted Response' in formatted_df.columns:
            best_target = formatted_df['Predicted Response'].max() if maximize else formatted_df['Predicted Response'].min()
            st.metric(
                "Best Expected Target",
                f"{best_target:.4f}",
                delta=None
            )

    with col3:
        if 'Acquisition_Value' in suggested_df.columns:
            avg_acq = suggested_df['Acquisition_Value'].mean()
            st.metric("Avg Acquisition", f"{avg_acq:.4f}")
        elif 'Acquisition Score' in formatted_df.columns:
            avg_acq = formatted_df['Acquisition Score'].mean()
            st.metric("Avg Acquisition", f"{avg_acq:.4f}")

    # ========================================================================
    # SUGGESTED EXPERIMENTS IN REAL UNITS
    # ========================================================================

    st.markdown("---")
    st.markdown("### 📋 Suggested Experiments (Real Units)")

    # Build original_bounds from current_data (original, untransformed)
    # We need to get the ORIGINAL data before any transformations
    original_data = st.session_state.get('current_data', current_data)

    # Build bounds dictionary
    original_bounds = {}
    for factor in selected_factors:
        if factor in original_data.columns:
            try:
                original_bounds[factor] = {
                    'original_min': float(original_data[factor].min()),
                    'original_max': float(original_data[factor].max())
                }
            except:
                # If factor doesn't exist in original data, use current data
                if factor in current_data.columns:
                    original_bounds[factor] = {
                        'original_min': float(current_data[factor].min()),
                        'original_max': float(current_data[factor].max())
                    }

    # Check if data was transformed
    transformation_applied = st.session_state.get('transformation_applied', False)

    if transformation_applied and original_bounds:
        # Data is coded - convert back to real units
        try:
            # Extract factor columns from suggestions
            suggestions_factors_coded = suggested_df[selected_factors].copy()

            # Use inverse_transform_predictions if available
            real_suggestions = inverse_transform_predictions(
                suggestions_factors_coded,
                original_bounds
            )

            st.caption("✓ Values converted from coded [-1, +1] to original measurement units")

        except Exception as e:
            # Fallback: manual conversion
            real_suggestions = suggestions_factors_coded.copy()

            for factor in selected_factors:
                if factor in original_bounds:
                    min_val = original_bounds[factor]['original_min']
                    max_val = original_bounds[factor]['original_max']

                    # Inverse formula: real = (coded + 1) * (max - min) / 2 + min
                    real_suggestions[factor] = (
                        (real_suggestions[factor] + 1) * (max_val - min_val) / 2 + min_val
                    )

            st.caption(f"✓ Values converted using inverse transformation (fallback method)")
    else:
        # Data is NOT coded - values are already in real units
        real_suggestions = suggested_df[selected_factors].copy()
        st.caption("ℹ️ Values are already in original measurement units (no transformation was applied)")

    # Build display dataframe
    # Add Rank column if exists
    display_cols = []

    if 'Rank' in suggested_df.columns:
        display_cols.append('Rank')

    # Combine: Rank | Factor columns (real units) | Predicted Response | Uncertainty | Acquisition Score
    real_suggestions_display = pd.DataFrame()

    # Add Rank
    if 'Rank' in suggested_df.columns:
        real_suggestions_display['Rank'] = suggested_df['Rank'].values

    # Add real factor values
    for factor in selected_factors:
        if factor in real_suggestions.columns:
            real_suggestions_display[factor] = real_suggestions[factor].values

    # Add prediction metrics
    if 'Expected_Target' in suggested_df.columns:
        real_suggestions_display['Predicted Response'] = suggested_df['Expected_Target'].values
    elif 'Predicted Response' in formatted_df.columns:
        real_suggestions_display['Predicted Response'] = formatted_df['Predicted Response'].values

    if 'Uncertainty' in suggested_df.columns:
        real_suggestions_display['Uncertainty (±)'] = suggested_df['Uncertainty'].values
    elif 'Uncertainty (±)' in formatted_df.columns:
        real_suggestions_display['Uncertainty (±)'] = formatted_df['Uncertainty (±)'].values

    if 'Acquisition_Value' in suggested_df.columns:
        real_suggestions_display['Acquisition Score'] = suggested_df['Acquisition_Value'].values
    elif 'Acquisition Score' in formatted_df.columns:
        real_suggestions_display['Acquisition Score'] = formatted_df['Acquisition Score'].values

    # Format display - round numeric columns
    for col in real_suggestions_display.columns:
        if col not in ['Rank']:
            try:
                real_suggestions_display[col] = real_suggestions_display[col].round(4)
            except:
                pass

    # Display table
    st.dataframe(real_suggestions_display, use_container_width=True, hide_index=True)

    # Download button
    st.markdown("---")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"bo_suggested_experiments_{timestamp}.csv"

    csv_data = suggested_df.to_csv(index=False)

    st.download_button(
        label="📥 Download Suggested Experiments (CSV)",
        data=csv_data,
        file_name=csv_filename,
        mime="text/csv",
        use_container_width=True
    )

    # ========================================================================
    # SECTION 7 - VISUALIZATIONS (150 lines)
    # ========================================================================

    st.header("6️⃣ Acquisition Landscape Visualization")

    tab1, tab2, tab3 = st.tabs(["📈 1D Analysis", "🗺️ 2D Analysis", "📊 Summary"])

    # --- TAB 1: 1D Analysis ---
    with tab1:
        st.subheader("1D Acquisition Landscape")

        st.markdown("""
        Visualize how the GP model predicts the response across the range of a single factor,
        with other factors held at their center values.
        """)

        if len(selected_factors) < 1:
            st.warning("Need at least 1 factor for 1D visualization")
        else:
            # Factor selection
            selected_factor_1d = st.selectbox(
                "Select factor to visualize:",
                options=selected_factors,
                key="viz_1d_factor"
            )

            factor_idx = selected_factors.index(selected_factor_1d)

            # Resolution slider
            resolution = st.slider(
                "Resolution (number of points):",
                min_value=50,
                max_value=200,
                value=100,
                step=10,
                key="viz_1d_resolution"
            )

            # Generate plot
            try:
                with st.spinner("Generating 1D plot..."):
                    fig_1d = bo_manager.visualize_acquisition_1d(
                        factor_idx=factor_idx,
                        resolution=resolution
                    )
                    st.plotly_chart(fig_1d, use_container_width=True)

                st.info("""
                **📖 Interpretation:**
                - **Green line**: GP mean prediction
                - **Orange band**: Uncertainty (±1 std dev)
                - **Blue dots**: Observed data points
                - **Red line**: Suggested point
                """)
            except Exception as e:
                st.error(f"Error generating 1D plot: {str(e)}")

    # --- TAB 2: 2D Analysis ---
    with tab2:
        st.subheader("2D Acquisition Landscape")

        st.markdown("""
        Visualize the interaction between two factors and their effect on the predicted response.
        """)

        if len(selected_factors) < 2:
            st.warning("Need at least 2 factors for 2D visualization")
        else:
            col1, col2, col3 = st.columns(3)

            with col1:
                factor1_2d = st.selectbox(
                    "Factor 1 (X-axis):",
                    options=selected_factors,
                    key="viz_2d_factor1"
                )

            with col2:
                # Ensure factor 2 is different from factor 1
                factor2_options = [f for f in selected_factors if f != factor1_2d]
                factor2_2d = st.selectbox(
                    "Factor 2 (Y-axis):",
                    options=factor2_options,
                    key="viz_2d_factor2"
                )

            with col3:
                plot_type_2d = st.radio(
                    "Plot Type:",
                    options=["heatmap", "surface"],
                    key="viz_2d_type"
                )

            factor1_idx = selected_factors.index(factor1_2d)
            factor2_idx = selected_factors.index(factor2_2d)

            # Generate plot
            try:
                with st.spinner("Generating 2D plot..."):
                    fig_2d = bo_manager.visualize_acquisition_2d(
                        factor1_idx=factor1_idx,
                        factor2_idx=factor2_idx,
                        plot_type=plot_type_2d
                    )
                    st.plotly_chart(fig_2d, use_container_width=True)

                st.info("""
                **📖 Interpretation:**
                - **Color intensity**: Predicted response value
                - **White X markers**: Observed data points (if applicable)
                - Look for regions of high/low response based on optimization goal
                """)
            except Exception as e:
                st.error(f"Error generating 2D plot: {str(e)}")

    # --- TAB 3: Summary ---
    with tab3:
        st.subheader("Optimization Summary")

        try:
            summary = bo_manager.get_optimization_summary()

            # Display key metrics
            col1, col2 = st.columns(2)

            with col1:
                st.write("**Model Information:**")
                st.write(f"- Factors: {summary['n_factors']}")
                st.write(f"- Training observations: {summary['n_existing_experiments']}")
                st.write(f"- GP R² Score: {summary['gp_model_score']:.4f}")
                st.write(f"- Target: {summary['y_col']}")

            with col2:
                st.write("**Optimization Results:**")
                st.write(f"- Suggested points: {summary['n_suggested_points']}")
                st.write(f"- Kernel: {kernel_type}")
                st.write(f"- Acquisition: {acquisition_type}")
                st.write(f"- Direction: {'Maximize' if maximize else 'Minimize'}")

            # Factor bounds
            if 'bounds_summary' in summary:
                st.write("**Factor Bounds:**")
                bounds_summary_df = pd.DataFrame([
                    {'Factor': k, 'Range': v}
                    for k, v in summary['bounds_summary'].items()
                ])
                st.dataframe(bounds_summary_df, use_container_width=True, hide_index=True)

            # Optimization history
            if summary['optimization_history']:
                st.write("**Optimization History:**")
                history_df = pd.DataFrame(summary['optimization_history'])
                st.dataframe(history_df, use_container_width=True, hide_index=True)

        except Exception as e:
            st.error(f"Error displaying summary: {str(e)}")

    # ========================================================================
    # SECTION 8 - ITERATIVE REFINEMENT (100 lines)
    # ========================================================================

    st.header("7️⃣ Iterative Refinement")

    with st.expander("📤 Upload Experimental Results & Continue Optimization", expanded=False):
        st.markdown("""
        After running the suggested experiments, upload the results here to update
        the GP model and suggest the next batch of experiments.
        """)

        st.info("""
        **Workflow:**
        1. Download suggested experiments (above)
        2. Run the experiments in your lab
        3. Add the measured response values to the CSV
        4. Upload the completed CSV here
        5. The GP model will be updated with new data
        6. Run optimization again to get next batch
        """)

        # File uploader
        uploaded_file = st.file_uploader(
            "Upload experimental results (CSV):",
            type=['csv'],
            help="CSV must contain factor columns and measured response",
            key="bo_upload_results"
        )

        if uploaded_file is not None:
            try:
                # Read uploaded data
                uploaded_df = pd.read_csv(uploaded_file)

                st.write("**Preview of uploaded data:**")
                st.dataframe(uploaded_df.head(10), use_container_width=True)

                # Check columns
                st.write(f"Columns found: {list(uploaded_df.columns)}")

                # Select target column from uploaded data
                available_response_cols = [col for col in uploaded_df.columns
                                          if col in numeric_columns or col == target_column]

                if not available_response_cols:
                    st.error("❌ No valid response columns found in uploaded file")
                else:
                    response_col = st.selectbox(
                        "Select response column:",
                        options=available_response_cols,
                        key="upload_response_col"
                    )

                    # Verify factor columns exist
                    missing_factors = [f for f in selected_factors if f not in uploaded_df.columns]
                    if missing_factors:
                        st.error(f"❌ Missing factor columns: {missing_factors}")
                    else:
                        # Button to import and update
                        if st.button("📥 Import & Update GP Model", type="primary"):
                            with st.spinner("Updating GP model with new data..."):
                                try:
                                    # Combine original data with new results
                                    combined_data = pd.concat([
                                        current_data,
                                        uploaded_df
                                    ], ignore_index=True)

                                    # Remove duplicates if any
                                    combined_data = combined_data.drop_duplicates()

                                    st.write(f"Combined dataset: {len(combined_data)} observations")

                                    # Create new BO manager with combined data
                                    updated_bo_manager = BayesianOptimizationDesigner(combined_data)

                                    # Set bounds
                                    for factor, bounds in bounds_dict.items():
                                        updated_bo_manager.set_factor_bounds(
                                            factor,
                                            bounds['lower'],
                                            bounds['upper'],
                                            bounds['type'],
                                            bounds.get('step')
                                        )

                                    # Refit GP
                                    success = updated_bo_manager.fit_gaussian_process_from_data(
                                        X_cols=selected_factors,
                                        y_col=response_col,
                                        kernel_type=kernel_type
                                    )

                                    if success:
                                        # Update session state
                                        st.session_state.current_data = combined_data
                                        st.session_state.bo_manager = updated_bo_manager

                                        new_score = updated_bo_manager.gp_score
                                        old_score = bo_manager.gp_score

                                        st.success(f"""
                                        ✅ **GP Model Updated Successfully!**

                                        - New observations: {len(uploaded_df)}
                                        - Total observations: {len(combined_data)}
                                        - Updated GP R² score: {new_score:.4f}
                                        - Previous score: {old_score:.4f}
                                        - Change: {(new_score - old_score):.4f}

                                        You can now run optimization again to get the next batch of suggestions.
                                        """)

                                        # Offer to re-run optimization
                                        if st.button("🔄 Re-run Optimization with Updated Model"):
                                            st.rerun()
                                    else:
                                        st.error("❌ Failed to refit GP model")

                                except Exception as e:
                                    st.error(f"❌ Error updating model: {str(e)}")
                                    import traceback
                                    st.markdown("**🔍 Error Details:**")
                                    st.code(traceback.format_exc())

            except Exception as e:
                st.error(f"❌ Error reading uploaded file: {str(e)}")

    # ========================================================================
    # SECTION 9 - VALIDATION MODE (150 lines)
    # ========================================================================

    st.header("8️⃣ Validation Mode (Testing on Historical Data)")

    with st.expander("🧪 Validate BO on existing datasets", expanded=False):
        st.info("""
        **Test BO performance on historical data before running real experiments.**

        This mode splits your existing data into training and test sets, fits the GP model
        on training data only, and validates predictions on the held-out test set.
        """)

        st.markdown("""
        **Why validate?**
        - Assess if your data is sufficient for reliable BO predictions
        - Check if GP model generalizes well (not overfitting)
        - Identify if you need more data before suggesting new experiments
        - Verify that selected factors have predictive power
        """)

        # Train/Test split configuration
        st.subheader("Configure Train/Test Split")

        col_split1, col_split2 = st.columns(2)
        with col_split1:
            train_ratio = st.slider(
                "Training set %",
                min_value=50,
                max_value=90,
                value=70,
                step=5,
                help="Percentage of data to use for training. Rest is used for testing."
            )
        with col_split2:
            random_seed = st.number_input(
                "Random seed",
                min_value=0,
                max_value=9999,
                value=42,
                help="Set random seed for reproducible splits"
            )

        st.info(f"""
        **Current split:**
        - Training: {int(len(current_data) * train_ratio / 100)} observations ({train_ratio}%)
        - Testing: {int(len(current_data) * (100 - train_ratio) / 100)} observations ({100 - train_ratio}%)
        """)

        # Validation button
        if st.button("🧪 Split Data & Validate BO", type="primary", use_container_width=True):
            with st.spinner("🔄 Splitting data and validating GP model..."):
                try:
                    from sklearn.model_selection import train_test_split

                    # Extract features and target
                    X_all = current_data[selected_factors].values
                    y_all = current_data[target_column].values

                    # Split data
                    X_train, X_test, y_train, y_test = train_test_split(
                        X_all, y_all,
                        train_size=train_ratio/100,
                        random_state=random_seed
                    )

                    st.write(f"✅ Data split: {len(X_train)} training, {len(X_test)} test observations")

                    # Create training dataframe
                    train_df = pd.DataFrame(X_train, columns=selected_factors)
                    train_df[target_column] = y_train

                    # Initialize BO Designer with training data only
                    bo_val = BayesianOptimizationDesigner(train_df)

                    # Set bounds from training data
                    for i, factor in enumerate(selected_factors):
                        lower = float(X_train[:, i].min())
                        upper = float(X_train[:, i].max())
                        bo_val.set_factor_bounds(factor, lower, upper, "continuous")

                    st.write("✅ Factor bounds configured from training data")

                    # Fit GP on training data
                    success = bo_val.fit_gaussian_process_from_data(
                        X_cols=selected_factors,
                        y_col=target_column,
                        kernel_type=kernel_type
                    )

                    if not success:
                        st.error("❌ GP fitting failed on training data")
                        st.stop()

                    st.write(f"✅ GP model fitted on training data (R² train = {bo_val.gp_score:.4f})")

                    # Predict on test set
                    y_pred_test, y_std_test = bo_val.gp_model.predict(X_test, return_std=True)

                    # Calculate validation metrics
                    mae = np.mean(np.abs(y_test - y_pred_test))
                    rmse = np.sqrt(np.mean((y_test - y_pred_test)**2))
                    r2_test = bo_val.gp_model.score(X_test, y_test)

                    # Also calculate metrics on training set for comparison
                    y_pred_train = bo_val.gp_model.predict(X_train)
                    mae_train = np.mean(np.abs(y_train - y_pred_train))
                    rmse_train = np.sqrt(np.mean((y_train - y_pred_train)**2))

                    # Display validation metrics
                    st.markdown("---")
                    st.subheader("📊 Validation Metrics")

                    col_m1, col_m2, col_m3 = st.columns(3)
                    with col_m1:
                        st.metric(
                            "MAE (Test)",
                            f"{mae:.4f}",
                            delta=f"{mae - mae_train:.4f} vs train",
                            delta_color="inverse"
                        )
                    with col_m2:
                        st.metric(
                            "RMSE (Test)",
                            f"{rmse:.4f}",
                            delta=f"{rmse - rmse_train:.4f} vs train",
                            delta_color="inverse"
                        )
                    with col_m3:
                        st.metric(
                            "R² (Test)",
                            f"{r2_test:.4f}",
                            delta=f"{r2_test - bo_val.gp_score:.4f} vs train"
                        )

                    # Interpretation
                    st.markdown("---")
                    st.subheader("📖 Interpretation")

                    if r2_test > 0.8:
                        st.success("""
                        ✅ **Excellent validation performance (R² > 0.8)**

                        Your BO model is highly reliable:
                        - GP predictions generalize well to unseen data
                        - You can confidently use BO to suggest new experiments
                        - The selected factors have strong predictive power
                        """)
                    elif r2_test >= 0.6:
                        st.warning("""
                        ⚠️ **Acceptable validation performance (R² 0.6-0.8)**

                        BO can be used with caution:
                        - GP model has moderate predictive power
                        - Consider collecting more data for better reliability
                        - Use BO suggestions as guidance, not absolute truth
                        - Verify suggested experiments are physically meaningful
                        """)
                    else:
                        st.error("""
                        ❌ **Poor validation performance (R² < 0.6)**

                        BO model needs improvement:
                        - Insufficient predictive power on test data
                        - **DO NOT trust BO suggestions yet**

                        **Recommendations:**
                        - Collect more experimental data (aim for 5-10x number of factors)
                        - Use Plackett-Burman to screen factors first (reduce dimensionality)
                        - Check for data quality issues (outliers, measurement errors)
                        - Consider if selected factors truly affect the response
                        """)

                    # Overfitting check
                    if bo_val.gp_score - r2_test > 0.2:
                        st.warning("""
                        ⚠️ **Potential overfitting detected!**

                        Training R² is much higher than test R² (difference > 0.2).
                        This suggests the model memorizes training data but doesn't generalize well.

                        **Solutions:**
                        - Collect more training data
                        - Simplify model (try different kernel)
                        - Check for outliers in training data
                        """)

                    # Plot: Predicted vs Actual
                    st.markdown("---")
                    st.subheader("📈 Predicted vs Actual (Test Set)")

                    fig_val = go.Figure()

                    # Scatter plot with error bars (uncertainty)
                    fig_val.add_trace(go.Scatter(
                        x=y_test,
                        y=y_pred_test,
                        error_y=dict(
                            type='data',
                            array=y_std_test,
                            visible=True,
                            color='lightblue'
                        ),
                        mode='markers',
                        name='Test Predictions',
                        marker=dict(size=10, color='blue', opacity=0.6),
                        text=[f"Actual: {yt:.3f}<br>Predicted: {yp:.3f}<br>Std: {ys:.3f}"
                              for yt, yp, ys in zip(y_test, y_pred_test, y_std_test)],
                        hovertemplate='%{text}<extra></extra>'
                    ))

                    # Perfect prediction line (diagonal)
                    min_y = min(y_test.min(), y_pred_test.min())
                    max_y = max(y_test.max(), y_pred_test.max())
                    fig_val.add_trace(go.Scatter(
                        x=[min_y, max_y],
                        y=[min_y, max_y],
                        mode='lines',
                        name='Perfect Prediction',
                        line=dict(dash='dash', color='red', width=2)
                    ))

                    fig_val.update_layout(
                        title=f"BO Validation: Predicted vs Actual (R² = {r2_test:.4f})",
                        xaxis_title=f"Actual {target_column}",
                        yaxis_title=f"Predicted {target_column}",
                        template='plotly_white',
                        hovermode='closest',
                        height=500
                    )

                    st.plotly_chart(fig_val, use_container_width=True)

                    st.info("""
                    **How to read this plot:**
                    - **Points on the red diagonal**: Perfect predictions ✓
                    - **Points far from diagonal**: Poor predictions ❌
                    - **Error bars**: GP uncertainty (wider = less confident)
                    - **Clustered points near diagonal**: Model generalizes well
                    - **Scattered points**: Model struggles to predict
                    """)

                    # Residuals plot
                    st.markdown("---")
                    st.subheader("📉 Residuals Analysis")

                    residuals = y_test - y_pred_test

                    fig_residuals = go.Figure()

                    fig_residuals.add_trace(go.Scatter(
                        x=y_pred_test,
                        y=residuals,
                        mode='markers',
                        name='Residuals',
                        marker=dict(size=10, color='green', opacity=0.6),
                        text=[f"Predicted: {yp:.3f}<br>Residual: {r:.3f}"
                              for yp, r in zip(y_pred_test, residuals)],
                        hovertemplate='%{text}<extra></extra>'
                    ))

                    # Zero line
                    fig_residuals.add_trace(go.Scatter(
                        x=[y_pred_test.min(), y_pred_test.max()],
                        y=[0, 0],
                        mode='lines',
                        name='Zero',
                        line=dict(dash='dash', color='red', width=2)
                    ))

                    fig_residuals.update_layout(
                        title="Residuals vs Predicted Values",
                        xaxis_title=f"Predicted {target_column}",
                        yaxis_title="Residual (Actual - Predicted)",
                        template='plotly_white',
                        hovermode='closest',
                        height=400
                    )

                    st.plotly_chart(fig_residuals, use_container_width=True)

                    st.info("""
                    **Residuals interpretation:**
                    - **Random scatter around zero**: Good model ✓
                    - **Patterns or trends**: Model bias (systematic errors) ❌
                    - **Funnel shape**: Heteroscedasticity (variance changes with prediction level)
                    """)

                    # Summary table
                    st.markdown("---")
                    st.subheader("📋 Validation Summary Table")

                    summary_table = pd.DataFrame({
                        'Metric': ['MAE', 'RMSE', 'R²', 'Training Size', 'Test Size', 'Kernel', 'Factors'],
                        'Training': [f"{mae_train:.4f}", f"{rmse_train:.4f}", f"{bo_val.gp_score:.4f}",
                                    len(X_train), '-', kernel_type, len(selected_factors)],
                        'Testing': [f"{mae:.4f}", f"{rmse:.4f}", f"{r2_test:.4f}",
                                   '-', len(X_test), '-', '-']
                    })

                    st.dataframe(summary_table, use_container_width=True, hide_index=True)

                    st.success("""
                    ✅ **Validation complete!**

                    Review the metrics and plots above to decide if your BO model is ready for
                    suggesting new experiments.
                    """)

                except Exception as e:
                    st.error(f"❌ Validation failed: {str(e)}")
                    import traceback
                    st.markdown("**🔍 Error Details:**")
                    st.code(traceback.format_exc())

    # Footer
    st.markdown("---")
    st.markdown("""
    **🔬 ChemoMetric Solutions** | Bayesian Optimization Module v1.0

    *For questions or support, refer to the documentation or contact your system administrator.*
    """)


# Entry point for testing
if __name__ == "__main__":
    st.set_page_config(
        page_title="Bayesian Optimization",
        page_icon="🎯",
        layout="wide"
    )

    # Mock session state for testing
    if 'current_data' not in st.session_state:
        # Create dummy data
        np.random.seed(42)
        n_samples = 20
        st.session_state.current_data = pd.DataFrame({
            'Temperature': np.random.uniform(20, 100, n_samples),
            'Pressure': np.random.uniform(1, 10, n_samples),
            'Catalyst': np.random.uniform(0, 5, n_samples),
            'Yield': np.random.uniform(50, 95, n_samples)
        })
        st.session_state.current_workspace = "demo_workspace"

    show_bayesian_optimization_page()
