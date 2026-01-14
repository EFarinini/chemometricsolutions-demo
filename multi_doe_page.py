"""
Multi-DOE Analysis Page - Multi-Response Design of Experiments
Analyze multiple response variables (Y) simultaneously with shared predictor variables (X)

This module provides a complete interface for Multi-DOE analysis,
equivalent to running multiple MLR/DOE analyses in parallel with unified UI.

Features:
- Define X variables once, multiple Y variables
- Automatic model fitting for each Y
- Parallel surface analysis across all responses
- Multi-criteria decision making with weighted objectives
- Comparison views across response variables
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
import os

# Add parent directory to path (only if needed for mlr_doe import)
# This is needed because multi_doe_page.py is in the root directory
# but might be called from different contexts
if os.path.dirname(os.path.abspath(__file__)) not in sys.path:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import Multi-DOE utility functions
from mlr_utils.model_computation_multidoe import (
    fit_all_multidoe_models,
    statistical_summary_multidoe
)
from mlr_utils.model_diagnostics_multidoe import (
    show_model_diagnostics_ui_multidoe
)
from mlr_utils.surface_analysis_multidoe import (
    show_surface_analysis_ui_multidoe
)
from mlr_utils.predictions_multidoe import (
    show_predictions_ui_multidoe
)
from mlr_utils.export_multidoe import (
    show_export_ui_multidoe
)
from mlr_utils.pareto_ui_multidoe import (
    show_pareto_ui_multidoe
)

# Import candidate points generator (reused from MLR/DOE)
from mlr_utils.candidate_points import show_candidate_points_ui

# Import model computation helper functions for Tab 1
from mlr_utils.model_computation import (
    analyze_design_structure,
    create_term_selection_matrix,
    display_term_selection_ui,
    build_model_formula,
    create_model_matrix,
    fit_mlr_model,
    sort_coefficients_correct_order,
    generate_model_equation
)
from mlr_utils.model_computation_multidoe import (
    extract_coefficients_comparison,
    normalize_coefficients_for_comparison,
    get_coefficient_colors,
    categorize_coefficients_by_type,
    get_separator_positions
)
# Import detection functions from design_detection module
from mlr_utils.design_detection import (
    detect_central_points,
    detect_pseudo_central_points
)


def sort_coefficients_by_type(coef_names):
    """
    Sort coefficient names in standard order:
    1. Linear terms (no * or ^)
    2. Interaction terms (contains *)
    3. Quadratic terms (contains ^2)

    Args:
        coef_names: list of coefficient name strings

    Returns:
        sorted_names: list sorted by term type, maintaining alphabetical within each group
    """
    linear = []
    interactions = []
    quadratic = []

    for name in coef_names:
        if '^2' in name or ('^' in name and '2' in name):
            quadratic.append(name)
        elif '*' in name:
            interactions.append(name)
        else:
            linear.append(name)

    # Sort each group alphabetically for consistency
    linear.sort()
    interactions.sort()
    quadratic.sort()

    # Combine in correct order
    sorted_names = linear + interactions + quadratic
    return sorted_names


def pvalue_to_stars(p_value):
    """Convert p-value to significance stars"""
    if p_value is None or pd.isna(p_value):
        return "ns"
    elif p_value <= 0.001:
        return "***"
    elif p_value <= 0.01:
        return "**"
    elif p_value <= 0.05:
        return "*"
    else:
        return "ns"


def sort_and_format_coefficients_table(models_dict, x_vars=None, y_order=None):
    """
    Create sorted coefficients table with significance stars

    Order: Intercept → Linear (b1, b2, b3...) → Interactions (b12, b13, b23...) → Quadratic (b11, b22, b33...)

    CRITICAL: Uses CORRECT model sequence based on x_vars order, NOT alphabetical!

    Args:
        models_dict: Dictionary of fitted models {response_name: model_results}
        x_vars: List of X variable names in correct order (for proper coefficient sorting)
        y_order: List of Y variable names in desired column order (preserves original dataset order)

    Returns:
        DataFrame with sorted coefficients and significance columns
    """

    # Extract coefficients and p-values
    coef_data = {}
    pval_data = {}

    for response, model in models_dict.items():
        if isinstance(model, dict) and 'coefficients' in model:
            coef_data[response] = model['coefficients']
            if 'p_values' in model:
                pval_data[response] = model['p_values']

    if not coef_data:
        return None

    # Get all unique coefficient names
    all_coefs = set()
    for coefs in coef_data.values():
        all_coefs.update(coefs.index)

    # Separate intercept from other coefficients
    intercept = [c for c in all_coefs if 'Intercept' in str(c)]
    non_intercept = [c for c in all_coefs if 'Intercept' not in str(c)]

    # Sort non-intercept coefficients using CORRECT sequence
    if x_vars and non_intercept:
        sorted_non_intercept = sort_coefficients_correct_order(non_intercept, x_vars)
    else:
        # Fallback to alphabetical sorting if x_vars not provided
        def sort_coef_names_fallback(names):
            linear = []
            interactions = []
            quadratic = []

            for n in names:
                if '*' in str(n):
                    interactions.append(n)
                elif '^2' in str(n) or ('^' in str(n) and '2' in str(n)):
                    quadratic.append(n)
                else:
                    linear.append(n)

            linear.sort()
            interactions.sort()
            quadratic.sort()

            return linear + interactions + quadratic

        sorted_non_intercept = sort_coef_names_fallback(non_intercept)

    # Combine: Intercept FIRST, then sorted coefficients
    sorted_coefs = intercept + sorted_non_intercept

    # Build table with alternating columns: value | significance
    result_dict = {}

    # Determine Y variable order (use provided order or preserve dict order)
    if y_order is not None:
        response_order = [y for y in y_order if y in coef_data]
    else:
        response_order = list(coef_data.keys())

    for response in response_order:
        coefs = coef_data[response]
        pvals = pval_data.get(response, None)

        # Coefficient values
        coef_vals = []
        sig_vals = []

        for coef_name in sorted_coefs:
            coef_val = coefs.get(coef_name, np.nan)
            coef_vals.append(coef_val)

            if pvals is not None:
                p_val = pvals.get(coef_name, np.nan)
                sig_vals.append(pvalue_to_stars(p_val))
            else:
                sig_vals.append("—")

        # Create column pairs: (value, significance)
        result_dict[response] = coef_vals
        if pvals is not None:
            result_dict[f"{response} (sig)"] = sig_vals

    # Create DataFrame
    df = pd.DataFrame(result_dict, index=sorted_coefs)
    df.index.name = "Coefficient"

    return df


def _display_coefficients_barplot_multidoe(model_results, y_var, x_vars=None, normalize=False, coef_normalized_df=None, max_values_dict=None):
    """
    Display coefficients bar plot for Multi-DOE model with proper ordering: Linear, Interactions, Quadratic

    Based on model_computation.py _display_coefficients_barplot()
    Color scheme:
    - Red = Linear terms
    - Green = Two-term interactions
    - Cyan = Quadratic terms

    Args:
        model_results: Model results dictionary
        y_var: Response variable name
        x_vars: List of X variable names in correct order (for proper coefficient sorting)
        normalize: If True, use normalized coefficients (0-1 scale)
        coef_normalized_df: DataFrame with normalized coefficients (from normalize_coefficients_for_comparison)
        max_values_dict: Dictionary with max absolute values per Y variable
    """
    st.markdown(f"#### {y_var} - Coefficients")

    coefficients = model_results['coefficients']
    coef_no_intercept = coefficients[coefficients.index != 'Intercept']

    if len(coef_no_intercept) == 0:
        st.warning(f"No coefficients to plot for {y_var} (model contains only intercept)")
        return

    # ========================================================================
    # SORT COEFFICIENTS: Use CORRECT model sequence (b1, b2, b3, b12, b13, b23, b11, b22, b33)
    # ========================================================================
    coef_names_raw = coef_no_intercept.index.tolist()
    if x_vars:
        coef_names_sorted = sort_coefficients_correct_order(coef_names_raw, x_vars)
    else:
        # Fallback to alphabetical if x_vars not provided
        coef_names_sorted = sort_coefficients_by_type(coef_names_raw)

    # Get values in sorted order (NORMALIZED or ORIGINAL)
    if normalize and coef_normalized_df is not None and y_var in coef_normalized_df.columns:
        # Use normalized coefficients
        coef_values = []
        for coef_name in coef_names_sorted:
            if coef_name in coef_normalized_df.index:
                coef_values.append(coef_normalized_df.loc[coef_name, y_var])
            else:
                coef_values.append(0)  # Fallback if coefficient not found
        coef_values = np.array(coef_values)
    else:
        # Use original coefficients
        coef_values = coef_no_intercept.loc[coef_names_sorted].values

    coef_names = coef_names_sorted  # Use sorted names

    # ========================================================================
    # DETERMINE COLORS based on term type
    # ========================================================================
    colors = []
    for name in coef_names:
        if '*' in name:
            # Interaction term
            n_asterisks = name.count('*')
            colors.append('cyan' if n_asterisks > 1 else 'green')
        elif '^2' in name or '^' in name:
            # Quadratic term
            colors.append('cyan')
        else:
            # Linear term
            colors.append('red')

    # ========================================================================
    # CREATE BAR CHART
    # ========================================================================
    fig = go.Figure()

    # Prepare hover template based on normalization
    if normalize:
        # Get original values for hover
        original_values = coef_no_intercept.loc[coef_names_sorted].values
        hover_template = '<b>%{x}</b><br>Normalized: %{y:.4f}<br>Original: %{customdata:.4f}<extra></extra>'
        customdata_values = original_values
    else:
        hover_template = '<b>%{x}</b><br>Value: %{y:.4f}<extra></extra>'
        customdata_values = None

    fig.add_trace(go.Bar(
        x=coef_names,
        y=coef_values,
        marker_color=colors,
        marker_line_color='black',
        marker_line_width=1,
        name='Coefficients',
        showlegend=False,
        hovertemplate=hover_template,
        customdata=customdata_values
    ))

    # ========================================================================
    # ADD CONFIDENCE INTERVALS (if available)
    # ========================================================================
    if 'ci_lower' in model_results and 'ci_upper' in model_results:
        try:
            ci_lower = model_results['ci_lower'].loc[coef_names_sorted].values
            ci_upper = model_results['ci_upper'].loc[coef_names_sorted].values

            error_minus = coef_values - ci_lower
            error_plus = ci_upper - coef_values

            fig.update_traces(
                error_y=dict(
                    type='data',
                    symmetric=False,
                    array=error_plus,
                    arrayminus=error_minus,
                    color='black',
                    thickness=2,
                    width=4
                )
            )
        except Exception as e:
            pass  # Continue without CI if error

    # ========================================================================
    # ADD SIGNIFICANCE MARKERS (if available)
    # ========================================================================
    if 'p_values' in model_results:
        try:
            p_values = model_results['p_values'].loc[coef_names_sorted].values
            for i, (name, coef, p) in enumerate(zip(coef_names, coef_values, p_values)):
                y_pos = coef
                y_offset = max(abs(coef) * 0.05, 0.01) if coef >= 0 else -max(abs(coef) * 0.05, 0.01)

                if p <= 0.001:
                    sig_text = '***'
                elif p <= 0.01:
                    sig_text = '**'
                elif p <= 0.05:
                    sig_text = '*'
                else:
                    sig_text = None

                if sig_text:
                    fig.add_annotation(
                        x=name, y=y_pos + y_offset,
                        text=sig_text,
                        showarrow=False,
                        font=dict(size=20, color='black'),
                        yshift=5 if coef >= 0 else -5
                    )
        except Exception as e:
            pass  # Continue without p-values if error

    # Update layout (adjust title and y-axis based on normalization)
    if normalize:
        yaxis_title = "Normalized Coefficient Value (0-1 scale)"
        title_suffix = " (Normalized)"
        # Set consistent y-axis range for normalized values
        yaxis_config = dict(
            range=[0, 1.2],
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor='gray'
        )
    else:
        yaxis_title = "Coefficient Value"
        title_suffix = ""
        # Dynamic y-axis range for original values
        yaxis_config = dict(
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor='gray'
        )

    fig.update_layout(
        title=f"Coefficients - {y_var}{title_suffix}",
        xaxis_title="Term",
        yaxis_title=yaxis_title,
        height=400,
        xaxis={'tickangle': 45},
        showlegend=False,
        yaxis=yaxis_config,
        margin=dict(b=100)
    )

    st.plotly_chart(fig, use_container_width=True)

    # Display color legend
    col1, col2, col3 = st.columns(3)
    with col1:
        st.caption("🔴 Red = Linear terms")
    with col2:
        st.caption("🟢 Green = 2-term interactions")
    with col3:
        st.caption("🔵 Cyan = Quadratic terms")

    if 'p_values' in model_results:
        st.caption("Significance: *** p≤0.001, ** p≤0.01, * p≤0.05")


def show():
    """Display the Multi-DOE Analysis page"""

    st.markdown("# 🎯 Multi-DOE Analysis")
    st.markdown("*Multiple Response MLR & Design of Experiments*")
    st.info("""
    Analyze multiple response variables simultaneously with shared predictor variables.
    This powerful approach allows you to:
    - Fit models for all responses at once
    - Compare surfaces across different responses
    - Optimize multiple criteria simultaneously (Pareto optimization)
    """)

    # Check if data is loaded
    data_loaded = 'current_data' in st.session_state and st.session_state.current_data is not None
    data = st.session_state.current_data if data_loaded else None

    # Create 7 tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "Model Computation",          # Multi-model fitting
        "Model Diagnostics",          # Select response + show diagnostics
        "Surface Analysis",           # Show contours for all responses
        "Predictions",                # Predict all responses at once
        "Multi-Criteria Decision",    # Optimize weighted combination of responses
        "Generate Matrix",            # Standalone (reuse from mlr_doe)
        "Extract & Export"            # Export all models to Excel
    ])

    # ========================================================================
    # TAB 1: MODEL COMPUTATION
    # ========================================================================
    with tab1:
        if not data_loaded:
            st.warning("⚠️ No data loaded")
            st.info("💡 Go to **Data Handling** to load data")
            return

        # ====================================================================
        # SECTION 1: DATA PREVIEW (identical to model_computation.py)
        # ====================================================================
        st.markdown("### 👁️ Data Preview")
        with st.expander("Show current dataset", expanded=True):
            # Full scrollable dataframe
            st.dataframe(data, use_container_width=True, height=400)

            col_info1, col_info2, col_info3 = st.columns(3)
            with col_info1:
                st.metric("Total Samples", data.shape[0])
            with col_info2:
                st.metric("Total Variables", data.shape[1])
            with col_info3:
                numeric_cols_count = len(data.select_dtypes(include=[np.number]).columns)
                st.metric("Numeric Variables", numeric_cols_count)

        st.markdown("---")

        # ====================================================================
        # SECTION 2: VARIABLE SELECTION (adapted for Multi-DOE)
        # ====================================================================
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 📊 Variable Selection")

            numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()

            if not numeric_columns:
                st.error("❌ No numeric columns found!")
                return

            # X variables (shared across all models)
            x_vars = st.multiselect(
                "Select X variables (predictors):",
                numeric_columns,
                default=numeric_columns[:min(3, len(numeric_columns))],
                key="multidoe_x_vars_widget",
                help="These predictors will be shared across ALL response models"
            )

            # Y variables (MULTIPLE - one model per Y)
            remaining_cols = [col for col in numeric_columns if col not in x_vars]
            if remaining_cols:
                y_vars = st.multiselect(
                    "Select Y variables (responses - 2+ required):",
                    remaining_cols,
                    key="multidoe_y_vars_widget",
                    help="Select 2+ response variables - one model will be fitted for each"
                )

                # Validation
                if not y_vars:
                    st.warning("⚠️ Select at least 2 Y variables for Multi-DOE")
                    return
                elif len(y_vars) < 2:
                    st.error("❌ Multi-DOE requires at least 2 Y variables")
                    return
                else:
                    st.success(f"✅ Will fit {len(y_vars)} models (one per Y variable)")
            else:
                st.warning("⚠️ Select at least one X variable")
                return

            # Show configuration summary
            if x_vars and y_vars:
                x_vars_str = [str(var) for var in x_vars]
                y_vars_str = [str(var) for var in y_vars]
                st.info(f"**Multi-DOE Configuration:**\n\n"
                       f"X (shared): {' + '.join(x_vars_str)}\n\n"
                       f"Y (multiple): {', '.join(y_vars_str)}")

        with col2:
            st.markdown("### 🎯 Sample Selection")

            # Sample selection options
            sample_selection_mode = st.radio(
                "Select samples:",
                ["Use all samples", "Select by index", "Select by range"],
                key="multidoe_sample_selection_mode"
            )

            if sample_selection_mode == "Use all samples":
                selected_samples = data.index.tolist()
                st.success(f"Using all {len(selected_samples)} samples")

            elif sample_selection_mode == "Select by index":
                sample_input = st.text_input(
                    "Enter sample indices (1-based, comma-separated or ranges):",
                    value=f"1-{data.shape[0]}",
                    help="Examples: 1,2,5-10,15 or 1-20",
                    key="multidoe_sample_input"
                )

                try:
                    selected_indices = []
                    for part in sample_input.split(','):
                        part = part.strip()
                        if '-' in part:
                            start, end = map(int, part.split('-'))
                            selected_indices.extend(range(start-1, end))
                        else:
                            selected_indices.append(int(part)-1)

                    selected_indices = sorted(list(set(selected_indices)))
                    valid_indices = [i for i in selected_indices if 0 <= i < len(data)]
                    selected_samples = data.index[valid_indices].tolist()

                    st.success(f"Selected {len(selected_samples)} samples")

                except Exception as e:
                    st.error(f"Invalid format: {e}")
                    selected_samples = data.index.tolist()

            else:  # Select by range
                col_range1, col_range2 = st.columns(2)
                with col_range1:
                    start_idx = st.number_input("From sample:", 1, len(data), 1, key="multidoe_start_idx")
                with col_range2:
                    end_idx = st.number_input("To sample:", start_idx, len(data), len(data), key="multidoe_end_idx")

                selected_samples = data.index[start_idx-1:end_idx].tolist()
                st.success(f"Selected {len(selected_samples)} samples (rows {start_idx}-{end_idx})")

            # Show selected samples preview
            if len(selected_samples) < len(data):
                with st.expander("Preview selected samples"):
                    st.dataframe(data.loc[selected_samples].head(10), use_container_width=True)

        st.markdown("---")

        # ====================================================================
        # SECTION 3: DESIGN STRUCTURE ANALYSIS (identical to model_computation.py)
        # ====================================================================
        if not x_vars:
            st.warning("Please select X variables first")
            return

        st.markdown("### 🎛️ Model Configuration")
        st.markdown("#### 🔍 Design Structure Analysis")

        # Prepare X data for analysis
        X_for_analysis = data.loc[selected_samples, x_vars].copy()

        with st.spinner("Analyzing design structure..."):
            try:
                design_analysis = analyze_design_structure(X_for_analysis)

                # Display design analysis results
                col_analysis1, col_analysis2 = st.columns([2, 1])

                with col_analysis1:
                    st.info(design_analysis['interpretation'])

                with col_analysis2:
                    st.metric("Design Type", design_analysis['design_type'])
                    st.metric("Center Points", len(design_analysis['center_points_indices']))

                # Show warnings if any
                if design_analysis['warnings']:
                    for warning_msg in design_analysis['warnings']:
                        st.warning(warning_msg)

                # Display levels per variable
                st.markdown("**Levels per Variable (excluding center points)**")
                levels_df = pd.DataFrame([
                    {
                        'Variable': var_name,
                        'Levels': n_levels,
                        'Type': 'Quantitative' if design_analysis['is_quantitative'].get(var_name, True) else 'Categorical'
                    }
                    for var_name, n_levels in design_analysis['n_levels_per_var'].items()
                ])

                st.dataframe(levels_df, use_container_width=True, hide_index=True)

            except Exception as e:
                st.warning(f"⚠️ Design analysis failed: {str(e)}")
                st.info("Using default configuration (intercept + linear terms)")

                # Fallback defaults
                design_analysis = {
                    'design_type': 'unknown',
                    'recommended_terms': {
                        'intercept': True,
                        'linear': True,
                        'interactions': False,
                        'quadratic': False
                    },
                    'n_levels_per_var': {var: 2 for var in x_vars},
                    'is_quantitative': {var: True for var in x_vars},
                    'center_points_indices': [],
                    'warnings': []
                }

        # ====================================================================
        # SECTION 4: MODEL CONFIGURATION (identical to model_computation.py)
        # ====================================================================
        st.markdown("### 🔧 Model Configuration")

        # Show design analysis info as compact caption
        if design_analysis['design_type'] == "2-level":
            st.caption("✅ 2-Level Design - Interactions OK, no quadratic")
        elif design_analysis['design_type'] == ">2-level":
            st.caption("✅ >2-Level Design - All terms available")
        elif design_analysis['design_type'] == "qualitative_only":
            st.caption("⚠️ Qualitative Only - Linear terms only")
        else:
            st.caption(f"ℹ️ Design: {design_analysis['design_type']}")

        # Top controls (3 checkboxes - matching model_computation.py)
        col_top1, col_top2, col_top3 = st.columns(3)

        with col_top1:
            include_intercept = st.checkbox(
                "Include intercept",
                value=True,
                disabled=True,
                help="Always included in model",
                key="multidoe_include_intercept"
            )

        with col_top2:
            # Disable interactions for qualitative-only designs
            should_disable_interactions = (design_analysis['design_type'] == "qualitative_only")

            include_interactions = st.checkbox(
                "Include interactions",
                value=design_analysis['recommended_terms']['interactions'],
                disabled=should_disable_interactions,
                help="Two-way interaction terms (X1*X2)" if not should_disable_interactions else "Not available for qualitative-only",
                key="multidoe_include_interactions"
            )

        with col_top3:
            # Disable quadratic for 2-level or qualitative-only designs
            should_disable_quadratic = (design_analysis['design_type'] in ["2-level", "qualitative_only"])

            include_quadratic = st.checkbox(
                "Include quadratic terms",
                value=design_analysis['recommended_terms']['quadratic'] if not should_disable_quadratic else False,
                disabled=should_disable_quadratic,
                help="Quadratic terms (X1²)" if not should_disable_quadratic else "Only for >2-level designs",
                key="multidoe_include_quadratic"
            )

        st.markdown("---")

        # ====================================================================
        # SECTION 5: TERM SELECTION MATRIX (updated for 3-checkbox model)
        # ====================================================================
        if (include_interactions or include_quadratic) and design_analysis['design_type'] != "qualitative_only":

            st.markdown("### 📊 Select Model Terms")
            st.info("Use the matrix below to select interactions and quadratic terms")

            # Get the term selection matrix UI (pass individual flags)
            term_matrix, selected_terms = display_term_selection_ui(
                x_vars,
                key_prefix="multidoe_config",
                design_analysis=design_analysis,
                allow_interactions=include_interactions,
                allow_quadratic=include_quadratic
            )

            # Display Summary
            st.markdown("#### Summary")

            col_sum1, col_sum2, col_sum3 = st.columns(3)

            with col_sum1:
                linear_count = len(selected_terms['linear'])
                st.metric("Linear Terms", linear_count)

            with col_sum2:
                interaction_count = len(selected_terms['interactions'])
                st.metric("Interactions", interaction_count)

            with col_sum3:
                quadratic_count = len(selected_terms['quadratic'])
                st.metric("Quadratic Terms", quadratic_count)

            # Saturation check
            n_total = 1 + linear_count + interaction_count + quadratic_count

            st.markdown("---")

            if n_total > len(X_for_analysis):
                st.error(f"❌ Model is saturated! {n_total} terms > {len(X_for_analysis)} observations")
            elif n_total >= len(X_for_analysis) * 0.8:
                st.warning(f"⚠️  Model is near saturation: {n_total} terms ≈ {len(X_for_analysis)} observations")
            else:
                st.success(f"✅ Model has {len(X_for_analysis) - n_total} degrees of freedom")

        else:
            # Higher-order disabled (qualitative-only or user unchecked)
            st.info("📊 **Select Model Terms** - Higher-order terms disabled")

            # Build simple selected_terms with linear only
            selected_terms = {
                'linear': x_vars.copy(),
                'interactions': [],
                'quadratic': []
            }

            # Create empty term_matrix (all zeros)
            term_matrix = create_term_selection_matrix(x_vars)
            for i in range(len(x_vars)):
                for j in range(len(x_vars)):
                    term_matrix.iloc[i, j] = 0

        st.markdown("---")

        # ====================================================================
        # SECTION 6: ADDITIONAL SETTINGS (identical to model_computation.py)
        # ====================================================================
        st.markdown("### ⚙️ Additional Model Settings")

        col_set1, col_set2 = st.columns(2)

        with col_set1:
            exclude_central_points = st.checkbox(
                "Exclude central points (0,0,0...)",
                value=False,
                help="Central points are typically used only for validation in factorial designs",
                key="multidoe_exclude_central"
            )

        with col_set2:
            run_cv = st.checkbox(
                "Run cross-validation",
                value=True,
                help="Leave-one-out CV (only for n≤100)",
                key="multidoe_run_cv"
            )

        # Detect pseudo-central points (pattern-based detection)
        pseudo_central_indices = detect_pseudo_central_points(X_for_analysis, design_analysis)

        # Show checkbox if:
        # 1. Found pseudo-central points AND
        # 2. User did NOT select quadratic (pseudo-centrals don't matter for quadratic models)
        show_pseudo_central_option = (
            len(pseudo_central_indices) > 0 and
            not include_quadratic
        )

        if show_pseudo_central_option:
            st.markdown("---")

            # Debug output to help understand what's being detected
            with st.expander("🔍 Debug: Pseudo-Central Detection"):
                st.write(f"**Pseudo-central points detected at indices:** {pseudo_central_indices}")
                st.write(f"**Include quadratic?** {include_quadratic}")

                if pseudo_central_indices:
                    st.write("**Detected pseudo-central points:**")
                    for idx in pseudo_central_indices:
                        row_data = X_for_analysis.iloc[idx].to_dict()
                        st.write(f"  Row {X_for_analysis.index[idx]}: {row_data}")

            exclude_pseudo_central = st.checkbox(
                f"Exclude pseudo-central points ({len(pseudo_central_indices)} found)",
                value=False,
                help="Points with some (but not all) coordinates at 0. Repeated points used for validation and variance estimation.",
                key="multidoe_exclude_pseudo_central"
            )
        else:
            exclude_pseudo_central = False

        # ====================================================================
        # SECTION 7: MODEL FORMULAS (NEW - show all Y formulas)
        # ====================================================================
        st.markdown("---")
        st.markdown("### 📐 Postulated Model Formulas")

        # Display formula for each Y variable
        with st.expander("View all model formulas", expanded=True):
            for y_var in y_vars:
                try:
                    formula = build_model_formula(y_var, selected_terms, include_intercept)
                    st.code(formula, language="text")
                except Exception as e:
                    st.warning(f"Could not generate formula for {y_var}: {str(e)}")
                    st.code(f"{y_var} = b0 + b1·X + ... (formula generation error)", language="text")

        # Summary of selected terms
        total_terms = len(selected_terms['linear']) + len(selected_terms['interactions']) + len(selected_terms['quadratic'])
        if include_intercept:
            total_terms += 1

        st.info(f"""
        **Multi-DOE Summary:**
        - Total parameters per model: {total_terms}
        - Number of response models: {len(y_vars)}
        - Response variables: {', '.join(y_vars)}
        - Cross-validation: {'Enabled' if run_cv else 'Disabled'}
        """)

        # ====================================================================
        # SECTION 8: FIT BUTTON & RESULTS (NEW - Multi-DOE logic)
        # ====================================================================
        st.markdown("---")

        if st.button("🚀 Fit All Models", type="primary", key="fit_all_multidoe"):
            try:
                with st.spinner(f"Fitting {len(y_vars)} models..."):
                    # Prepare data with selected samples
                    X_data = data.loc[selected_samples, x_vars].copy()

                    # Remove missing values in X
                    valid_idx_X = ~X_data.isnull().any(axis=1)

                    # Store original X (before excluding central points)
                    X_data_original = X_data[valid_idx_X].copy()

                    # Detect and optionally exclude central points
                    central_points = detect_central_points(X_data_original)

                    if central_points:
                        # st.info(f"🎯 Detected {len(central_points)} central point(s) at indices: {[i+1 for i in central_points]}")

                        if exclude_central_points:
                            # Remove central points from X
                            X_data_for_fitting = X_data_original.drop(X_data_original.index[central_points])
                            st.warning(f"⚠️ Excluded {len(central_points)} central point(s) from analysis")
                            st.info(f"ℹ️ Using {len(X_data_for_fitting)} samples (excluding central points)")
                        else:
                            X_data_for_fitting = X_data_original

                    else:
                        X_data_for_fitting = X_data_original

                    # Detect and optionally exclude pseudo-central points
                    # (only relevant for designs without quadratic, matching checkbox logic)
                    if exclude_pseudo_central and len(pseudo_central_indices) > 0:
                        # Get indices in current X_data_for_fitting (after potential central point removal)
                        # Need to map from original indices to current X_data_for_fitting indices
                        remaining_pseudo_central = [i for i in range(len(X_data_for_fitting))
                                                   if X_data_for_fitting.index.tolist()[i] in
                                                   [data.index.tolist()[pi] for pi in pseudo_central_indices]]

                        if remaining_pseudo_central:
                            pseudo_central_samples_original = X_data_for_fitting.index[remaining_pseudo_central].tolist()

                            st.info(f"🎯 Detected {len(remaining_pseudo_central)} pseudo-central point(s)")

                            # Remove pseudo-central points from modeling data
                            X_data_for_fitting = X_data_for_fitting.drop(X_data_for_fitting.index[remaining_pseudo_central])

                            st.warning(f"⚠️ Excluded {len(remaining_pseudo_central)} pseudo-central point(s) from analysis")

                            st.info(f"ℹ️ Using {len(X_data_for_fitting)} samples (after all exclusions)")

                            # Store excluded pseudo-central points for later validation
                            # (will be used for all Y variables in Multi-DOE)
                            st.session_state.multi_doe_pseudo_central_points = {
                                'X': data.loc[pseudo_central_samples_original, x_vars],
                                'indices': pseudo_central_samples_original
                            }

                    # Create Y dictionary (remove NaNs per Y variable)
                    y_dict = {}
                    for y_var in y_vars:
                        y_series = data.loc[selected_samples, y_var].copy()
                        # Filter to match X_data_for_fitting indices
                        y_series = y_series.loc[X_data_for_fitting.index]
                        # Remove NaNs in this Y
                        valid_y = ~y_series.isnull()
                        y_dict[y_var] = y_series[valid_y]

                    if len(X_data_for_fitting) < len(x_vars) + 1:
                        st.error("❌ Not enough samples for model fitting!")
                        return

                    # st.info(f"ℹ️ Using {len(X_data_for_fitting)} samples for model fitting")

                    # ================================================================
                    # CRITICAL FIX: Create model matrix ONCE, reuse for ALL Y
                    # ================================================================



                    try:
                        X_model, term_names = create_model_matrix(
                            X_data_for_fitting,
                            include_intercept=include_intercept,
                            include_interactions=True if len(selected_terms['interactions']) > 0 else False,
                            include_quadratic=True if len(selected_terms['quadratic']) > 0 else False,
                            interaction_matrix=term_matrix
                        )

                        # st.success(f"✅ Created model matrix: {X_model.shape[0]} samples × {X_model.shape[1]} terms")
                        # st.info(f"📋 Model terms: {', '.join(term_names)}")

                    except Exception as e:
                        st.error(f"❌ Failed to create model matrix: {str(e)}")
                        import traceback
                        with st.expander("🐛 Matrix creation error details"):
                            st.code(traceback.format_exc())
                        return

                    # Step 2: Align Y data to X_model index

                    y_dict_aligned = {}
                    for y_var in y_vars:
                        if y_var in y_dict:
                            y_series = y_dict[y_var]
                            # Align to X_model index (same samples)
                            try:
                                y_aligned = y_series.reindex(X_model.index)
                                # Remove NaNs
                                valid_mask = ~y_aligned.isna()
                                if valid_mask.sum() > 0:
                                    y_dict_aligned[y_var] = y_aligned[valid_mask]
                                    # st.info(f"  • {y_var}: {valid_mask.sum()} valid samples")
                                else:
                                    st.warning(f"⚠️ {y_var} has no valid data after alignment")
                            except Exception as e:
                                st.warning(f"⚠️ Could not align {y_var}: {str(e)}")

                    if not y_dict_aligned:
                        st.error("❌ No valid Y variables after alignment!")
                        return

                    # Step 3: Fit all models using SAME X_model
                    # st.info(f"🚀 Fitting {len(y_dict_aligned)} models...")
                    models_dict = {}

                    for y_var, y_data in y_dict_aligned.items():
                        try:
                            # Align X_model to y_data indices (remove rows where Y is NaN)
                            X_model_aligned = X_model.loc[y_data.index]

                            # Fit model (X_model already has correct terms, pass None for terms)
                            model = fit_mlr_model(
                                X_model_aligned,
                                y_data,
                                return_diagnostics=run_cv
                            )

                            # Add metadata
                            model['y_name'] = y_var
                            model['X'] = X_model_aligned
                            model['y'] = y_data
                            model['term_names'] = term_names

                            models_dict[y_var] = model
                            # st.success(f"  ✅ {y_var}: R²_adj = {model.get('r_squared_adj', model.get('r_squared', 0)):.4f}")

                        except Exception as e:
                            st.error(f"  ❌ {y_var} failed: {str(e)}")
                            models_dict[y_var] = {
                                'error': str(e),
                                'y_name': y_var,
                                'status': 'Failed'
                            }

                # Store results in session state
                st.session_state.multi_doe_models = models_dict
                st.session_state.multi_doe_x_vars = x_vars
                st.session_state.multi_doe_y_vars = y_vars
                st.session_state.y_variable_order = y_vars  # Store Y order for visualization consistency
                st.session_state.X_data = X_data_for_fitting
                st.session_state.multi_doe_X_model = X_model
                st.session_state.multi_doe_term_names = term_names

                # st.success(f"✅ Successfully fitted {len(models_dict)} models!")

                # ============================================================
                # COMPACT SUMMARY TABLE (replaces verbose status messages)
                # ============================================================
                st.markdown("#### ✅ Model Fitting Summary")

                # Build summary data
                summary_data = []
                for y_var, model in models_dict.items():
                    if 'error' not in model:
                        r2_adj = model.get('r_squared_adj', model.get('r_squared', 0))
                        status = "✅"
                    else:
                        r2_adj = np.nan
                        status = "❌"

                    summary_data.append({
                        'Response': y_var,
                        'Samples': len(X_data_for_fitting),
                        'R²_adj': f"{r2_adj:.4f}" if pd.notna(r2_adj) else "Error",
                        'Status': status
                    })

                summary_df_compact = pd.DataFrame(summary_data)

                # Display summary metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Central Points", len(central_points))
                with col2:
                    st.metric("Samples Used", len(X_data_for_fitting))
                with col3:
                    st.metric("Model Terms", X_model.shape[1])

                # Display results table
                st.dataframe(summary_df_compact, use_container_width=True, hide_index=True)

                # ============================================================
                # RESULTS DISPLAY
                # ============================================================
                st.markdown("---")
                st.markdown("## 📊 Multi-DOE Results")

                # Result 1: Summary Table
                st.markdown("### 📈 Model Summary Table")
                summary_df = statistical_summary_multidoe(models_dict)

                # Format numeric columns
                for col in ['R²_adj', 'RMSE', 'Q²_adj', 'RMSECV']:
                    if col in summary_df.columns:
                        summary_df[col] = summary_df[col].apply(
                            lambda x: f"{x:.4f}" if not pd.isna(x) else "N/A"
                        )

                st.dataframe(summary_df, use_container_width=True, hide_index=True)

                # Result 2: Adjusted R² Comparison Chart
                st.markdown("### 📈 Adjusted R² Comparison Across Models")
                r2_data = []
                for y_var, model in models_dict.items():
                    if 'error' not in model:
                        r2_data.append({
                            'Response': y_var,
                            'R²_adj': model.get('r_squared_adj', model.get('r_squared', 0))
                        })

                if r2_data:
                    r2_df = pd.DataFrame(r2_data)
                    fig = px.bar(
                        r2_df,
                        x='Response',
                        y='R²_adj',
                        title='Model Quality Comparison (Adjusted R²)',
                        labels={'R²_adj': 'R²_adj Value'},
                        color='R²_adj',
                        color_continuous_scale='Viridis'
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)

                # Result 3: Coefficients Comparison Table (WITH SORTING AND SIGNIFICANCE)
                st.markdown("### 🔢 Coefficients Comparison (Side-by-Side)")
                st.info("Compare model coefficients across all response variables with significance indicators")

                # Use new sorting function with significance stars (pass x_vars for correct ordering and y_vars for Y variable order)
                coef_table_sorted = sort_and_format_coefficients_table(models_dict, x_vars, y_order=y_vars)

                if coef_table_sorted is not None and not coef_table_sorted.empty:
                    st.markdown("#### Coefficients Table")

                    # Reset index to show coefficient names as column
                    coef_display = coef_table_sorted.reset_index()

                    # Format numeric columns (only coefficient values, not significance)
                    for col in coef_display.columns:
                        if col != 'Coefficient' and '(sig)' not in col:
                            coef_display[col] = coef_display[col].apply(
                                lambda x: f"{x:.6f}" if pd.notna(x) and isinstance(x, (int, float)) else "—"
                            )

                    st.dataframe(coef_display, use_container_width=True, hide_index=True)
                    st.caption("**Significance codes:** *** p≤0.001, ** p≤0.01, * p≤0.05, ns = not significant")

                    # NEW: Add coefficient comparison chart (EXCLUDE INTERCEPT)
                    st.markdown("#### 📊 Coefficients Comparison Chart")

                    # ============================================================
                    # STEP 1: Get coefficient names (exclude Intercept)
                    # ============================================================
                    coef_names_raw = [row['Coefficient'] for _, row in coef_display.iterrows()
                                     if row['Coefficient'] != 'Intercept']

                    if not coef_names_raw:
                        st.info("No coefficients to plot (excluding intercept)")
                    else:
                        # ========================================================
                        # STEP 2: SORT IN CORRECT MODEL SEQUENCE (b1, b2, b3, b12, b13, b23, b11, b22, b33)
                        # ========================================================
                        coef_names_sorted = sort_coefficients_correct_order(coef_names_raw, x_vars)

                        # DEBUG: Verify coefficient order matches table (optional)
                        if st.session_state.get('debug_mode', False):
                            st.write("**DEBUG - Coefficient Ordering Verification:**")
                            st.write("Chart order (coef_names_sorted):")
                            for i, name in enumerate(coef_names_sorted):
                                st.write(f"  {i}: {name}")
                            st.write("Table order (coef_table_sorted non-Intercept rows):")
                            for i, name in enumerate(coef_table_sorted.index):
                                if name != 'Intercept':
                                    st.write(f"  {i}: {name}")

                        # ========================================================
                        # STEP 2.5: NORMALIZATION (NEW)
                        # ========================================================
                        # Extract coefficients matrix for normalization (with Y order preservation)
                        coef_comparison_df = extract_coefficients_comparison(models_dict, y_order=y_vars)

                        # Normalize coefficients for comparison
                        coef_normalized_df, max_values_dict = normalize_coefficients_for_comparison(coef_comparison_df)

                        # Get coefficient colors
                        coef_colors = get_coefficient_colors(coef_comparison_df)

                        # Calculate separator positions for visual dividers between term types
                        separator_positions = get_separator_positions(coef_comparison_df)

                        # ========================================================
                        # STEP 3: Prepare data with sorted order (USING NORMALIZED VALUES)
                        # ========================================================
                        coef_chart_data = []
                        for coef_name in coef_names_sorted:
                            for y_var in [col for col in coef_display.columns if col != 'Coefficient' and '(sig)' not in col]:
                                # Use NORMALIZED values from coef_normalized_df
                                if coef_name in coef_normalized_df.index and y_var in coef_normalized_df.columns:
                                    normalized_value = coef_normalized_df.loc[coef_name, y_var]
                                    original_value = coef_comparison_df.loc[coef_name, y_var]  # Keep original for hover

                                    if pd.notna(normalized_value):
                                        coef_chart_data.append({
                                            'Coefficient': coef_name,
                                            'Response': y_var,
                                            'Value_Normalized': normalized_value,
                                            'Value_Original': original_value,
                                            'Max_Abs_Y': max_values_dict.get(y_var, 1.0)
                                        })

                        if coef_chart_data:
                            coef_chart_df = pd.DataFrame(coef_chart_data)

                            # ====================================================
                            # STEP 4: Create simplified grouped bar chart
                            # ====================================================
                            fig_coef = go.Figure()

                            # Get unique responses in ORIGINAL Y variable order (EXCLUDE significance columns)
                            # Use y_vars which preserves order from fit_all_multidoe_models()
                            responses = [y_var for y_var in y_vars
                                        if y_var in coef_display.columns]

                            # Count term types for legend
                            linear_count = sum(1 for n in coef_names_sorted if '*' not in n and '^' not in n)
                            interaction_count = sum(1 for n in coef_names_sorted if '*' in n)
                            quadratic_count = len(coef_names_sorted) - linear_count - interaction_count

                            # Add a bar trace for each response variable
                            for response in responses:
                                response_data = coef_chart_df[coef_chart_df['Response'] == response]
                                # Maintain sorted order
                                response_values_normalized = []
                                response_values_original = []
                                for coef in coef_names_sorted:
                                    match = response_data[response_data['Coefficient'] == coef]
                                    if not match.empty:
                                        response_values_normalized.append(match.iloc[0]['Value_Normalized'])
                                        response_values_original.append(match.iloc[0]['Value_Original'])
                                    else:
                                        response_values_normalized.append(None)
                                        response_values_original.append(None)

                                fig_coef.add_trace(go.Bar(
                                    x=coef_names_sorted,
                                    y=response_values_normalized,  # Use normalized values for y-axis
                                    name=response,
                                    customdata=response_values_original,  # Store original values
                                    hovertemplate='<b>%{x}</b><br>' +
                                                  response + ' (normalized): %{y:.4f}<br>' +
                                                  'Original value: %{customdata:.4f}<br>' +
                                                  f'Scaling factor: {max_values_dict.get(response, 1.0):.4f}<extra></extra>'
                                ))

                            # ====================================================
                            # STEP 5: Add significance stars ON bars (centered)
                            # ====================================================
                            # Extract p-values from models_dict
                            for i, coef_name in enumerate(coef_names_sorted):
                                for j, response in enumerate(responses):
                                    # Get p-value for this coefficient and response
                                    if response in models_dict:
                                        model = models_dict[response]
                                        if 'p_values' in model and coef_name in model['p_values'].index:
                                            p_val = model['p_values'][coef_name]

                                            # Determine significance stars
                                            if pd.notna(p_val):
                                                if p_val <= 0.001:
                                                    sig_text = '***'
                                                elif p_val <= 0.01:
                                                    sig_text = '**'
                                                elif p_val <= 0.05:
                                                    sig_text = '*'
                                                else:
                                                    sig_text = None

                                                if sig_text:
                                                    # Get NORMALIZED coefficient value for positioning
                                                    if coef_name in coef_normalized_df.index and response in coef_normalized_df.columns:
                                                        try:
                                                            value = coef_normalized_df.loc[coef_name, response]

                                                            if pd.notna(value):
                                                                # Calculate x position for grouped bars
                                                                num_responses = len(responses)
                                                                bar_width = 0.8 / num_responses
                                                                x_pos = i + (j - num_responses/2 + 0.5) * bar_width

                                                                # Position star at CENTER of bar
                                                                fig_coef.add_annotation(
                                                                    x=x_pos,
                                                                    y=value,
                                                                    text=sig_text,
                                                                    showarrow=False,
                                                                    font=dict(size=20, color='black'),
                                                                    xanchor='center',
                                                                    yanchor='middle'
                                                                )
                                                        except (ValueError, TypeError):
                                                            pass

                            # ====================================================
                            # Add vertical separators between term types
                            # Based on DISPLAYED coefficients (coef_names_sorted), not indices
                            # ====================================================

                            # Categorize the DISPLAYED coefficients
                            linear_terms = [c for c in coef_names_sorted if '*' not in c and '^2' not in c and '^' not in c]
                            interaction_terms = [c for c in coef_names_sorted if '*' in c or ':' in c]
                            quadratic_terms = [c for c in coef_names_sorted if '^2' in c or ('^' in c and '2' in c)]

                            # Count how many of each type
                            n_linear = len(linear_terms)
                            n_interaction = len(interaction_terms)
                            n_quadratic = len(quadratic_terms)

                            # Separator AFTER linear terms (between linear and interaction)
                            if n_linear > 0 and n_interaction > 0:
                                sep_after_linear = n_linear - 0.5
                                fig_coef.add_vline(
                                    x=sep_after_linear,
                                    line_dash="dash",
                                    line_color="rgba(128, 128, 128, 0.5)",
                                    line_width=2
                                )

                            # Separator AFTER interaction terms (between interaction and quadratic)
                            if n_interaction > 0 and n_quadratic > 0:
                                sep_after_interaction = n_linear + n_interaction - 0.5
                                fig_coef.add_vline(
                                    x=sep_after_interaction,
                                    line_dash="dash",
                                    line_color="rgba(128, 128, 128, 0.5)",
                                    line_width=2
                                )

                            # ====================================================
                            # STEP 6: Updated layout with normalization info
                            # ====================================================
                            fig_coef.update_layout(
                                title='Coefficient Comparison Across Response Variables (Normalized by Max Absolute Value)',
                                xaxis_title='Coefficient',
                                yaxis_title='Normalized Coefficient Value (0-1 scale)',
                                barmode='group',
                                height=500,
                                xaxis={'tickangle': -45},
                                margin=dict(b=100),
                                showlegend=True
                            )

                            st.plotly_chart(fig_coef, use_container_width=True)

                            # Add normalization explanation
                            st.caption("""
**Normalization:** Each Y variable is normalized independently by its maximum absolute coefficient value.
This allows visual comparison across responses with very different scales.
Original values and scaling factors are shown in hover tooltips.
""")

                            # Display significance legend
                            st.caption("**Significance:** *** p≤0.001, ** p≤0.01, * p≤0.05")

                        else:
                            st.info("No coefficients to plot (excluding intercept)")

                    # Allow download
                    csv = coef_table_sorted.to_csv(index=True)
                    st.download_button(
                        label="📥 Download Coefficients Comparison (CSV)",
                        data=csv,
                        file_name="multi_doe_coefficients_comparison.csv",
                        mime="text/csv"
                    )
                else:
                    st.warning("Could not generate coefficients comparison")

                # ============================================================
                # RESULTS DISPLAY: Coefficient Charts (one per Y)
                # ============================================================
                st.markdown("---")
                st.markdown("### 📊 Coefficient Analysis by Response")
                st.info("Individual coefficient plots for each response variable")

                # ============================================================
                # ALWAYS USE ORIGINAL COEFFICIENT VALUES (NO NORMALIZATION)
                # ============================================================
                st.caption("📊 Using original coefficient values - Y-axis range adjusts dynamically per chart")

                # Always use original coefficient values (normalization disabled)
                coef_normalized_df_for_charts = None
                max_values_dict_for_charts = None

                # Create vertical stack of coefficient charts
                for y_var in y_vars:
                    model = models_dict[y_var]

                    if 'error' in model:
                        st.warning(f"⚠️ {y_var}: Model fitting failed")
                        continue

                    if 'coefficients' not in model:
                        st.warning(f"⚠️ {y_var}: No coefficients available")
                        continue

                    # Display chart for this Y variable (always using original values)
                    _display_coefficients_barplot_multidoe(
                        model,
                        y_var,
                        x_vars,
                        normalize=False,
                        coef_normalized_df=coef_normalized_df_for_charts,
                        max_values_dict=max_values_dict_for_charts
                    )

                    # ============================================================
                    # NEW: Display Model Equation
                    # ============================================================
                    st.markdown("#### 📐 Model Equation")

                    try:
                        # Generate numeric equation with subscripts
                        equation_numeric = generate_model_equation(
                            coefficients=model['coefficients'],
                            variable_names=x_vars,
                            y_variable_name=y_var,
                            use_subscripts=True,
                            show_coefficient_names=False,
                            decimals=4,
                            use_hat=True
                        )

                        # Generate symbolic equation (with coefficient names)
                        equation_symbolic = generate_model_equation(
                            coefficients=model['coefficients'],
                            variable_names=x_vars,
                            y_variable_name=y_var,
                            use_subscripts=True,
                            show_coefficient_names=True,
                            decimals=4,
                            use_hat=True
                        )

                        # Display both equations in expandable sections
                        col_eq1, col_eq2 = st.columns([5, 1])

                        with col_eq1:
                            st.markdown("**Numeric Equation:**")
                            st.code(equation_numeric, language="text")

                            with st.expander("Show symbolic form (coefficient names)"):
                                st.code(equation_symbolic, language="text")

                        with col_eq2:
                            # Optional: Add copy button if pyperclip is available
                            if st.button("📋", key=f"copy_eq_{y_var}", help="Copy numeric equation"):
                                try:
                                    import pyperclip
                                    pyperclip.copy(equation_numeric)
                                    st.success("✅")
                                except ImportError:
                                    st.info("Install pyperclip to enable copy")

                    except Exception as e:
                        st.warning(f"Could not generate equation: {str(e)}")

                    # Add spacing between charts
                    st.markdown("---")

                # NEW: Debug Section for Verification
                with st.expander("🔍 Model Details (Debug)", expanded=False):
                    st.markdown("### Model Structure Verification")
                    st.info("Use this section to verify that all models have identical structure")

                    for y_var, model in models_dict.items():
                        if 'error' not in model:
                            st.markdown(f"#### {y_var}")

                            col_debug1, col_debug2 = st.columns(2)

                            with col_debug1:
                                # Show coefficient vector
                                if 'coefficients' in model:
                                    coef_series = model['coefficients']
                                    st.write(f"**Coefficients:** {coef_series.shape[0]} terms")
                                    st.write(f"**Names:** {coef_series.index.tolist()}")

                                    # Show first few values
                                    st.dataframe(
                                        coef_series.to_frame('Value').head(10),
                                        use_container_width=True
                                    )

                            with col_debug2:
                                # Show model matrix shape
                                if 'X' in model:
                                    st.write(f"**X matrix shape:** {model['X'].shape}")
                                    if hasattr(model['X'], 'columns'):
                                        st.write(f"**X columns ({len(model['X'].columns)}):**")
                                        st.write(model['X'].columns.tolist())
                                    else:
                                        st.write("**X columns:** N/A (not a DataFrame)")

                                # Show term names if available
                                if 'term_names' in model:
                                    st.write(f"**Term names ({len(model['term_names'])}):**")
                                    st.write(model['term_names'])

                            # Show R²_adj and key metrics
                            st.write(f"**R²_adj:** {model.get('r_squared_adj', model.get('r_squared', 'N/A')):.6f}")
                            st.write(f"**RMSE:** {model.get('rmse', 'N/A'):.6f}")

                            st.divider()

                # Result 4: Error Summary
                errors = [y for y, model in models_dict.items() if 'error' in model]
                if errors:
                    st.markdown("### ⚠️ Failed Models")
                    for y in errors:
                        st.error(f"**{y}:** {models_dict[y]['error']}")

            except Exception as e:
                st.error(f"❌ Error fitting models: {str(e)}")
                import traceback
                with st.expander("🐛 Error details"):
                    st.code(traceback.format_exc())

    # ========================================================================
    # TAB 2: MODEL DIAGNOSTICS
    # ========================================================================
    with tab2:
        if not data_loaded:
            st.warning("⚠️ No data loaded")
        elif 'multi_doe_models' not in st.session_state:
            st.warning("⚠️ No models fitted")
            st.info("💡 Go to **Model Computation** tab to fit models")
        else:
            st.markdown("## 📊 Model Diagnostics")

            # Response selector
            col_sel1, col_sel2 = st.columns([2, 1])
            with col_sel1:
                selected_y = st.selectbox(
                    "Select response variable to analyze:",
                    st.session_state.multi_doe_y_vars,
                    help="Choose which model's diagnostics to view",
                    key="multidoe_diag_selected_y"
                )

            with col_sel2:
                model = st.session_state.multi_doe_models[selected_y]
                if 'error' not in model:
                    r2_adj = model.get('r_squared_adj', model.get('r_squared', np.nan))
                    st.metric("R²_adj", f"{r2_adj:.4f}" if not np.isnan(r2_adj) else "N/A")

            st.markdown("---")

            # Show diagnostics for selected model
            model = st.session_state.multi_doe_models[selected_y]
            if 'error' in model:
                st.error(f"❌ Model for {selected_y} failed: {model['error']}")
            else:
                show_model_diagnostics_ui_multidoe(
                    model_results=model,
                    X=model['X'],
                    y=model['y'],
                    y_name=selected_y
                )

    # ========================================================================
    # TAB 3: SURFACE ANALYSIS
    # ========================================================================
    with tab3:
        if not data_loaded:
            st.warning("⚠️ No data loaded")
        elif 'multi_doe_models' not in st.session_state:
            st.warning("⚠️ No models fitted")
        else:
            show_surface_analysis_ui_multidoe(
                st.session_state.multi_doe_models,
                st.session_state.multi_doe_x_vars,
                st.session_state.multi_doe_y_vars
            )

    # ========================================================================
    # TAB 4: PREDICTIONS
    # ========================================================================
    with tab4:
        if not data_loaded:
            st.warning("⚠️ No data loaded")
        elif 'multi_doe_models' not in st.session_state:
            st.warning("⚠️ No models fitted")
        else:
            show_predictions_ui_multidoe(
                st.session_state.multi_doe_models,
                st.session_state.multi_doe_x_vars,
                st.session_state.multi_doe_y_vars,
                data
            )

    # ========================================================================
    # TAB 5: MULTI-CRITERIA DECISION MAKING
    # ========================================================================
    with tab5:
        if not data_loaded:
            st.warning("⚠️ No data loaded")
        elif 'multi_doe_models' not in st.session_state:
            st.warning("⚠️ No models fitted")
        else:
            show_pareto_ui_multidoe(
                st.session_state.multi_doe_models,
                st.session_state.multi_doe_x_vars,
                st.session_state.multi_doe_y_vars
            )

    # ========================================================================
    # TAB 6: GENERATE MATRIX (Standalone - reuse from mlr_doe)
    # ========================================================================
    with tab6:
        st.markdown("## Experimental Design Matrix Generator")
        st.markdown("*Standalone tool - works independently of loaded data or fitted models*")
        st.info("Create custom experimental designs without needing to load data first")
        try:
            show_candidate_points_ui()
        except Exception as e:
            st.error(f"❌ Error in Generate Matrix tab: {str(e)}")
            import traceback
            with st.expander("🐛 Error Details"):
                st.code(traceback.format_exc())

    # ========================================================================
    # TAB 7: EXPORT
    # ========================================================================
    with tab7:
        if not data_loaded:
            st.warning("⚠️ No data loaded")
        elif 'multi_doe_models' not in st.session_state:
            st.warning("⚠️ No models fitted")
        else:
            show_export_ui_multidoe(
                st.session_state.multi_doe_models,
                st.session_state.multi_doe_x_vars,
                st.session_state.multi_doe_y_vars
            )


if __name__ == "__main__":
    show()
