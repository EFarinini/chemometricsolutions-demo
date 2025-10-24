"""
MLR Model Computation UI
Equivalent to DOE_model_computation.r
Complete model fitting workflow with term selection, diagnostics, and statistical tests
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy import stats


def show_model_computation_ui(data, dataset_name):
    """
    Display the MLR Model Computation UI

    Args:
        data: DataFrame with experimental data
        dataset_name: name of the current dataset
    """
    # Import helper functions from parent module
    from mlr_doe import (
        create_model_matrix, fit_mlr_model, detect_replicates,
        detect_central_points, display_term_selection_ui,
        create_term_selection_matrix, build_model_formula, design_analysis
    )

    st.markdown("## 🔧 MLR Model Computation")
    st.markdown("*Equivalent to DOE_model_computation.r*")

    # DATA PREVIEW SECTION
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

    # Variable and sample selection
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📊 Variable Selection")

        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()

        if not numeric_columns:
            st.error("❌ No numeric columns found!")
            return

        # X variables
        x_vars = st.multiselect(
            "Select X variables (predictors):",
            numeric_columns,
            default=numeric_columns[:min(3, len(numeric_columns))],
            key="mlr_x_vars_widget"
        )

        # Y variable (OPTIONAL - for design analysis mode)
        remaining_cols = [col for col in numeric_columns if col not in x_vars]
        if remaining_cols:
            y_options = ["(None - Design Analysis Only)"] + remaining_cols
            y_var_selected = st.selectbox(
                "Select Y variable (response - optional):",
                y_options,
                key="mlr_y_var_widget",
                help="Select '(None)' for design screening without response variable"
            )

            # Parse selection
            if y_var_selected == "(None - Design Analysis Only)":
                y_var = None
                st.info("**Design Analysis Mode**: No Y variable - will analyze design matrix only (VIF, Leverage, Dispersion)")
            else:
                y_var = y_var_selected
        else:
            st.warning("⚠️ Select at least one X variable")
            return

        # Show selected variables info
        if x_vars and y_var:
            x_vars_str = [str(var) for var in x_vars]
            st.info(f"Model: {y_var} ~ {' + '.join(x_vars_str)}")
        elif x_vars and y_var is None:
            x_vars_str = [str(var) for var in x_vars]
            st.info(f"Design Matrix: {' + '.join(x_vars_str)}")

    with col2:
        st.markdown("### 🎯 Sample Selection")

        # Sample selection options
        sample_selection_mode = st.radio(
            "Select samples:",
            ["Use all samples", "Select by index", "Select by range"],
            key="sample_selection_mode"
        )

        if sample_selection_mode == "Use all samples":
            selected_samples = data.index.tolist()
            st.success(f"Using all {len(selected_samples)} samples")

        elif sample_selection_mode == "Select by index":
            sample_input = st.text_input(
                "Enter sample indices (1-based, comma-separated or ranges):",
                value=f"1-{data.shape[0]}",
                help="Examples: 1,2,5-10,15 or 1-20"
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
                start_idx = st.number_input("From sample:", 1, len(data), 1)
            with col_range2:
                end_idx = st.number_input("To sample:", start_idx, len(data), len(data))

            selected_samples = data.index[start_idx-1:end_idx].tolist()
            st.success(f"Selected {len(selected_samples)} samples (rows {start_idx}-{end_idx})")

        # Show selected samples preview
        if len(selected_samples) < len(data):
            with st.expander("Preview selected samples"):
                st.dataframe(data.loc[selected_samples].head(10), use_container_width=True)

    st.markdown("---")

    # Interactive Term Selection UI
    if not x_vars:
        st.warning("Please select X variables first")
        return

    st.markdown("### 🎛️ Model Configuration")

    # Basic model configuration checkboxes
    col_config1, col_config2 = st.columns(2)

    with col_config1:
        include_intercept = st.checkbox(
            "Include intercept",
            value=True,
            help="Include b0 constant term in the model"
        )

    with col_config2:
        include_higher_terms = st.checkbox(
            "Include higher-order terms",
            value=False,
            help="Include interactions and quadratic terms (polynomial model)"
        )

    st.markdown("---")

    # Conditional term selection matrix
    if include_higher_terms:
        # Display term selection matrix with validation
        try:
            term_matrix, selected_terms = display_term_selection_ui(x_vars, key_prefix="mlr_tab1")

            # Validate that we got valid results
            if term_matrix is None or selected_terms is None:
                st.error("Error: Term selection matrix failed to initialize")
                # Create fallback
                term_matrix = create_term_selection_matrix(x_vars)
                selected_terms = {
                    'linear': x_vars.copy(),
                    'interactions': [f"{x_vars[i]}*{x_vars[j]}" for i in range(len(x_vars)) for j in range(i+1, len(x_vars))],
                    'quadratic': [f"{var}^2" for var in x_vars]
                }
                st.info("Using default term selection: all linear, interaction, and quadratic terms")

        except Exception as e:
            st.error(f"Error in term selection: {str(e)}")
            # Create fallback
            term_matrix = create_term_selection_matrix(x_vars)
            selected_terms = {
                'linear': x_vars.copy(),
                'interactions': [f"{x_vars[i]}*{x_vars[j]}" for i in range(len(x_vars)) for j in range(i+1, len(x_vars))],
                'quadratic': [f"{var}^2" for var in x_vars]
            }
            st.info("Using default term selection: all linear, interaction, and quadratic terms")

        # Show term counts
        col_terms1, col_terms2, col_terms3 = st.columns(3)
        with col_terms1:
            st.metric("Linear Terms", len(selected_terms['linear']))
        with col_terms2:
            st.metric("Interactions", len(selected_terms['interactions']))
        with col_terms3:
            st.metric("Quadratic Terms", len(selected_terms['quadratic']))
    else:
        # Linear model only - no higher-order terms
        selected_terms = {
            'linear': x_vars.copy(),
            'interactions': [],
            'quadratic': []
        }
        term_matrix = create_term_selection_matrix(x_vars)
        # Set all to zero except diagonal for linear terms
        for i in range(len(x_vars)):
            for j in range(len(x_vars)):
                term_matrix.iloc[i, j] = 0

        st.info(f"**Linear Model Only:** {len(x_vars)} linear terms (no interactions or quadratic terms)")

    st.markdown("---")

    # Model Settings
    st.markdown("### ⚙️ Additional Model Settings")

    col_set1, col_set2 = st.columns(2)

    with col_set1:
        exclude_central_points = st.checkbox(
            "Exclude central points (0,0,0...)",
            value=False,
            help="Central points are typically used only for validation in factorial designs"
        )

    with col_set2:
        # Variance method selector
        variance_method = st.radio(
            "Variance estimation method:",
            ["Residuals", "Independent measurements"],
            help="Choose how to estimate model error variance"
        )

        run_cv = st.checkbox("Run cross-validation", value=True,
                            help="Leave-one-out CV (only for n≤100)")

    # Display model formula (only if Y variable is selected)
    st.markdown("---")
    if y_var:
        st.markdown("### 📐 Model Formula")

        try:
            formula = build_model_formula(y_var, selected_terms, include_intercept)
            st.code(formula, language="text")
        except Exception as e:
            st.warning(f"Could not generate formula display: {str(e)}")
            st.code(f"{y_var} = b0 + b1·X + ... (formula generation error)", language="text")
    else:
        st.markdown("### 📐 Design Structure")
        # Show design structure without Y variable
        terms_list = []
        if include_intercept:
            terms_list.append("Intercept")
        terms_list.extend(selected_terms['linear'])
        terms_list.extend(selected_terms['interactions'])
        terms_list.extend(selected_terms['quadratic'])

        st.code(f"Design Matrix Terms: {', '.join(terms_list)}", language="text")

    # Summary of selected terms
    total_terms = len(selected_terms['linear']) + len(selected_terms['interactions']) + len(selected_terms['quadratic'])
    if include_intercept:
        total_terms += 1

    if y_var:
        st.info(f"""
        **Model Summary:**
        - Total parameters: {total_terms}
        - Response variable: {y_var}
        - Variance method: {variance_method}
        """)
    else:
        st.info(f"""
        **Design Analysis Summary:**
        - Total design terms: {total_terms}
        - Mode: Design screening (no response variable)
        - Analysis: Dispersion matrix, VIF, Leverage
        """)

    # Use term_matrix as interaction_matrix for backward compatibility
    interaction_matrix = term_matrix

    # Debug output
    with st.expander("🐛 Debug Info (click to expand)"):
        st.write("**Selected X variables:**", x_vars)
        st.write("**Selected Y variable:**", y_var)
        st.write("**Term matrix shape:**", term_matrix.shape if term_matrix is not None else "None")
        st.write("**Selected terms:**", selected_terms)
        st.write("**Number of samples:**", len(selected_samples))

    # Fit model or analyze design button
    button_text = "🚀 Fit MLR Model" if y_var else "🔍 Analyze Design"
    button_type = "primary"

    if st.button(button_text, type=button_type):
        try:
            # Prepare data with selected samples
            X_data = data.loc[selected_samples, x_vars].copy()

            # Handle Y variable (if present)
            if y_var:
                y_data = data.loc[selected_samples, y_var].copy()
                # Remove missing values
                valid_idx = ~(X_data.isnull().any(axis=1) | y_data.isnull())
                X_data = X_data[valid_idx]
                y_data = y_data[valid_idx]

                if len(X_data) < len(x_vars) + 1:
                    st.error("❌ Not enough samples for model fitting!")
                    return

                st.info(f"ℹ️ Using {len(X_data)} samples after removing missing values")
            else:
                # Design analysis mode - no Y variable
                # Remove rows with missing X values only
                valid_idx = ~X_data.isnull().any(axis=1)
                X_data = X_data[valid_idx]
                y_data = None

                if len(X_data) < len(x_vars):
                    st.error("❌ Not enough samples for design analysis!")
                    return

                st.info(f"ℹ️ Using {len(X_data)} samples for design analysis")

            # Detect and optionally exclude central points
            central_points = detect_central_points(X_data)

            if central_points:
                st.info(f"🎯 Detected {len(central_points)} central point(s) at indices: {[i+1 for i in central_points]}")

                if exclude_central_points:
                    # Store original indices before filtering
                    central_samples_original = X_data.index[central_points].tolist()

                    # Remove central points from modeling data
                    X_data = X_data.drop(X_data.index[central_points])
                    if y_data is not None:
                        y_data = y_data.drop(y_data.index[central_points])

                    st.warning(f"⚠️ Excluded {len(central_points)} central point(s) from analysis")
                    st.info(f"ℹ️ Using {len(X_data)} samples (excluding central points)")

                    # Store excluded central points for later validation (only if Y exists)
                    if y_var:
                        st.session_state.mlr_central_points = {
                            'X': data.loc[central_samples_original, x_vars],
                            'y': data.loc[central_samples_original, y_var],
                            'indices': central_samples_original
                        }
                else:
                    st.info("ℹ️ Central points included in the analysis")

            # Validate term_matrix
            if interaction_matrix is None:
                st.error("❌ Term selection matrix is None! Cannot create model.")
                st.info("This is a bug - please report with your data configuration.")
                return

            # Debug: Show what we're passing to create_model_matrix
            st.write("**DEBUG - Model Matrix Creation:**")
            st.write(f"- X_data shape: {X_data.shape}")
            st.write(f"- Interaction matrix shape: {interaction_matrix.shape}")
            st.write(f"- Include intercept: {include_intercept}")
            st.write("- Selected terms from UI:", selected_terms)

            # Create model matrix
            with st.spinner("Creating model matrix..."):
                X_model, term_names = create_model_matrix(
                    X_data,
                    include_intercept=include_intercept,
                    include_interactions=True,  # Always True - term_matrix controls selection
                    include_quadratic=True,  # Always True - term_matrix controls selection
                    interaction_matrix=interaction_matrix
                )

            st.success(f"✅ Model matrix created: {X_model.shape[0]} × {X_model.shape[1]}")
            st.write(f"**Model terms:** {term_names}")

            # BRANCH: Model fitting vs Design analysis
            if y_var is not None:
                # ===== MODEL FITTING MODE (Y variable present) =====
                with st.spinner("Fitting MLR model..."):
                    model_results = fit_mlr_model(X_model, y_data, return_diagnostics=run_cv)

                if model_results is None:
                    return

                # Store results
                st.session_state.mlr_model = model_results
                st.session_state.mlr_y_var = y_var
                st.session_state.mlr_x_vars = x_vars

                st.success("✅ MLR model fitted successfully!")

                # Show model results (calling the display function)
                _display_model_results(
                    model_results, y_var, x_vars, data, selected_samples,
                    central_points, exclude_central_points, X_data, y_data
                )

            else:
                # ===== DESIGN ANALYSIS MODE (No Y variable) =====
                with st.spinner("Analyzing design matrix..."):
                    # Detect replicates in X data only (for experimental variance)
                    replicate_info = detect_replicates(X_data, pd.Series(np.zeros(len(X_data)), index=X_data.index))

                    # Run design analysis
                    design_results = design_analysis(X_model, X_data, replicate_info)

                if design_results is None:
                    return

                st.success("✅ Design analysis completed successfully!")

                # Display design analysis results
                _display_design_analysis_results(design_results, x_vars, X_data)

        except Exception as e:
            st.error(f"❌ Error fitting model: {str(e)}")
            import traceback
            if st.checkbox("Show debug info"):
                st.code(traceback.format_exc())


def _display_model_results(model_results, y_var, x_vars, data, selected_samples,
                           central_points, exclude_central_points, X_data, y_data):
    """
    Display complete model results with diagnostics and statistical tests

    GENERIC IMPLEMENTATION - Works with ANY dataset structure:
    - With or without replicates
    - With or without central points
    - Any number of samples and variables

    ALWAYS DISPLAYS:
    - R², RMSE (model quality)
    - VIF (multicollinearity)
    - Leverage (influential points)
    - Coefficients with significance tests
    - Cross-validation (if enabled)

    CONDITIONALLY DISPLAYS (if data structure allows):
    - Replicate analysis (if replicates exist)
    - Lack of fit test (if replicates exist)
    - Factor effects F-test (if replicates exist)
    - Central point validation (if central points excluded)
    """
    # Import helper function
    from mlr_doe import detect_replicates

    # DEBUG: Show what keys are in model_results
    with st.expander("🔍 Model Results Debug Info"):
        st.write("**Available keys in model_results:**")
        st.write(list(model_results.keys()))
        st.write("**Model results summary:**")
        for key, value in model_results.items():
            if isinstance(value, (int, float, np.integer, np.floating)):
                st.write(f"- {key}: {value}")
            elif isinstance(value, (pd.Series, pd.DataFrame)):
                st.write(f"- {key}: {type(value).__name__} with shape {value.shape}")
            elif isinstance(value, np.ndarray):
                st.write(f"- {key}: numpy array with shape {value.shape}")
            else:
                st.write(f"- {key}: {type(value)}")

    # Show number of experiments used for fitting
    st.info(f"📊 **Model fitted using {model_results['n_samples']} experiments** (after excluding central points if selected)")

    # ===== ALWAYS: Basic Model Quality =====
    st.markdown("### 📈 Model Quality Summary")

    summary_col1, summary_col2 = st.columns(2)

    with summary_col1:
        if 'r_squared' in model_results:
            var_explained_pct = model_results['r_squared'] * 100
            st.metric("% Explained Variance (R²)", f"{var_explained_pct:.2f}%")

    with summary_col2:
        if 'rmse' in model_results:
            st.metric("Std Dev of Residuals (RMSE)", f"{model_results['rmse']:.4f}")

    # ========== AUTOMATIC REPLICATE DETECTION ==========
    # ALWAYS use ALL original data (including central points) for experimental variability calculation
    all_X_data = data.loc[selected_samples, x_vars].copy()
    all_y_data = data.loc[selected_samples, y_var].copy()
    all_valid_idx = ~(all_X_data.isnull().any(axis=1) | all_y_data.isnull())
    all_X_data = all_X_data[all_valid_idx]
    all_y_data = all_y_data[all_valid_idx]

    replicate_info_full = detect_replicates(all_X_data, all_y_data)

    # ===== CONDITIONAL: Replicate Analysis (only if replicates exist) =====
    if replicate_info_full:
        _display_replicate_analysis(replicate_info_full, model_results, central_points,
                                    exclude_central_points, y_data, all_y_data)
    else:
        st.info("ℹ️ No replicates detected - pure experimental error cannot be estimated")

    # ===== CONDITIONAL: Central Points Validation (only if excluded) =====
    if central_points and exclude_central_points:
        _display_central_points_validation(central_points)

    # ===== CONDITIONAL: Model Data Replicates Check =====
    replicate_info = detect_replicates(X_data, y_data)
    if replicate_info:
        _display_model_data_replicates(replicate_info, replicate_info_full)

    # ===== ALWAYS: Statistical Analysis Summary (adapts to available data) =====
    _display_statistical_summary(model_results, all_y_data, y_data, central_points,
                                 exclude_central_points, replicate_info_full)

    # ===== ALWAYS: Dispersion Matrix, VIF, Leverage =====
    _display_model_summary(model_results)

    # ===== CONDITIONAL: Error Comparison (only if replicates exist) =====
    if replicate_info and 'rmse' in model_results:
        _display_error_comparison(model_results, replicate_info)

    # ===== ALWAYS: Coefficients Table =====
    _display_coefficients_table(model_results)

    # ===== ALWAYS: Coefficients Bar Plot =====
    _display_coefficients_barplot(model_results, y_var)

    # ===== ALWAYS: Cross-Validation Results (if CV was run) =====
    if 'q2' in model_results:
        st.markdown("### 🔄 Cross-Validation Results")

        cv_col1, cv_col2 = st.columns(2)
        with cv_col1:
            st.metric("RMSECV", f"{model_results['rmsecv']:.4f}")
        with cv_col2:
            st.metric("Q² (LOO-CV)", f"{model_results['q2']:.4f}")


def _display_replicate_analysis(replicate_info_full, model_results, central_points,
                                exclude_central_points, y_data, all_y_data):
    """Display experimental variability analysis from replicates"""
    st.markdown("### 🔬 Experimental Variability (Pure Error)")
    st.info("""
    **Pure experimental error** estimated from replicate measurements
    (including ALL points - central points always included for experimental error calculation).
    This represents the baseline measurement variability.
    """)

    rep_col1, rep_col2, rep_col3, rep_col4 = st.columns(4)

    with rep_col1:
        st.metric("Replicate Groups", replicate_info_full['n_replicate_groups'])
    with rep_col2:
        st.metric("Total Replicates", replicate_info_full['total_replicates'])
    with rep_col3:
        st.metric("Pooled Std Dev (σ_exp)", f"{replicate_info_full['pooled_std']:.4f}")
    with rep_col4:
        st.metric("Pure Error DOF", replicate_info_full['pooled_dof'])

    with st.expander("📋 Replicate Groups Details"):
        rep_data = []
        for i, group in enumerate(replicate_info_full['group_stats'], 1):
            rep_data.append({
                'Group': i,
                'Samples': ', '.join([str(idx+1) for idx in group['indices']]),
                'N': group['n_replicates'],
                'Mean Y': f"{group['mean']:.4f}",
                'Std Dev': f"{group['std']:.4f}",
                'Variance': f"{group['variance']:.6f}",
                'DOF': group['dof']
            })

        rep_df = pd.DataFrame(rep_data)
        st.dataframe(rep_df, use_container_width=True)

        st.markdown(f"""
        **Pooled Standard Deviation Formula:**

        σ_pooled = √[Σ(s²ᵢ × dfᵢ) / Σ(dfᵢ)]

        Where s²ᵢ is the variance of group i and dfᵢ is its degrees of freedom.

        **Result:** σ_exp = {replicate_info_full['pooled_std']:.4f}
        (from {replicate_info_full['pooled_dof']} degrees of freedom)
        """)

    # Statistical tests
    _display_statistical_tests(model_results, replicate_info_full, central_points,
                               exclude_central_points, y_data, all_y_data)


def _display_statistical_tests(model_results, replicate_info_full, central_points,
                               exclude_central_points, y_data, all_y_data):
    """Display statistical tests for model quality"""
    st.markdown("---")
    st.markdown("### 📊 Statistical Analysis of Model Quality")

    # 1. DoE Factor Variability vs Experimental Variability
    st.markdown("#### 1️⃣ DoE Factor Variability vs Experimental Variability")

    if 'var_y' in model_results:
        # Determine which data to use for DoE variance
        if central_points and exclude_central_points:
            var_y_doe = np.var(y_data, ddof=1)
            dof_y_doe = len(y_data) - 1

            st.info(f"""
            **DoE Variability**: Calculated from {len(y_data)} DoE experimental points
            (central points excluded as they don't contribute to factor-induced variation).
            """)
        else:
            var_y_doe = model_results['var_y']
            dof_y_doe = len(all_y_data) - 1

            st.info("""
            **DoE Variability**: Calculated from all experimental points
            (central points included in model).
            """)

        # F-test: σ²_DoE / σ²_exp
        f_global = var_y_doe / replicate_info_full['pooled_variance']
        f_crit_global = stats.f.ppf(0.95, dof_y_doe, replicate_info_full['pooled_dof'])
        p_global = 1 - stats.f.cdf(f_global, dof_y_doe, replicate_info_full['pooled_dof'])

        test_col1, test_col2, test_col3 = st.columns(3)

        with test_col1:
            st.metric("DoE Variance (σ²_DoE)", f"{var_y_doe:.6f}")
            st.metric("DOF", dof_y_doe)

        with test_col2:
            st.metric("Experimental Variance (σ²_exp)", f"{replicate_info_full['pooled_variance']:.6f}")
            st.metric("DOF", replicate_info_full['pooled_dof'])

        with test_col3:
            st.metric("F-statistic", f"{f_global:.2f}")
            st.metric("p-value", f"{p_global:.4f}")

        if p_global < 0.05:
            st.success(f"✅ DoE factors induce significant variation in response (p={p_global:.4f})")
            st.info("The experimental factors have meaningful effects on the response variable.")
        else:
            st.warning(f"⚠️ DoE factor effects not significantly different from experimental noise (p={p_global:.4f})")
            st.info("The factors may have weak effects or the experimental error is too large.")

        # Show variance ratio
        variance_ratio = var_y_doe / replicate_info_full['pooled_variance']
        st.markdown(f"""
        **Variance Ratio**: σ²_DoE / σ²_exp = {variance_ratio:.2f}

        - Ratio > 4: Strong factor effects
        - Ratio 2-4: Moderate factor effects
        - Ratio < 2: Weak factor effects
        """)

    # 2. Lack of Fit test
    st.markdown("---")
    st.markdown("#### 2️⃣ Lack of Fit Test (Model Adequacy)")
    st.info("""
    **F-test**: Compares model residual variance vs pure experimental variance.
    - H₀: Model is adequate (σ²_model = σ²_exp)
    - H₁: Significant lack of fit (σ²_model > σ²_exp)
    """)

    if 'rmse' in model_results:
        lof_col1, lof_col2, lof_col3 = st.columns(3)

        with lof_col1:
            st.metric("Model RMSE", f"{model_results['rmse']:.4f}")
            st.caption(f"Variance: {model_results['var_res']:.6f}")
            st.caption(f"DOF: {model_results['dof']}")

        with lof_col2:
            st.metric("Experimental Std Dev", f"{replicate_info_full['pooled_std']:.4f}")
            st.caption(f"Variance: {replicate_info_full['pooled_variance']:.6f}")
            st.caption(f"DOF: {replicate_info_full['pooled_dof']}")

        with lof_col3:
            # F = variance_model / variance_exp
            f_lof = model_results['var_res'] / replicate_info_full['pooled_variance']
            f_crit = stats.f.ppf(0.95, model_results['dof'], replicate_info_full['pooled_dof'])
            p_lof = 1 - stats.f.cdf(f_lof, model_results['dof'], replicate_info_full['pooled_dof'])

            st.metric("F-statistic", f"{f_lof:.2f}")
            st.caption(f"F-crit (95%): {f_crit:.2f}")
            st.caption(f"p-value: {p_lof:.4f}")

        # Unified interpretation with ratio
        ratio = model_results['rmse'] / replicate_info_full['pooled_std']

        st.markdown("---")

        result_col1, result_col2 = st.columns([1, 3])

        with result_col1:
            st.metric("RMSE / σ_exp", f"{ratio:.2f}")

        with result_col2:
            if p_lof > 0.05:
                st.success(f"✅ No significant Lack of Fit (p={p_lof:.4f})")
                if ratio < 1.2:
                    st.info("🎯 Model error ≈ experimental error - excellent fit")
                elif ratio < 2.0:
                    st.info("✅ Model error is reasonable")
                else:
                    st.warning("⚠️ Model error exceeds experimental error despite non-significant test")
            else:
                st.error(f"❌ Significant Lack of Fit detected (p={p_lof:.4f})")
                st.warning("""
                **Model inadequate!** Consider:
                - Adding missing interaction or quadratic terms
                - Checking for outliers or influential points
                - Data transformations (log, sqrt, etc.)
                - Verifying model assumptions
                """)
    else:
        st.warning("Insufficient data for Lack of Fit test")


def _display_central_points_validation(central_points):
    """Display central points validation section"""
    st.markdown("---")
    st.markdown("### 🎯 Central Points Validation")

    st.info(f"""
    **{len(central_points)} central point(s)** excluded from model fitting - reserved for validation.
    These points assess model adequacy and curvature effects at the center of the experimental domain.
    """)

    if 'mlr_central_points' in st.session_state:
        central_X = st.session_state.mlr_central_points['X']
        central_y = st.session_state.mlr_central_points['y']

        # Calculate central point statistics
        central_mean = central_y.mean()
        central_std = central_y.std(ddof=1) if len(central_y) > 1 else 0

        central_stats_col1, central_stats_col2, central_stats_col3 = st.columns(3)

        with central_stats_col1:
            st.metric("Central Points Count", len(central_y))
        with central_stats_col2:
            st.metric("Mean Response", f"{central_mean:.4f}")
        with central_stats_col3:
            if len(central_y) > 1:
                st.metric("Std Dev", f"{central_std:.4f}")
            else:
                st.metric("Std Dev", "N/A (single point)")

        with st.expander("📋 Central Points Details"):
            central_display = pd.DataFrame({
                'Sample': [str(idx) for idx in st.session_state.mlr_central_points['indices']],
                'Observed Y': central_y.values
            })

            for col in central_X.columns:
                central_display[col] = central_X[col].values

            st.dataframe(central_display, use_container_width=True)

        st.info("""
        **Central Point Validation**: Use these points for model validation in the Predictions tab.
        They help assess curvature and lack of fit at the experimental center.
        """)


def _display_model_data_replicates(replicate_info, replicate_info_full):
    """Display replicates found in the model data"""
    st.markdown("### 🔬 Experimental Replicates in Model Data")

    rep_col1, rep_col2, rep_col3, rep_col4 = st.columns(4)

    with rep_col1:
        st.metric("Replicate Groups", replicate_info['n_replicate_groups'])
    with rep_col2:
        st.metric("Total Replicates", replicate_info['total_replicates'])
    with rep_col3:
        st.metric("Pooled Std Dev", f"{replicate_info['pooled_std']:.4f}")
    with rep_col4:
        st.metric("Replicate DOF", replicate_info['pooled_dof'])

    with st.expander("📋 Model Data Replicate Groups Details"):
        rep_data = []
        for i, group in enumerate(replicate_info['group_stats'], 1):
            rep_data.append({
                'Group': i,
                'Samples': ', '.join([str(idx+1) for idx in group['indices']]),
                'N': group['n_replicates'],
                'Mean Y': f"{group['mean']:.4f}",
                'Std Dev': f"{group['std']:.4f}",
                'DOF': group['dof']
            })

        rep_df = pd.DataFrame(rep_data)
        st.dataframe(rep_df, use_container_width=True)

    st.info(f"""
    **Model Data Experimental Error** = {replicate_info['pooled_std']:.4f}
    (from {replicate_info['pooled_dof']} degrees of freedom)

    This represents the experimental error in the data actually used for modeling.
    """)

    # Compare model replicates vs full replicates
    if replicate_info['pooled_std'] != replicate_info_full['pooled_std']:
        st.warning(f"""
        **Note**: Model data experimental error ({replicate_info['pooled_std']:.4f}) differs from
        full dataset experimental error ({replicate_info_full['pooled_std']:.4f}).
        This occurs when central point replicates are excluded from modeling.
        """)


def _display_statistical_summary(model_results, all_y_data, y_data, central_points,
                                 exclude_central_points, replicate_info_full):
    """
    Display statistical analysis summary - FULLY GENERIC VERSION

    Dynamically builds summary based on available metrics.
    Never assumes any specific keys exist except those explicitly checked.

    ALWAYS SHOWS (if available):
    - R², RMSE, DOF, parameters
    - Coefficients, p-values (shown elsewhere)
    - VIF (shown in model summary)

    CONDITIONALLY SHOWS (only if keys exist):
    - Pure error and Lack of Fit (if replicates detected)
    - Central points validation (if excluded)
    - Cross-validation Q², RMSECV (if 'q2' in results)
    """
    st.markdown("---")
    st.markdown("### 📋 Statistical Analysis Summary")

    # Build summary text dynamically based on available data
    summary_parts = []

    # ===== ALWAYS: Data Structure =====
    summary_parts.append(f"""
    📊 **Data Structure:**
    - Total samples: {len(all_y_data)}
    - Model samples: {len(y_data)}
    - Central points: {len(central_points) if central_points else 0}""")

    if replicate_info_full:
        summary_parts.append(f"    - Replicate groups: {replicate_info_full['n_replicate_groups']}")
    else:
        summary_parts.append("    - Replicate groups: 0 (no replicates detected)")

    # ===== CONDITIONAL: Model Diagnostics (check each key) =====
    diagnostics_lines = ["", "    🎯 **Model Diagnostics:**"]

    if 'r_squared' in model_results:
        diagnostics_lines.append(f"    - R² (explained variance): {model_results['r_squared']:.4f}")

    if 'rmse' in model_results:
        diagnostics_lines.append(f"    - RMSE (model error): {model_results['rmse']:.4f}")

    if 'dof' in model_results:
        diagnostics_lines.append(f"    - Degrees of freedom: {model_results['dof']}")

    if 'n_features' in model_results:
        diagnostics_lines.append(f"    - Number of parameters: {model_results['n_features']}")

    # Add diagnostics if at least one metric was found
    if len(diagnostics_lines) > 2:
        summary_parts.append("\n".join(diagnostics_lines))

    # ===== CONDITIONAL: Cross-Validation (only if keys exist) =====
    if 'q2' in model_results and 'rmsecv' in model_results:
        summary_parts.append(f"    - Q² (cross-validation): {model_results['q2']:.4f}")
        summary_parts.append(f"    - RMSECV: {model_results['rmsecv']:.4f}")

    # ===== CONDITIONAL: Experimental Error Analysis (only if replicates exist) =====
    if replicate_info_full and 'rmse' in model_results:
        error_ratio = model_results['rmse'] / replicate_info_full['pooled_std']
        summary_parts.append(f"""
    🔬 **Experimental Error (from replicates):**
    - Pure error: σ_exp = {replicate_info_full['pooled_std']:.4f} (DOF = {replicate_info_full['pooled_dof']})
    - Error ratio: RMSE/σ_exp = {error_ratio:.2f}""")

        # Interpret error ratio
        if error_ratio < 1.2:
            summary_parts.append("    - ✅ Excellent: Model error ≈ experimental error")
        elif error_ratio < 2.0:
            summary_parts.append("    - ✅ Good: Model error is reasonable")
        else:
            summary_parts.append("    - ⚠️ Warning: Model error exceeds experimental error")

        # ===== CONDITIONAL: Factor Effects F-test (only with replicates AND var_y) =====
        if 'var_y' in model_results or len(y_data) > 1:
            # Calculate DoE variance
            var_y_doe = model_results.get('var_y', 0)
            dof_y_doe = len(all_y_data) - 1

            # Recalculate if central points were excluded
            if central_points and exclude_central_points:
                var_y_doe = np.var(y_data, ddof=1)
                dof_y_doe = len(y_data) - 1

            if var_y_doe > 0:  # Only proceed if variance is valid
                f_global = var_y_doe / replicate_info_full['pooled_variance']
                p_global = 1 - stats.f.cdf(f_global, dof_y_doe, replicate_info_full['pooled_dof'])
                variance_ratio = var_y_doe / replicate_info_full['pooled_variance']

                summary_parts.append(f"""
    📈 **Factor Effects:**
    - DoE variance: σ²_DoE = {var_y_doe:.6f}
    - F-test p-value: {p_global:.4f}
    - Variance amplification: {variance_ratio:.1f}×""")

                # Interpret variance ratio
                if variance_ratio > 4:
                    summary_parts.append("    - ✅ Strong factor effects")
                elif variance_ratio > 2:
                    summary_parts.append("    - ✅ Moderate factor effects")
                else:
                    summary_parts.append("    - ⚠️ Weak factor effects")

    elif replicate_info_full and 'rmse' not in model_results:
        # Replicates exist but RMSE is missing
        summary_parts.append("""
    🔬 **Experimental Error (from replicates):**
    - Pure error: Available from replicates
    - Error ratio: Cannot calculate (RMSE not available)""")

    else:
        # No replicates case
        summary_parts.append("""
    🔬 **Experimental Error:**
    - No replicates detected - pure error cannot be estimated
    - Model quality assessed using R², RMSE, and cross-validation only""")

    # ===== CONDITIONAL: Central Points Validation (only if excluded) =====
    if central_points and exclude_central_points:
        if 'mlr_central_points' in st.session_state:
            central_mean = st.session_state.mlr_central_points['y'].mean()
            summary_parts.append(f"""
    🎯 **Central Points:**
    - Excluded from model: {len(central_points)} points
    - Reserved for validation
    - Mean response: {central_mean:.4f}""")

    # Combine all parts and display
    summary_text = "\n".join(summary_parts)
    st.info(summary_text)


def _display_model_summary(model_results):
    """
    Display model summary: Dispersion Matrix, VIF, Leverage

    ALWAYS SHOWS (if available):
    - Dispersion Matrix (X'X)^-1
    - VIF (Variance Inflation Factors) - multicollinearity check
    - Leverage (hat values) - influential points
    """
    st.markdown("### 📋 Model Summary")

    # ===== CONDITIONAL: Dispersion Matrix =====
    if 'XtX_inv' in model_results and 'X' in model_results:
        st.markdown("#### Dispersion Matrix (X'X)^-1")
        try:
            dispersion_df = pd.DataFrame(
                model_results['XtX_inv'],
                index=model_results['X'].columns,
                columns=model_results['X'].columns
            )
            st.dataframe(dispersion_df.round(4), use_container_width=True)

            trace = np.trace(model_results['XtX_inv'])
            st.info(f"**Trace of Dispersion Matrix:** {trace:.4f}")
        except Exception as e:
            st.warning(f"⚠️ Could not display dispersion matrix: {str(e)}")

    # ===== CONDITIONAL: VIF (only if key exists) =====
    if 'vif' in model_results and model_results['vif'] is not None:
        st.markdown("#### Variance Inflation Factors (VIF)")
        try:
            vif_df = model_results['vif'].to_frame('VIF')
            vif_df_clean = vif_df[~vif_df.index.str.contains('Intercept', case=False, na=False)]
            vif_df_clean = vif_df_clean.dropna()

            if not vif_df_clean.empty:
                def interpret_vif(vif_val):
                    if vif_val <= 1:
                        return "✅ No correlation"
                    elif vif_val <= 2:
                        return "✅ OK"
                    elif vif_val <= 4:
                        return "⚠️ Good"
                    elif vif_val <= 8:
                        return "⚠️ Acceptable"
                    else:
                        return "❌ High multicollinearity"

                vif_df_clean['Interpretation'] = vif_df_clean['VIF'].apply(interpret_vif)
                st.dataframe(vif_df_clean.round(4), use_container_width=True)

                st.info("""
                **VIF Interpretation:**
                - VIF = 1: No correlation
                - VIF < 2: OK
                - VIF < 4: Good
                - VIF < 8: Acceptable
                - VIF > 8: High multicollinearity (problematic)
                """)
            else:
                st.info("VIF not applicable for this model")
        except Exception as e:
            st.warning(f"⚠️ Could not display VIF: {str(e)}")
    else:
        st.info("ℹ️ VIF not calculated for this model")

    # ===== CONDITIONAL: Leverage (only if key exists) =====
    if 'leverage' in model_results and model_results['leverage'] is not None:
        st.markdown("#### Leverage of Experimental Points")
        try:
            leverage_series = pd.Series(
                model_results['leverage'],
                index=range(1, len(model_results['leverage']) + 1)
            )
            st.dataframe(leverage_series.to_frame('Leverage').T.round(4), use_container_width=True)
            st.info(f"**Maximum Leverage:** {model_results['leverage'].max():.4f}")
        except Exception as e:
            st.warning(f"⚠️ Could not display leverage: {str(e)}")
    else:
        st.info("ℹ️ Leverage not calculated for this model")


def _display_error_comparison(model_results, replicate_info):
    """Display comparison between model error and experimental error"""
    st.markdown("#### 🎯 Model vs Experimental Error Comparison")

    comparison_col1, comparison_col2, comparison_col3 = st.columns(3)

    with comparison_col1:
        st.metric("Model RMSE", f"{model_results['rmse']:.4f}")

    with comparison_col2:
        st.metric("Experimental Std Dev", f"{replicate_info['pooled_std']:.4f}")

    with comparison_col3:
        ratio = model_results['rmse'] / replicate_info['pooled_std']
        st.metric("RMSE / Exp. Std Dev", f"{ratio:.2f}")

    if ratio < 1.2:
        st.success("✅ Model error is close to experimental error - excellent fit!")
    elif ratio < 2.0:
        st.info("ℹ️ Model error is reasonable compared to experimental error")
    else:
        st.warning("⚠️ Model error significantly exceeds experimental error - consider additional terms or transformation")


def _display_coefficients_table(model_results):
    """Display coefficients table with statistics"""
    st.markdown("### 📊 Model Coefficients")

    try:
        # Validate that coefficients exist
        if 'coefficients' not in model_results or model_results['coefficients'] is None:
            st.error("❌ Coefficients data not available in model results")
        else:
            coef_df = pd.DataFrame({'Coefficient': model_results['coefficients']})

            # Check if ALL statistical keys exist
            has_statistics = (
                'se_coef' in model_results and model_results['se_coef'] is not None and
                't_stats' in model_results and model_results['t_stats'] is not None and
                'p_values' in model_results and model_results['p_values'] is not None and
                'ci_lower' in model_results and model_results['ci_lower'] is not None and
                'ci_upper' in model_results and model_results['ci_upper'] is not None
            )

            if has_statistics:
                # Add all statistical columns
                coef_df['Std. Error'] = model_results['se_coef']
                coef_df['t-statistic'] = model_results['t_stats']
                coef_df['p-value'] = model_results['p_values']
                coef_df['CI Lower'] = model_results['ci_lower']
                coef_df['CI Upper'] = model_results['ci_upper']

                def add_stars(p):
                    if p <= 0.001:
                        return '***'
                    elif p <= 0.01:
                        return '**'
                    elif p <= 0.05:
                        return '*'
                    else:
                        return ''

                coef_df['Sig.'] = coef_df['p-value'].apply(add_stars)

                st.dataframe(coef_df.round(4), use_container_width=True)
                st.info("Significance codes: *** p≤0.001, ** p≤0.01, * p≤0.05")
            else:
                # Fallback: Show only coefficients
                st.dataframe(coef_df.round(4), use_container_width=True)
                st.warning("⚠️ Statistical information (standard errors, p-values, confidence intervals) not available")
                st.info("This may occur when degrees of freedom ≤ 0 (not enough samples for the model complexity)")

    except Exception as e:
        st.error(f"❌ Error displaying coefficients: {str(e)}")
        import traceback
        with st.expander("🐛 Full error traceback"):
            st.code(traceback.format_exc())


def _display_coefficients_barplot(model_results, y_var):
    """Display coefficients bar plot"""
    st.markdown("#### Coefficients Bar Plot")

    coefficients = model_results['coefficients']
    coef_no_intercept = coefficients[coefficients.index != 'Intercept']
    coef_names = coef_no_intercept.index.tolist()

    if len(coef_names) == 0:
        st.warning("No coefficients to plot (model contains only intercept)")
    else:
        colors = []
        for name in coef_names:
            if '*' in name:
                n_asterisks = name.count('*')
                colors.append('cyan' if n_asterisks > 1 else 'green')
            elif '^2' in name or '^' in name:
                colors.append('cyan')
            else:
                colors.append('red')

        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=coef_names,
            y=coef_no_intercept.values,
            marker_color=colors,
            marker_line_color='black',
            marker_line_width=1,
            name='Coefficients',
            showlegend=False
        ))

        if 'ci_lower' in model_results and 'ci_upper' in model_results:
            ci_lower = model_results['ci_lower'][coef_no_intercept.index].values
            ci_upper = model_results['ci_upper'][coef_no_intercept.index].values

            error_minus = coef_no_intercept.values - ci_lower
            error_plus = ci_upper - coef_no_intercept.values

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

        if 'p_values' in model_results:
            p_values = model_results['p_values'][coef_no_intercept.index].values
            for i, (name, coef, p) in enumerate(zip(coef_names, coef_no_intercept.values, p_values)):
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
                        yshift=10 if coef >= 0 else -10
                    )

        fig.update_layout(
            title=f"Coefficients - {y_var}",
            xaxis_title="Term",
            yaxis_title="Coefficient Value",
            height=500,
            xaxis={'tickangle': 45},
            showlegend=False,
            yaxis=dict(zeroline=True, zerolinewidth=2, zerolinecolor='gray')
        )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        **Color legend:**
        - Red = Linear terms
        - Green = Two-term interactions
        - Cyan = Quadratic terms
        """)

        st.info("Significance markers: *** p≤0.001, ** p≤0.01, * p≤0.05")


def _display_design_analysis_results(design_results, x_vars, X_data):
    """
    Display design analysis results (without Y variable)

    Shows:
    - Design matrix information
    - Dispersion Matrix (X'X)^-1
    - VIF (multicollinearity check)
    - Leverage (influential points)
    - Prediction confidence intervals (if replicates exist)

    Args:
        design_results: dict from design_analysis()
        x_vars: list of X variable names
        X_data: original X data (before model matrix expansion)
    """
    st.markdown("---")
    st.markdown("## 📊 Design Analysis Results")
    st.info("**Design Screening Mode**: Analyzing experimental design quality without response variable")

    # ===== DESIGN MATRIX INFO =====
    st.markdown("### 📐 Design Matrix Information")

    info_col1, info_col2, info_col3 = st.columns(3)

    with info_col1:
        st.metric("Experimental Points", design_results['n_samples'])

    with info_col2:
        st.metric("Model Terms", design_results['n_features'])

    with info_col3:
        st.metric("Degrees of Freedom", design_results['dof'])

    if design_results['dof'] <= 0:
        st.error(f"""
        ❌ **Insufficient degrees of freedom!**
        - You have {design_results['n_samples']} experimental points
        - The model requires {design_results['n_features']} parameters
        - Need at least {design_results['n_features'] + 1} points to fit a model
        """)
        st.warning("**Recommendation**: Add more experimental points or reduce model complexity")
    elif design_results['dof'] < 5:
        st.warning(f"""
        ⚠️ **Low degrees of freedom** (DOF = {design_results['dof']})
        - Model will have limited statistical power
        - Consider adding more experimental points for robust estimation
        """)
    else:
        st.success(f"✅ Adequate degrees of freedom (DOF = {design_results['dof']})")

    # ===== DISPERSION MATRIX =====
    st.markdown("---")
    st.markdown("### 📊 Dispersion Matrix (X'X)^-1")
    st.info("""
    The dispersion matrix shows the variance-covariance structure of model parameters.
    - **Diagonal elements**: Variance of coefficient estimates (smaller is better)
    - **Off-diagonal elements**: Correlation between coefficients
    """)

    try:
        dispersion_df = pd.DataFrame(
            design_results['XtX_inv'],
            index=design_results['X'].columns,
            columns=design_results['X'].columns
        )
        st.dataframe(dispersion_df.round(6), use_container_width=True)

        trace = np.trace(design_results['XtX_inv'])
        determinant = np.linalg.det(design_results['XtX_inv'])

        disp_metric_col1, disp_metric_col2 = st.columns(2)
        with disp_metric_col1:
            st.metric("Trace", f"{trace:.4f}", help="Sum of diagonal elements - measure of total variance")
        with disp_metric_col2:
            st.metric("Determinant", f"{determinant:.2e}", help="Measure of design efficiency")

    except Exception as e:
        st.error(f"❌ Could not display dispersion matrix: {str(e)}")

    # ===== VIF (Multicollinearity) =====
    st.markdown("---")
    st.markdown("### 🔍 Variance Inflation Factors (VIF)")
    st.info("""
    **VIF measures multicollinearity** between predictor variables:
    - VIF = 1: No correlation
    - VIF < 2: Excellent
    - VIF < 4: Good
    - VIF < 8: Acceptable
    - VIF > 8: **High multicollinearity** (problematic)
    """)

    if 'vif' in design_results and design_results['vif'] is not None:
        vif_df = design_results['vif'].to_frame('VIF')
        vif_df_clean = vif_df[~vif_df.index.str.contains('Intercept', case=False, na=False)]
        vif_df_clean = vif_df_clean.dropna()

        if not vif_df_clean.empty:
            def interpret_vif(vif_val):
                if vif_val <= 1:
                    return "✅ No correlation"
                elif vif_val <= 2:
                    return "✅ Excellent"
                elif vif_val <= 4:
                    return "✅ Good"
                elif vif_val <= 8:
                    return "⚠️ Acceptable"
                else:
                    return "❌ High multicollinearity"

            vif_df_clean['Interpretation'] = vif_df_clean['VIF'].apply(interpret_vif)
            st.dataframe(vif_df_clean.round(4), use_container_width=True)

            # Check for problematic VIF
            max_vif = vif_df_clean['VIF'].max()
            if max_vif > 8:
                st.error(f"""
                ❌ **High multicollinearity detected!** (Max VIF = {max_vif:.2f})
                Consider:
                - Removing correlated variables
                - Using centered/orthogonal coding
                - Reducing interaction/quadratic terms
                """)
            elif max_vif > 4:
                st.warning(f"⚠️ Moderate multicollinearity detected (Max VIF = {max_vif:.2f})")
            else:
                st.success(f"✅ Low multicollinearity (Max VIF = {max_vif:.2f})")
        else:
            st.info("VIF not applicable (single term model)")
    else:
        st.info("ℹ️ VIF not calculated")

    # ===== LEVERAGE =====
    st.markdown("---")
    st.markdown("### 📍 Leverage of Experimental Points")
    st.info("""
    **Leverage** measures how influential each experimental point is on model predictions:
    - Higher leverage = more influential point
    - Average leverage = p/n (where p = parameters, n = samples)
    - Points with leverage > 2×average may be influential
    """)

    if 'leverage' in design_results and design_results['leverage'] is not None:
        leverage_series = pd.Series(
            design_results['leverage'],
            index=range(1, len(design_results['leverage']) + 1),
            name='Leverage'
        )

        # Display as horizontal table (transposed)
        st.dataframe(leverage_series.to_frame().T.round(4), use_container_width=True)

        avg_leverage = design_results['n_features'] / design_results['n_samples']
        max_leverage = design_results['leverage'].max()
        max_leverage_idx = np.argmax(design_results['leverage']) + 1

        lev_col1, lev_col2, lev_col3 = st.columns(3)

        with lev_col1:
            st.metric("Average Leverage", f"{avg_leverage:.4f}")

        with lev_col2:
            st.metric("Max Leverage", f"{max_leverage:.4f}")

        with lev_col3:
            st.metric("Max at Point", max_leverage_idx)

        # Check for high leverage points
        high_leverage_threshold = 2 * avg_leverage
        high_leverage_points = np.where(design_results['leverage'] > high_leverage_threshold)[0] + 1

        if len(high_leverage_points) > 0:
            st.warning(f"""
            ⚠️ **{len(high_leverage_points)} point(s) with high leverage** (> {high_leverage_threshold:.4f}):
            Points: {', '.join(map(str, high_leverage_points))}

            High leverage points have strong influence on model predictions.
            """)
        else:
            st.success("✅ No unusually high leverage points detected")

    # ===== EXPERIMENTAL VARIANCE (if replicates exist) =====
    if 'experimental_std' in design_results:
        st.markdown("---")
        st.markdown("### 🔬 Experimental Variability")
        st.info("""
        **Pure experimental error** estimated from replicate measurements.
        This can be used to assess prediction uncertainty even without fitting a model.
        """)

        exp_col1, exp_col2, exp_col3 = st.columns(3)

        with exp_col1:
            st.metric("Experimental Std Dev (σ_exp)", f"{design_results['experimental_std']:.4f}")

        with exp_col2:
            st.metric("Degrees of Freedom", design_results['experimental_dof'])

        with exp_col3:
            st.metric("t-critical (95%)", f"{design_results['t_critical']:.3f}")

        st.markdown("#### Prediction Standard Errors")
        st.info("Standard error for predictions at each experimental point (σ_exp × √leverage)")

        se_pred_series = pd.Series(
            design_results['prediction_se'],
            index=range(1, len(design_results['prediction_se']) + 1),
            name='Prediction SE'
        )

        st.dataframe(se_pred_series.to_frame().T.round(4), use_container_width=True)

        st.success("""
        ✅ **Prediction confidence intervals can be computed** once a response variable is measured.
        The prediction uncertainty will be: ±{:.4f} × t-critical for each point.
        """.format(design_results['experimental_std']))

    else:
        st.markdown("---")
        st.info("""
        ℹ️ **No experimental replicates detected** in the design matrix.
        Prediction uncertainty cannot be estimated without replicate measurements or a fitted model.
        """)
