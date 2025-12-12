"""
Mixture Design Analysis Page
Simplex Centroid Design and Scheffe Polynomial Models
Equivalent to CAT DOE_model_computation_mixt.r and DOE_prediction_mixt.r scripts

This module provides complete mixture design functionality:
- Simplex centroid design generation
- Pseudo-component transformation
- Scheffe polynomial model fitting
- Ternary/quaternary response surfaces
- Component effect analysis
"""

import streamlit as st
import numpy as np
import pandas as pd
from scipy import stats

# Import mixture design functions from mixture_utils
try:
    from mixture_utils.mixture_generation import (
        generate_simplex_centroid_design,
        apply_pseudo_components,
        apply_constraints,
        apply_d_optimal_design,
        validate_mixture_design
    )
    from mixture_utils.mixture_computation import (
        fit_mixture_model,
        statistical_summary,
        scheffe_polynomial_prediction,
        calculate_component_effects,
        detect_mixture_design,
        apply_mixture_transformation
    )
    from mixture_utils.mixture_diagnostics import show_mixture_diagnostics_ui
    from mixture_utils.mixture_surface import show_mixture_surface_ui
    from mixture_utils.mixture_predictions import show_mixture_predictions_ui
    from mixture_utils.mixture_export import show_mixture_export_ui
    from mixture_utils.mixture_ui_utils import (
        show_constraint_editor_ui,
        show_pseudo_component_ui,
        show_design_selection_ui,
        show_model_formula_builder,
        show_design_summary_ui,
        plot_ternary_design
    )
    MIXTURE_UTILS_AVAILABLE = True
except ImportError as e:
    MIXTURE_UTILS_AVAILABLE = False
    IMPORT_ERROR = str(e)


# ============================================================================
# MAIN PAGE FUNCTION
# ============================================================================

def show():
    """Display the Mixture Design Analysis page"""

    # ✅ INITIALIZE SESSION STATE AT FUNCTION START (before tabs)
    # This ensures session_state persists correctly across all tabs
    if 'mixture_design_matrix' not in st.session_state:
        st.session_state.mixture_design_matrix = None
    if 'mixture_model_results' not in st.session_state:
        st.session_state.mixture_model_results = None
    if 'mixture_component_names' not in st.session_state:
        st.session_state.mixture_component_names = []
    if 'mixture_n_components' not in st.session_state:
        st.session_state.mixture_n_components = 3
    if 'mixture_y_var' not in st.session_state:
        st.session_state.mixture_y_var = None
    if 'fitted_model_degree' not in st.session_state:
        st.session_state.fitted_model_degree = None

    st.markdown("# Mixture Design Analysis")
    st.markdown("*Simplex Centroid Design with Scheffe Polynomial Models*")

    # Check if mixture_utils is available
    if not MIXTURE_UTILS_AVAILABLE:
        st.error("❌ **Mixture Design utilities not available**")
        st.info(f"Import error: {IMPORT_ERROR}")
        st.markdown("---")
        st.markdown("### Implementation Status")
        st.warning("The mixture_utils module is being created. Please run the implementation steps.")
        return

    # Check if data is loaded (but don't block the entire page)
    data_loaded = 'current_data' in st.session_state and st.session_state.current_data is not None
    data = st.session_state.current_data if data_loaded else None

    # Create 6 tabs (Design Generation moved to Homepage)
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Model Computation",          # TAB 1: Fit Scheffe polynomial
        "Model Diagnostics",          # TAB 2: Leverage, residuals, VIF
        "Response Surface",           # TAB 3: Ternary plots
        "Predictions & Effects",      # TAB 4: Component effects
        "Multi-Criteria Decision",    # TAB 5: Pareto for mixtures
        "Extract & Export"            # TAB 6: Export results
    ])

    # ========================================================================
    # TAB 1: MODEL COMPUTATION (Workspace-Based)
    # ========================================================================
    with tab1:
        st.markdown("## 🧮 Scheffe Polynomial Model Computation")
        st.markdown("*Complete mixture model fitting with statistical analysis*")

        st.info("""
        **How to provide mixture data:**
        - **From Workspace:** Load experimental data from Data Handling (below)
        - **Design Generation:** Use Homepage → Generate DoE for theoretical designs
        """)

        # ═══════════════════════════════════════════════════════════════════════════
        # SECTION 1: LOAD DATA FROM WORKSPACE
        # ═══════════════════════════════════════════════════════════════════════════
        st.markdown("## 📂 Step 1: Load Mixture Data from Workspace")

        # Import workspace utilities
        try:
            from workspace_utils import display_workspace_dataset_selector
            workspace_available = True
        except ImportError:
            workspace_available = False
            st.error("❌ workspace_utils not available")

        design_df = None
        y_data = None

        if workspace_available:
            # Use workspace selector
            dataset_result = display_workspace_dataset_selector(
                label="Select dataset with design matrix:",
                key="mixture_workspace_dataset_selector",
                help_text="Choose dataset from your workspace (Data Handling)",
                show_info=True
            )

            if dataset_result is not None:
                dataset_name, selected_data = dataset_result

                st.markdown("---")

                # ═══════════════════════════════════════════════════════════════════
                # SECTION 2: SELECT DESIGN MATRIX COLUMNS
                # ═══════════════════════════════════════════════════════════════════
                st.markdown("## 🎯 Step 2: Define Design Matrix (Component Columns)")

                st.info("""
                **Select mixture component columns:**
                - Components should sum to 1.0 (or 100%)
                - Typical: X1, X2, X3... or Component_A, Component_B, etc.
                - If they don't sum to 1.0, normalization will be offered
                """)

                # Get all numeric columns
                numeric_cols = selected_data.select_dtypes(include=['number']).columns.tolist()

                if len(numeric_cols) < 2:
                    st.error("❌ Need at least 2 numeric columns for mixture design")
                    st.stop()

                # Multi-select for design matrix columns
                design_cols = st.multiselect(
                    "Select component columns (must sum to 1.0)",
                    options=numeric_cols,
                    default=numeric_cols[:3] if len(numeric_cols) >= 3 else numeric_cols,
                    key="mixture_design_cols_selector",
                    help="Choose columns representing mixture components"
                )

                if len(design_cols) < 2:
                    st.warning("⚠️ Select at least 2 component columns")
                    st.stop()

                # Extract design matrix
                design_df = selected_data[design_cols].copy()

                # ═══════════════════════════════════════════════════════════════════
                # SECTION 3: VALIDATE & NORMALIZE
                # ═══════════════════════════════════════════════════════════════════
                st.markdown("### ✓ Validation")

                # Check if rows sum to 1.0
                row_sums = design_df.sum(axis=1)
                all_sum_to_one = np.allclose(row_sums, 1.0, atol=1e-6)

                col_val1, col_val2, col_val3 = st.columns(3)

                with col_val1:
                    st.metric("Components", len(design_cols))
                with col_val2:
                    st.metric("Experiments", len(design_df))
                with col_val3:
                    if all_sum_to_one:
                        st.metric("Sum Check", "✓ Valid")
                    else:
                        st.metric("Sum Check", "⚠️ Need normalization")

                # Show row sums
                with st.expander("📊 View row sums"):
                    sum_df = pd.DataFrame({
                        'Experiment': range(1, len(row_sums) + 1),
                        'Row Sum': row_sums,
                        'Valid': np.abs(row_sums - 1.0) < 1e-6
                    })
                    st.dataframe(
                        sum_df.style.format({'Row Sum': '{:.6f}'}),
                        use_container_width=True
                    )

                # Offer normalization if needed
                if not all_sum_to_one:
                    st.warning("⚠️ **Components don't sum to 1.0**")
                    st.info("Normalization: Each row will be divided by its sum")

                    if st.button("🔄 Normalize to sum=1.0", key="normalize_mixture_btn"):
                        design_df = design_df.div(row_sums, axis=0)
                        st.success("✓ Normalized! All rows now sum to 1.0")
                        st.rerun()
                else:
                    st.success("✓ All components sum to 1.0 (valid mixture design)")

                # Display design matrix preview
                st.markdown("### 📋 Design Matrix Preview")
                st.dataframe(
                    design_df.style.format("{:.6f}"),
                    use_container_width=True,
                    height=min(300, (len(design_df) + 1) * 35 + 3)
                )

                st.markdown("---")

                # ═══════════════════════════════════════════════════════════════════
                # SECTION 2.5: DATA PRE-PROCESSING (MIXTURE TRANSFORMATION)
                # ═══════════════════════════════════════════════════════════════════
                st.markdown("## 🔄 Step 2.5: Data Pre-Processing")

                # Auto-detect mixture design
                try:
                    mixture_detection = detect_mixture_design(design_df)

                    if mixture_detection['is_mixture']:
                        st.success(f"✅ {mixture_detection['reason']}")

                        # Display detection details
                        col_detect1, col_detect2 = st.columns(2)
                        with col_detect1:
                            st.metric("Components Detected", mixture_detection['n_components'])
                        with col_detect2:
                            component_list = ', '.join(mixture_detection['component_names'])
                            st.info(f"**Components**: {component_list}")

                        # Offer transformation checkbox
                        st.markdown("### 🔄 Pseudo-Component Transformation")

                        st.info("""
                        **Pseudo-Component Transformation:**

                        Rescales the mixture simplex to orthogonal [0, 1]^n coordinates:

                        **Formula:** PseudoCompi = (Xi - min_i) / (max_i - min_i)

                        **Where:**
                        - Xi = real value of component i
                        - min_i = minimum value of component i in the design
                        - max_i = maximum value of component i in the design

                        **Benefits:**
                        - Maps vertices to pure components: (1, 0, 0), (0, 1, 0), (0, 0, 1)
                        - Simplifies model interpretation
                        - Enables orthogonal analysis

                        **Note:** This is NOT Principal Component Analysis (PCA)!
                        "PseudoComp" refers to mixture design pseudo-components.
                        """)

                        apply_mixture_coding = st.checkbox(
                            "🔄 Apply pseudo-component transformation",
                            value=True,
                            key="apply_mixture_coding",
                            help="Transform to PseudoComp1, PseudoComp2, PseudoComp3... coordinates"
                        )

                        if apply_mixture_coding:
                            try:
                                # Apply transformation
                                coded_df = apply_mixture_transformation(
                                    design_df,
                                    mixture_detection['component_names']
                                )

                                # Store both versions in session state
                                st.session_state.mixture_design_matrix_original = design_df.copy()
                                st.session_state.mixture_design_matrix_coded = coded_df.copy()

                                # Show comparison
                                st.markdown("### 📊 Transformation Comparison")

                                # Create comparison DataFrame
                                col_compare1, col_compare2 = st.columns(2)

                                with col_compare1:
                                    st.markdown("**Original (Real Compositions):**")
                                    st.dataframe(
                                        design_df.head(5).style.format("{:.6f}"),
                                        use_container_width=True
                                    )

                                with col_compare2:
                                    st.markdown("**Pseudo-Components:**")
                                    st.dataframe(
                                        coded_df.head(5).style.format("{:.6f}"),
                                        use_container_width=True
                                    )

                                # Show mapping and statistics
                                st.markdown("**Transformation Statistics:**")

                                # Calculate transformation parameters
                                orig_data = design_df[mixture_detection['component_names']]
                                transform_stats = []
                                for i, comp_name in enumerate(mixture_detection['component_names']):
                                    min_val = orig_data[comp_name].min()
                                    max_val = orig_data[comp_name].max()
                                    range_val = max_val - min_val
                                    transform_stats.append({
                                        'Original Name': comp_name,
                                        'Pseudo-Comp Name': coded_df.columns[i],
                                        'Min (Real)': min_val,
                                        'Max (Real)': max_val,
                                        'Range': range_val
                                    })

                                stats_df = pd.DataFrame(transform_stats)
                                st.dataframe(
                                    stats_df.style.format({
                                        'Min (Real)': '{:.4f}',
                                        'Max (Real)': '{:.4f}',
                                        'Range': '{:.4f}'
                                    }),
                                    use_container_width=True,
                                    hide_index=True
                                )

                                st.caption("Formula: PseudoCompi = (Xi - Min) / Range")

                                # Use coded version for model fitting
                                design_df = coded_df
                                design_cols = coded_df.columns.tolist()

                                st.success("✅ **Using pseudo-component coordinates (PseudoComp1, PseudoComp2, PseudoComp3...) for model fitting**")

                            except Exception as e:
                                st.error(f"❌ Transformation failed: {str(e)}")
                                st.warning("Falling back to original data")
                                # Fall back to original
                                apply_mixture_coding = False

                        else:
                            st.info("ℹ️ Using original component names for model fitting")

                    else:
                        # Not a mixture design - show reason
                        st.info(f"ℹ️ {mixture_detection['reason']}")
                        st.warning("""
                        **Not detected as mixture design:**
                        - Transformation skipped
                        - Standard model fitting will be used
                        """)
                        apply_mixture_coding = False

                except Exception as e:
                    st.warning(f"⚠️ Mixture detection failed: {str(e)}")
                    st.info("Proceeding without transformation")
                    apply_mixture_coding = False

                st.markdown("---")

                # Store design matrix in session state
                st.session_state.mixture_design_matrix = design_df
                st.session_state.mixture_component_names = design_cols
                st.session_state.mixture_n_components = len(design_cols)

                # ═══════════════════════════════════════════════════════════════════
                # SECTION 4: SELECT RESPONSE VARIABLE
                # ═══════════════════════════════════════════════════════════════════
                st.markdown("## 📊 Step 3: Select Response Variable (Y)")

                st.info("""
                **Response variable:**
                - Select the experimental outcome you want to model
                - Must be numeric (e.g., Yield, Strength, Viscosity, etc.)
                """)

                # Get remaining numeric columns (excluding design columns)
                remaining_cols = [col for col in numeric_cols if col not in design_cols]

                if len(remaining_cols) == 0:
                    st.error("❌ No response columns available (all numeric columns used for design matrix)")
                    st.info("Your dataset should have: design matrix columns + at least 1 response column")
                    st.stop()

                response_var = st.selectbox(
                    "Response Variable (Y)",
                    options=remaining_cols,
                    key="mixture_response_selector",
                    help="Select the outcome variable to model"
                )

                y_data = selected_data[response_var].values

                # Show summary
                col_y1, col_y2, col_y3, col_y4 = st.columns(4)
                with col_y1:
                    st.metric("Response", response_var)
                with col_y2:
                    st.metric("Mean", f"{np.mean(y_data):.4f}")
                with col_y3:
                    st.metric("Std Dev", f"{np.std(y_data, ddof=1):.4f}")
                with col_y4:
                    st.metric("Range", f"{np.ptp(y_data):.4f}")

                st.markdown("---")

                # ═══════════════════════════════════════════════════════════════════
                # SECTION 5: MODEL SPECIFICATION
                # ═══════════════════════════════════════════════════════════════════
                st.markdown("## 🧬 Step 4: Model Specification")

                col_spec1, col_spec2 = st.columns([1.5, 1.5])

                with col_spec1:
                    st.markdown("### Scheffe Polynomial Degree")

                    degree = st.radio(
                        "Choose model complexity:",
                        options=["linear", "reduced_cubic", "quadratic", "cubic"],
                        captions=[
                            "Linear: Y = β₁X₁ + β₂X₂ + β₃X₃",
                            "Reduced Cubic: Y = Linear + β₁₂₃X₁X₂X₃",
                            "Quadratic: Y = Linear + β₁₂X₁X₂ + β₁₃X₁X₃ + β₂₃X₂X₃",
                            "Cubic (Full): Y = Linear + Quadratic + β₁₂₃X₁X₂X₃"
                        ],
                        index=2,  # Default to quadratic
                        key="mixture_model_degree"
                    )

                    component_names = st.session_state.get('mixture_component_names', design_df.columns.tolist())

                    # Model descriptions
                    model_descriptions = {
                        'linear': {
                            'terms': 'Pure components only',
                            'binary': False,
                            'ternary': False
                        },
                        'reduced_cubic': {
                            'terms': 'Pure components + Ternary interaction (skip binary)',
                            'binary': False,
                            'ternary': True
                        },
                        'quadratic': {
                            'terms': 'Pure components + Binary interactions',
                            'binary': True,
                            'ternary': False
                        },
                        'cubic': {
                            'terms': 'Pure components + Binary + Ternary interactions',
                            'binary': True,
                            'ternary': True
                        }
                    }

                    desc = model_descriptions.get(degree, model_descriptions['linear'])

                    st.markdown(f"""
                    **Selected:** {degree.upper().replace('_', ' ')}
                    - Pure components (vertices): estimated
                    - Binary interactions: {'estimated' if desc['binary'] else 'not estimated'}
                    - Ternary interactions: {'estimated' if desc['ternary'] else 'not estimated'}

                    📝 *{desc['terms']}*
                    """)

                with col_spec2:
                    st.markdown("### Model Information")

                    # Calculate model complexity
                    n_components = design_df.shape[1]
                    if degree == "linear":
                        n_terms = n_components
                    elif degree == "reduced_cubic":
                        # Linear + Ternary only (skip binary interactions)
                        n_terms = n_components
                        if n_components >= 3:
                            n_terms += (n_components * (n_components - 1) * (n_components - 2)) // 6
                    elif degree == "quadratic":
                        n_terms = n_components + (n_components * (n_components - 1)) // 2
                    else:  # cubic (full)
                        n_terms = n_components + (n_components * (n_components - 1)) // 2
                        if n_components >= 3:
                            n_terms += (n_components * (n_components - 1) * (n_components - 2)) // 6

                    dof = len(y_data) - n_terms

                    st.metric("Components", n_components)
                    st.metric("Model Terms", n_terms)

                    # ✓ ALLOW DOF = 0 (saturated model is OK for mixtures)
                    color = "🔴" if dof < 0 else "🟡" if dof == 0 else "🟢"
                    st.metric("DOF (n - p)", f"{color} {dof}")

                    # Only block if truly over-parameterized (DOF < 0)
                    if dof < 0:
                        st.error(f"❌ Over-parameterized! {len(y_data)} experiments with {n_terms} terms (need ≥ {n_terms})")
                        st.info("💡 Use simpler model (fewer components or lower degree)")
                        st.stop()

                    # Warnings for reduced diagnostic capability, but DON'T block
                    if dof == 0:
                        st.warning("⚠️ **Saturated Model (DOF = 0)**")
                        st.info("""
    **Your design perfectly matches the model complexity:**
    - All parameters are exactly identified
    - No residual variance to estimate
    - NO diagnostic plots available (residuals ≈ 0, R² = 1.0)
    - BUT: You can use model for predictions, sensitivity analysis, optimization

    **This is NORMAL and EXPECTED for Simplex Centroid designs!**
                        """)
                    elif dof < 3:
                        st.warning(f"⚠️ Very limited DOF ({dof}). Diagnostic power is limited.")
                        st.info("Consider adding replicates for better variance estimation")

                st.markdown("---")

                # ═══════════════════════════════════════════════════════════════════
                # SECTION 6: VARIANCE SPECIFICATION
                # ═══════════════════════════════════════════════════════════════════
                st.markdown("## 📈 Step 5: Experimental Variance Estimation")

                st.info("""
                **Where to estimate experimental error?**
                - **From residuals (LSQ):** Standard approach - variance estimated from model residuals
                - **From independent measurements:** If you have replicate standard deviation from external experiments
                """)

                variance_source = st.radio(
                    "Variance source",
                    options=["From residuals (LSQ)", "From independent measurements"],
                    key="mixture_variance_source"
                )

                # Store variance choice
                use_external_variance = (variance_source == "From independent measurements")

                if use_external_variance:
                    st.markdown("**External Variance Input:**")
                    col_var1, col_var2 = st.columns(2)

                    with col_var1:
                        rmsef_exp = st.number_input(
                            "Experimental standard deviation (σ)",
                            min_value=0.0,
                            value=0.01,
                            format="%.6f",
                            key="rmsef_exp_input"
                        )

                    with col_var2:
                        dof_exp = st.number_input(
                            "Degrees of freedom (DOF)",
                            min_value=1,
                            value=3,
                            step=1,
                            key="dof_exp_input"
                        )

                    st.info(f"Using external σ = {rmsef_exp:.6f} with DOF = {dof_exp}")
                else:
                    rmsef_exp = None
                    dof_exp = None

                st.markdown("---")

                # ═══════════════════════════════════════════════════════════════════
                # SECTION 7: FIT MODEL
                # ═══════════════════════════════════════════════════════════════════
                st.markdown("## 🚀 Step 6: Fit Model")

                col_fit1, col_fit2 = st.columns([2, 1])

                with col_fit1:
                    if st.button("🚀 Fit Scheffe Polynomial Model", type="primary", key="fit_mixture_model_btn"):
                        try:
                            with st.spinner("Fitting Scheffe polynomial model..."):
                                # Fit model
                                model_results = fit_mixture_model(design_df, y_data, degree=degree)

                                # Override variance if external provided
                                if use_external_variance:
                                    model_results['rmsef_exp'] = rmsef_exp
                                    model_results['dof_exp'] = dof_exp
                                    model_results['use_external_variance'] = True
                                else:
                                    model_results['use_external_variance'] = False

                                # Store in session state
                                st.session_state.mixture_model_results = model_results
                                st.session_state.fitted_model_degree = degree  # Different key to avoid conflict with widget
                                st.session_state.mixture_y_var = response_var

                            st.success("✅ Model fitted successfully!")
                            st.rerun()

                        except Exception as e:
                            st.error(f"❌ Model fitting failed: {str(e)}")
                            import traceback
                            with st.expander("🐛 Error details"):
                                st.code(traceback.format_exc())

                # ═══════════════════════════════════════════════════════════════════
                # SECTION 8: MODEL RESULTS & STATISTICS (if model fitted)
                # ═══════════════════════════════════════════════════════════════════

                if st.session_state.mixture_model_results is not None:
                    model_results = st.session_state.mixture_model_results

                    st.markdown("---")
                    st.markdown("## ✅ Model Fitting Results")

                    # Show DOF=0 capability notice if saturated model
                    if model_results['dof'] == 0:
                        st.info("✓ **Saturated model fitted (DOF=0)**")
                        col_cap1, col_cap2 = st.columns(2)
                        with col_cap1:
                            st.markdown("**✓ Available:**")
                            st.write("- Coefficients ✓")
                            st.write("- Predictions ✓")
                            st.write("- Response surface plots ✓")
                            st.write("- Component effects ✓")
                            st.write("- Optimization ✓")
                        with col_cap2:
                            st.markdown("**⚠️ NOT available (DOF=0):**")
                            st.write("- Residual plots (all ≈ 0)")
                            st.write("- R² check (always = 1.0)")
                            st.write("- Model diagnostics")
                            st.write("- Standard errors/CI")
                        st.markdown("---")
                    elif model_results['dof'] > 0:
                        st.success("✓ Model fitted with degrees of freedom for full diagnostics")

                    # Subsection 5.1: Dispersion Matrix (X'X)^-1 (following R script lines 138-145)
                    with st.expander("🔍 Dispersion Matrix (X'X)⁻¹"):
                        st.markdown("*Shows parameter covariance and precision*")

                        if 'XtX_inv' in model_results:
                            disp_df = pd.DataFrame(
                                model_results['XtX_inv'],
                                index=model_results['feature_names'],
                                columns=model_results['feature_names']
                            )

                            st.dataframe(
                                disp_df.style.format('{:.6f}'),
                                use_container_width=True
                            )

                            # Trace (sum of diagonal)
                            trace = np.trace(model_results['XtX_inv'])
                            st.metric("Trace (Σ diagonal)", f"{trace:.6f}")
                            st.caption("Lower trace = more precise parameters")

                    # Subsection 5.2: Leverage (following R script lines 149-154)
                    st.markdown("### 📈 Leverage of Experimental Points")

                    leverage = model_results['leverage']
                    max_leverage = np.max(leverage)
                    avg_leverage = np.mean(leverage)

                    col_lev1, col_lev2 = st.columns(2)

                    with col_lev1:
                        st.metric("Maximum Leverage", f"{max_leverage:.6f}")

                    with col_lev2:
                        st.metric("Average Leverage", f"{avg_leverage:.6f}")

                    # Leverage table
                    leverage_df = pd.DataFrame({
                        'Experiment': range(1, len(leverage) + 1),
                        'Leverage': leverage
                    })

                    st.dataframe(
                        leverage_df.style.format({'Leverage': '{:.6f}'}),
                        use_container_width=True,
                        height=min(300, (len(leverage) + 1) * 35 + 3)
                    )

                    st.markdown("---")

                    # Subsection 5.3: Model Summary Metrics
                    st.markdown("### 📊 Model Summary")

                    col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)

                    with col_m1:
                        st.metric("R²", f"{model_results['r_squared']:.6f}")
                    with col_m2:
                        q2 = model_results.get('q2', np.nan)
                        st.metric("Q² (CV)", f"{q2:.6f}")
                    with col_m3:
                        st.metric("RMSE", f"{model_results['rmse']:.6f}")
                    with col_m4:
                        rmsecv = model_results.get('rmsecv', np.nan)
                        st.metric("RMSECV", f"{rmsecv:.6f}")
                    with col_m5:
                        dof_used = model_results.get('dof_exp') if model_results.get('use_external_variance') else model_results['dof']
                        st.metric("DOF", dof_used)

                    # Variance explained
                    vary = np.var(model_results['y'], ddof=1)
                    pct_explained = (1 - (model_results['rmse']**2) / vary) * 100

                    st.info(f"""
                    **Variance of Y:** {vary:.6f}
                    **Standard deviation of residuals:** {model_results['rmse']:.6f}
                    **% Explained Variance:** {pct_explained:.2f}%
                    """)

                    st.markdown("---")

                    # Subsection 5.4: Coefficient Table (following R script lines 168-235)
                    st.markdown("### 📈 Scheffe Polynomial Coefficients")

                    coef_df = pd.DataFrame({
                        'Term': model_results['feature_names'],
                        'Coefficient': model_results['coefficients'],
                        'Std.dev.': model_results.get('se_coef', [np.nan] * len(model_results['coefficients'])),
                        'Conf.Int.': [
                            f"±{ci:.6f}" for ci in (model_results.get('ci_upper', model_results['coefficients']) -
                                                   model_results['coefficients'])
                        ] if 'ci_upper' in model_results else [''] * len(model_results['coefficients']),
                        'p-value': model_results.get('p_values', [np.nan] * len(model_results['coefficients'])),
                        'Sig.': ['***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
                                for p in model_results.get('p_values', [np.nan] * len(model_results['coefficients']))]
                    })

                    st.dataframe(
                        coef_df.style.format({
                            'Coefficient': '{:.6e}',
                            'Std.dev.': '{:.6e}',
                            'p-value': '{:.6f}'
                        }),
                        use_container_width=True
                    )

                    st.caption("Significance codes: *** p<0.001,  ** p<0.01,  * p<0.05")

                    st.markdown("---")

                    # Subsection 5.5: Fitted Values and Residuals (following R script lines 248-253)
                    st.markdown("### 📊 Fitted Values & Residuals")

                    col_fit1, col_fit2 = st.columns(2)

                    with col_fit1:
                        st.markdown("**Fitted Values:**")
                        fitted_df = pd.DataFrame({
                            'Exp': range(1, len(model_results['y_pred']) + 1),
                            'Observed': model_results['y'],
                            'Predicted': model_results['y_pred']
                        })
                        st.dataframe(
                            fitted_df.style.format({
                                'Observed': '{:.6f}',
                                'Predicted': '{:.6f}'
                            }),
                            use_container_width=True,
                            height=min(300, (len(fitted_df) + 1) * 35 + 3)
                        )

                    with col_fit2:
                        st.markdown("**Residuals:**")
                        residuals_df = pd.DataFrame({
                            'Exp': range(1, len(model_results['residuals']) + 1),
                            'Residuals': model_results['residuals']
                        })
                        st.dataframe(
                            residuals_df.style.format({'Residuals': '{:.6f}'}),
                            use_container_width=True,
                            height=min(300, (len(residuals_df) + 1) * 35 + 3)
                        )

                    st.markdown("---")

                    # Subsection 5.6: Cross-Validation Results (following R script lines 254-282)
                    st.markdown("### 🔄 Cross-Validation (Leave-One-Out)")

                    col_cv1, col_cv2, col_cv3 = st.columns(3)

                    with col_cv1:
                        st.metric("RMSECV", f"{model_results.get('rmsecv', np.nan):.6f}")

                    with col_cv2:
                        cv_var = np.var(model_results.get('cv_residuals', [0]), ddof=0) if 'cv_residuals' in model_results else 0
                        pct_cv_explained = (1 - cv_var / vary) * 100 if vary > 0 else 0
                        st.metric("% CV Explained Variance", f"{pct_cv_explained:.2f}%")

                    with col_cv3:
                        # R² vs Q² comparison
                        r2 = model_results['r_squared']
                        q2 = model_results.get('q2', 0)
                        diff = abs(r2 - q2)
                        st.metric("R² - Q² Gap", f"{diff:.4f}",
                                 delta="Good" if diff < 0.1 else "Check overfitting" if diff < 0.2 else "Overfitting!")

                    # CV predictions and residuals
                    if 'cv_predictions' in model_results:
                        col_cvdata1, col_cvdata2 = st.columns(2)

                        with col_cvdata1:
                            st.markdown("**CV Predicted Values:**")
                            cv_pred_df = pd.DataFrame({
                                'Exp': range(1, len(model_results['cv_predictions']) + 1),
                                'CV Predicted': model_results['cv_predictions']
                            })
                            st.dataframe(
                                cv_pred_df.style.format({'CV Predicted': '{:.6f}'}),
                                use_container_width=True,
                                height=min(250, (len(cv_pred_df) + 1) * 35 + 3)
                            )

                        with col_cvdata2:
                            st.markdown("**CV Residuals:**")
                            cv_res_df = pd.DataFrame({
                                'Exp': range(1, len(model_results['cv_residuals']) + 1),
                                'CV Residuals': model_results['cv_residuals']
                            })
                            st.dataframe(
                                cv_res_df.style.format({'CV Residuals': '{:.6f}'}),
                                use_container_width=True,
                                height=min(250, (len(cv_res_df) + 1) * 35 + 3)
                            )

                    st.info("✨ Model ready! Proceed to Tab 2 for diagnostics and Tab 3 for response surface visualization.")

    # ========================================================================
    # TAB 2: MODEL DIAGNOSTICS
    # ========================================================================
    with tab2:
        if st.session_state.mixture_model_results is None:
            st.warning("⚠️ **No mixture model fitted**")
            st.info("💡 Go to **Model Computation** tab to fit a model first")
        else:
            # show_mixture_diagnostics_ui handles both saturated (DOF=0) and non-saturated models
            # No need to call st.stop() - just let the function return
            show_mixture_diagnostics_ui(st.session_state.mixture_model_results)

    # ========================================================================
    # TAB 3: RESPONSE SURFACE
    # ========================================================================
    with tab3:
        if st.session_state.mixture_model_results is None:
            st.warning("⚠️ **No mixture model fitted**")
            st.info("💡 Go to **Model Computation** tab to fit a model first")
        else:
            show_mixture_surface_ui(
                st.session_state.mixture_model_results,
                st.session_state.mixture_design_matrix,
                data
            )

    # ========================================================================
    # TAB 4: PREDICTIONS & COMPONENT EFFECTS
    # ========================================================================
    with tab4:
        if st.session_state.mixture_model_results is None:
            st.warning("⚠️ **No mixture model fitted**")
            st.info("💡 Go to **Model Computation** tab to fit a model first")
        else:
            show_mixture_predictions_ui(
                st.session_state.mixture_model_results,
                st.session_state.mixture_design_matrix,
                data
            )

    # ========================================================================
    # TAB 5: MULTI-CRITERIA DECISION
    # ========================================================================
    with tab5:
        st.markdown("## ⚖️ Multi-Criteria Decision Making")
        st.markdown("*Pareto optimization for mixture compositions*")

        # ═══════════════════════════════════════════════════════════════════════════
        # CHECK MODEL FITTED
        # ═══════════════════════════════════════════════════════════════════════════

        if st.session_state.mixture_model_results is None:
            st.warning("⚠️ **No mixture model fitted**")
            st.info("👈 Go to **Model Computation** tab to fit a model first")
            st.stop()

        model_results = st.session_state.mixture_model_results
        component_names = model_results['component_names']

        st.markdown("### 🎯 Define Optimization Objectives")

        st.info("""
        **Pareto Optimization:**
        Find mixtures that optimize multiple responses simultaneously.
        A Pareto-optimal mixture cannot improve one objective without worsening another.
        """)

        st.markdown("---")

        # ═══════════════════════════════════════════════════════════════════════════
        # STEP 1: GRID SETTINGS
        # ═══════════════════════════════════════════════════════════════════════════

        st.markdown("### 📊 Step 1: Grid Configuration")

        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("**Grid Resolution:**")
            st.caption("Higher = finer Pareto front")

        with col2:
            opt_grid = st.slider(
                "Points per edge",
                min_value=20,
                max_value=100,
                value=50,
                step=5,
                key="pareto_grid_resolution"
            )

        st.markdown("---")

        # ═══════════════════════════════════════════════════════════════════════════
        # STEP 2: OBJECTIVE DEFINITION
        # ═══════════════════════════════════════════════════════════════════════════

        st.markdown("### 🎯 Step 2: Define Objectives")

        # For mixture design, we typically have 1 response variable
        # But we can add constraints on components as secondary objectives

        st.markdown("**Response Variable:**")
        st.markdown(f"Y = Predicted Response from model")

        col_obj1, col_obj2 = st.columns([2, 1])

        with col_obj1:
            st.markdown("**Primary Objective:**")

        with col_obj2:
            objective_y = st.radio(
                "Y Optimization",
                options=["Maximize Y", "Minimize Y"],
                key="pareto_objective_y"
            )

        st.markdown("---")

        # Component-based secondary objectives
        st.markdown("### 📋 Secondary Objectives (Optional)")

        st.info("""
        Add component constraints to find balanced compositions:
        - **Range constraint:** Keep component in [min, max]
        - **Minimize:** Reduce a component
        - **Maximize:** Increase a component
        """)

        secondary_objectives = {}

        col_sec1, col_sec2 = st.columns(2)

        with col_sec1:
            st.markdown("**Component 1 Constraint:**")
            comp_constraint_1 = st.selectbox(
                f"{component_names[0]} constraint",
                options=["None", "Minimize", "Maximize", "Range"],
                key="pareto_constraint_1"
            )

            if comp_constraint_1 == "Minimize":
                secondary_objectives[component_names[0]] = "minimize"
            elif comp_constraint_1 == "Maximize":
                secondary_objectives[component_names[0]] = "maximize"
            elif comp_constraint_1 == "Range":
                col_r1a, col_r1b = st.columns(2)
                with col_r1a:
                    min_val_1 = st.number_input(f"Min {component_names[0]}", 0.0, 1.0, 0.1, key="comp1_min")
                with col_r1b:
                    max_val_1 = st.number_input(f"Max {component_names[0]}", 0.0, 1.0, 0.9, key="comp1_max")

        with col_sec2:
            st.markdown("**Component 2 Constraint:**")
            comp_constraint_2 = st.selectbox(
                f"{component_names[1]} constraint",
                options=["None", "Minimize", "Maximize", "Range"],
                key="pareto_constraint_2"
            )

            if comp_constraint_2 == "Minimize":
                secondary_objectives[component_names[1]] = "minimize"
            elif comp_constraint_2 == "Maximize":
                secondary_objectives[component_names[1]] = "maximize"
            elif comp_constraint_2 == "Range":
                col_r2a, col_r2b = st.columns(2)
                with col_r2a:
                    min_val_2 = st.number_input(f"Min {component_names[1]}", 0.0, 1.0, 0.1, key="comp2_min")
                with col_r2b:
                    max_val_2 = st.number_input(f"Max {component_names[1]}", 0.0, 1.0, 0.9, key="comp2_max")

        st.markdown("---")

        # ═══════════════════════════════════════════════════════════════════════════
        # RUN PARETO OPTIMIZATION
        # ═══════════════════════════════════════════════════════════════════════════

        if st.button("🚀 Calculate Pareto Front", type="primary", key="run_pareto_btn"):
            with st.spinner("Calculating Pareto front..."):
                try:
                    from mixture_utils.mixture_computation import scheffe_polynomial_prediction

                    # Try importing pareto functions - if not available, use fallback
                    try:
                        from mlr_utils.pareto_optimization import calculate_pareto_front, plot_pareto_2d
                        pareto_available = True
                    except ImportError:
                        pareto_available = False

                    # Create grid
                    step = 1.0 / opt_grid
                    grid_points = []
                    grid_responses = []

                    for i in range(opt_grid + 1):
                        for j in range(opt_grid + 1 - i):
                            x1 = i * step
                            x2 = j * step
                            x3 = 1.0 - x1 - x2

                            if x3 >= -1e-10:
                                # Predict
                                pred = scheffe_polynomial_prediction(
                                    model_results,
                                    {component_names[0]: x1, component_names[1]: x2, component_names[2]: x3}
                                )

                                grid_points.append((x1, x2, x3))
                                grid_responses.append(pred['predicted_value'])

                    # Create DataFrame
                    pareto_df = pd.DataFrame({
                        component_names[0]: [p[0] for p in grid_points],
                        component_names[1]: [p[1] for p in grid_points],
                        component_names[2]: [p[2] for p in grid_points],
                        'Y_predicted': grid_responses
                    })

                    # Build objectives dictionary
                    objectives_dict = {}

                    # Primary objective (Y)
                    if objective_y == "Maximize Y":
                        objectives_dict['Y_predicted'] = 'maximize'
                    else:
                        objectives_dict['Y_predicted'] = 'minimize'

                    # Secondary objectives (components)
                    objectives_dict.update(secondary_objectives)

                    # Calculate Pareto front
                    if pareto_available:
                        pareto_ranked = calculate_pareto_front(pareto_df, objectives_dict, n_fronts=3)
                    else:
                        # Fallback: simple optimization without Pareto
                        st.warning("Pareto optimization module not available. Using simple optimization.")
                        if objective_y == "Maximize Y":
                            best_idx = pareto_df['Y_predicted'].idxmax()
                        else:
                            best_idx = pareto_df['Y_predicted'].idxmin()

                        pareto_ranked = pareto_df.copy()
                        pareto_ranked['pareto_rank'] = 999
                        pareto_ranked.loc[best_idx, 'pareto_rank'] = 1
                        pareto_ranked['crowding_distance'] = 0.0

                    # Store results
                    st.session_state.pareto_results = pareto_ranked
                    st.session_state.pareto_objectives = objectives_dict

                    st.success("✅ Pareto front calculated successfully!")
                    st.rerun()

                except Exception as e:
                    st.error(f"❌ Pareto calculation failed: {str(e)}")
                    import traceback
                    with st.expander("🐛 Error details"):
                        st.code(traceback.format_exc())

        st.markdown("---")

        # ═══════════════════════════════════════════════════════════════════════════
        # DISPLAY PARETO RESULTS
        # ═══════════════════════════════════════════════════════════════════════════

        if st.session_state.get('pareto_results') is not None:
            pareto_ranked = st.session_state.pareto_results

            st.markdown("## ✅ Pareto Front Results")

            # Summary statistics
            col_stat1, col_stat2, col_stat3 = st.columns(3)

            with col_stat1:
                n_front_1 = (pareto_ranked['pareto_rank'] == 1).sum()
                st.metric("Front 1 Solutions", n_front_1)

            with col_stat2:
                n_front_2 = (pareto_ranked['pareto_rank'] == 2).sum()
                st.metric("Front 2 Solutions", n_front_2)

            with col_stat3:
                n_total = len(pareto_ranked)
                st.metric("Total Candidates", n_total)

            st.markdown("---")

            # Pareto Front 2D Visualization
            if len(pareto_ranked) > 0:
                st.markdown("### 📉 Pareto Front Visualization")

                try:
                    from mlr_utils.pareto_optimization import plot_pareto_2d

                    # Plot Y vs each component
                    fig_pareto = plot_pareto_2d(
                        pareto_ranked,
                        'Y_predicted',
                        component_names[0],
                        st.session_state.pareto_objectives
                    )

                    st.plotly_chart(fig_pareto, use_container_width=True)

                except Exception as e:
                    st.warning(f"Could not create 2D visualization: {str(e)}")

            st.markdown("---")

            # Front 1 Table
            st.markdown("### 📊 Pareto Front 1 (Best Compromises)")

            front_1_df = pareto_ranked[pareto_ranked['pareto_rank'] == 1].sort_values(
                'crowding_distance', ascending=False
            )

            if len(front_1_df) > 0:
                # Get actual pseudo-component column names from the dataframe
                pseudo_comp_cols = [col for col in front_1_df.columns if col.startswith('PseudoComp')]
                pseudo_comp_cols = sorted(pseudo_comp_cols, key=lambda x: int(x.replace('PseudoComp', '')))

                # Build display columns list using actual column names
                display_cols = pseudo_comp_cols + ['Y_predicted', 'crowding_distance', 'pareto_rank']

                # Format dict with actual column names
                format_dict = {col: '{:.4f}' for col in pseudo_comp_cols}
                format_dict['Y_predicted'] = '{:.4f}'
                format_dict['crowding_distance'] = '{:.4f}'

                st.dataframe(
                    front_1_df[display_cols].style.format(format_dict),
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info("No solutions in Front 1")

            st.markdown("---")

            # Export Pareto Results
            st.markdown("### 💾 Export Pareto Results")

            if st.button("📥 Download Pareto Analysis Excel", key="export_pareto_btn"):
                try:
                    from mlr_utils.pareto_optimization import export_pareto_results

                    excel_buffer = export_pareto_results(
                        pareto_ranked,
                        st.session_state.pareto_objectives,
                        f"Pareto_Analysis.xlsx"
                    )

                    st.download_button(
                        "📥 Download Excel File",
                        excel_buffer,
                        "Pareto_Analysis.xlsx",
                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key="download_pareto_excel"
                    )

                    st.success("✅ Excel report ready for download!")

                except Exception as e:
                    st.error(f"Export failed: {str(e)}")

        else:
            st.info("👈 Configure objectives and click 'Calculate Pareto Front' to begin")

    # ========================================================================
    # TAB 6: EXTRACT & EXPORT
    # ========================================================================
    with tab6:
        if st.session_state.mixture_model_results is None:
            st.warning("⚠️ **No mixture model fitted**")
            st.info("💡 Go to **Model Computation** tab to fit a model first")
        else:
            show_mixture_export_ui(
                st.session_state.mixture_model_results,
                st.session_state.mixture_design_matrix,
                data,
                st.session_state.mixture_y_var
            )


if __name__ == "__main__":
    # For testing
    st.set_page_config(page_title="Mixture Design", layout="wide")
    show()
