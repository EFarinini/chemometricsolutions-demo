"""
MLR/DoE Analysis Page - Refactored Clean Version
Complete Design of Experiments and Multiple Linear Regression suite
Equivalent to CAT DOE_* R scripts

This module imports core functions from mlr_utils submodules.
Only page UI logic and TAB1 workflow remain in this file.
"""

import streamlit as st
import numpy as np
import pandas as pd
from scipy import stats

# Import core MLR/DoE functions from mlr_utils
from mlr_utils.model_computation import (
    fit_mlr_model,
    statistical_summary,
    create_model_matrix
)
from mlr_utils.model_diagnostics import (
    calculate_vif,
    check_model_saturated,
    show_model_diagnostics_ui
)
from mlr_utils.surface_analysis import show_surface_analysis_ui
# Backward compatibility (deprecated - now unified in surface_analysis):
# from mlr_utils.response_surface import show_response_surface_ui
# from mlr_utils.confidence_intervals import show_confidence_intervals_ui
from mlr_utils.predictions import show_predictions_ui
from mlr_utils.candidate_points import show_candidate_points_ui
from mlr_utils.export import show_export_ui
from mlr_utils.pareto_ui import show_pareto_ui


# ============================================================================
# HELPER FUNCTIONS (used in TAB1 only)
# ============================================================================

def detect_replicates(X_data, y_data, tolerance=1e-10):
    """
    Detect experimental replicates in the design matrix
    Returns replicate statistics or None if no replicates found
    """
    n_samples = len(X_data)
    X_values = X_data.values
    y_values = y_data.values

    replicate_groups = []
    used_indices = set()

    for i in range(n_samples):
        if i in used_indices:
            continue
        group = [i]
        used_indices.add(i)
        for j in range(i + 1, n_samples):
            if j in used_indices:
                continue
            if np.allclose(X_values[i], X_values[j], atol=tolerance):
                group.append(j)
                used_indices.add(j)
        if len(group) > 1:
            replicate_groups.append(group)

    if not replicate_groups:
        return None

    # Calculate pooled standard deviation
    variance_sum = 0
    dof_sum = 0
    group_stats = []

    for group in replicate_groups:
        y_group = y_values[group]
        n_reps = len(group)
        mean_y = np.mean(y_group)
        if n_reps > 1:
            var_y = np.var(y_group, ddof=1)
            std_y = np.sqrt(var_y)
            dof = n_reps - 1
            variance_sum += var_y * dof
            dof_sum += dof
            group_stats.append({
                'indices': group,
                'n_replicates': n_reps,
                'mean': mean_y,
                'std': std_y,
                'variance': var_y,
                'dof': dof
            })

    if dof_sum > 0:
        pooled_variance = variance_sum / dof_sum
        pooled_std = np.sqrt(pooled_variance)
    else:
        return None

    return {
        'n_replicate_groups': len(replicate_groups),
        'total_replicates': sum(len(g) for g in replicate_groups),
        'group_stats': group_stats,
        'pooled_std': pooled_std,
        'pooled_variance': pooled_variance,
        'pooled_dof': dof_sum
    }


def detect_central_points(X_data, tolerance=1e-6):
    """
    Detect central points in the design matrix
    A central point has ALL variables at their central value (coded: 0, natural: midpoint)
    Returns list of central point indices
    """
    central_indices = []
    X_values = X_data.values

    for i in range(len(X_data)):
        # Method 1: Check for ALL zeros (coded variables)
        if np.allclose(X_values[i], 0, atol=tolerance):
            central_indices.append(i)
            continue

        # Method 2: Check if ALL values are at midpoint of their ranges
        is_central = True
        for j in range(X_data.shape[1]):
            col_values = X_values[:, j]
            unique_vals = np.unique(col_values)
            if len(unique_vals) <= 1:
                continue
            if set(unique_vals).issubset({-1, 0, 1}):
                if not np.isclose(X_values[i, j], 0, atol=tolerance):
                    is_central = False
                    break
            else:
                min_val = col_values.min()
                max_val = col_values.max()
                mid_val = (min_val + max_val) / 2
                if not np.isclose(X_values[i, j], mid_val, atol=tolerance):
                    is_central = False
                    break

        if is_central:
            central_indices.append(i)

    return central_indices


# ============================================================================
# MAIN PAGE FUNCTION
# ============================================================================

def show():
    """Display the MLR/DOE Analysis page"""

    st.markdown("# MLR & Design of Experiments")
    st.markdown("*Complete MLR/DoE analysis suite equivalent to CAT DOE_* R scripts*")

    # Check if data is loaded (but don't block the entire page)
    data_loaded = 'current_data' in st.session_state and st.session_state.current_data is not None
    data = st.session_state.current_data if data_loaded else None

    # Create 7 tabs (merged Response Surface + Confidence Intervals into Surface Analysis)
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "Model Computation",
        "Model Diagnostics",
        "Surface Analysis",          # Merged tab (was Response Surface + Confidence Intervals)
        "Predictions",               # Tab 4
        "Multi-Criteria Decision",   # NEW TAB 5 - Pareto Optimization
        "Generate Matrix",           # Shifted from tab5 to tab6
        "Extract & Export"           # Shifted from tab6 to tab7
    ])

    # ========================================================================
    # TAB 1: MODEL COMPUTATION - Complete workflow with all statistical tests
    # ========================================================================
    with tab1:


        if not data_loaded:
            st.warning("⚠️ **No data loaded**")
            st.info("💡 **Two options:**\n"
                   "1. Go to **Data Handling** to load your dataset\n"
                   "2. Use **Generate Matrix** tab to create a DoE design")
        else:
            # ════════════════════════════════════════════════════════════════
            # DATA PRE-PROCESSING SECTION
            # ════════════════════════════════════════════════════════════════
            st.markdown("## 🔄 Data Pre-Processing")

            # Create two columns: Info + Auto-Coding Toggle
            col_preproc1, col_preproc2 = st.columns([2, 1])

            with col_preproc1:
                st.info("""
**Check if data needs automatic DoE coding:**
- Raw numeric data (natural units) → coded to [-1, +1]
- Categorical data → coded appropriately
- Already coded data → skip this step
                """)

            with col_preproc2:
                needs_coding = st.checkbox("🔧 Apply automatic coding", value=False, key="auto_code_toggle")

            if needs_coding:
                st.markdown("### Automatic DoE Coding")

                # Show current data statistics
                col_stat1, col_stat2, col_stat3 = st.columns(3)
                with col_stat1:
                    st.metric("Rows", data.shape[0])
                with col_stat2:
                    st.metric("Columns", data.shape[1])
                with col_stat3:
                    st.metric("Missing Values", data.isna().sum().sum())

                # Display column analysis
                st.markdown("#### Column-by-Column Analysis")
                col_analysis_container = st.container()

                with col_analysis_container:
                    # Import detect_column_type from column_transforms
                    from transforms.column_transforms import detect_column_type

                    analysis_results = {}
                    for col_name in data.columns:
                        col_info = detect_column_type(data[col_name])
                        analysis_results[col_name] = col_info

                    # Create analysis dataframe for display
                    analysis_display = []
                    for col_name, info in analysis_results.items():
                        # Limit unique values display to 50 chars
                        unique_str = str(info['unique_values'])[:50]
                        if len(str(info['unique_values'])) > 50:
                            unique_str += "..."

                        analysis_display.append({
                            'Variable': col_name,
                            'Type': info['dtype_detected'],
                            'Levels': info['n_levels'],
                            'Unique Values': unique_str
                        })

                    analysis_df = pd.DataFrame(analysis_display)
                    st.dataframe(analysis_df, use_container_width=True, hide_index=True)

                # ════════════════════════════════════════════════════════════════
                # Column Selection for Auto-Coding
                # ════════════════════════════════════════════════════════════════
                st.markdown("### Select Columns to Code")
                st.info("Select which columns to apply DoE coding to")

                # Multiselect for columns (default: all except last)
                cols_to_code = st.multiselect(
                    "Columns to code:",
                    options=data.columns.tolist(),
                    default=data.columns[:-1].tolist(),
                    key="auto_code_columns"
                )

                # Calculate excluded columns
                cols_excluded = [col for col in data.columns if col not in cols_to_code]

                # Display 2-column summary
                col_select1, col_select2 = st.columns(2)

                with col_select1:
                    st.markdown(f"**To Code ({len(cols_to_code)}):**")
                    if cols_to_code:
                        for col in cols_to_code:
                            st.write(f"- {col}")
                    else:
                        st.write("*(None selected)*")

                with col_select2:
                    st.markdown(f"**Excluded ({len(cols_excluded)}):**")
                    if cols_excluded:
                        for col in cols_excluded:
                            st.write(f"- {col}")
                    else:
                        st.write("*(None)*")

                st.markdown("---")

                # Apply coding button
                if st.button("🚀 Apply Automatic Coding", type="primary", key="apply_auto_coding"):
                    try:
                        with st.spinner("Encoding data..."):
                            # Import column_doe_coding from column_transforms
                            from transforms.column_transforms import column_doe_coding

                            # Prepare data for encoding
                            if cols_to_code:
                                # Create subset with only columns to code
                                data_to_code = data[cols_to_code].copy()

                                # Apply automatic DoE encoding to selected columns
                                coded_data, encoding_metadata, multiclass_info = column_doe_coding(
                                    data_to_code,
                                    col_range=(0, len(cols_to_code)),
                                    reference_levels={}  # Use auto-suggested reference levels
                                )

                                # Merge coded data with excluded columns
                                if cols_excluded:
                                    excluded_data = data[cols_excluded].copy()
                                    # Combine: coded columns first, then excluded columns
                                    coded_data = pd.concat([coded_data, excluded_data], axis=1)
                            else:
                                st.error("❌ No columns selected for coding!")
                                return

                        st.success("✅ Data coded successfully!")

                        # Show encoding summary
                        st.markdown("#### Encoding Summary")

                        summary_rows = []
                        for col_name, meta in encoding_metadata.items():
                            summary_rows.append({
                                'Column': col_name,
                                'Type': meta['type'],
                                'N Levels': meta['n_levels'],
                                'Encoding': meta['encoding_rule']
                            })

                        summary_df = pd.DataFrame(summary_rows)
                        st.dataframe(summary_df, use_container_width=True, hide_index=True)

                        # Expandable detailed info
                        with st.expander("📋 Detailed Encoding Information"):
                            for col_name, meta in encoding_metadata.items():
                                st.markdown(f"**{col_name}** ({meta['type']})")

                                if 'encoding_map' in meta:
                                    st.write(f"**Mapping:** {meta['encoding_map']}")

                                if 'dummy_columns' in meta:
                                    st.write(f"**Dummy Columns Created:** {', '.join(meta['dummy_columns'])}")
                                    st.write(f"**Reference Level (implicit):** {meta['reference_level']}")

                                st.markdown("---")

                        # Store coded data in session state
                        st.session_state.current_data = coded_data
                        st.session_state.encoding_metadata = encoding_metadata
                        st.session_state.auto_coded = True

                        st.info("✨ Coded data is now active in session. Proceeding to model computation...")
                        st.rerun()

                    except Exception as e:
                        st.error(f"❌ Coding failed: {str(e)}")
                        import traceback
                        with st.expander("🐛 Error details"):
                            st.code(traceback.format_exc())

            st.markdown("---")

            # ════════════════════════════════════════════════════════════════
            # MODEL COMPUTATION UI
            # ════════════════════════════════════════════════════════════════
            # Use potentially coded data from session state
            current_working_data = st.session_state.current_data

            from mlr_utils.model_computation import show_model_computation_ui
            show_model_computation_ui(current_working_data, "current_data")

    # ========================================================================
    # TAB 2: MODEL DIAGNOSTICS
    # ========================================================================
    with tab2:
        if not data_loaded:
            st.warning("⚠️ **No data loaded**")
            st.info("💡 Load data in **Data Handling** or create a design in **Generate Matrix**")
        elif 'mlr_model' not in st.session_state:
            st.warning("⚠️ **No MLR model fitted**")
            st.info("💡 Go to **Model Computation** tab to fit a model first")
        else:
            show_model_diagnostics_ui(
                model_results=st.session_state.mlr_model,
                X=st.session_state.mlr_model['X'],
                y=st.session_state.mlr_model['y']
            )

    # ========================================================================
    # TAB 3: SURFACE ANALYSIS (Merged Response Surface + Confidence Intervals)
    # ========================================================================
    with tab3:
        if not data_loaded:
            st.warning("⚠️ **No data loaded**")
            st.info("💡 Load data in **Data Handling** or create a design in **Generate Matrix**")
        elif st.session_state.get('mlr_model'):
            show_surface_analysis_ui(
                st.session_state.mlr_model,
                st.session_state.mlr_x_vars,
                st.session_state.mlr_y_var
            )
        else:
            st.warning("⚠️ **No MLR model fitted**")
            st.info("💡 Go to **Model Computation** tab to fit a model first")

    # ========================================================================
    # TAB 4: PREDICTIONS (shifted from tab5)
    # ========================================================================
    with tab4:
        if not data_loaded:
            st.warning("⚠️ **No data loaded**")
            st.info("💡 Load data in **Data Handling** or create a design in **Generate Matrix**")
        elif st.session_state.get('mlr_model'):
            show_predictions_ui(
                st.session_state.mlr_model,
                st.session_state.mlr_x_vars,
                st.session_state.mlr_y_var,
                data
            )
        else:
            st.warning("⚠️ **No MLR model fitted**")
            st.info("💡 Go to **Model Computation** tab to fit a model first")

    # ========================================================================
    # TAB 5: MULTI-CRITERIA DECISION MAKING (Pareto Optimization) - NEW
    # ========================================================================
    with tab5:
        if not data_loaded:
            st.warning("⚠️ **No data loaded**")
            st.info("💡 Load data in **Data Handling** or create a design in **Generate Matrix**")
        elif st.session_state.get('mlr_model'):
            show_pareto_ui(
                st.session_state.mlr_model,
                st.session_state.mlr_x_vars,
                st.session_state.mlr_y_var
            )
        else:
            st.warning("⚠️ **No MLR model fitted**")
            st.info("💡 Go to **Model Computation** tab to fit a model first")

    # ========================================================================
    # TAB 6: GENERATE MATRIX - STANDALONE (shifted from tab5)
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
    # TAB 7: EXPORT (shifted from tab6)
    # ========================================================================
    with tab7:
        if not data_loaded:
            st.warning("⚠️ **No data loaded**")
            st.info("💡 Load data in **Data Handling** or create a design in **Generate Matrix**")
        elif 'mlr_model' not in st.session_state:
            st.warning("⚠️ **No MLR model fitted**")
            st.info("💡 Go to **Model Computation** tab to fit a model first")
        else:
            try:
                show_export_ui(
                    st.session_state.mlr_model,
                    st.session_state.mlr_y_var
                )
            except Exception as e:
                st.error(f"❌ Error in Export tab: {str(e)}")
                import traceback
                with st.expander("🐛 Error Details"):
                    st.code(traceback.format_exc())
