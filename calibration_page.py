"""
PLS-1 Calibration Module - Main Streamlit Page

This module provides a complete interface for building and validating PLS-1 regression models
using the NIPALS algorithm. It includes:
- Model calibration with cross-validation
- Optimal latent variable selection
- Test set validation
- Comprehensive visualizations and diagnostics

Author: ChemoMetric Solutions
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any

# Import utilities
from calibration_utils.pls_calculations import (
    pls_nipals,
    pls_predict,
    calculate_metrics,
    calculate_residuals
)
from calibration_utils.pls_cv import (
    repeated_kfold_cv,
    select_optimal_lv,
    full_cv_summary
)
from calibration_utils.pls_preprocessing import (
    prepare_calibration_data,
    apply_scaler,
    validate_pls_input,
    split_xy_by_column_name
)
from calibration_utils.pls_plots import (
    plot_rmsecv_vs_lv,
    plot_predictions_vs_observed,
    plot_residuals,
    plot_loadings,
    plot_regression_coefficients
)
# Diagnostic plots
from calibration_utils.pls_diagnostics import (
    plot_loading_plot,
    plot_vip,
    plot_score_plot_with_ellipse,
    plot_residuals_histogram,
    plot_qq_plot,
    create_diagnostic_info_dict
)
# Smart coefficient plots
from calibration_utils.pls_coefficients_smart import (
    plot_regression_coefficients_smart,
    analyze_overfitting,
    plot_coefficient_comparison
)
from calibration_utils.pls_validation import (
    load_test_set_from_workspace,
    validate_on_test,
    generate_validation_report
)

# Import shared utilities
from workspace_utils import get_workspace_datasets
import color_utils

# Import TAB 3 Test Set Validation
from calibration_utils.tab3_test_set_validation import show_tab3_test_set_validation


# Note: Page configuration is handled by homepage.py/streamlit_app.py
# Do NOT call st.set_page_config() here as it can only be called once per app


def render_header():
    """Render the page header with title and description."""
    st.title("📊 PLS-1 Calibration Module")
    st.markdown("""
    Build and validate Partial Least Squares regression models using the NIPALS algorithm.
    This module supports cross-validation, optimal component selection, and test set validation.
    """)


def render_sidebar():
    """Render the sidebar with data selection and model parameters."""
    st.sidebar.header("Model Configuration")

    # Data selection
    st.sidebar.subheader("1. Data Selection")

    # Placeholder for workspace data loading
    # (datasets loaded in main tab)

    # Model parameters
    st.sidebar.subheader("2. Model Parameters")

    # Placeholder for parameter inputs

    return {}


def render_calibration_tab():
    """Render the Calibration & Cross-Validation tab."""
    st.header("PLS Calibration Model Computation")

    ### STEP 1: Load Calibration Data
    st.markdown("### 📊 Step 1: Load Calibration Dataset")

    st.info("""
    **Load your dataset and define X (predictors) and Y (response)**
    - X = Numeric matrix (features/variables)
    - Y = Single numeric response column (continuous target)
    """)

    # Get available datasets from workspace
    datasets = get_workspace_datasets()
    available_datasets = list(datasets.keys())

    if not available_datasets:
        st.warning("⚠️ No datasets found in workspace. Please upload data first in Data Handling module.")
        return

    dataset_name = st.selectbox(
        "📂 Select dataset:",
        options=available_datasets,
        key="dataset_select_pls",
        help="Choose which dataset contains both X and Y"
    )

    full_dataset = datasets[dataset_name]
    n_samples_total = full_dataset.shape[0]

    st.write(f"**Dataset info:** {n_samples_total} rows × {full_dataset.shape[1]} columns")

    # Show data preview
    with st.expander("📊 Data Preview"):
        st.dataframe(full_dataset.head(10), use_container_width=True)

    # Sample selection
    st.markdown("#### 🔢 Select Sample Range (rows)")

    col_s1, col_s2 = st.columns(2)

    with col_s1:
        first_sample = st.number_input(
            "First sample (1-based):",
            min_value=1,
            max_value=n_samples_total,
            value=1,
            key="first_sample_pls"
        )

    with col_s2:
        last_sample = st.number_input(
            "Last sample (1-based):",
            min_value=first_sample,
            max_value=n_samples_total,
            value=n_samples_total,
            key="last_sample_pls"
        )

    # Filter samples
    sample_range = list(range(first_sample - 1, last_sample))
    full_dataset = full_dataset.iloc[sample_range].copy()

    st.write(f"**Selected samples:** {len(full_dataset)} rows (from {first_sample} to {last_sample})")

    st.divider()

    ### STEP 2: Select X Matrix (Predictor Range)
    st.markdown("### 📋 Step 2: Select X Matrix (Predictor Columns)")

    all_columns = full_dataset.columns.tolist()
    numeric_cols = full_dataset.select_dtypes(include=[np.number]).columns.tolist()
    non_numeric_cols = [c for c in all_columns if c not in numeric_cols]

    col_info1, col_info2, col_info3 = st.columns(3)
    with col_info1:
        st.metric("Total columns", len(all_columns))
    with col_info2:
        st.metric("Numeric columns", len(numeric_cols))
    with col_info3:
        st.metric("Non-numeric columns", len(non_numeric_cols))

    if len(non_numeric_cols) > 0:
        st.write(f"**Non-numeric columns (potential categorical Y):** {non_numeric_cols}")

    # Column range selection for X
    st.markdown("#### 🎯 Define X column range:")
    col_range1, col_range2 = st.columns(2)

    # Determine default range for X (all numeric columns)
    if len(numeric_cols) > 0:
        first_numeric_idx = all_columns.index(numeric_cols[0]) + 1
        last_numeric_idx = all_columns.index(numeric_cols[-1]) + 1
    else:
        first_numeric_idx = 1
        last_numeric_idx = len(all_columns)

    with col_range1:
        first_col = st.number_input(
            "First column (1-based):",
            min_value=1,
            max_value=len(all_columns),
            value=first_numeric_idx,
            key="first_col_pls",
            help="Start of X predictor columns"
        )

    with col_range2:
        last_col = st.number_input(
            "Last column (1-based):",
            min_value=first_col,
            max_value=len(all_columns),
            value=last_numeric_idx,
            key="last_col_pls",
            help="End of X predictor columns (inclusive)"
        )

    # Extract X columns
    x_cols = all_columns[first_col - 1:last_col]
    st.write(f"**Selected X columns:** {first_col} to {last_col} = `{x_cols[0]}` ... `{x_cols[-1]}`")

    # Display X matrix preview
    with st.expander("🔍 X Matrix Preview"):
        X_df_preview = full_dataset[x_cols]
        st.dataframe(X_df_preview.head(), use_container_width=True)
        st.caption(f"X shape: {X_df_preview.shape[0]} samples × {X_df_preview.shape[1]} features")

    ### STEP 3: Select Y (Response Variable)
    st.markdown("### 🎯 Step 3: Select Y (Response Variable)")

    # Only allow numeric columns for Y (regression target)
    y_candidates = [col for col in all_columns if col not in x_cols and col in numeric_cols]

    if len(y_candidates) == 0:
        st.error("❌ No numeric columns available for Y (response). All numeric columns are in X range!")
        return

    y_col = st.selectbox(
        "Select Y (response column):",
        options=y_candidates,
        key="y_select_pls",
        help="Must be a numeric column (continuous response for regression)"
    )

    # Display Y statistics
    st.markdown("#### 📊 Y Variable Statistics:")
    y_series = full_dataset[y_col]

    col_y1, col_y2, col_y3, col_y4 = st.columns(4)
    with col_y1:
        st.metric("Min", f"{y_series.min():.3f}")
    with col_y2:
        st.metric("Max", f"{y_series.max():.3f}")
    with col_y3:
        st.metric("Mean", f"{y_series.mean():.3f}")
    with col_y4:
        st.metric("Std", f"{y_series.std():.3f}")

    ### STEP 4: Validation Checks
    st.markdown("### ✅ Step 4: Data Validation")

    # Perform validation checks
    X_df = full_dataset[x_cols]
    n_samples = len(full_dataset)
    n_features = len(x_cols)

    # Validation checks (only critical ones that block execution)
    check1 = y_col in numeric_cols  # Y is numeric
    check2 = all(col in numeric_cols for col in x_cols)  # X columns are all numeric
    check3 = y_col not in x_cols  # Y not in X
    check4 = n_features >= 3  # At least 3 features

    col_v1, col_v2, col_v3, col_v4 = st.columns(4)

    with col_v1:
        st.metric("✅ Y numeric" if check1 else "❌ Y numeric", "PASS" if check1 else "FAIL")
    with col_v2:
        st.metric("✅ X numeric" if check2 else "❌ X numeric", "PASS" if check2 else "FAIL")
    with col_v3:
        st.metric("✅ Y ∉ X" if check3 else "❌ Y ∉ X", "PASS" if check3 else "FAIL")
    with col_v4:
        st.metric("✅ p ≥ 3" if check4 else "❌ p ≥ 3", f"{n_features}" if check4 else f"{n_features}")

    all_checks_pass = check1 and check2 and check3 and check4

    if not all_checks_pass:
        st.error("❌ Data validation failed. Please adjust your selections.")
        if not check2:
            non_numeric_x = [col for col in x_cols if col not in numeric_cols]
            st.error(f"Non-numeric X columns: {non_numeric_x}")
        return

    # Show n vs p info (not a blocker - PLS handles n < p!)
    st.info(f"📊 Dataset: **{n_samples} samples** × **{n_features} features** (n/p ratio: {n_samples/n_features:.2f})")

    # Warning only if ratio is very low
    if n_samples < n_features:
        st.warning(
            f"ℹ️ More features ({n_features}) than samples ({n_samples}). "
            f"**PLS is designed for this!** Regularization handles it, but ensure data quality."
        )

    st.success(f"✅ All validation checks passed! Ready to proceed with PLS calibration.")
    st.info(f"📊 Final matrix: {n_samples} samples × {n_features} features → {y_col}")

    ### STEP 5: Preprocessing Options
    st.markdown("### 🔧 Step 5: Data Preprocessing")

    col3, col4 = st.columns(2)

    with col3:
        scaling = st.radio(
            "Scaling Method",
            options=['mean_center', 'autoscale', 'none'],
            index=0,
            help="mean_center: center only (recommended for PLS)\nautoscale: center + scale to unit variance"
        )

    with col4:
        st.markdown("**Note:**")
        st.markdown("- **mean_center**: Recommended for PLS")
        st.markdown("- **autoscale**: For different scales")
        st.markdown("- **none**: No preprocessing")

    max_components = min(n_features, n_samples - 1, 20)

    ### STEP 6: Cross-Validation Setup
    st.markdown("### 🔄 Step 6: Cross-Validation Strategy")

    col5, col6, col7 = st.columns(3)

    with col5:
        n_folds = st.slider(
            "K-fold Splits",
            min_value=3,
            max_value=min(20, n_samples // 2),
            value=10,
            help="Number of CV folds"
        )

    with col6:
        n_repeats = st.slider(
            "Number of Repeats",
            min_value=1,
            max_value=100,
            value=10,
            step=5,
            help="Repeat CV for robust estimates"
        )

    with col7:
        n_components_max = st.slider(
            "Max Components to Test",
            min_value=2,
            max_value=max_components,
            value=min(10, max_components),
            help="Test up to this many LVs"
        )

    ### STEP 7: Run CV Analysis
    if st.button("▶️ Start Repeated CV Analysis", key="start_cv", type="primary"):
        with st.spinner("Running cross-validation... This may take a moment."):
            try:
                # Prepare data with selected X columns
                # NOTE: prepare_calibration_data() now returns RAW data ONLY
                X_df = full_dataset[x_cols]
                prep_result = prepare_calibration_data(
                    data=pd.concat([X_df, full_dataset[[y_col]]], axis=1),
                    response_column=y_col,
                    scaling=scaling  # DEPRECATED - kept for compatibility
                )
                X_raw = prep_result['X']  # RAW data (no preprocessing)
                Y_raw = prep_result['y']  # RAW data (no preprocessing)
                feature_names = prep_result['feature_names']

                # Run CV with RAW data
                # Preprocessing is applied INSIDE pls_nipals() on each fold
                st.write(f"🔴 DEBUG: scaling variable received = '{scaling}'")
                st.write(f"🔍 DEBUG: CV starting with preprocessing='{scaling}'")

                cv_results = repeated_kfold_cv(
                    X=X_raw,
                    y=Y_raw,
                    max_components=n_components_max,
                    n_folds=n_folds,
                    n_repeats=n_repeats,
                    randomize=True,
                    random_state=42,
                    preprocessing=scaling  # Pass preprocessing to CV
                )

                # Select optimal LV
                optimal_result = select_optimal_lv(cv_results, method='min')
                optimal_lv = optimal_result['optimal_lv']

                # Save to session state
                # NOTE: Saving RAW data (not preprocessed!)
                st.session_state['X_raw'] = X_raw
                st.session_state['Y_raw'] = Y_raw
                st.session_state['preprocessing'] = scaling  # Save preprocessing method
                st.session_state['feature_names'] = feature_names
                st.session_state['cv_results'] = cv_results
                st.session_state['optimal_lv'] = optimal_lv
                st.session_state['x_columns'] = x_cols
                st.session_state['y_column'] = y_col
                st.session_state['dataframe'] = full_dataset
                st.session_state['n_components_max'] = n_components_max

                st.success("✅ CV completed successfully!")
                st.rerun()

            except Exception as e:
                st.error(f"❌ Error during CV: {str(e)}")
                import traceback
                st.code(traceback.format_exc())

    # Display results if available
    if 'cv_results' in st.session_state:
        cv_results = st.session_state['cv_results']
        optimal_lv = st.session_state.get('optimal_lv', None)

        # Validate CV results structure
        if not isinstance(cv_results, dict) or 'RMSECV' not in cv_results:
            st.error("❌ Cross-validation results are corrupted. Please re-run the CV analysis.")
            # Clear corrupted results
            if 'cv_results' in st.session_state:
                del st.session_state['cv_results']
            if 'optimal_lv' in st.session_state:
                del st.session_state['optimal_lv']
            return

        if optimal_lv is None:
            st.error("❌ Optimal LV not found. Please re-run the CV analysis.")
            return

        st.markdown("### 📊 Cross-Validation Results")

        # RMSECV plot
        try:
            fig_rmsecv = plot_rmsecv_vs_lv(
                cv_results,
                optimal_lv,
                theme='light'
            )
            st.plotly_chart(fig_rmsecv, use_container_width=True)
        except Exception as e:
            st.error(f"Error plotting RMSECV: {str(e)}")
            import traceback
            st.code(traceback.format_exc())

        # Results table
        try:
            cv_summary_df = full_cv_summary(cv_results, optimal_lv)
            st.dataframe(cv_summary_df, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating CV summary: {str(e)}")
            import traceback
            st.code(traceback.format_exc())

        st.info(f"🎯 Minimum RMSECV at LV = {optimal_lv}")

        ### STEP 8: Select Model Complexity
        st.markdown("### 🎛️ Step 8: Select Number of Components")

        selected_lv = st.slider(
            "Number of Latent Variables",
            min_value=1,
            max_value=st.session_state['n_components_max'],
            value=optimal_lv,
            help="Minimum RMSECV suggested, but you can adjust for parsimony"
        )

        if st.button("✅ Confirm Model and Continue", key="confirm_model", type="primary"):
            try:
                # Final calibration with selected LV
                # NOTE: Using RAW data with preprocessing applied INSIDE pls_nipals()
                X_raw = st.session_state['X_raw']
                Y_raw = st.session_state['Y_raw']
                preprocessing = st.session_state['preprocessing']

                final_model = pls_nipals(X_raw, Y_raw, n_components=selected_lv,
                                       preprocessing=preprocessing)

                # Add feature names for diagnostics
                final_model['feature_names'] = st.session_state['x_columns']
                final_model['y_true'] = Y_raw.flatten()

                # Save to session
                st.session_state['final_model'] = final_model
                st.session_state['selected_lv'] = selected_lv
                # Also save unscaled Y for score plot coloring
                st.session_state['pls_Y'] = st.session_state['dataframe'][st.session_state['y_column']].values
                st.session_state['pls_X_raw'] = X_raw
                st.session_state['pls_Y_raw'] = Y_raw

                st.success(f"✅ Model with {selected_lv} LV confirmed! Go to TAB 2 for validation.")

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")


def render_validation_tab():
    """Render the Test Set Validation tab - TAB 2: Results & Graphics."""

    # Check if model is available
    if 'final_model' not in st.session_state:
        st.info("👉 Complete calibration in TAB 1 first to view results")
        return

    # Get model and data from session state
    model = st.session_state['final_model']
    selected_lv = st.session_state['selected_lv']
    cv_results = st.session_state['cv_results']

    # Handle both old and new session state structures
    # Old: 'scaler_info', 'X_scaled', 'Y_scaled'
    # New: 'feature_names', 'X_raw', 'Y_raw', 'preprocessing'

    # Try new structure first, fall back to old
    if 'feature_names' in st.session_state:
        feature_names = st.session_state['feature_names']
    elif 'scaler_info' in st.session_state and 'feature_names' in st.session_state['scaler_info']:
        feature_names = st.session_state['scaler_info']['feature_names']
    elif 'x_columns' in st.session_state:
        feature_names = st.session_state['x_columns']
    else:
        feature_names = [f"Var_{i+1}" for i in range(model.get('n_features', 10))]

    # Get raw data (or fallback to scaled data for old sessions)
    if 'X_raw' in st.session_state:
        X_data = st.session_state['X_raw']
        Y_data = st.session_state['Y_raw']
    elif 'X_scaled' in st.session_state:
        # Old session - use scaled data
        X_data = st.session_state['X_scaled']
        Y_data = st.session_state['Y_scaled']
        st.warning("⚠️ Using old session data. Re-run calibration in TAB 1 for best results.")
    else:
        st.error("❌ Session data not found. Please re-run calibration in TAB 1.")
        return

    st.header("PLS Model Results & Diagnostics")

    ### Model Summary Info Box
    col1, col2, col3, col4 = st.columns(4)

    # Get CV predictions for selected LV (use selected_lv, not optimal_lv)
    # Note: selected_lv may differ from optimal_lv if user adjusted the slider
    optimal_lv = st.session_state['optimal_lv']

    # Check if cv_predictions is 2D or 1D
    if cv_results['cv_predictions'].ndim == 2:
        cv_predictions = cv_results['cv_predictions'][:, selected_lv - 1]
    else:
        # If already 1D, it's already for the selected LV
        cv_predictions = cv_results['cv_predictions']

    cv_true = cv_results['cv_true']

    # Calculate CV metrics
    cv_metrics = calculate_metrics(cv_true, cv_predictions)

    with col1:
        st.metric("Latent Variables", selected_lv)

    with col2:
        st.metric("RMSECV", f"{cv_results['RMSECV'][optimal_lv-1]:.4f}")

    with col3:
        r2_cv = cv_results['R2CV'][optimal_lv-1] * 100
        st.metric("R² CV (%)", f"{r2_cv:.2f}")

    with col4:
        st.metric("Samples", X_data.shape[0])

    ### Tabbed Results Display
    results_tab1, results_tab2, results_tab3, results_tab4, results_tab5 = st.tabs([
        "📈 VIP",
        "📊 Predictions",
        "📉 Residuals",
        "📋 Coefficients",
        "🔄 Loadings"
    ])

    with results_tab1:
        st.subheader("Variable Importance in Projection (VIP)")

        # VIP slider
        col1, col2 = st.columns(2)
        with col1:
            n_show = st.slider("Show top N variables:", 5, min(50, model.get('n_features', 20)), 15, key="vip_n")
        with col2:
            threshold = st.slider("VIP Threshold:", 0.5, 2.0, 1.0, 0.1, key="vip_thresh")

        try:
            fig_vip = plot_vip(model, threshold=threshold, n_features_show=n_show)
            st.plotly_chart(fig_vip, use_container_width=True)

            with st.expander("📖 What is VIP?"):
                st.markdown("""
                **Variable Importance in Projection (VIP)**
                - Ranks variables by importance to predictions
                - VIP > 1.0 (green): Important → KEEP
                - VIP < 0.5 (red): Not important → REMOVE
                - Use for feature selection
                """)
        except Exception as e:
            st.error(f"Error creating VIP plot: {str(e)}")

    with results_tab2:
        st.subheader("Predictions vs Observed (Cross-Validation)")

        # Plot predictions vs observed
        try:
            fig_pred = plot_predictions_vs_observed(
                y_true=cv_true,
                y_pred=cv_predictions,
                title="Cross-Validation: Predicted vs Observed",
                sample_names=[f"Sample_{i+1}" for i in range(len(cv_true))],
                theme='light'
            )
            st.plotly_chart(fig_pred, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating predictions plot: {str(e)}")

        # Show metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("R²", f"{cv_metrics['R2']:.4f}")
        with col2:
            st.metric("RMSE", f"{cv_metrics['RMSE']:.4f}")
        with col3:
            st.metric("Bias", f"{cv_metrics['Bias']:.4f}")

    with results_tab3:
        st.subheader("Residuals Analysis (Cross-Validation)")

        # Calculate residuals
        residuals = cv_true - cv_predictions

        # Plot residuals diagnostics
        try:
            fig_resid = plot_residuals(
                y_true=cv_true,
                y_pred=cv_predictions,
                sample_names=[f"Sample_{i+1}" for i in range(len(cv_true))],
                theme='light'
            )
            st.plotly_chart(fig_resid, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating residuals plot: {str(e)}")

        # Show residual statistics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Mean Residual", f"{np.mean(residuals):.4f}")
        with col2:
            st.metric("Std Residual", f"{np.std(residuals):.4f}")

    with results_tab4:
        st.subheader("Regression Coefficients")

        # Get coefficients from model
        B = model['B']
        # feature_names already loaded from session state at top of function

        # Plot regression coefficients using SMART auto-detection
        try:
            fig_coef = plot_regression_coefficients_smart(
                model=model,
                feature_names=feature_names
            )
            st.plotly_chart(fig_coef, use_container_width=True)

            # Show info about plot type
            n_features = len(B)
            if n_features > 100:
                st.info(f"📊 Auto-detected: **Spectral data** ({n_features} variables) → Line plot")
            else:
                st.info(f"📊 Auto-detected: **Tabular data** ({n_features} variables) → Bar chart")

        except Exception as e:
            st.error(f"Error creating coefficients plot: {str(e)}")

        # Coefficients table
        coef_df = pd.DataFrame({
            'Feature': feature_names,
            'Coefficient': B.flatten()
        }).sort_values('Coefficient', key=abs, ascending=False)

        st.dataframe(coef_df, use_container_width=True)

    with results_tab5:
        st.subheader("X-Loadings Biplot")

        n_comp = model.get('n_components', 1)

        # LV selection
        col1, col2 = st.columns(2)
        with col1:
            lv1 = st.slider("LV X-axis:", 1, n_comp, 1, key="loading_lv1")
        with col2:
            lv2 = st.slider("LV Y-axis:", 1, n_comp, min(2, n_comp), key="loading_lv2")

        try:
            fig_loading = plot_loading_plot(model, lv=lv1, lv_y=lv2, n_features_highlight=10)
            st.plotly_chart(fig_loading, use_container_width=True)

            with st.expander("📖 How to read Loading Plot?"):
                st.markdown("""
                **Loading Plot Explanation:**
                - Each point = one X variable
                - Distance from center = importance
                - Angle between variables = correlation
                - Unit circle = normalization reference
                - Use to understand LV meaning
                """)
        except Exception as e:
            st.error(f"Error creating loading plot: {str(e)}")

        # Create detailed table
        detail_df = pd.DataFrame({
            'Sample': [f"Sample_{i+1}" for i in range(len(cv_true))],
            'Observed': cv_true,
            'Predicted_CV': cv_predictions,
            'Residual': residuals,
            'Std_Residual': residuals / np.std(residuals) if np.std(residuals) > 0 else residuals,
            'Error_%': (residuals / cv_true * 100) if np.all(cv_true != 0) else residuals
        })

        st.dataframe(detail_df, use_container_width=True, height=400)

    ### Export Options
    st.divider()
    st.subheader("Export Options")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("💾 Save Model to Workspace", key="save_model"):
            try:
                from datetime import datetime
                import json

                # Prepare model data for export
                model_export = {
                    'n_components': selected_lv,
                    'feature_names': feature_names,
                    'response_name': st.session_state['y_column'],
                    'B': B.tolist(),
                    'X_mean': model['X_mean'].tolist(),
                    'y_mean': float(model['y_mean']),
                    'cv_metrics': {k: float(v) if isinstance(v, (np.floating, float)) else int(v)
                                  for k, v in cv_metrics.items()},
                    'timestamp': datetime.now().isoformat()
                }

                filename = f"pls_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

                # Save to workspace
                with open(f"workspace/{filename}", 'w') as f:
                    json.dump(model_export, f, indent=2)

                st.success(f"✅ Model saved: {filename}")
            except Exception as e:
                st.error(f"Error saving model: {str(e)}")

    with col2:
        # Prepare predictions CSV
        export_df = pd.DataFrame({
            'Sample': [f"Sample_{i+1}" for i in range(len(cv_true))],
            'Observed': cv_true,
            'Predicted_CV': cv_predictions,
            'Residuals': residuals
        })
        csv = export_df.to_csv(index=False)

        st.download_button(
            label="📥 Download Predictions CSV",
            data=csv,
            file_name="pls_cv_predictions.csv",
            mime="text/csv",
            key="download_predictions"
        )

    with col3:
        # Prepare coefficients CSV
        coef_csv = coef_df.to_csv(index=False)

        st.download_button(
            label="📥 Download Coefficients CSV",
            data=coef_csv,
            file_name="pls_coefficients.csv",
            mime="text/csv",
            key="download_coefficients"
        )

    # Add two more tabs for comprehensive diagnostics
    st.markdown("---")
    st.subheader("Additional Diagnostics")

    diag_tab1, diag_tab2, diag_tab3 = st.tabs([
        "🎯 Score Plot (Outliers)",
        "📉 Residuals Histogram",
        "📍 Q-Q Plot"
    ])

    with diag_tab1:
        st.subheader("Score Plot with Confidence Ellipse")
        n_comp = model.get('n_components', 1)

        col1, col2 = st.columns(2)
        with col1:
            lv_s1 = st.selectbox(
                "LV1 (X-axis):",
                options=list(range(1, n_comp + 1)),
                index=0,
                key="score_lv1"
            )
        with col2:
            lv_s2 = st.selectbox(
                "LV2 (Y-axis):",
                options=list(range(1, n_comp + 1)),
                index=min(1, n_comp - 1),
                key="score_lv2"
            )

        try:
            y_vals = st.session_state.get('pls_Y', None)
            if y_vals is None:
                y_vals = st.session_state['dataframe'][st.session_state['y_column']].values
            fig_score = plot_score_plot_with_ellipse(model, lv1=lv_s1, lv2=lv_s2, y_data=y_vals)
            st.plotly_chart(fig_score, use_container_width=True)

            with st.expander("📖 Score Plot Meaning?"):
                st.markdown("""
                - Inside ellipse (95%): Normal samples
                - Outside ellipse: Outliers (check data quality!)
                - Color: Response variable value
                """)
        except Exception as e:
            st.error(f"Error: {str(e)}")

    with diag_tab2:
        st.subheader("Residuals Histogram (Normality Check)")
        residuals = model.get('residuals', np.array([]))

        try:
            fig_hist = plot_residuals_histogram(residuals)
            st.plotly_chart(fig_hist, use_container_width=True)

            with st.expander("📖 Histogram Meaning?"):
                st.markdown("""
                ✓ Mean ≈ 0: No systematic bias
                ✓ Bell-shaped: Normal distribution
                ✓ Shapiro-Wilk p > 0.05: Normal confirmed
                """)
        except Exception as e:
            st.error(f"Error: {str(e)}")

    with diag_tab3:
        st.subheader("Q-Q Plot (Normality Assessment)")
        residuals = model.get('residuals', np.array([]))

        try:
            fig_qq = plot_qq_plot(residuals)
            st.plotly_chart(fig_qq, use_container_width=True)

            with st.expander("📖 Q-Q Plot Meaning?"):
                st.markdown("""
                ✓ Points on diagonal: Normal distribution
                ✗ S-curve: Heavy tails (outliers)
                ✗ Deviation: Non-normal errors
                """)
        except Exception as e:
            st.error(f"Error: {str(e)}")


def render_diagnostics_tab():
    """Render the Model Diagnostics tab - TAB 3: External Test Set Validation."""

    # Check if model is available
    if 'final_model' not in st.session_state:
        st.info("👉 Complete calibration in TAB 1 first to perform test validation")
        return

    # Get model and data from session state
    model = st.session_state['final_model']
    scaler_info = st.session_state['scaler_info']
    selected_lv = st.session_state['selected_lv']

    st.header("External Test Set Validation")

    ### Load Test Set
    st.subheader("Step 1: Load Test Set")

    col1, col2 = st.columns([1, 1])

    with col1:
        test_file = st.file_uploader(
            "Upload Test Set (CSV/Excel)",
            type=['csv', 'xlsx', 'xls'],
            key='test_upload',
            help="Upload a file with the same features as calibration set"
        )

    with col2:
        if test_file:
            try:
                if test_file.name.endswith('.csv'):
                    test_df = pd.read_csv(test_file)
                else:
                    test_df = pd.read_excel(test_file)

                st.success(f"✅ Loaded {test_df.shape[0]} samples × {test_df.shape[1]} columns")
                st.session_state['test_df'] = test_df

                with st.expander("📊 Test Data Preview"):
                    st.dataframe(test_df.head(10), use_container_width=True)

            except Exception as e:
                st.error(f"❌ Error loading file: {str(e)}")

    if 'test_df' in st.session_state:
        test_df = st.session_state['test_df']

        ### Select X and Y from Test
        st.subheader("Step 2: Define Test Set Variables")

        col1, col2 = st.columns(2)

        with col1:
            # Get training X columns for comparison
            training_x_cols = st.session_state.get('x_columns', [])

            test_x_cols = st.multiselect(
                "X Variables (must match training)",
                options=list(test_df.columns),
                default=[col for col in training_x_cols if col in test_df.columns],
                key='test_x_cols',
                help=f"Training used: {training_x_cols}"
            )

        with col2:
            training_y_col = st.session_state.get('y_column', None)

            test_y_col = st.selectbox(
                "Y Variable (response)",
                options=list(test_df.columns),
                index=list(test_df.columns).index(training_y_col) if training_y_col in test_df.columns else 0,
                key='test_y_col',
                help=f"Training used: {training_y_col}"
            )

        # Validation
        if test_x_cols and test_y_col:
            # Check if variables match training
            if set(test_x_cols) != set(st.session_state['x_columns']):
                st.warning("⚠️ X variables don't exactly match training set!")
                missing = set(st.session_state['x_columns']) - set(test_x_cols)
                extra = set(test_x_cols) - set(st.session_state['x_columns'])
                if missing:
                    st.error(f"Missing: {missing}")
                if extra:
                    st.error(f"Extra: {extra}")
            else:
                st.info(f"✅ Variables match training: {len(test_x_cols)} features")

            # Run validation
            if st.button("▶️ Validate on Test Set", key="run_test_validation", type="primary"):
                with st.spinner("Running test set validation..."):
                    try:
                        # Prepare test data
                        X_test_df = test_df[test_x_cols]
                        y_test = test_df[test_y_col].values

                        # Reorder to match training
                        X_test_df = X_test_df[st.session_state['x_columns']]

                        # Apply scaling
                        from calibration_utils.pls_preprocessing import apply_scaler
                        X_test_scaled = apply_scaler(X_test_df.values, scaler_info)

                        # Create test data dict
                        test_data = {
                            'X_test': X_test_scaled,
                            'y_test': y_test,
                            'sample_names': [f"Test_{i+1}" for i in range(len(y_test))],
                            'feature_names': list(X_test_df.columns),
                            'n_samples': len(y_test),
                            'n_features': X_test_scaled.shape[1]
                        }

                        # Validate
                        X_cal = st.session_state.get('X_scaled', None)
                        val_result = validate_on_test(model, test_data, X_cal)

                        st.session_state['test_results'] = val_result
                        st.session_state['test_y_true'] = y_test

                        st.success("✅ Test set validation completed!")
                        st.rerun()

                    except Exception as e:
                        st.error(f"❌ Error during validation: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())

    # Display test results if available
    if 'test_results' in st.session_state:
        test_res = st.session_state['test_results']
        y_test_true = st.session_state['test_y_true']

        st.divider()
        st.subheader("Test Set Results")

        ### Metrics Summary
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("RMSE Test", f"{test_res['metrics']['RMSE']:.4f}")

        with col2:
            r2_test = test_res['metrics']['R2']
            st.metric("R² Test", f"{r2_test:.4f}")

        with col3:
            st.metric("Bias", f"{test_res['metrics']['Bias']:.4f}")

        with col4:
            st.metric("Test Samples", len(test_res['y_pred']))

        ### Result Tabs
        test_tab1, test_tab2, test_tab3, test_tab4 = st.tabs([
            "📊 Predictions",
            "📉 Residuals",
            "📋 Comparison",
            "🎯 Sample Details"
        ])

        with test_tab1:
            st.subheader("Test Set: Predictions vs Observed")

            # Plot test predictions vs observed
            try:
                fig_test_pred = plot_predictions_vs_observed(
                    y_true=test_res['y_true'],
                    y_pred=test_res['y_pred'],
                    title="Test Set: Predicted vs Observed",
                    sample_names=test_res.get('sample_names', [f"Test_{i+1}" for i in range(len(test_res['y_pred']))]),
                    theme='light'
                )
                st.plotly_chart(fig_test_pred, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating test predictions plot: {str(e)}")

            # Show metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("MAE", f"{test_res['metrics']['MAE']:.4f}")
            with col2:
                st.metric("RMSE", f"{test_res['metrics']['RMSE']:.4f}")
            with col3:
                st.metric("R²", f"{test_res['metrics']['R2']:.4f}")

        with test_tab2:
            st.subheader("Test Set: Residuals Analysis")

            # Plot test residuals diagnostics
            try:
                fig_test_resid = plot_residuals(
                    y_true=test_res['y_true'],
                    y_pred=test_res['y_pred'],
                    sample_names=test_res.get('sample_names', [f"Test_{i+1}" for i in range(len(test_res['y_pred']))]),
                    theme='light'
                )
                st.plotly_chart(fig_test_resid, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating test residuals plot: {str(e)}")

            # Show outliers
            if test_res['n_outliers'] > 0:
                st.warning(f"⚠️ {test_res['n_outliers']} outlier samples detected (|std_residual| > 2.5)")

            if test_res['n_extrapolating'] > 0:
                st.warning(f"⚠️ {test_res['n_extrapolating']} samples extrapolating beyond calibration range")

        with test_tab3:
            st.subheader("Calibration vs Cross-Validation vs Test Comparison")

            # Get CV metrics
            cv_results = st.session_state['cv_results']
            optimal_lv = st.session_state['optimal_lv']

            # Build comparison table
            comparison_data = {
                'Metric': ['RMSE', 'R²', 'MAE', 'Bias', 'N Samples'],
                'Calibration': [
                    f"{model['RMSE']:.4f}",
                    f"{model['R2']:.4f}",
                    '-',
                    '-',
                    str(model['n_samples'])
                ],
                'Cross-Validation': [
                    f"{cv_results['RMSECV'][optimal_lv-1]:.4f}",
                    f"{cv_results['R2CV'][optimal_lv-1]:.4f}",
                    '-',
                    '-',
                    str(model['n_samples'])
                ],
                'Test Set': [
                    f"{test_res['metrics']['RMSE']:.4f}",
                    f"{test_res['metrics']['R2']:.4f}",
                    f"{test_res['metrics']['MAE']:.4f}",
                    f"{test_res['metrics']['Bias']:.4f}",
                    str(len(test_res['y_pred']))
                ]
            }

            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True, hide_index=True)

            # Performance assessment
            cv_rmse = cv_results['RMSECV'][optimal_lv-1]
            test_rmse = test_res['metrics']['RMSE']

            if test_rmse > cv_rmse * 1.5:
                st.error("❌ Test RMSE significantly higher than CV - possible overfitting or distribution shift")
            elif test_rmse > cv_rmse * 1.2:
                st.warning("⚠️ Test RMSE moderately higher than CV - monitor model performance")
            else:
                st.success("✅ Test performance consistent with cross-validation")

        with test_tab4:
            st.subheader("Sample-by-Sample Test Analysis")

            # Create detailed table
            test_detail_df = pd.DataFrame({
                'Sample': test_res.get('sample_names', [f"Test_{i+1}" for i in range(len(test_res['y_pred']))]),
                'Observed': y_test_true,
                'Predicted': test_res['y_pred'],
                'Residual': test_res['residuals'],
                'Std_Residual': test_res['std_residuals'],
                'Outlier': test_res['outlier_flags'],
                'Extrapolating': test_res['extrapolation_flags']
            })

            # Highlight outliers
            def highlight_outliers(row):
                if row['Outlier']:
                    return ['background-color: #ffcccc'] * len(row)
                elif row['Extrapolating']:
                    return ['background-color: #ffffcc'] * len(row)
                return [''] * len(row)

            st.dataframe(
                test_detail_df.style.apply(highlight_outliers, axis=1),
                use_container_width=True,
                height=400
            )

            st.caption("🔴 Red: Outliers | 🟡 Yellow: Extrapolating")

        ### Export Report
        st.divider()
        st.subheader("Export Test Results")

        col1, col2 = st.columns(2)

        with col1:
            # Prepare comprehensive report
            from datetime import datetime

            report_data = {
                'Model_Info': [
                    f"N Components: {selected_lv}",
                    f"N Features: {model['n_features']}",
                    f"N Cal Samples: {model['n_samples']}",
                    f"N Test Samples: {len(test_res['y_pred'])}"
                ],
                'Performance': [
                    f"Cal RMSE: {model['RMSE']:.4f}",
                    f"CV RMSE: {cv_results['RMSECV'][optimal_lv-1]:.4f}",
                    f"Test RMSE: {test_res['metrics']['RMSE']:.4f}",
                    f"Test R²: {test_res['metrics']['R2']:.4f}"
                ]
            }

            report_df = pd.DataFrame(report_data)
            csv = report_df.to_csv(index=False)

            st.download_button(
                label="📥 Download Summary Report CSV",
                data=csv,
                file_name=f"pls_test_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                key="download_summary"
            )

        with col2:
            # Prepare detailed predictions
            predictions_csv = test_detail_df.to_csv(index=False)

            st.download_button(
                label="📥 Download Test Predictions CSV",
                data=predictions_csv,
                file_name=f"pls_test_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                key="download_predictions_test"
            )


def main():
    """Main application entry point."""
    # Render header
    render_header()

    # Render sidebar and get configuration
    config = render_sidebar()

    # Create tabs
    tab1, tab2, tab3 = st.tabs([
        "📈 Calibration & CV",
        "📊 Results & Graphics",
        "🧪 Test Set Validation"
    ])

    with tab1:
        render_calibration_tab()

    with tab2:
        render_validation_tab()

    with tab3:
        show_tab3_test_set_validation()


def show():
    """
    Entry point for homepage navigation.
    Wrapper function for compatibility with other modules.
    """
    main()


if __name__ == "__main__":
    main()
