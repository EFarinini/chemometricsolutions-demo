"""
PLS Test Set Validation - TAB 3 Implementation
==============================================

Complete Streamlit implementation following classification_page.py pattern:
1. Select test dataset from workspace
2. Choose X/Y columns
3. Select split strategy (venetian blind, block, random)
4. Validate model
5. Show results & diagnostics

Author: ChemoMetric Solutions
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Dict, Any, Optional, Tuple


def show_tab3_test_set_validation():
    """TAB 3: External Test Set Validation"""

    st.header("🧪 External Test Set Validation")

    st.markdown("""
    Validate your PLS model on independent test data.

    **Steps:**
    1. Select test dataset from workspace
    2. Choose X and Y columns
    3. Run validation
    4. Analyze results
    """)

    st.divider()

    # ===== STEP 1: Check if model available =====
    if 'final_model' not in st.session_state:
        st.info("👉 Complete calibration in **Calibration & CV** TAB first")
        return

    model = st.session_state['final_model']

    # ===== STEP 2: Load datasets from workspace =====
    st.markdown("### 📥 Step 1: Select Test Dataset")

    try:
        from workspace_utils import get_workspace_datasets
        available_datasets = get_workspace_datasets()

        if not available_datasets:
            st.warning("⚠️ No datasets available in workspace. Load data first.")
            return
    except Exception as e:
        st.error(f"Error loading workspace: {str(e)}")
        return

    # Dataset selector
    test_dataset_name = st.selectbox(
        "📂 Select test dataset:",
        options=list(available_datasets.keys()),
        key="test_dataset_select",
        help="Choose dataset for external validation"
    )

    if test_dataset_name not in available_datasets:
        st.error("Dataset not found")
        return

    test_data = available_datasets[test_dataset_name]

    st.write(f"**Dataset info:** {test_data.shape[0]} rows × {test_data.shape[1]} columns")

    # Sample selection
    st.markdown("#### 🔢 Select Sample Range (rows)")

    n_samples_test = len(test_data)

    col_s1, col_s2 = st.columns(2)

    with col_s1:
        first_sample_test = st.number_input(
            "First sample (1-based):",
            min_value=1,
            max_value=n_samples_test,
            value=1,
            key="first_sample_test"
        )

    with col_s2:
        last_sample_test = st.number_input(
            "Last sample (1-based):",
            min_value=first_sample_test,
            max_value=n_samples_test,
            value=n_samples_test,
            key="last_sample_test"
        )

    # Filter samples
    sample_range_test = list(range(first_sample_test - 1, last_sample_test))
    test_data = test_data.iloc[sample_range_test].copy()

    st.write(f"**Selected samples:** {len(test_data)} rows (from {first_sample_test} to {last_sample_test})")

    st.divider()

    with st.expander("📊 Data Preview"):
        st.dataframe(test_data.head(10), use_container_width=True)

    st.divider()

    # ===== STEP 3: Select X and Y columns =====
    st.markdown("### 📋 Step 2: Select X and Y Columns")

    all_columns = test_data.columns.tolist()
    numeric_cols = test_data.select_dtypes(include=[np.number]).columns.tolist()
    non_numeric_cols = [c for c in all_columns if c not in numeric_cols]

    col_info1, col_info2, col_info3 = st.columns(3)
    with col_info1:
        st.metric("Total columns", len(all_columns))
    with col_info2:
        st.metric("Numeric columns", len(numeric_cols))
    with col_info3:
        st.metric("Non-numeric columns", len(non_numeric_cols))

    if len(non_numeric_cols) > 0:
        st.write(f"**Non-numeric columns (potential Y):** {non_numeric_cols}")

    # ===== SELECT X COLUMNS (same as calibration) =====
    st.markdown("#### 📊 Select X Column Range (numeric predictors)")

    if len(numeric_cols) > 0:
        first_numeric_idx = all_columns.index(numeric_cols[0]) + 1
        last_numeric_idx = all_columns.index(numeric_cols[-1]) + 1
    else:
        first_numeric_idx = 1
        last_numeric_idx = len(all_columns)

    col_range1, col_range2 = st.columns(2)

    with col_range1:
        first_col = st.number_input(
            "First column (1-based):",
            min_value=1,
            max_value=len(all_columns),
            value=first_numeric_idx,
            key="test_first_col",
            help="Start of predictor columns"
        )

    with col_range2:
        last_col = st.number_input(
            "Last column (1-based):",
            min_value=first_col,
            max_value=len(all_columns),
            value=last_numeric_idx,
            key="test_last_col",
            help="End of predictor columns"
        )

    # Extract X columns
    x_col_indices = list(range(first_col - 1, last_col))
    x_columns = [all_columns[i] for i in x_col_indices]
    X_test_df = test_data.iloc[:, x_col_indices].copy()

    # Validate X
    if X_test_df.shape[1] == 0:
        st.error("❌ No columns selected for X! Adjust column range.")
        st.stop()

    if not X_test_df.select_dtypes(include=[np.number]).shape[1] == X_test_df.shape[1]:
        st.error("❌ X must contain only numeric columns!")
        st.stop()

    st.success(f"✅ X Matrix Selected: {X_test_df.shape[0]} samples × {X_test_df.shape[1]} variables")

    with st.expander("🔍 X Matrix Preview"):
        st.dataframe(X_test_df.head(), use_container_width=True)

    # ===== SELECT Y COLUMN =====
    st.markdown("#### 🎯 Select Target Variable Y (response)")

    remaining_cols = [c for c in all_columns if c not in x_columns]
    numeric_remaining = test_data[remaining_cols].select_dtypes(include=[np.number]).columns.tolist()

    if len(numeric_remaining) == 0:
        st.error("❌ No numeric columns available for Y!")
        st.stop()

    y_column = st.selectbox(
        "Select response variable (quantitative):",
        options=numeric_remaining,
        key="test_y_select",
        help="Choose numeric column for regression target"
    )

    Y_test_df = test_data[[y_column]].copy()

    # Validate Y
    if Y_test_df.isnull().any().any():
        st.warning(f"⚠️ Y contains {Y_test_df.isnull().sum().sum()} missing values")

    st.success(f"✅ Y Variable Selected: **{y_column}**")

    y_stats = test_data[y_column].describe()
    col_y1, col_y2, col_y3, col_y4 = st.columns(4)
    with col_y1:
        st.metric("Min", f"{y_stats['min']:.3f}")
    with col_y2:
        st.metric("Max", f"{y_stats['max']:.3f}")
    with col_y3:
        st.metric("Mean", f"{y_stats['mean']:.3f}")
    with col_y4:
        st.metric("Std", f"{y_stats['std']:.3f}")

    st.divider()

    # ===== STEP 4: Run validation =====
    st.markdown("### ✅ Step 3: Run Validation")

    X_test = X_test_df.values.astype(np.float64)
    y_test = Y_test_df.values.astype(np.float64).flatten()

    # Check if X columns match calibration
    cal_x_columns = st.session_state.get('x_columns', [])
    if set(x_columns) != set(cal_x_columns):
        st.warning("⚠️ Test X columns don't match calibration X columns!")
        st.write(f"**Calibration columns:** {len(cal_x_columns)}")
        st.write(f"**Test columns:** {len(x_columns)}")

    # Display preprocessing info
    preprocessing = st.session_state.get('preprocessing', None)
    if preprocessing:
        st.info(f"📊 Preprocessing method from calibration: **{preprocessing}**")
        st.info("✅ Test data will be preprocessed automatically using training parameters")

    if st.button("▶️ Run Test Set Validation", key="test_run_validation", type="primary", use_container_width=True):

        with st.spinner("Validating model on test set..."):
            try:
                # Import PLS functions
                from calibration_utils.pls_calculations import pls_predict, calculate_metrics, calculate_residuals

                # NOTE: Pass RAW test data to pls_predict()
                # pls_predict() will automatically apply the SAME preprocessing as calibration
                # using the scaler_info stored inside the model
                y_pred = pls_predict(model, X_test)

                # Calculate metrics
                metrics = calculate_metrics(y_test, y_pred)
                residuals_dict = calculate_residuals(y_test, y_pred)
                residuals = residuals_dict['residuals']

                # Store results
                st.session_state['test_validation_results'] = {
                    'y_test': y_test,
                    'y_pred': y_pred,
                    'residuals': residuals,
                    'metrics': metrics,
                    'dataset_name': test_dataset_name,
                    'n_samples': len(y_test),
                    'x_columns': x_columns,
                    'y_column': y_column
                }

                st.success("✅ Validation complete!")
                st.rerun()

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                import traceback
                st.code(traceback.format_exc())

    st.divider()

    # ===== STEP 5: Display results =====
    if 'test_validation_results' in st.session_state:
        results = st.session_state['test_validation_results']

        st.markdown("### 📊 Validation Results")

        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("R² (Test)", f"{results['metrics']['R2']:.3f}")
        with col2:
            st.metric("RMSE (Test)", f"{results['metrics']['RMSE']:.3f}")
        with col3:
            st.metric("MAE (Test)", f"{results['metrics']['MAE']:.3f}")
        with col4:
            st.metric("Bias (Test)", f"{results['metrics']['Bias']:.3f}")

        # ===== PREPROCESSING VERIFICATION =====
        with st.expander("🔍 Preprocessing Verification (MATLAB Compatible)"):
            preprocessing_method = st.session_state.get('preprocessing', 'unknown')

            # Get scaler info from model
            scaler_info = model.get('scaler_info', None)

            if scaler_info and preprocessing_method:
                col1, col2 = st.columns(2)

                with col1:
                    st.write("**Calibration Preprocessing:**")
                    st.write(f"Method: **{preprocessing_method}**")
                    st.write(f"N Features: {model.get('n_features', '?')}")
                    st.write(f"N Samples: {model.get('n_samples', '?')}")

                    if preprocessing_method == 'autoscale':
                        has_scaler = 'X_scaler' in scaler_info and scaler_info['X_scaler'] is not None
                        st.success(f"✅ Using sklearn StandardScaler: {has_scaler}")
                        if 'X_mean_raw' in scaler_info:
                            st.write(f"**Training Mean (first 5):**")
                            st.code(f"{scaler_info['X_mean_raw'][:5]}")
                        if 'X_scale_raw' in scaler_info:
                            st.write(f"**Training Std (first 5):**")
                            st.code(f"{scaler_info['X_scale_raw'][:5]}")
                    elif preprocessing_method == 'mean_center':
                        if 'X_mean_raw' in scaler_info:
                            st.write(f"**Training Mean (first 5):**")
                            st.code(f"{scaler_info['X_mean_raw'][:5]}")

                with col2:
                    st.write("**Test Data (RAW, before preprocessing):**")
                    st.write(f"Shape: {results['n_samples']} × {len(results['x_columns'])}")
                    st.write(f"**Note:** Test data is passed RAW to pls_predict()")
                    st.write(f"Preprocessing applied automatically inside pls_predict()")

                st.divider()
                st.info("✅ Test data preprocessed using **SAME** parameters as training (MATLAB compatible)")
            else:
                st.warning("No preprocessing info available in model")

        st.divider()

        # ===== RESULTS TABS =====
        results_tab1, results_tab2, results_tab3, results_tab4 = st.tabs([
            "📈 Predictions vs Observed",
            "📉 Residuals Analysis",
            "📋 Sample Details",
            "📊 Statistics"
        ])

        with results_tab1:
            st.subheader("Predictions vs Observed")

            y_test = results['y_test']
            y_pred = results['y_pred']

            # Scatter plot
            fig = go.Figure()

            # Perfect prediction line
            min_val = min(y_test.min(), y_pred.min())
            max_val = max(y_test.max(), y_pred.max())

            fig.add_trace(go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='Perfect Prediction',
                line=dict(color='red', width=2, dash='dash')
            ))

            # Predictions
            fig.add_trace(go.Scatter(
                x=y_test,
                y=y_pred,
                mode='markers',
                name='Predictions',
                marker=dict(size=8, color='blue', opacity=0.6),
                text=[f"Obs: {o:.2f}<br>Pred: {p:.2f}" for o, p in zip(y_test, y_pred)],
                hovertemplate='%{text}<extra></extra>'
            ))

            fig.update_layout(
                title=f"Test Set: Predictions vs Observed ({len(y_test)} samples)",
                xaxis_title="Observed (Test)",
                yaxis_title="Predicted",
                width=800,
                height=600,
                template='plotly_white',
                xaxis=dict(scaleanchor="y", scaleratio=1),
                yaxis=dict(scaleanchor="x", scaleratio=1)
            )

            st.plotly_chart(fig, use_container_width=True)

            with st.expander("📖 Interpretation"):
                st.markdown("""
                **Good model:**
                - Points clustered on red diagonal line
                - Symmetric scatter around line
                - Few outliers

                **Warning signs:**
                - Large scatter around line
                - Systematic deviation (curve pattern)
                - Multiple outliers
                """)

        with results_tab2:
            st.subheader("Residuals Analysis")

            residuals = results['residuals']

            # Histogram
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(
                x=residuals,
                nbinsx=20,
                name='Residuals',
                marker=dict(color='skyblue', line=dict(color='navy', width=1))
            ))

            fig_hist.add_vline(x=0, line_dash="dash", line_color="red", line_width=2)
            fig_hist.update_layout(
                title="Residuals Distribution",
                xaxis_title="Residual",
                yaxis_title="Frequency",
                template='plotly_white',
                width=800,
                height=500
            )

            st.plotly_chart(fig_hist, use_container_width=True)

            # Stats
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Mean", f"{np.mean(residuals):.4f}")
            with col2:
                st.metric("Std Dev", f"{np.std(residuals):.4f}")
            with col3:
                st.metric("Max Abs", f"{np.max(np.abs(residuals)):.4f}")

        with results_tab3:
            st.subheader("Sample Details")

            # Create sample table
            sample_df = pd.DataFrame({
                'Sample': range(1, len(y_test) + 1),
                'Observed': results['y_test'],
                'Predicted': results['y_pred'],
                'Residual': results['residuals'],
                'Error %': (results['residuals'] / results['y_test'] * 100)
            })

            # Highlight errors
            st.dataframe(
                sample_df.style.background_gradient(
                    subset=['Error %'],
                    cmap='RdYlGn_r'
                ),
                use_container_width=True,
                height=400
            )

            # Download button
            csv = sample_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Sample Details CSV",
                data=csv,
                file_name="pls_test_sample_details.csv",
                mime="text/csv"
            )

        with results_tab4:
            st.subheader("Statistical Summary")

            # Comparison: Calibration vs Cross-Validation vs Test
            cv_results = st.session_state.get('cv_results', None)
            optimal_lv = st.session_state.get('optimal_lv', None)

            comparison_data = {
                'Metric': ['R²', 'RMSE', 'MAE', 'Bias'],
                'Calibration': [
                    f"{model.get('R2', 0):.4f}",
                    f"{model.get('RMSE', 0):.4f}",
                    '-',
                    '-'
                ],
                'Cross-Validation': [
                    f"{cv_results['R2CV'][optimal_lv-1]:.4f}" if cv_results and optimal_lv else '-',
                    f"{cv_results['RMSECV'][optimal_lv-1]:.4f}" if cv_results and optimal_lv else '-',
                    '-',
                    '-'
                ] if cv_results else ['-', '-', '-', '-'],
                'Test Set': [
                    f"{results['metrics']['R2']:.4f}",
                    f"{results['metrics']['RMSE']:.4f}",
                    f"{results['metrics']['MAE']:.4f}",
                    f"{results['metrics']['Bias']:.4f}"
                ]
            }

            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True, hide_index=True)

            # Performance assessment
            if cv_results and optimal_lv:
                cv_rmse = cv_results['RMSECV'][optimal_lv-1]
                test_rmse = results['metrics']['RMSE']

                if test_rmse > cv_rmse * 1.5:
                    st.error("❌ Test RMSE significantly higher than CV - possible overfitting or distribution shift")
                elif test_rmse > cv_rmse * 1.2:
                    st.warning("⚠️ Test RMSE moderately higher than CV - monitor model performance")
                else:
                    st.success("✅ Test performance consistent with cross-validation")
