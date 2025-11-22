"""
Multi-DOE Predictions Module
Make predictions for all response variables simultaneously

This module provides prediction UI for Multi-DOE analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np

# Import existing prediction utilities
from .surface_analysis import create_prediction_matrix
from .model_computation_multidoe import calculate_ci_at_point_experimental


def show_predictions_ui_multidoe(models_dict, x_vars, y_vars, data):
    """
    Predictions UI for multi-DOE

    Args:
        models_dict (dict): Dictionary {y_name: model_result}
        x_vars (list): List of X variable names
        y_vars (list): List of Y variable names
        data (DataFrame): Original dataset
    """
    st.markdown("## 🔮 Multi-Response Predictions")
    st.info("""
    Make predictions for all response variables simultaneously.
    Enter values for the predictor variables and get predictions for all responses at once.
    """)

    # ========================================================================
    # SECTION 1: Prediction Mode Selection
    # ========================================================================
    st.markdown("### Prediction Mode")
    pred_mode = st.radio(
        "Choose prediction mode:",
        ["Single Point", "Data Row Selection"],
        key="multidoe_pred_mode"
    )

    # ========================================================================
    # SECTION 2: Single Point Prediction
    # ========================================================================
    if pred_mode == "Single Point":
        st.markdown("#### Enter Point Coordinates")
        st.info("Enter values for all X variables to get predictions for all Y responses")

        # Create input fields for all X variables
        input_dict = {}
        cols = st.columns(3)
        for i, x_var in enumerate(x_vars):
            with cols[i % 3]:
                input_dict[x_var] = st.number_input(
                    f"{x_var}:",
                    value=0.0,
                    step=0.1,
                    format="%.3f",
                    key=f"multidoe_pred_input_{x_var}"
                )

        if st.button("🔮 Predict All Responses", type="primary", key="multidoe_predict_single"):
            try:
                # Create DataFrame for prediction point
                X_point = pd.DataFrame([input_dict])

                # Get CI parameters from session state (if available)
                ci_params = st.session_state.get('multidoe_ci_params', {})
                use_ci = ci_params.get('ci_type') == 'Experimental' and all(
                    k in ci_params for k in ['s_exp', 'dof_exp', 'n_replicates']
                )

                # Predict all Y responses for this point
                predictions = {}
                for y_name, model in models_dict.items():
                    if 'error' in model:
                        predictions[y_name] = {'value': np.nan, 'error': model['error']}
                        continue

                    try:
                        # Get coefficient names from model
                        coef_names = model['coefficients'].index.tolist()

                        # Create model matrix for prediction
                        X_model = create_prediction_matrix(
                            X_point.values,
                            x_vars,
                            coef_names
                        )

                        # Calculate prediction
                        pred = X_model @ model['coefficients'].values

                        # Calculate CI if experimental parameters available
                        ci_result = None
                        if use_ci:
                            try:
                                X_point_array = np.array([input_dict[xv] for xv in x_vars])
                                ci_result = calculate_ci_at_point_experimental(
                                    model_result=model,
                                    X_point=X_point_array,
                                    x_vars=x_vars,
                                    ci_type='Experimental',
                                    s_model=ci_params.get('s_model'),
                                    dof_model=ci_params.get('dof_model'),
                                    n_replicates=ci_params.get('n_replicates', 1),
                                    s_exp=ci_params.get('s_exp'),
                                    dof_exp=ci_params.get('dof_exp')
                                )
                            except Exception as ci_error:
                                # CI calculation failed, continue with point prediction
                                ci_result = None

                        predictions[y_name] = {
                            'value': pred[0],
                            'error': None,
                            'ci_result': ci_result
                        }

                    except Exception as e:
                        predictions[y_name] = {'value': np.nan, 'error': str(e), 'ci_result': None}

                # ============================================================
                # DISPLAY: Results for all responses
                # ============================================================
                st.markdown("#### 🎯 Predictions for All Responses")

                # Display predictions in columns
                n_cols = min(len(y_vars), 4)
                cols = st.columns(n_cols)

                for i, y_name in enumerate(y_vars):
                    with cols[i % n_cols]:
                        pred_info = predictions[y_name]
                        if pred_info['error']:
                            st.metric(
                                y_name,
                                "Error",
                                delta=None
                            )
                            st.caption(f"❌ {pred_info['error'][:30]}")
                        else:
                            st.metric(
                                y_name,
                                f"{pred_info['value']:.4f}"
                            )
                            # Show CI if available
                            if pred_info.get('ci_result'):
                                ci = pred_info['ci_result']
                                st.caption(f"CI: [{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}]")
                                st.caption(f"± {ci['ci_semiamplitude']:.4f}")

                # Display prediction table
                st.markdown("---")
                st.markdown("#### Prediction Summary Table")

                # Build table with CI columns if available
                pred_table = []
                has_ci = any(predictions[y].get('ci_result') for y in y_vars)

                for y_name in y_vars:
                    pred_info = predictions[y_name]
                    row = {
                        'Response': y_name,
                        'Prediction': pred_info['value'] if not pred_info['error'] else np.nan,
                    }

                    if has_ci:
                        if pred_info.get('ci_result'):
                            ci = pred_info['ci_result']
                            row['CI_Lower'] = ci['ci_lower']
                            row['CI_Upper'] = ci['ci_upper']
                            row['CI_±'] = ci['ci_semiamplitude']
                        else:
                            row['CI_Lower'] = np.nan
                            row['CI_Upper'] = np.nan
                            row['CI_±'] = np.nan

                    row['Status'] = '✅ OK' if not pred_info['error'] else f"❌ {pred_info['error'][:30]}"
                    pred_table.append(row)

                pred_df = pd.DataFrame(pred_table)
                st.dataframe(pred_df, use_container_width=True, hide_index=True)

                # Show CI details if available
                if has_ci and use_ci:
                    with st.expander("📊 Confidence Interval Details", expanded=False):
                        st.info(f"""
                        **CI Type**: Experimental (Model + Measurement Uncertainty)

                        **Parameters**:
                        - Model std dev: {ci_params.get('s_model', 0):.6f}
                        - Model DOF: {ci_params.get('dof_model', 0)}
                        - Experimental std dev: {ci_params.get('s_exp', 0):.6f}
                        - Experimental DOF: {ci_params.get('dof_exp', 0)}
                        - Replicates: {ci_params.get('n_replicates', 1)}

                        **Formula**: CI_total = √((CI_model)² + (CI_experimental)²)
                        """)

                        # Show per-response CI breakdown
                        ci_breakdown = []
                        for y_name in y_vars:
                            pred_info = predictions[y_name]
                            if pred_info.get('ci_result'):
                                ci = pred_info['ci_result']
                                ci_breakdown.append({
                                    'Response': y_name,
                                    'CI_model': ci.get('ci_model', np.nan),
                                    'CI_exp': ci.get('ci_exp', np.nan),
                                    'CI_total': ci['ci_semiamplitude'],
                                    'Leverage': ci['leverage']
                                })

                        if ci_breakdown:
                            ci_df = pd.DataFrame(ci_breakdown)
                            st.dataframe(ci_df, use_container_width=True, hide_index=True)

                # Store for later use
                st.session_state.multidoe_last_prediction = {
                    'X_input': input_dict,
                    'predictions': predictions
                }

            except Exception as e:
                st.error(f"❌ Prediction failed: {str(e)}")
                import traceback
                with st.expander("🐛 Error details"):
                    st.code(traceback.format_exc())

    # ========================================================================
    # SECTION 3: Data Row Selection
    # ========================================================================
    elif pred_mode == "Data Row Selection":
        st.markdown("#### Select Rows from Dataset")

        # Display data with index
        display_df = data.copy()
        display_df.insert(0, 'Row', range(len(display_df)))

        st.dataframe(display_df, use_container_width=True, height=300)

        # Row selection
        selected_rows = st.multiselect(
            "Select rows to predict:",
            range(len(data)),
            key="multidoe_pred_selected_rows"
        )

        if len(selected_rows) > 0:
            if st.button("🔮 Predict Selected Rows", type="primary", key="multidoe_predict_rows"):
                try:
                    results = []

                    for row_idx in selected_rows:
                        X_point = data.iloc[row_idx:row_idx+1][x_vars]

                        for y_name, model in models_dict.items():
                            if 'error' in model:
                                results.append({
                                    'Row': row_idx,
                                    'Y_Variable': y_name,
                                    'Prediction': np.nan,
                                    'Status': f"❌ {model['error'][:30]}"
                                })
                                continue

                            try:
                                coef_names = model['coefficients'].index.tolist()
                                X_model = create_prediction_matrix(
                                    X_point.values,
                                    x_vars,
                                    coef_names
                                )
                                pred = X_model @ model['coefficients'].values

                                results.append({
                                    'Row': row_idx,
                                    'Y_Variable': y_name,
                                    'Prediction': pred[0],
                                    'Status': '✅ OK'
                                })

                            except Exception as e:
                                results.append({
                                    'Row': row_idx,
                                    'Y_Variable': y_name,
                                    'Prediction': np.nan,
                                    'Status': f"❌ {str(e)[:30]}"
                                })

                    # Display results
                    results_df = pd.DataFrame(results)

                    st.markdown("#### Prediction Results")
                    st.dataframe(results_df, use_container_width=True, hide_index=True)

                    # Pivot table view
                    with st.expander("📊 Pivot Table View", expanded=True):
                        pivot_df = results_df.pivot(
                            index='Row',
                            columns='Y_Variable',
                            values='Prediction'
                        )
                        st.dataframe(pivot_df, use_container_width=True)

                    # Store results
                    st.session_state.multidoe_row_predictions = results_df

                except Exception as e:
                    st.error(f"❌ Prediction failed: {str(e)}")
                    import traceback
                    with st.expander("🐛 Error details"):
                        st.code(traceback.format_exc())
        else:
            st.info("Select one or more rows to generate predictions")
