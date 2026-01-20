"""
Predictions for New Points
Equivalent to DOE_prediction.r
Predict response for new experimental points with confidence intervals and leverage
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats


def predict_new_points(model_results, X_new, y_new=None, s=None, dof=None):
    """
    Predict response for new experimental points
    
    Args:
        model_results: dict from fit_mlr_model
        X_new: DataFrame with new predictor values (same structure as training X)
        y_new: optional experimental Y values for comparison
        s: experimental standard deviation (if None, uses model RMSE)
        dof: degrees of freedom (if None, uses model dof)
    
    Returns:
        DataFrame with predictions, leverage, CI, and optionally residuals
    """
    coefficients = model_results['coefficients']
    dispersion = model_results['XtX_inv']
    
    # Use model values if not provided
    if s is None:
        s = model_results.get('rmse', 1.0)
    if dof is None:
        dof = model_results.get('dof', 1)
    
    # Convert to model matrix format
    X_model = X_new.values
    
    # Predictions
    y_pred = X_model @ coefficients.values
    
    # Leverage for new points
    leverage = np.diag(X_model @ dispersion @ X_model.T)
    
    # Build results DataFrame
    predictions_df = pd.DataFrame({
        'Predicted': y_pred,
        'Leverage': np.round(leverage, 3)
    }, index=X_new.index)
    
    # Confidence intervals (if dof > 0)
    if dof > 0:
        t_critical = stats.t.ppf(0.975, dof)  # 95% CI
        se_pred = s * np.sqrt(leverage)
        ci_lower = y_pred - t_critical * se_pred
        ci_upper = y_pred + t_critical * se_pred
        
        predictions_df['SE_Pred'] = se_pred
        predictions_df['CI_Lower'] = ci_lower
        predictions_df['CI_Upper'] = ci_upper
    
    # Add experimental values and residuals if provided
    if y_new is not None:
        predictions_df['Experimental'] = y_new
        predictions_df['Residuals'] = y_pred - y_new
    
    return predictions_df


def show_predictions_ui(model_results, x_vars, y_var, data):
    """
    Streamlit UI for predictions on new points
    
    Args:
        model_results: dict from fit_mlr_model
        x_vars: list of X variable names
        y_var: Y variable name
        data: current dataset (for reference)
    """
    st.markdown("## 🔮 Predictions for New Points")
    st.markdown("*Equivalent to DOE_prediction.r*")
    
    if model_results is None:
        st.warning("⚠️ No MLR model fitted. Please fit a model first.")
        return
    
    st.info("""
    Predict the response for new experimental conditions using the fitted model.
    
    You can:
    - Enter new experimental conditions manually
    - Upload a file with new conditions
    - Validate central points (if previously excluded from modeling)
    """)
    
    # Variance estimation method
    variance_method = st.radio(
        "Experimental variance from:",
        ["Residuals (from model)", "Independent measurements"],
        key="pred_variance_method"
    )
    
    if variance_method == "Residuals (from model)":
        s = model_results.get('rmse')
        dof = model_results.get('dof')
        
        if s is None or dof is None:
            st.warning("⚠️ Model RMSE or DOF not available")
            s, dof = 1.0, 1
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Std. Deviation", f"{s:.4f}")
        with col2:
            st.metric("DOF", dof)
    else:
        col1, col2 = st.columns(2)
        with col1:
            s = st.number_input("Experimental std. deviation:", value=1.0, 
                               min_value=0.0001, format="%.4f", step=0.1)
        with col2:
            dof = st.number_input("Degrees of freedom:", value=5, min_value=1, step=1)
    
    # Prediction mode selection
    pred_mode = st.radio(
        "Prediction mode:",
        ["Manual input", "Upload file", "Validate central points"],
        key="pred_mode"
    )
    
    # ===== MANUAL INPUT MODE =====
    if pred_mode == "Manual input":
        st.markdown("### Enter New Experimental Conditions")
        
        # Create input fields for each X variable
        new_values = {}
        cols = st.columns(min(3, len(x_vars)))
        
        for i, var in enumerate(x_vars):
            with cols[i % len(cols)]:
                new_values[var] = st.number_input(
                    f"{var}:",
                    value=0.0,
                    step=0.1,
                    format="%.3f",
                    key=f"pred_input_{var}"
                )
        
        # Optional: experimental Y value for comparison
        st.markdown("### Optional: Experimental Response (for validation)")
        has_y = st.checkbox("I have an experimental Y value", key="pred_has_y")
        
        y_exp = None
        if has_y:
            y_exp = st.number_input(f"{y_var} (experimental):", value=0.0, 
                                   step=0.1, format="%.4f", key="pred_y_exp")
        
        if st.button("🚀 Predict Response", type="primary", key="pred_button_manual"):
            try:
                # Create X_new with model matrix structure
                # Need to create full model matrix with interactions and quadratic terms
                X_base = pd.DataFrame([new_values])
                
                # Get the model matrix structure from stored model
                X_stored = model_results['X']
                coef_names = model_results['coefficients'].index.tolist()
                
                # Build model matrix for new point
                from .response_surface import create_prediction_matrix
                X_model = create_prediction_matrix(X_base.values, x_vars, coef_names)
                X_new_df = pd.DataFrame(X_model, columns=coef_names, index=[0])
                
                # Make prediction
                y_new_series = pd.Series([y_exp]) if has_y else None
                predictions = predict_new_points(model_results, X_new_df, y_new_series, s, dof)
                
                st.success("✅ Prediction completed!")
                
                # Display results
                st.markdown("### Prediction Results")
                
                col_res1, col_res2, col_res3 = st.columns(3)
                
                with col_res1:
                    st.metric("Predicted Response", f"{predictions['Predicted'].iloc[0]:.4f}")
                
                with col_res2:
                    st.metric("Leverage", f"{predictions['Leverage'].iloc[0]:.3f}")
                
                with col_res3:
                    if 'CI_Lower' in predictions.columns:
                        ci_width = predictions['CI_Upper'].iloc[0] - predictions['CI_Lower'].iloc[0]
                        st.metric("CI Width (95%)", f"{ci_width:.4f}")
                
                # Confidence interval
                if 'CI_Lower' in predictions.columns:
                    st.info(f"""
                    **95% Confidence Interval:** 
                    [{predictions['CI_Lower'].iloc[0]:.4f}, {predictions['CI_Upper'].iloc[0]:.4f}]
                    """)
                
                # Validation if Y provided
                if has_y:
                    residual = predictions['Residuals'].iloc[0]
                    st.markdown("### Validation")
                    
                    col_val1, col_val2 = st.columns(2)
                    with col_val1:
                        st.metric("Experimental Value", f"{y_exp:.4f}")
                    with col_val2:
                        st.metric("Residual (Pred - Exp)", f"{residual:.4f}")
                    
                    # Check if within CI
                    if 'CI_Lower' in predictions.columns:
                        in_ci = (predictions['CI_Lower'].iloc[0] <= y_exp <= 
                                predictions['CI_Upper'].iloc[0])
                        if in_ci:
                            st.success("✅ Experimental value is within the 95% confidence interval")
                        else:
                            st.warning("⚠️ Experimental value is outside the 95% confidence interval")
                
                # Full results table
                st.markdown("### Detailed Results")
                st.dataframe(predictions.round(4), use_container_width=True)
                
            except Exception as e:
                st.error(f"❌ Error in prediction: {str(e)}")
                import traceback
                with st.expander("🐛 Error details"):
                    st.code(traceback.format_exc())
    
    # ===== FILE UPLOAD MODE =====
    elif pred_mode == "Upload file":
        st.markdown("### Upload File with New Conditions")
        
        st.info("""
        Upload a CSV/Excel file with new experimental conditions.
        
        **Required:** Columns matching your X variables
        **Optional:** Y variable column for validation
        """)
        
        uploaded_file = st.file_uploader(
            "Choose file:",
            type=['csv', 'xlsx', 'txt'],
            key="pred_file_upload"
        )
        
        if uploaded_file is not None:
            try:
                # Read file
                if uploaded_file.name.endswith('.csv'):
                    pred_data = pd.read_csv(uploaded_file)
                elif uploaded_file.name.endswith('.xlsx'):
                    pred_data = pd.read_excel(uploaded_file)
                else:
                    pred_data = pd.read_csv(uploaded_file, sep='\t')
                
                st.success(f"✅ File loaded: {pred_data.shape[0]} rows × {pred_data.shape[1]} columns")
                
                # Show preview
                with st.expander("📋 Data preview"):
                    st.dataframe(pred_data.head(10), use_container_width=True)
                
                # Variable selection
                st.markdown("### Variable Selection")
                
                col_sel1, col_sel2 = st.columns(2)
                
                with col_sel1:
                    # X variables selection
                    x_cols = st.multiselect(
                        "Select X variable columns:",
                        pred_data.columns.tolist(),
                        default=[col for col in x_vars if col in pred_data.columns],
                        key="pred_x_cols"
                    )
                
                with col_sel2:
                    # Optional Y variable
                    has_y_file = st.checkbox("File contains Y variable", key="pred_file_has_y")
                    y_col = None
                    if has_y_file:
                        y_col = st.selectbox(
                            "Select Y variable column:",
                            [col for col in pred_data.columns if col not in x_cols],
                            key="pred_y_col"
                        )
                
                # Row selection
                row_selection = st.radio(
                    "Rows to predict:",
                    ["All rows", "Specific range"],
                    key="pred_row_selection"
                )
                
                if row_selection == "Specific range":
                    col_row1, col_row2 = st.columns(2)
                    with col_row1:
                        start_row = st.number_input("From row:", 1, len(pred_data), 1)
                    with col_row2:
                        end_row = st.number_input("To row:", start_row, len(pred_data), len(pred_data))
                    
                    selected_rows = list(range(start_row-1, end_row))
                else:
                    selected_rows = list(range(len(pred_data)))
                
                if st.button("🚀 Predict for All Points", type="primary", key="pred_button_file"):
                    try:
                        # Extract X data
                        X_new_base = pred_data.loc[selected_rows, x_cols].copy()
                        
                        # Build model matrix
                        coef_names = model_results['coefficients'].index.tolist()
                        from .response_surface import create_prediction_matrix
                        X_model = create_prediction_matrix(X_new_base.values, x_vars, coef_names)
                        X_new_df = pd.DataFrame(X_model, columns=coef_names, 
                                               index=X_new_base.index)
                        
                        # Get Y if available
                        y_new = pred_data.loc[selected_rows, y_col] if has_y_file and y_col else None
                        
                        # Make predictions
                        predictions = predict_new_points(model_results, X_new_df, y_new, s, dof)
                        
                        st.success(f"✅ Predictions completed for {len(predictions)} points!")
                        
                        # Display results
                        st.markdown("### Prediction Results")
                        st.dataframe(predictions.round(4), use_container_width=True)
                        
                        # Statistics
                        col_stat1, col_stat2, col_stat3 = st.columns(3)
                        with col_stat1:
                            st.metric("Mean Predicted", f"{predictions['Predicted'].mean():.4f}")
                        with col_stat2:
                            st.metric("Mean Leverage", f"{predictions['Leverage'].mean():.3f}")
                        with col_stat3:
                            if 'Residuals' in predictions.columns:
                                st.metric("RMSE", f"{np.sqrt((predictions['Residuals']**2).mean()):.4f}")
                        
                        # Plots if Y available
                        if has_y_file and y_col:
                            st.markdown("### Prediction Plots")
                            
                            fig = make_subplots(
                                rows=1, cols=2,
                                subplot_titles=('Experimental vs Predicted', 'Residuals Plot')
                            )
                            
                            # Exp vs Pred
                            fig.add_trace(
                                go.Scatter(
                                    x=predictions['Experimental'],
                                    y=predictions['Predicted'],
                                    mode='markers',
                                    name='Predictions',
                                    marker=dict(size=8, color='red')
                                ),
                                row=1, col=1
                            )
                            
                            # 1:1 line
                            min_val = min(predictions['Experimental'].min(), predictions['Predicted'].min())
                            max_val = max(predictions['Experimental'].max(), predictions['Predicted'].max())
                            fig.add_trace(
                                go.Scatter(
                                    x=[min_val, max_val],
                                    y=[min_val, max_val],
                                    mode='lines',
                                    name='1:1 line',
                                    line=dict(color='green', dash='dash')
                                ),
                                row=1, col=1
                            )
                            
                            # Residuals
                            fig.add_trace(
                                go.Scatter(
                                    x=list(range(1, len(predictions)+1)),
                                    y=predictions['Residuals'],
                                    mode='markers',
                                    name='Residuals',
                                    marker=dict(size=8, color='blue')
                                ),
                                row=1, col=2
                            )
                            
                            fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=2)
                            
                            fig.update_xaxes(title_text="Experimental", row=1, col=1)
                            fig.update_yaxes(title_text="Predicted", row=1, col=1)
                            fig.update_xaxes(title_text="Object Number", row=1, col=2)
                            fig.update_yaxes(title_text="Residuals", row=1, col=2)
                            
                            fig.update_layout(height=500, showlegend=True)
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Export
                        st.markdown("### Export Predictions")
                        csv_data = predictions.to_csv(index=True)
                        st.download_button(
                            "💾 Download Predictions CSV",
                            csv_data,
                            f"{y_var}_predictions.csv",
                            "text/csv"
                        )
                        
                    except Exception as e:
                        st.error(f"❌ Error in batch prediction: {str(e)}")
                        import traceback
                        with st.expander("🐛 Error details"):
                            st.code(traceback.format_exc())
                
            except Exception as e:
                st.error(f"❌ Error loading file: {str(e)}")
    
    # ===== CENTRAL POINTS VALIDATION =====
    else:  # Validate central points
        st.markdown("### Validate Central Points")
        
        if 'mlr_central_points' not in st.session_state:
            st.warning("⚠️ No central points were excluded during model fitting")
            st.info("Central points are only available when you exclude them in the Model Computation tab")
        else:
            central_data = st.session_state.mlr_central_points
            
            st.info(f"""
            **Central points validation**
            
            Central points: {len(central_data['y'])} points
            
            These points were excluded from model fitting and can be used to validate the model at the experimental center.
            """)
            
            # Show central points
            with st.expander("📋 Central Points Data"):
                display_df = pd.DataFrame({
                    'Sample': [str(idx) for idx in central_data['indices']],
                    y_var: central_data['y'].values
                })
                for col in central_data['X'].columns:
                    display_df[col] = central_data['X'][col].values
                st.dataframe(display_df, use_container_width=True)
            
            if st.button("🚀 Predict Central Points", type="primary", key="pred_button_central"):
                try:
                    # Build model matrix for central points
                    X_central = central_data['X']
                    coef_names = model_results['coefficients'].index.tolist()
                    from .response_surface import create_prediction_matrix
                    X_model = create_prediction_matrix(X_central.values, x_vars, coef_names)
                    X_new_df = pd.DataFrame(X_model, columns=coef_names, index=X_central.index)
                    
                    # Predict
                    predictions = predict_new_points(model_results, X_new_df, central_data['y'], s, dof)
                    
                    st.success(f"✅ Central points validated!")
                    
                    # Results
                    st.markdown("### Validation Results")
                    st.dataframe(predictions.round(4), use_container_width=True)
                    
                    # Statistics
                    col_stat1, col_stat2, col_stat3 = st.columns(3)
                    with col_stat1:
                        st.metric("Mean Experimental", f"{central_data['y'].mean():.4f}")
                    with col_stat2:
                        st.metric("Mean Predicted", f"{predictions['Predicted'].mean():.4f}")
                    with col_stat3:
                        rmse_central = np.sqrt((predictions['Residuals']**2).mean())
                        st.metric("RMSE (Central)", f"{rmse_central:.4f}")
                    
                    # Assessment
                    if rmse_central < s * 1.5:
                        st.success("✅ Model validates well at the central point!")
                    else:
                        st.warning("⚠️ Model shows larger errors at the central point - may indicate lack of fit or curvature issues")
                    
                except Exception as e:
                    st.error(f"❌ Error validating central points: {str(e)}")
                    import traceback
                    with st.expander("🐛 Error details"):
                        st.code(traceback.format_exc())
