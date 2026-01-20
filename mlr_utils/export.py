"""
Extract & Export MLR Results
Equivalent to DOE_extract.r
Export various model results to CSV/Excel format
"""

import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO


def extract_dispersion_matrix(model_results):
    """Extract dispersion matrix (X'X)^-1"""
    if 'XtX_inv' not in model_results:
        return None
    
    dispersion = pd.DataFrame(
        model_results['XtX_inv'],
        index=model_results['X'].columns,
        columns=model_results['X'].columns
    )
    return dispersion


def extract_coefficients(model_results):
    """Extract model coefficients"""
    if 'coefficients' not in model_results:
        return None
    
    coef_df = pd.DataFrame({
        'Coefficient': model_results['coefficients']
    })
    
    # Add statistics if available
    if 'se_coef' in model_results and model_results['se_coef'] is not None:
        coef_df['Std_Error'] = model_results['se_coef']
        coef_df['t_statistic'] = model_results['t_stats']
        coef_df['p_value'] = model_results['p_values']
        coef_df['CI_Lower'] = model_results['ci_lower']
        coef_df['CI_Upper'] = model_results['ci_upper']
    
    return coef_df


def extract_fitted_values(model_results):
    """Extract fitted values"""
    if 'y_pred' not in model_results:
        return None
    
    fitted_df = pd.DataFrame({
        'Fitted': model_results['y_pred']
    }, index=range(1, len(model_results['y_pred']) + 1))
    
    return fitted_df


def extract_residuals(model_results):
    """Extract residuals"""
    if 'residuals' not in model_results:
        return None
    
    residuals_df = pd.DataFrame({
        'Residuals': model_results['residuals']
    }, index=range(1, len(model_results['residuals']) + 1))
    
    return residuals_df


def extract_cv_predictions(model_results):
    """Extract cross-validation predictions"""
    if 'cv_predictions' not in model_results:
        return None
    
    cv_pred_df = pd.DataFrame({
        'CV_Predicted': model_results['cv_predictions']
    }, index=range(1, len(model_results['cv_predictions']) + 1))
    
    return cv_pred_df


def extract_cv_residuals(model_results):
    """Extract cross-validation residuals"""
    if 'cv_residuals' not in model_results:
        return None
    
    cv_res_df = pd.DataFrame({
        'CV_Residuals': model_results['cv_residuals']
    }, index=range(1, len(model_results['cv_residuals']) + 1))
    
    return cv_res_df


def extract_leverage(model_results):
    """Extract leverage values"""
    if 'leverage' not in model_results:
        return None
    
    leverage_df = pd.DataFrame({
        'Leverage': model_results['leverage']
    }, index=range(1, len(model_results['leverage']) + 1))
    
    return leverage_df


def extract_vif(model_results):
    """Extract VIF values"""
    if 'vif' not in model_results:
        return None
    
    vif_df = model_results['vif'].to_frame('VIF')
    # Remove intercept and NaN values
    vif_df = vif_df[~vif_df.index.str.contains('Intercept', case=False, na=False)]
    vif_df = vif_df.dropna()
    
    return vif_df


def create_complete_export(model_results, y_var):
    """
    Create complete Excel export with multiple sheets
    
    Args:
        model_results: dict from fit_mlr_model
        y_var: Y variable name
    
    Returns:
        BytesIO buffer with Excel file
    """
    excel_buffer = BytesIO()
    
    with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
        # 1. Model Summary
        summary_data = {
            'Metric': ['Response_Variable', 'N_Samples', 'N_Features', 'DOF', 'R_Squared', 'RMSE', 'Q2', 'RMSECV'],
            'Value': [
                y_var,
                model_results.get('n_samples', 'N/A'),
                model_results.get('n_features', 'N/A'),
                model_results.get('dof', 'N/A'),
                model_results.get('r_squared', 'N/A'),
                model_results.get('rmse', 'N/A'),
                model_results.get('q2', 'N/A'),
                model_results.get('rmsecv', 'N/A')
            ]
        }
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        # 2. Coefficients
        coef_df = extract_coefficients(model_results)
        if coef_df is not None:
            coef_df.to_excel(writer, sheet_name='Coefficients', index=True)
        
        # 3. Fitted Values
        fitted_df = extract_fitted_values(model_results)
        if fitted_df is not None:
            # Combine with experimental values
            if 'y' in model_results:
                fitted_df['Experimental'] = model_results['y'].values
            if 'residuals' in model_results:
                fitted_df['Residuals'] = model_results['residuals']
            fitted_df.to_excel(writer, sheet_name='Fitted_Values', index=True)
        
        # 4. Cross-Validation
        if 'cv_predictions' in model_results:
            cv_df = pd.DataFrame({
                'CV_Predicted': model_results['cv_predictions'],
                'CV_Residuals': model_results['cv_residuals']
            })
            cv_df.to_excel(writer, sheet_name='Cross_Validation', index=True)
        
        # 5. Diagnostics
        diag_df = pd.DataFrame()
        if 'leverage' in model_results:
            diag_df['Leverage'] = model_results['leverage']
        
        if not diag_df.empty:
            diag_df.to_excel(writer, sheet_name='Diagnostics', index=True)
        
        # 6. VIF
        vif_df = extract_vif(model_results)
        if vif_df is not None:
            vif_df.to_excel(writer, sheet_name='VIF', index=True)
        
        # 7. Dispersion Matrix
        dispersion_df = extract_dispersion_matrix(model_results)
        if dispersion_df is not None:
            dispersion_df.to_excel(writer, sheet_name='Dispersion_Matrix', index=True)
    
    excel_buffer.seek(0)
    return excel_buffer


def show_export_ui(model_results, y_var):
    """
    Streamlit UI for extracting and exporting MLR results
    
    Args:
        model_results: dict from fit_mlr_model
        y_var: Y variable name
    """
    st.markdown("## 💾 Extract & Export")
    st.markdown("*Equivalent to DOE_extract.r*")
    
    if model_results is None:
        st.warning("⚠️ No MLR model fitted. Please fit a model first.")
        return
    
    st.info("Extract and export various model results in CSV or Excel format")
    
    # Individual exports
    st.markdown("### 📊 Available Data for Export")
    
    export_options = {}
    
    # Build available export options
    dispersion = extract_dispersion_matrix(model_results)
    if dispersion is not None:
        export_options["📐 Dispersion Matrix"] = dispersion
    
    coefficients = extract_coefficients(model_results)
    if coefficients is not None:
        export_options["📊 Coefficients"] = coefficients
    
    fitted = extract_fitted_values(model_results)
    if fitted is not None:
        export_options["📈 Fitted Values"] = fitted
    
    residuals = extract_residuals(model_results)
    if residuals is not None:
        export_options["📉 Residuals"] = residuals
    
    cv_pred = extract_cv_predictions(model_results)
    if cv_pred is not None:
        export_options["🔄 CV Predictions"] = cv_pred
    
    cv_res = extract_cv_residuals(model_results)
    if cv_res is not None:
        export_options["📊 CV Residuals"] = cv_res
    
    leverage_df = extract_leverage(model_results)
    if leverage_df is not None:
        export_options["🎯 Leverage"] = leverage_df
    
    vif_df = extract_vif(model_results)
    if vif_df is not None:
        export_options["📐 VIF"] = vif_df
    
    # Display each export option
    for name, df in export_options.items():
        with st.expander(f"{name} ({df.shape[0]}×{df.shape[1]})"):
            st.dataframe(df, use_container_width=True)
            
            # CSV download button
            csv_data = df.to_csv(index=True)
            clean_name = name.replace("📐 ", "").replace("📊 ", "").replace("📈 ", "").replace("📉 ", "").replace("🔄 ", "").replace("🎯 ", "").replace(" ", "_")
            
            col_btn1, col_btn2 = st.columns(2)
            
            with col_btn1:
                st.download_button(
                    f"💾 Download {name} as CSV",
                    csv_data,
                    f"MLR_{clean_name}.csv",
                    "text/csv",
                    key=f"download_{clean_name}"
                )
            
            with col_btn2:
                # Statistics about the data
                if df.select_dtypes(include=[np.number]).shape[1] > 0:
                    numeric_cols = df.select_dtypes(include=[np.number])
                    st.caption(f"Mean: {numeric_cols.mean().mean():.4f}")
    
    # Complete Excel export
    st.markdown("---")
    st.markdown("### 📦 Complete MLR Analysis Export")
    
    st.info("""
    Export all model results in a single Excel file with multiple sheets:
    - Summary (R², RMSE, Q², etc.)
    - Coefficients with statistics
    - Fitted values and residuals
    - Cross-validation results
    - Diagnostics (leverage, VIF)
    - Dispersion matrix
    """)
    
    col_export1, col_export2 = st.columns([2, 1])
    
    with col_export1:
        if st.button("📦 Export Complete MLR Analysis", type="primary"):
            try:
                with st.spinner("Creating Excel file..."):
                    excel_buffer = create_complete_export(model_results, y_var)
                
                st.success("✅ Complete MLR analysis ready for download!")
                
                st.download_button(
                    "📄 Download Complete Analysis (XLSX)",
                    excel_buffer.getvalue(),
                    f"Complete_MLR_Analysis_{y_var}.xlsx",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="download_complete_excel"
                )
                
            except Exception as e:
                st.error(f"❌ Excel export failed: {str(e)}")
                st.info("Individual CSV exports are still available above")
                import traceback
                with st.expander("🐛 Error details"):
                    st.code(traceback.format_exc())
    
    with col_export2:
        # Export statistics
        st.markdown("**Export includes:**")
        n_sheets = 1  # Summary always included
        if coefficients is not None:
            n_sheets += 1
        if fitted is not None:
            n_sheets += 1
        if cv_pred is not None:
            n_sheets += 1
        if leverage_df is not None or vif_df is not None:
            n_sheets += 1
        if dispersion is not None:
            n_sheets += 1
        
        st.metric("Excel Sheets", n_sheets)
        st.metric("Total Variables", len(export_options))
