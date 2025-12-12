"""
Mixture Design Export Module
Export results to CSV, Excel, and other formats
"""

import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import openpyxl
from datetime import datetime


def extract_mixture_coefficients(model_results):
    """
    Extract coefficients as DataFrame

    Returns:
        pd.DataFrame with coefficient statistics
    """
    coef_df = pd.DataFrame({
        'Term': model_results['feature_names'],
        'Coefficient': model_results['coefficients'],
        'Std_Error': model_results.get('se_coef', [np.nan] * len(model_results['coefficients'])),
        't_statistic': model_results.get('t_stats', [np.nan] * len(model_results['coefficients'])),
        'p_value': model_results.get('p_values', [np.nan] * len(model_results['coefficients'])),
        'CI_95_Lower': model_results.get('ci_lower', [np.nan] * len(model_results['coefficients'])),
        'CI_95_Upper': model_results.get('ci_upper', [np.nan] * len(model_results['coefficients']))
    })

    return coef_df


def extract_mixture_predictions(model_results):
    """
    Extract fitted values and residuals

    Returns:
        pd.DataFrame with predictions
    """
    pred_df = pd.DataFrame({
        'Experiment_Number': range(1, len(model_results['y']) + 1),
        'Observed_Y': model_results['y'],
        'Predicted_Y': model_results['y_pred'],
        'Residuals': model_results['residuals'],
        'Leverage': model_results['leverage']
    })

    return pred_df


def show_mixture_export_ui(model_results, mixture_design_matrix, data, y_var):
    """
    Display export UI with download buttons

    Args:
        model_results: fitted mixture model
        mixture_design_matrix: design matrix
        data: original dataset
        y_var: response variable name
    """
    st.markdown("## 📤 Extract & Export Results")

    st.info("Download model results in various formats")

    # Section 1: Coefficients
    with st.expander("📋 Coefficients", expanded=True):
        coef_df = extract_mixture_coefficients(model_results)

        st.dataframe(
            coef_df.style.format({
                'Coefficient': '{:.6f}',
                'Std_Error': '{:.6f}',
                't_statistic': '{:.4f}',
                'p_value': '{:.4f}',
                'CI_95_Lower': '{:.6f}',
                'CI_95_Upper': '{:.6f}'
            }),
            use_container_width=True
        )

        csv_coef = coef_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Download Coefficients CSV",
            csv_coef,
            f"mixture_coefficients_{y_var}.csv",
            "text/csv",
            key="download_coef"
        )

    # Section 2: Fitted Values
    with st.expander("📊 Fitted Values & Residuals"):
        pred_df = extract_mixture_predictions(model_results)

        st.dataframe(
            pred_df.style.format({
                'Observed_Y': '{:.6f}',
                'Predicted_Y': '{:.6f}',
                'Residuals': '{:.6f}',
                'Leverage': '{:.4f}'
            }),
            use_container_width=True
        )

        csv_pred = pred_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Download Predictions CSV",
            csv_pred,
            f"mixture_predictions_{y_var}.csv",
            "text/csv",
            key="download_pred"
        )

    # Section 3: Design Matrix
    with st.expander("🧪 Design Matrix"):
        if mixture_design_matrix is not None:
            st.dataframe(mixture_design_matrix, use_container_width=True)

            csv_design = mixture_design_matrix.to_csv(index=True).encode('utf-8')
            st.download_button(
                "📥 Download Design Matrix CSV",
                csv_design,
                "mixture_design_matrix.csv",
                "text/csv",
                key="download_design"
            )
        else:
            st.warning("No design matrix available")

    # Section 4: Complete Export
    st.markdown("---")
    st.markdown("### 📦 Complete Analysis Package")

    if st.button("Generate Complete Excel Report", type="primary"):
        with st.spinner("Generating Excel report..."):
            try:
                # Create Excel file
                output = BytesIO()

                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    # Sheet 1: Summary
                    summary_data = {
                        'Parameter': [
                            'Model Type',
                            'Degree',
                            'Response Variable',
                            'N Samples',
                            'N Components',
                            'N Parameters',
                            'R²',
                            'Q²',
                            'RMSE',
                            'RMSECV',
                            'Export Date'
                        ],
                        'Value': [
                            model_results['model_type'],
                            model_results['degree'],
                            y_var,
                            model_results['n_samples'],
                            model_results['n_components'],
                            model_results['n_features'],
                            f"{model_results['r_squared']:.6f}",
                            f"{model_results.get('q2', np.nan):.6f}",
                            f"{model_results['rmse']:.6f}",
                            f"{model_results.get('rmsecv', np.nan):.6f}",
                            datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        ]
                    }
                    pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)

                    # Sheet 2: Coefficients
                    coef_df.to_excel(writer, sheet_name='Coefficients', index=False)

                    # Sheet 3: Fitted Values
                    pred_df.to_excel(writer, sheet_name='Fitted_Values', index=False)

                    # Sheet 4: Design Matrix
                    if mixture_design_matrix is not None:
                        mixture_design_matrix.to_excel(writer, sheet_name='Design_Matrix')

                output.seek(0)

                st.download_button(
                    "📥 Download Excel Report",
                    output,
                    f"mixture_analysis_{y_var}_{datetime.now().strftime('%Y%m%d')}.xlsx",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="download_excel"
                )

                st.success("✅ Excel report generated successfully!")

            except Exception as e:
                st.error(f"❌ Export failed: {str(e)}")
