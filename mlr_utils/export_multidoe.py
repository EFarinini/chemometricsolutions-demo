"""
Multi-DOE Export Module
Export all models to structured Excel or CSV files

This module provides export functionality for Multi-DOE analysis,
including comprehensive Excel export with multiple sheets.
"""

import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import datetime

# Import summary function
from .model_computation_multidoe import (
    statistical_summary_multidoe,
    extract_coefficients_comparison
)


def create_multidoe_excel_export(models_dict, x_vars, y_vars):
    """
    Create comprehensive Excel export with multiple sheets

    Sheets:
    - Summary: Overall model comparison
    - Coefficients_Comparison: All coefficients side-by-side
    - Model_Y1, Model_Y2, ...: Per-model details
    - Fitted_Y1, Residuals_Y1, ...: Per-model results

    Args:
        models_dict (dict): Dictionary {y_name: model_result}
        x_vars (list): List of X variable names
        y_vars (list): List of Y variable names

    Returns:
        BytesIO: Excel file buffer
    """
    excel_buffer = BytesIO()

    with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
        # ====================================================================
        # SHEET 1: SUMMARY (All models comparison)
        # ====================================================================
        summary_df = statistical_summary_multidoe(models_dict)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)

        # ====================================================================
        # SHEET 2: COEFFICIENTS COMPARISON (side-by-side)
        # ====================================================================
        coef_comparison = extract_coefficients_comparison(models_dict)
        coef_comparison.to_excel(writer, sheet_name='Coefficients_Comparison')

        # ====================================================================
        # SHEETS 3+: Per-model details
        # ====================================================================
        for y_name, model in models_dict.items():
            if 'error' in model:
                # Create error sheet
                error_df = pd.DataFrame({
                    'Status': ['Error'],
                    'Message': [model['error']]
                })
                error_df.to_excel(writer, sheet_name=f'Error_{y_name[:20]}', index=False)
                continue

            # Model summary sheet
            model_summary = pd.DataFrame({
                'Metric': ['R²', 'RMSE', 'Q²', 'RMSECV', 'DOF', 'N_samples', 'N_coefficients'],
                'Value': [
                    model.get('r_squared', np.nan),
                    model.get('rmse', np.nan),
                    model.get('q2', np.nan),
                    model.get('rmsecv', np.nan),
                    model.get('dof', np.nan),
                    len(model.get('y', [])),
                    len(model.get('coefficients', []))
                ]
            })
            # Truncate sheet name to 31 chars (Excel limit)
            sheet_name = f'Summary_{y_name}'[:31]
            model_summary.to_excel(writer, sheet_name=sheet_name, index=False)

            # Coefficients sheet
            if 'coefficients' in model:
                coef_df = pd.DataFrame({
                    'Term': model['coefficients'].index,
                    'Coefficient': model['coefficients'].values
                })
                sheet_name = f'Coef_{y_name}'[:31]
                coef_df.to_excel(writer, sheet_name=sheet_name, index=False)

            # Fitted values and residuals sheet
            if 'y_pred' in model and 'residuals' in model:
                fitted_df = pd.DataFrame({
                    'Observed': model.get('y', []),
                    'Fitted': model.get('y_pred', []),
                    'Residuals': model.get('residuals', [])
                })
                sheet_name = f'Fitted_{y_name}'[:31]
                fitted_df.to_excel(writer, sheet_name=sheet_name, index=False)

        # ====================================================================
        # METADATA SHEET
        # ====================================================================
        metadata = pd.DataFrame({
            'Property': [
                'Export Date',
                'Number of Models',
                'X Variables',
                'Y Variables',
                'X Variable Names',
                'Y Variable Names'
            ],
            'Value': [
                datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                len(models_dict),
                len(x_vars),
                len(y_vars),
                ', '.join(x_vars),
                ', '.join(y_vars)
            ]
        })
        metadata.to_excel(writer, sheet_name='Metadata', index=False)

    excel_buffer.seek(0)
    return excel_buffer


def show_export_ui_multidoe(models_dict, x_vars, y_vars):
    """
    Export UI for multi-DOE models

    Args:
        models_dict (dict): Dictionary {y_name: model_result}
        x_vars (list): List of X variable names
        y_vars (list): List of Y variable names
    """
    st.markdown("## 📦 Extract & Export")
    st.info("Export all models and results to Excel or CSV format")

    # ========================================================================
    # SECTION 1: Export Mode Selection
    # ========================================================================
    export_mode = st.radio(
        "Export scope:",
        ["All Models (Single Excel)", "Individual Model", "Comparison Only"],
        key="multidoe_export_mode"
    )

    # ========================================================================
    # SECTION 2: All Models Export
    # ========================================================================
    if export_mode == "All Models (Single Excel)":
        st.markdown("### Complete Multi-DOE Analysis Export")
        st.info("""
        Export all models to a single Excel file with multiple sheets:
        - Summary: Model comparison table
        - Coefficients_Comparison: All coefficients side-by-side
        - Individual sheets for each model (summary, coefficients, fitted values)
        - Metadata sheet with export information
        """)

        if st.button("📦 Create Excel Export", type="primary", key="multidoe_export_excel"):
            try:
                with st.spinner("Creating Excel file..."):
                    excel_buffer = create_multidoe_excel_export(models_dict, x_vars, y_vars)

                # Generate filename
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"Multi_DOE_Analysis_{timestamp}.xlsx"

                st.download_button(
                    label="📥 Download Complete Analysis (XLSX)",
                    data=excel_buffer.getvalue(),
                    file_name=filename,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="multidoe_download_excel"
                )

                st.success("✅ Excel file ready for download")

            except Exception as e:
                st.error(f"❌ Export failed: {str(e)}")
                import traceback
                with st.expander("🐛 Error details"):
                    st.code(traceback.format_exc())

    # ========================================================================
    # SECTION 3: Individual Model Export
    # ========================================================================
    elif export_mode == "Individual Model":
        st.markdown("### Export Single Model")

        selected_y = st.selectbox(
            "Select response to export:",
            y_vars,
            key="multidoe_export_selected_y"
        )

        if st.button("📦 Export Selected Model", type="primary", key="multidoe_export_individual"):
            try:
                model = models_dict[selected_y]

                if 'error' in model:
                    st.error(f"❌ Cannot export {selected_y}: {model['error']}")
                    return

                # Create export DataFrame
                export_df = pd.DataFrame({
                    'Metric': ['R²', 'RMSE', 'Q²', 'RMSECV', 'DOF'],
                    'Value': [
                        model.get('r_squared', np.nan),
                        model.get('rmse', np.nan),
                        model.get('q2', np.nan),
                        model.get('rmsecv', np.nan),
                        model.get('dof', np.nan)
                    ]
                })

                csv_data = export_df.to_csv(index=False)

                st.download_button(
                    label=f"📥 Download {selected_y} Model (CSV)",
                    data=csv_data,
                    file_name=f"Model_{selected_y}.csv",
                    mime="text/csv",
                    key="multidoe_download_individual_csv"
                )

                st.success(f"✅ Model {selected_y} ready for download")

            except Exception as e:
                st.error(f"❌ Export failed: {str(e)}")

    # ========================================================================
    # SECTION 4: Comparison Only Export
    # ========================================================================
    elif export_mode == "Comparison Only":
        st.markdown("### 📊 Models Comparison Table")

        summary_df = statistical_summary_multidoe(models_dict)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

        # Export comparison as CSV
        csv_data = summary_df.to_csv(index=False)

        st.download_button(
            label="📥 Download Comparison (CSV)",
            data=csv_data,
            file_name="Multi_DOE_Comparison.csv",
            mime="text/csv",
            key="multidoe_download_comparison_csv"
        )

        # Also show coefficients comparison
        st.markdown("---")
        st.markdown("### Coefficients Comparison")

        coef_comparison = extract_coefficients_comparison(models_dict)
        st.dataframe(coef_comparison, use_container_width=True)

        # Export coefficients comparison
        csv_coef = coef_comparison.to_csv()

        st.download_button(
            label="📥 Download Coefficients Comparison (CSV)",
            data=csv_coef,
            file_name="Multi_DOE_Coefficients.csv",
            mime="text/csv",
            key="multidoe_download_coef_csv"
        )
