"""
Univariate Analysis Workspace Management

Handle saving, loading, and exporting univariate analysis results.
"""

import pandas as pd
import json
from datetime import datetime
from typing import Dict, Any, Optional
from io import BytesIO
import streamlit as st


def save_univariate_results(
    results: Dict[str, Any],
    result_name: str,
    category: str = "univariate_analysis"
) -> bool:
    """
    Save univariate analysis results to Streamlit session state.

    Parameters
    ----------
    results : dict
        Dictionary containing analysis results
    result_name : str
        Name for this result set
    category : str
        Category for organization

    Returns
    -------
    bool
        Success flag
    """
    try:
        if 'univariate_results' not in st.session_state:
            st.session_state.univariate_results = {}

        st.session_state.univariate_results[result_name] = {
            'data': results,
            'timestamp': datetime.now().isoformat(),
            'category': category
        }
        return True
    except Exception as e:
        st.error(f"Error saving results: {e}")
        return False


def load_univariate_results(result_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Load previously saved univariate results.

    Parameters
    ----------
    result_name : str, optional
        Name of result set to load. If None, returns all results.

    Returns
    -------
    dict or None
        Results if found, None otherwise
    """
    if 'univariate_results' not in st.session_state:
        return None

    # If no name provided, return all results
    if result_name is None:
        return st.session_state.univariate_results

    # Return specific result
    if result_name not in st.session_state.univariate_results:
        return None

    return st.session_state.univariate_results[result_name]['data']


def export_statistics_to_csv(
    statistics_dict: Dict[str, Dict[str, float]],
    column_names: Optional[list] = None,
    filepath: Optional[str] = None
) -> pd.DataFrame:
    """
    Export statistical results to CSV format.

    Parameters
    ----------
    statistics_dict : dict
        Dictionary of statistics for each column
    column_names : list, optional
        List of column names. If None, uses all keys from statistics_dict.
    filepath : str, optional
        Output filepath. If None, returns DataFrame without writing to file.

    Returns
    -------
    pd.DataFrame
        Exported dataframe
    """
    if column_names is None:
        column_names = list(statistics_dict.keys())

    export_data = []

    for col_name in column_names:
        if col_name in statistics_dict:
            stats = statistics_dict[col_name]
            row = {'Column': col_name}
            row.update(stats)
            export_data.append(row)

    df_export = pd.DataFrame(export_data)

    if filepath is not None:
        df_export.to_csv(filepath, index=False)

    return df_export


def export_statistics_to_excel(
    statistics_dict: Dict[str, pd.DataFrame],
    include_metadata: bool = True
) -> BytesIO:
    """
    Export multiple statistics tables to Excel with multiple sheets.

    Parameters
    ----------
    statistics_dict : dict
        Dictionary of dataframes to export (sheet_name: dataframe)
    include_metadata : bool, default=True
        Include metadata sheet with analysis information

    Returns
    -------
    BytesIO
        In-memory Excel file buffer
    """
    buffer = BytesIO()

    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        # Write each dataframe to a separate sheet
        for sheet_name, df in statistics_dict.items():
            # Ensure sheet name is valid (max 31 chars, no special chars)
            safe_sheet_name = sheet_name[:31].replace('[', '').replace(']', '').replace(':', '')
            df.to_excel(writer, sheet_name=safe_sheet_name, index=False)

        # Add metadata sheet if requested
        if include_metadata:
            metadata_df = pd.DataFrame({
                'Property': [
                    'Analysis Type',
                    'Generated Date',
                    'Number of Sheets',
                    'Software',
                    'Version'
                ],
                'Value': [
                    'Univariate Statistics',
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    len(statistics_dict),
                    'ChemometricSolutions',
                    '1.0.0'
                ]
            })
            metadata_df.to_excel(writer, sheet_name='Metadata', index=False)

    buffer.seek(0)
    return buffer


def get_workspace_datasets(session_state: Optional[dict] = None) -> Dict[str, pd.DataFrame]:
    """
    Get available datasets from workspace session state.

    Parameters
    ----------
    session_state : dict, optional
        Streamlit session state. If None, uses st.session_state

    Returns
    -------
    dict
        Dictionary of available datasets (name: dataframe)
    """
    if session_state is None:
        session_state = st.session_state

    datasets = {}

    # Check for uploaded data
    if 'uploaded_data' in session_state and session_state['uploaded_data'] is not None:
        datasets['Uploaded Data'] = session_state['uploaded_data']

    # Check for PCA results
    if 'pca_results' in session_state:
        pca_dict = session_state['pca_results']
        if 'original_data' in pca_dict:
            datasets['PCA Original Data'] = pca_dict['original_data']
        if 'scores' in pca_dict:
            scores_df = pd.DataFrame(
                pca_dict['scores'],
                columns=[f"PC{i+1}" for i in range(pca_dict['scores'].shape[1])]
            )
            datasets['PCA Scores'] = scores_df

    # Check for transformed data
    if 'transformed_data' in session_state and session_state['transformed_data'] is not None:
        datasets['Transformed Data'] = session_state['transformed_data']

    # Check for calibration data
    if 'calibration_data' in session_state:
        cal_dict = session_state['calibration_data']
        if 'X_train' in cal_dict:
            datasets['Calibration X (Training)'] = cal_dict['X_train']
        if 'X_test' in cal_dict:
            datasets['Calibration X (Test)'] = cal_dict['X_test']

    return datasets


def format_statistics_for_display(stats_dict: Dict[str, Any]) -> pd.DataFrame:
    """
    Format statistics dictionary for nice display in Streamlit.

    Parameters
    ----------
    stats_dict : dict
        Statistics dictionary from calculation functions

    Returns
    -------
    pd.DataFrame
        Formatted dataframe with categories and formatting
    """
    # Define statistic categories and their display names
    stat_display_names = {
        # Sample size
        'n': ('Sample Size', 'Sample Size'),
        'n_na': ('Sample Size', 'Missing Values'),

        # Descriptive statistics
        'mean_arithmetic': ('Descriptive', 'Mean (Arithmetic)'),
        'mean_geometric': ('Descriptive', 'Mean (Geometric)'),
        'median': ('Descriptive', 'Median'),

        # Dispersion
        'std_dev': ('Dispersion', 'Standard Deviation'),
        'variance': ('Dispersion', 'Variance'),
        'rsd': ('Dispersion', 'RSD (%)'),
        'min': ('Dispersion', 'Minimum'),
        'max': ('Dispersion', 'Maximum'),
        'range': ('Dispersion', 'Range'),

        # Robust statistics
        'iqr': ('Robust', 'IQR'),
        'mad': ('Robust', 'MAD'),
        'robust_cv': ('Robust', 'Robust CV (%)'),
        'q1': ('Robust', '1st Quartile (Q1)'),
        'q3': ('Robust', '3rd Quartile (Q3)')
    }

    # Build formatted dataframe
    rows = []
    for key, value in stats_dict.items():
        if key in stat_display_names:
            category, display_name = stat_display_names[key]
            rows.append({
                'Category': category,
                'Statistic': display_name,
                'Value': value
            })

    df = pd.DataFrame(rows)
    return df


def get_all_saved_results() -> Dict[str, Dict[str, Any]]:
    """
    Get all saved univariate results from session state.

    Returns
    -------
    dict
        Dictionary of all saved results
    """
    if 'univariate_results' not in st.session_state:
        return {}

    return st.session_state.univariate_results


def clear_univariate_results(result_name: Optional[str] = None) -> bool:
    """
    Clear saved univariate results.

    Parameters
    ----------
    result_name : str, optional
        Name of specific result to clear. If None, clears all results.

    Returns
    -------
    bool
        Success flag
    """
    try:
        if 'univariate_results' not in st.session_state:
            return True

        if result_name is None:
            # Clear all results
            st.session_state.univariate_results = {}
        else:
            # Clear specific result
            if result_name in st.session_state.univariate_results:
                del st.session_state.univariate_results[result_name]

        return True
    except Exception as e:
        st.error(f"Error clearing results: {e}")
        return False
