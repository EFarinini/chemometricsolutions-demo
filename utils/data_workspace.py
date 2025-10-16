"""
Workspace management functions
Handles transformation history and dataset management
"""

import pandas as pd
import streamlit as st


def save_original_to_history(data, dataset_name):
    """
    Save original dataset to transformation history for reference

    Parameters:
    -----------
    data : pd.DataFrame
        Original dataset to save
    dataset_name : str
        Name of the dataset

    Returns:
    --------
    None (modifies st.session_state)
    """
    if 'transformation_history' not in st.session_state:
        st.session_state.transformation_history = {}

    # Save original only if not exists
    original_name = f"{dataset_name.split('.')[0]}_ORIGINAL"

    if original_name not in st.session_state.transformation_history:
        st.session_state.transformation_history[original_name] = {
            'data': data.copy(),
            'transform': 'Original (Untransformed)',
            'params': {},
            'col_range': None,
            'timestamp': pd.Timestamp.now(),
            'transform_type': 'original'
        }
