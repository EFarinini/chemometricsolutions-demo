"""
Workspace Utilities Module

Reusable utilities for working with datasets in the workspace.
Provides consistent dataset selection across all analysis modules.

Author: ChemometricSolutions
"""

import streamlit as st
import pandas as pd
from typing import Dict, Optional, Tuple


def get_workspace_datasets() -> Dict[str, pd.DataFrame]:
    """
    Get all available datasets from the workspace.

    Collects datasets from:
    - Current dataset (st.session_state.current_data)
    - Transformation history (st.session_state.transformation_history)
    - Split datasets (st.session_state.split_datasets)

    Returns
    -------
    dict
        Dictionary mapping dataset names to DataFrames
    """
    available_datasets = {}

    # Add current dataset
    if 'current_data' in st.session_state and st.session_state.current_data is not None:
        dataset_name = st.session_state.get('current_dataset', 'Current Dataset')
        available_datasets[dataset_name] = st.session_state.current_data

    # Add transformation history
    if 'transformation_history' in st.session_state:
        for name, info in st.session_state.transformation_history.items():
            if 'data' in info and info['data'] is not None:
                # Don't duplicate if already in current
                if name not in available_datasets:
                    available_datasets[name] = info['data']

    # Add split datasets
    if 'split_datasets' in st.session_state:
        for name, info in st.session_state.split_datasets.items():
            if 'data' in info and info['data'] is not None:
                # Prefix with "Split: " to distinguish
                split_name = f"Split: {name}"
                available_datasets[split_name] = info['data']

    return available_datasets


def display_workspace_dataset_selector(
    label: str = "Select dataset:",
    key: str = "workspace_dataset_selector",
    help_text: Optional[str] = None,
    show_info: bool = True
) -> Optional[Tuple[str, pd.DataFrame]]:
    """
    Display a dataset selector from workspace with consistent UI.

    Parameters
    ----------
    label : str
        Label for the selectbox
    key : str
        Unique key for the selectbox widget
    help_text : str, optional
        Help text for the selectbox
    show_info : bool
        Whether to show dataset info after selection

    Returns
    -------
    tuple or None
        (dataset_name, dataframe) if dataset selected, None otherwise
    """
    available_datasets = get_workspace_datasets()

    if len(available_datasets) == 0:
        st.warning("⚠️ **No datasets available in workspace.**")
        st.info("💡 Load data in the **Data Handling** page first")
        return None

    # Create selectbox
    if help_text is None:
        help_text = "Choose a dataset from your workspace"

    selected_name = st.selectbox(
        label,
        options=list(available_datasets.keys()),
        key=key,
        help=help_text
    )

    if selected_name:
        selected_data = available_datasets[selected_name].copy()

        if show_info:
            # Display dataset info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Samples", selected_data.shape[0])
            with col2:
                st.metric("Variables", selected_data.shape[1])
            with col3:
                numeric_cols = selected_data.select_dtypes(include=['number']).columns
                st.metric("Numeric", len(numeric_cols))

        return selected_name, selected_data

    return None


def display_workspace_summary():
    """
    Display a summary of all datasets in the workspace.
    Useful for sidebar or workspace overview sections.
    """
    available_datasets = get_workspace_datasets()

    if len(available_datasets) == 0:
        st.info("📊 Workspace is empty - load data to get started")
        return

    st.markdown(f"### 📊 Workspace ({len(available_datasets)} datasets)")

    for name, data in available_datasets.items():
        with st.expander(f"📁 {name}", expanded=False):
            st.write(f"**Shape**: {data.shape[0]} samples × {data.shape[1]} variables")

            numeric_cols = data.select_dtypes(include=['number']).columns
            non_numeric_cols = data.select_dtypes(exclude=['number']).columns

            st.write(f"**Numeric columns**: {len(numeric_cols)}")
            st.write(f"**Non-numeric columns**: {len(non_numeric_cols)}")

            # Check if it's from transformation history
            if 'transformation_history' in st.session_state:
                if name in st.session_state.transformation_history:
                    transform_info = st.session_state.transformation_history[name]
                    st.caption(f"📝 Transform: {transform_info.get('transform', 'Unknown')}")
                    if 'timestamp' in transform_info:
                        st.caption(f"🕐 {transform_info['timestamp'].strftime('%Y-%m-%d %H:%M')}")


def activate_dataset_in_workspace(dataset_name: str, dataset: pd.DataFrame):
    """
    Activate a dataset in the workspace (make it the current dataset).

    This updates st.session_state.current_data and st.session_state.current_dataset
    so the dataset becomes active across all modules.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset
    dataset : pd.DataFrame
        The dataset to activate
    """
    st.session_state.current_data = dataset.copy()
    st.session_state.current_dataset = dataset_name
    st.success(f"✅ **{dataset_name}** is now active in the workspace")


def get_dataset_metadata(dataset_name: str) -> Optional[Dict]:
    """
    Get metadata about a dataset if available.

    Returns transformation info, split info, or None if not found.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset

    Returns
    -------
    dict or None
        Metadata dictionary or None
    """
    # Check transformation history
    if 'transformation_history' in st.session_state:
        if dataset_name in st.session_state.transformation_history:
            return st.session_state.transformation_history[dataset_name]

    # Check split datasets (remove "Split: " prefix if present)
    if dataset_name.startswith("Split: "):
        split_name = dataset_name.replace("Split: ", "")
        if 'split_datasets' in st.session_state:
            if split_name in st.session_state.split_datasets:
                return st.session_state.split_datasets[split_name]

    return None
