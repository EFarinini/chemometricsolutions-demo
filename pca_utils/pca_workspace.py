"""
PCA Workspace Management Functions

Functions for managing PCA analysis workspaces, dataset splitting,
and saving/loading analysis results.
"""

import pandas as pd
import numpy as np
import streamlit as st
import json
from datetime import datetime
from pathlib import Path
from io import StringIO
from typing import Dict, Any, Optional, Tuple


def save_workspace_to_file(filepath: Optional[str] = None) -> bool:
    """
    Save PCA workspace (split datasets) to a JSON file.

    Exports all datasets in session_state.split_datasets to a JSON file,
    preserving metadata and data structure for later restoration.

    Parameters
    ----------
    filepath : str, optional
        Path to save the workspace file. If None, saves to 'pca_workspace.json'
        in the current directory. Default is None.

    Returns
    -------
    bool
        True if workspace was successfully saved, False if no split_datasets exist
        or if an error occurred.

    Examples
    --------
    >>> # Save workspace to default location
    >>> success = save_workspace_to_file()
    >>> if success:
    ...     print("Workspace saved successfully")

    >>> # Save to custom location
    >>> save_workspace_to_file("my_analysis_workspace.json")

    Notes
    -----
    The saved JSON structure contains:
    - Dataset names as keys
    - For each dataset:
        - 'type': Dataset type (e.g., 'PCA_Selection', 'PCA_Remaining')
        - 'parent': Name of parent dataset
        - 'n_samples': Number of samples in dataset
        - 'creation_time': ISO format timestamp
        - 'csv_data': DataFrame serialized as CSV string

    The function accesses Streamlit session state to retrieve split datasets.
    """
    if 'split_datasets' not in st.session_state:
        return False

    if not st.session_state.split_datasets:
        return False

    try:
        workspace_data = {}
        for name, info in st.session_state.split_datasets.items():
            # Save metadata and CSV data
            workspace_data[name] = {
                'type': info['type'],
                'parent': info['parent'],
                'n_samples': info['n_samples'],
                'creation_time': info['creation_time'].isoformat(),
                'csv_data': info['data'].to_csv(index=True)
            }

        # Save to file
        workspace_file = Path(filepath) if filepath else Path("pca_workspace.json")
        with open(workspace_file, 'w') as f:
            json.dump(workspace_data, f, indent=2)

        return True
    except Exception:
        return False


def load_workspace_from_file(filepath: Optional[str] = None) -> bool:
    """
    Load PCA workspace (split datasets) from a JSON file.

    Restores previously saved split datasets into session_state.split_datasets,
    reconstructing DataFrames and metadata.

    Parameters
    ----------
    filepath : str, optional
        Path to the workspace file to load. If None, loads from 'pca_workspace.json'
        in the current directory. Default is None.

    Returns
    -------
    bool
        True if workspace was successfully loaded, False if file doesn't exist
        or if an error occurred.

    Examples
    --------
    >>> # Load workspace from default location
    >>> if load_workspace_from_file():
    ...     print(f"Loaded {len(st.session_state.split_datasets)} datasets")

    >>> # Load from custom location
    >>> load_workspace_from_file("my_analysis_workspace.json")

    Notes
    -----
    - Overwrites existing split_datasets in session state
    - Reconstructs pandas DataFrames from CSV strings
    - Converts ISO timestamp strings back to pandas Timestamp objects
    - Displays error message via Streamlit if loading fails

    The function modifies Streamlit session state directly.
    """
    workspace_file = Path(filepath) if filepath else Path("pca_workspace.json")

    if not workspace_file.exists():
        return False

    try:
        with open(workspace_file, 'r') as f:
            workspace_data = json.load(f)

        # Reconstruct datasets
        st.session_state.split_datasets = {}
        for name, info in workspace_data.items():
            # Reconstruct DataFrame from CSV
            csv_data = StringIO(info['csv_data'])
            df = pd.read_csv(csv_data, index_col=0)

            st.session_state.split_datasets[name] = {
                'data': df,
                'type': info['type'],
                'parent': info['parent'],
                'n_samples': info['n_samples'],
                'creation_time': pd.Timestamp.fromisoformat(info['creation_time'])
            }

        return True
    except Exception as e:
        st.error(f"Error loading workspace: {str(e)}")
        return False


def save_dataset_split(
    selected_data: pd.DataFrame,
    remaining_data: pd.DataFrame,
    pc_x: str,
    pc_y: str,
    parent_name: Optional[str] = None,
    selection_method: str = 'Manual'
) -> Tuple[str, str]:
    """
    Save dataset split (selected and remaining samples) to workspace.

    Stores both selected and remaining datasets from a PCA sample selection
    operation into session_state.split_datasets with appropriate metadata.

    Parameters
    ----------
    selected_data : pd.DataFrame
        DataFrame containing the selected samples (rows).
    remaining_data : pd.DataFrame
        DataFrame containing the remaining (non-selected) samples.
    pc_x : str
        Name of the PC on x-axis (e.g., 'PC1') used for selection.
    pc_y : str
        Name of the PC on y-axis (e.g., 'PC2') used for selection.
    parent_name : str, optional
        Name of the parent dataset. If None, retrieves from
        st.session_state.current_dataset or uses 'Dataset'. Default is None.
    selection_method : str, optional
        Method used for selection ('Manual', 'Coordinate', 'Lasso', etc.).
        Default is 'Manual'.

    Returns
    -------
    selected_name : str
        Name assigned to the selected dataset in workspace.
    remaining_name : str
        Name assigned to the remaining dataset in workspace.

    Examples
    --------
    >>> selected_df = scores_df.iloc[[1, 5, 10]]
    >>> remaining_df = scores_df.drop([1, 5, 10])
    >>> sel_name, rem_name = save_dataset_split(
    ...     selected_df, remaining_df, 'PC1', 'PC2',
    ...     parent_name='NIR_Data', selection_method='Lasso'
    ... )
    >>> print(f"Saved: {sel_name} and {rem_name}")

    Notes
    -----
    Dataset names are created with format: {parent}_Selected_{timestamp} and
    {parent}_Remaining_{timestamp} where timestamp is HHMMSS format.

    Each dataset entry contains:
    - 'data': DataFrame with samples
    - 'type': 'PCA_Selection' or 'PCA_Remaining'
    - 'parent': Name of parent dataset
    - 'n_samples': Number of samples
    - 'creation_time': Timestamp object
    - 'selection_method': Method used for selection
    - 'pc_axes': String describing PC axes used (e.g., 'PC1 vs PC2')

    The function modifies st.session_state.split_datasets directly.
    """
    # Initialize split_datasets if not exists
    if 'split_datasets' not in st.session_state:
        st.session_state.split_datasets = {}

    # Get parent name
    if parent_name is None:
        parent_name = st.session_state.get('current_dataset', 'Dataset')

    # Generate unique names with timestamp
    timestamp = pd.Timestamp.now().strftime('%H%M%S')
    selected_name = f"{parent_name}_Selected_{timestamp}"
    remaining_name = f"{parent_name}_Remaining_{timestamp}"

    # Save selected dataset
    st.session_state.split_datasets[selected_name] = {
        'data': selected_data,
        'type': 'PCA_Selection',
        'parent': parent_name,
        'n_samples': len(selected_data),
        'creation_time': pd.Timestamp.now(),
        'selection_method': selection_method,
        'pc_axes': f"{pc_x} vs {pc_y}"
    }

    # Save remaining dataset
    st.session_state.split_datasets[remaining_name] = {
        'data': remaining_data,
        'type': 'PCA_Remaining',
        'parent': parent_name,
        'n_samples': len(remaining_data),
        'creation_time': pd.Timestamp.now(),
        'selection_method': selection_method,
        'pc_axes': f"{pc_x} vs {pc_y}"
    }

    return selected_name, remaining_name


def get_split_datasets_info() -> Dict[str, Any]:
    """
    Get summary information about all split datasets in workspace.

    Returns
    -------
    dict
        Dictionary with summary statistics:
        - 'count': Total number of split datasets
        - 'datasets': List of dataset names
        - 'total_samples': Total number of samples across all datasets
        - 'by_type': Count of datasets by type
        - 'by_parent': Count of datasets by parent

    Examples
    --------
    >>> info = get_split_datasets_info()
    >>> print(f"Workspace contains {info['count']} datasets")
    >>> print(f"Dataset types: {info['by_type']}")

    Notes
    -----
    Returns empty structure if no split_datasets exist in session state.
    """
    if 'split_datasets' not in st.session_state or not st.session_state.split_datasets:
        return {
            'count': 0,
            'datasets': [],
            'total_samples': 0,
            'by_type': {},
            'by_parent': {}
        }

    datasets = st.session_state.split_datasets

    # Count by type
    by_type = {}
    for info in datasets.values():
        dataset_type = info.get('type', 'Unknown')
        by_type[dataset_type] = by_type.get(dataset_type, 0) + 1

    # Count by parent
    by_parent = {}
    for info in datasets.values():
        parent = info.get('parent', 'Unknown')
        by_parent[parent] = by_parent.get(parent, 0) + 1

    # Total samples
    total_samples = sum(info.get('n_samples', 0) for info in datasets.values())

    return {
        'count': len(datasets),
        'datasets': list(datasets.keys()),
        'total_samples': total_samples,
        'by_type': by_type,
        'by_parent': by_parent
    }


def delete_split_dataset(dataset_name: str) -> bool:
    """
    Delete a specific dataset from the workspace.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset to delete from split_datasets.

    Returns
    -------
    bool
        True if dataset was successfully deleted, False otherwise.

    Examples
    --------
    >>> delete_split_dataset("NIR_Data_Selected_143052")
    True

    Notes
    -----
    The function modifies st.session_state.split_datasets directly.
    """
    if 'split_datasets' not in st.session_state:
        return False

    if dataset_name in st.session_state.split_datasets:
        del st.session_state.split_datasets[dataset_name]
        return True

    return False


def clear_all_split_datasets() -> int:
    """
    Clear all split datasets from workspace.

    Returns
    -------
    int
        Number of datasets that were deleted.

    Examples
    --------
    >>> n_deleted = clear_all_split_datasets()
    >>> print(f"Cleared {n_deleted} datasets from workspace")

    Notes
    -----
    This operation cannot be undone unless datasets were previously
    saved to file using save_workspace_to_file().
    """
    if 'split_datasets' not in st.session_state:
        return 0

    count = len(st.session_state.split_datasets)
    st.session_state.split_datasets = {}
    return count
