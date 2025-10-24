"""
Missing Data Reconstruction for PCA

Functions for reconstructing missing values using PCA scores and loadings.
"""

import numpy as np
import pandas as pd
from typing import Union, Tuple, Optional


def count_missing_values(X: Union[pd.DataFrame, np.ndarray]) -> Tuple[int, int, float]:
    """
    Count missing values in dataset.

    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
        Data matrix

    Returns
    -------
    n_missing : int
        Total number of missing values
    n_total : int
        Total number of cells
    pct_missing : float
        Percentage of missing values
    """
    if isinstance(X, pd.DataFrame):
        X_array = X.values
    else:
        X_array = np.asarray(X)

    n_missing = np.isnan(X_array).sum()
    n_total = X_array.size
    pct_missing = (n_missing / n_total) * 100 if n_total > 0 else 0.0

    return int(n_missing), int(n_total), float(pct_missing)


def reconstruct_missing_data(
    X: Union[pd.DataFrame, np.ndarray],
    scores: Union[pd.DataFrame, np.ndarray],
    loadings: Union[pd.DataFrame, np.ndarray],
    n_components: Optional[int] = None
) -> pd.DataFrame:
    """
    Reconstruct missing values using PCA model.

    Uses the PCA reconstruction formula: X_reconstructed = Scores @ Loadings.T
    Missing values in the original data are replaced with reconstructed values.
    Non-missing values are preserved.

    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
        Original data with missing values
    scores : pd.DataFrame or np.ndarray
        PCA scores matrix (n_samples, n_components)
    loadings : pd.DataFrame or np.ndarray
        PCA loadings matrix (n_features, n_components)
    n_components : int, optional
        Number of components to use for reconstruction.
        If None, uses all available components.

    Returns
    -------
    X_reconstructed : pd.DataFrame
        Data with missing values filled by PCA reconstruction.
        Preserves original DataFrame structure if input is DataFrame.

    Examples
    --------
    >>> # After PCA computation
    >>> X_full = reconstruct_missing_data(X_with_nan, scores, loadings, n_components=5)
    >>> # Check reconstruction
    >>> n_missing_after, _, _ = count_missing_values(X_full)
    >>> print(f"Missing values after reconstruction: {n_missing_after}")  # Should be 0

    Notes
    -----
    - Only missing values are replaced; original values are preserved
    - Reconstruction quality depends on variance captured by selected components
    - More components generally give better reconstruction but may overfit
    """
    # Store DataFrame info if input is DataFrame
    is_dataframe = isinstance(X, pd.DataFrame)
    if is_dataframe:
        index = X.index
        columns = X.columns
        X_array = X.values.astype(float)
    else:
        X_array = np.asarray(X, dtype=float)
        index = None
        columns = None

    # Convert scores and loadings to arrays
    if isinstance(scores, pd.DataFrame):
        scores_array = scores.values
    else:
        scores_array = np.asarray(scores)

    if isinstance(loadings, pd.DataFrame):
        loadings_array = loadings.values
    else:
        loadings_array = np.asarray(loadings)

    # Determine number of components to use
    available_components = min(scores_array.shape[1], loadings_array.shape[1])
    if n_components is None:
        n_components = available_components
    else:
        n_components = min(n_components, available_components)

    # Use only selected components
    scores_subset = scores_array[:, :n_components]
    loadings_subset = loadings_array[:, :n_components]

    # Reconstruct data: X_hat = T @ P.T
    X_reconstructed = scores_subset @ loadings_subset.T

    # Replace only missing values, preserve original non-missing values
    X_filled = X_array.copy()
    missing_mask = np.isnan(X_array)
    X_filled[missing_mask] = X_reconstructed[missing_mask]

    # Convert back to DataFrame if input was DataFrame
    if is_dataframe:
        X_filled_df = pd.DataFrame(X_filled, index=index, columns=columns)
        return X_filled_df
    else:
        return X_filled


def get_reconstruction_info(
    X_original: Union[pd.DataFrame, np.ndarray],
    X_reconstructed: Union[pd.DataFrame, np.ndarray]
) -> dict:
    """
    Calculate statistics about the reconstruction.

    Parameters
    ----------
    X_original : pd.DataFrame or np.ndarray
        Original data with missing values
    X_reconstructed : pd.DataFrame or np.ndarray
        Reconstructed data without missing values

    Returns
    -------
    info : dict
        Dictionary with reconstruction statistics:
        - 'n_missing_before': Number of missing values in original
        - 'n_missing_after': Number of missing values after reconstruction
        - 'n_filled': Number of values filled
        - 'filled_mean': Mean of filled values
        - 'filled_std': Standard deviation of filled values
        - 'filled_min': Minimum filled value
        - 'filled_max': Maximum filled value
    """
    if isinstance(X_original, pd.DataFrame):
        X_orig_array = X_original.values
    else:
        X_orig_array = np.asarray(X_original)

    if isinstance(X_reconstructed, pd.DataFrame):
        X_recon_array = X_reconstructed.values
    else:
        X_recon_array = np.asarray(X_reconstructed)

    # Identify missing values
    missing_mask = np.isnan(X_orig_array)
    n_missing_before = missing_mask.sum()
    n_missing_after = np.isnan(X_recon_array).sum()
    n_filled = n_missing_before - n_missing_after

    # Statistics of filled values
    if n_filled > 0:
        filled_values = X_recon_array[missing_mask]
        filled_values = filled_values[~np.isnan(filled_values)]  # Remove any remaining NaN

        filled_mean = np.mean(filled_values) if len(filled_values) > 0 else 0.0
        filled_std = np.std(filled_values) if len(filled_values) > 0 else 0.0
        filled_min = np.min(filled_values) if len(filled_values) > 0 else 0.0
        filled_max = np.max(filled_values) if len(filled_values) > 0 else 0.0
    else:
        filled_mean = filled_std = filled_min = filled_max = 0.0

    return {
        'n_missing_before': int(n_missing_before),
        'n_missing_after': int(n_missing_after),
        'n_filled': int(n_filled),
        'filled_mean': float(filled_mean),
        'filled_std': float(filled_std),
        'filled_min': float(filled_min),
        'filled_max': float(filled_max)
    }


def save_reconstructed_data(
    X_reconstructed: pd.DataFrame,
    base_filename: str,
    output_format: str = 'excel'
) -> str:
    """
    Save reconstructed data to file.

    Parameters
    ----------
    X_reconstructed : pd.DataFrame
        Reconstructed data without missing values
    base_filename : str
        Base filename (without extension)
    output_format : str, optional
        Output format: 'excel', 'csv', or 'both'. Default is 'excel'.

    Returns
    -------
    filename : str
        Path to saved file (or comma-separated paths if 'both')

    Examples
    --------
    >>> filename = save_reconstructed_data(X_full, "dataset_reconstructed", "excel")
    >>> print(f"Saved to: {filename}")
    """
    import os

    # Ensure DataFrame
    if not isinstance(X_reconstructed, pd.DataFrame):
        X_reconstructed = pd.DataFrame(X_reconstructed)

    saved_files = []

    if output_format in ['excel', 'both']:
        excel_path = f"{base_filename}.xlsx"
        X_reconstructed.to_excel(excel_path, index=True)
        saved_files.append(excel_path)

    if output_format in ['csv', 'both']:
        csv_path = f"{base_filename}.csv"
        X_reconstructed.to_csv(csv_path, index=True)
        saved_files.append(csv_path)

    return ', '.join(saved_files) if len(saved_files) > 1 else saved_files[0]
