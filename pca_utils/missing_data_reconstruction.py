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
        # Select only numeric columns to avoid errors with object/string types
        X_numeric = X.select_dtypes(include=[np.number])
        X_array = X_numeric.values
    else:
        X_array = np.asarray(X)

    n_missing = np.isnan(X_array).sum()
    n_total = X_array.size
    pct_missing = (n_missing / n_total) * 100 if n_total > 0 else 0.0

    return int(n_missing), int(n_total), float(pct_missing)


def reconstruct_missing_data(
    X: Union[pd.DataFrame, np.ndarray],
    n_components: int,
    max_iter: int = 1000,
    tol: float = 1e-6,
    center: bool = True,
    scale: bool = False
) -> Tuple[pd.DataFrame, dict]:
    """
    Reconstruct missing values using NIPALS PCA algorithm.

    This function implements the same algorithm as R's pcaMethods::pca(method="nipals").
    It performs PCA on data with missing values using the NIPALS iterative algorithm,
    then reconstructs ONLY the missing values while preserving original data.

    CRITICAL: Unlike standard PCA, this uses NIPALS which handles missing data natively.

    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
        Original data matrix with missing values (NaN)
    n_components : int
        Number of principal components to use for reconstruction.
        More components = better reconstruction but possible overfitting.
    max_iter : int, optional
        Maximum iterations for NIPALS convergence. Default 1000.
    tol : float, optional
        Convergence tolerance for NIPALS. Default 1e-6.
    center : bool, optional
        Whether to center data (subtract mean). Default True.
        Matches R parameter: center=TRUE
    scale : bool, optional
        Whether to scale data (divide by std). Default False.
        Matches R parameter: scale="uv" (unit variance) if True, "none" if False

    Returns
    -------
    X_reconstructed : pd.DataFrame
        Data with missing values filled by PCA reconstruction.
        Original non-missing values are PRESERVED exactly.
    info : dict
        Dictionary with reconstruction information:
        - 'n_components_used': Number of components used
        - 'explained_variance': % variance explained by each component
        - 'total_variance_explained': Total % variance explained
        - 'n_iterations': Iterations needed for convergence
        - 'converged': Whether NIPALS converged
        - 'n_missing_before': Number of missing values before
        - 'n_missing_after': Number of missing values after (should be 0)

    Examples
    --------
    >>> # Data with missing values
    >>> X_with_nan = pd.DataFrame([[1.0, 2.0, np.nan],
    ...                             [2.0, np.nan, 4.0],
    ...                             [3.0, 4.0, 5.0]])
    >>>
    >>> # Reconstruct using 2 components
    >>> X_full, info = reconstruct_missing_data(X_with_nan, n_components=2)
    >>> print(f"Variance explained: {info['total_variance_explained']:.1f}%")
    >>> print(f"Missing values filled: {info['n_missing_before']}")

    Notes
    -----
    - Matches R/CAT algorithm: pca(M_, method="nipals", center=pre, scale=sc, nPcs=npc)
    - NIPALS = Nonlinear Iterative Partial Least Squares
    - Algorithm handles missing data by iteratively estimating them
    - Original non-missing values are NEVER modified
    - After reconstruction: M.rec[!M.na] <- M_[!M.na] (preserve originals)

    Algorithm Steps (matching R):
    1. Store mask of missing values
    2. Initialize missing values with column means
    3. Run NIPALS PCA for n_components
    4. Reconstruct data: X_hat = T @ P.T (de-centered, de-scaled)
    5. Replace ONLY missing values: X_full[missing_mask] = X_hat[missing_mask]
    6. Original values preserved: X_full[~missing_mask] = X_original[~missing_mask]
    """
    # Store DataFrame info
    is_dataframe = isinstance(X, pd.DataFrame)
    if is_dataframe:
        index = X.index
        columns = X.columns
        X_array = X.values.astype(float).copy()
    else:
        X_array = np.asarray(X, dtype=float).copy()
        index = None
        columns = None

    # Store original data and missing mask
    X_original = X_array.copy()
    missing_mask = np.isnan(X_array)
    n_missing_before = missing_mask.sum()

    if n_missing_before == 0:
        # No missing data - return as is
        if is_dataframe:
            return pd.DataFrame(X_array, index=index, columns=columns), {
                'n_components_used': 0,
                'explained_variance': [],
                'total_variance_explained': 0.0,
                'n_iterations': 0,
                'converged': True,
                'n_missing_before': 0,
                'n_missing_after': 0
            }
        else:
            return X_array, {}

    # === NIPALS PCA WITH MISSING DATA ===
    # Step 1: Initialize missing values with column means (of non-missing)
    X_filled = X_array.copy()
    for j in range(X_array.shape[1]):
        col_data = X_array[:, j]
        col_mean = np.nanmean(col_data)
        X_filled[np.isnan(X_filled[:, j]), j] = col_mean

    # Step 2: Center and scale (if requested)
    # Calculate statistics ONLY from non-missing original values
    col_means = np.zeros(X_array.shape[1])
    col_stds = np.ones(X_array.shape[1])

    if center or scale:
        for j in range(X_array.shape[1]):
            col_data_orig = X_original[:, j]
            non_missing = ~np.isnan(col_data_orig)

            if center:
                col_means[j] = np.mean(col_data_orig[non_missing])

            if scale:
                col_stds[j] = np.std(col_data_orig[non_missing], ddof=1)
                if col_stds[j] < 1e-10:
                    col_stds[j] = 1.0

    # Apply centering/scaling to filled data
    X_preprocessed = X_filled.copy()
    if center:
        X_preprocessed = X_preprocessed - col_means
    if scale:
        X_preprocessed = X_preprocessed / col_stds

    # Step 3: NIPALS algorithm for n_components
    scores_list = []
    loadings_list = []
    explained_var_list = []
    total_iter = 0
    converged = True

    X_residual = X_preprocessed.copy()

    for comp in range(n_components):
        # Initialize score vector with first column
        t = X_residual[:, 0].copy()
        t_old = np.zeros_like(t)

        iteration = 0
        while iteration < max_iter:
            # Compute loadings: p = X.T @ t / (t.T @ t)
            p = X_residual.T @ t / (t @ t)
            p = p / np.linalg.norm(p)  # Normalize

            # Compute scores: t = X @ p / (p.T @ p)
            t = X_residual @ p / (p @ p)

            # Check convergence
            if np.linalg.norm(t - t_old) < tol:
                break

            t_old = t.copy()
            iteration += 1

        total_iter += iteration

        if iteration >= max_iter:
            converged = False

        # Store component
        scores_list.append(t)
        loadings_list.append(p)

        # Explained variance
        var_explained = (t @ t) / (np.sum(X_preprocessed**2) + 1e-10) * 100
        explained_var_list.append(var_explained)

        # Deflate: X_residual = X_residual - t @ p.T
        X_residual = X_residual - np.outer(t, p)

    # Step 4: Reconstruct preprocessed data
    T = np.column_stack(scores_list)  # scores matrix (n_samples, n_components)
    P = np.column_stack(loadings_list)  # loadings matrix (n_features, n_components)

    X_reconstructed_preprocessed = T @ P.T

    # Step 5: Reverse preprocessing (de-center, de-scale)
    X_reconstructed = X_reconstructed_preprocessed.copy()
    if scale:
        X_reconstructed = X_reconstructed * col_stds
    if center:
        X_reconstructed = X_reconstructed + col_means

    # Step 6: CRITICAL - Preserve original non-missing values
    # This matches R: M.rec[!M.na] <- M_[!M.na]
    X_final = X_reconstructed.copy()
    X_final[~missing_mask] = X_original[~missing_mask]

    # Count remaining missing values (should be 0)
    n_missing_after = np.isnan(X_final).sum()

    # Prepare info dictionary
    info = {
        'n_components_used': n_components,
        'explained_variance': explained_var_list,
        'total_variance_explained': sum(explained_var_list),
        'n_iterations': total_iter,
        'converged': converged,
        'n_missing_before': int(n_missing_before),
        'n_missing_after': int(n_missing_after),
        'center': center,
        'scale': scale
    }

    # Convert back to DataFrame if needed
    if is_dataframe:
        X_final_df = pd.DataFrame(X_final, index=index, columns=columns)
        return X_final_df, info
    else:
        return X_final, info


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
        # Select only numeric columns to avoid errors with object/string types
        X_orig_numeric = X_original.select_dtypes(include=[np.number])
        X_orig_array = X_orig_numeric.values
    else:
        X_orig_array = np.asarray(X_original)

    if isinstance(X_reconstructed, pd.DataFrame):
        # Select only numeric columns to avoid errors with object/string types
        X_recon_numeric = X_reconstructed.select_dtypes(include=[np.number])
        X_recon_array = X_recon_numeric.values
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
