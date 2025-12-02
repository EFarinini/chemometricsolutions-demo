"""
Row transformation functions for spectral/analytical data
Transformations applied across variables (along rows)
"""

import pandas as pd
import numpy as np
from scipy.signal import savgol_filter


def snv_transform(data, col_range):
    """
    Standard Normal Variate (row autoscaling)

    Parameters:
    -----------
    data : pd.DataFrame
        Input data
    col_range : tuple
        (start_col, end_col) - column range to transform

    Returns:
    --------
    pd.DataFrame : SNV-transformed data
    """
    M = data.iloc[:, col_range[0]:col_range[1]].copy()
    M_t = M.T
    M_scaled = (M_t - M_t.mean(axis=0)) / M_t.std(axis=0, ddof=1)
    return M_scaled.T


def first_derivative_row(data, col_range):
    """
    First derivative by row (across variables)

    Parameters:
    -----------
    data : pd.DataFrame
        Input data
    col_range : tuple
        (start_col, end_col) - column range to transform

    Returns:
    --------
    pd.DataFrame : First derivative (shape: rows × cols-1)
    """
    M = data.iloc[:, col_range[0]:col_range[1]].copy()
    M_diff = M.diff(axis=1).iloc[:, 1:]
    return M_diff


def second_derivative_row(data, col_range):
    """
    Second derivative by row (across variables)

    Parameters:
    -----------
    data : pd.DataFrame
        Input data
    col_range : tuple
        (start_col, end_col) - column range to transform

    Returns:
    --------
    pd.DataFrame : Second derivative (shape: rows × cols-2)
    """
    M = data.iloc[:, col_range[0]:col_range[1]].copy()
    M_diff = M.diff(axis=1).diff(axis=1).iloc[:, 2:]
    return M_diff


def savitzky_golay_transform(data, col_range, window_length, polyorder, deriv):
    """
    Savitzky-Golay filter for smoothing and derivatives

    Parameters:
    -----------
    data : pd.DataFrame
        Input data
    col_range : tuple
        (start_col, end_col) - column range to transform
    window_length : int
        Window length (must be odd)
    polyorder : int
        Polynomial order
    deriv : int
        Derivative order (0 = smoothing, 1 = first derivative, 2 = second derivative)

    Returns:
    --------
    pd.DataFrame : Savitzky-Golay filtered data
    """
    M = data.iloc[:, col_range[0]:col_range[1]].copy()
    M_sg = M.apply(lambda row: savgol_filter(row, window_length, polyorder, deriv=deriv), axis=1)
    return pd.DataFrame(M_sg.tolist(), index=M.index)


def moving_average_row(data, col_range, window):
    """
    Moving average by row (across variables)

    Parameters:
    -----------
    data : pd.DataFrame
        Input data
    col_range : tuple
        (start_col, end_col) - column range to transform
    window : int
        Window size

    Returns:
    --------
    pd.DataFrame : Moving averaged data (reduced dimensions)
    """
    M = data.iloc[:, col_range[0]:col_range[1]].copy()
    M_ma = M.rolling(window=window, axis=1, center=True).mean()
    return M_ma.dropna(axis=1)


def row_sum100(data, col_range):
    """
    Normalize row sum to 100

    Parameters:
    -----------
    data : pd.DataFrame
        Input data
    col_range : tuple
        (start_col, end_col) - column range to transform

    Returns:
    --------
    pd.DataFrame : Normalized data (row sum = 100)
    """
    M = data.iloc[:, col_range[0]:col_range[1]].copy()
    row_sums = M.sum(axis=1)
    M_norm = M.div(row_sums, axis=0) * 100
    return M_norm


def binning_transform(data, col_range, bin_width):
    """
    Binning (averaging adjacent variables)

    Parameters:
    -----------
    data : pd.DataFrame
        Input data
    col_range : tuple
        (start_col, end_col) - column range to transform
    bin_width : int
        Number of variables to average per bin

    Returns:
    --------
    pd.DataFrame : Binned data (reduced dimensions)

    Raises:
    -------
    ValueError : If number of columns not divisible by bin_width
    """
    M = data.iloc[:, col_range[0]:col_range[1]].copy()
    n_cols = M.shape[1]

    if n_cols % bin_width != 0:
        raise ValueError(f"Number of columns ({n_cols}) must be multiple of bin width ({bin_width})")

    n_bins = n_cols // bin_width
    binned_data = []

    for i in range(n_bins):
        start_idx = i * bin_width
        end_idx = start_idx + bin_width
        bin_mean = M.iloc[:, start_idx:end_idx].mean(axis=1)
        binned_data.append(bin_mean)

    return pd.DataFrame(binned_data).T
