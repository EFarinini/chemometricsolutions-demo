"""
Univariate Calculations Module

Descriptive and dispersion statistics for univariate analysis.

Functions implement algorithms from R-CAT reference software:
- Arithmetic, geometric, median averages
- Standard deviation, variance, RSD
- Robust measures: IQR, MAD, Robust CV
"""

import numpy as np
import pandas as pd
from typing import Dict, Union


# ========== DESCRIPTIVE STATISTICS ==========

def calculate_mean_arithmetic(data: Union[np.ndarray, pd.Series]) -> float:
    """Arithmetic mean, ignoring NaN values"""
    data = np.asarray(data).flatten()
    return np.nanmean(data)


def calculate_mean_geometric(data: Union[np.ndarray, pd.Series]) -> float:
    """
    Geometric mean, ignoring NaN values.

    Uses log transformation: GM = exp(mean(log(x)))
    Only for positive values.
    """
    data = np.asarray(data).flatten()
    data_clean = data[~np.isnan(data)]

    if np.any(data_clean <= 0):
        raise ValueError("Geometric mean requires positive values only")

    return np.exp(np.mean(np.log(data_clean)))


def calculate_mean_median(data: Union[np.ndarray, pd.Series]) -> float:
    """Median value, ignoring NaN values"""
    data = np.asarray(data).flatten()
    return np.nanmedian(data)


# ========== DISPERSION STATISTICS ==========

def calculate_std_dev(data: Union[np.ndarray, pd.Series], ddof: int = 1) -> float:
    """
    Standard deviation (sample, ddof=1 by default).
    Ignoring NaN values.
    """
    data = np.asarray(data).flatten()
    return np.nanstd(data, ddof=ddof)


def calculate_variance(data: Union[np.ndarray, pd.Series], ddof: int = 1) -> float:
    """Variance (sample, ddof=1 by default). Ignoring NaN values."""
    data = np.asarray(data).flatten()
    return np.nanvar(data, ddof=ddof)


def calculate_rsd(data: Union[np.ndarray, pd.Series], ddof: int = 1) -> float:
    """
    Relative Standard Deviation (RSD) = (std_dev / mean) * 100
    RSD in percentage. Ignoring NaN values.
    """
    data = np.asarray(data).flatten()
    data_clean = data[~np.isnan(data)]

    mean_val = np.mean(data_clean)
    std_val = np.std(data_clean, ddof=ddof)

    if mean_val == 0:
        return np.nan

    return (std_val / np.abs(mean_val)) * 100


# ========== ROBUST STATISTICS ==========

def calculate_iqr(data: Union[np.ndarray, pd.Series]) -> float:
    """
    Interquartile Range (IQR) = Q3 - Q1.
    Robust dispersion measure. Ignoring NaN values.
    """
    data = np.asarray(data).flatten()
    data_clean = data[~np.isnan(data)]

    q1 = np.percentile(data_clean, 25)
    q3 = np.percentile(data_clean, 75)

    return q3 - q1


def calculate_mad(data: Union[np.ndarray, pd.Series]) -> float:
    """
    Median Absolute Deviation (MAD).
    Robust dispersion measure.
    MAD = median(|xi - median(x)|)
    """
    data = np.asarray(data).flatten()
    data_clean = data[~np.isnan(data)]

    median_val = np.median(data_clean)
    deviations = np.abs(data_clean - median_val)

    return np.median(deviations)


def calculate_robust_cv(data: Union[np.ndarray, pd.Series]) -> float:
    """
    Robust Coefficient of Variation.
    Uses median and MAD instead of mean and std.

    RobustCV = (MAD / median) * 100
    """
    data = np.asarray(data).flatten()
    data_clean = data[~np.isnan(data)]

    median_val = np.median(data_clean)
    mad_val = calculate_mad(data_clean)

    if median_val == 0:
        return np.nan

    return (mad_val / np.abs(median_val)) * 100


# ========== COMPOSITE FUNCTIONS ==========

def calculate_descriptive_stats(data: Union[np.ndarray, pd.Series]) -> Dict[str, float]:
    """
    Compute all descriptive statistics for a vector.

    Returns
    -------
    dict
        Keys: 'mean_arithmetic', 'mean_geometric', 'median', 'n', 'n_na'
    """
    data = np.asarray(data).flatten()
    data_clean = data[~np.isnan(data)]

    result = {
        'mean_arithmetic': calculate_mean_arithmetic(data),
        'mean_geometric': None,
        'median': calculate_mean_median(data),
        'n': len(data_clean),
        'n_na': np.sum(np.isnan(data))
    }

    # Geometric mean only if all positive
    try:
        if np.all(data_clean > 0):
            result['mean_geometric'] = calculate_mean_geometric(data_clean)
    except:
        pass

    return result


def calculate_dispersion_stats(data: Union[np.ndarray, pd.Series]) -> Dict[str, float]:
    """
    Compute all dispersion statistics for a vector.

    Returns
    -------
    dict
        Keys: 'std_dev', 'variance', 'rsd', 'min', 'max', 'range'
    """
    data = np.asarray(data).flatten()

    return {
        'std_dev': calculate_std_dev(data),
        'variance': calculate_variance(data),
        'rsd': calculate_rsd(data),
        'min': np.nanmin(data),
        'max': np.nanmax(data),
        'range': np.nanmax(data) - np.nanmin(data)
    }


def calculate_robust_stats(data: Union[np.ndarray, pd.Series]) -> Dict[str, float]:
    """
    Compute all robust statistics for a vector.

    Returns
    -------
    dict
        Keys: 'iqr', 'mad', 'robust_cv', 'q1', 'q3'
    """
    data = np.asarray(data).flatten()
    data_clean = data[~np.isnan(data)]

    q1 = np.percentile(data_clean, 25)
    q3 = np.percentile(data_clean, 75)

    return {
        'iqr': calculate_iqr(data),
        'mad': calculate_mad(data),
        'robust_cv': calculate_robust_cv(data),
        'q1': q1,
        'q3': q3
    }


def get_column_statistics_summary(
    dataframe: pd.DataFrame,
    column_name: str
) -> Dict[str, Dict[str, float]]:
    """
    Get comprehensive statistics for a single column.

    Parameters
    ----------
    dataframe : pd.DataFrame
    column_name : str
        Column name to analyze

    Returns
    -------
    dict
        Nested dict with 'descriptive', 'dispersion', 'robust' keys
    """
    column_data = dataframe[column_name].values

    return {
        'descriptive': calculate_descriptive_stats(column_data),
        'dispersion': calculate_dispersion_stats(column_data),
        'robust': calculate_robust_stats(column_data)
    }


def get_row_profile_stats(
    dataframe: pd.DataFrame,
    row_index: int
) -> Dict[str, float]:
    """
    Get basic statistics for a single row (analytical profile).

    Parameters
    ----------
    dataframe : pd.DataFrame
    row_index : int
        Row index (0-based)

    Returns
    -------
    dict
        Keys: 'mean', 'min', 'max', 'std_dev', 'range', 'n_na'
    """
    row_data = dataframe.iloc[row_index].values

    # Convert to float and handle non-numeric values
    try:
        row_float = row_data.astype(float)
    except (ValueError, TypeError):
        # If conversion fails, try to extract numeric values only
        row_float = pd.to_numeric(row_data, errors='coerce').values

    row_clean = row_float[~np.isnan(row_float)]

    if len(row_clean) == 0:
        return {
            'mean': np.nan,
            'min': np.nan,
            'max': np.nan,
            'std_dev': np.nan,
            'range': np.nan,
            'n_na': len(row_data)
        }

    return {
        'mean': np.mean(row_clean),
        'min': np.min(row_clean),
        'max': np.max(row_clean),
        'std_dev': np.std(row_clean, ddof=1),
        'range': np.max(row_clean) - np.min(row_clean),
        'n_na': np.sum(np.isnan(row_float))
    }
