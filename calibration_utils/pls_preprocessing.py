"""
PLS Preprocessing Module

This module provides data preparation and validation utilities for PLS regression.
It handles data loading, validation, scaling, and splitting.

Key Features:
- Data loading from workspace
- Input validation (missing values, dimensions)
- X/Y splitting by column name
- Scaling/centering options
- Data consistency checks

Author: ChemoMetric Solutions
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List, Any
from sklearn.preprocessing import StandardScaler
import warnings


def prepare_calibration_data(data: pd.DataFrame,
                             response_column: str,
                             scaling: str = 'mean_center') -> Dict[str, Any]:
    """
    Prepare calibration data for PLS modeling.

    NOTE: Preprocessing is now handled INSIDE pls_nipals()
          This function only validates and splits data.

    Parameters
    ----------
    data : pd.DataFrame
        Input dataframe with predictors and response
    response_column : str
        Name of the response column
    scaling : str, optional
        DEPRECATED - kept for backward compatibility
        Actual preprocessing is passed to pls_nipals()

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - 'X': Predictor matrix (RAW, no preprocessing) (np.ndarray)
        - 'y': Response vector (RAW, no preprocessing) (np.ndarray)
        - 'feature_names': List of predictor column names
        - 'response_name': Name of response column
        - 'n_samples': Number of samples
        - 'n_features': Number of predictors

    Raises
    ------
    ValueError
        If response_column not in data or data contains NaN values

    Notes
    -----
    Preprocessing is now applied inside pls_nipals() to avoid double-centering.
    This function only validates data and converts to numpy arrays.
    """
    # Validate response column exists
    if response_column not in data.columns:
        raise ValueError(f"Response column '{response_column}' not found in data")

    # Check for missing values
    if data.isnull().any().any():
        missing_info = detect_missing_values(data)
        raise ValueError(
            f"Data contains {missing_info['n_missing']} missing values. "
            f"PLS-NIPALS cannot handle missing data. Please impute or remove."
        )

    # Split into X and y
    X_df, y_series = split_xy_by_column_name(data, response_column)

    # Convert to numpy arrays
    X = X_df.values.astype(np.float64)
    y = y_series.values.astype(np.float64)

    # Store metadata
    feature_names = list(X_df.columns)
    n_samples, n_features = X.shape

    # Validate input
    is_valid, errors = validate_pls_input(X, y)
    if not is_valid:
        raise ValueError(f"Invalid input data:\n" + "\n".join(errors))

    # Check for constant features
    X_df_clean, removed_features = remove_constant_features(X_df, threshold=1e-10)
    if removed_features:
        warnings.warn(
            f"Removed {len(removed_features)} constant features: {removed_features}"
        )
        X = X_df_clean.values.astype(np.float64)
        feature_names = list(X_df_clean.columns)
        n_features = X.shape[1]

    # Return RAW data - preprocessing is done inside pls_nipals()
    return {
        'X': X,  # RAW data
        'y': y,  # RAW data
        'feature_names': feature_names,
        'response_name': response_column,
        'n_samples': n_samples,
        'n_features': n_features
    }


def apply_scaler(X_new: np.ndarray,
                scaler_info: Dict[str, Any]) -> np.ndarray:
    """
    Apply saved scaler parameters to new data.

    Parameters
    ----------
    X_new : np.ndarray
        New data matrix to scale (n_samples, n_features)
    scaler_info : Dict[str, Any]
        Scaler information from prepare_calibration_data()

    Returns
    -------
    np.ndarray
        Scaled data matrix

    Raises
    ------
    ValueError
        If X_new has wrong number of features

    Notes
    -----
    Applies the same scaling transformation used on calibration data.
    Uses saved mean and std from training set.

    CRITICAL: For autoscale, uses sklearn StandardScaler.transform() for
    exact MATLAB compatibility.
    """
    X_new = np.asarray(X_new, dtype=np.float64)

    # Validate dimensions
    if X_new.ndim == 1:
        X_new = X_new.reshape(1, -1)

    if X_new.shape[1] != scaler_info['n_features']:
        raise ValueError(
            f"X_new has {X_new.shape[1]} features, but scaler expects "
            f"{scaler_info['n_features']} features"
        )

    # Apply scaling based on saved method
    scaling_method = scaler_info['scaling']

    if scaling_method == 'none':
        # No preprocessing
        X_scaled = X_new.copy()

    elif scaling_method == 'mean_center':
        # Mean center only
        X_scaled = X_new - scaler_info['X_mean']

    elif scaling_method == 'autoscale':
        # CRITICAL: Use sklearn StandardScaler.transform() if available
        # This ensures EXACT same transformation as calibration (MATLAB compatible)
        if 'X_scaler' in scaler_info and scaler_info['X_scaler'] is not None:
            # Use the fitted StandardScaler object
            X_scaled = scaler_info['X_scaler'].transform(X_new)
        else:
            # Fallback to manual scaling (legacy compatibility)
            X_scaled = (X_new - scaler_info['X_mean']) / scaler_info['X_std']

            # Handle features with zero variance (set to 0)
            zero_var_mask = scaler_info['X_std'] < 1e-10
            if np.any(zero_var_mask):
                X_scaled[:, zero_var_mask] = 0.0

    else:
        raise ValueError(f"Unknown scaling method in scaler_info: {scaling_method}")

    return X_scaled


def validate_pls_input(X: np.ndarray, y: np.ndarray,
                      max_components: Optional[int] = None) -> Tuple[bool, List[str]]:
    """
    Validate input data for PLS modeling.

    Parameters
    ----------
    X : np.ndarray
        Predictor matrix
    y : np.ndarray
        Response vector
    max_components : Optional[int], optional
        Maximum number of components to validate against

    Returns
    -------
    Tuple[bool, List[str]]
        (is_valid, error_messages)

    Validation checks:
    - No NaN or Inf values
    - Matching number of samples
    - Sufficient samples (n > n_features recommended)
    - max_components <= min(n_samples, n_features)
    - At least 2 samples
    """
    errors = []

    # Convert to numpy arrays
    X = np.asarray(X)
    y = np.asarray(y).flatten()

    # Check dimensions
    if X.ndim != 2:
        errors.append(f"X must be 2-dimensional, got {X.ndim} dimensions")
        return False, errors

    if y.ndim != 1:
        errors.append(f"y must be 1-dimensional, got {y.ndim} dimensions")
        return False, errors

    n_samples_X, n_features = X.shape
    n_samples_y = len(y)

    # Check matching sample sizes
    if n_samples_X != n_samples_y:
        errors.append(
            f"X and y must have same number of samples. "
            f"X has {n_samples_X}, y has {n_samples_y}"
        )

    # Check minimum samples
    if n_samples_X < 2:
        errors.append(f"Need at least 2 samples, got {n_samples_X}")

    # Check for NaN values
    if np.any(np.isnan(X)):
        n_nan_X = np.sum(np.isnan(X))
        errors.append(f"X contains {n_nan_X} NaN values")

    if np.any(np.isnan(y)):
        n_nan_y = np.sum(np.isnan(y))
        errors.append(f"y contains {n_nan_y} NaN values")

    # Check for Inf values
    if np.any(np.isinf(X)):
        n_inf_X = np.sum(np.isinf(X))
        errors.append(f"X contains {n_inf_X} Inf values")

    if np.any(np.isinf(y)):
        n_inf_y = np.sum(np.isinf(y))
        errors.append(f"y contains {n_inf_y} Inf values")

    # Check sufficient samples vs features
    if n_samples_X <= n_features:
        warnings.warn(
            f"Number of samples ({n_samples_X}) should be greater than "
            f"number of features ({n_features}) for robust PLS modeling"
        )

    # Check variance in y
    if n_samples_y > 1:
        y_var = np.var(y)
        if y_var < 1e-10:
            errors.append("y has near-zero variance. Cannot build regression model.")

    # Check max_components
    if max_components is not None:
        max_possible = min(n_samples_X, n_features)
        if max_components > max_possible:
            errors.append(
                f"max_components ({max_components}) cannot exceed "
                f"min(n_samples, n_features) = {max_possible}"
            )
        if max_components < 1:
            errors.append(f"max_components must be >= 1, got {max_components}")

    # Return validation result
    is_valid = len(errors) == 0
    return is_valid, errors


def split_xy_by_column_name(data: pd.DataFrame,
                            response_column: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Split dataframe into X (predictors) and y (response).

    Parameters
    ----------
    data : pd.DataFrame
        Input dataframe
    response_column : str
        Name of response column

    Returns
    -------
    Tuple[pd.DataFrame, pd.Series]
        (X, y) where X contains all columns except response

    Raises
    ------
    ValueError
        If response_column not in data

    Notes
    -----
    Response column is removed from X and returned as y.
    Column order in X is preserved.
    """
    if response_column not in data.columns:
        raise ValueError(
            f"Response column '{response_column}' not found in data. "
            f"Available columns: {list(data.columns)}"
        )

    # Extract y
    y = data[response_column]

    # Extract X (all columns except response)
    X = data.drop(columns=[response_column])

    return X, y


def detect_missing_values(data: pd.DataFrame) -> Dict[str, Any]:
    """
    Detect and report missing values in data.

    Parameters
    ----------
    data : pd.DataFrame
        Input dataframe

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - 'has_missing': Boolean indicating presence of NaN
        - 'n_missing': Total number of missing values
        - 'missing_by_column': Series with missing count per column
        - 'missing_percentage': Percentage of missing values

    Notes
    -----
    PLS-NIPALS cannot handle missing values directly.
    Options: remove samples, impute, or use specialized algorithms.
    """
    missing_by_column = data.isnull().sum()
    n_missing = missing_by_column.sum()
    total_values = data.shape[0] * data.shape[1]

    return {
        'has_missing': n_missing > 0,
        'n_missing': int(n_missing),
        'missing_by_column': missing_by_column,
        'missing_percentage': (n_missing / total_values * 100) if total_values > 0 else 0.0
    }


def remove_constant_features(X: pd.DataFrame,
                            threshold: float = 0.0) -> Tuple[pd.DataFrame, List[str]]:
    """
    Remove constant or near-constant features.

    Parameters
    ----------
    X : pd.DataFrame
        Predictor dataframe
    threshold : float, optional
        Variance threshold (default: 0.0 = only exact constants)

    Returns
    -------
    Tuple[pd.DataFrame, List[str]]
        (X_filtered, removed_columns)

    Notes
    -----
    Constant features provide no information and can cause numerical issues.
    """
    # Calculate variance for each column
    variances = X.var(ddof=1)

    # Find columns with variance below threshold
    constant_mask = variances <= threshold
    constant_columns = list(X.columns[constant_mask])

    # Remove constant columns
    X_filtered = X.loc[:, ~constant_mask]

    return X_filtered, constant_columns


def check_data_consistency(X_cal: np.ndarray, X_test: np.ndarray) -> Dict[str, Any]:
    """
    Check consistency between calibration and test sets.

    Parameters
    ----------
    X_cal : np.ndarray
        Calibration predictor matrix
    X_test : np.ndarray
        Test predictor matrix

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - 'same_n_features': Boolean
        - 'test_in_range': Boolean (all test values within calibration range)
        - 'warnings': List of warning messages

    Notes
    -----
    Extrapolation warnings are issued if test samples fall outside
    the calibration range.
    """
    X_cal = np.asarray(X_cal)
    X_test = np.asarray(X_test)

    warnings_list = []

    # Check number of features
    same_n_features = X_cal.shape[1] == X_test.shape[1]
    if not same_n_features:
        warnings_list.append(
            f"Feature count mismatch: calibration has {X_cal.shape[1]} features, "
            f"test has {X_test.shape[1]} features"
        )
        return {
            'same_n_features': False,
            'test_in_range': False,
            'warnings': warnings_list
        }

    # Check if test values are within calibration range
    cal_min = np.min(X_cal, axis=0)
    cal_max = np.max(X_cal, axis=0)
    test_min = np.min(X_test, axis=0)
    test_max = np.max(X_test, axis=0)

    # Find features where test set extrapolates
    below_range = test_min < cal_min
    above_range = test_max > cal_max

    extrapolating_features = np.where(below_range | above_range)[0]

    test_in_range = len(extrapolating_features) == 0

    if not test_in_range:
        warnings_list.append(
            f"{len(extrapolating_features)} features have test values outside "
            f"calibration range (extrapolation)"
        )

        # Report details for first few extrapolating features
        for i, feat_idx in enumerate(extrapolating_features[:5]):
            if below_range[feat_idx]:
                warnings_list.append(
                    f"  Feature {feat_idx}: test min ({test_min[feat_idx]:.3f}) < "
                    f"cal min ({cal_min[feat_idx]:.3f})"
                )
            if above_range[feat_idx]:
                warnings_list.append(
                    f"  Feature {feat_idx}: test max ({test_max[feat_idx]:.3f}) > "
                    f"cal max ({cal_max[feat_idx]:.3f})"
                )

        if len(extrapolating_features) > 5:
            warnings_list.append(
                f"  ... and {len(extrapolating_features) - 5} more features"
            )

    return {
        'same_n_features': same_n_features,
        'test_in_range': test_in_range,
        'extrapolating_features': extrapolating_features.tolist() if not test_in_range else [],
        'warnings': warnings_list
    }
