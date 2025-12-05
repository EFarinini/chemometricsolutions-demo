"""
PLS Validation Module

This module provides test set validation functionality for PLS regression models.
It handles loading test data, applying models, and generating validation reports.

Key Features:
- Test set loading from workspace
- Model application on test data
- Validation metrics calculation
- Comprehensive validation reports
- Extrapolation detection

Author: ChemoMetric Solutions
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
import warnings
import json
import os
from scipy import stats
from .pls_calculations import pls_predict, calculate_metrics, calculate_residuals
from .pls_preprocessing import check_data_consistency, apply_scaler, split_xy_by_column_name


def load_test_set_from_workspace(test_file: str,
                                 response_column: str,
                                 feature_columns: Optional[List[str]] = None,
                                 scaler_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Load test set data from workspace.

    Parameters
    ----------
    test_file : str
        Name of test set file in workspace
    response_column : str
        Name of response column
    feature_columns : Optional[List[str]], optional
        Expected feature columns (validates against calibration)
    scaler_info : Optional[Dict[str, Any]], optional
        Scaler info from calibration for applying same scaling

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - 'X_test': Test predictor matrix (scaled if scaler_info provided)
        - 'y_test': Test response vector
        - 'sample_names': Test sample identifiers
        - 'feature_names': Feature column names
        - 'n_samples': Number of test samples
        - 'n_features': Number of features

    Raises
    ------
    ValueError
        If test file not found or columns don't match calibration

    Notes
    -----
    Validates that test set has same features as calibration set.
    Issues warnings for extrapolation.
    """
    # Load data from workspace
    try:
        from workspace_utils import get_workspace_datasets
        datasets = get_workspace_datasets()
        if test_file not in datasets:
            raise ValueError(f"Test file '{test_file}' not found in workspace")
        test_data = datasets[test_file]
    except Exception as e:
        raise ValueError(f"Failed to load test file '{test_file}': {str(e)}")

    # Check if response column exists
    if response_column not in test_data.columns:
        raise ValueError(
            f"Response column '{response_column}' not found in test data. "
            f"Available columns: {list(test_data.columns)}"
        )

    # Split X and y
    X_test_df, y_test_series = split_xy_by_column_name(test_data, response_column)

    # Get sample names (index or generate)
    if test_data.index.name:
        sample_names = test_data.index.tolist()
    else:
        sample_names = [f"Sample_{i+1}" for i in range(len(test_data))]

    # Validate feature columns if provided
    if feature_columns is not None:
        test_features = list(X_test_df.columns)
        if test_features != feature_columns:
            missing_in_test = set(feature_columns) - set(test_features)
            extra_in_test = set(test_features) - set(feature_columns)

            error_msg = "Feature mismatch between calibration and test sets:\n"
            if missing_in_test:
                error_msg += f"  Missing in test: {missing_in_test}\n"
            if extra_in_test:
                error_msg += f"  Extra in test: {extra_in_test}\n"
            raise ValueError(error_msg)

        # Reorder columns to match calibration
        X_test_df = X_test_df[feature_columns]

    # Convert to numpy
    X_test = X_test_df.values.astype(np.float64)
    y_test = y_test_series.values.astype(np.float64)
    feature_names = list(X_test_df.columns)

    # Apply scaling if scaler_info provided
    if scaler_info is not None:
        X_test = apply_scaler(X_test, scaler_info)

    return {
        'X_test': X_test,
        'y_test': y_test,
        'sample_names': sample_names,
        'feature_names': feature_names,
        'n_samples': len(y_test),
        'n_features': X_test.shape[1]
    }


def validate_on_test(model: Dict[str, Any],
                    test_data: Dict[str, Any],
                    X_cal: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """
    Validate PLS model on independent test set.

    Parameters
    ----------
    model : Dict[str, Any]
        Fitted PLS model from pls_nipals()
    test_data : Dict[str, Any]
        Test data from load_test_set_from_workspace()
    X_cal : Optional[np.ndarray], optional
        Calibration X matrix for extrapolation detection

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - 'y_pred': Predicted values on test set
        - 'y_true': True values on test set
        - 'residuals': Prediction residuals
        - 'metrics': Dict with RMSE, R², MAE, bias
        - 'sample_names': Test sample identifiers
        - 'extrapolation_flags': Boolean array indicating extrapolation
        - 'outlier_flags': Boolean array for outliers (|std_residual| > 2.5)

    Notes
    -----
    Test data must be scaled using calibration scaling parameters.
    Predictions are back-transformed to original scale.
    """
    # Extract test data
    X_test = test_data['X_test']
    y_test = test_data['y_test']
    sample_names = test_data.get('sample_names', None)

    # Make predictions
    y_pred = pls_predict(model, X_test)

    # Calculate metrics
    metrics = calculate_metrics(y_test, y_pred)

    # Calculate residuals
    residuals_dict = calculate_residuals(y_test, y_pred)
    residuals = residuals_dict['residuals']
    std_residuals = residuals_dict['standardized_residuals']

    # Identify outliers (|standardized residual| > 2.5)
    outlier_flags = np.abs(std_residuals) > 2.5

    # Detect extrapolation if calibration data provided
    extrapolation_flags = np.zeros(len(y_test), dtype=bool)
    if X_cal is not None:
        extrap_result = detect_extrapolation(X_cal, X_test, method='range')
        extrapolation_flags = extrap_result['extrapolation_flags']

    return {
        'y_pred': y_pred,
        'y_true': y_test,
        'residuals': residuals,
        'std_residuals': std_residuals,
        'metrics': metrics,
        'sample_names': sample_names,
        'extrapolation_flags': extrapolation_flags,
        'outlier_flags': outlier_flags,
        'n_outliers': int(np.sum(outlier_flags)),
        'n_extrapolating': int(np.sum(extrapolation_flags))
    }


def generate_validation_report(validation_results: Dict[str, Any],
                               cv_results: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """
    Generate comprehensive validation report comparing calibration and test performance.

    Parameters
    ----------
    validation_results : Dict[str, Any]
        Results from validate_on_test()
    cv_results : Optional[Dict[str, Any]], optional
        Cross-validation results for comparison

    Returns
    -------
    pd.DataFrame
        Summary table with columns:
        - 'Metric': Metric name (RMSE, R², MAE, Bias)
        - 'Calibration': Calibration set performance
        - 'Cross-Validation': CV performance (if provided)
        - 'Test Set': Test set performance
        - 'Difference': Test - CV (if applicable)

    Notes
    -----
    Large differences between CV and test performance may indicate:
    - Overfitting
    - Different sample distributions
    - Extrapolation
    """
    # TODO: Implement validation report
    pass


def detect_extrapolation(X_cal: np.ndarray, X_test: np.ndarray,
                        method: str = 'range') -> Dict[str, Any]:
    """
    Detect extrapolation in test set relative to calibration set.

    Parameters
    ----------
    X_cal : np.ndarray
        Calibration predictor matrix
    X_test : np.ndarray
        Test predictor matrix
    method : str, optional
        Detection method:
        - 'range': Check if test values fall outside calibration range (default)
        - 'hotelling': Use Hotelling's T² statistic
        - 'leverage': Use leverage (hat) values

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - 'extrapolation_flags': Boolean array (n_test_samples,)
        - 'extrapolation_score': Degree of extrapolation per sample
        - 'warning_features': Features causing extrapolation
        - 'n_extrapolating': Number of samples extrapolating

    Notes
    -----
    Predictions for extrapolating samples are less reliable.
    Range method: checks if any feature value exceeds [min, max] from calibration.
    """
    # TODO: Implement extrapolation detection
    pass


def calculate_prediction_intervals(model: Dict[str, Any],
                                   X_test: np.ndarray,
                                   confidence_level: float = 0.95,
                                   cv_results: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Calculate prediction intervals for test set predictions.

    Parameters
    ----------
    model : Dict[str, Any]
        Fitted PLS model
    X_test : np.ndarray
        Test predictor matrix
    confidence_level : float, optional
        Confidence level for intervals (default: 0.95)
    cv_results : Optional[Dict[str, Any]], optional
        CV results for error estimation

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - 'lower_bound': Lower prediction interval
        - 'upper_bound': Upper prediction interval
        - 'std_error': Standard error of prediction

    Notes
    -----
    Prediction intervals estimated from CV residuals.
    Assumes normal distribution of prediction errors.
    """
    # TODO: Implement prediction intervals
    pass


def compare_calibration_test_distributions(X_cal: np.ndarray, X_test: np.ndarray,
                                          feature_names: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Compare feature distributions between calibration and test sets.

    Parameters
    ----------
    X_cal : np.ndarray
        Calibration predictor matrix
    X_test : np.ndarray
        Test predictor matrix
    feature_names : Optional[List[str]], optional
        Feature names for reporting

    Returns
    -------
    pd.DataFrame
        Summary table with columns:
        - 'Feature': Feature name
        - 'Cal_Mean': Mean in calibration set
        - 'Test_Mean': Mean in test set
        - 'Cal_Std': Std in calibration set
        - 'Test_Std': Std in test set
        - 'KS_Statistic': Kolmogorov-Smirnov test statistic
        - 'KS_PValue': KS test p-value

    Notes
    -----
    Significant differences may indicate that test set is not representative
    of calibration set.
    """
    # TODO: Implement distribution comparison
    pass


def batch_predict(model: Dict[str, Any],
                 X_new: np.ndarray,
                 scaling_params: Optional[Dict[str, Any]] = None,
                 sample_names: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Make predictions on multiple new samples and return as DataFrame.

    Parameters
    ----------
    model : Dict[str, Any]
        Fitted PLS model
    X_new : np.ndarray
        New predictor matrix
    scaling_params : Optional[Dict[str, Any]], optional
        Scaling parameters from calibration
    sample_names : Optional[List[str]], optional
        Sample identifiers

    Returns
    -------
    pd.DataFrame
        Predictions with columns:
        - 'Sample': Sample identifier
        - 'Prediction': Predicted value
        - 'Extrapolation': Boolean flag

    Notes
    -----
    Useful for production predictions on new samples.
    """
    # TODO: Implement batch prediction
    pass
