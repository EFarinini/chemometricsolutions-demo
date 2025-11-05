"""
PLS Calculations Module

This module implements the NIPALS algorithm for PLS-1 regression and provides
functions for predictions, metrics calculation, and residual analysis.

Key Features:
- NIPALS algorithm for PLS-1 (compatible with R pls package and MATLAB)
- Mean-centered data handling
- Prediction on new data
- Comprehensive metrics (RMSE, R², MAE, bias)
- Residual calculations

Author: ChemoMetric Solutions
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, Any
from sklearn.preprocessing import StandardScaler


def pls_nipals(X: np.ndarray, y: np.ndarray, n_components: int,
               tol: float = 1e-6, max_iter: int = 500,
               preprocessing: str = 'none') -> Dict[str, Any]:
    """
    Perform PLS-1 regression using the NIPALS algorithm.

    This implementation follows the NIPALS algorithm as described in the R package 'pls'
    and MATLAB PLS implementations. Preprocessing is applied BEFORE NIPALS centering.

    Parameters
    ----------
    X : np.ndarray
        Predictor matrix (n_samples, n_features) in ORIGINAL scale
    y : np.ndarray
        Response vector (n_samples,) in ORIGINAL scale
    n_components : int
        Number of latent variables to extract
    tol : float, optional
        Convergence tolerance for NIPALS iterations (default: 1e-6)
    max_iter : int, optional
        Maximum number of iterations per component (default: 500)
    preprocessing : str, optional
        Data preprocessing method:
        - 'none': No preprocessing (default, data assumed centered)
        - 'mean_center': Center X and y
        - 'autoscale': Standardize X and y (mean=0, std=1)

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - 'T': X scores (n_samples, n_components)
        - 'P': X loadings (n_features, n_components)
        - 'W': X weights (n_features, n_components)
        - 'Q': y loadings (n_components,)
        - 'B': regression coefficients for preprocessed data (n_features,)
        - 'B_scaled': regression coefficients for scaled data (n_features,)
        - 'X_mean': mean of X for NIPALS centering
        - 'y_mean': mean of y for NIPALS centering
        - 'fitted_values': fitted values on calibration set (original scale)
        - 'residuals': residuals on calibration set (original scale)
        - 'R2': R² on calibration set
        - 'RMSE': RMSE on calibration set
        - 'preprocessing': preprocessing method used
        - 'scaler_info': preprocessing parameters for test data

    Notes
    -----
    The NIPALS algorithm iteratively extracts latent variables by:
    1. Preprocessing data (autoscale/mean_center/none)
    2. NIPALS centering (separate from preprocessing!)
    3. Deflating X and y matrices after each component
    4. Computing weights, scores, and loadings
    5. Ensuring convergence within tolerance

    References
    ----------
    - Wold, H. (1966). Estimation of principal components and related models
    - Martens & Naes (1989). Multivariate Calibration
    - R package 'pls': Mevik & Wehrens (2007)
    """
    # Convert inputs to numpy arrays and ensure proper shape
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if y.ndim == 1:
        y = y.reshape(-1, 1)

    n_samples, n_features = X.shape

    # Store original data for back-transform
    original_X = X.copy()
    original_y = y.copy().flatten()

    # ========== STEP 1: APPLY PREPROCESSING ==========
    scaler_info = {
        'preprocessing': preprocessing,
        'n_samples': n_samples,
        'n_features': n_features
    }

    if preprocessing == 'autoscale':
        print(f"🔴 DEBUG autoscale START: X shape={X.shape}, X mean={np.mean(X):.2f}, X std={np.std(X):.2f}")
        print(f"🔴 DEBUG autoscale START: y mean={np.mean(original_y):.2f}, y std={np.std(original_y):.2f}")

        # Scale X using StandardScaler
        X_scaler = StandardScaler()
        X = X_scaler.fit_transform(X)

        print(f"🔴 DEBUG autoscale AFTER SCALE X: X mean={np.mean(X):.6f}, X std={np.std(X):.6f}")

        # Scale y
        y_mean_raw = np.mean(original_y)
        y_std_raw = np.std(original_y, ddof=0)
        y = (original_y - y_mean_raw) / y_std_raw
        y = y.reshape(-1, 1)

        print(f"🔴 DEBUG autoscale AFTER SCALE y: y mean={np.mean(y):.6f}, y std={np.std(y):.6f}")
        print(f"🔴 DEBUG autoscale END: Scaling completed successfully")

        # Store scaler info
        scaler_info['X_scaler'] = X_scaler
        scaler_info['X_mean_raw'] = X_scaler.mean_
        scaler_info['X_scale_raw'] = X_scaler.scale_
        scaler_info['y_mean_raw'] = y_mean_raw
        scaler_info['y_std_raw'] = y_std_raw

    elif preprocessing == 'mean_center':
        # Center X
        X_mean_raw = np.mean(X, axis=0)
        X = X - X_mean_raw

        # Center y
        y_mean_raw = np.mean(original_y)
        y = original_y - y_mean_raw
        y = y.reshape(-1, 1)

        # Store scaler info
        scaler_info['X_scaler'] = None
        scaler_info['X_mean_raw'] = X_mean_raw
        scaler_info['X_scale_raw'] = np.ones(n_features)
        scaler_info['y_mean_raw'] = y_mean_raw
        scaler_info['y_std_raw'] = 1.0

    else:  # preprocessing == 'none'
        # No preprocessing - assume data already centered
        scaler_info['X_scaler'] = None
        scaler_info['X_mean_raw'] = np.zeros(n_features)
        scaler_info['X_scale_raw'] = np.ones(n_features)
        scaler_info['y_mean_raw'] = 0.0
        scaler_info['y_std_raw'] = 1.0

    # ========== STEP 2: NIPALS CENTERING (on preprocessed data) ==========
    # Store means for NIPALS centering (NOT preprocessing!)
    X_mean = np.mean(X, axis=0)
    y_mean = np.mean(y, axis=0)

    # Center the data for NIPALS
    X_centered = X - X_mean
    y_centered = y - y_mean

    # Initialize storage matrices
    T = np.zeros((n_samples, n_components))  # X scores
    P = np.zeros((n_features, n_components))  # X loadings
    W = np.zeros((n_features, n_components))  # X weights
    Q = np.zeros((1, n_components))  # y loadings
    U = np.zeros((n_samples, n_components))  # y scores

    # Working copies for deflation
    X_work = X_centered.copy()
    y_work = y_centered.copy()

    iterations_per_component = []
    converged_per_component = []

    # Extract each component
    for comp in range(n_components):
        # Initialize weight vector with X.T @ y
        w = X_work.T @ y_work
        w_norm = np.linalg.norm(w)

        if w_norm < 1e-10:
            # No more variance to explain
            break

        w = w / w_norm

        # NIPALS iterations for current component
        converged = False
        for iteration in range(max_iter):
            w_old = w.copy()

            # 1. Calculate X scores
            t = X_work @ w
            t_norm_sq = np.dot(t.flatten(), t.flatten())

            if t_norm_sq < 1e-10:
                break

            # 2. Calculate y loading
            q = (y_work.T @ t) / t_norm_sq

            # 3. Calculate y scores
            u = y_work * q[0, 0]

            # 4. Update weight vector
            w = X_work.T @ u
            w_norm = np.linalg.norm(w)

            if w_norm < 1e-10:
                break

            w = w / w_norm

            # Check convergence
            diff = np.linalg.norm(w - w_old)
            if diff < tol:
                converged = True
                break

        iterations_per_component.append(iteration + 1)
        converged_per_component.append(converged)

        # Recalculate final t with converged w
        t = X_work @ w
        t_norm_sq = np.dot(t.flatten(), t.flatten())

        # Calculate X loadings
        p = (X_work.T @ t) / t_norm_sq

        # Calculate y loading
        q = (y_work.T @ t) / t_norm_sq

        # Store results
        T[:, comp] = t.flatten()
        P[:, comp] = p.flatten()
        W[:, comp] = w.flatten()
        Q[0, comp] = q[0, 0]
        U[:, comp] = (y_work * q[0, 0]).flatten()

        # Deflate X and y
        X_work = X_work - np.outer(t, p)
        y_work = y_work - t * q[0, 0]

    # Calculate regression coefficients
    # B = W @ inv(P.T @ W) @ Q.T
    # For numerical stability, use pseudo-inverse
    W_used = W[:, :comp+1]
    P_used = P[:, :comp+1]
    Q_used = Q[:, :comp+1]

    PTW = P_used.T @ W_used
    try:
        PTW_inv = np.linalg.inv(PTW)
    except np.linalg.LinAlgError:
        PTW_inv = np.linalg.pinv(PTW)

    B_scaled = W_used @ PTW_inv @ Q_used.T
    B_scaled = B_scaled.flatten()

    # Calculate fitted values on calibration set (in preprocessed scale)
    fitted_values_preprocessed = X_centered @ B_scaled + y_mean[0]

    # Back-transform to original scale
    if preprocessing == 'autoscale':
        fitted_values = fitted_values_preprocessed * scaler_info['y_std_raw'] + scaler_info['y_mean_raw']
    elif preprocessing == 'mean_center':
        fitted_values = fitted_values_preprocessed + scaler_info['y_mean_raw']
    else:
        fitted_values = fitted_values_preprocessed

    # Calculate residuals (in original scale)
    residuals = original_y - fitted_values

    # Calculate metrics (in original scale)
    metrics = calculate_metrics(original_y, fitted_values)

    # Return model dictionary
    model = {
        'T': T[:, :comp+1],
        'P': P[:, :comp+1],
        'W': W[:, :comp+1],
        'Q': Q[:, :comp+1].flatten(),
        'U': U[:, :comp+1],
        'B': B_scaled,
        'B_scaled': B_scaled,
        'X_mean': X_mean,  # NIPALS centering mean
        'y_mean': y_mean[0],  # NIPALS centering mean
        'fitted_values': fitted_values,
        'residuals': residuals,
        'R2': metrics['R2'],
        'RMSE': metrics['RMSE'],
        'n_components': comp + 1,
        'converged': all(converged_per_component),
        'iterations': iterations_per_component,
        'n_samples': n_samples,
        'n_features': n_features,
        # NEW: Preprocessing info
        'preprocessing': preprocessing,
        'scaler_info': scaler_info
    }

    return model


def pls_predict(model: Dict[str, Any], X_new: np.ndarray) -> np.ndarray:
    """
    Predict response values for new data using a fitted PLS model.

    Parameters
    ----------
    model : Dict[str, Any]
        Fitted PLS model from pls_nipals()
    X_new : np.ndarray
        New predictor matrix (n_samples, n_features) in ORIGINAL scale
        MUST be in original scale (not preprocessed)

    Returns
    -------
    np.ndarray
        Predicted response values (n_samples,) in ORIGINAL scale

    Notes
    -----
    Predictions are computed as:
    1. Apply SAME preprocessing as training
    2. Center using NIPALS mean
    3. Apply regression coefficients
    4. Back-transform to original scale
    """
    # Convert to numpy array
    X_new = np.asarray(X_new, dtype=np.float64)

    # Ensure X_new is 2D
    if X_new.ndim == 1:
        X_new = X_new.reshape(1, -1)

    # Check dimensionality
    if X_new.shape[1] != len(model['X_mean']):
        raise ValueError(
            f"X_new has {X_new.shape[1]} features, but model was trained with "
            f"{len(model['X_mean'])} features"
        )

    # ========== STEP 1: Apply SAME preprocessing as training ==========
    preprocessing = model.get('preprocessing', 'none')
    scaler_info = model.get('scaler_info', {})

    if preprocessing == 'autoscale':
        # Use saved StandardScaler object
        X_scaler = scaler_info.get('X_scaler', None)
        if X_scaler is None:
            raise ValueError("StandardScaler not found in model!")

        X_preprocessed = X_scaler.transform(X_new)

    elif preprocessing == 'mean_center':
        X_mean_raw = scaler_info.get('X_mean_raw', 0.0)
        X_preprocessed = X_new - X_mean_raw

    else:  # preprocessing == 'none'
        X_preprocessed = X_new

    # ========== STEP 2: Center using NIPALS mean ==========
    X_centered = X_preprocessed - model['X_mean']

    # ========== STEP 3: Apply regression coefficients ==========
    y_pred_preprocessed = X_centered @ model['B'] + model['y_mean']

    # ========== STEP 4: Back-transform to original scale ==========
    if preprocessing == 'autoscale':
        y_std_raw = scaler_info['y_std_raw']
        y_mean_raw = scaler_info['y_mean_raw']
        y_pred = y_pred_preprocessed * y_std_raw + y_mean_raw
    elif preprocessing == 'mean_center':
        y_mean_raw = scaler_info['y_mean_raw']
        y_pred = y_pred_preprocessed + y_mean_raw
    else:
        y_pred = y_pred_preprocessed

    return y_pred


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate regression performance metrics.

    Parameters
    ----------
    y_true : np.ndarray
        True response values
    y_pred : np.ndarray
        Predicted response values

    Returns
    -------
    Dict[str, float]
        Dictionary containing:
        - 'RMSE': Root Mean Squared Error
        - 'R2': Coefficient of determination
        - 'MAE': Mean Absolute Error
        - 'Bias': Mean bias (average residual)
        - 'n_samples': Number of samples

    Notes
    -----
    R² is calculated as: 1 - SS_res / SS_tot
    Bias is the mean of residuals: mean(y_true - y_pred)
    """
    # Convert to numpy arrays and flatten
    y_true = np.asarray(y_true, dtype=np.float64).flatten()
    y_pred = np.asarray(y_pred, dtype=np.float64).flatten()

    # Check same length
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"y_true and y_pred must have same length. "
            f"Got {len(y_true)} and {len(y_pred)}"
        )

    n_samples = len(y_true)

    # Calculate residuals
    residuals = y_true - y_pred

    # RMSE (Root Mean Squared Error)
    rmse = np.sqrt(np.mean(residuals ** 2))

    # R² (Coefficient of Determination)
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    if ss_tot < 1e-10:
        # No variance in y_true
        r2 = 0.0
    else:
        r2 = 1.0 - (ss_res / ss_tot)

    # MAE (Mean Absolute Error)
    mae = np.mean(np.abs(residuals))

    # Bias (mean of residuals)
    bias = np.mean(residuals)

    # Calculate slope and intercept of y_pred vs y_true
    # (useful for diagnostic plots)
    if np.std(y_pred) > 1e-10:
        slope = np.corrcoef(y_true, y_pred)[0, 1] * (np.std(y_true) / np.std(y_pred))
        intercept = np.mean(y_true) - slope * np.mean(y_pred)
    else:
        slope = 0.0
        intercept = np.mean(y_true)

    return {
        'RMSE': rmse,
        'R2': r2,
        'MAE': mae,
        'Bias': bias,
        'RMSEP': rmse,  # Alias for RMSE (commonly used for prediction error)
        'slope': slope,
        'intercept': intercept,
        'n_samples': n_samples
    }


def calculate_residuals(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Calculate various residual metrics for diagnostics.

    Parameters
    ----------
    y_true : np.ndarray
        True response values
    y_pred : np.ndarray
        Predicted response values

    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary containing:
        - 'residuals': Raw residuals (y_true - y_pred)
        - 'standardized_residuals': Standardized residuals
        - 'percent_error': Percent error ((y_true - y_pred) / y_true * 100)

    Notes
    -----
    Standardized residuals are calculated as: residuals / std(residuals)
    """
    # Convert to numpy arrays and flatten
    y_true = np.asarray(y_true, dtype=np.float64).flatten()
    y_pred = np.asarray(y_pred, dtype=np.float64).flatten()

    # Calculate raw residuals
    residuals = y_true - y_pred

    # Calculate standardized residuals
    residual_std = np.std(residuals, ddof=1) if len(residuals) > 1 else 1.0
    if residual_std < 1e-10:
        standardized_residuals = np.zeros_like(residuals)
    else:
        standardized_residuals = residuals / residual_std

    # Calculate percent error (handle division by zero)
    percent_error = np.zeros_like(residuals)
    non_zero_mask = np.abs(y_true) > 1e-10
    percent_error[non_zero_mask] = (residuals[non_zero_mask] / y_true[non_zero_mask]) * 100

    # Calculate residuals mean and std for reporting
    residuals_mean = np.mean(residuals)
    residuals_std = residual_std

    return {
        'residuals': residuals,
        'standardized_residuals': standardized_residuals,
        'percent_error': percent_error,
        'residuals_mean': residuals_mean,
        'residuals_std': residuals_std
    }


def extract_components_subset(model: Dict[str, Any], n_comp: int) -> Dict[str, Any]:
    """
    Extract a subset of latent variables from a fitted PLS model.

    Parameters
    ----------
    model : Dict[str, Any]
        Full PLS model
    n_comp : int
        Number of components to keep (must be <= model's n_components)

    Returns
    -------
    Dict[str, Any]
        New model with reduced number of components

    Notes
    -----
    Useful for comparing models with different numbers of latent variables
    without refitting.
    """
    if n_comp < 1:
        raise ValueError(f"n_comp must be >= 1, got {n_comp}")

    if n_comp > model['n_components']:
        raise ValueError(
            f"n_comp ({n_comp}) cannot exceed model's n_components ({model['n_components']})"
        )

    # Extract subset of components
    W_subset = model['W'][:, :n_comp]
    P_subset = model['P'][:, :n_comp]
    Q_subset = model['Q'][:n_comp]

    # Recalculate regression coefficients with subset
    PTW = P_subset.T @ W_subset
    try:
        PTW_inv = np.linalg.inv(PTW)
    except np.linalg.LinAlgError:
        PTW_inv = np.linalg.pinv(PTW)

    B_subset = W_subset @ PTW_inv @ Q_subset.reshape(-1, 1)
    B_subset = B_subset.flatten()

    # Recalculate fitted values and metrics with subset
    X_centered = np.zeros((model['n_samples'], model['n_features']))  # Placeholder
    # Note: We don't have original X, so we can't recalculate fitted values
    # This is a limitation of the subset extraction

    # Create new model dictionary
    subset_model = {
        'T': model['T'][:, :n_comp],
        'P': P_subset,
        'W': W_subset,
        'Q': Q_subset,
        'U': model['U'][:, :n_comp],
        'B': B_subset,
        'B_scaled': B_subset,
        'X_mean': model['X_mean'],
        'y_mean': model['y_mean'],
        'n_components': n_comp,
        'n_samples': model['n_samples'],
        'n_features': model['n_features'],
        'converged': model.get('converged', True),
        'iterations': model.get('iterations', [])[:n_comp],
        'preprocessing': model.get('preprocessing', 'none'),
        'scaler_info': model.get('scaler_info', {})
    }

    return subset_model
