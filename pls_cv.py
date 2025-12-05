"""
PLS Cross-Validation Module

This module implements K-fold cross-validation for PLS-1 regression with optional
randomization and repeated CV. It provides functions for optimal latent variable
selection based on RMSECV.

Key Features:
- K-fold cross-validation with optional randomization
- Repeated CV for robust estimates
- Automatic optimal LV selection (minimum RMSECV or 1-SE rule)
- Full CV summary with all metrics per LV

Author: ChemoMetric Solutions
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from sklearn.model_selection import KFold
import warnings
from .pls_calculations import pls_nipals, pls_predict, calculate_metrics


def repeated_kfold_cv(X: np.ndarray, y: np.ndarray,
                      max_components: int,
                      n_folds: int = 10,
                      n_repeats: int = 1,
                      randomize: bool = False,
                      random_state: Optional[int] = None,
                      preprocessing: str = 'none') -> Dict[str, Any]:
    """
    Perform repeated K-fold cross-validation for PLS regression.

    This function implements K-fold CV similar to the R package 'pls' with optional
    randomization of sample order. CV is repeated multiple times to get robust estimates.

    Parameters
    ----------
    X : np.ndarray
        Predictor matrix (RAW, unpreprocessed) (n_samples, n_features)
    y : np.ndarray
        Response vector (RAW, unpreprocessed) (n_samples,)
    max_components : int
        Maximum number of latent variables to test
    n_folds : int, optional
        Number of CV folds (default: 10)
    n_repeats : int, optional
        Number of times to repeat CV (default: 1)
    randomize : bool, optional
        Whether to randomize sample order before CV (default: False)
    random_state : Optional[int], optional
        Random seed for reproducibility (default: None)
    preprocessing : str, optional
        Preprocessing method: 'none', 'mean_center', 'autoscale' (default: 'none')
        Preprocessing is applied INSIDE pls_nipals() on each CV fold

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - 'RMSECV': Mean RMSECV per component (max_components,)
        - 'RMSECV_std': Standard deviation of RMSECV across repeats
        - 'R2CV': Mean R²CV per component
        - 'R2CV_std': Standard deviation of R²CV across repeats
        - 'cv_predictions': Predictions for each sample (for plotting)
        - 'cv_residuals': Residuals for each sample
        - 'n_folds': Number of folds used
        - 'n_repeats': Number of repeats used
        - 'randomized': Whether data was randomized

    Notes
    -----
    CV strategy:
    1. Optionally randomize sample order
    2. Split data into K folds
    3. For each fold: train on K-1 folds, predict on held-out fold
    4. Repeat process n_repeats times
    5. Average metrics across all repeats

    References
    ----------
    - Mevik & Wehrens (2007). The pls Package: Principal Component and
      Partial Least Squares Regression in R. Journal of Statistical Software.
    """
    # Convert to numpy arrays
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).flatten()

    n_samples = X.shape[0]

    # Initialize storage for results across repeats
    # Shape: (n_repeats, max_components)
    rmsecv_all_repeats = np.zeros((n_repeats, max_components))
    r2cv_all_repeats = np.zeros((n_repeats, max_components))

    # Storage for CV predictions (last repeat only, for plotting)
    cv_predictions_last = np.zeros((n_samples, max_components))
    cv_true_last = y.copy()

    # Set up random state
    if random_state is not None:
        rng = np.random.RandomState(random_state)
    else:
        rng = np.random.RandomState()

    # Perform repeated CV
    for repeat_idx in range(n_repeats):
        # Optionally randomize sample order
        if randomize:
            shuffle_idx = rng.permutation(n_samples)
            X_shuffled = X[shuffle_idx]
            y_shuffled = y[shuffle_idx]
        else:
            X_shuffled = X.copy()
            y_shuffled = y.copy()

        # Initialize KFold splitter
        kfold = KFold(n_splits=n_folds, shuffle=False, random_state=None)

        # Storage for CV predictions for this repeat
        cv_predictions = np.zeros((n_samples, max_components))
        cv_predictions[:] = np.nan  # Initialize with NaN to track which samples were predicted

        # Perform K-fold CV
        for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X_shuffled)):
            # Split data
            X_train = X_shuffled[train_idx]
            y_train = y_shuffled[train_idx]
            X_test = X_shuffled[test_idx]
            y_test = y_shuffled[test_idx]

            # Fit PLS model with max_components on training fold
            # NOTE: X_train and y_train are RAW data - preprocessing applied inside pls_nipals()
            try:
                model = pls_nipals(X_train, y_train, n_components=max_components,
                                 preprocessing=preprocessing)

                # Debug logging (first fold of first repeat only)
                if repeat_idx == 0 and fold_idx == 0:
                    try:
                        import streamlit as st
                        st.write(f"🔍 DEBUG CV: preprocessing='{preprocessing}', model uses '{model['preprocessing']}'")
                    except:
                        pass  # Streamlit not available in some contexts

            except Exception as e:
                warnings.warn(
                    f"PLS fitting failed for repeat {repeat_idx+1}, fold {fold_idx+1}: {str(e)}"
                )
                continue

            # Predict for each number of components (1 to max_components)
            for n_comp in range(1, max_components + 1):
                # Extract subset model with n_comp components
                if n_comp < model['n_components']:
                    from .pls_calculations import extract_components_subset
                    model_subset = extract_components_subset(model, n_comp)
                else:
                    model_subset = model

                # Predict on test fold
                # NOTE: X_test is RAW data - pls_predict() applies same preprocessing
                try:
                    y_pred_fold = pls_predict(model_subset, X_test)
                    cv_predictions[test_idx, n_comp - 1] = y_pred_fold
                except Exception as e:
                    warnings.warn(
                        f"Prediction failed for repeat {repeat_idx+1}, fold {fold_idx+1}, "
                        f"n_comp={n_comp}: {str(e)}"
                    )

        # Calculate RMSECV and R²CV for each number of components
        for n_comp in range(max_components):
            # Get predictions for this component count
            y_pred_comp = cv_predictions[:, n_comp]

            # Remove any NaN predictions (shouldn't happen, but safety check)
            valid_mask = ~np.isnan(y_pred_comp)
            y_true_valid = y_shuffled[valid_mask]
            y_pred_valid = y_pred_comp[valid_mask]

            if len(y_true_valid) > 0:
                # Calculate metrics
                metrics = calculate_metrics(y_true_valid, y_pred_valid)
                rmsecv_all_repeats[repeat_idx, n_comp] = metrics['RMSE']
                r2cv_all_repeats[repeat_idx, n_comp] = metrics['R2']
            else:
                rmsecv_all_repeats[repeat_idx, n_comp] = np.nan
                r2cv_all_repeats[repeat_idx, n_comp] = np.nan

        # Store last repeat's predictions for plotting
        if repeat_idx == n_repeats - 1:
            if randomize:
                # Map predictions back to original order
                cv_predictions_last[shuffle_idx] = cv_predictions
            else:
                cv_predictions_last = cv_predictions

    # Calculate mean and std across repeats
    rmsecv_mean = np.nanmean(rmsecv_all_repeats, axis=0)
    rmsecv_std = np.nanstd(rmsecv_all_repeats, axis=0, ddof=1)
    r2cv_mean = np.nanmean(r2cv_all_repeats, axis=0)
    r2cv_std = np.nanstd(r2cv_all_repeats, axis=0, ddof=1)

    # Calculate residuals for last repeat
    cv_residuals_last = cv_true_last[:, np.newaxis] - cv_predictions_last

    # Return results
    return {
        'RMSECV': rmsecv_mean,
        'RMSECV_std': rmsecv_std,
        'R2CV': r2cv_mean,
        'R2CV_std': r2cv_std,
        'RMSECV_all': rmsecv_all_repeats,
        'R2CV_all': r2cv_all_repeats,
        'cv_predictions': cv_predictions_last,
        'cv_residuals': cv_residuals_last,
        'cv_true': cv_true_last,
        'n_folds': n_folds,
        'n_repeats': n_repeats,
        'randomized': randomize,
        'n_components_range': np.arange(1, max_components + 1)
    }


def select_optimal_lv(cv_results: Dict[str, Any],
                      method: str = 'min',
                      se_factor: float = 1.0) -> Dict[str, Any]:
    """
    Select optimal number of latent variables from CV results.

    Parameters
    ----------
    cv_results : Dict[str, Any]
        Results from repeated_kfold_cv()
    method : str, optional
        Selection method:
        - 'min': Select LV with minimum RMSECV (default)
        - '1se': Select most parsimonious model within 1 SE of minimum
        - 'custom_se': Use custom SE factor
    se_factor : float, optional
        Factor for SE rule (default: 1.0)

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - 'optimal_lv': Selected number of latent variables
        - 'optimal_rmsecv': RMSECV at optimal LV
        - 'optimal_r2cv': R²CV at optimal LV
        - 'method': Selection method used
        - 'all_rmsecv': RMSECV for all tested LVs
        - 'se_threshold': SE threshold (if applicable)

    Notes
    -----
    1-SE rule (Breiman et al., 1984):
    Select the most parsimonious model whose RMSECV is within 1 standard error
    of the minimum RMSECV. This helps prevent overfitting.

    References
    ----------
    - Breiman, Friedman, Olshen, & Stone (1984). Classification and Regression Trees.
    """
    rmsecv_mean = cv_results['RMSECV']
    rmsecv_std = cv_results['RMSECV_std']
    r2cv_mean = cv_results['R2CV']

    # Find minimum RMSECV
    min_idx = np.argmin(rmsecv_mean)
    min_rmsecv = rmsecv_mean[min_idx]
    min_std = rmsecv_std[min_idx]

    se_threshold = None

    if method == 'min':
        # Select LV with minimum RMSECV
        optimal_idx = min_idx

    elif method in ['1se', 'custom_se']:
        # One standard error rule (or custom SE factor)
        # Find the most parsimonious (smallest) model whose RMSECV is within
        # se_factor * SE of the minimum RMSECV
        se_threshold = min_rmsecv + se_factor * min_std

        # Find the first (smallest) LV where RMSECV <= threshold
        candidates = np.where(rmsecv_mean <= se_threshold)[0]

        if len(candidates) > 0:
            optimal_idx = candidates[0]
        else:
            # Fallback to minimum if no candidate found
            optimal_idx = min_idx
            warnings.warn(
                "No model found within SE threshold, using minimum RMSECV instead"
            )

    else:
        raise ValueError(
            f"Unknown method '{method}'. Choose from: 'min', '1se', 'custom_se'"
        )

    # Get optimal LV (1-indexed)
    optimal_lv = optimal_idx + 1

    return {
        'optimal_lv': optimal_lv,
        'optimal_rmsecv': rmsecv_mean[optimal_idx],
        'optimal_r2cv': r2cv_mean[optimal_idx],
        'optimal_rmsecv_std': rmsecv_std[optimal_idx],
        'method': method,
        'all_rmsecv': rmsecv_mean,
        'all_rmsecv_std': rmsecv_std,
        'all_r2cv': r2cv_mean,
        'se_threshold': se_threshold,
        'min_rmsecv': min_rmsecv,
        'min_rmsecv_lv': min_idx + 1,
        'se_factor': se_factor
    }


def full_cv_summary(cv_results: Dict[str, Any],
                    optimal_lv: int) -> pd.DataFrame:
    """
    Generate a comprehensive CV summary table.

    Parameters
    ----------
    cv_results : Dict[str, Any]
        Results from repeated_kfold_cv()
    optimal_lv : int
        Selected optimal number of latent variables

    Returns
    -------
    pd.DataFrame
        Summary table with columns:
        - 'n_comp': Number of components
        - 'RMSECV': Mean RMSECV
        - 'RMSECV_std': Standard deviation of RMSECV
        - 'R2CV': Mean R²CV
        - 'R2CV_std': Standard deviation of R²CV
        - 'is_optimal': Boolean indicating optimal LV

    Notes
    -----
    This table can be displayed in Streamlit for easy interpretation of CV results.
    """
    n_components = len(cv_results['RMSECV'])

    # Create summary dataframe
    summary_df = pd.DataFrame({
        'n_comp': np.arange(1, n_components + 1),
        'RMSECV': cv_results['RMSECV'],
        'RMSECV_std': cv_results['RMSECV_std'],
        'R2CV': cv_results['R2CV'],
        'R2CV_std': cv_results['R2CV_std'],
        'is_optimal': False
    })

    # Mark optimal LV
    if 1 <= optimal_lv <= n_components:
        summary_df.loc[optimal_lv - 1, 'is_optimal'] = True

    # Format for display
    summary_df['RMSECV'] = summary_df['RMSECV'].round(4)
    summary_df['RMSECV_std'] = summary_df['RMSECV_std'].round(4)
    summary_df['R2CV'] = summary_df['R2CV'].round(4)
    summary_df['R2CV_std'] = summary_df['R2CV_std'].round(4)

    return summary_df


def calculate_press(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate PRESS (Predicted Residual Sum of Squares).

    Parameters
    ----------
    y_true : np.ndarray
        True response values
    y_pred : np.ndarray
        Predicted response values from CV

    Returns
    -------
    float
        PRESS statistic

    Notes
    -----
    PRESS = sum((y_true - y_pred)^2)
    Used to calculate Q² statistic: Q² = 1 - PRESS/SS_total
    """
    y_true = np.asarray(y_true, dtype=np.float64).flatten()
    y_pred = np.asarray(y_pred, dtype=np.float64).flatten()

    residuals = y_true - y_pred
    press = np.sum(residuals ** 2)

    return press


def calculate_q2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Q² statistic from cross-validation predictions.

    Parameters
    ----------
    y_true : np.ndarray
        True response values
    y_pred : np.ndarray
        Cross-validated predictions

    Returns
    -------
    float
        Q² statistic

    Notes
    -----
    Q² = 1 - PRESS / SS_total
    Q² is similar to R², but calculated from CV predictions.
    """
    y_true = np.asarray(y_true, dtype=np.float64).flatten()
    y_pred = np.asarray(y_pred, dtype=np.float64).flatten()

    press = calculate_press(y_true, y_pred)
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)

    if ss_total < 1e-10:
        return 0.0

    q2 = 1.0 - (press / ss_total)

    return q2


def stratified_cv_split(y: np.ndarray, n_folds: int,
                       random_state: Optional[int] = None) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Create stratified CV splits for continuous response.

    Parameters
    ----------
    y : np.ndarray
        Response vector
    n_folds : int
        Number of folds
    random_state : Optional[int], optional
        Random seed

    Returns
    -------
    List[Tuple[np.ndarray, np.ndarray]]
        List of (train_indices, test_indices) tuples

    Notes
    -----
    For continuous responses, stratification is done by binning the response
    into quantiles to ensure balanced distribution across folds.
    """
    y = np.asarray(y, dtype=np.float64).flatten()
    n_samples = len(y)

    # Bin the response into quantiles
    try:
        # Use n_folds quantiles for stratification
        quantiles = np.percentile(y, np.linspace(0, 100, n_folds + 1))
        bins = np.digitize(y, quantiles[1:-1])  # Exclude min and max
    except Exception:
        # Fallback to simple binning if quantiles fail
        bins = np.digitize(y, np.linspace(y.min(), y.max(), n_folds + 1)[1:-1])

    # Use stratified KFold on the binned response
    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    splits = []
    for train_idx, test_idx in skf.split(np.zeros(n_samples), bins):
        splits.append((train_idx, test_idx))

    return splits
