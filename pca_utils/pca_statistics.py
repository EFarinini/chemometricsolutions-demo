"""
PCA Statistical Functions

Statistical analysis and diagnostic functions for Principal Component Analysis (PCA).
Includes calculations for T2 statistics, Q residuals, contributions, leverage,
and cross-validation metrics.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import f, chi2, t
from sklearn.decomposition import PCA
from typing import Tuple, Dict, Any, Union, Optional


def calculate_hotelling_t2(
    scores: Union[np.ndarray, pd.DataFrame],
    eigenvalues: np.ndarray,
    alpha: float = 0.95
) -> Tuple[np.ndarray, float]:
    """
    Calculate Hotelling's T2 statistic for PCA scores.

    T2 measures the Mahalanobis distance of each sample from the model center
    in the principal component space. It indicates how far a sample is from
    the multivariate mean, accounting for variance in each PC direction.

    Parameters
    ----------
    scores : np.ndarray or pd.DataFrame
        PCA scores matrix of shape (n_samples, n_components).
        Each row is a sample projected into PC space.
    eigenvalues : np.ndarray
        Eigenvalues (explained variance) for each PC. Shape: (n_components,).
        Used to normalize distances by variance in each PC direction.
    alpha : float, optional
        Confidence level for critical limit (0 < alpha < 1).
        Default is 0.95 (95% confidence).

    Returns
    -------
    t2_values : np.ndarray
        T2 statistic for each sample. Shape: (n_samples,).
        Higher values indicate samples farther from model center.
    t2_limit : float
        Critical value at specified confidence level.
        Samples with T2 > t2_limit are considered outliers.

    Notes
    -----
    T2 statistic formula:

    .. math::
        T^2_i = \sum_{j=1}^{a} \frac{t_{ij}^2}{\lambda_j}

    where:
    - :math:`t_{ij}` is the score for sample i on PC j
    - :math:`\lambda_j` is the eigenvalue (variance) of PC j
    - :math:`a` is the number of components

    Critical limit (F-distribution approximation):

    .. math::
        T^2_{crit} = \frac{(n-1) \cdot a}{n-a} \cdot F_{a,n-a,\\alpha}

    where:
    - :math:`n` is the number of samples
    - :math:`F_{a,n-a,\\alpha}` is the F-distribution critical value

    Examples
    --------
    >>> scores = np.array([[1.2, 0.5], [0.8, -0.3], [3.5, 2.1]])
    >>> eigenvalues = np.array([2.5, 1.2])
    >>> t2, limit = calculate_hotelling_t2(scores, eigenvalues, alpha=0.95)
    >>> outliers = t2 > limit
    >>> print(f"Outliers: {np.where(outliers)[0]}")

    References
    ----------
    .. [1] Jackson, J.E. (1991). A User's Guide to Principal Components.
    .. [2] Nomikos & MacGregor (1995). Multivariate SPC charts for
           monitoring batch processes. Technometrics, 37(1), 41-59.
    """
    # Convert to numpy array if DataFrame
    if isinstance(scores, pd.DataFrame):
        scores_array = scores.values
    else:
        scores_array = np.asarray(scores)

    eigenvalues = np.asarray(eigenvalues)

    n_samples, n_components = scores_array.shape

    # Ensure eigenvalues are positive (numerical stability)
    eigenvalues = np.maximum(eigenvalues, 1e-10)

    # Calculate T2 for each sample
    # T2b = Sigma| (t_ij2 / lambda|)
    t2_values = np.sum((scores_array ** 2) / eigenvalues, axis=1)

    # Calculate critical value using F-distribution
    # T2_limit = [(n-1)  * a / (n-a)]  * F(a, n-a, alpha)
    if n_samples <= n_components:
        # Edge case: not enough samples
        t2_limit = 1e10
    else:
        df1 = n_components
        df2 = n_samples - n_components
        f_value = f.ppf(alpha, df1, df2)
        t2_limit = ((n_samples - 1) * n_components / (n_samples - n_components)) * f_value

    return t2_values, t2_limit


def calculate_q_residuals(
    X: Union[np.ndarray, pd.DataFrame],
    scores: Union[np.ndarray, pd.DataFrame],
    loadings: Union[np.ndarray, pd.DataFrame],
    alpha: float = 0.95
) -> Tuple[np.ndarray, float]:
    """
    Calculate Q residuals (SPE - Squared Prediction Error) for PCA.

    Q statistic measures the distance of each sample from the PCA model plane.
    It represents the variation not captured by the retained principal components,
    indicating how well the model describes each sample.

    Parameters
    ----------
    X : np.ndarray or pd.DataFrame
        Original data matrix. Shape: (n_samples, n_features).
    scores : np.ndarray or pd.DataFrame
        PCA scores matrix. Shape: (n_samples, n_components).
    loadings : np.ndarray or pd.DataFrame
        PCA loadings matrix. Shape: (n_features, n_components).
    alpha : float, optional
        Confidence level for critical limit. Default is 0.95.

    Returns
    -------
    q_values : np.ndarray
        Q residual (SPE) for each sample. Shape: (n_samples,).
        Lower values indicate better model fit.
    q_limit : float
        Critical value at specified confidence level.
        Samples with Q > q_limit are considered outliers.

    Notes
    -----
    Q statistic formula (Squared Prediction Error):

    .. math::
        Q_i = SPE_i = \sum_{j=1}^{p} (x_{ij} - \hat{x}_{ij})^2

    where:
    - :math:`x_{ij}` is the original value
    - :math:`\hat{x}_{ij}` is the reconstructed value
    - :math:`\hat{X} = T \cdot P^T` (scores times loadings transpose)
    - :math:`p` is the number of features

    Critical limit (Jackson-Mudholkar approximation):

    .. math::
        Q_{crit} = \\theta_1 \left[1 - \\frac{\\theta_2 h_0^2}{2\\theta_1^2} +
                   z_\\alpha\sqrt{\\frac{2\\theta_2 h_0^2}{\\theta_1}}\\right]^{1/h_0}

    where:
    - :math:`\\theta_i = \sum \\lambda_j^i` (sum over residual eigenvalues)
    - :math:`h_0 = 1 - \\frac{2\\theta_1\\theta_3}{3\\theta_2^2}`
    - :math:`z_\\alpha` is the normal quantile at confidence level :math:`\\alpha`

    Examples
    --------
    >>> X = np.random.randn(100, 50)
    >>> # After PCA...
    >>> X_recon = scores @ loadings.T
    >>> q, limit = calculate_q_residuals(X, scores, loadings, alpha=0.95)
    >>> outliers = q > limit

    References
    ----------
    .. [1] Jackson, J.E. & Mudholkar, G.S. (1979). Control procedures for
           residuals associated with principal component analysis.
           Technometrics, 21(3), 341-349.
    .. [2] Wise et al. (2006). Chemometrics with PCA.
    """
    # Convert to numpy arrays
    if isinstance(X, pd.DataFrame):
        X_array = X.values
    else:
        X_array = np.asarray(X)

    if isinstance(scores, pd.DataFrame):
        scores_array = scores.values
    else:
        scores_array = np.asarray(scores)

    if isinstance(loadings, pd.DataFrame):
        loadings_array = loadings.values
    else:
        loadings_array = np.asarray(loadings)

    # Reconstruct data: X_reconstructed = scores @ loadings.T
    X_reconstructed = scores_array @ loadings_array.T

    # Calculate residuals
    residuals = X_array - X_reconstructed

    # Q statistic = sum of squared residuals for each sample
    q_values = np.sum(residuals ** 2, axis=1)

    # Calculate critical value using Jackson-Mudholkar approximation
    # For this, we need the residual eigenvalues
    # Simplified approach: use chi-square approximation
    n_samples, n_features = X_array.shape
    n_components = scores_array.shape[1]

    # Degrees of freedom for residual space
    df_residual = n_features - n_components

    if df_residual > 0:
        # Estimate mean and variance of Q from data
        q_mean = np.mean(q_values)
        q_var = np.var(q_values)

        # Chi-square approximation
        g = q_var / (2 * q_mean) if q_mean > 0 else 1
        h = (2 * q_mean ** 2) / q_var if q_var > 0 else df_residual

        q_limit = g * chi2.ppf(alpha, h)
    else:
        q_limit = 0.0

    return q_values, q_limit


def calculate_contributions(
    loadings: Union[np.ndarray, pd.DataFrame],
    explained_variance_ratio: np.ndarray,
    n_components: Optional[int] = None,
    normalize: bool = True
) -> pd.DataFrame:
    """
    Calculate variable contributions to total variance explained.

    Quantifies how much each variable contributes to the variance captured
    by the retained principal components. Useful for identifying the most
    important variables in the model.

    Parameters
    ----------
    loadings : np.ndarray or pd.DataFrame
        PCA loadings. Shape: (n_features, n_components).
        Each column represents loadings for one PC.
    explained_variance_ratio : np.ndarray
        Proportion of variance explained by each PC.
        Shape: (n_components,).
    n_components : int, optional
        Number of components to include in contribution calculation.
        If None, uses all available components. Default is None.
    normalize : bool, optional
        Whether to return contributions as percentages summing to 100.
        Default is True.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - 'Variable': variable names (index from loadings if DataFrame)
        - 'Contribution_%': contribution percentage
        - 'PC{i}_Loading2_x_Var': individual PC contributions (optional)

    Notes
    -----
    Contribution formula:

    .. math::
        C_j = \sum_{i=1}^{a} (p_{ji}^2 \\times \\lambda_i)

    where:
    - :math:`C_j` is the contribution of variable j
    - :math:`p_{ji}` is the loading of variable j on PC i
    - :math:`\\lambda_i` is the variance explained by PC i
    - :math:`a` is the number of retained components

    Normalized contribution (percentage):

    .. math::
        C_j^{\\%} = \\frac{C_j}{\sum_{k=1}^{p} C_k} \\times 100

    Examples
    --------
    >>> loadings = pd.DataFrame({'PC1': [0.8, 0.5, 0.2],
    ...                           'PC2': [0.1, 0.6, 0.9]},
    ...                          index=['var1', 'var2', 'var3'])
    >>> var_ratio = np.array([0.6, 0.3])
    >>> contrib = calculate_contributions(loadings, var_ratio, n_components=2)
    >>> print(contrib)  # Shows percentage contribution of each variable

    References
    ----------
    .. [1] Wold et al. (1987). Principal Component Analysis.
    .. [2] Jackson, J.E. (1991). A User's Guide to Principal Components.
    """
    # Convert to arrays
    is_dataframe = isinstance(loadings, pd.DataFrame)

    if is_dataframe:
        loadings_array = loadings.values
        var_names = loadings.index.tolist()
        pc_names = loadings.columns.tolist()
    else:
        loadings_array = np.asarray(loadings)
        var_names = [f'Var{i+1}' for i in range(loadings_array.shape[0])]
        pc_names = [f'PC{i+1}' for i in range(loadings_array.shape[1])]

    n_features, total_components = loadings_array.shape

    # Determine number of components to use
    if n_components is None:
        n_components = total_components
    else:
        n_components = min(n_components, total_components)

    # Extract relevant loadings and variance ratios
    loadings_subset = loadings_array[:, :n_components]
    variance_subset = explained_variance_ratio[:n_components]

    # Calculate weighted contributions
    # Contribution = Sigma(loading2  * variance_explained)
    contributions = np.zeros(n_features)

    # Store individual PC contributions for detailed output
    pc_contributions = {}

    for i in range(n_components):
        pc_contrib = (loadings_subset[:, i] ** 2) * variance_subset[i]
        contributions += pc_contrib
        pc_contributions[f'{pc_names[i]}_Loading2_x_Var'] = pc_contrib

    # Normalize to percentage if requested
    if normalize:
        total = np.sum(contributions)
        if total > 0:
            contributions_pct = (contributions / total) * 100
        else:
            contributions_pct = contributions
    else:
        contributions_pct = contributions

    # Create DataFrame with detailed information
    contrib_df = pd.DataFrame({
        'Variable': var_names,
        'Contribution_%': contributions_pct
    })

    # Add individual PC contributions
    for pc_name, pc_contrib in pc_contributions.items():
        contrib_df[pc_name] = pc_contrib

    # Add cumulative percentage
    sorted_idx = np.argsort(contributions_pct)[::-1]
    sorted_contributions = contributions_pct[sorted_idx]
    cumulative = np.zeros(n_features)
    cumulative[sorted_idx] = np.cumsum(sorted_contributions)
    contrib_df['Cumulative_%'] = cumulative

    return contrib_df


def calculate_leverage(
    scores: Union[np.ndarray, pd.DataFrame]
) -> np.ndarray:
    """
    Calculate leverage (hat matrix diagonal) for PCA samples.

    Leverage measures the influence of each sample on the PCA model.
    High leverage samples have unusual score patterns and can strongly
    influence the model parameters.

    Parameters
    ----------
    scores : np.ndarray or pd.DataFrame
        PCA scores matrix. Shape: (n_samples, n_components).

    Returns
    -------
    np.ndarray
        Leverage values for each sample. Shape: (n_samples,).
        Higher values indicate greater influence on the model.

    Notes
    -----
    Leverage formula:

    .. math::
        h_{ii} = t_i^T (T^T T)^{-1} t_i

    where:
    - :math:`h_{ii}` is the leverage for sample i
    - :math:`t_i` is the score vector for sample i
    - :math:`T` is the scores matrix

    Typical threshold for high leverage:

    .. math::
        h_{threshold} = \\frac{2a}{n} \\text{ or } \\frac{3a}{n}

    where :math:`a` is the number of components and :math:`n` is the number of samples.

    Examples
    --------
    >>> scores = np.array([[1.2, 0.5], [0.8, -0.3], [3.5, 2.1]])
    >>> leverage = calculate_leverage(scores)
    >>> threshold = 2 * scores.shape[1] / scores.shape[0]
    >>> high_leverage = leverage > threshold

    References
    ----------
    .. [1] Jackson, J.E. (1991). A User's Guide to Principal Components.
    .. [2] Hoaglin & Welsch (1978). The hat matrix in regression and ANOVA.
    """
    # Convert to numpy array
    if isinstance(scores, pd.DataFrame):
        scores_array = scores.values
    else:
        scores_array = np.asarray(scores)

    n_samples, n_components = scores_array.shape

    # Calculate T^T T
    TtT = scores_array.T @ scores_array

    # Calculate (T^T T)^{-1}
    try:
        TtT_inv = np.linalg.inv(TtT)
    except np.linalg.LinAlgError:
        # Use pseudo-inverse if matrix is singular
        TtT_inv = np.linalg.pinv(TtT)

    # Calculate leverage for each sample
    # h_i = t_i^T (T^T T)^{-1} t_i
    leverage = np.zeros(n_samples)
    for i in range(n_samples):
        t_i = scores_array[i, :]
        leverage[i] = t_i @ TtT_inv @ t_i

    return leverage


def cross_validate_pca(
    X: Union[np.ndarray, pd.DataFrame],
    max_components: int,
    n_folds: int = 7,
    center: bool = True,
    scale: bool = False
) -> Dict[str, Any]:
    """
    Perform cross-validation for PCA model selection.

    Uses k-fold cross-validation to determine the optimal number of components
    based on predictive ability (Q2) and root mean squared error (RMSECV).

    Parameters
    ----------
    X : np.ndarray or pd.DataFrame
        Original data matrix. Shape: (n_samples, n_features).
    max_components : int
        Maximum number of components to test.
    n_folds : int, optional
        Number of folds for cross-validation. Default is 7.
    center : bool, optional
        Whether to center the data. Default is True.
    scale : bool, optional
        Whether to scale the data. Default is False.

    Returns
    -------
    dict
        Dictionary containing cross-validation results:

        - 'n_components' : np.ndarray
            Array of component numbers tested
        - 'Q2' : np.ndarray
            Q2 (predictive ability) for each number of components
        - 'RMSECV' : np.ndarray
            Root Mean Squared Error of Cross-Validation
        - 'PRESS' : np.ndarray
            Predicted Residual Error Sum of Squares
        - 'optimal_components' : int
            Optimal number of components (maximum Q2)

    Notes
    -----
    Q2 (predictive ability):

    .. math::
        Q^2 = 1 - \\frac{PRESS}{TSS}

    where:
    - PRESS = Predicted Residual Error Sum of Squares
    - TSS = Total Sum of Squares

    RMSECV (Root Mean Squared Error of Cross-Validation):

    .. math::
        RMSECV = \sqrt{\\frac{PRESS}{n \\times p}}

    Cross-validation procedure:
    1. Split data into k folds
    2. For each fold:
       - Train PCA on k-1 folds
       - Predict left-out fold
       - Calculate prediction error
    3. Sum errors across all folds (PRESS)
    4. Calculate Q2 and RMSECV

    Examples
    --------
    >>> X = np.random.randn(100, 50)
    >>> cv_results = cross_validate_pca(X, max_components=10, n_folds=7)
    >>> print(f"Optimal components: {cv_results['optimal_components']}")
    >>> print(f"Q2 values: {cv_results['Q2']}")

    References
    ----------
    .. [1] Wold, S. (1978). Cross-validatory estimation of the number of
           components in factor and principal components models.
    .. [2] Eastment & Krzanowski (1982). Cross-validatory choice of the
           number of components from a principal component analysis.
    """
    # Convert to numpy array
    if isinstance(X, pd.DataFrame):
        X_array = X.values
    else:
        X_array = np.asarray(X)

    n_samples, n_features = X_array.shape

    # Limit max components
    max_components = min(max_components, n_samples - 1, n_features)

    # Initialize results storage
    component_range = np.arange(1, max_components + 1)
    press_values = np.zeros(max_components)

    # K-fold cross-validation
    fold_size = n_samples // n_folds

    for n_comp_idx, n_comp in enumerate(component_range):
        fold_press = 0

        for fold in range(n_folds):
            # Define test indices for this fold
            test_start = fold * fold_size
            test_end = test_start + fold_size if fold < n_folds - 1 else n_samples
            test_indices = np.arange(test_start, test_end)
            train_indices = np.setdiff1d(np.arange(n_samples), test_indices)

            # Split data
            X_train = X_array[train_indices]
            X_test = X_array[test_indices]

            # Center/scale training data
            if center:
                train_mean = np.mean(X_train, axis=0)
                X_train_proc = X_train - train_mean
                X_test_proc = X_test - train_mean
            else:
                X_train_proc = X_train
                X_test_proc = X_test

            if scale:
                train_std = np.std(X_train_proc, axis=0)
                train_std[train_std == 0] = 1  # Avoid division by zero
                X_train_proc = X_train_proc / train_std
                X_test_proc = X_test_proc / train_std

            # Fit PCA on training data
            pca = PCA(n_components=n_comp)
            pca.fit(X_train_proc)

            # Project test data and reconstruct
            scores_test = pca.transform(X_test_proc)
            X_test_reconstructed = scores_test @ pca.components_

            # Calculate prediction error for this fold
            fold_error = np.sum((X_test_proc - X_test_reconstructed) ** 2)
            fold_press += fold_error

        press_values[n_comp_idx] = fold_press

    # Calculate Q2 and RMSECV
    # Total sum of squares (TSS)
    if center:
        X_centered = X_array - np.mean(X_array, axis=0)
    else:
        X_centered = X_array

    tss = np.sum(X_centered ** 2)

    # Q2 = 1 - PRESS/TSS
    q2_values = 1 - (press_values / tss)

    # RMSECV = sqrt(PRESS / (n * p))
    rmsecv_values = np.sqrt(press_values / (n_samples * n_features))

    # Find optimal number of components (maximum Q2)
    optimal_idx = np.argmax(q2_values)
    optimal_components = component_range[optimal_idx]

    return {
        'n_components': component_range,
        'Q2': q2_values,
        'RMSECV': rmsecv_values,
        'PRESS': press_values,
        'optimal_components': optimal_components
    }
