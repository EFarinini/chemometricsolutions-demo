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

    CRITICAL: NIPALS eigenvalues use population variance (t't/n), but T²
    requires sample variance (t't/(n-1)). We apply the correction factor
    to match R/CAT and standard statistical references.

    Parameters
    ----------
    scores : np.ndarray or pd.DataFrame
        PCA scores matrix of shape (n_samples, n_components).
        Each row is a sample projected into PC space.
    eigenvalues : np.ndarray
        Eigenvalues from NIPALS (population variance: t't/n).
        Shape: (n_components,).
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
    **NIPALS vs Sample Variance Correction:**

    NIPALS computes: λ_NIPALS = t't / n  (population variance)

    T² requires:     λ_sample = t't / (n-1)  (sample variance)

    Correction: λ_sample = λ_NIPALS × n/(n-1)

    This ensures T² values match R/CAT exactly. Without this correction,
    Python T² values would be systematically lower by factor (n-1)/n.

    **T2 statistic formula:**

    .. math::
        T^2_i = \sum_{j=1}^{a} \frac{t_{ij}^2}{\lambda_{sample,j}}

    where:
    - :math:`t_{ij}` is the score for sample i on PC j
    - :math:`\lambda_{sample,j}` is the sample variance eigenvalue of PC j
    - :math:`a` is the number of components

    **Critical limit (F-distribution approximation):**

    .. math::
        T^2_{crit} = \frac{(n-1) \cdot a}{n-a} \cdot F_{a,n-a,\\alpha}

    where:
    - :math:`n` is the number of samples
    - :math:`F_{a,n-a,\\alpha}` is the F-distribution critical value

    Examples
    --------
    >>> scores = np.array([[1.2, 0.5], [0.8, -0.3], [3.5, 2.1]])
    >>> eigenvalues_nipals = np.array([2.5, 1.2])  # From NIPALS
    >>> t2, limit = calculate_hotelling_t2(scores, eigenvalues_nipals, alpha=0.95)
    >>> outliers = t2 > limit
    >>> print(f"Outliers: {np.where(outliers)[0]}")

    References
    ----------
    .. [1] Jackson, J.E. (1991). A User's Guide to Principal Components.
    .. [2] Nomikos & MacGregor (1995). Multivariate SPC charts for
           monitoring batch processes. Technometrics, 37(1), 41-59.
    .. [3] R/CAT: PCA_t2vsq_Dataset.r (uses sDev^2 = sample variance)
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

    # === CRITICAL CORRECTION ===
    # Convert NIPALS eigenvalues (population variance) to sample variance
    # λ_sample = λ_NIPALS × n/(n-1)
    # This correction factor ensures T² values match R/CAT standard
    if n_samples > 1:
        eigenvalues_corrected = eigenvalues * (n_samples / (n_samples - 1))
    else:
        eigenvalues_corrected = eigenvalues

    # === T² CALCULATION (R/CAT STANDARD) ===
    # T²_i = Σ(t_ij² / λ_sample_j)
    t2_values = np.sum((scores_array ** 2) / eigenvalues_corrected, axis=1)

    # Calculate critical value using F-distribution
    # T2_limit = [(n-1)  * a / (n-a)]  * F(a, n-a, alpha)
    if n_samples <= n_components:
        # Edge case: not enough samples
        t2_limit = 1e10
    else:
        df1 = n_components  # a = number of PCs
        df2 = n_samples - n_components  # n - a

        # Get F critical value at specified confidence level
        f_value = f.ppf(alpha, df1, df2)

        # Apply Hotelling T² limit formula with safe division
        # Ensure denominator is at least 1 to avoid division issues
        t2_limit = ((n_samples - 1) * n_components / max(1, n_samples - n_components)) * f_value

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
    Q statistic formula (from R script PCA_t2vsq_Dataset.r):

    .. math::
        Q_i = diag(X \\cdot M_Q \\cdot X^T)

    where:
    - :math:`M_Q = I - P \\cdot P^T` (residual projection matrix)
    - :math:`P` is the loadings matrix (n_features × n_components)
    - :math:`I` is the identity matrix
    - :math:`X` is the centered/scaled data matrix

    This is mathematically equivalent to:

    .. math::
        Q_i = SPE_i = \sum_{j=1}^{p} (x_{ij} - \hat{x}_{ij})^2

    where :math:`\hat{x}_{ij}` is the reconstructed value from :math:`T \\cdot P^T`.

    Critical limit (log-normal approximation from R script):

    .. math::
        Q_{crit} = 10^{\\mu_{log} + t_{\\alpha,n-1} \\cdot \\sigma_{log}}

    where:
    - :math:`\\mu_{log}` = mean of log10(Q) values
    - :math:`\\sigma_{log}` = standard deviation of log10(Q) values
    - :math:`t_{\\alpha,n-1}` is the t-distribution critical value

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

    # === CORRECT Q CALCULATION (from R script PCA_t2vsq_Dataset.r) ===
    # Q = diag(X * MQ * X^T)
    # where MQ = I - P * P^T (residual projection matrix)

    n_samples, n_features = X_array.shape
    n_components = scores_array.shape[1]

    # Create residual projection matrix: MQ = I - P * P^T
    # This projects data onto the residual space (not explained by PCA model)
    I = np.eye(n_features)
    MQ = I - (loadings_array @ loadings_array.T)

    # Calculate Q for each sample: Q = diag(X * MQ * X^T)
    # This is the squared residual distance from PCA model plane
    q_values = np.diag(X_array @ MQ @ X_array.T)

    # === CORRECT Q_LIMIT using log-normal approximation (from R script) ===
    # Q_limit = 10^(mean(log10(Q)) + t(alpha, n-1) * sd(log10(Q)))
    # This uses a log-normal distribution fit to the Q residuals

    # Calculate log-transformed Q values
    q_log = np.log10(q_values + 1e-10)  # Add small value to avoid log(0)
    q_mean_log = np.mean(q_log)
    q_std_log = np.std(q_log, ddof=1)  # Use sample std (ddof=1)

    # Get t-distribution critical value
    t_val = t.ppf(alpha, n_samples - 1)

    # Calculate Q limit in log space, then convert back to original scale
    q_limit = 10 ** (q_mean_log + t_val * q_std_log)

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


def calculate_variable_variance_explained(
    X_preprocessed: Union[pd.DataFrame, np.ndarray],
    scores: Union[pd.DataFrame, np.ndarray],
    loadings: Union[pd.DataFrame, np.ndarray],
    n_components: int
) -> pd.DataFrame:
    """
    Calculate fraction of variance explained by each variable.

    Equivalent to R chemometrics::pcaVarexpl()

    This implements the EXACT R formula:
        varexpl = 1 - Σ(residuals²) / Σ(X²)

    where residuals = X - T[1:a] × P[1:a]ᵀ

    This measures how well each variable can be reconstructed using
    the first n_components principal components.

    Parameters
    ----------
    X_preprocessed : pd.DataFrame or np.ndarray
        Preprocessed (centered/scaled) data matrix.
        Shape: (n_samples, n_variables).

    scores : pd.DataFrame or np.ndarray
        PCA scores matrix (T in PCA notation).
        Shape: (n_samples, n_total_components).

    loadings : pd.DataFrame or np.ndarray
        PCA loadings matrix (P in PCA notation).
        Shape: (n_variables, n_total_components).

    n_components : int
        Number of components to use for reconstruction (a in R formula).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - 'Variable': variable names
        - 'Variance_Explained_Ratio': fraction of variance explained (0-1)

        Sorted by 'Variance_Explained_Ratio' in descending order.

    Notes
    -----
    R chemometrics::pcaVarexpl() formula:

    .. math::
        VarExpl_j = 1 - \\frac{\\sum (X_j - \\hat{X}_j)^2}{\\sum X_j^2}

    where:
    - :math:`X_j` is the preprocessed data for variable j
    - :math:`\\hat{X}_j = T_{1:a} \\cdot P_{j,1:a}^T` is reconstruction with a components
    - :math:`a` is the number of components used

    Interpretation:
    - Close to 1.0: variable is very well represented by the model
    - 0.7-0.9: good representation
    - 0.5-0.7: moderate representation
    - < 0.5: poor representation (may follow different patterns)

    Examples
    --------
    >>> X_preprocessed = ...  # Centered/scaled data
    >>> scores = ...  # T from PCA
    >>> loadings = ...  # P from PCA
    >>> result = calculate_variable_variance_explained(
    ...     X_preprocessed, scores, loadings, n_components=2
    ... )

    References
    ----------
    .. [1] R package 'chemometrics' - pcaVarexpl function
    .. [2] Varmuza & Filzmoser (2009). Introduction to Multivariate
           Statistical Analysis in Chemometrics. CRC Press.
    """
    # Convert inputs to numpy arrays
    if isinstance(X_preprocessed, pd.DataFrame):
        X_array = X_preprocessed.values
        var_names = X_preprocessed.columns.tolist()
    else:
        X_array = np.asarray(X_preprocessed)
        var_names = [f'Var{i+1}' for i in range(X_array.shape[1])]

    if isinstance(scores, pd.DataFrame):
        T_array = scores.values
    else:
        T_array = np.asarray(scores)

    if isinstance(loadings, pd.DataFrame):
        P_array = loadings.values
    else:
        P_array = np.asarray(loadings)

    # EXACT R formula: varexpl = 1 - Σ(residuals²) / Σ(X²)
    #
    # Step 1: Reconstruct data using first n_components
    T_a = T_array[:, :n_components]  # Scores for first a components
    P_a = P_array[:, :n_components]  # Loadings for first a components
    X_reconstructed = T_a @ P_a.T  # Reconstruction: T[1:a] × P[1:a]ᵀ

    # Step 2: Calculate residuals
    residuals = X_array - X_reconstructed

    # Step 3: Calculate unexplained variance per variable (sum over samples)
    unexplained_var = np.sum(residuals ** 2, axis=0)

    # Step 4: Calculate total variance per variable
    total_var = np.sum(X_array ** 2, axis=0)

    # Step 5: Calculate variance explained (R formula)
    # Handle division by zero
    variance_explained = np.zeros(len(var_names))
    non_zero_mask = total_var > 1e-10

    variance_explained[non_zero_mask] = 1 - (
        unexplained_var[non_zero_mask] / total_var[non_zero_mask]
    )

    # Clip to [0, 1] range (negative values can occur due to numerical errors)
    variance_explained = np.clip(variance_explained, 0, 1)

    # Create result DataFrame
    result_df = pd.DataFrame({
        'Variable': var_names,
        'Variance_Explained_Ratio': variance_explained
    })

    # Sort by variance explained (descending)
    result_df = result_df.sort_values('Variance_Explained_Ratio', ascending=False)

    return result_df


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


def calculate_hotelling_t2_matricial(
    X_test: np.ndarray,
    loadings: np.ndarray,
    eigenvalues: np.ndarray,
    X_train_mean: np.ndarray,
    X_train_std: Optional[np.ndarray] = None,
    n_train: int = None
) -> Tuple[np.ndarray, float]:
    """
    Calculate Hotelling's T² statistic using matricial formula for TEST DATA projection.
    
    This function implements the EXACT R/CAT formula from PCA_t2q_Dataset.r:
        T² = diag(X · MT · X')
        where MT = P · diag(1/λ) · P'
    
    This matricial approach correctly handles the centering/scaling transformation
    when projecting new data onto a PCA model trained on different data.
    
    **CRITICAL**: This is ONLY for TEST DATA projection. For training data,
    use the simpler `calculate_hotelling_t2()` function.
    
    Parameters
    ----------
    X_test : np.ndarray
        Test data matrix (ORIGINAL scale, not preprocessed).
        Shape: (n_test_samples, n_features).
    loadings : np.ndarray
        PCA loadings matrix from training model.
        Shape: (n_features, n_components).
    eigenvalues : np.ndarray
        Eigenvalues from training model (sample variance).
        Shape: (n_components,).
    X_train_mean : np.ndarray
        Mean values from training data for centering.
        Shape: (n_features,).
    X_train_std : np.ndarray, optional
        Standard deviations from training data for scaling.
        If None, only centering is applied. Shape: (n_features,).
    n_train : int, optional
        Number of training samples (for limit calculation).
        If None, limit is not calculated.
    
    Returns
    -------
    t2_values : np.ndarray
        T² statistic for each test sample. Shape: (n_test_samples,).
    t2_limit : float
        Critical value at 97.5% confidence (if n_train provided).
        Otherwise returns np.nan.
    
    Notes
    -----
    **R/CAT Matricial Formula (from PCA_t2q_Dataset.r):**
    
    .. code-block:: r
    
        # Lines 29-33 from PCA_t2q_Dataset.r
        if(PCA$center) M <- M - (unity %*% PCA$centered)
        if(PCA$scale) M <- M / (unity %*% PCA$scaled)
        MT <- P %*% (diag(length(L)) * (1/L)) %*% t(P)
        TN <- diag(M %*% MT %*% t(M))
    
    In Python:
    
    .. math::
        T^2_i = diag(X_{test} \\cdot M_T \\cdot X_{test}^T)
        
        M_T = P \\cdot diag(1/\\lambda) \\cdot P^T
    
    where:
    - :math:`X_{test}` is the centered/scaled test data matrix
    - :math:`P` is the loadings matrix (n_features × n_components)
    - :math:`\\lambda` are the eigenvalues (sample variance)
    
    **Why this formula?**
    
    The matricial formula automatically accounts for:
    1. Centering with training mean
    2. Scaling with training std (if applicable)
    3. Projection onto PCA space
    4. Mahalanobis distance calculation
    
    All in one compact operation that matches R/CAT exactly.
    
    **Training vs Test Data:**
    
    - Training: Use `calculate_hotelling_t2(scores, eigenvalues)`
      Simple formula: T² = Σ(t²/λ) works because scores are from the model itself
    
    - Test: Use this function `calculate_hotelling_t2_matricial(X_test, ...)`
      Matricial formula needed because data is projected from external source
    
    Examples
    --------
    >>> # After training PCA model
    >>> X_test = np.array([[1.2, 0.5, 0.8], [0.9, 1.1, 0.7]])
    >>> loadings = pca_model['loadings'][:, :2]  # First 2 PCs
    >>> eigenvalues = pca_model['eigenvalues'][:2]
    >>> X_train_mean = pca_model['means']
    >>> X_train_std = pca_model['stds']  # Or None if not scaled
    >>> 
    >>> t2, limit = calculate_hotelling_t2_matricial(
    ...     X_test, loadings, eigenvalues, X_train_mean, X_train_std, n_train=100
    ... )
    >>> outliers = t2 > limit
    
    References
    ----------
    .. [1] R/CAT: PCA_t2q_Dataset.r, lines 9-33
    .. [2] Jackson, J.E. (1991). A User's Guide to Principal Components
    .. [3] Nomikos & MacGregor (1995). Multivariate SPC charts for monitoring
    """
    # Ensure numpy arrays
    X_test = np.asarray(X_test)
    loadings = np.asarray(loadings)
    eigenvalues = np.asarray(eigenvalues)
    X_train_mean = np.asarray(X_train_mean)
    
    n_test_samples, n_features = X_test.shape
    n_components = loadings.shape[1]
    
    # === STEP 1: Preprocess test data (center/scale with TRAINING statistics) ===
    # R/CAT lines 29-30:
    # if(PCA$center) M <- M - (unity %*% PCA$centered)
    # if(PCA$scale) M <- M / (unity %*% PCA$scaled)
    
    X_test_processed = X_test - X_train_mean  # Center with training mean
    
    if X_train_std is not None:
        # Scale with training std (avoid division by zero)
        std_safe = np.where(X_train_std > 1e-10, X_train_std, 1.0)
        X_test_processed = X_test_processed / std_safe
    
    # === STEP 2: Create T² projection matrix ===
    # R/CAT line 14:
    # MT <- P %*% (diag(length(L)) * (1/L)) %*% t(P)

    # === CRITICAL CORRECTION ===
    # Convert NIPALS eigenvalues (population variance) to sample variance
    # λ_sample = λ_NIPALS × n_train/(n_train-1)
    # This matches the correction in calculate_hotelling_t2()
    if n_train is not None and n_train > 1:
        eigenvalues_corrected = eigenvalues * (n_train / (n_train - 1))
    else:
        eigenvalues_corrected = eigenvalues

    # Ensure eigenvalues are positive (numerical stability)
    eigenvalues_safe = np.maximum(eigenvalues_corrected, 1e-10)

    # Create diagonal matrix: diag(1/λ_sample)
    inv_eigenvalues_diag = np.diag(1.0 / eigenvalues_safe)
    
    # MT = P · diag(1/λ) · P'
    MT = loadings @ inv_eigenvalues_diag @ loadings.T
    
    # === STEP 3: Calculate T² for each test sample ===
    # R/CAT line 33:
    # TN <- diag(M %*% MT %*% t(M))
    
    # T² = diag(X · MT · X')
    # For each sample i: T²_i = X_i · MT · X_i'
    t2_values = np.diag(X_test_processed @ MT @ X_test_processed.T)
    
    # === STEP 4: Calculate critical limit (if training size provided) ===
    # R/CAT line 18:
    # Tlim <- (n-1)*ncp/(n-ncp)*qf(0.95,ncp,n-ncp)
    
    if n_train is not None and n_train > n_components:
        alpha = 0.975  # 97.5% confidence level
        df1 = n_components
        df2 = n_train - n_components
        f_value = f.ppf(alpha, df1, df2)
        t2_limit = ((n_train - 1) * n_components / (n_train - n_components)) * f_value
    else:
        t2_limit = np.nan
    
    return t2_values, t2_limit
