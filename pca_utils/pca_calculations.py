"""
PCA Calculation Functions

Core computation functions for Principal Component Analysis (PCA).
Includes standard PCA decomposition and Varimax rotation.
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional, Dict, Any, Union


def compute_pca(
    X: Union[pd.DataFrame, np.ndarray],
    n_components: int,
    center: bool = True,
    scale: bool = False
) -> Dict[str, Any]:
    """
    Perform Principal Component Analysis on input data.

    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
        Input data matrix of shape (n_samples, n_features).
        Rows represent samples/observations, columns represent variables/features.
    n_components : int
        Number of principal components to compute.
    center : bool, optional
        Whether to center the data (mean centering). Default is True.
    scale : bool, optional
        Whether to scale the data to unit variance (autoscaling). Default is False.

    Returns
    -------
    dict
        Dictionary containing PCA results with the following keys:

        - 'model' : sklearn.decomposition.PCA
            Fitted PCA model object
        - 'scores' : pd.DataFrame
            Principal component scores (n_samples, n_components)
        - 'loadings' : pd.DataFrame
            Principal component loadings (n_features, n_components)
        - 'explained_variance' : np.ndarray
            Variance explained by each component (eigenvalues)
        - 'explained_variance_ratio' : np.ndarray
            Proportion of variance explained by each component
        - 'cumulative_variance' : np.ndarray
            Cumulative proportion of variance explained
        - 'eigenvalues' : np.ndarray
            Eigenvalues (same as explained_variance)
        - 'scaler' : StandardScaler or None
            Scaler object used for preprocessing (if scale=True)
        - 'processed_data' : pd.DataFrame or np.ndarray
            Preprocessed data after centering/scaling

    Examples
    --------
    >>> import pandas as pd
    >>> data = pd.DataFrame(np.random.randn(100, 50))
    >>> results = compute_pca(data, n_components=5, center=True, scale=True)
    >>> print(results['explained_variance_ratio'])
    >>> print(results['scores'].shape)  # (100, 5)

    Notes
    -----
    - Centering is standard for PCA analysis
    - Scaling (autoscaling) is recommended when variables have different units
    - The function follows the convention: scores = data @ loadings
    """
    # Input validation
    if X is None:
        raise ValueError("Input data X cannot be None")

    # Store original index and columns if DataFrame
    is_dataframe = isinstance(X, pd.DataFrame)
    if is_dataframe:
        index = X.index
        columns = X.columns
        X_array = X.values
    else:
        X_array = np.asarray(X)
        index = None
        columns = None

    # Validate shape
    if X_array.ndim != 2:
        raise ValueError(f"Input data must be 2-dimensional, got shape {X_array.shape}")

    n_samples, n_features = X_array.shape

    if n_samples < 2:
        raise ValueError(f"Need at least 2 samples for PCA, got {n_samples}")

    if n_features < 1:
        raise ValueError(f"Need at least 1 feature for PCA, got {n_features}")

    # Validate n_components
    if not isinstance(n_components, (int, np.integer)):
        raise TypeError(f"n_components must be an integer, got {type(n_components)}")

    if n_components < 1:
        raise ValueError(f"n_components must be >= 1, got {n_components}")

    if n_components > min(n_samples, n_features):
        raise ValueError(
            f"n_components ({n_components}) cannot exceed min(n_samples, n_features) "
            f"= min({n_samples}, {n_features}) = {min(n_samples, n_features)}"
        )

    # Check for missing values
    if np.any(np.isnan(X_array)):
        raise ValueError("Input data contains NaN values. Please remove or impute missing values before PCA.")

    # Preprocessing
    scaler = None
    if scale:
        scaler = StandardScaler()
        X_processed = scaler.fit_transform(X_array)
    elif center:
        X_processed = X_array - np.mean(X_array, axis=0)
    else:
        X_processed = X_array.copy()

    # Perform PCA
    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(X_processed)

    # Get loadings (components transposed)
    loadings = pca.components_.T

    # Calculate variance metrics
    explained_variance = pca.explained_variance_
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance_ratio)

    # Prepare output with proper naming
    pc_names = [f'PC{i+1}' for i in range(n_components)]

    if is_dataframe:
        scores_df = pd.DataFrame(scores, columns=pc_names, index=index)
        loadings_df = pd.DataFrame(loadings, columns=pc_names, index=columns)
        processed_df = pd.DataFrame(X_processed, columns=columns, index=index)
    else:
        scores_df = scores
        loadings_df = loadings
        processed_df = X_processed

    # Compile results
    results = {
        'model': pca,
        'scores': scores_df,
        'loadings': loadings_df,
        'explained_variance': explained_variance,
        'explained_variance_ratio': explained_variance_ratio,
        'cumulative_variance': cumulative_variance,
        'eigenvalues': explained_variance,
        'scaler': scaler,
        'processed_data': processed_df
    }

    return results


def varimax_rotation(
    loadings: Union[np.ndarray, pd.DataFrame],
    max_iter: int = 100,
    tol: float = 1e-6
) -> Tuple[Union[np.ndarray, pd.DataFrame], int]:
    """
    Perform Varimax rotation on PCA loadings matrix.

    Varimax rotation is an orthogonal rotation method that maximizes the variance
    of squared loadings within each component, leading to simpler and more
    interpretable factor structure.

    Parameters
    ----------
    loadings : np.ndarray or pd.DataFrame
        Loading matrix of shape (n_features, n_components).
        Each column represents one principal component's loadings.
    max_iter : int, optional
        Maximum number of iterations for convergence. Default is 100.
    tol : float, optional
        Convergence tolerance. Algorithm stops when the change in loadings
        between iterations is less than this value. Default is 1e-6.

    Returns
    -------
    rotated_loadings : np.ndarray or pd.DataFrame
        Rotated loading matrix with the same shape as input.
        Type matches input (DataFrame if input is DataFrame, else ndarray).
    iterations : int
        Number of iterations performed until convergence.

    Examples
    --------
    >>> loadings = np.random.randn(50, 3)
    >>> rotated, n_iter = varimax_rotation(loadings)
    >>> print(f"Converged in {n_iter} iterations")
    >>> print(rotated.shape)  # (50, 3)

    Notes
    -----
    - Varimax rotation seeks a "simple structure" where each variable loads
      highly on few components and weakly on others
    - The rotation is orthogonal, preserving orthogonality of components
    - After rotation, components should be sorted by explained variance
    - This implementation uses pairwise angle search for robustness

    References
    ----------
    Kaiser, H. F. (1958). The varimax criterion for analytic rotation
    in factor analysis. Psychometrika, 23(3), 187-200.
    """
    # Input validation
    if loadings is None:
        raise ValueError("Loadings matrix cannot be None")

    # Handle DataFrame input
    is_dataframe = isinstance(loadings, pd.DataFrame)
    if is_dataframe:
        index = loadings.index
        columns = loadings.columns
        loadings_array = loadings.values
    else:
        loadings_array = np.asarray(loadings)

    # Validate shape
    if loadings_array.ndim != 2:
        raise ValueError(f"Loadings must be 2-dimensional, got shape {loadings_array.shape}")

    p, k = loadings_array.shape  # p = n_features, k = n_components

    if p < 2:
        raise ValueError(f"Need at least 2 features for Varimax rotation, got {p}")

    if k < 2:
        raise ValueError(f"Need at least 2 components for Varimax rotation, got {k}")

    # Validate parameters
    if not isinstance(max_iter, (int, np.integer)) or max_iter < 1:
        raise ValueError(f"max_iter must be a positive integer, got {max_iter}")

    if tol <= 0:
        raise ValueError(f"Tolerance must be positive, got {tol}")

    rotated_loadings = loadings_array.copy()

    # Iterative rotation algorithm
    converged = False
    iteration = 0

    while not converged and iteration < max_iter:
        prev_loadings = rotated_loadings.copy()

        # Pairwise rotations between all component pairs
        for i in range(k - 1):
            for j in range(i + 1, k):
                # Extract two columns for rotation
                lo = rotated_loadings[:, [i, j]]

                # Find optimal rotation angle
                best_angle = 0
                best_criterion = np.sum(lo**4)

                # Search for best rotation angle (-90 to 90 degrees)
                for angle_deg in np.arange(-90, 90.1, 0.1):
                    angle_rad = angle_deg * np.pi / 180

                    # Create rotation matrix
                    cos_a = np.cos(angle_rad)
                    sin_a = np.sin(angle_rad)
                    rotation_matrix = np.array([
                        [cos_a, -sin_a],
                        [sin_a, cos_a]
                    ])

                    # Apply rotation
                    rotated_pair = lo @ rotation_matrix
                    criterion = np.sum(rotated_pair**4)

                    if criterion > best_criterion:
                        best_criterion = criterion
                        best_angle = angle_deg

                # Apply best rotation if improvement found
                if best_angle != 0:
                    angle_rad = best_angle * np.pi / 180
                    cos_a = np.cos(angle_rad)
                    sin_a = np.sin(angle_rad)
                    rotation_matrix = np.array([
                        [cos_a, -sin_a],
                        [sin_a, cos_a]
                    ])

                    rotated_loadings[:, [i, j]] = lo @ rotation_matrix

        # Check convergence
        if np.allclose(rotated_loadings, prev_loadings, atol=tol):
            converged = True

        iteration += 1

    # Convert back to DataFrame if input was DataFrame
    if is_dataframe:
        rotated_loadings = pd.DataFrame(
            rotated_loadings,
            index=index,
            columns=columns
        )

    return rotated_loadings, iteration


def calculate_explained_variance(eigenvalues: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Calculate variance explained metrics from eigenvalues.

    Parameters
    ----------
    eigenvalues : np.ndarray
        Array of eigenvalues (variance explained by each component).
        Should be in descending order (largest eigenvalue first).

    Returns
    -------
    dict
        Dictionary containing variance metrics:
        - 'explained_variance' : np.ndarray - Original eigenvalues
        - 'explained_variance_ratio' : np.ndarray - Proportion of total variance
        - 'cumulative_variance' : np.ndarray - Cumulative proportion
        - 'total_variance' : float - Sum of all eigenvalues

    Examples
    --------
    >>> eigenvalues = np.array([5.2, 3.1, 1.8, 0.9, 0.5])
    >>> variance_metrics = calculate_explained_variance(eigenvalues)
    >>> print(variance_metrics['explained_variance_ratio'])

    Notes
    -----
    Explained variance ratio = eigenvalue / sum(all eigenvalues)
    """
    eigenvalues = np.asarray(eigenvalues)
    total_variance = np.sum(eigenvalues)

    if total_variance > 0:
        explained_variance_ratio = eigenvalues / total_variance
    else:
        explained_variance_ratio = np.zeros_like(eigenvalues)

    cumulative_variance = np.cumsum(explained_variance_ratio)

    return {
        'explained_variance': eigenvalues,
        'explained_variance_ratio': explained_variance_ratio,
        'cumulative_variance': cumulative_variance,
        'total_variance': total_variance
    }
