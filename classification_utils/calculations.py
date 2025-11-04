"""
Classification Calculations
============================

Core classification algorithms including:
- LDA (Linear Discriminant Analysis)
- QDA (Quadratic Discriminant Analysis)
- kNN (k-Nearest Neighbors with multiple distance metrics)
- SIMCA (Soft Independent Modeling of Class Analogy)
- UNEQ (Class modeling with unequal class dispersions)
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Union, Tuple, List
from scipy.stats import f as f_dist, beta as beta_dist
from scipy.spatial.distance import cdist
from .config import (
    DEFAULT_CV_FOLDS,
    DEFAULT_K_MAX,
    DEFAULT_CONFIDENCE_LEVEL,
    NIPALS_MAX_ITER,
    NIPALS_TOLERANCE
)


# ============================================================================
# LDA - Linear Discriminant Analysis
# ============================================================================

def fit_lda(
    X: np.ndarray,
    y: np.ndarray
) -> Dict[str, any]:
    """
    Fit Linear Discriminant Analysis model.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Training data
    y : ndarray of shape (n_samples,)
        Class labels

    Returns
    -------
    dict
        LDA model containing:
        - means: class means
        - cov: pooled covariance matrix
        - inv_cov: inverse of pooled covariance
        - classes: unique class labels
        - priors: class prior probabilities
    """
    classes = np.unique(y)
    n_classes = len(classes)
    n_samples, n_features = X.shape

    # Calculate class means
    means = np.zeros((n_classes, n_features))
    for i, cls in enumerate(classes):
        means[i] = X[y == cls].mean(axis=0)

    # Calculate pooled covariance matrix
    pooled_cov = np.zeros((n_features, n_features))
    for cls in classes:
        X_cls = X[y == cls]
        n_cls = len(X_cls)
        if n_cls > 1:
            # Covariance for this class
            cov_cls = np.cov(X_cls, rowvar=False, bias=False)
            pooled_cov += cov_cls * (n_cls - 1)

    pooled_cov /= (n_samples - n_classes)

    # Regularization for numerical stability
    pooled_cov += np.eye(n_features) * 1e-10

    # Calculate inverse
    try:
        inv_cov = np.linalg.inv(pooled_cov)
    except np.linalg.LinAlgError:
        # Use pseudo-inverse if singular
        inv_cov = np.linalg.pinv(pooled_cov)

    # Calculate priors
    priors = np.array([np.sum(y == cls) / n_samples for cls in classes])

    return {
        'means': means,
        'cov': pooled_cov,
        'inv_cov': inv_cov,
        'classes': classes,
        'priors': priors,
        'n_features': n_features,
        'n_classes': n_classes
    }


def predict_lda(
    X: np.ndarray,
    model: Dict[str, any]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Predict classes using LDA model.

    Parameters
    ----------
    X : ndarray
        Data to predict
    model : dict
        LDA model from fit_lda

    Returns
    -------
    predictions : ndarray
        Predicted class labels
    distances : ndarray of shape (n_samples, n_classes)
        Mahalanobis distances to each class
    """
    n_samples = X.shape[0]
    n_classes = len(model['classes'])

    # Calculate Mahalanobis distances to each class
    distances = np.zeros((n_samples, n_classes))
    for i, cls in enumerate(model['classes']):
        diff = X - model['means'][i]
        distances[:, i] = np.sum(diff @ model['inv_cov'] * diff, axis=1)

    # Predict class with minimum distance
    predictions = model['classes'][np.argmin(distances, axis=1)]

    return predictions, distances


def predict_lda_detailed(
    X: np.ndarray,
    model: Dict[str, any]
) -> Dict[str, np.ndarray]:
    """
    Predict classes using LDA model with detailed output including posterior probabilities.

    Uses discriminant function approach with log-sum-exp trick for numerical stability.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Data to predict
    model : dict
        LDA model from fit_lda

    Returns
    -------
    dict
        Dictionary containing:
        - predicted_classes : ndarray (n_samples,)
            Predicted class labels
        - discriminant_scores : ndarray (n_samples, n_classes)
            Discriminant function values for each class
        - posterior_probabilities : ndarray (n_samples, n_classes)
            Posterior probabilities for each class (softmax of discriminant scores)
        - prediction_confidence : ndarray (n_samples,)
            Maximum posterior probability for each sample
        - mahalanobis_distances : ndarray (n_samples, n_classes)
            Mahalanobis distances to each class mean
        - classes : ndarray
            Class labels

    Reference
    ---------
    CL_prediction_LDA.r
    """
    n_samples = X.shape[0]
    n_classes = len(model['classes'])
    classes = model['classes']

    # Calculate Mahalanobis distances to each class
    distances = np.zeros((n_samples, n_classes))
    discriminant_scores = np.zeros((n_samples, n_classes))

    for i, cls in enumerate(classes):
        diff = X - model['means'][i]
        # Mahalanobis distance squared
        md_squared = np.sum(diff @ model['inv_cov'] * diff, axis=1)
        distances[:, i] = md_squared

        # Discriminant function: -0.5 * d² + log(prior)
        # Equivalent to: z_j = X @ coeff_j where coeff_j = inv_cov @ mean_j
        discriminant_scores[:, i] = -0.5 * md_squared + np.log(model['priors'][i] + 1e-10)

    # Compute posterior probabilities using log-sum-exp trick for numerical stability
    # P(class_j | x) = exp(z_j) / sum_k(exp(z_k))
    max_scores = np.max(discriminant_scores, axis=1, keepdims=True)
    exp_scores = np.exp(discriminant_scores - max_scores)
    posterior_probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    # Predicted class: argmax of posterior probabilities (or discriminant scores)
    predicted_indices = np.argmax(posterior_probs, axis=1)
    predicted_classes = classes[predicted_indices]

    # Prediction confidence: maximum posterior probability
    prediction_confidence = np.max(posterior_probs, axis=1)

    return {
        'predicted_classes': predicted_classes,
        'discriminant_scores': discriminant_scores,
        'posterior_probabilities': posterior_probs,
        'prediction_confidence': prediction_confidence,
        'mahalanobis_distances': np.sqrt(distances),
        'mahalanobis_distances_squared': distances,
        'classes': classes
    }


# ============================================================================
# QDA - Quadratic Discriminant Analysis
# ============================================================================

def fit_qda(
    X: np.ndarray,
    y: np.ndarray
) -> Dict[str, any]:
    """
    Fit Quadratic Discriminant Analysis model.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Training data
    y : ndarray of shape (n_samples,)
        Class labels

    Returns
    -------
    dict
        QDA model containing means, covariances, and inverse covariances for each class
    """
    classes = np.unique(y)
    n_classes = len(classes)
    n_features = X.shape[1]

    means = []
    covs = []
    inv_covs = []
    priors = []

    for cls in classes:
        X_cls = X[y == cls]
        n_cls = len(X_cls)

        # Class mean
        mean_cls = X_cls.mean(axis=0)
        means.append(mean_cls)

        # Class covariance
        if n_cls > 1:
            cov_cls = np.cov(X_cls, rowvar=False, bias=False)
        else:
            cov_cls = np.eye(n_features)

        # Regularization
        cov_cls += np.eye(n_features) * 1e-10
        covs.append(cov_cls)

        # Inverse covariance
        try:
            inv_cov_cls = np.linalg.inv(cov_cls)
        except np.linalg.LinAlgError:
            inv_cov_cls = np.linalg.pinv(cov_cls)
        inv_covs.append(inv_cov_cls)

        # Prior
        priors.append(n_cls / len(y))

    return {
        'means': np.array(means),
        'covs': covs,
        'inv_covs': inv_covs,
        'classes': classes,
        'priors': np.array(priors),
        'n_features': n_features,
        'n_classes': n_classes
    }


def predict_qda(
    X: np.ndarray,
    model: Dict[str, any]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Predict classes using QDA model.

    Parameters
    ----------
    X : ndarray
        Data to predict
    model : dict
        QDA model from fit_qda

    Returns
    -------
    predictions : ndarray
        Predicted class labels
    distances : ndarray of shape (n_samples, n_classes)
        Mahalanobis distances to each class
    """
    n_samples = X.shape[0]
    n_classes = len(model['classes'])

    # Calculate Mahalanobis distances to each class
    distances = np.zeros((n_samples, n_classes))
    for i, cls in enumerate(model['classes']):
        diff = X - model['means'][i]
        distances[:, i] = np.sum(diff @ model['inv_covs'][i] * diff, axis=1)

    # Predict class with minimum distance
    predictions = model['classes'][np.argmin(distances, axis=1)]

    return predictions, distances


def predict_qda_detailed(
    X: np.ndarray,
    model: Dict[str, any]
) -> Dict[str, np.ndarray]:
    """
    Predict classes using QDA model with detailed output including posterior probabilities.

    Uses quadratic discriminant function with class-specific covariances.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Data to predict
    model : dict
        QDA model from fit_qda

    Returns
    -------
    dict
        Dictionary containing:
        - predicted_classes : ndarray (n_samples,)
            Predicted class labels
        - discriminant_scores : ndarray (n_samples, n_classes)
            Quadratic discriminant function values for each class
        - posterior_probabilities : ndarray (n_samples, n_classes)
            Posterior probabilities for each class (softmax of discriminant scores)
        - prediction_confidence : ndarray (n_samples,)
            Maximum posterior probability for each sample
        - mahalanobis_distances : ndarray (n_samples, n_classes)
            Mahalanobis distances to each class mean
        - mahalanobis_distances_squared : ndarray (n_samples, n_classes)
            Squared Mahalanobis distances
        - classes : ndarray
            Class labels

    Reference
    ---------
    CL_prediction_QDA.r
    """
    n_samples = X.shape[0]
    n_classes = len(model['classes'])
    classes = model['classes']

    # Calculate Mahalanobis distances and discriminant scores
    distances = np.zeros((n_samples, n_classes))
    discriminant_scores = np.zeros((n_samples, n_classes))

    for i, cls in enumerate(classes):
        diff = X - model['means'][i]

        # Mahalanobis distance squared (class-specific covariance)
        md_squared = np.sum(diff @ model['inv_covs'][i] * diff, axis=1)
        distances[:, i] = md_squared

        # Quadratic discriminant function:
        # D_j = -0.5 * log|Σ_j| - 0.5 * (x - μ_j)ᵀ Σ_j⁻¹ (x - μ_j) + log(π_j)
        # Calculate log determinant using slogdet for numerical stability
        sign, logdet = np.linalg.slogdet(model['covs'][i])
        discriminant_scores[:, i] = (
            -0.5 * logdet
            - 0.5 * md_squared
            + np.log(model['priors'][i] + 1e-10)
        )

    # Compute posterior probabilities using log-sum-exp trick
    max_scores = np.max(discriminant_scores, axis=1, keepdims=True)
    exp_scores = np.exp(discriminant_scores - max_scores)
    posterior_probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    # Predicted class: argmax of posterior probabilities
    predicted_indices = np.argmax(posterior_probs, axis=1)
    predicted_classes = classes[predicted_indices]

    # Prediction confidence: maximum posterior probability
    prediction_confidence = np.max(posterior_probs, axis=1)

    return {
        'predicted_classes': predicted_classes,
        'discriminant_scores': discriminant_scores,
        'posterior_probabilities': posterior_probs,
        'prediction_confidence': prediction_confidence,
        'mahalanobis_distances': np.sqrt(distances),
        'mahalanobis_distances_squared': distances,
        'classes': classes
    }


# ============================================================================
# kNN - k-Nearest Neighbors
# ============================================================================

def calculate_distance_matrix(
    X1: np.ndarray,
    X2: np.ndarray,
    metric: str = 'euclidean',
    cov: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Calculate pairwise distances between samples.

    Parameters
    ----------
    X1 : ndarray
        First set of samples
    X2 : ndarray
        Second set of samples
    metric : str
        Distance metric: 'euclidean', 'manhattan', 'chebyshev', 'minkowski', 'mahalanobis'
    cov : ndarray, optional
        Covariance matrix for Mahalanobis distance

    Returns
    -------
    ndarray
        Distance matrix of shape (n_samples_X1, n_samples_X2)
    """
    if metric == 'mahalanobis':
        if cov is None:
            # Calculate covariance from combined data
            X_combined = np.vstack([X1, X2])
            cov = np.cov(X_combined, rowvar=False) + np.eye(X1.shape[1]) * 1e-10

        try:
            inv_cov = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            inv_cov = np.linalg.pinv(cov)

        # Calculate Mahalanobis distances
        distances = np.zeros((X1.shape[0], X2.shape[0]))
        for i in range(X1.shape[0]):
            diff = X2 - X1[i]
            distances[i] = np.sqrt(np.sum(diff @ inv_cov * diff, axis=1))

        return distances

    elif metric in ['euclidean', 'manhattan', 'chebyshev', 'minkowski']:
        return cdist(X1, X2, metric=metric)

    else:
        raise ValueError(f"Unknown metric: {metric}")


def fit_knn(
    X: np.ndarray,
    y: np.ndarray,
    metric: str = 'euclidean'
) -> Dict[str, any]:
    """
    Prepare kNN model (store training data).

    Parameters
    ----------
    X : ndarray
        Training data
    y : ndarray
        Training labels
    metric : str
        Distance metric

    Returns
    -------
    dict
        kNN model
    """
    classes = np.unique(y)

    # Calculate covariance if using Mahalanobis
    cov = None
    if metric == 'mahalanobis':
        cov = np.cov(X, rowvar=False) + np.eye(X.shape[1]) * 1e-10

    return {
        'X_train': X,
        'y_train': y,
        'classes': classes,
        'metric': metric,
        'cov': cov
    }


def predict_knn(
    X: np.ndarray,
    model: Dict[str, any],
    k: int = 3
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Predict classes using kNN.

    Parameters
    ----------
    X : ndarray
        Data to predict
    model : dict
        kNN model
    k : int
        Number of neighbors

    Returns
    -------
    predictions : ndarray
        Predicted class labels
    neighbor_info : ndarray
        Nearest neighbor information
    """
    # Calculate distances
    distances = calculate_distance_matrix(
        X,
        model['X_train'],
        metric=model['metric'],
        cov=model['cov']
    )

    # Find k nearest neighbors
    n_samples = X.shape[0]
    predictions = np.zeros(n_samples, dtype=model['y_train'].dtype)

    for i in range(n_samples):
        # Get k nearest neighbors
        nearest_indices = np.argsort(distances[i])[:k]
        nearest_labels = model['y_train'][nearest_indices]

        # Vote
        unique_labels, counts = np.unique(nearest_labels, return_counts=True)
        # In case of tie, take first one
        predictions[i] = unique_labels[np.argmax(counts)]

    return predictions, distances


def predict_knn_detailed(
    X: np.ndarray,
    model: Dict[str, any],
    k: int = 3
) -> Dict[str, np.ndarray]:
    """
    Predict classes using kNN with detailed neighbor information.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Data to predict
    model : dict
        kNN model from fit_knn
    k : int
        Number of neighbors to consider

    Returns
    -------
    dict
        Dictionary containing:
        - predicted_classes : ndarray (n_samples,)
            Predicted class labels
        - neighbor_distances : ndarray (n_samples, k)
            Distances to k nearest neighbors
        - neighbor_indices : ndarray (n_samples, k)
            Indices of k nearest neighbors in training set
        - neighbor_classes : ndarray (n_samples, k)
            Class labels of k nearest neighbors
        - prediction_confidence : ndarray (n_samples,)
            Confidence = count(same_class) / k
        - vote_counts : ndarray (n_samples, n_classes)
            Vote count for each class
        - k : int
            Number of neighbors used
        - classes : ndarray
            Class labels

    Reference
    ---------
    CL_prediction_KNN.R
    """
    # Calculate distances
    distances = calculate_distance_matrix(
        X,
        model['X_train'],
        metric=model['metric'],
        cov=model['cov']
    )

    n_samples = X.shape[0]
    n_classes = len(model['classes'])
    classes = model['classes']

    # Initialize output arrays
    predicted_classes = np.zeros(n_samples, dtype=model['y_train'].dtype)
    neighbor_indices = np.zeros((n_samples, k), dtype=int)
    neighbor_distances = np.zeros((n_samples, k))
    neighbor_classes = np.zeros((n_samples, k), dtype=model['y_train'].dtype)
    prediction_confidence = np.zeros(n_samples)
    vote_counts = np.zeros((n_samples, n_classes))

    for i in range(n_samples):
        # Get k nearest neighbors
        nearest_idx = np.argsort(distances[i])[:k]
        nearest_labels = model['y_train'][nearest_idx]

        neighbor_indices[i] = nearest_idx
        neighbor_distances[i] = distances[i][nearest_idx]
        neighbor_classes[i] = nearest_labels

        # Vote: count occurrences of each class
        for j, cls in enumerate(classes):
            vote_counts[i, j] = np.sum(nearest_labels == cls)

        # Prediction: class with maximum votes
        # In case of tie, take first one (lowest class index)
        max_votes = np.max(vote_counts[i])
        predicted_classes[i] = classes[np.argmax(vote_counts[i])]

        # Confidence: proportion of votes for predicted class
        prediction_confidence[i] = max_votes / k

    return {
        'predicted_classes': predicted_classes,
        'neighbor_distances': neighbor_distances,
        'neighbor_indices': neighbor_indices,
        'neighbor_classes': neighbor_classes,
        'prediction_confidence': prediction_confidence,
        'vote_counts': vote_counts,
        'k': k,
        'classes': classes
    }


# ============================================================================
# NIPALS Algorithm (for SIMCA and UNEQ)
# ============================================================================

def nipals_pca(
    X: np.ndarray,
    n_components: int,
    max_iter: int = NIPALS_MAX_ITER,
    tol: float = NIPALS_TOLERANCE
) -> Dict[str, any]:
    """
    NIPALS algorithm for PCA (used in SIMCA/UNEQ).

    Parameters
    ----------
    X : ndarray
        Data matrix (should be centered/scaled)
    n_components : int
        Number of components to extract
    max_iter : int
        Maximum iterations
    tol : float
        Convergence tolerance

    Returns
    -------
    dict
        PCA results with scores, loadings, and explained variance
    """
    n_samples, n_features = X.shape
    scores = np.zeros((n_samples, n_components))
    loadings = np.zeros((n_components, n_features))
    explained_var = np.zeros(n_components)

    X_residual = X.copy()
    total_var = np.sum(X ** 2)

    for comp in range(n_components):
        # Initialize with column having maximum variance
        var_cols = np.sum(X_residual ** 2, axis=0)
        max_var_idx = np.argmax(var_cols)
        t = X_residual[:, max_var_idx].reshape(-1, 1)

        # NIPALS iterations
        for iteration in range(max_iter):
            # Calculate loading vector
            p = (X_residual.T @ t) / (t.T @ t)
            p = p / np.linalg.norm(p)  # Normalize

            # Calculate score vector
            t_new = (X_residual @ p) / (p.T @ p)

            # Check convergence
            diff = np.abs(np.sum(t_new ** 2) - np.sum(t ** 2))
            if diff < tol:
                break

            t = t_new

        # Store results
        scores[:, comp] = t.flatten()
        loadings[comp, :] = p.flatten()

        # Calculate explained variance
        explained_var[comp] = np.sum(t ** 2)

        # Deflate matrix
        X_residual = X_residual - t @ p.T

    # Calculate percentage of explained variance
    explained_var_ratio = explained_var / total_var * 100

    return {
        'scores': scores,
        'loadings': loadings,
        'explained_variance': explained_var,
        'explained_variance_ratio': explained_var_ratio,
        'residuals': X_residual
    }


# ============================================================================
# SIMCA - Soft Independent Modeling of Class Analogy
# ============================================================================

def fit_simca(
    X: np.ndarray,
    y: np.ndarray,
    n_components: int,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    scaling_params: Optional[Dict] = None
) -> Dict[str, any]:
    """
    Fit SIMCA model.

    Parameters
    ----------
    X : ndarray
        Training data (should already be scaled)
    y : ndarray
        Class labels
    n_components : int
        Number of PCs per class model
    confidence_level : float
        Confidence level for acceptance threshold
    scaling_params : dict, optional
        Mean and std for each class

    Returns
    -------
    dict
        SIMCA model
    """
    classes = np.unique(y)
    class_models = {}

    for cls in classes:
        X_cls = X[y == cls]
        n_samples_cls, n_features = X_cls.shape

        # Get scaling parameters for this class
        if scaling_params and cls in scaling_params:
            mean_cls = scaling_params[cls]['mean']
            std_cls = scaling_params[cls]['std']
            X_cls_scaled = (X_cls - mean_cls) / std_cls
        else:
            mean_cls = X_cls.mean(axis=0)
            std_cls = X_cls.std(axis=0, ddof=1)
            std_cls[std_cls == 0] = 1.0
            X_cls_scaled = (X_cls - mean_cls) / std_cls

        # Perform PCA using NIPALS
        pca_result = nipals_pca(X_cls_scaled, n_components)

        # Calculate residual variance
        residuals = X_cls_scaled - pca_result['scores'] @ pca_result['loadings']
        residual_var = np.sum(residuals ** 2) / ((n_features - n_components) * (n_samples_cls - n_components - 1))

        # Calculate F critical value
        df1 = n_features - n_components
        df2 = (n_features - n_components) * (n_samples_cls - n_components - 1)
        f_crit = f_dist.ppf(confidence_level, df1, df2)

        class_models[cls] = {
            'mean': mean_cls,
            'std': std_cls,
            'scores': pca_result['scores'],
            'loadings': pca_result['loadings'],
            'residual_var': residual_var,
            'f_critical': f_crit,
            'n_samples': n_samples_cls,
            'n_components': n_components,
            'n_features': n_features
        }

    return {
        'class_models': class_models,
        'classes': classes,
        'n_components': n_components,
        'confidence_level': confidence_level
    }


def predict_simca(
    X: np.ndarray,
    model: Dict[str, any]
) -> Tuple[Dict[any, np.ndarray], Dict[any, np.ndarray]]:
    """
    Predict using SIMCA model.

    Parameters
    ----------
    X : ndarray
        Data to predict
    model : dict
        SIMCA model

    Returns
    -------
    distances : dict
        Dictionary mapping class to distance values for each sample
    acceptance : dict
        Dictionary mapping class to acceptance (1) or rejection (0) for each sample
    """
    n_samples = X.shape[0]
    classes = model['classes']

    distances = {}
    acceptance = {}

    for cls in classes:
        class_model = model['class_models'][cls]

        # Scale data using class parameters
        X_scaled = (X - class_model['mean']) / class_model['std']

        # Project onto PCA space
        scores_new = X_scaled @ class_model['loadings'].T

        # Reconstruct
        X_reconstructed = scores_new @ class_model['loadings']

        # Calculate residuals
        residuals = X_scaled - X_reconstructed
        n_features = class_model['n_features']
        n_components = class_model['n_components']

        residual_sum = np.sum(residuals ** 2, axis=1) / (n_features - n_components)

        # Calculate augmented distance (consider score space boundaries)
        score_min = class_model['scores'].min(axis=0)
        score_max = class_model['scores'].max(axis=0)

        # Check if outside score space
        augmented_penalty = np.zeros(n_samples)
        for pc in range(n_components):
            below_min = np.maximum(0, score_min[pc] - scores_new[:, pc])
            above_max = np.maximum(0, scores_new[:, pc] - score_max[pc])
            score_var = np.var(class_model['scores'][:, pc])
            if score_var > 0:
                augmented_penalty += (below_min ** 2 + above_max ** 2) * class_model['residual_var'] / score_var

        # Total distance (F-statistic)
        f_values = (residual_sum + augmented_penalty) / class_model['residual_var']

        distances[cls] = f_values

        # Acceptance based on F-critical
        acceptance[cls] = (f_values <= class_model['f_critical']).astype(int)

    return distances, acceptance


def predict_simca_detailed(
    X: np.ndarray,
    model: Dict[str, any]
) -> Dict[str, any]:
    """
    Predict using SIMCA model with detailed classification output.

    Classification logic:
    - If exactly 1 class accepts → predict it (type='accepted')
    - If multiple classes accept → minimum distance (type='ambiguous')
    - If no classes accept → minimum distance (type='rejected')

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Data to predict
    model : dict
        SIMCA model from fit_simca

    Returns
    -------
    dict
        Dictionary containing:
        - predicted_classes : ndarray (n_samples,)
            Predicted class labels
        - distances_per_class : ndarray (n_samples, n_classes)
            F-statistic distances to each class
        - distances_squared : ndarray (n_samples, n_classes)
            Squared F-statistic distances
        - class_acceptance : ndarray (n_samples, n_classes) bool
            Acceptance/rejection for each class
        - num_accepted_per_sample : ndarray (n_samples,)
            Number of classes that accepted each sample
        - prediction_type : ndarray (n_samples,) dtype=str
            'accepted', 'ambiguous', or 'rejected'
        - prediction_confidence : ndarray (n_samples,)
            Confidence based on acceptance pattern
        - min_distances : ndarray (n_samples,)
            Minimum distance across all classes
        - classes : ndarray
            Class labels

    Reference
    ---------
    simca.m prediction section
    """
    n_samples = X.shape[0]
    classes = model['classes']
    n_classes = len(classes)

    # Arrays to store results
    distances_array = np.zeros((n_samples, n_classes))
    acceptance_array = np.zeros((n_samples, n_classes), dtype=bool)

    # Get distances and acceptance from basic predict_simca
    distances_dict, acceptance_dict = predict_simca(X, model)

    # Convert dictionaries to arrays
    for i, cls in enumerate(classes):
        distances_array[:, i] = distances_dict[cls]
        acceptance_array[:, i] = acceptance_dict[cls].astype(bool)

    # Count number of accepting classes per sample
    num_accepted = np.sum(acceptance_array, axis=1)

    # Determine prediction type and predicted class
    predicted_classes = np.zeros(n_samples, dtype=classes.dtype)
    prediction_type = np.empty(n_samples, dtype='U10')
    prediction_confidence = np.zeros(n_samples)
    min_distances = np.min(distances_array, axis=1)

    for i in range(n_samples):
        if num_accepted[i] == 1:
            # Exactly one class accepts → predict it
            predicted_classes[i] = classes[np.where(acceptance_array[i])[0][0]]
            prediction_type[i] = 'accepted'
            prediction_confidence[i] = 1.0

        elif num_accepted[i] > 1:
            # Multiple classes accept → ambiguous, use minimum distance
            predicted_classes[i] = classes[np.argmin(distances_array[i])]
            prediction_type[i] = 'ambiguous'
            # Confidence inversely related to number of accepting classes
            prediction_confidence[i] = 1.0 / num_accepted[i]

        else:  # num_accepted[i] == 0
            # No classes accept → rejected, use minimum distance
            predicted_classes[i] = classes[np.argmin(distances_array[i])]
            prediction_type[i] = 'rejected'
            # Low confidence for rejected samples
            prediction_confidence[i] = 0.0

    return {
        'predicted_classes': predicted_classes,
        'distances_per_class': distances_array,
        'distances_squared': distances_array ** 2,
        'class_acceptance': acceptance_array,
        'num_accepted_per_sample': num_accepted,
        'prediction_type': prediction_type,
        'prediction_confidence': prediction_confidence,
        'min_distances': min_distances,
        'classes': classes
    }


# ============================================================================
# UNEQ - Unequal Class Dispersions
# ============================================================================

def fit_uneq(
    X: np.ndarray,
    y: np.ndarray,
    n_components: Optional[int] = None,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    use_pca: bool = False
) -> Dict[str, any]:
    """
    Fit UNEQ model.

    Parameters
    ----------
    X : ndarray
        Training data
    y : ndarray
        Class labels
    n_components : int, optional
        Number of PCs (if using PCA)
    confidence_level : float
        Confidence level
    use_pca : bool
        Whether to use PCA

    Returns
    -------
    dict
        UNEQ model
    """
    classes = np.unique(y)
    class_models = {}

    for cls in classes:
        X_cls = X[y == cls]
        n_samples_cls, n_features = X_cls.shape

        if use_pca and n_components is not None:
            # Use PCA
            mean_cls = X_cls.mean(axis=0)
            X_cls_centered = X_cls - mean_cls

            pca_result = nipals_pca(X_cls_centered, n_components)

            # Model in PC space
            mean_model = np.zeros(n_components)
            cov_model = np.cov(pca_result['scores'], rowvar=False)
            cov_model += np.eye(n_components) * 1e-10

            class_models[cls] = {
                'use_pca': True,
                'mean_original': mean_cls,
                'loadings': pca_result['loadings'],
                'mean_pc': mean_model,
                'cov_pc': cov_model,
                'inv_cov_pc': np.linalg.inv(cov_model),
                'n_samples': n_samples_cls,
                'n_components': n_components,
                'n_features': n_features
            }

            # Calculate thresholds
            # Beta distribution for own class (Mahalanobis distance)
            v = n_components
            beta_crit = beta_dist.ppf(confidence_level, v/2, (n_samples_cls - v - 1)/2)
            md_beta = beta_crit / (n_samples_cls / (n_samples_cls - 1)**2)

            # Hotelling T2 for other classes
            f_val = f_dist.ppf(confidence_level, v, n_samples_cls - v)
            t2_crit = ((n_samples_cls - 1) * v / (n_samples_cls - v)) * f_val
            t2_crit_corrected = t2_crit * ((n_samples_cls + 1) / n_samples_cls)

            class_models[cls]['md_beta'] = md_beta
            class_models[cls]['t2_critical'] = t2_crit_corrected

        else:
            # Direct Mahalanobis distance (no PCA)
            mean_cls = X_cls.mean(axis=0)
            cov_cls = np.cov(X_cls, rowvar=False)
            cov_cls += np.eye(n_features) * 1e-10

            try:
                inv_cov_cls = np.linalg.inv(cov_cls)
            except np.linalg.LinAlgError:
                inv_cov_cls = np.linalg.pinv(cov_cls)

            class_models[cls] = {
                'use_pca': False,
                'mean': mean_cls,
                'cov': cov_cls,
                'inv_cov': inv_cov_cls,
                'n_samples': n_samples_cls,
                'n_features': n_features
            }

            # Calculate thresholds
            v = n_features
            beta_crit = beta_dist.ppf(confidence_level, v/2, (n_samples_cls - v - 1)/2)
            md_beta = beta_crit / (n_samples_cls / (n_samples_cls - 1)**2)

            f_val = f_dist.ppf(confidence_level, v, n_samples_cls - v)
            t2_crit = ((n_samples_cls - 1) * v / (n_samples_cls - v)) * f_val
            t2_crit_corrected = t2_crit * ((n_samples_cls + 1) / n_samples_cls)

            class_models[cls]['md_beta'] = md_beta
            class_models[cls]['t2_critical'] = t2_crit_corrected

    return {
        'class_models': class_models,
        'classes': classes,
        'confidence_level': confidence_level,
        'use_pca': use_pca,
        'n_components': n_components
    }


def predict_uneq(
    X: np.ndarray,
    model: Dict[str, any],
    y_true: Optional[np.ndarray] = None
) -> Tuple[Dict[any, np.ndarray], Dict[any, np.ndarray]]:
    """
    Predict using UNEQ model.

    Parameters
    ----------
    X : ndarray
        Data to predict
    model : dict
        UNEQ model
    y_true : ndarray, optional
        True labels (to apply different thresholds for own class)

    Returns
    -------
    distances : dict
        Mahalanobis distances to each class
    acceptance : dict
        Acceptance/rejection for each class
    """
    n_samples = X.shape[0]
    classes = model['classes']

    distances = {}
    acceptance = {}

    for cls in classes:
        class_model = model['class_models'][cls]

        if class_model['use_pca']:
            # Project to PC space
            X_centered = X - class_model['mean_original']
            scores = X_centered @ class_model['loadings'].T

            # Mahalanobis distance in PC space
            diff = scores - class_model['mean_pc']
            md_squared = np.sum(diff @ class_model['inv_cov_pc'] * diff, axis=1)

        else:
            # Direct Mahalanobis distance
            diff = X - class_model['mean']
            md_squared = np.sum(diff @ class_model['inv_cov'] * diff, axis=1)

        distances[cls] = md_squared

        # Apply appropriate threshold
        if y_true is not None:
            # Use beta threshold for own class, T2 for others
            own_class_mask = (y_true == cls)
            accept = np.zeros(n_samples, dtype=int)
            accept[own_class_mask] = (md_squared[own_class_mask] <= class_model['md_beta']).astype(int)
            accept[~own_class_mask] = (md_squared[~own_class_mask] <= class_model['t2_critical']).astype(int)
            acceptance[cls] = accept
        else:
            # Use T2 threshold for all (prediction mode)
            acceptance[cls] = (md_squared <= class_model['t2_critical']).astype(int)

    return distances, acceptance


def predict_uneq_detailed(
    X: np.ndarray,
    model: Dict[str, any]
) -> Dict[str, any]:
    """
    Predict using UNEQ model with detailed acceptance pattern output.

    Classification logic (same as SIMCA):
    - If exactly 1 class accepts → predict it (type='accepted')
    - If multiple classes accept → minimum distance (type='ambiguous')
    - If no classes accept → minimum distance (type='rejected')

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Data to predict
    model : dict
        UNEQ model from fit_uneq

    Returns
    -------
    dict
        Dictionary containing:
        - predicted_classes : ndarray (n_samples,)
            Predicted class labels
        - distances_per_class : ndarray (n_samples, n_classes)
            Mahalanobis distances (squared) to each class
        - class_acceptance : ndarray (n_samples, n_classes) bool
            Acceptance/rejection for each class
        - num_accepted_per_sample : ndarray (n_samples,)
            Number of classes that accepted each sample
        - prediction_type : ndarray (n_samples,) dtype=str
            'accepted', 'ambiguous', or 'rejected'
        - prediction_confidence : ndarray (n_samples,)
            Confidence based on acceptance pattern
        - min_distances : ndarray (n_samples,)
            Minimum distance across all classes
        - classes : ndarray
            Class labels

    Reference
    ---------
    uneq.m lines 188-205
    """
    n_samples = X.shape[0]
    classes = model['classes']
    n_classes = len(classes)

    # Arrays to store results
    distances_array = np.zeros((n_samples, n_classes))
    acceptance_array = np.zeros((n_samples, n_classes), dtype=bool)

    # Get distances and acceptance from basic predict_uneq (without y_true)
    distances_dict, acceptance_dict = predict_uneq(X, model, y_true=None)

    # Convert dictionaries to arrays
    for i, cls in enumerate(classes):
        distances_array[:, i] = distances_dict[cls]
        acceptance_array[:, i] = acceptance_dict[cls].astype(bool)

    # Count number of accepting classes per sample
    num_accepted = np.sum(acceptance_array, axis=1)

    # Determine prediction type and predicted class
    predicted_classes = np.zeros(n_samples, dtype=classes.dtype)
    prediction_type = np.empty(n_samples, dtype='U10')
    prediction_confidence = np.zeros(n_samples)
    min_distances = np.min(distances_array, axis=1)

    for i in range(n_samples):
        if num_accepted[i] == 1:
            # Exactly one class accepts → predict it
            predicted_classes[i] = classes[np.where(acceptance_array[i])[0][0]]
            prediction_type[i] = 'accepted'
            prediction_confidence[i] = 1.0

        elif num_accepted[i] > 1:
            # Multiple classes accept → ambiguous, use minimum distance
            predicted_classes[i] = classes[np.argmin(distances_array[i])]
            prediction_type[i] = 'ambiguous'
            # Confidence inversely related to number of accepting classes
            prediction_confidence[i] = 1.0 / num_accepted[i]

        else:  # num_accepted[i] == 0
            # No classes accept → rejected, use minimum distance
            predicted_classes[i] = classes[np.argmin(distances_array[i])]
            prediction_type[i] = 'rejected'
            # Low confidence for rejected samples
            prediction_confidence[i] = 0.0

    return {
        'predicted_classes': predicted_classes,
        'distances_per_class': distances_array,
        'class_acceptance': acceptance_array,
        'num_accepted_per_sample': num_accepted,
        'prediction_type': prediction_type,
        'prediction_confidence': prediction_confidence,
        'min_distances': min_distances,
        'classes': classes
    }
