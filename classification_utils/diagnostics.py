"""
Classification Diagnostics and Validation
==========================================

Functions for evaluating classification models including:
- Confusion matrices and performance metrics
- Cross-validation
- Sensitivity, specificity, and efficiency calculations
- Model comparison tools
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, List, Callable
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
from .config import DEFAULT_CV_FOLDS
from .preprocessing import create_cv_folds, create_stratified_cv_folds, create_stratified_cv_folds_with_groups
from . import calculations


def calculate_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: Optional[np.ndarray] = None
) -> pd.DataFrame:
    """
    Calculate confusion matrix as DataFrame.

    Parameters
    ----------
    y_true : ndarray
        True labels
    y_pred : ndarray
        Predicted labels
    classes : ndarray, optional
        Class labels in desired order

    Returns
    -------
    DataFrame
        Confusion matrix with class labels
    """
    if classes is None:
        classes = np.unique(np.concatenate([y_true, y_pred]))

    # Filter classes to only those actually present in y_true
    classes_in_test = np.intersect1d(classes, y_true)
    cm = confusion_matrix(y_true, y_pred, labels=classes_in_test)

    return pd.DataFrame(
        cm,
        index=[f"True_{cls}" for cls in classes_in_test],
        columns=[f"Pred_{cls}" for cls in classes_in_test]
    )


def calculate_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: Optional[np.ndarray] = None
) -> Dict[str, any]:
    """
    Calculate comprehensive classification metrics.

    Parameters
    ----------
    y_true : ndarray
        True labels
    y_pred : ndarray
        Predicted labels
    classes : ndarray, optional
        Class labels

    Returns
    -------
    dict
        Dictionary of metrics including accuracy, precision, recall, F1-score
    """
    # Validate inputs
    if y_true is None or len(y_true) == 0:
        raise ValueError("y_true is empty or None. Cannot calculate metrics.")

    if y_pred is None or len(y_pred) == 0:
        raise ValueError("y_pred is empty or None. Cannot calculate metrics.")

    # Convert to numpy arrays for consistent type handling
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    if classes is None:
        classes = np.unique(np.concatenate([y_true_arr, y_pred_arr]))
    else:
        classes = np.asarray(classes)

    if len(classes) == 0:
        raise ValueError("classes is empty. Cannot calculate metrics.")

    # Overall accuracy
    accuracy = accuracy_score(y_true_arr, y_pred_arr)

    # Filter classes to only those actually present in y_true
    classes_in_test = np.intersect1d(classes, y_true_arr)

    # Validate that we have common classes
    if len(classes_in_test) == 0:
        import streamlit as st
        st.error(f"❌ DEBUG: classes_in_test is EMPTY!")
        st.write(f"  - classes parameter: {classes}")
        st.write(f"  - unique values in y_true: {np.unique(y_true_arr)}")
        st.write(f"  - unique values in y_pred: {np.unique(y_pred_arr)}")
        st.write(f"  - y_true dtype: {y_true_arr.dtype}")
        st.write(f"  - classes dtype: {classes.dtype}")
        raise ValueError(
            f"No common classes found between y_true and classes parameter. "
            f"y_true contains: {np.unique(y_true_arr).tolist()}, "
            f"classes parameter contains: {classes.tolist()}"
        )

    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true_arr, y_pred_arr, labels=classes_in_test, zero_division=0
    )

    # Confusion matrix
    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=classes_in_test)

    # Calculate sensitivity (recall) and specificity per class
    n_classes = len(classes_in_test)
    sensitivity = np.zeros(n_classes)
    specificity = np.zeros(n_classes)

    for i, cls in enumerate(classes_in_test):
        # True positives
        tp = cm[i, i]

        # False negatives
        fn = cm[i, :].sum() - tp

        # False positives
        fp = cm[:, i].sum() - tp

        # True negatives
        tn = cm.sum() - tp - fn - fp

        # Sensitivity (True Positive Rate)
        sensitivity[i] = tp / (tp + fn) if (tp + fn) > 0 else 0

        # Specificity (True Negative Rate)
        specificity[i] = tn / (tn + fp) if (tn + fp) > 0 else 0

    # Calculate efficiency (geometric mean of sensitivity and specificity)
    efficiency = np.sqrt(sensitivity * specificity) * 100

    # Per-class results
    class_metrics = {}
    for i, cls in enumerate(classes_in_test):
        class_metrics[cls] = {
            'sensitivity': sensitivity[i] * 100,
            'specificity': specificity[i] * 100,
            'efficiency': efficiency[i],
            'precision': precision[i] * 100,
            'recall': recall[i] * 100,
            'f1_score': f1[i] * 100,
            'support': int(support[i])
        }

    # Flatten class_metrics into per-class dictionaries for easier UI access
    sensitivity_per_class = {cls: class_metrics[cls]['sensitivity'] for cls in class_metrics}
    specificity_per_class = {cls: class_metrics[cls]['specificity'] for cls in class_metrics}
    precision_per_class = {cls: class_metrics[cls]['precision'] for cls in class_metrics}
    recall_per_class = {cls: class_metrics[cls]['recall'] for cls in class_metrics}
    f1_per_class = {cls: class_metrics[cls]['f1_score'] for cls in class_metrics}

    return {
        'accuracy': accuracy * 100,
        'avg_sensitivity': np.mean(sensitivity) * 100,
        'avg_specificity': np.mean(specificity) * 100,
        'avg_efficiency': np.mean(efficiency),
        'class_metrics': class_metrics,
        # Add flattened versions for easier access in UI
        'sensitivity_per_class': sensitivity_per_class,
        'specificity_per_class': specificity_per_class,
        'precision_per_class': precision_per_class,
        'recall_per_class': recall_per_class,
        'f1_per_class': f1_per_class,
        'confusion_matrix': cm,
        'classes': classes_in_test
    }


def calculate_simca_uneq_metrics(
    acceptance: Dict[any, np.ndarray],
    y_true: np.ndarray,
    classes: np.ndarray
) -> Dict[str, any]:
    """
    Calculate sensitivity and specificity for SIMCA/UNEQ models.

    Parameters
    ----------
    acceptance : dict
        Dictionary mapping class to acceptance arrays
    y_true : ndarray
        True class labels
    classes : ndarray
        Class labels

    Returns
    -------
    dict
        Sensitivity, specificity, and efficiency per class
    """
    n_classes = len(classes)
    sensitivity = np.zeros(n_classes)
    specificity = np.zeros(n_classes)

    for i, cls in enumerate(classes):
        # Get acceptance for this class
        accept_cls = acceptance[cls]

        # Own class samples (should be accepted)
        own_class_mask = (y_true == cls)
        n_own_class = own_class_mask.sum()

        if n_own_class > 0:
            sensitivity[i] = accept_cls[own_class_mask].sum() / n_own_class

        # Other class samples (should be rejected)
        other_class_mask = ~own_class_mask
        n_other_class = other_class_mask.sum()

        if n_other_class > 0:
            # Specificity = correctly rejected / total other class
            specificity[i] = (1 - accept_cls[other_class_mask]).sum() / n_other_class

    # Calculate efficiency
    efficiency = np.sqrt(sensitivity * specificity) * 100

    # Per-class results
    class_metrics = {}
    for i, cls in enumerate(classes):
        class_metrics[cls] = {
            'sensitivity': sensitivity[i] * 100,
            'specificity': specificity[i] * 100,
            'efficiency': efficiency[i]
        }

    return {
        'avg_sensitivity': np.mean(sensitivity) * 100,
        'avg_specificity': np.mean(specificity) * 100,
        'avg_efficiency': np.mean(efficiency),
        'class_metrics': class_metrics
    }


def cross_validate_lda(
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = DEFAULT_CV_FOLDS,
    random_state: Optional[int] = None
) -> Dict[str, any]:
    """
    Cross-validation for LDA.

    Parameters
    ----------
    X : ndarray
        Feature matrix
    y : ndarray
        Class labels
    n_folds : int
        Number of CV folds
    random_state : int, optional
        Random seed

    Returns
    -------
    dict
        CV results including predictions and metrics
    """
    n_samples = len(y)
    classes = np.unique(y)
    fold_indices = create_cv_folds(n_samples, n_folds, random_state)

    y_pred_cv = np.zeros_like(y)
    cv_details = []

    for fold in range(n_folds):
        # Split data
        test_mask = fold_indices == fold
        train_mask = ~test_mask

        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]

        # Fit and predict
        model = calculations.fit_lda(X_train, y_train)
        y_pred, _ = calculations.predict_lda(X_test, model)

        y_pred_cv[test_mask] = y_pred

        # Calculate fold metrics
        fold_metrics = calculate_classification_metrics(y_test, y_pred, classes=classes)

        # Calculate Mahalanobis distances for this fold as matrix
        mahal_dist_matrix = np.zeros((len(X_test), len(classes)))
        try:
            for class_idx, cls in enumerate(classes):
                class_mask = (y_train == cls)
                X_train_class = X_train[class_mask]

                if len(X_train_class) > 0:
                    # Calculate covariance and inverse
                    cov_matrix = np.cov(X_train_class.T)

                    # Handle singular matrix
                    try:
                        inv_cov = np.linalg.inv(cov_matrix)
                    except np.linalg.LinAlgError:
                        inv_cov = np.linalg.pinv(cov_matrix)

                    # Class centroid
                    centroid = np.mean(X_train_class, axis=0)

                    # Mahalanobis distance from test samples to this class centroid
                    diff = X_test - centroid
                    mahal_dist = np.sqrt(np.sum(diff @ inv_cov * diff, axis=1))
                    mahal_dist_matrix[:, class_idx] = mahal_dist
        except Exception:
            # If calculation fails, matrix remains zeros
            pass

        cv_details.append({
            'fold': fold,
            'accuracy': fold_metrics['accuracy'],
            'n_train': len(np.where(train_mask)[0]),
            'n_test': len(np.where(test_mask)[0]),
            'mahal_distances_matrix': mahal_dist_matrix,
            'test_indices': np.where(test_mask)[0]
        })

    # Calculate metrics
    metrics = calculate_classification_metrics(y, y_pred_cv, classes=classes)

    # Add misclassified samples
    misclassified = np.where(y != y_pred_cv)[0]

    # Reconstruct full Mahalanobis distance matrix from folds
    all_mahal_distances = np.zeros((len(y), len(classes)))
    for fold_detail in cv_details:
        test_idx = fold_detail['test_indices']
        all_mahal_distances[test_idx] = fold_detail['mahal_distances_matrix']

    return {
        'predictions': y_pred_cv,
        'metrics': metrics,
        'misclassified_indices': misclassified,
        'n_misclassified': len(misclassified),
        'mahalanobis_distances': all_mahal_distances,
        'cv_details': cv_details
    }


def cross_validate_qda(
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = DEFAULT_CV_FOLDS,
    random_state: Optional[int] = None
) -> Dict[str, any]:
    """
    Cross-validation for QDA.

    Parameters
    ----------
    X : ndarray
        Feature matrix
    y : ndarray
        Class labels
    n_folds : int
        Number of CV folds
    random_state : int, optional
        Random seed

    Returns
    -------
    dict
        CV results
    """
    n_samples = len(y)
    classes = np.unique(y)
    fold_indices = create_cv_folds(n_samples, n_folds, random_state)

    y_pred_cv = np.zeros_like(y)
    cv_details = []

    for fold in range(n_folds):
        test_mask = fold_indices == fold
        train_mask = ~test_mask

        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]

        model = calculations.fit_qda(X_train, y_train)
        y_pred, _ = calculations.predict_qda(X_test, model)

        y_pred_cv[test_mask] = y_pred

        # Calculate fold metrics
        fold_metrics = calculate_classification_metrics(y_test, y_pred, classes=classes)

        # Calculate Mahalanobis distances for this fold as matrix
        mahal_dist_matrix = np.zeros((len(X_test), len(classes)))
        try:
            for class_idx, cls in enumerate(classes):
                class_mask = (y_train == cls)
                X_train_class = X_train[class_mask]

                if len(X_train_class) > 0:
                    # Calculate covariance and inverse
                    cov_matrix = np.cov(X_train_class.T)

                    # Handle singular matrix
                    try:
                        inv_cov = np.linalg.inv(cov_matrix)
                    except np.linalg.LinAlgError:
                        inv_cov = np.linalg.pinv(cov_matrix)

                    # Class centroid
                    centroid = np.mean(X_train_class, axis=0)

                    # Mahalanobis distance from test samples to this class centroid
                    diff = X_test - centroid
                    mahal_dist = np.sqrt(np.sum(diff @ inv_cov * diff, axis=1))
                    mahal_dist_matrix[:, class_idx] = mahal_dist
        except Exception:
            # If calculation fails, matrix remains zeros
            pass

        cv_details.append({
            'fold': fold,
            'accuracy': fold_metrics['accuracy'],
            'n_train': len(np.where(train_mask)[0]),
            'n_test': len(np.where(test_mask)[0]),
            'mahal_distances_matrix': mahal_dist_matrix,
            'test_indices': np.where(test_mask)[0]
        })

    metrics = calculate_classification_metrics(y, y_pred_cv, classes=classes)
    misclassified = np.where(y != y_pred_cv)[0]

    # Reconstruct full Mahalanobis distance matrix from folds
    all_mahal_distances = np.zeros((len(y), len(classes)))
    for fold_detail in cv_details:
        test_idx = fold_detail['test_indices']
        all_mahal_distances[test_idx] = fold_detail['mahal_distances_matrix']

    return {
        'predictions': y_pred_cv,
        'metrics': metrics,
        'misclassified_indices': misclassified,
        'n_misclassified': len(misclassified),
        'mahalanobis_distances': all_mahal_distances,
        'cv_details': cv_details
    }


def cross_validate_knn(
    X: np.ndarray,
    y: np.ndarray,
    k_values: List[int],
    metric: str = 'euclidean',
    n_folds: int = DEFAULT_CV_FOLDS,
    random_state: Optional[int] = None
) -> Dict[int, Dict[str, any]]:
    """
    Cross-validation for kNN with multiple k values.

    Parameters
    ----------
    X : ndarray
        Feature matrix
    y : ndarray
        Class labels
    k_values : list of int
        List of k values to test
    metric : str
        Distance metric
    n_folds : int
        Number of CV folds
    random_state : int, optional
        Random seed

    Returns
    -------
    dict
        Results for each k value
    """
    n_samples = len(y)
    fold_indices = create_cv_folds(n_samples, n_folds, random_state)

    results = {}

    for k in k_values:
        y_pred_cv = np.zeros_like(y)

        for fold in range(n_folds):
            test_mask = fold_indices == fold
            train_mask = ~test_mask

            X_train, y_train = X[train_mask], y[train_mask]
            X_test = X[test_mask]

            model = calculations.fit_knn(X_train, y_train, metric=metric)
            y_pred, _ = calculations.predict_knn(X_test, model, k=k)

            y_pred_cv[test_mask] = y_pred

        metrics = calculate_classification_metrics(y, y_pred_cv)
        misclassified = np.where(y != y_pred_cv)[0]

        results[k] = {
            'predictions': y_pred_cv,
            'metrics': metrics,
            'misclassified_indices': misclassified,
            'n_misclassified': len(misclassified)
        }

    return results


# ============================================================================
# Cross-Validation with PCA Preprocessing
# ============================================================================

def cross_validate_lda_with_pca(
    X: np.ndarray,
    y: np.ndarray,
    n_components_pca: int,
    n_folds: int = DEFAULT_CV_FOLDS,
    scaling_method: Optional[str] = None,
    groups: Optional[np.ndarray] = None,
    random_state: Optional[int] = None
) -> Dict[str, any]:
    """
    Cross-validation for LDA with PCA preprocessing.

    IMPORTANT: PCA is fitted independently for each fold ONLY on the training set
    of that fold. The evaluation set is NEVER included in PCA fitting - it is only
    projected onto the fitted PCA model. This ensures proper validation and prevents
    data leakage through preprocessing.

    Data Flow per Fold:
    1. Split data into train/eval indices
    2. FIT PCA on X_train ONLY -> pca_model
    3. PROJECT X_train onto PCA -> X_train_pca
    4. PROJECT X_eval onto PCA -> X_eval_pca (using same pca_model)
    5. FIT LDA on (X_train_pca, y_train)
    6. PREDICT on X_eval_pca
    7. Store predictions and metrics

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Feature matrix (should already be scaled if desired)
    y : ndarray of shape (n_samples,)
        Class labels
    n_components_pca : int
        Number of PCA components for preprocessing
    n_folds : int, default=5
        Number of CV folds
    scaling_method : str, optional
        Scaling method (for documentation purposes - data should already be scaled)
    groups : ndarray of shape (n_samples,), optional
        Cancellation groups - samples with same group stay together in train/test
    random_state : int, optional
        Random seed for reproducibility

    Returns
    -------
    dict
        CV results containing:
        - predictions : ndarray - out-of-fold predictions
        - y_true : ndarray - true labels (original order)
        - metrics : dict - classification metrics (accuracy, sensitivity, etc.)
        - cv_details : list - per-fold metrics and info
        - n_components_pca : int - components used
        - folds_info : dict - fold split information

    Examples
    --------
    >>> cv_results = cross_validate_lda_with_pca(
    ...     X_scaled, y, n_components_pca=5, n_folds=5
    ... )
    >>> print(f"Accuracy: {cv_results['metrics']['accuracy']:.2f}%")
    """
    n_samples = len(y)
    classes = np.unique(y)

    # Get fold indices using groups if provided
    folds_dict = create_stratified_cv_folds_with_groups(y, groups, n_folds, random_state)

    # Storage for predictions
    y_pred_cv = np.zeros_like(y)
    cv_details = []

    for fold in range(n_folds):
        train_idx = folds_dict[fold]['train_indices']
        eval_idx = folds_dict[fold]['test_indices']

        X_train, y_train = X[train_idx], y[train_idx]
        X_eval, y_eval = X[eval_idx], y[eval_idx]

        # Fit PCA on training data ONLY
        pca_model = calculations.fit_pca_preprocessor(X_train, n_components=n_components_pca)

        # Project both train and eval onto PCA
        X_train_pca = calculations.project_onto_pca(X_train, pca_model)
        X_eval_pca = calculations.project_onto_pca(X_eval, pca_model)

        # Fit LDA on projected training data
        lda_model = calculations.fit_lda(X_train_pca, y_train)

        # Predict on projected evaluation data
        y_pred, _ = calculations.predict_lda(X_eval_pca, lda_model)

        # Store predictions
        y_pred_cv[eval_idx] = y_pred

        # Calculate fold metrics
        fold_metrics = calculate_classification_metrics(y_eval, y_pred, classes=classes)

        # Calculate Mahalanobis distances for this fold as matrix
        mahal_dist_matrix = np.zeros((len(X_eval), len(classes)))
        try:
            for class_idx, cls in enumerate(classes):
                class_mask = (y_train == cls)
                X_train_class_pca = X_train_pca[class_mask]

                if len(X_train_class_pca) > 0:
                    # Calculate covariance and inverse
                    cov_matrix = np.cov(X_train_class_pca.T)

                    # Handle singular matrix
                    try:
                        inv_cov = np.linalg.inv(cov_matrix)
                    except np.linalg.LinAlgError:
                        inv_cov = np.linalg.pinv(cov_matrix)

                    # Class centroid
                    centroid = np.mean(X_train_class_pca, axis=0)

                    # Mahalanobis distance from eval samples to this class centroid
                    diff = X_eval_pca - centroid
                    mahal_dist = np.sqrt(np.sum(diff @ inv_cov * diff, axis=1))
                    mahal_dist_matrix[:, class_idx] = mahal_dist
        except Exception:
            # If calculation fails, matrix remains zeros
            pass

        cv_details.append({
            'fold': fold,
            'accuracy': fold_metrics['accuracy'],
            'n_train': len(train_idx),
            'n_eval': len(eval_idx),
            'variance_explained': float(pca_model['cumulative_variance_ratio'][-1]),
            'mahal_distances_matrix': mahal_dist_matrix,
            'eval_indices': eval_idx
        })

    # Calculate overall metrics
    metrics = calculate_classification_metrics(y, y_pred_cv, classes=classes)
    misclassified = np.where(y != y_pred_cv)[0]

    # Reconstruct full Mahalanobis distance matrix from folds
    all_mahal_distances = np.zeros((len(y), len(classes)))
    for fold_detail in cv_details:
        eval_idx = fold_detail['eval_indices']
        all_mahal_distances[eval_idx] = fold_detail['mahal_distances_matrix']

    # ✅ TRAIN FINAL PCA ON ALL DATA FOR EXPORT/DIAGNOSTICS
    final_pca_preprocessor = calculations.fit_pca_preprocessor(X, n_components=n_components_pca)

    return {
        'predictions': y_pred_cv,
        'y_true': y,
        'metrics': metrics,
        'cv_details': cv_details,
        'n_components_pca': n_components_pca,
        'misclassified_indices': misclassified,
        'n_misclassified': len(misclassified),
        'mahalanobis_distances': all_mahal_distances,
        'folds_info': folds_dict,
        'scaling_method': scaling_method,
        # ✅ ADD THESE THREE LINES:
        'pca_preprocessor': final_pca_preprocessor,
        'pca_loadings': final_pca_preprocessor.get('loadings'),
        'pca_variance_explained': final_pca_preprocessor.get('explained_variance_ratio'),
        'use_pca_preprocessing': True
    }


def cross_validate_qda_with_pca(
    X: np.ndarray,
    y: np.ndarray,
    n_components_pca: int,
    n_folds: int = DEFAULT_CV_FOLDS,
    scaling_method: Optional[str] = None,
    groups: Optional[np.ndarray] = None,
    random_state: Optional[int] = None
) -> Dict[str, any]:
    """
    Cross-validation for QDA with PCA preprocessing.

    Same logic as cross_validate_lda_with_pca but using QDA classifier.
    PCA is fitted independently per fold on training data only.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Feature matrix
    y : ndarray of shape (n_samples,)
        Class labels
    n_components_pca : int
        Number of PCA components
    n_folds : int, default=5
        Number of CV folds
    scaling_method : str, optional
        Scaling method (documentation)
    groups : ndarray, optional
        Cancellation groups
    random_state : int, optional
        Random seed

    Returns
    -------
    dict
        CV results (same format as cross_validate_lda_with_pca)
    """
    n_samples = len(y)
    classes = np.unique(y)

    folds_dict = create_stratified_cv_folds_with_groups(y, groups, n_folds, random_state)

    y_pred_cv = np.zeros_like(y)
    cv_details = []

    for fold in range(n_folds):
        train_idx = folds_dict[fold]['train_indices']
        eval_idx = folds_dict[fold]['test_indices']

        X_train, y_train = X[train_idx], y[train_idx]
        X_eval, y_eval = X[eval_idx], y[eval_idx]

        # Fit PCA on training data ONLY
        pca_model = calculations.fit_pca_preprocessor(X_train, n_components=n_components_pca)

        # Project both sets
        X_train_pca = calculations.project_onto_pca(X_train, pca_model)
        X_eval_pca = calculations.project_onto_pca(X_eval, pca_model)

        # Fit QDA on projected training data
        qda_model = calculations.fit_qda(X_train_pca, y_train)

        # Predict on projected evaluation data
        y_pred, _ = calculations.predict_qda(X_eval_pca, qda_model)

        y_pred_cv[eval_idx] = y_pred

        fold_metrics = calculate_classification_metrics(y_eval, y_pred, classes=classes)

        # Calculate Mahalanobis distances for this fold as matrix
        mahal_dist_matrix = np.zeros((len(X_eval), len(classes)))
        try:
            for class_idx, cls in enumerate(classes):
                class_mask = (y_train == cls)
                X_train_class_pca = X_train_pca[class_mask]

                if len(X_train_class_pca) > 0:
                    # Calculate covariance and inverse
                    cov_matrix = np.cov(X_train_class_pca.T)

                    # Handle singular matrix
                    try:
                        inv_cov = np.linalg.inv(cov_matrix)
                    except np.linalg.LinAlgError:
                        inv_cov = np.linalg.pinv(cov_matrix)

                    # Class centroid
                    centroid = np.mean(X_train_class_pca, axis=0)

                    # Mahalanobis distance from eval samples to this class centroid
                    diff = X_eval_pca - centroid
                    mahal_dist = np.sqrt(np.sum(diff @ inv_cov * diff, axis=1))
                    mahal_dist_matrix[:, class_idx] = mahal_dist
        except Exception:
            # If calculation fails, matrix remains zeros
            pass

        cv_details.append({
            'fold': fold,
            'accuracy': fold_metrics['accuracy'],
            'n_train': len(train_idx),
            'n_eval': len(eval_idx),
            'variance_explained': float(pca_model['cumulative_variance_ratio'][-1]),
            'mahal_distances_matrix': mahal_dist_matrix,
            'eval_indices': eval_idx
        })

    metrics = calculate_classification_metrics(y, y_pred_cv, classes=classes)
    misclassified = np.where(y != y_pred_cv)[0]

    # Reconstruct full Mahalanobis distance matrix from folds
    all_mahal_distances = np.zeros((len(y), len(classes)))
    for fold_detail in cv_details:
        eval_idx = fold_detail['eval_indices']
        all_mahal_distances[eval_idx] = fold_detail['mahal_distances_matrix']

    # ✅ TRAIN FINAL PCA ON ALL DATA FOR EXPORT/DIAGNOSTICS
    final_pca_preprocessor = calculations.fit_pca_preprocessor(X, n_components=n_components_pca)

    return {
        'predictions': y_pred_cv,
        'y_true': y,
        'metrics': metrics,
        'cv_details': cv_details,
        'n_components_pca': n_components_pca,
        'misclassified_indices': misclassified,
        'n_misclassified': len(misclassified),
        'mahalanobis_distances': all_mahal_distances,
        'folds_info': folds_dict,
        'scaling_method': scaling_method,
        # ✅ ADD THESE THREE LINES:
        'pca_preprocessor': final_pca_preprocessor,
        'pca_loadings': final_pca_preprocessor.get('loadings'),
        'pca_variance_explained': final_pca_preprocessor.get('explained_variance_ratio'),
        'use_pca_preprocessing': True
    }


def cross_validate_knn_with_pca(
    X: np.ndarray,
    y: np.ndarray,
    n_components_pca: int,
    k_values: Optional[List[int]] = None,
    metric: str = 'euclidean',
    n_folds: int = DEFAULT_CV_FOLDS,
    scaling_method: Optional[str] = None,
    groups: Optional[np.ndarray] = None,
    random_state: Optional[int] = None
) -> Dict[str, any]:
    """
    Cross-validation for kNN with PCA preprocessing.

    Tests multiple k values if provided. PCA is fitted independently per fold
    on training data only.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Feature matrix
    y : ndarray of shape (n_samples,)
        Class labels
    n_components_pca : int
        Number of PCA components
    k_values : list of int, optional
        List of k values to test. If None, uses [1, 3, 5, 7]
    metric : str, default='euclidean'
        Distance metric for kNN
    n_folds : int, default=5
        Number of CV folds
    scaling_method : str, optional
        Scaling method (documentation)
    groups : ndarray, optional
        Cancellation groups
    random_state : int, optional
        Random seed

    Returns
    -------
    dict
        CV results containing:
        - predictions : ndarray - predictions using best k
        - y_true : ndarray - true labels
        - metrics : dict - metrics using best k
        - cv_details : list - per-fold info
        - n_components_pca : int
        - best_k : int - optimal k value
        - k_results : dict - results for each k value tested
    """
    if k_values is None:
        k_values = [1, 3, 5, 7]

    n_samples = len(y)
    classes = np.unique(y)

    folds_dict = create_stratified_cv_folds_with_groups(y, groups, n_folds, random_state)

    # Storage for each k value
    k_results = {}

    for k in k_values:
        y_pred_cv = np.zeros_like(y)
        cv_details = []

        for fold in range(n_folds):
            train_idx = folds_dict[fold]['train_indices']
            eval_idx = folds_dict[fold]['test_indices']

            X_train, y_train = X[train_idx], y[train_idx]
            X_eval, y_eval = X[eval_idx], y[eval_idx]

            # Fit PCA on training data ONLY
            pca_model = calculations.fit_pca_preprocessor(X_train, n_components=n_components_pca)

            # Project both sets
            X_train_pca = calculations.project_onto_pca(X_train, pca_model)
            X_eval_pca = calculations.project_onto_pca(X_eval, pca_model)

            # Fit kNN on projected training data
            knn_model = calculations.fit_knn(X_train_pca, y_train, metric=metric)

            # Predict on projected evaluation data
            y_pred, _ = calculations.predict_knn(X_eval_pca, knn_model, k=k)

            y_pred_cv[eval_idx] = y_pred

            fold_metrics = calculate_classification_metrics(y_eval, y_pred, classes=classes)
            cv_details.append({
                'fold': fold,
                'accuracy': fold_metrics['accuracy'],
                'n_train': len(train_idx),
                'n_eval': len(eval_idx),
                'variance_explained': float(pca_model['cumulative_variance_ratio'][-1])
            })

        metrics = calculate_classification_metrics(y, y_pred_cv, classes=classes)
        misclassified = np.where(y != y_pred_cv)[0]

        k_results[k] = {
            'predictions': y_pred_cv,
            'metrics': metrics,
            'cv_details': cv_details,
            'misclassified_indices': misclassified,
            'n_misclassified': len(misclassified)
        }

    # Find best k
    best_k = max(k_results.keys(), key=lambda k: k_results[k]['metrics']['accuracy'])

    # ✅ TRAIN FINAL PCA ON ALL DATA FOR EXPORT/DIAGNOSTICS
    final_pca_preprocessor = calculations.fit_pca_preprocessor(X, n_components=n_components_pca)

    return {
        'predictions': k_results[best_k]['predictions'],
        'y_true': y,
        'metrics': k_results[best_k]['metrics'],
        'cv_details': k_results[best_k]['cv_details'],
        'n_components_pca': n_components_pca,
        'misclassified_indices': k_results[best_k]['misclassified_indices'],
        'n_misclassified': k_results[best_k]['n_misclassified'],
        'best_k': best_k,
        'k_results': k_results,
        'metric': metric,
        'folds_info': folds_dict,
        'scaling_method': scaling_method,
        # ✅ ADD THESE THREE LINES:
        'pca_preprocessor': final_pca_preprocessor,
        'pca_loadings': final_pca_preprocessor.get('loadings'),
        'pca_variance_explained': final_pca_preprocessor.get('explained_variance_ratio'),
        'use_pca_preprocessing': True
    }


def cross_validate_simca(
    X: np.ndarray,
    y: np.ndarray,
    n_components: int,
    confidence_level: float = 0.95,
    n_folds: int = DEFAULT_CV_FOLDS,
    random_state: Optional[int] = None
) -> Dict[str, any]:
    """
    Cross-validation for SIMCA.

    Parameters
    ----------
    X : ndarray
        Feature matrix
    y : ndarray
        Class labels
    n_components : int
        Number of PCs per class
    confidence_level : float
        Confidence level
    n_folds : int
        Number of CV folds
    random_state : int, optional
        Random seed

    Returns
    -------
    dict
        CV results
    """
    n_samples = len(y)
    classes = np.unique(y)
    fold_indices = create_cv_folds(n_samples, n_folds, random_state)

    # Store acceptance for each fold
    acceptance_all = {cls: np.zeros(n_samples, dtype=int) for cls in classes}

    for fold in range(n_folds):
        test_mask = fold_indices == fold
        train_mask = ~test_mask

        X_train, y_train = X[train_mask], y[train_mask]
        X_test = X[test_mask]

        model = calculations.fit_simca(X_train, y_train, n_components, confidence_level)
        _, acceptance = calculations.predict_simca(X_test, model)

        for cls in classes:
            acceptance_all[cls][test_mask] = acceptance[cls]

    # Calculate metrics
    metrics = calculate_simca_uneq_metrics(acceptance_all, y, classes)

    return {
        'acceptance': acceptance_all,
        'metrics': metrics
    }


def cross_validate_uneq(
    X: np.ndarray,
    y: np.ndarray,
    n_components: Optional[int] = None,
    use_pca: bool = False,
    confidence_level: float = 0.95,
    n_folds: int = DEFAULT_CV_FOLDS,
    random_state: Optional[int] = None
) -> Dict[str, any]:
    """
    Cross-validation for UNEQ.

    Parameters
    ----------
    X : ndarray
        Feature matrix
    y : ndarray
        Class labels
    n_components : int, optional
        Number of PCs (if using PCA)
    use_pca : bool
        Whether to use PCA
    confidence_level : float
        Confidence level
    n_folds : int
        Number of CV folds
    random_state : int, optional
        Random seed

    Returns
    -------
    dict
        CV results
    """
    n_samples = len(y)
    classes = np.unique(y)
    fold_indices = create_cv_folds(n_samples, n_folds, random_state)

    acceptance_all = {cls: np.zeros(n_samples, dtype=int) for cls in classes}

    for fold in range(n_folds):
        test_mask = fold_indices == fold
        train_mask = ~test_mask

        X_train, y_train = X[train_mask], y[train_mask]
        X_test = X[test_mask]
        y_test = y[test_mask]

        model = calculations.fit_uneq(X_train, y_train, n_components, confidence_level, use_pca)
        _, acceptance = calculations.predict_uneq(X_test, model, y_test)

        for cls in classes:
            acceptance_all[cls][test_mask] = acceptance[cls]

    metrics = calculate_simca_uneq_metrics(acceptance_all, y, classes)

    return {
        'acceptance': acceptance_all,
        'metrics': metrics
    }


def find_best_k(
    cv_results: Dict[int, Dict[str, any]],
    metric: str = 'accuracy'
) -> Tuple[int, float]:
    """
    Find best k value from kNN cross-validation results.

    Parameters
    ----------
    cv_results : dict
        Results from cross_validate_knn
    metric : str
        Metric to optimize: 'accuracy', 'avg_efficiency'

    Returns
    -------
    best_k : int
        Best k value
    best_score : float
        Best metric score
    """
    k_values = []
    scores = []

    for k, results in cv_results.items():
        k_values.append(k)
        scores.append(results['metrics'][metric])

    best_idx = np.argmax(scores)
    best_k = k_values[best_idx]
    best_score = scores[best_idx]

    return best_k, best_score


def compare_models(
    results_dict: Dict[str, Dict[str, any]],
    metric: str = 'accuracy'
) -> pd.DataFrame:
    """
    Compare performance of multiple models.

    Parameters
    ----------
    results_dict : dict
        Dictionary mapping model names to result dictionaries
    metric : str
        Metric to compare

    Returns
    -------
    DataFrame
        Comparison table
    """
    comparison_data = []

    for model_name, results in results_dict.items():
        if 'metrics' in results:
            metrics = results['metrics']
            row = {
                'Model': model_name,
                'Accuracy': metrics.get('accuracy', np.nan),
                'Avg Sensitivity': metrics.get('avg_sensitivity', np.nan),
                'Avg Specificity': metrics.get('avg_specificity', np.nan),
                'Avg Efficiency': metrics.get('avg_efficiency', np.nan)
            }
            comparison_data.append(row)

    df = pd.DataFrame(comparison_data)

    if not df.empty:
        # Sort by specified metric (descending)
        metric_col_map = {
            'accuracy': 'Accuracy',
            'avg_sensitivity': 'Avg Sensitivity',
            'avg_specificity': 'Avg Specificity',
            'avg_efficiency': 'Avg Efficiency'
        }
        sort_col = metric_col_map.get(metric, 'Accuracy')
        df = df.sort_values(sort_col, ascending=False).reset_index(drop=True)

    return df


def get_misclassified_samples(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_names: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Get information about misclassified samples.

    Parameters
    ----------
    y_true : ndarray
        True labels
    y_pred : ndarray
        Predicted labels
    sample_names : list of str, optional
        Sample names/labels

    Returns
    -------
    DataFrame
        Misclassified samples with true and predicted labels
    """
    misclassified_mask = y_true != y_pred
    misclassified_indices = np.where(misclassified_mask)[0]

    if sample_names is None:
        sample_names = [f"Sample_{i+1}" for i in range(len(y_true))]

    data = []
    for idx in misclassified_indices:
        data.append({
            'Index': idx,
            'Sample': sample_names[idx],
            'True_Class': y_true[idx],
            'Predicted_Class': y_pred[idx]
        })

    return pd.DataFrame(data)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[np.ndarray] = None
) -> Dict[str, any]:
    """
    Compute comprehensive classification metrics following CAT-Chemometrics standards.

    Calculates confusion matrix (k×k) where [i,j] = count(true=i, pred=j).
    For each class computes TP, FN, FP, TN and derives:
    - Sensitivity = TP/(TP+FN) [True Positive Rate]
    - Specificity = TN/(TN+FP) [True Negative Rate]
    - Precision = TP/(TP+FP) [Positive Predictive Value]
    - F1 = 2*(Precision*Sensitivity)/(Precision+Sensitivity)

    All metrics are expressed as percentages.

    Parameters
    ----------
    y_true : ndarray of shape (n_samples,)
        True class labels
    y_pred : ndarray of shape (n_samples,)
        Predicted class labels
    class_names : ndarray, optional
        Class names in desired order. If None, uses unique values from y_true and y_pred

    Returns
    -------
    dict
        Dictionary containing:
        - accuracy : float
            Overall accuracy as percentage (trace/total * 100)
        - macro_f1 : float
            Macro-averaged F1 score (mean of per-class F1 scores)
        - weighted_f1 : float
            Weighted F1 score (mean of F1 * class_fraction)
        - average_sensitivity : float
            Average sensitivity across all classes
        - average_specificity : float
            Average specificity across all classes
        - confusion_matrix : ndarray of shape (k, k)
            Confusion matrix where [i,j] = count(true=i, pred=j)
        - sensitivity_per_class : dict
            Sensitivity for each class (as percentage)
        - specificity_per_class : dict
            Specificity for each class (as percentage)
        - precision_per_class : dict
            Precision for each class (as percentage)
        - f1_per_class : dict
            F1-score for each class (as percentage)
        - class_counts : dict
            Number of samples per class in y_true
        - misclassified_samples : ndarray
            Indices of misclassified samples
        - n_misclassified : int
            Total number of misclassified samples

    Notes
    -----
    - Division by zero is handled by setting metric to 0.0
    - Follows standard ML metrics definitions
    - Compatible with CAT-Chemometrics R implementations

    References
    ----------
    Standard machine learning metrics (scikit-learn conventions)

    Examples
    --------
    >>> y_true = np.array([0, 0, 1, 1, 2, 2])
    >>> y_pred = np.array([0, 1, 1, 1, 2, 0])
    >>> metrics = compute_classification_metrics(y_true, y_pred)
    >>> print(f"Accuracy: {metrics['accuracy']:.2f}%")
    Accuracy: 66.67%
    """
    # Validate inputs
    if y_true is None or len(y_true) == 0:
        raise ValueError("y_true is empty or None. Cannot calculate metrics.")

    if y_pred is None or len(y_pred) == 0:
        raise ValueError("y_pred is empty or None. Cannot calculate metrics.")

    # Convert to numpy arrays for consistent type handling
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    # Determine class labels
    if class_names is None:
        class_names = np.unique(np.concatenate([y_true_arr, y_pred_arr]))
    else:
        class_names = np.asarray(class_names)
        # Filter classes to only those actually present in y_true
        class_names = np.intersect1d(class_names, y_true_arr)

    if len(class_names) == 0:
        import streamlit as st
        st.error(f"❌ DEBUG: class_names is EMPTY in compute_classification_metrics!")
        st.write(f"  - class_names parameter: {class_names if class_names is not None else 'None'}")
        st.write(f"  - unique values in y_true: {np.unique(y_true_arr)}")
        st.write(f"  - unique values in y_pred: {np.unique(y_pred_arr)}")
        st.write(f"  - y_true dtype: {y_true_arr.dtype}")
        raise ValueError(
            f"No classes found. "
            f"y_true contains: {np.unique(y_true_arr).tolist()}, "
            f"y_pred contains: {np.unique(y_pred_arr).tolist()}"
        )

    n_classes = len(class_names)
    n_samples = len(y_true_arr)

    # Create confusion matrix: [i,j] = count(true=i, pred=j)
    conf_matrix = confusion_matrix(y_true_arr, y_pred_arr, labels=class_names)

    # Overall accuracy: trace(conf_matrix) / total
    accuracy = np.trace(conf_matrix) / n_samples * 100.0

    # Initialize per-class metrics
    sensitivity_per_class = {}
    specificity_per_class = {}
    precision_per_class = {}
    f1_per_class = {}
    class_counts = {}

    # Arrays for averaging
    sensitivity_values = []
    specificity_values = []
    f1_values = []
    class_fractions = []

    # Compute per-class metrics
    for i, cls in enumerate(class_names):
        # True Positives: diagonal element
        TP = conf_matrix[i, i]

        # False Negatives: row sum minus TP
        FN = conf_matrix[i, :].sum() - TP

        # False Positives: column sum minus TP
        FP = conf_matrix[:, i].sum() - TP

        # True Negatives: total minus TP, FN, FP
        TN = conf_matrix.sum() - TP - FN - FP

        # Class count
        class_count = int(conf_matrix[i, :].sum())
        class_counts[cls] = class_count
        class_fractions.append(class_count / n_samples)

        # Sensitivity = TP / (TP + FN)
        if (TP + FN) > 0:
            sensitivity = TP / (TP + FN) * 100.0
        else:
            sensitivity = 0.0
        sensitivity_per_class[cls] = sensitivity
        sensitivity_values.append(sensitivity)

        # Specificity = TN / (TN + FP)
        if (TN + FP) > 0:
            specificity = TN / (TN + FP) * 100.0
        else:
            specificity = 0.0
        specificity_per_class[cls] = specificity
        specificity_values.append(specificity)

        # Precision = TP / (TP + FP)
        if (TP + FP) > 0:
            precision = TP / (TP + FP) * 100.0
        else:
            precision = 0.0
        precision_per_class[cls] = precision

        # F1 = 2 * (Precision * Sensitivity) / (Precision + Sensitivity)
        # Convert back from percentage for calculation
        prec_frac = precision / 100.0
        sens_frac = sensitivity / 100.0

        if (prec_frac + sens_frac) > 0:
            f1 = 2 * (prec_frac * sens_frac) / (prec_frac + sens_frac) * 100.0
        else:
            f1 = 0.0
        f1_per_class[cls] = f1
        f1_values.append(f1)

    # Macro-averaged F1: mean of per-class F1 scores
    macro_f1 = np.mean(f1_values)

    # Weighted F1: mean of (F1 * class_fraction)
    weighted_f1 = np.sum(np.array(f1_values) * np.array(class_fractions))

    # Average sensitivity and specificity
    average_sensitivity = np.mean(sensitivity_values)
    average_specificity = np.mean(specificity_values)

    # Misclassified samples
    misclassified_samples = np.where(y_true != y_pred)[0]
    n_misclassified = len(misclassified_samples)

    return {
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'average_sensitivity': average_sensitivity,
        'average_specificity': average_specificity,
        'confusion_matrix': conf_matrix,
        'sensitivity_per_class': sensitivity_per_class,
        'specificity_per_class': specificity_per_class,
        'precision_per_class': precision_per_class,
        'f1_per_class': f1_per_class,
        'class_counts': class_counts,
        'misclassified_samples': misclassified_samples,
        'n_misclassified': n_misclassified,
        'classes': class_names
    }


def cross_validate_classifier(
    X: np.ndarray,
    y: np.ndarray,
    classifier_type: str,
    n_folds: int = 5,
    classifier_params: Optional[Dict[str, any]] = None,
    random_state: Optional[int] = None
) -> Dict[str, any]:
    """
    Perform stratified K-fold cross-validation for any classifier.

    Uses stratified K-fold CV to split data while maintaining class distribution.
    For each fold, trains on (n_folds-1) folds and predicts on held-out fold.
    Collects out-of-fold predictions and computes comprehensive metrics.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Feature matrix
    y : ndarray of shape (n_samples,)
        Class labels
    classifier_type : str
        Type of classifier: 'lda', 'qda', 'knn', 'simca', 'uneq'
    n_folds : int, default=5
        Number of cross-validation folds
    classifier_params : dict, optional
        Classifier-specific parameters:
        - For kNN: {'k': int, 'metric': str}
        - For SIMCA: {'n_components': int, 'confidence_level': float}
        - For UNEQ: {'n_components': int, 'confidence_level': float, 'use_pca': bool}
        - For LDA/QDA: None or {}
    random_state : int, optional
        Random seed for reproducibility

    Returns
    -------
    dict
        Dictionary containing:
        - cv_predictions : ndarray of shape (n_samples,)
            Out-of-fold predictions in original sample order
        - cv_accuracy : float
            Average cross-validation accuracy (percentage)
        - cv_sensitivity_per_class : dict
            Average sensitivity for each class (percentage)
        - cv_specificity_per_class : dict
            Average specificity for each class (percentage)
        - cv_f1_per_class : dict
            Average F1-score for each class (percentage)
        - cv_confusion_matrix : ndarray
            Aggregated confusion matrix across all folds
        - fold_results : list of dict
            Results for each fold containing:
            - fold : int (fold number)
            - accuracy : float
            - sensitivity_per_class : dict
            - specificity_per_class : dict
            - confusion_matrix : ndarray
        - n_folds : int
            Number of folds used
        - fold_assignments : ndarray of shape (n_samples,)
            Fold assignment for each sample (0 to n_folds-1)
        - classifier_type : str
            Type of classifier used
        - classifier_params : dict
            Parameters used for classifier

    Notes
    -----
    - Uses StratifiedKFold to maintain class proportions in each fold
    - Out-of-fold predictions are in original sample order for reproducibility
    - Fold assignments stored for tracking which samples were in which fold
    - Compatible with CAT-Chemometrics _rnd.r routines

    References
    ----------
    CAT-R _rnd.r routines for randomized cross-validation

    Examples
    --------
    >>> X = np.random.randn(100, 5)
    >>> y = np.array([0]*50 + [1]*50)
    >>> results = cross_validate_classifier(X, y, 'lda', n_folds=5)
    >>> print(f"CV Accuracy: {results['cv_accuracy']:.2f}%")
    CV Accuracy: 85.00%
    """
    # Validate inputs
    if classifier_params is None:
        classifier_params = {}

    classifier_type = classifier_type.lower()
    valid_classifiers = ['lda', 'qda', 'knn', 'simca', 'uneq']

    if classifier_type not in valid_classifiers:
        raise ValueError(
            f"classifier_type must be one of {valid_classifiers}, got '{classifier_type}'"
        )

    n_samples = len(y)
    classes = np.unique(y)
    n_classes = len(classes)

    # Create stratified fold assignments (maintains class proportions)
    fold_assignments = create_stratified_cv_folds(y, n_folds, random_state)

    # Initialize storage for out-of-fold predictions
    cv_predictions = np.zeros_like(y)

    # Storage for per-fold results
    fold_results = []

    # Per-class metrics across folds
    fold_sensitivity = {cls: [] for cls in classes}
    fold_specificity = {cls: [] for cls in classes}
    fold_f1 = {cls: [] for cls in classes}

    # Aggregated confusion matrix
    cv_confusion_matrix = np.zeros((n_classes, n_classes), dtype=int)

    # Iterate through folds
    for fold in range(n_folds):
        # Create train/test split
        test_mask = fold_assignments == fold
        train_mask = ~test_mask

        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]

        # Train classifier based on type
        if classifier_type == 'lda':
            model = calculations.fit_lda(X_train, y_train)
            y_pred, _ = calculations.predict_lda(X_test, model)

        elif classifier_type == 'qda':
            model = calculations.fit_qda(X_train, y_train)
            y_pred, _ = calculations.predict_qda(X_test, model)

        elif classifier_type == 'knn':
            k = classifier_params.get('k', 3)
            metric = classifier_params.get('metric', 'euclidean')
            model = calculations.fit_knn(X_train, y_train, metric=metric)
            y_pred, _ = calculations.predict_knn(X_test, model, k=k)

        elif classifier_type == 'simca':
            n_components = classifier_params.get('n_components', 2)
            confidence_level = classifier_params.get('confidence_level', 0.95)
            model = calculations.fit_simca(X_train, y_train, n_components, confidence_level)
            # Use detailed prediction to get predicted classes
            pred_detailed = calculations.predict_simca_detailed(X_test, model)
            y_pred = pred_detailed['predicted_classes']

        elif classifier_type == 'uneq':
            n_components = classifier_params.get('n_components', None)
            confidence_level = classifier_params.get('confidence_level', 0.95)
            use_pca = classifier_params.get('use_pca', False)
            model = calculations.fit_uneq(X_train, y_train, n_components, confidence_level, use_pca)
            # Use detailed prediction to get predicted classes
            pred_detailed = calculations.predict_uneq_detailed(X_test, model)
            y_pred = pred_detailed['predicted_classes']

        # Store out-of-fold predictions
        cv_predictions[test_mask] = y_pred

        # Compute fold metrics
        fold_metrics = compute_classification_metrics(y_test, y_pred, class_names=classes)

        # Store per-fold results
        fold_result = {
            'fold': fold,
            'accuracy': fold_metrics['accuracy'],
            'sensitivity_per_class': fold_metrics['sensitivity_per_class'],
            'specificity_per_class': fold_metrics['specificity_per_class'],
            'f1_per_class': fold_metrics['f1_per_class'],
            'confusion_matrix': fold_metrics['confusion_matrix']
        }
        fold_results.append(fold_result)

        # Accumulate per-class metrics
        for cls in classes:
            # Use .get() with default 0.0 for classes not in this fold's test set
            fold_sensitivity[cls].append(fold_metrics['sensitivity_per_class'].get(cls, 0.0))
            fold_specificity[cls].append(fold_metrics['specificity_per_class'].get(cls, 0.0))
            fold_f1[cls].append(fold_metrics['f1_per_class'].get(cls, 0.0))

        # Accumulate confusion matrix
        cv_confusion_matrix += fold_metrics['confusion_matrix']

    # Compute overall CV metrics from out-of-fold predictions
    overall_metrics = compute_classification_metrics(y, cv_predictions, class_names=classes)

    # Average per-class metrics across folds
    cv_sensitivity_per_class = {cls: np.mean(fold_sensitivity[cls]) for cls in classes}
    cv_specificity_per_class = {cls: np.mean(fold_specificity[cls]) for cls in classes}
    cv_f1_per_class = {cls: np.mean(fold_f1[cls]) for cls in classes}

    return {
        'cv_predictions': cv_predictions,
        'cv_accuracy': overall_metrics['accuracy'],
        'cv_sensitivity_per_class': cv_sensitivity_per_class,
        'cv_specificity_per_class': cv_specificity_per_class,
        'cv_f1_per_class': cv_f1_per_class,
        'cv_confusion_matrix': cv_confusion_matrix,
        'fold_results': fold_results,
        'n_folds': n_folds,
        'fold_assignments': fold_assignments,
        'classifier_type': classifier_type,
        'classifier_params': classifier_params
    }


# ============================================================================
# kNN Neighbor Analysis Functions
# ============================================================================

def analyze_sample_neighbors_by_k(
    sample: np.ndarray,
    model: Dict,
    k_max: int = 7,
    class_labels: Optional[np.ndarray] = None
) -> pd.DataFrame:
    """
    Analyze how class composition of k-nearest neighbors changes as k increases.

    For a given sample, compute for each k value (1 to k_max):
    - Number of neighbors from each class
    - Predicted class at that k
    - Confidence at that k

    Parameters
    ----------
    sample : ndarray of shape (1, n_features) or (n_features,)
        Single sample to analyze
    model : dict
        Trained kNN model from fit_knn()
    k_max : int
        Maximum k value to test (default 7)
    class_labels : ndarray
        Class labels (used for labeling). If None, uses model['classes']

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - k: k value (1 to k_max)
        - For each class: '{class}_count' (number of neighbors from that class)
        - 'predicted_class': Class predicted at this k
        - 'confidence': k_votes / k (prediction confidence)
        - 'votes_by_class': dict showing votes for each class

    Examples
    --------
    >>> analysis = analyze_sample_neighbors_by_k(
    ...     sample=X_test[0],
    ...     model=knn_model,
    ...     k_max=7,
    ...     class_labels=['A', 'B']
    ... )
    >>> print(analysis)
       k  A_count  B_count  predicted_class  confidence
    0  1        1        0              A          1.0
    1  2        1        1              A          0.5
    2  3        2        1              A          0.67
    3  4        2        2              A          0.5
    ...
    """
    from scipy.spatial.distance import cdist

    # Ensure sample is 2D
    if sample.ndim == 1:
        sample = sample.reshape(1, -1)

    # Get class labels from model if not provided
    if class_labels is None:
        class_labels = model['classes']

    n_classes = len(class_labels)

    # Calculate distances to all training samples
    distances = cdist(
        sample,
        model['X_train'],
        metric=model.get('metric', 'euclidean')
    )[0]  # Get first row (our sample)

    # Sort by distance to find nearest neighbors
    sorted_indices = np.argsort(distances)
    sorted_labels = model['y_train'][sorted_indices]

    # Analyze for each k
    results = []

    for k in range(1, k_max + 1):
        # Get k nearest neighbors
        k_neighbor_labels = sorted_labels[:k]

        # Count votes for each class
        votes_by_class = {}
        predictions_by_class = np.zeros(n_classes)

        for i, cls in enumerate(class_labels):
            count = np.sum(k_neighbor_labels == cls)
            votes_by_class[cls] = int(count)
            predictions_by_class[i] = count

        # Determine predicted class (max votes)
        predicted_class = class_labels[np.argmax(predictions_by_class)]
        confidence = np.max(predictions_by_class) / k

        # Build row
        row = {
            'k': k,
            'predicted_class': predicted_class,
            'confidence': confidence,
            'votes_by_class': votes_by_class
        }

        # Add count for each class
        for cls in class_labels:
            row[f'{cls}_count'] = votes_by_class[cls]

        results.append(row)

    df = pd.DataFrame(results)
    return df


def get_sample_metadata(
    sample_idx: int,
    X_data: np.ndarray,
    y_data: np.ndarray,
    feature_names: Optional[list] = None,
    class_labels: Optional[np.ndarray] = None
) -> Dict:
    """
    Get metadata about a specific sample (true class, feature values, etc.)

    Parameters
    ----------
    sample_idx : int
        Index of sample
    X_data : ndarray
        Feature matrix
    y_data : ndarray
        Class labels
    feature_names : list
        Feature names (optional)
    class_labels : ndarray
        Class label names

    Returns
    -------
    dict
        Sample metadata: true_class, index, feature_values, etc.
    """
    true_class = y_data[sample_idx]
    sample = X_data[sample_idx]

    if feature_names is None:
        feature_names = [f"Feature {i+1}" for i in range(len(sample))]

    return {
        'index': sample_idx,
        'true_class': true_class,
        'sample': sample,
        'feature_values': dict(zip(feature_names, sample)),
        'sample_id': f"Sample {sample_idx}: True={true_class}"
    }
