"""
Data Preprocessing for Classification
======================================

Functions for preparing data for classification analysis including:
- Data validation and cleaning
- Scaling and centering
- Train/test splitting
- Class balance checking
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional, Union
from .config import MIN_SAMPLES_PER_CLASS, MAX_CLASSES, PCA_VARIANCE_THRESHOLD


def validate_classification_data(
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    min_samples_per_class: int = MIN_SAMPLES_PER_CLASS
) -> Dict[str, any]:
    """
    Validate classification dataset and return statistics.

    Parameters
    ----------
    X : DataFrame or ndarray
        Feature matrix
    y : Series or ndarray
        Class labels
    min_samples_per_class : int
        Minimum required samples per class

    Returns
    -------
    dict
        Dictionary with validation results and statistics
    """
    # Convert to arrays if needed
    if isinstance(X, pd.DataFrame):
        X_arr = X.values
        feature_names = X.columns.tolist()
    else:
        X_arr = np.array(X)
        feature_names = [f"Var{i+1}" for i in range(X_arr.shape[1])]

    if isinstance(y, pd.Series):
        y_arr = y.values
    else:
        y_arr = np.array(y)

    # Get unique classes and counts
    unique_classes, class_counts = np.unique(y_arr, return_counts=True)
    n_classes = len(unique_classes)
    n_samples, n_features = X_arr.shape

    # Validation checks
    validation_results = {
        'valid': True,
        'messages': [],
        'warnings': [],
        'n_samples': n_samples,
        'n_features': n_features,
        'n_classes': n_classes,
        'classes': unique_classes.tolist(),
        'class_counts': dict(zip(unique_classes, class_counts)),
        'feature_names': feature_names
    }

    # Check for missing values
    if isinstance(X, pd.DataFrame):
        n_missing = X.isnull().sum().sum()
    else:
        n_missing = np.isnan(X_arr).sum()

    if n_missing > 0:
        validation_results['valid'] = False
        validation_results['messages'].append(
            f"Dataset contains {n_missing} missing values. Please handle missing data first."
        )

    # Check number of classes
    if n_classes < 2:
        validation_results['valid'] = False
        validation_results['messages'].append(
            "Need at least 2 classes for classification."
        )

    if n_classes > MAX_CLASSES:
        validation_results['warnings'].append(
            f"Dataset has {n_classes} classes. Consider reducing for better performance."
        )

    # Check samples per class
    for cls, count in zip(unique_classes, class_counts):
        if count < min_samples_per_class:
            validation_results['valid'] = False
            validation_results['messages'].append(
                f"Class '{cls}' has only {count} samples. Need at least {min_samples_per_class}."
            )

    # Check feature/sample ratio
    if n_features >= n_samples:
        validation_results['warnings'].append(
            f"Number of features ({n_features}) >= number of samples ({n_samples}). "
            "Consider dimensionality reduction for some methods (LDA, QDA)."
        )

    # Check for constant features
    if isinstance(X, pd.DataFrame):
        const_features = (X.std() == 0).sum()
    else:
        const_features = (X_arr.std(axis=0) == 0).sum()

    if const_features > 0:
        validation_results['warnings'].append(
            f"{const_features} constant features detected. Consider removing them."
        )

    return validation_results


def scale_data(
    X: Union[pd.DataFrame, np.ndarray],
    method: str = 'autoscale',
    center: bool = True,
    scale: bool = True,
    reference_mean: Optional[np.ndarray] = None,
    reference_std: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Scale and/or center data.

    Parameters
    ----------
    X : DataFrame or ndarray
        Data to scale
    method : str
        Scaling method: 'autoscale', 'center', 'scale', 'none'
    center : bool
        Whether to center data (subtract mean)
    scale : bool
        Whether to scale data (divide by std)
    reference_mean : ndarray, optional
        Mean values to use for centering (for test set)
    reference_std : ndarray, optional
        Std values to use for scaling (for test set)

    Returns
    -------
    X_scaled : ndarray
        Scaled data
    mean_values : ndarray
        Mean values used
    std_values : ndarray
        Std values used
    """
    # Convert to array if needed
    if isinstance(X, pd.DataFrame):
        X_arr = X.values
    else:
        X_arr = np.array(X, dtype=float)

    # Calculate or use reference statistics
    if reference_mean is None:
        mean_values = np.mean(X_arr, axis=0) if center else np.zeros(X_arr.shape[1])
    else:
        mean_values = reference_mean

    if reference_std is None:
        std_values = np.std(X_arr, axis=0, ddof=1) if scale else np.ones(X_arr.shape[1])
        # Avoid division by zero
        std_values[std_values == 0] = 1.0
    else:
        std_values = reference_std

    # Apply scaling based on method
    if method == 'autoscale':
        X_scaled = (X_arr - mean_values) / std_values
    elif method == 'center':
        X_scaled = X_arr - mean_values
    elif method == 'scale':
        X_scaled = X_arr / std_values
    else:  # 'none'
        X_scaled = X_arr.copy()
        mean_values = np.zeros(X_arr.shape[1])
        std_values = np.ones(X_arr.shape[1])

    return X_scaled, mean_values, std_values


def split_by_class(
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray]
) -> Dict[any, np.ndarray]:
    """
    Split data by class labels.

    Parameters
    ----------
    X : DataFrame or ndarray
        Feature matrix
    y : Series or ndarray
        Class labels

    Returns
    -------
    dict
        Dictionary mapping class labels to feature matrices
    """
    # Convert to arrays if needed
    if isinstance(X, pd.DataFrame):
        X_arr = X.values
    else:
        X_arr = np.array(X)

    if isinstance(y, pd.Series):
        y_arr = y.values
    else:
        y_arr = np.array(y)

    # Split by class
    class_data = {}
    for cls in np.unique(y_arr):
        mask = y_arr == cls
        class_data[cls] = X_arr[mask]

    return class_data


def create_cv_folds(
    n_samples: int,
    n_folds: int,
    random_state: Optional[int] = None
) -> np.ndarray:
    """
    Create cross-validation fold indices.

    Parameters
    ----------
    n_samples : int
        Number of samples
    n_folds : int
        Number of folds
    random_state : int, optional
        Random seed for reproducibility

    Returns
    -------
    ndarray
        Array of fold indices (0 to n_folds-1)
    """
    if random_state is not None:
        np.random.seed(random_state)

    # Create sequential fold indices
    fold_indices = np.tile(np.arange(n_folds), n_samples // n_folds + 1)[:n_samples]

    # Shuffle
    np.random.shuffle(fold_indices)

    return fold_indices


def create_stratified_cv_folds(
    y: Union[pd.Series, np.ndarray],
    n_folds: int,
    random_state: Optional[int] = None
) -> np.ndarray:
    """
    Create stratified K-fold indices maintaining class distribution per fold.
    Each fold preserves the class proportions of the full dataset.

    Parameters
    ----------
    y : Series or ndarray
        Class labels
    n_folds : int
        Number of folds
    random_state : int, optional
        Random seed for reproducibility

    Returns
    -------
    ndarray
        Array of fold indices (0 to n_folds-1) maintaining class proportions
    """
    if random_state is not None:
        np.random.seed(random_state)

    # Convert to array if needed
    if isinstance(y, pd.Series):
        y_arr = y.values
    else:
        y_arr = np.array(y)

    n_samples = len(y_arr)
    fold_indices = np.zeros(n_samples, dtype=int)

    # For each class, distribute samples across folds cyclically
    for cls in np.unique(y_arr):
        cls_indices = np.where(y_arr == cls)[0]
        np.random.shuffle(cls_indices)

        # Assign to folds cyclically (preserves class proportion in each fold)
        fold_assignment = np.arange(len(cls_indices)) % n_folds
        fold_indices[cls_indices] = fold_assignment

    # Final shuffle to randomize sample order within each fold
    shuffle_idx = np.random.permutation(n_samples)
    y_shuffled = y_arr[shuffle_idx]
    fold_shuffled = fold_indices[shuffle_idx]

    # Create reverse mapping to restore original order
    fold_indices_original = np.zeros(n_samples, dtype=int)
    fold_indices_original[shuffle_idx] = fold_shuffled

    return fold_indices_original


def create_stratified_cv_folds_with_groups(
    y: Union[pd.Series, np.ndarray],
    groups: Optional[Union[pd.Series, np.ndarray]] = None,
    n_folds: int = 5,
    random_state: Optional[int] = None
) -> Dict[int, Dict[str, np.ndarray]]:
    """
    Create stratified K-fold indices with optional cancellation groups support.

    This function creates CV fold indices while ensuring that samples from the
    same group (cancellation group) NEVER appear in both training and test sets
    within the same fold. This is critical for proper validation when samples
    are related (e.g., replicates, time series from same subject).

    IMPORTANT: When using PCA preprocessing with CV, PCA must be fitted ONLY on
    the training set of each fold. The evaluation set is projected onto this
    PCA model without being used in fitting. This function provides the indices
    needed to implement this correctly.

    Parameters
    ----------
    y : Series or ndarray of shape (n_samples,)
        Class labels for stratification
    groups : Series or ndarray of shape (n_samples,), optional
        Group/sample identifiers for cancellation. Samples with the same group
        value will always stay together (never split between train and test).
        If None, behaves like standard stratified K-fold.
    n_folds : int, default=5
        Number of cross-validation folds
    random_state : int, optional
        Random seed for reproducibility

    Returns
    -------
    dict
        Nested dictionary with fold information:
        {
            fold_idx: {
                'train_indices': ndarray - indices for training set
                'test_indices': ndarray - indices for test/evaluation set
                'train_groups': ndarray - group values in training (if groups provided)
                'test_groups': ndarray - group values in test (if groups provided)
            }
        }

    Notes
    -----
    - With groups: Uses GroupKFold-like logic with stratification
    - Without groups: Falls back to standard stratified K-fold
    - Class proportions are maintained per fold when possible
    - All groups are represented across the full CV procedure

    Examples
    --------
    >>> y = np.array([0, 0, 0, 1, 1, 1, 0, 0, 1, 1])
    >>> groups = np.array(['A', 'A', 'B', 'B', 'C', 'C', 'D', 'D', 'E', 'E'])
    >>> folds = create_stratified_cv_folds_with_groups(y, groups, n_folds=5)
    >>> # Samples in same group (e.g., 'A') will always be in same split

    >>> # Without groups - standard stratified K-fold
    >>> folds = create_stratified_cv_folds_with_groups(y, groups=None, n_folds=5)

    See Also
    --------
    create_stratified_cv_folds : Standard stratified K-fold (returns array)
    """
    if random_state is not None:
        np.random.seed(random_state)

    # Convert to arrays if needed
    if isinstance(y, pd.Series):
        y_arr = y.values
    else:
        y_arr = np.array(y)

    n_samples = len(y_arr)
    unique_classes = np.unique(y_arr)

    # If no groups provided, fall back to standard stratified K-fold
    if groups is None:
        fold_indices = create_stratified_cv_folds(y_arr, n_folds, random_state)
        folds_dict = {}
        for fold in range(n_folds):
            test_mask = fold_indices == fold
            train_mask = ~test_mask
            folds_dict[fold] = {
                'train_indices': np.where(train_mask)[0],
                'test_indices': np.where(test_mask)[0],
                'train_groups': None,
                'test_groups': None
            }
        return folds_dict

    # Convert groups to array
    if isinstance(groups, pd.Series):
        groups_arr = groups.values
    else:
        groups_arr = np.array(groups)

    unique_groups = np.unique(groups_arr)
    n_groups = len(unique_groups)

    if n_groups < n_folds:
        raise ValueError(
            f"Number of groups ({n_groups}) must be >= n_folds ({n_folds}). "
            f"Consider using fewer folds or not using groups."
        )

    # Strategy: Assign entire groups to folds while trying to maintain stratification
    # First, determine majority class for each group
    group_class_info = {}
    for grp in unique_groups:
        grp_mask = groups_arr == grp
        grp_classes, grp_counts = np.unique(y_arr[grp_mask], return_counts=True)
        majority_class = grp_classes[np.argmax(grp_counts)]
        group_class_info[grp] = {
            'majority_class': majority_class,
            'n_samples': grp_mask.sum(),
            'indices': np.where(grp_mask)[0]
        }

    # Group groups by majority class for stratified assignment
    class_groups = {cls: [] for cls in unique_classes}
    for grp, info in group_class_info.items():
        class_groups[info['majority_class']].append(grp)

    # Shuffle groups within each class
    for cls in unique_classes:
        np.random.shuffle(class_groups[cls])

    # Assign groups to folds in round-robin fashion within each class
    group_to_fold = {}
    for cls in unique_classes:
        for i, grp in enumerate(class_groups[cls]):
            group_to_fold[grp] = i % n_folds

    # Build fold indices
    folds_dict = {}
    for fold in range(n_folds):
        test_groups = [grp for grp, f in group_to_fold.items() if f == fold]
        train_groups = [grp for grp, f in group_to_fold.items() if f != fold]

        test_indices = []
        for grp in test_groups:
            test_indices.extend(group_class_info[grp]['indices'].tolist())

        train_indices = []
        for grp in train_groups:
            train_indices.extend(group_class_info[grp]['indices'].tolist())

        folds_dict[fold] = {
            'train_indices': np.array(train_indices),
            'test_indices': np.array(test_indices),
            'train_groups': np.array(train_groups),
            'test_groups': np.array(test_groups)
        }

    return folds_dict


def prepare_training_test(
    X_train: Union[pd.DataFrame, np.ndarray],
    y_train: Union[pd.Series, np.ndarray],
    X_test: Optional[Union[pd.DataFrame, np.ndarray]] = None,
    scaling_method: str = 'autoscale'
) -> Dict[str, any]:
    """
    Prepare training and optional test data with consistent scaling.

    Parameters
    ----------
    X_train : DataFrame or ndarray
        Training features
    y_train : Series or ndarray
        Training labels
    X_test : DataFrame or ndarray, optional
        Test features
    scaling_method : str
        Scaling method to apply

    Returns
    -------
    dict
        Dictionary containing scaled data and parameters
    """
    # Validate training data
    validation = validate_classification_data(X_train, y_train)
    if not validation['valid']:
        raise ValueError(f"Training data validation failed: {validation['messages']}")

    # Scale training data
    center = scaling_method in ['autoscale', 'center']
    scale = scaling_method in ['autoscale', 'scale']

    X_train_scaled, mean_values, std_values = scale_data(
        X_train,
        method=scaling_method,
        center=center,
        scale=scale
    )

    # Convert labels to array
    if isinstance(y_train, pd.Series):
        y_train_arr = y_train.values
    else:
        y_train_arr = np.array(y_train)

    result = {
        'X_train': X_train_scaled,
        'y_train': y_train_arr,
        'mean_values': mean_values,
        'std_values': std_values,
        'scaling_method': scaling_method,
        'n_features': X_train_scaled.shape[1],
        'classes': np.unique(y_train_arr),
        'n_classes': len(np.unique(y_train_arr))
    }

    # Scale test data if provided
    if X_test is not None:
        X_test_scaled, _, _ = scale_data(
            X_test,
            method=scaling_method,
            center=center,
            scale=scale,
            reference_mean=mean_values,
            reference_std=std_values
        )
        result['X_test'] = X_test_scaled

    return result


def balance_classes(
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    method: str = 'undersample',
    random_state: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Balance class distribution in dataset.

    Parameters
    ----------
    X : DataFrame or ndarray
        Feature matrix
    y : Series or ndarray
        Class labels
    method : str
        Balancing method: 'undersample' or 'oversample'
    random_state : int, optional
        Random seed

    Returns
    -------
    X_balanced : ndarray
        Balanced feature matrix
    y_balanced : ndarray
        Balanced labels
    """
    if random_state is not None:
        np.random.seed(random_state)

    # Convert to arrays
    if isinstance(X, pd.DataFrame):
        X_arr = X.values
    else:
        X_arr = np.array(X)

    if isinstance(y, pd.Series):
        y_arr = y.values
    else:
        y_arr = np.array(y)

    # Get class distribution
    unique_classes, class_counts = np.unique(y_arr, return_counts=True)

    if method == 'undersample':
        # Undersample to smallest class size
        target_size = class_counts.min()

        X_balanced = []
        y_balanced = []

        for cls in unique_classes:
            mask = y_arr == cls
            indices = np.where(mask)[0]
            selected = np.random.choice(indices, size=target_size, replace=False)
            X_balanced.append(X_arr[selected])
            y_balanced.append(y_arr[selected])

        X_balanced = np.vstack(X_balanced)
        y_balanced = np.concatenate(y_balanced)

    elif method == 'oversample':
        # Oversample to largest class size
        target_size = class_counts.max()

        X_balanced = []
        y_balanced = []

        for cls in unique_classes:
            mask = y_arr == cls
            indices = np.where(mask)[0]
            selected = np.random.choice(indices, size=target_size, replace=True)
            X_balanced.append(X_arr[selected])
            y_balanced.append(y_arr[selected])

        X_balanced = np.vstack(X_balanced)
        y_balanced = np.concatenate(y_balanced)

    else:
        raise ValueError(f"Unknown balancing method: {method}")

    # Shuffle
    shuffle_idx = np.random.permutation(len(y_balanced))
    X_balanced = X_balanced[shuffle_idx]
    y_balanced = y_balanced[shuffle_idx]

    return X_balanced, y_balanced


def suggest_n_components_pca(
    X: Union[pd.DataFrame, np.ndarray],
    explained_variance_ratio: Optional[np.ndarray] = None,
    cumsum_threshold: float = PCA_VARIANCE_THRESHOLD,
    max_components: Optional[int] = None
) -> Dict[str, any]:
    """
    Suggest optimal number of PCA components based on cumulative variance explained.

    This function helps determine how many PCA components to use for preprocessing.
    It analyzes variance distribution and recommends a number of components that
    retains a specified proportion of total variance.

    Parameters
    ----------
    X : DataFrame or ndarray of shape (n_samples, n_features)
        Training data (used to determine max sensible components if variance not provided)
    explained_variance_ratio : ndarray of shape (n_components,), optional
        Variance explained by each component (from a fitted PCA model).
        If None, a temporary PCA will be fitted to calculate this.
    cumsum_threshold : float, default=0.95
        Cumulative variance threshold (0 to 1). Components are selected until
        this proportion of total variance is explained.
    max_components : int, optional
        Hard limit on maximum components to consider.

    Returns
    -------
    dict
        Dictionary containing:
        - recommended_n_components : int
            Number of components at cumsum_threshold
        - variance_explained : float
            Actual cumulative variance at recommended components (0 to 1)
        - max_possible : int
            Maximum sensible number of components based on data shape
        - suggested_range : tuple (min, max)
            Recommended range for user slider
        - all_variances : list
            Cumulative variance at each component count

    Notes
    -----
    - This is the MOST COMMON approach in chemometrics
    - Default threshold of 95% is industry standard
    - For small datasets, consider lower threshold (90%) to avoid overfitting

    Examples
    --------
    >>> from classification_utils import fit_pca_preprocessor
    >>> pca_model = fit_pca_preprocessor(X_train, n_components=20)
    >>> suggestion = suggest_n_components_pca(
    ...     X_train,
    ...     explained_variance_ratio=pca_model['explained_variance_ratio'],
    ...     cumsum_threshold=0.95
    ... )
    >>> print(f"Recommended: {suggestion['recommended_n_components']} components")
    >>> print(f"Variance retained: {suggestion['variance_explained']*100:.1f}%")

    See Also
    --------
    fit_pca_preprocessor : Fit PCA model to get explained_variance_ratio
    """
    # Convert to array if needed
    if isinstance(X, pd.DataFrame):
        X_arr = X.values
    else:
        X_arr = np.array(X)

    n_samples, n_features = X_arr.shape
    max_sensible = min(n_samples - 1, n_features)

    if max_components is not None:
        max_sensible = min(max_components, max_sensible)

    # If variance not provided, fit temporary PCA to get it
    if explained_variance_ratio is None:
        # Import here to avoid circular import
        from .calculations import fit_pca_preprocessor

        n_comp_temp = min(max_sensible, 50)  # Fit enough components
        pca_temp = fit_pca_preprocessor(X_arr, n_components=n_comp_temp)
        explained_variance_ratio = pca_temp['explained_variance_ratio']

    # Calculate cumulative sum
    cumsum = np.cumsum(explained_variance_ratio)

    # Find first component where cumsum >= threshold
    idx_threshold = np.where(cumsum >= cumsum_threshold)[0]
    if len(idx_threshold) > 0:
        recommended_n = idx_threshold[0] + 1  # +1 because indexing starts at 0
    else:
        recommended_n = len(explained_variance_ratio)  # All components

    # Apply hard caps
    recommended_n = min(recommended_n, max_sensible)
    recommended_n = max(recommended_n, 1)  # At least 1 component

    # Calculate suggested range for user slider
    min_suggested = max(1, recommended_n - 3)
    max_suggested = min(max_sensible, recommended_n + 5, 15)

    # Get variance at recommended
    if recommended_n <= len(cumsum):
        variance_at_recommended = float(cumsum[recommended_n - 1])
    else:
        variance_at_recommended = float(cumsum[-1])

    return {
        'recommended_n_components': recommended_n,
        'variance_explained': variance_at_recommended,
        'max_possible': max_sensible,
        'suggested_range': (min_suggested, max_suggested),
        'all_variances': cumsum.tolist(),
        'cumsum_threshold': cumsum_threshold
    }
