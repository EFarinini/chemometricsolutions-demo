"""
Classification Utility Modules for ChemometricSolutions
========================================================

A comprehensive package for classification analysis including:
- LDA (Linear Discriminant Analysis)
- QDA (Quadratic Discriminant Analysis)
- kNN (k-Nearest Neighbors with multiple distance metrics)
- SIMCA (Soft Independent Modeling of Class Analogy)
- UNEQ (Unequal Class Dispersions modeling)

Package Structure
-----------------
calculations     : Core classification algorithms (LDA, QDA, kNN, SIMCA, UNEQ)
preprocessing    : Data validation, scaling, and preparation
diagnostics      : Performance metrics, confusion matrices, cross-validation
plots            : Plotly-based visualization functions
config           : Package-level configuration constants

Quick Start
-----------
>>> from classification_utils import fit_lda, predict_lda, cross_validate_lda
>>> import pandas as pd
>>> import numpy as np
>>>
>>> # Prepare data
>>> X = data[feature_columns].values
>>> y = data['class'].values
>>>
>>> # Fit LDA model
>>> from classification_utils.preprocessing import prepare_training_test
>>> prepared = prepare_training_test(X, y, scaling_method='autoscale')
>>>
>>> # Train and evaluate with cross-validation
>>> cv_results = cross_validate_lda(prepared['X_train'], prepared['y_train'], n_folds=5)
>>> print(f"Accuracy: {cv_results['metrics']['accuracy']:.2f}%")
>>>
>>> # Visualize results
>>> from classification_utils.plots import plot_confusion_matrix
>>> fig = plot_confusion_matrix(cv_results['metrics']['confusion_matrix'], classes)
"""

# Import configuration constants
from .config import (
    DEFAULT_CV_FOLDS,
    DEFAULT_CONFIDENCE_LEVEL,
    DEFAULT_K_MAX,
    AVAILABLE_DISTANCE_METRICS,
    DEFAULT_N_COMPONENTS_SIMCA,
    PLOT_COLORS,
    COOMANS_AXIS_LIMIT,
    # PCA Preprocessing constants
    DEFAULT_N_COMPONENTS_PCA,
    DEFAULT_PCA_MAX_ITER,
    DEFAULT_PCA_TOLERANCE,
    MIN_N_COMPONENTS_PCA,
    MAX_N_COMPONENTS_PCA_RATIO,
    PCA_VARIANCE_THRESHOLD
)

# Import preprocessing functions
from .preprocessing import (
    validate_classification_data,
    scale_data,
    split_by_class,
    create_cv_folds,
    create_stratified_cv_folds,
    create_stratified_cv_folds_with_groups,
    prepare_training_test,
    balance_classes,
    suggest_n_components_pca
)

# Import calculation functions
from .calculations import (
    # LDA
    fit_lda,
    predict_lda,
    predict_lda_detailed,

    # QDA
    fit_qda,
    predict_qda,
    predict_qda_detailed,

    # kNN
    fit_knn,
    predict_knn,
    predict_knn_detailed,
    calculate_distance_matrix,

    # SIMCA
    fit_simca,
    predict_simca,
    predict_simca_detailed,

    # UNEQ
    fit_uneq,
    predict_uneq,
    predict_uneq_detailed,

    # NIPALS (for SIMCA/UNEQ)
    nipals_pca,

    # PCA Preprocessing
    fit_pca_preprocessor,
    project_onto_pca,
    fit_lda_with_pca,
    predict_lda_with_pca,
    fit_qda_with_pca,
    predict_qda_with_pca,
    fit_knn_with_pca,
    predict_knn_with_pca
)

# Import diagnostic functions
from .diagnostics import (
    calculate_confusion_matrix,
    calculate_classification_metrics,
    calculate_simca_uneq_metrics,
    compute_classification_metrics,
    cross_validate_lda,
    cross_validate_qda,
    cross_validate_knn,
    cross_validate_simca,
    cross_validate_uneq,
    cross_validate_classifier,
    find_best_k,
    compare_models,
    get_misclassified_samples,
    # CV with PCA preprocessing
    cross_validate_lda_with_pca,
    cross_validate_qda_with_pca,
    cross_validate_knn_with_pca,
    # kNN Neighbor Analysis
    analyze_sample_neighbors_by_k,
    get_sample_metadata
)

# Import plotting functions
from .plots import (
    plot_confusion_matrix,
    plot_classification_metrics,
    plot_coomans,
    plot_distance_distributions,
    plot_knn_performance,
    plot_model_comparison,
    plot_decision_boundary_2d,
    plot_class_separation,
    plot_mahalanobis_distances,
    plot_mahalanobis_distance_closest_category,
    plot_mahalanobis_distance_category,
    plot_mahalanobis_distance_object,
    plot_knn_neighbors,
    plot_classification_report,
    # PCA Preprocessing plots
    plot_pca_variance_explained,
    plot_pca_scores_2d,
    plot_pca_loadings,
    # kNN Neighbor Analysis plots
    plot_sample_neighbors_by_k
)

# Define public API
__all__ = [
    # Configuration
    'DEFAULT_CV_FOLDS',
    'DEFAULT_CONFIDENCE_LEVEL',
    'DEFAULT_K_MAX',
    'AVAILABLE_DISTANCE_METRICS',
    'DEFAULT_N_COMPONENTS_SIMCA',
    'PLOT_COLORS',
    'COOMANS_AXIS_LIMIT',
    # PCA Preprocessing config
    'DEFAULT_N_COMPONENTS_PCA',
    'DEFAULT_PCA_MAX_ITER',
    'DEFAULT_PCA_TOLERANCE',
    'MIN_N_COMPONENTS_PCA',
    'MAX_N_COMPONENTS_PCA_RATIO',
    'PCA_VARIANCE_THRESHOLD',

    # Preprocessing
    'validate_classification_data',
    'scale_data',
    'split_by_class',
    'create_cv_folds',
    'create_stratified_cv_folds',
    'create_stratified_cv_folds_with_groups',
    'prepare_training_test',
    'balance_classes',
    'suggest_n_components_pca',

    # LDA
    'fit_lda',
    'predict_lda',
    'predict_lda_detailed',

    # QDA
    'fit_qda',
    'predict_qda',
    'predict_qda_detailed',

    # kNN
    'fit_knn',
    'predict_knn',
    'predict_knn_detailed',
    'calculate_distance_matrix',

    # SIMCA
    'fit_simca',
    'predict_simca',
    'predict_simca_detailed',

    # UNEQ
    'fit_uneq',
    'predict_uneq',
    'predict_uneq_detailed',

    # NIPALS
    'nipals_pca',

    # PCA Preprocessing
    'fit_pca_preprocessor',
    'project_onto_pca',
    'fit_lda_with_pca',
    'predict_lda_with_pca',
    'fit_qda_with_pca',
    'predict_qda_with_pca',
    'fit_knn_with_pca',
    'predict_knn_with_pca',

    # Diagnostics
    'calculate_confusion_matrix',
    'calculate_classification_metrics',
    'calculate_simca_uneq_metrics',
    'compute_classification_metrics',
    'cross_validate_lda',
    'cross_validate_qda',
    'cross_validate_knn',
    'cross_validate_simca',
    'cross_validate_uneq',
    'cross_validate_classifier',
    'find_best_k',
    'compare_models',
    'get_misclassified_samples',
    # CV with PCA preprocessing
    'cross_validate_lda_with_pca',
    'cross_validate_qda_with_pca',
    'cross_validate_knn_with_pca',
    # kNN Neighbor Analysis
    'analyze_sample_neighbors_by_k',
    'get_sample_metadata',

    # Plotting
    'plot_confusion_matrix',
    'plot_classification_metrics',
    'plot_coomans',
    'plot_distance_distributions',
    'plot_knn_performance',
    'plot_model_comparison',
    'plot_decision_boundary_2d',
    'plot_class_separation',
    'plot_mahalanobis_distances',
    'plot_mahalanobis_distance_closest_category',
    'plot_mahalanobis_distance_category',
    'plot_mahalanobis_distance_object',
    'plot_knn_neighbors',
    'plot_classification_report',
    # PCA Preprocessing plots
    'plot_pca_variance_explained',
    'plot_pca_scores_2d',
    'plot_pca_loadings',
    # kNN Neighbor Analysis plots
    'plot_sample_neighbors_by_k',
]

# Package metadata
__version__ = '1.0.0'
__author__ = 'ChemometricSolutions - Dr. Emanuele Farinini'
__description__ = 'Classification utility modules for chemometric analysis'
__license__ = 'Proprietary'

# Convenience function to get all available classifiers
def get_available_classifiers():
    """
    Get list of available classification methods.

    Returns
    -------
    list of dict
        List of dictionaries with classifier information
    """
    return [
        {
            'name': 'LDA',
            'full_name': 'Linear Discriminant Analysis',
            'description': 'Assumes equal covariance matrices across classes',
            'fit_fn': fit_lda,
            'predict_fn': predict_lda,
            'cv_fn': cross_validate_lda,
            'supports_multiclass': True,
            'requires_pca': False
        },
        {
            'name': 'QDA',
            'full_name': 'Quadratic Discriminant Analysis',
            'description': 'Allows different covariance matrices per class',
            'fit_fn': fit_qda,
            'predict_fn': predict_qda,
            'cv_fn': cross_validate_qda,
            'supports_multiclass': True,
            'requires_pca': False
        },
        {
            'name': 'kNN',
            'full_name': 'k-Nearest Neighbors',
            'description': 'Non-parametric method based on distance metrics',
            'fit_fn': fit_knn,
            'predict_fn': predict_knn,
            'cv_fn': cross_validate_knn,
            'supports_multiclass': True,
            'requires_pca': False,
            'has_hyperparameters': True
        },
        {
            'name': 'SIMCA',
            'full_name': 'Soft Independent Modeling of Class Analogy',
            'description': 'Class modeling using PCA for each class',
            'fit_fn': fit_simca,
            'predict_fn': predict_simca,
            'cv_fn': cross_validate_simca,
            'supports_multiclass': True,
            'requires_pca': True,
            'is_class_modeling': True
        },
        {
            'name': 'UNEQ',
            'full_name': 'Unequal Class Dispersions',
            'description': 'Class modeling with Mahalanobis distances',
            'fit_fn': fit_uneq,
            'predict_fn': predict_uneq,
            'cv_fn': cross_validate_uneq,
            'supports_multiclass': True,
            'requires_pca': False,
            'is_class_modeling': True
        }
    ]
