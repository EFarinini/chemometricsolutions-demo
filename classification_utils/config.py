"""
Configuration Constants for Classification Module
==================================================

This module contains global constants used throughout the classification package.
"""

# Cross-validation defaults
DEFAULT_CV_FOLDS = 5
DEFAULT_CONFIDENCE_LEVEL = 0.95

# kNN defaults
DEFAULT_K_MAX = 7
AVAILABLE_DISTANCE_METRICS = [
    'euclidean',
    'mahalanobis',
    'manhattan',
    'chebyshev',
    'minkowski'
]

# SIMCA/UNEQ defaults
DEFAULT_N_COMPONENTS_SIMCA = 3
NIPALS_MAX_ITER = 1000
NIPALS_TOLERANCE = 1e-7

# Plotting defaults
PLOT_COLORS = [
    '#FF0000',  # Red
    '#0000FF',  # Blue
    '#008000',  # Green
    '#FF00FF',  # Magenta
    '#00FFFF',  # Cyan
    '#FF8040',  # Orange
    '#00FF00',  # Lime
    '#000000'   # Black
]

# Coomans plot settings
COOMANS_AXIS_LIMIT = 6
COOMANS_THRESHOLD = 1.0
COOMANS_CLIP_VALUE = 4.0  # Values > 6 are clipped to 4-6 range

# Validation thresholds
MIN_SAMPLES_PER_CLASS = 3
MAX_CLASSES = 10

# PCA Preprocessing defaults
DEFAULT_N_COMPONENTS_PCA = 5
DEFAULT_PCA_MAX_ITER = 1000
DEFAULT_PCA_TOLERANCE = 1e-7
MIN_N_COMPONENTS_PCA = 1
MAX_N_COMPONENTS_PCA_RATIO = 0.95  # Max 95% of n_features
PCA_VARIANCE_THRESHOLD = 0.95  # Default 95% cumulative variance threshold
