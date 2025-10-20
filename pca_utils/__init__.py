"""
PCA Utility Modules for ChemometricSolutions
=============================================

A comprehensive package for Principal Component Analysis (PCA) including:
- Core PCA calculations and Varimax rotation
- Statistical diagnostics (T², Q residuals, cross-validation)
- Visualization functions (scores, loadings, biplots, scree plots)
- Workspace management (dataset splitting, saving/loading)

Package Structure
-----------------
pca_calculations : PCA computation, Varimax rotation, explained variance
pca_statistics   : Statistical diagnostics and validation metrics
pca_plots        : Plotly-based visualization functions
pca_workspace    : Dataset splitting and workspace management
config           : Package-level configuration constants

Quick Start
-----------
>>> from pca_utils import compute_pca, plot_scores, calculate_hotelling_t2
>>> import pandas as pd
>>>
>>> # Perform PCA
>>> data = pd.DataFrame(...)
>>> pca_results = compute_pca(data, n_components=5, scale=True)
>>>
>>> # Create scores plot
>>> fig = plot_scores(pca_results['scores'], 'PC1', 'PC2',
...                   pca_results['explained_variance_ratio'])
>>>
>>> # Calculate T² statistic
>>> t2_values, t2_limit = calculate_hotelling_t2(pca_results['scores'],
...                                               pca_results['eigenvalues'])
"""

# Import configuration constants
from .config import (
    DEFAULT_N_COMPONENTS,
    DEFAULT_CONFIDENCE_LEVEL,
    VARIMAX_MAX_ITER,
    VARIMAX_TOLERANCE
)

# Import calculation functions
from .pca_calculations import (
    compute_pca,
    varimax_rotation,
    calculate_explained_variance
)

# Import plotting functions
from .pca_plots import (
    plot_scree,
    plot_cumulative_variance,
    plot_scores,
    plot_loadings,
    plot_loadings_line,
    plot_biplot,
    add_convex_hulls
)

# Import statistical functions
from .pca_statistics import (
    calculate_hotelling_t2,
    calculate_q_residuals,
    calculate_contributions,
    calculate_leverage,
    cross_validate_pca
)

# Import workspace management functions
from .pca_workspace import (
    save_workspace_to_file,
    load_workspace_from_file,
    save_dataset_split,
    get_split_datasets_info,
    delete_split_dataset,
    clear_all_split_datasets
)

# Define public API
__all__ = [
    # Configuration constants
    'DEFAULT_N_COMPONENTS',
    'DEFAULT_CONFIDENCE_LEVEL',
    'VARIMAX_MAX_ITER',
    'VARIMAX_TOLERANCE',

    # Calculation functions
    'compute_pca',
    'varimax_rotation',
    'calculate_explained_variance',

    # Plotting functions
    'plot_scree',
    'plot_cumulative_variance',
    'plot_scores',
    'plot_loadings',
    'plot_loadings_line',
    'plot_biplot',
    'add_convex_hulls',

    # Statistical functions
    'calculate_hotelling_t2',
    'calculate_q_residuals',
    'calculate_contributions',
    'calculate_leverage',
    'cross_validate_pca',

    # Workspace management functions
    'save_workspace_to_file',
    'load_workspace_from_file',
    'save_dataset_split',
    'get_split_datasets_info',
    'delete_split_dataset',
    'clear_all_split_datasets',
]

# Package metadata
__version__ = '1.0.0'
__author__ = 'ChemometricSolutions'
__description__ = 'PCA utility modules for chemometric analysis'
