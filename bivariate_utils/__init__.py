"""
Bivariate Analysis Utilities
Modular utilities for bivariate statistical analysis and visualization
"""

from .statistics import (
    compute_correlation_matrix,
    compute_covariance_matrix,
    compute_spearman_correlation,
    compute_kendall_correlation
)

from .plotting import (
    create_scatter_plot,
    create_pairs_plot,
    create_correlation_heatmap
)

__all__ = [
    'compute_correlation_matrix',
    'compute_covariance_matrix',
    'compute_spearman_correlation',
    'compute_kendall_correlation',
    'create_scatter_plot',
    'create_pairs_plot',
    'create_correlation_heatmap'
]
