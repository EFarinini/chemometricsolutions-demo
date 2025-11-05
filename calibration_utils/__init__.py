"""
Calibration Utilities Package

This package provides modular utilities for PLS-1 regression modeling including:
- NIPALS algorithm implementation
- Cross-validation with K-fold
- Data preprocessing and validation
- Comprehensive visualizations
- Test set validation

Author: ChemoMetric Solutions
"""

# Import from pls_calculations
from .pls_calculations import (
    pls_nipals,
    pls_predict,
    calculate_metrics,
    calculate_residuals
)

# Import from pls_cv
from .pls_cv import (
    repeated_kfold_cv,
    select_optimal_lv,
    full_cv_summary
)

# Import from pls_preprocessing
from .pls_preprocessing import (
    prepare_calibration_data,
    apply_scaler,
    validate_pls_input,
    split_xy_by_column_name
)

# Import from pls_plots
from .pls_plots import (
    plot_rmsecv_vs_lv,
    plot_predictions_vs_observed,
    plot_residuals,
    plot_loadings,
    plot_regression_coefficients
)

# Import from pls_validation
from .pls_validation import (
    load_test_set_from_workspace,
    validate_on_test,
    generate_validation_report
)

# Import from pls_diagnostics
from .pls_diagnostics import (
    plot_loading_plot,
    calculate_vip,
    plot_vip,
    plot_score_plot_with_ellipse,
    plot_residuals_histogram,
    plot_qq_plot,
    display_diagnostic_with_explanation,
    create_diagnostic_info_dict
)

# Import from pls_coefficients_smart
from .pls_coefficients_smart import (
    plot_regression_coefficients_smart,
    analyze_overfitting,
    plot_coefficient_comparison
)

__all__ = [
    # Calculations
    'pls_nipals',
    'pls_predict',
    'calculate_metrics',
    'calculate_residuals',
    # Cross-validation
    'repeated_kfold_cv',
    'select_optimal_lv',
    'full_cv_summary',
    # Preprocessing
    'prepare_calibration_data',
    'apply_scaler',
    'validate_pls_input',
    'split_xy_by_column_name',
    # Plots
    'plot_rmsecv_vs_lv',
    'plot_predictions_vs_observed',
    'plot_residuals',
    'plot_loadings',
    'plot_regression_coefficients',
    # Validation
    'load_test_set_from_workspace',
    'validate_on_test',
    'generate_validation_report',
    # Diagnostics
    'plot_loading_plot',
    'calculate_vip',
    'plot_vip',
    'plot_score_plot_with_ellipse',
    'plot_residuals_histogram',
    'plot_qq_plot',
    'display_diagnostic_with_explanation',
    'create_diagnostic_info_dict',
    # Smart Coefficients
    'plot_regression_coefficients_smart',
    'analyze_overfitting',
    'plot_coefficient_comparison'
]
