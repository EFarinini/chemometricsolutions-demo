"""
MLR/DoE utility modules for ChemometricSolutions

This package contains:
- model_computation.py: Core MLR fitting and model matrix creation
- model_diagnostics.py: VIF, leverage, diagnostic plots
- surface_analysis.py: Unified response surface & confidence interval analysis
- predictions.py: Prediction utilities
- candidate_points.py: Candidate points generation
- export.py: Export utilities

DEPRECATED (replaced by surface_analysis.py):
- response_surface.py: Now included in surface_analysis.py
- confidence_intervals.py: Now included in surface_analysis.py
"""

# Core computation functions
from .model_computation import (
    create_model_matrix,
    fit_mlr_model,
    statistical_summary
)

# Diagnostic functions
from .model_diagnostics import (
    calculate_vif,
    check_model_saturated,
    show_model_diagnostics_ui
)

# Surface Analysis (unified Response Surface + CI)
from .surface_analysis import (
    show_surface_analysis_ui
)

# Predictions
from .predictions import (
    show_predictions_ui
)

# Candidate Points (Design Generation)
from .candidate_points import (
    show_candidate_points_ui
)

# Export utilities
from .export import (
    show_export_ui
)

# Pareto optimization UI
from .pareto_ui import (
    show_pareto_ui
)

# Multi-DOE functions
from .model_computation_multidoe import (
    fit_multidoe_model,
    fit_all_multidoe_models,
    statistical_summary_multidoe,
    extract_coefficients_comparison
)

from .model_diagnostics_multidoe import (
    show_model_diagnostics_ui_multidoe
)

from .surface_analysis_multidoe import (
    show_surface_analysis_ui_multidoe
)

from .predictions_multidoe import (
    show_predictions_ui_multidoe
)

from .export_multidoe import (
    show_export_ui_multidoe,
    create_multidoe_excel_export
)

from .pareto_ui_multidoe import (
    show_pareto_ui_multidoe
)

__all__ = [
    # Model computation
    'create_model_matrix',
    'fit_mlr_model',
    'statistical_summary',

    # Model diagnostics
    'calculate_vif',
    'check_model_saturated',
    'show_model_diagnostics_ui',

    # Surface Analysis
    'show_surface_analysis_ui',

    # Predictions
    'show_predictions_ui',

    # Candidate Points
    'show_candidate_points_ui',

    # Export
    'show_export_ui',

    # Pareto optimization
    'show_pareto_ui',

    # Multi-DOE
    'fit_multidoe_model',
    'fit_all_multidoe_models',
    'statistical_summary_multidoe',
    'extract_coefficients_comparison',
    'show_model_diagnostics_ui_multidoe',
    'show_surface_analysis_ui_multidoe',
    'show_predictions_ui_multidoe',
    'show_export_ui_multidoe',
    'create_multidoe_excel_export',
    'show_pareto_ui_multidoe',
]

# For backward compatibility, also expose old imports
# (can be removed after deprecation period)
try:
    from .response_surface import show_response_surface_ui
    from .confidence_intervals import show_confidence_intervals_ui
    __all__.extend(['show_response_surface_ui', 'show_confidence_intervals_ui'])
except ImportError:
    pass
