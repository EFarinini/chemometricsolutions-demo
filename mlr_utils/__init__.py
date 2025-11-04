"""
MLR/DoE utility modules for ChemometricSolutions

This package contains:
- model_computation.py: Core MLR fitting and model matrix creation
- model_diagnostics.py: VIF, leverage, diagnostic plots
- response_surface.py: Response surface analysis
- confidence_intervals.py: Confidence interval calculations
- predictions.py: Prediction utilities
- candidate_points.py: Candidate points generation
- export.py: Export utilities
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

__all__ = [
    # Model computation
    'create_model_matrix',
    'fit_mlr_model',
    'statistical_summary',

    # Model diagnostics
    'calculate_vif',
    'check_model_saturated',
    'show_model_diagnostics_ui',
]
