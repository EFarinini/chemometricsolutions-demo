"""
Generate DoE Utilities Package
Contains modules for generating experimental designs and D-optimal optimization.
"""

from .doe_designs import (
    generate_full_factorial,
    generate_plackett_burman,
    generate_central_composite,
    generate_custom_design
)

from .candidate_generator import (
    create_candidate_matrix,
    apply_constraints
)

from .doptimal_algorithm import (
    doptimal_design,
    calculate_inflation_factors,
    exchange_algorithm_iteration
)

from .doptimal_by_addition import (
    doptimal_by_addition,
    calculate_inflation_factors_addition,
    format_addition_results,
    extract_added_experiments,
    calculate_model_matrix
)

__all__ = [
    'generate_full_factorial',
    'generate_plackett_burman',
    'generate_central_composite',
    'generate_custom_design',
    'create_candidate_matrix',
    'apply_constraints',
    'doptimal_design',
    'calculate_inflation_factors',
    'exchange_algorithm_iteration',
    'doptimal_by_addition',
    'calculate_inflation_factors_addition',
    'format_addition_results',
    'extract_added_experiments',
    'calculate_model_matrix'
]
