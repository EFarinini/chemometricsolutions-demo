"""
Genetic Algorithm Variable Selection Module
===========================================

Implementation of Riccardo Leardi's genetic algorithms for variable selection
in chemometrics applications. Optimized for Streamlit execution.

Main Components:
- ga_engine: Core genetic algorithm implementation
- cv_evaluators: Fitness functions for different problem types
- ga_validators: Input validation utilities
- results_visualization: Plotting utilities for results
- spectral_visualization: Spectral plot utilities for spectroscopic data

Supported Problem Types:
- PLS Regression (continuous target)
- LDA Classification (categorical)
- FDA Classification (Fisher discriminant)
- Mahalanobis Distance (fault detection)
- Distance-based similarity
"""

__version__ = "1.0.0"
__author__ = "ChemometricSolutions"

from .ga_engine import GeneticAlgorithm
from .cv_evaluators import (
    FitnessEvaluator,
    PLSEvaluator,
    LDAEvaluator,
    MahalanobisEvaluator,
    DistanceEvaluator
)

# Spectral visualization (optional)
try:
    from .spectral_visualization import (
        plot_spectral_regions,
        plot_spectral_regions_interactive,
        plot_spectral_leardi_style,
        create_spectral_summary,
        is_spectroscopic_data,
        extract_wavelengths_from_columns
    )
    SPECTRAL_VIZ_AVAILABLE = True
except ImportError:
    SPECTRAL_VIZ_AVAILABLE = False

# Spectral BANDS visualization (CORRECT Leardi approach)
try:
    from .spectral_bands import (
        plot_spectrum_with_bands,
        plot_multiple_runs_with_bands,
        create_bands_table
    )
    SPECTRAL_BANDS_AVAILABLE = True
except ImportError:
    SPECTRAL_BANDS_AVAILABLE = False

# Empirical configuration dashboard (Leardi's ACTUAL methodology)
try:
    from .empirical_config import (
        create_empirical_config_dashboard,
        simulate_true_vs_random_curve,
        create_cv_performance_curve,
        plot_true_vs_random,
        plot_cv_curve_with_plateau,
        create_selection_frequency_histogram,
        analyze_plateau_region
    )
    EMPIRICAL_CONFIG_AVAILABLE = True
except ImportError:
    EMPIRICAL_CONFIG_AVAILABLE = False

# GAPLSSP Original (100% faithful Leardi implementation)
try:
    from .gaplssp_original import GAPLSSP, gaplssp
    GAPLSSP_AVAILABLE = True
except ImportError:
    GAPLSSP_AVAILABLE = False

# GAPLSOPT - Randomization test for stop criterion (Step 1 of Leardi workflow)
try:
    from .gaplsopt import GAPLSOPT, gaplsopt
    GAPLSOPT_AVAILABLE = True
except ImportError:
    GAPLSOPT_AVAILABLE = False

# Results visualization (plotting utilities)
try:
    from .results_visualization import (
        plot_selection_frequency,
        plot_fitness_evolution,
        plot_cv_curve,
        plot_rmsecv_curve
    )
    RESULTS_VIZ_AVAILABLE = True
except ImportError:
    RESULTS_VIZ_AVAILABLE = False

__all__ = [
    'GeneticAlgorithm',
    'FitnessEvaluator',
    'PLSEvaluator',
    'LDAEvaluator',
    'MahalanobisEvaluator',
    'DistanceEvaluator',
    'SPECTRAL_VIZ_AVAILABLE',
    'SPECTRAL_BANDS_AVAILABLE',
    'EMPIRICAL_CONFIG_AVAILABLE',
    'GAPLSSP_AVAILABLE',
    'GAPLSOPT_AVAILABLE'
]

if SPECTRAL_VIZ_AVAILABLE:
    __all__.extend([
        'plot_spectral_regions',
        'plot_spectral_regions_interactive',
        'plot_spectral_leardi_style',
        'create_spectral_summary',
        'is_spectroscopic_data',
        'extract_wavelengths_from_columns'
    ])

if SPECTRAL_BANDS_AVAILABLE:
    __all__.extend([
        'plot_spectrum_with_bands',
        'plot_multiple_runs_with_bands',
        'create_bands_table'
    ])

if EMPIRICAL_CONFIG_AVAILABLE:
    __all__.extend([
        'create_empirical_config_dashboard',
        'simulate_true_vs_random_curve',
        'create_cv_performance_curve',
        'plot_true_vs_random',
        'plot_cv_curve_with_plateau',
        'create_selection_frequency_histogram',
        'analyze_plateau_region'
    ])

if GAPLSSP_AVAILABLE:
    __all__.extend([
        'GAPLSSP',
        'gaplssp'
    ])

if GAPLSOPT_AVAILABLE:
    __all__.extend([
        'GAPLSOPT',
        'gaplsopt'
    ])

if RESULTS_VIZ_AVAILABLE:
    __all__.extend([
        'plot_selection_frequency',
        'plot_fitness_evolution',
        'plot_cv_curve',
        'plot_rmsecv_curve',
        'RESULTS_VIZ_AVAILABLE'
    ])
