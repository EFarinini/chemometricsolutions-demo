"""
Input Validation for GA Variable Selection
==========================================

Validation utilities for genetic algorithm inputs and configuration.
"""

import numpy as np
import pandas as pd
from typing import Union, Optional, Tuple, Dict, Any
import warnings


class GAValidator:
    """Validation utilities for GA inputs and configuration."""

    @staticmethod
    def validate_dataset(
        X: Union[np.ndarray, pd.DataFrame],
        min_samples: int = 10,
        max_samples: int = 50000,
        max_vars: int = 10000,
        check_variance: bool = True
    ) -> Tuple[bool, str]:
        """
        Validate dataset for GA analysis.

        Parameters
        ----------
        X : np.ndarray or pd.DataFrame
            Input dataset
        min_samples : int
            Minimum number of samples required
        max_samples : int
            Maximum number of samples allowed
        max_vars : int
            Maximum number of variables allowed
        check_variance : bool
            Check for zero-variance variables

        Returns
        -------
        is_valid : bool
            True if dataset is valid
        message : str
            Error message if invalid, empty string if valid
        """
        # Convert to numpy if DataFrame
        if isinstance(X, pd.DataFrame):
            X = X.values

        # Check type
        if not isinstance(X, np.ndarray):
            return False, "Dataset must be numpy array or pandas DataFrame"

        # Check dimensions
        if X.ndim != 2:
            return False, f"Dataset must be 2D, got {X.ndim}D"

        n_samples, n_vars = X.shape

        # Check sample count
        if n_samples < min_samples:
            return False, f"Too few samples: {n_samples} (minimum: {min_samples})"

        if n_samples > max_samples:
            return False, f"Too many samples: {n_samples} (maximum: {max_samples})"

        # Check variable count
        if n_vars < 2:
            return False, f"Need at least 2 variables, got {n_vars}"

        if n_vars > max_vars:
            return False, f"Too many variables: {n_vars} (maximum: {max_vars})"

        # Check for NaN/Inf
        if np.any(np.isnan(X)):
            return False, "Dataset contains NaN values"

        if np.any(np.isinf(X)):
            return False, "Dataset contains Inf values"

        # Check for zero variance
        if check_variance:
            variances = np.var(X, axis=0)
            zero_var_count = np.sum(variances == 0)

            if zero_var_count > 0:
                warnings.warn(
                    f"Found {zero_var_count} zero-variance variables. "
                    "Consider removing them before analysis.",
                    RuntimeWarning
                )

        return True, ""

    @staticmethod
    def validate_target(
        y: Union[np.ndarray, pd.Series],
        problem_type: str,
        n_samples: int
    ) -> Tuple[bool, str]:
        """
        Validate target variable for supervised methods.

        Parameters
        ----------
        y : np.ndarray or pd.Series
            Target variable
        problem_type : str
            Type of problem ('pls', 'lda', 'fda', 'distance')
        n_samples : int
            Number of samples in dataset (for length check)

        Returns
        -------
        is_valid : bool
            True if target is valid
        message : str
            Error message if invalid
        """
        # Convert to numpy if Series
        if isinstance(y, pd.Series):
            y = y.values

        # Check type
        if not isinstance(y, np.ndarray):
            return False, "Target must be numpy array or pandas Series"

        # Check length
        if len(y) != n_samples:
            return False, f"Target length ({len(y)}) doesn't match dataset ({n_samples})"

        # Check for NaN/Inf
        if np.any(np.isnan(y)):
            return False, "Target contains NaN values"

        if np.any(np.isinf(y)):
            return False, "Target contains Inf values"

        # Problem-specific validation
        if problem_type in ['lda', 'fda', 'distance']:
            # Classification: check classes
            unique_classes = np.unique(y)
            n_classes = len(unique_classes)

            if n_classes < 2:
                return False, f"Classification requires at least 2 classes, found {n_classes}"

            # Check samples per class
            for cls in unique_classes:
                n_cls_samples = np.sum(y == cls)
                if n_cls_samples < 2:
                    return False, f"Class {cls} has only {n_cls_samples} sample(s), need at least 2"

            # Warn if imbalanced
            class_counts = [np.sum(y == cls) for cls in unique_classes]
            min_count = min(class_counts)
            max_count = max(class_counts)

            if max_count > 5 * min_count:
                warnings.warn(
                    f"Highly imbalanced classes: {min_count} to {max_count} samples. "
                    "Consider resampling or stratified splitting.",
                    RuntimeWarning
                )

        elif problem_type == 'pls':
            # Regression: check for constant target
            if np.var(y) == 0:
                return False, "Target variable has zero variance"

        return True, ""

    @staticmethod
    def validate_config(config: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Validate GA configuration parameters.

        Parameters
        ----------
        config : dict
            Configuration dictionary

        Returns
        -------
        is_valid : bool
            True if config is valid
        message : str
            Error message if invalid
        """
        required_keys = ['runs', 'population_size', 'evaluations']

        # Check required keys
        for key in required_keys:
            if key not in config:
                return False, f"Missing required config parameter: {key}"

        # Validate runs
        if not isinstance(config['runs'], int) or config['runs'] < 1:
            return False, f"'runs' must be positive integer, got {config['runs']}"

        if config['runs'] > 1000:
            return False, f"'runs' too high: {config['runs']} (maximum: 1000)"

        # Validate population_size
        if not isinstance(config['population_size'], int) or config['population_size'] < 4:
            return False, f"'population_size' must be >= 4, got {config['population_size']}"

        if config['population_size'] > 100:
            return False, f"'population_size' too high: {config['population_size']} (maximum: 100)"

        # Validate evaluations
        if not isinstance(config['evaluations'], int) or config['evaluations'] < 10:
            return False, f"'evaluations' must be >= 10, got {config['evaluations']}"

        if config['evaluations'] > 1000:
            return False, f"'evaluations' too high: {config['evaluations']} (maximum: 1000)"

        # Validate optional parameters
        if 'mutation_prob' in config:
            if not 0 < config['mutation_prob'] <= 1:
                return False, f"'mutation_prob' must be in (0, 1], got {config['mutation_prob']}"

        if 'crossover_prob' in config:
            if not 0 < config['crossover_prob'] <= 1:
                return False, f"'crossover_prob' must be in (0, 1], got {config['crossover_prob']}"

        if 'cv_groups' in config:
            if not isinstance(config['cv_groups'], int) or config['cv_groups'] < 2:
                return False, f"'cv_groups' must be >= 2, got {config['cv_groups']}"

        if 'min_vars' in config:
            if not isinstance(config['min_vars'], int) or config['min_vars'] < 1:
                return False, f"'min_vars' must be >= 1, got {config['min_vars']}"

        if 'max_vars' in config:
            if not isinstance(config['max_vars'], int) or config['max_vars'] < 1:
                return False, f"'max_vars' must be >= 1, got {config['max_vars']}"

            # Check consistency
            if 'min_vars' in config and config['max_vars'] < config['min_vars']:
                return False, f"'max_vars' ({config['max_vars']}) < 'min_vars' ({config['min_vars']})"

        return True, ""

    @staticmethod
    def validate_fitness_function(
        fitness_fn: callable,
        test_X: np.ndarray,
        test_y: Optional[np.ndarray] = None
    ) -> Tuple[bool, str]:
        """
        Test fitness function with sample data.

        Parameters
        ----------
        fitness_fn : callable
            Fitness evaluation function
        test_X : np.ndarray
            Small test dataset
        test_y : np.ndarray, optional
            Test target variable

        Returns
        -------
        is_valid : bool
            True if function works correctly
        message : str
            Error message if invalid
        """
        if not callable(fitness_fn):
            return False, "Fitness function must be callable"

        try:
            # Test with subset of variables
            n_vars = test_X.shape[1]
            test_indices = np.arange(min(5, n_vars))

            # Call fitness function
            if test_y is not None:
                result = fitness_fn(test_X[:, test_indices], test_y, test_indices)
            else:
                result = fitness_fn(test_X[:, test_indices], test_indices)

            # Check result type
            if not isinstance(result, (int, float, np.number)):
                return False, f"Fitness function must return numeric value, got {type(result)}"

            # Check result is finite
            result = float(result)
            if np.isnan(result) or np.isinf(result):
                return False, f"Fitness function returned invalid value: {result}"

            return True, ""

        except Exception as e:
            return False, f"Fitness function failed: {str(e)}"

    @staticmethod
    def estimate_memory_usage(
        n_samples: int,
        n_vars: int,
        config: Dict[str, Any]
    ) -> float:
        """
        Estimate memory usage for GA run.

        Parameters
        ----------
        n_samples : int
            Number of samples
        n_vars : int
            Number of variables
        config : dict
            GA configuration

        Returns
        -------
        memory_mb : float
            Estimated memory usage in MB
        """
        # Dataset storage
        dataset_mb = (n_samples * n_vars * 8) / (1024 ** 2)  # float64

        # Population storage
        population_size = config.get('population_size', 20)
        population_mb = (population_size * n_vars) / (1024 ** 2)  # bool

        # Library storage (all evaluated chromosomes)
        runs = config.get('runs', 20)
        evaluations = config.get('evaluations', 50)
        library_mb = (runs * evaluations * n_vars) / (1024 ** 2)

        # Overhead (CV, temporary arrays, etc.)
        overhead_mb = dataset_mb * 2  # Conservative estimate

        total_mb = dataset_mb + population_mb + library_mb + overhead_mb

        return total_mb

    @staticmethod
    def estimate_runtime(
        n_samples: int,
        n_vars: int,
        config: Dict[str, Any],
        problem_type: str = 'pls'
    ) -> float:
        """
        Estimate runtime for GA run.

        Parameters
        ----------
        n_samples : int
            Number of samples
        n_vars : int
            Number of variables
        config : dict
            GA configuration
        problem_type : str
            Type of problem (affects CV time)

        Returns
        -------
        runtime_minutes : float
            Estimated runtime in minutes
        """
        runs = config.get('runs', 20)
        population_size = config.get('population_size', 20)
        evaluations = config.get('evaluations', 50)
        cv_groups = config.get('cv_groups', 3)

        # Total fitness evaluations
        total_evals = runs * evaluations

        # Time per evaluation (empirical estimates)
        # Depends on: n_samples, n_vars, cv_groups, problem_type
        base_time_sec = 0.01  # Base overhead

        # CV time (dominant factor)
        if problem_type == 'pls':
            cv_time = 0.001 * n_samples * n_vars * cv_groups
        elif problem_type in ['lda', 'fda']:
            cv_time = 0.0005 * n_samples * n_vars * cv_groups
        else:  # mahalanobis, distance
            cv_time = 0.0002 * n_samples * n_vars

        time_per_eval = base_time_sec + cv_time

        # Total time
        total_seconds = total_evals * time_per_eval

        # Add overhead for initialization, backward steps, etc. (20%)
        total_seconds *= 1.2

        return total_seconds / 60  # Convert to minutes


def validate_ga_inputs(
    X: Union[np.ndarray, pd.DataFrame],
    y: Optional[Union[np.ndarray, pd.Series]],
    problem_type: str,
    config: Dict[str, Any]
) -> Tuple[bool, str]:
    """
    Comprehensive validation of all GA inputs.

    Parameters
    ----------
    X : np.ndarray or pd.DataFrame
        Feature matrix
    y : np.ndarray or pd.Series, optional
        Target variable (for supervised methods)
    problem_type : str
        Type of problem
    config : dict
        GA configuration

    Returns
    -------
    is_valid : bool
        True if all inputs are valid
    message : str
        Error message if invalid, empty string if valid
    """
    validator = GAValidator()

    # Validate dataset
    is_valid, message = validator.validate_dataset(X)
    if not is_valid:
        return False, f"Dataset validation failed: {message}"

    # Get dimensions
    if isinstance(X, pd.DataFrame):
        n_samples, n_vars = X.shape
    else:
        n_samples, n_vars = X.shape

    # Validate target (if provided)
    if y is not None and problem_type != 'mahalanobis':
        is_valid, message = validator.validate_target(y, problem_type, n_samples)
        if not is_valid:
            return False, f"Target validation failed: {message}"

    # Validate configuration
    is_valid, message = validator.validate_config(config)
    if not is_valid:
        return False, f"Configuration validation failed: {message}"

    # Check memory
    memory_mb = validator.estimate_memory_usage(n_samples, n_vars, config)
    if memory_mb > 2000:  # 2 GB limit
        warnings.warn(
            f"High memory usage expected: {memory_mb:.0f} MB. "
            "Consider reducing dataset size or GA parameters.",
            RuntimeWarning
        )

    # Check runtime
    runtime_min = validator.estimate_runtime(n_samples, n_vars, config, problem_type)
    if runtime_min > 60:  # 1 hour
        warnings.warn(
            f"Long runtime expected: {runtime_min:.0f} minutes. "
            "Consider using Fast mode or reducing parameters.",
            RuntimeWarning
        )

    return True, ""
