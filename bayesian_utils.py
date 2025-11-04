"""
Bayesian Optimization Utility Functions
========================================

Support utilities for Bayesian Optimization Designer including:
- Domain specification and conversion
- Constraint parsing
- Bounds validation
- Acquisition function computation
- Coordinate transformations (natural ↔ coded)
- Encoding detection (coded vs natural units)
- Inverse transformations (coded → natural)
- Results formatting

Author: ChemoMetric Solutions
License: MIT
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from typing import Dict, List, Tuple, Callable, Union, Optional
import warnings


def create_gpyopt_domain(bounds_dict: Dict[str, Dict]) -> List[Dict]:
    """
    Convert bounds dictionary to GPyOpt domain format.

    Takes a bounds dictionary with factor specifications and creates
    a list of domain dictionaries compatible with GPyOpt's domain format.

    Args:
        bounds_dict: Dictionary with structure:
            {
                'factor_name': {
                    'lower': float,
                    'upper': float,
                    'type': str ('continuous' or 'discrete'),
                    'step': float or None
                }
            }

    Returns:
        List of dictionaries in GPyOpt domain format:
        [
            {
                'name': 'factor_name',
                'type': 'continuous' or 'discrete',
                'domain': (lower, upper),
                'dimensionality': 1
            }
        ]

    Raises:
        ValueError: If bounds_dict is empty or improperly formatted

    Example:
        >>> bounds = {
        ...     'Temperature': {'lower': 20, 'upper': 100, 'type': 'continuous', 'step': None},
        ...     'Catalyst': {'lower': 0, 'upper': 4, 'type': 'discrete', 'step': 1}
        ... }
        >>> domain = create_gpyopt_domain(bounds)
        >>> print(domain[0]['name'])
        Temperature
    """
    if not bounds_dict:
        raise ValueError("bounds_dict cannot be empty")

    domain_list = []

    for factor_name, bounds_spec in bounds_dict.items():
        # Validate required keys
        required_keys = ['lower', 'upper', 'type']
        missing_keys = [k for k in required_keys if k not in bounds_spec]
        if missing_keys:
            raise ValueError(
                f"Factor '{factor_name}' missing required keys: {missing_keys}"
            )

        # Extract bounds
        lower = float(bounds_spec['lower'])
        upper = float(bounds_spec['upper'])
        var_type = bounds_spec['type'].lower()

        # Validate bounds
        if lower >= upper:
            raise ValueError(
                f"Factor '{factor_name}': lower bound ({lower}) must be < upper bound ({upper})"
            )

        # Validate type
        if var_type not in ['continuous', 'discrete']:
            raise ValueError(
                f"Factor '{factor_name}': type must be 'continuous' or 'discrete', got '{var_type}'"
            )

        # Build domain specification
        domain_spec = {
            'name': factor_name,
            'type': var_type,
            'domain': (lower, upper),
            'dimensionality': 1
        }

        # Add discrete values if applicable
        if var_type == 'discrete':
            step = bounds_spec.get('step')
            if step is not None and step > 0:
                # Generate discrete values
                n_steps = int((upper - lower) / step) + 1
                discrete_values = np.linspace(lower, upper, n_steps)
                domain_spec['domain'] = tuple(discrete_values)
            else:
                # Default: integer steps
                discrete_values = np.arange(lower, upper + 1, 1)
                domain_spec['domain'] = tuple(discrete_values)

        domain_list.append(domain_spec)

    return domain_list


def parse_constraint_string(constraint_expr: str) -> Callable:
    """
    Convert constraint string expression to callable function.

    Parses mathematical constraint expressions (e.g., "x[:, 0] + x[:, 1] <= 100")
    and returns a lambda function that evaluates the constraint on input arrays.

    Args:
        constraint_expr: String expression using 'x' as variable array
                        Supports operators: +, -, *, /, **, <=, >=, ==
                        Example: "x[:, 0]**2 + x[:, 1]**2 <= 100"

    Returns:
        Callable function that takes array x and returns boolean array
        True where constraint is satisfied

    Raises:
        ValueError: If expression has invalid syntax
        SyntaxError: If expression cannot be parsed

    Example:
        >>> constraint_fn = parse_constraint_string("x[:, 0] + x[:, 1] <= 10")
        >>> X = np.array([[1, 2], [5, 6]])
        >>> result = constraint_fn(X)
        >>> print(result)
        [True, False]

    Security Note:
        Uses eval() with restricted namespace. Only use with trusted input.
    """
    if not constraint_expr or not isinstance(constraint_expr, str):
        raise ValueError("constraint_expr must be a non-empty string")

    # Clean expression
    expr = constraint_expr.strip()

    # Validate expression contains 'x'
    if 'x' not in expr.lower():
        raise ValueError("Constraint expression must contain variable 'x'")

    # Check for valid comparison operators
    valid_operators = ['<=', '>=', '==', '<', '>']
    has_operator = any(op in expr for op in valid_operators)
    if not has_operator:
        raise ValueError(
            f"Constraint must contain comparison operator: {valid_operators}"
        )

    # Security: restrict available functions
    safe_dict = {
        'np': np,
        'abs': abs,
        'min': min,
        'max': max,
        'sum': sum,
        'sqrt': np.sqrt,
        'exp': np.exp,
        'log': np.log,
        'sin': np.sin,
        'cos': np.cos,
        '__builtins__': {}
    }

    try:
        # Parse constraint into left and right sides
        for op in ['<=', '>=', '==', '<', '>']:
            if op in expr:
                parts = expr.split(op)
                if len(parts) != 2:
                    raise ValueError(f"Invalid constraint format around operator '{op}'")

                left_expr, right_expr = parts[0].strip(), parts[1].strip()

                # Create lambda function
                if op == '<=':
                    constraint_fn = lambda x, l=left_expr, r=right_expr, sd=safe_dict: \
                        eval(l, sd, {'x': x}) <= eval(r, sd, {'x': x})
                elif op == '>=':
                    constraint_fn = lambda x, l=left_expr, r=right_expr, sd=safe_dict: \
                        eval(l, sd, {'x': x}) >= eval(r, sd, {'x': x})
                elif op == '==':
                    constraint_fn = lambda x, l=left_expr, r=right_expr, sd=safe_dict: \
                        eval(l, sd, {'x': x}) == eval(r, sd, {'x': x})
                elif op == '<':
                    constraint_fn = lambda x, l=left_expr, r=right_expr, sd=safe_dict: \
                        eval(l, sd, {'x': x}) < eval(r, sd, {'x': x})
                else:  # '>'
                    constraint_fn = lambda x, l=left_expr, r=right_expr, sd=safe_dict: \
                        eval(l, sd, {'x': x}) > eval(r, sd, {'x': x})

                # Test the function with dummy data
                test_x = np.array([[0.0, 0.0]])
                _ = constraint_fn(test_x)

                return constraint_fn

        raise ValueError("No valid comparison operator found")

    except SyntaxError as e:
        raise SyntaxError(f"Invalid syntax in constraint expression: {str(e)}")
    except Exception as e:
        raise ValueError(f"Error parsing constraint: {str(e)}")


def validate_bounds(bounds_dict: Dict[str, Dict], data_df: pd.DataFrame) -> Tuple[bool, str]:
    """
    Validate factor bounds against data and consistency rules.

    Checks that:
    1. All factors have valid lower < upper bounds
    2. Bounds encompass existing data (with warnings)
    3. Factor names exist in dataframe
    4. No numerical issues (NaN, inf)

    Args:
        bounds_dict: Dictionary of factor bounds
        data_df: DataFrame containing experimental data

    Returns:
        Tuple of (is_valid: bool, message: str)
        - is_valid: True if all validation passes
        - message: Description of validation result or errors

    Example:
        >>> bounds = {'Temperature': {'lower': 20, 'upper': 100, 'type': 'continuous'}}
        >>> data = pd.DataFrame({'Temperature': [25, 50, 75]})
        >>> valid, msg = validate_bounds(bounds, data)
        >>> print(valid)
        True
    """
    if not bounds_dict:
        return False, "bounds_dict is empty"

    if data_df is None or len(data_df) == 0:
        return False, "data_df is empty or None"

    issues = []
    warnings_list = []

    for factor_name, bounds_spec in bounds_dict.items():
        # Check factor exists in data
        if factor_name not in data_df.columns:
            issues.append(f"Factor '{factor_name}' not found in dataframe columns")
            continue

        # Extract bounds
        try:
            lower = float(bounds_spec['lower'])
            upper = float(bounds_spec['upper'])
        except (KeyError, ValueError, TypeError) as e:
            issues.append(f"Factor '{factor_name}': invalid bounds format ({str(e)})")
            continue

        # Check for NaN or inf
        if not np.isfinite(lower) or not np.isfinite(upper):
            issues.append(f"Factor '{factor_name}': bounds contain NaN or inf")
            continue

        # Check lower < upper
        if lower >= upper:
            issues.append(
                f"Factor '{factor_name}': lower bound ({lower}) must be < upper bound ({upper})"
            )
            continue

        # Check bounds encompass data
        factor_data = data_df[factor_name].dropna()
        if len(factor_data) > 0:
            data_min = factor_data.min()
            data_max = factor_data.max()

            if data_min < lower:
                warnings_list.append(
                    f"Factor '{factor_name}': data minimum ({data_min:.4f}) is below lower bound ({lower:.4f})"
                )

            if data_max > upper:
                warnings_list.append(
                    f"Factor '{factor_name}': data maximum ({data_max:.4f}) is above upper bound ({upper:.4f})"
                )

            # Check if bounds are too tight
            data_range = data_max - data_min
            bounds_range = upper - lower
            if bounds_range < data_range * 0.9:
                warnings_list.append(
                    f"Factor '{factor_name}': bounds range ({bounds_range:.4f}) is narrower than data range ({data_range:.4f})"
                )

        # Validate type if present
        if 'type' in bounds_spec:
            var_type = bounds_spec['type'].lower()
            if var_type not in ['continuous', 'discrete']:
                issues.append(
                    f"Factor '{factor_name}': invalid type '{var_type}' (must be 'continuous' or 'discrete')"
                )

        # Validate step for discrete variables
        if bounds_spec.get('type', '').lower() == 'discrete':
            step = bounds_spec.get('step')
            if step is not None:
                if step <= 0:
                    issues.append(f"Factor '{factor_name}': step must be positive")
                elif step > (upper - lower):
                    issues.append(f"Factor '{factor_name}': step is larger than bounds range")

    # Compile results
    if issues:
        return False, "Validation failed:\n" + "\n".join(f"- {issue}" for issue in issues)

    if warnings_list:
        warning_msg = "Validation passed with warnings:\n" + "\n".join(f"- {w}" for w in warnings_list)
        return True, warning_msg

    return True, f"All bounds valid for {len(bounds_dict)} factors"


def compute_acquisition_ei(mu: np.ndarray, sigma: np.ndarray,
                          y_max: float, jitter: float = 0.01) -> np.ndarray:
    """
    Compute Expected Improvement acquisition function.

    Expected Improvement (EI) balances exploitation (high predicted mean)
    and exploration (high uncertainty). It represents the expected amount
    by which a point will improve upon the current best observation.

    Formula:
        EI(x) = (μ(x) - y_max - ξ) * Φ(Z) + σ(x) * φ(Z)
        where Z = (μ(x) - y_max - ξ) / σ(x)
        Φ is standard normal CDF, φ is standard normal PDF
        ξ is jitter for exploration

    Args:
        mu: Array of predicted means from GP model
        sigma: Array of predicted standard deviations
        y_max: Current best observed value
        jitter: Exploration parameter (default: 0.01)

    Returns:
        Array of Expected Improvement values (same shape as mu)

    Example:
        >>> mu = np.array([1.0, 2.0, 3.0])
        >>> sigma = np.array([0.5, 0.3, 0.1])
        >>> y_max = 2.5
        >>> ei = compute_acquisition_ei(mu, sigma, y_max)
        >>> print(ei.shape)
        (3,)
    """
    # Validate inputs
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)

    if mu.shape != sigma.shape:
        raise ValueError(f"mu and sigma must have same shape, got {mu.shape} and {sigma.shape}")

    if not np.isfinite(y_max):
        raise ValueError("y_max must be finite")

    if jitter < 0:
        raise ValueError("jitter must be non-negative")

    # Handle edge case: zero or negative sigma
    # Add small epsilon to avoid division by zero
    epsilon = 1e-10
    sigma = np.maximum(sigma, epsilon)

    # Compute improvement
    improvement = mu - y_max - jitter

    # Compute Z-score
    Z = improvement / sigma

    # Compute Expected Improvement
    # EI = improvement * Φ(Z) + σ * φ(Z)
    ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)

    # Ensure non-negative (numerical stability)
    ei = np.maximum(ei, 0.0)

    return ei


def compute_acquisition_lcb(mu: np.ndarray, sigma: np.ndarray, weight: float = 2.0) -> np.ndarray:
    """
    Compute Lower Confidence Bound acquisition function.

    Lower Confidence Bound (LCB) is a pessimistic acquisition function
    that balances exploitation and exploration through the weight parameter.
    Lower values indicate more promising regions for minimization problems.

    Formula:
        LCB(x) = μ(x) - weight * σ(x)

    For maximization, negate the result or use weight as negative.

    Args:
        mu: Array of predicted means from GP model
        sigma: Array of predicted standard deviations
        weight: Exploration-exploitation trade-off parameter
               Higher weight → more exploration (default: 2.0)
               Typical range: [0.5, 3.0]

    Returns:
        Array of LCB values (same shape as mu)

    Example:
        >>> mu = np.array([1.0, 2.0, 3.0])
        >>> sigma = np.array([0.5, 0.3, 0.1])
        >>> lcb = compute_acquisition_lcb(mu, sigma, weight=2.0)
        >>> print(lcb)
        [0.  1.4 2.8]
    """
    # Validate inputs
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)

    if mu.shape != sigma.shape:
        raise ValueError(f"mu and sigma must have same shape, got {mu.shape} and {sigma.shape}")

    if not np.isfinite(weight):
        raise ValueError("weight must be finite")

    # Ensure sigma is non-negative
    sigma = np.maximum(sigma, 0.0)

    # Compute LCB
    lcb = mu - weight * sigma

    return lcb


def normalize_to_coded(X_natural: np.ndarray, bounds_dict: Dict[str, Dict]) -> np.ndarray:
    """
    Convert natural units to coded (normalized) coordinates [-1, 1].

    Transformation formula:
        X_coded = (2*X_natural - (upper + lower)) / (upper - lower)

    This maps:
        - lower bound → -1
        - center → 0
        - upper bound → +1

    Args:
        X_natural: Array in natural units, shape (n_samples, n_features)
        bounds_dict: Dictionary with 'lower' and 'upper' for each factor

    Returns:
        Array in coded units [-1, 1], same shape as input

    Raises:
        ValueError: If shapes don't match or bounds are invalid

    Example:
        >>> X = np.array([[25, 1.5], [75, 2.5]])
        >>> bounds = {
        ...     'Temperature': {'lower': 0, 'upper': 100},
        ...     'Pressure': {'lower': 1, 'upper': 3}
        ... }
        >>> X_coded = normalize_to_coded(X, bounds)
        >>> print(X_coded)
        [[-0.5 0. ]
         [ 0.5 0.5]]
    """
    # Validate inputs
    X_natural = np.asarray(X_natural)

    if X_natural.ndim == 1:
        X_natural = X_natural.reshape(-1, 1)

    n_features = X_natural.shape[1]

    if len(bounds_dict) != n_features:
        raise ValueError(
            f"Number of factors in bounds_dict ({len(bounds_dict)}) must match "
            f"number of columns in X_natural ({n_features})"
        )

    # Extract bounds in order
    factor_names = list(bounds_dict.keys())
    lowers = np.array([bounds_dict[f]['lower'] for f in factor_names])
    uppers = np.array([bounds_dict[f]['upper'] for f in factor_names])

    # Validate bounds
    if np.any(lowers >= uppers):
        invalid_factors = [factor_names[i] for i in range(len(factor_names))
                          if lowers[i] >= uppers[i]]
        raise ValueError(f"Invalid bounds (lower >= upper) for factors: {invalid_factors}")

    # Compute coded coordinates
    # X_coded = (2*X - (upper + lower)) / (upper - lower)
    centers = (uppers + lowers) / 2.0
    ranges = (uppers - lowers) / 2.0

    X_coded = (X_natural - centers) / ranges

    # Handle edge case: constant factor (range = 0)
    zero_range_mask = ranges == 0
    if np.any(zero_range_mask):
        warnings.warn(
            f"Some factors have zero range: {[factor_names[i] for i in np.where(zero_range_mask)[0]]}"
        )
        X_coded[:, zero_range_mask] = 0.0

    return X_coded


def denormalize_to_natural(X_coded: np.ndarray, bounds_dict: Dict[str, Dict]) -> np.ndarray:
    """
    Convert coded (normalized) coordinates [-1, 1] to natural units.

    Inverse transformation of normalize_to_coded:
        X_natural = X_coded * (upper - lower) / 2 + (upper + lower) / 2

    Args:
        X_coded: Array in coded units [-1, 1], shape (n_samples, n_features)
        bounds_dict: Dictionary with 'lower' and 'upper' for each factor

    Returns:
        Array in natural units, same shape as input

    Raises:
        ValueError: If shapes don't match or bounds are invalid

    Example:
        >>> X_coded = np.array([[-1, 0], [1, 1]])
        >>> bounds = {
        ...     'Temperature': {'lower': 0, 'upper': 100},
        ...     'Pressure': {'lower': 1, 'upper': 3}
        ... }
        >>> X_natural = denormalize_to_natural(X_coded, bounds)
        >>> print(X_natural)
        [[  0.  1.]
         [100.  3.]]
    """
    # Validate inputs
    X_coded = np.asarray(X_coded)

    if X_coded.ndim == 1:
        X_coded = X_coded.reshape(-1, 1)

    n_features = X_coded.shape[1]

    if len(bounds_dict) != n_features:
        raise ValueError(
            f"Number of factors in bounds_dict ({len(bounds_dict)}) must match "
            f"number of columns in X_coded ({n_features})"
        )

    # Extract bounds in order
    factor_names = list(bounds_dict.keys())
    lowers = np.array([bounds_dict[f]['lower'] for f in factor_names])
    uppers = np.array([bounds_dict[f]['upper'] for f in factor_names])

    # Validate bounds
    if np.any(lowers >= uppers):
        invalid_factors = [factor_names[i] for i in range(len(factor_names))
                          if lowers[i] >= uppers[i]]
        raise ValueError(f"Invalid bounds (lower >= upper) for factors: {invalid_factors}")

    # Compute natural coordinates
    # X_natural = X_coded * range + center
    centers = (uppers + lowers) / 2.0
    ranges = (uppers - lowers) / 2.0

    X_natural = X_coded * ranges + centers

    return X_natural


def detect_encoding(data: pd.DataFrame) -> Dict[str, Union[str, float, Dict]]:
    """
    Detect if DataFrame contains coded (DoE-style) or natural units data.

    Analyzes data characteristics to determine encoding type:
    - Coded data: Values close to {-1, 0, +1}, small ranges, std ~0.8-1.0
    - Natural data: Large ranges, continuous distributions, variable scales

    Detection Logic:
    1. Check if values cluster near -1, 0, +1 (tolerance ±0.15)
    2. Check if ranges are small (-1.5 to 1.5) vs large (>5)
    3. Check column statistics: std near 0.8-1.0 for coded
    4. Aggregate evidence across all numeric columns

    Args:
        data: DataFrame with experimental data

    Returns:
        Dictionary with:
        - 'encoding': str, 'coded' or 'natural'
        - 'confidence': float, 0-100% confidence level
        - 'columns_analysis': dict, per-column detection details
        - 'summary': dict, aggregate statistics

    Example:
        >>> # Coded data example
        >>> coded_df = pd.DataFrame({
        ...     'Factor_A': [-1, 0, 1, -1, 0, 1],
        ...     'Factor_B': [-1, -1, 0, 0, 1, 1]
        ... })
        >>> result = detect_encoding(coded_df)
        >>> print(result['encoding'])
        coded
        >>> print(f"{result['confidence']:.1f}%")
        95.0%

        >>> # Natural units example
        >>> natural_df = pd.DataFrame({
        ...     'Temperature': [25.5, 50.3, 75.1, 100.2],
        ...     'Pressure': [1.2, 2.4, 3.6, 4.8]
        ... })
        >>> result = detect_encoding(natural_df)
        >>> print(result['encoding'])
        natural
    """
    if data is None or len(data) == 0:
        return {
            'encoding': 'unknown',
            'confidence': 0.0,
            'columns_analysis': {},
            'summary': {'message': 'Empty or None DataFrame'}
        }

    # Get numeric columns only
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()

    if not numeric_cols:
        return {
            'encoding': 'unknown',
            'confidence': 0.0,
            'columns_analysis': {},
            'summary': {'message': 'No numeric columns found'}
        }

    # Analyze each column
    columns_analysis = {}
    coded_evidence = []
    natural_evidence = []

    for col in numeric_cols:
        values = data[col].dropna()

        if len(values) == 0:
            continue

        # Statistics
        col_min = values.min()
        col_max = values.max()
        col_range = col_max - col_min
        col_mean = values.mean()
        col_std = values.std()
        n_unique = len(values.unique())

        # Evidence 1: Proximity to DoE levels (-1, 0, +1)
        tolerance = 0.15
        near_minus_one = np.sum(np.abs(values + 1) < tolerance)
        near_zero = np.sum(np.abs(values) < tolerance)
        near_plus_one = np.sum(np.abs(values - 1) < tolerance)
        near_doe_levels = near_minus_one + near_zero + near_plus_one

        pct_near_doe = (near_doe_levels / len(values)) * 100

        # Evidence 2: Range check
        in_coded_range = (col_min >= -1.5 and col_max <= 1.5)

        # Evidence 3: Standard deviation check
        # Coded data typically has std around 0.8-1.0 (for -1, 0, +1 levels)
        std_coded_like = (0.5 <= col_std <= 1.2)

        # Evidence 4: Limited unique values
        limited_unique = (n_unique <= 5)

        # Evidence 5: Centered around zero
        centered = (abs(col_mean) < 0.3)

        # Decision for this column
        coded_score = 0
        natural_score = 0

        if pct_near_doe > 70 and in_coded_range and limited_unique:
            # Strong evidence for coded
            coded_score = 100
            col_type = 'coded'
        elif pct_near_doe > 50 and in_coded_range:
            # Moderate evidence for coded
            coded_score = 70
            col_type = 'coded'
        elif n_unique > 10 and col_range > 5.0:
            # Strong evidence for natural
            natural_score = 100
            col_type = 'natural'
        elif col_range > 2.5 and not in_coded_range:
            # Moderate evidence for natural
            natural_score = 70
            col_type = 'natural'
        elif pct_near_doe > 30 and in_coded_range and std_coded_like:
            # Some evidence for coded
            coded_score = 50
            col_type = 'coded_likely'
        else:
            # Uncertain
            col_type = 'uncertain'
            coded_score = 50
            natural_score = 50

        # Store analysis
        columns_analysis[col] = {
            'type': col_type,
            'min': float(col_min),
            'max': float(col_max),
            'range': float(col_range),
            'mean': float(col_mean),
            'std': float(col_std),
            'n_unique': int(n_unique),
            'pct_near_doe_levels': float(pct_near_doe),
            'in_coded_range': bool(in_coded_range),
            'std_coded_like': bool(std_coded_like),
            'centered': bool(centered),
            'coded_score': coded_score,
            'natural_score': natural_score
        }

        coded_evidence.append(coded_score)
        natural_evidence.append(natural_score)

    # Aggregate decision across all columns
    if not coded_evidence:
        return {
            'encoding': 'unknown',
            'confidence': 0.0,
            'columns_analysis': columns_analysis,
            'summary': {'message': 'No valid numeric data for analysis'}
        }

    avg_coded_score = np.mean(coded_evidence)
    avg_natural_score = np.mean(natural_evidence)

    # Count column types
    coded_count = sum(1 for c in columns_analysis.values() if 'coded' in c['type'])
    natural_count = sum(1 for c in columns_analysis.values() if c['type'] == 'natural')
    uncertain_count = sum(1 for c in columns_analysis.values() if c['type'] == 'uncertain')

    # Final decision
    if avg_coded_score > avg_natural_score:
        encoding = 'coded'
        confidence = min(avg_coded_score, 95.0)  # Cap at 95%
    elif avg_natural_score > avg_coded_score:
        encoding = 'natural'
        confidence = min(avg_natural_score, 95.0)
    else:
        encoding = 'uncertain'
        confidence = 50.0

    # Adjust confidence based on consistency
    if coded_count > 0 and natural_count > 0:
        # Mixed evidence reduces confidence
        confidence *= 0.7

    # Summary
    summary = {
        'n_columns_analyzed': len(coded_evidence),
        'coded_columns': coded_count,
        'natural_columns': natural_count,
        'uncertain_columns': uncertain_count,
        'avg_coded_score': float(avg_coded_score),
        'avg_natural_score': float(avg_natural_score),
        'recommendation': (
            'Data appears to be in DoE-style coded units (-1, 0, +1)'
            if encoding == 'coded' else
            'Data appears to be in natural units (physical/engineering scale)'
            if encoding == 'natural' else
            'Data encoding is uncertain - mixed or unusual patterns detected'
        )
    }

    return {
        'encoding': encoding,
        'confidence': float(confidence),
        'columns_analysis': columns_analysis,
        'summary': summary
    }


def inverse_transform_predictions(predictions_coded: pd.DataFrame,
                                  original_bounds: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """
    Convert coded predictions [-1, +1] back to natural units.

    Performs the inverse transformation of column_range_11, converting
    DoE-style coded values back to their original physical/engineering scales.
    This is used when Bayesian Optimization suggests optimal points in coded
    coordinates, and we need to report them in natural units.

    Transformation formula (inverse of column_range_11):
        natural_value = (coded_value + 1) * (max - min) / 2 + min

    This maps:
        - coded: -1 → natural: min
        - coded:  0 → natural: (min + max) / 2
        - coded: +1 → natural: max

    Args:
        predictions_coded: DataFrame with coded predictions [-1, +1]
                          Columns should be factor names
        original_bounds: Dictionary with original bounds for each factor
                        Structure:
                        {
                            'factor_name': {
                                'original_min': float,
                                'original_max': float,
                                ...
                            }
                        }
                        OR (alternative format from infer_factor_bounds):
                        {
                            'factor_name': {
                                'lower': float,
                                'upper': float,
                                'data_min': float,
                                'data_max': float,
                                ...
                            }
                        }

    Returns:
        DataFrame with natural units, same shape as predictions_coded
        Ready for reporting to users

    Raises:
        ValueError: If predictions_coded is empty or bounds are missing
        KeyError: If factor in predictions_coded not found in original_bounds

    Example:
        >>> # Coded predictions from BO
        >>> coded_df = pd.DataFrame({
        ...     'Temperature': [-0.5, 0.0, 0.5],
        ...     'Pressure': [-1.0, 0.0, 1.0]
        ... })
        >>>
        >>> # Original bounds from transformation
        >>> bounds = {
        ...     'Temperature': {'original_min': 20.0, 'original_max': 100.0},
        ...     'Pressure': {'original_min': 1.0, 'original_max': 5.0}
        ... }
        >>>
        >>> natural_df = inverse_transform_predictions(coded_df, bounds)
        >>> print(natural_df)
           Temperature  Pressure
        0         40.0       1.0
        1         60.0       3.0
        2         80.0       5.0

        >>> # Alternative: bounds from infer_factor_bounds
        >>> bounds2 = {
        ...     'Temperature': {'lower': 20.0, 'upper': 100.0, 'data_min': 25.0, 'data_max': 95.0},
        ...     'Pressure': {'lower': 1.0, 'upper': 5.0, 'data_min': 1.5, 'data_max': 4.5}
        ... }
        >>> natural_df2 = inverse_transform_predictions(coded_df, bounds2)
        >>> # Uses data_min/data_max if available, otherwise lower/upper

    Usage in Bayesian Optimization workflow:
        >>> # After BO suggests optimal point
        >>> optimal_coded = pd.DataFrame({
        ...     'Factor_A': [0.75],
        ...     'Factor_B': [-0.25]
        ... })
        >>>
        >>> # Convert to natural units for user
        >>> optimal_natural = inverse_transform_predictions(optimal_coded, transformation_metadata)
        >>>
        >>> # Display both
        >>> st.write("Optimal conditions (coded):", optimal_coded)
        >>> st.write("Optimal conditions (natural units):", optimal_natural)
    """
    # Validate inputs
    if predictions_coded is None or len(predictions_coded) == 0:
        raise ValueError("predictions_coded is empty or None")

    if not original_bounds or len(original_bounds) == 0:
        raise ValueError("original_bounds is empty or None")

    # Create copy to avoid modifying original
    predictions_natural = predictions_coded.copy()

    # Process each factor column
    for factor_name in predictions_coded.columns:
        # Check if factor exists in bounds
        if factor_name not in original_bounds:
            raise KeyError(
                f"Factor '{factor_name}' not found in original_bounds. "
                f"Available factors: {list(original_bounds.keys())}"
            )

        bounds_info = original_bounds[factor_name]

        # Extract min/max from bounds dictionary
        # Support two formats:
        # 1. Transformation metadata format: 'original_min', 'original_max'
        # 2. Infer bounds format: 'data_min', 'data_max' or 'lower', 'upper'

        if 'original_min' in bounds_info and 'original_max' in bounds_info:
            # Format 1: From transformation metadata
            orig_min = bounds_info['original_min']
            orig_max = bounds_info['original_max']
        elif 'data_min' in bounds_info and 'data_max' in bounds_info:
            # Format 2a: From infer_factor_bounds with data stats
            orig_min = bounds_info['data_min']
            orig_max = bounds_info['data_max']
        elif 'lower' in bounds_info and 'upper' in bounds_info:
            # Format 2b: From infer_factor_bounds, use bounds as fallback
            orig_min = bounds_info['lower']
            orig_max = bounds_info['upper']
        else:
            raise ValueError(
                f"Factor '{factor_name}': bounds_info must contain either "
                f"('original_min', 'original_max') or ('data_min', 'data_max') or ('lower', 'upper'). "
                f"Got keys: {list(bounds_info.keys())}"
            )

        # Validate bounds
        orig_min = float(orig_min)
        orig_max = float(orig_max)

        if not np.isfinite(orig_min) or not np.isfinite(orig_max):
            raise ValueError(f"Factor '{factor_name}': bounds contain NaN or inf")

        if orig_min >= orig_max:
            raise ValueError(
                f"Factor '{factor_name}': original_min ({orig_min}) must be < original_max ({orig_max})"
            )

        # Get coded values
        coded_values = predictions_coded[factor_name].values

        # Apply inverse transformation
        # Inverse of column_range_11:
        #   coded = 2 * (natural - min) / (max - min) - 1
        # Solving for natural:
        #   natural = (coded + 1) * (max - min) / 2 + min

        natural_values = (coded_values + 1) * (orig_max - orig_min) / 2 + orig_min

        # Store in output dataframe
        predictions_natural[factor_name] = natural_values

    # Round to reasonable precision based on magnitude
    for col in predictions_natural.columns:
        values = predictions_natural[col].values

        # Determine appropriate precision
        if factor_name in original_bounds:
            bounds_info = original_bounds[col]

            # Get range
            if 'original_min' in bounds_info and 'original_max' in bounds_info:
                val_range = bounds_info['original_max'] - bounds_info['original_min']
            elif 'data_min' in bounds_info and 'data_max' in bounds_info:
                val_range = bounds_info['data_max'] - bounds_info['data_min']
            elif 'lower' in bounds_info and 'upper' in bounds_info:
                val_range = bounds_info['upper'] - bounds_info['lower']
            else:
                val_range = np.ptp(values)  # Peak-to-peak (max - min)

            # Adaptive rounding
            if val_range < 1:
                decimals = 4
            elif val_range < 10:
                decimals = 3
            elif val_range < 100:
                decimals = 2
            else:
                decimals = 1

            predictions_natural[col] = predictions_natural[col].round(decimals)

    return predictions_natural


def infer_factor_bounds(data: pd.DataFrame,
                       encoding_info: Optional[Dict] = None) -> Dict[str, Dict[str, float]]:
    """
    Infer factor bounds from data based on encoding type.

    For coded data: Uses standard DoE bounds [-1, +1]
    For natural data: Uses data min/max with margin

    Args:
        data: DataFrame with experimental data
        encoding_info: Optional output from detect_encoding()
                      If None, will run detection automatically

    Returns:
        Dictionary with factor bounds:
        {
            'factor_name': {
                'lower': float,
                'upper': float,
                'type': 'continuous',
                'inferred_from': 'coded_standard' or 'data_range',
                'data_min': float,
                'data_max': float
            }
        }

    Example:
        >>> # Coded data
        >>> coded_df = pd.DataFrame({
        ...     'Factor_A': [-1, 0, 1, -1, 0],
        ...     'Factor_B': [-1, 0, 1, 0, 1]
        ... })
        >>> bounds = infer_factor_bounds(coded_df)
        >>> print(bounds['Factor_A'])
        {'lower': -1.0, 'upper': 1.0, 'type': 'continuous',
         'inferred_from': 'coded_standard', ...}

        >>> # Natural units
        >>> natural_df = pd.DataFrame({
        ...     'Temperature': [25, 50, 75, 100],
        ...     'Pressure': [1.0, 2.0, 3.0, 4.0]
        ... })
        >>> bounds = infer_factor_bounds(natural_df)
        >>> print(bounds['Temperature'])
        {'lower': 17.5, 'upper': 107.5, 'type': 'continuous',
         'inferred_from': 'data_range', ...}
    """
    if data is None or len(data) == 0:
        return {}

    # Run encoding detection if not provided
    if encoding_info is None:
        encoding_info = detect_encoding(data)

    encoding = encoding_info.get('encoding', 'natural')
    columns_analysis = encoding_info.get('columns_analysis', {})

    # Get numeric columns
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()

    if not numeric_cols:
        return {}

    bounds_dict = {}

    for col in numeric_cols:
        values = data[col].dropna()

        if len(values) == 0:
            continue

        # Get data statistics
        data_min = float(values.min())
        data_max = float(values.max())
        data_range = data_max - data_min

        # Get column-specific analysis if available
        col_analysis = columns_analysis.get(col, {})
        col_type = col_analysis.get('type', 'natural')

        # Infer bounds based on encoding
        if encoding == 'coded' or 'coded' in col_type:
            # Use standard coded bounds
            lower = -1.0
            upper = 1.0
            inferred_from = 'coded_standard'

            # If data extends beyond [-1, +1], use data range with small margin
            if data_min < -1.0 or data_max > 1.0:
                margin = 0.1 * data_range
                lower = data_min - margin
                upper = data_max + margin
                inferred_from = 'coded_extended'

        else:
            # Natural units: use data range with margin
            margin_pct = 0.1  # 10% margin

            if data_range > 0:
                margin = margin_pct * data_range
                lower = data_min - margin
                upper = data_max + margin
            else:
                # Constant column - add small symmetric margin
                lower = data_min - 0.5
                upper = data_max + 0.5

            inferred_from = 'data_range'

        # Round bounds for cleaner display
        # Use different precision based on magnitude
        if abs(upper - lower) < 1:
            decimals = 4
        elif abs(upper - lower) < 10:
            decimals = 2
        else:
            decimals = 1

        lower = round(lower, decimals)
        upper = round(upper, decimals)

        # Ensure lower < upper (handle edge cases)
        if lower >= upper:
            upper = lower + 0.1

        # Store bounds
        bounds_dict[col] = {
            'lower': lower,
            'upper': upper,
            'type': 'continuous',  # Default to continuous
            'inferred_from': inferred_from,
            'data_min': round(data_min, decimals),
            'data_max': round(data_max, decimals),
            'margin_added': inferred_from == 'data_range'
        }

    return bounds_dict


def format_results_display(suggested_df: pd.DataFrame,
                          factor_names: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Format Bayesian Optimization results for display.

    Applies formatting:
    - Rounds numeric columns to 4 decimals
    - Formats target and acquisition columns with descriptive names
    - Optionally reorders columns with factors first
    - Adds ranking column based on acquisition value

    Args:
        suggested_df: DataFrame with BO results
        factor_names: Optional list of factor names to display first

    Returns:
        Formatted DataFrame ready for display

    Example:
        >>> results = pd.DataFrame({
        ...     'Temperature': [25.123456, 75.987654],
        ...     'Pressure': [1.555555, 2.444444],
        ...     'Expected_Target': [10.123456, 12.987654],
        ...     'Acquisition_Value': [0.555555, 0.888888]
        ... })
        >>> formatted = format_results_display(results, ['Temperature', 'Pressure'])
        >>> print(formatted)
    """
    if suggested_df is None or len(suggested_df) == 0:
        raise ValueError("suggested_df is empty or None")

    # Create copy to avoid modifying original
    df_display = suggested_df.copy()

    # Round numeric columns to 4 decimals
    numeric_cols = df_display.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df_display[col] = df_display[col].round(4)

    # Rename common BO output columns for clarity
    column_renames = {
        'Expected_Target': 'Predicted Response',
        'Std_Dev': 'Uncertainty (±)',
        'Acquisition_Value': 'Acquisition Score'
    }

    for old_name, new_name in column_renames.items():
        if old_name in df_display.columns:
            df_display.rename(columns={old_name: new_name}, inplace=True)

    # Add rank column based on acquisition score
    if 'Acquisition Score' in df_display.columns:
        df_display['Rank'] = df_display['Acquisition Score'].rank(ascending=False).astype(int)

    # Reorder columns: factors first, then predictions, then scores
    if factor_names is not None:
        # Verify factor names exist
        existing_factors = [f for f in factor_names if f in df_display.columns]

        # Build column order
        other_cols = [c for c in df_display.columns if c not in existing_factors]

        # Prioritize certain columns
        priority_cols = ['Rank', 'Predicted Response', 'Uncertainty (±)', 'Acquisition Score']
        ordered_other = [c for c in priority_cols if c in other_cols]
        remaining = [c for c in other_cols if c not in ordered_other]

        new_order = existing_factors + ordered_other + remaining
        df_display = df_display[new_order]

    # Sort by rank if available
    if 'Rank' in df_display.columns:
        df_display = df_display.sort_values('Rank')

    # Reset index for clean display
    df_display.reset_index(drop=True, inplace=True)

    return df_display


# Testing utilities
def run_module_tests():
    """Run basic tests for all utility functions."""
    print("Running bayesian_utils tests...")
    print("=" * 50)

    # Test 1: create_gpyopt_domain
    print("\n1. Testing create_gpyopt_domain...")
    bounds = {
        'Temperature': {'lower': 20, 'upper': 100, 'type': 'continuous', 'step': None},
        'Catalyst': {'lower': 0, 'upper': 4, 'type': 'discrete', 'step': 1}
    }
    domain = create_gpyopt_domain(bounds)
    print(f"   Created domain with {len(domain)} factors")
    assert len(domain) == 2
    print("   ✓ Passed")

    # Test 2: parse_constraint_string
    print("\n2. Testing parse_constraint_string...")
    constraint_fn = parse_constraint_string("x[:, 0] + x[:, 1] <= 100")
    X_test = np.array([[10, 20], [60, 50]])
    result = constraint_fn(X_test)
    print(f"   Constraint results: {result}")
    assert result[0] == True and result[1] == False
    print("   ✓ Passed")

    # Test 3: validate_bounds
    print("\n3. Testing validate_bounds...")
    data = pd.DataFrame({'Temperature': [25, 50, 75], 'Catalyst': [1, 2, 3]})
    valid, msg = validate_bounds(bounds, data)
    print(f"   Valid: {valid}, Message: {msg[:50]}...")
    assert valid == True
    print("   ✓ Passed")

    # Test 4: compute_acquisition_ei
    print("\n4. Testing compute_acquisition_ei...")
    mu = np.array([1.0, 2.0, 3.0])
    sigma = np.array([0.5, 0.3, 0.1])
    ei = compute_acquisition_ei(mu, sigma, y_max=2.5)
    print(f"   EI values: {ei}")
    assert ei.shape == mu.shape
    print("   ✓ Passed")

    # Test 5: compute_acquisition_lcb
    print("\n5. Testing compute_acquisition_lcb...")
    lcb = compute_acquisition_lcb(mu, sigma, weight=2.0)
    print(f"   LCB values: {lcb}")
    assert lcb.shape == mu.shape
    print("   ✓ Passed")

    # Test 6: normalize_to_coded
    print("\n6. Testing normalize_to_coded...")
    X_natural = np.array([[25, 1], [75, 3]])
    bounds_norm = {
        'Temperature': {'lower': 0, 'upper': 100},
        'Pressure': {'lower': 1, 'upper': 3}
    }
    X_coded = normalize_to_coded(X_natural, bounds_norm)
    print(f"   Coded values:\n{X_coded}")
    assert X_coded.shape == X_natural.shape
    print("   ✓ Passed")

    # Test 7: denormalize_to_natural
    print("\n7. Testing denormalize_to_natural...")
    X_recovered = denormalize_to_natural(X_coded, bounds_norm)
    print(f"   Recovered values:\n{X_recovered}")
    assert np.allclose(X_recovered, X_natural)
    print("   ✓ Passed")

    # Test 8: format_results_display
    print("\n8. Testing format_results_display...")
    results = pd.DataFrame({
        'Temperature': [25.123456, 75.987654],
        'Pressure': [1.555555, 2.444444],
        'Expected_Target': [10.123456, 12.987654],
        'Acquisition_Value': [0.555555, 0.888888]
    })
    formatted = format_results_display(results, ['Temperature', 'Pressure'])
    print(f"   Formatted shape: {formatted.shape}")
    print(f"   Columns: {list(formatted.columns)}")
    assert 'Rank' in formatted.columns
    print("   ✓ Passed")

    print("\n" + "=" * 50)
    print("All tests passed! ✓")


if __name__ == "__main__":
    print("Bayesian Optimization Utilities Module")
    print("=" * 50)
    print("\nAvailable functions:")
    print("1. create_gpyopt_domain() - Convert bounds to GPyOpt format")
    print("2. parse_constraint_string() - Parse constraint expressions")
    print("3. validate_bounds() - Validate bounds against data")
    print("4. compute_acquisition_ei() - Expected Improvement")
    print("5. compute_acquisition_lcb() - Lower Confidence Bound")
    print("6. normalize_to_coded() - Natural → Coded coordinates")
    print("7. denormalize_to_natural() - Coded → Natural coordinates")
    print("8. format_results_display() - Format results for display")
    print("\n" + "=" * 50)

    # Run tests
    run_module_tests()
