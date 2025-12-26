"""
Candidate Matrix Generator
Creates candidate point matrices for D-optimal design.
"""

import numpy as np
import pandas as pd
import itertools
from typing import Dict, List, Optional, Callable


def create_candidate_matrix(variables_config: dict) -> pd.DataFrame:
    """
    Create the complete candidate point matrix from variable configuration.

    Supports TWO input formats for backward compatibility:

    Format 1 (NEW - with type specification):
        {
            'Temperature': {
                'type': 'Quantitative',
                'min': 10,
                'max': 50,
                'step': 10,
                'levels': [10, 20, 30, 40, 50]  # Auto-generated or provided
            },
            'Pressure': {
                'type': 'Categorical',
                'levels': ['high', 'low']
            }
        }

    Format 2 (OLD - backward compatible):
        {
            'Temperature': {'min': 50, 'max': 150, 'levels': [50, 75, 100, 125, 150]},
            'Pressure': {'min': 1, 'max': 5, 'step': 1}
        }

    Returns:
        DataFrame with ALL possible combinations of levels (NO Experiment_ID column yet)
    """
    # Extract variable names in order
    var_names = list(variables_config.keys())

    # Extract or generate levels for each variable
    levels_list = []
    for var_name in var_names:
        config = variables_config[var_name]

        # Check variable type (new format) or default to quantitative
        var_type = config.get('type', 'Quantitative')

        # Extract levels based on format
        if 'levels' in config and len(config['levels']) > 0:
            # Levels already provided
            levels = config['levels']

        elif 'step' in config:
            # Generate levels from min, max, and step
            min_val = config.get('min', 0)
            max_val = config.get('max', 100)
            step = config['step']

            if step > 0 and min_val < max_val:
                levels = list(np.arange(min_val, max_val + step/2, step))
                # Clean up floating point errors
                levels = [round(x, 10) for x in levels]
            else:
                # Invalid step, use min and max only
                levels = [min_val, max_val]

        elif 'min' in config and 'max' in config:
            # No step specified - default: generate 5 levels from min to max
            min_val = config['min']
            max_val = config['max']

            if min_val < max_val:
                levels = list(np.linspace(min_val, max_val, 5))
                levels = [round(x, 10) for x in levels]
            else:
                levels = [min_val, max_val]
        else:
            # No sufficient information - error
            raise ValueError(f"Variable '{var_name}' must have either 'levels' or 'min'+'max' specified")

        # For categorical variables, keep levels as-is (strings or numbers)
        # For quantitative, ensure numeric
        if var_type == 'Quantitative':
            try:
                levels = [float(x) for x in levels]
            except (ValueError, TypeError):
                # Cannot convert to float - treat as categorical
                pass

        levels_list.append(levels)

    # Generate all combinations using itertools.product
    combinations = list(itertools.product(*levels_list))

    # Create DataFrame
    df = pd.DataFrame(combinations, columns=var_names)

    # For quantitative columns, sort by all columns for consistency
    # For categorical, keep order as defined
    try:
        df = df.sort_values(by=var_names).reset_index(drop=True)
    except TypeError:
        # Mixed types - can't sort consistently, just reset index
        df = df.reset_index(drop=True)

    # DO NOT add Experiment_ID here - it will be added after encoding
    # This allows encoding step to work on clean data

    return df


def apply_constraints(candidate_matrix: pd.DataFrame, constraints: list) -> pd.DataFrame:
    """
    Remove candidate points that violate specified constraints.

    Args:
        candidate_matrix: Full candidate DataFrame
        constraints: List of constraint dictionaries
            Example: [
                {'type': 'exclude_rows', 'indices': [5, 12, 18]},
                {'type': 'custom_expression', 'expression': lambda row: row['Temperature'] > 2*row['Pressure']},
                {'type': 'range', 'variable': 'Temperature', 'min': 50, 'max': 150}
            ]

    Returns:
        Filtered DataFrame with remaining valid candidates
    """
    if not constraints:
        return candidate_matrix.copy()

    df = candidate_matrix.copy()
    initial_count = len(df)

    for constraint in constraints:
        constraint_type = constraint.get('type', 'custom_expression')

        if constraint_type == 'expression':
            # Parse and apply logical expression
            expr_str = constraint.get('expression', '')

            if expr_str:
                try:
                    # Replace column names with df column references
                    expr_eval = expr_str
                    for col_name in df.columns:
                        expr_eval = expr_eval.replace(col_name, f"row['{col_name}']")

                    # Apply expression as filter
                    mask = df.apply(lambda row: eval(expr_eval), axis=1)
                    df = df[mask]

                except Exception as e:
                    print(f"Warning: Could not evaluate expression '{expr_str}': {e}")

        elif constraint_type == 'exclude_rows':
            # Exclude specific row indices
            indices = constraint.get('indices', [])
            # Convert to Experiment_ID if provided as row numbers
            if 'Experiment_ID' in df.columns:
                df = df[~df['Experiment_ID'].isin(indices)]
            else:
                df = df.drop(index=indices, errors='ignore')

        elif constraint_type == 'custom_expression':
            # Apply custom boolean expression
            expression = constraint.get('expression')
            if expression and callable(expression):
                try:
                    mask = df.apply(expression, axis=1)
                    df = df[mask]
                except Exception as e:
                    print(f"Warning: Could not apply constraint expression: {e}")

        elif constraint_type == 'range':
            # Apply range constraint on a variable
            variable = constraint.get('variable')
            min_val = constraint.get('min', -np.inf)
            max_val = constraint.get('max', np.inf)
            if variable in df.columns:
                df = df[(df[variable] >= min_val) & (df[variable] <= max_val)]

        elif constraint_type == 'comparison':
            # Apply comparison between variables (e.g., var1 > var2)
            var1 = constraint.get('var1')
            var2 = constraint.get('var2')
            operator = constraint.get('operator', '>')

            if var1 in df.columns and var2 in df.columns:
                if operator == '>':
                    df = df[df[var1] > df[var2]]
                elif operator == '>=':
                    df = df[df[var1] >= df[var2]]
                elif operator == '<':
                    df = df[df[var1] < df[var2]]
                elif operator == '<=':
                    df = df[df[var1] <= df[var2]]
                elif operator == '==':
                    df = df[df[var1] == df[var2]]
                elif operator == '!=':
                    df = df[df[var1] != df[var2]]

    final_count = len(df)
    removed_count = initial_count - final_count

    if removed_count > 0:
        print(f"Constraints removed {removed_count} candidates ({initial_count} → {final_count})")

    # Reset index and Experiment_ID
    df = df.reset_index(drop=True)
    if 'Experiment_ID' in df.columns:
        df['Experiment_ID'] = range(1, len(df) + 1)

    return df


def validate_candidate_matrix(candidate_matrix: pd.DataFrame) -> tuple:
    """
    Validate candidate matrix for D-optimal design.

    Args:
        candidate_matrix: Candidate DataFrame

    Returns:
        Tuple (is_valid: bool, message: str)
    """
    if len(candidate_matrix) < 2:
        return False, "Candidate matrix must have at least 2 rows"

    # Check for duplicates
    if 'Experiment_ID' in candidate_matrix.columns:
        data_cols = [col for col in candidate_matrix.columns if col != 'Experiment_ID']
    else:
        data_cols = candidate_matrix.columns

    duplicates = candidate_matrix.duplicated(subset=data_cols).sum()
    if duplicates > 0:
        return False, f"Candidate matrix contains {duplicates} duplicate rows"

    # Check for NaN values
    nan_count = candidate_matrix[data_cols].isna().sum().sum()
    if nan_count > 0:
        return False, f"Candidate matrix contains {nan_count} NaN values"

    return True, "Candidate matrix is valid"
