"""
Column transformation functions
Transformations applied within each variable (along columns)
"""

import pandas as pd
import numpy as np
import streamlit as st


def detect_column_type(column, tolerance=1e-10):
    """
    Auto-detect if column is numeric or categorical and count levels.

    Parameters
    ----------
    column : pd.Series
        Column to analyze
    tolerance : float, default=1e-10
        Numerical tolerance (reserved for future use)

    Returns
    -------
    dict
        Dictionary containing:
        - is_numeric: bool, True if column is numeric
        - n_levels: int, count of unique non-null values
        - unique_values: list, sorted unique values (excluding NaN)
        - dtype_detected: str, 'numeric', 'binary_cat', or 'multiclass_cat'
        - can_encode_binary: bool, True if n_levels == 2
        - value_counts: dict, {value: frequency} for all non-null values

    Examples
    --------
    >>> # Binary categorical
    >>> col = pd.Series(['high', 'high', 'low', 'low'])
    >>> detect_column_type(col)
    {
        'is_numeric': False,
        'n_levels': 2,
        'unique_values': ['high', 'low'],
        'dtype_detected': 'binary_cat',
        'can_encode_binary': True,
        'value_counts': {'high': 2, 'low': 2}
    }

    >>> # Numeric 2-level
    >>> col = pd.Series([0, 10, 0, 10])
    >>> detect_column_type(col)
    {
        'is_numeric': True,
        'n_levels': 2,
        'unique_values': [0.0, 10.0],
        'dtype_detected': 'numeric',
        'can_encode_binary': True,
        'value_counts': {0.0: 2, 10.0: 2}
    }

    >>> # Multiclass categorical
    >>> col = pd.Series(['red', 'blue', 'green', 'red', 'blue', 'red'])
    >>> detect_column_type(col)
    {
        'is_numeric': False,
        'n_levels': 3,
        'unique_values': ['blue', 'green', 'red'],
        'dtype_detected': 'multiclass_cat',
        'can_encode_binary': False,
        'value_counts': {'red': 3, 'blue': 2, 'green': 1}
    }
    """
    # Remove NaN values for analysis
    col_clean = column.dropna()

    # Try to convert to numeric
    numeric_col = pd.to_numeric(col_clean, errors='coerce')

    # Check if all non-null values could be converted to numeric
    is_numeric = not numeric_col.isna().any()

    if is_numeric:
        # Use numeric values for analysis
        working_col = numeric_col
    else:
        # Use original values for categorical analysis
        working_col = col_clean

    # Get unique values and counts
    unique_values = sorted(working_col.unique())
    n_levels = len(unique_values)
    value_counts = working_col.value_counts().to_dict()

    # Determine data type
    if is_numeric:
        dtype_detected = 'numeric'
    elif n_levels == 2:
        dtype_detected = 'binary_cat'
    else:
        dtype_detected = 'multiclass_cat'

    # Binary encoding is possible if exactly 2 levels
    can_encode_binary = (n_levels == 2)

    return {
        'is_numeric': is_numeric,
        'n_levels': n_levels,
        'unique_values': unique_values,
        'dtype_detected': dtype_detected,
        'can_encode_binary': can_encode_binary,
        'value_counts': value_counts
    }


def column_auto_encode(data, col_range=None, exclude_cols=None):
    """
    Automatically encode all columns to [-1, +1] or one-hot encoding.

    Handles:
    - Numeric 2-level: maps to [-1, +1] range
    - Binary categorical: maps to [-1, +1] (sorted alphabetically)
    - Multiclass categorical (3+ levels): one-hot (k-1) encoding with auto reference level

    Parameters
    ----------
    data : pd.DataFrame
        Input dataframe to encode
    col_range : tuple (start, end), optional
        Column range to encode (start inclusive, end exclusive)
        If None, encodes all columns
    exclude_cols : list, optional
        List of column names to exclude from encoding

    Returns
    -------
    tuple
        (transformed_df, encoding_metadata)
        - transformed_df: DataFrame with encoded columns
        - encoding_metadata: dict with encoding information for each column

    Notes
    -----
    Multiclass encoding uses one-hot (k-1) with automatic reference level selection:
    - Reference level = most frequent category (implicit, all zeros)
    - Creates k-1 dummy columns for remaining categories
    - Original column is removed, replaced with dummy columns

    Examples
    --------
    >>> # Binary categorical
    >>> df = pd.DataFrame({'pouring_speed': ['high', 'high', 'low', 'low']})
    >>> transformed, meta = column_auto_encode(df)
    >>> transformed['pouring_speed'].tolist()
    [1.0, 1.0, -1.0, -1.0]

    >>> # Numeric 2-level
    >>> df = pd.DataFrame({'distance': [0, 10, 0, 10]})
    >>> transformed, meta = column_auto_encode(df)
    >>> transformed['distance'].tolist()
    [-1.0, 1.0, -1.0, 1.0]

    >>> # Multiclass categorical
    >>> df = pd.DataFrame({'color': ['red', 'blue', 'green', 'red', 'blue', 'red']})
    >>> transformed, meta = column_auto_encode(df)
    >>> list(transformed.columns)
    ['color_blue', 'color_green']
    >>> meta['color']['reference_level']
    'red'
    """
    # Initialize
    transformed_df = data.copy()
    encoding_metadata = {}

    if exclude_cols is None:
        exclude_cols = []

    # Determine column range
    if col_range is None:
        col_range = (0, len(data.columns))

    # Get columns to process
    cols_to_process = data.columns[col_range[0]:col_range[1]]
    cols_to_process = [col for col in cols_to_process if col not in exclude_cols]

    # Track columns to remove and add (for multiclass)
    columns_to_remove = []
    new_columns_data = {}

    # Process each column
    for col_name in cols_to_process:
        column = data[col_name]

        # Detect column type
        col_info = detect_column_type(column)

        dtype_detected = col_info['dtype_detected']
        n_levels = col_info['n_levels']
        unique_values = col_info['unique_values']
        value_counts = col_info['value_counts']

        # ═══════════════════════════════════════════════════════════════
        # CASE 1: Numeric with exactly 2 levels
        # ═══════════════════════════════════════════════════════════════
        if dtype_detected == 'numeric' and n_levels == 2:
            min_val = float(unique_values[0])
            max_val = float(unique_values[1])

            # Apply formula: 2*(x - min)/(max - min) - 1
            encoded = 2 * (column - min_val) / (max_val - min_val) - 1
            transformed_df[col_name] = encoded

            # Store metadata
            encoding_metadata[col_name] = {
                'type': 'numeric',
                'original_unique': unique_values,
                'n_levels': n_levels,
                'min_original': min_val,
                'max_original': max_val,
                'numeric_formula': f"2*(x - {min_val})/({max_val} - {min_val}) - 1",
                'encoding_rule': f"Numeric 2-level: [{min_val:.2f}, {max_val:.2f}] → [-1, +1]",
                'encoding_map': {min_val: -1.0, max_val: 1.0},
                'value_counts': value_counts
            }

        # ═══════════════════════════════════════════════════════════════
        # CASE 2: Binary categorical (2 levels, non-numeric)
        # ═══════════════════════════════════════════════════════════════
        elif dtype_detected == 'binary_cat':
            # Sort alphabetically: first → -1, second → +1
            sorted_values = sorted(unique_values)
            low_val = sorted_values[0]
            high_val = sorted_values[1]

            # Create encoding map
            encoding_map = {low_val: -1.0, high_val: 1.0}

            # Apply encoding
            encoded = column.map(encoding_map)
            transformed_df[col_name] = encoded

            # Store metadata
            encoding_metadata[col_name] = {
                'type': 'binary_cat',
                'original_unique': unique_values,
                'n_levels': n_levels,
                'encoding_rule': f"Binary categorical: {low_val}→-1, {high_val}→+1",
                'encoding_map': encoding_map,
                'value_counts': value_counts
            }

        # ═══════════════════════════════════════════════════════════════
        # CASE 3: Multiclass categorical (3+ levels)
        # ═══════════════════════════════════════════════════════════════
        elif dtype_detected == 'multiclass_cat':
            # STEP A: Auto-select reference level (highest frequency)
            reference_level = max(value_counts, key=value_counts.get)

            # STEP B: Create one-hot (k-1) encoding
            # Get non-reference levels, sorted alphabetically
            non_reference_levels = sorted([v for v in unique_values if v != reference_level])

            # Create dummy columns
            dummy_columns = []
            encoding_map = {reference_level: []}  # Will fill with zeros

            for level in non_reference_levels:
                dummy_col_name = f"{col_name}_{level}"
                dummy_columns.append(dummy_col_name)

                # Create dummy column: 1 where level matches, 0 elsewhere
                dummy_values = (column == level).astype(int)
                new_columns_data[dummy_col_name] = dummy_values

                # Update encoding map
                encoding_map[level] = []

            # Fill encoding map with actual encodings
            for level in unique_values:
                if level == reference_level:
                    encoding_map[level] = [0] * len(non_reference_levels)
                else:
                    # Create encoding: 1 at position of this level, 0 elsewhere
                    level_idx = non_reference_levels.index(level)
                    encoding = [0] * len(non_reference_levels)
                    encoding[level_idx] = 1
                    encoding_map[level] = encoding

            # Mark original column for removal
            columns_to_remove.append(col_name)

            # STEP C: Store metadata
            encoding_metadata[col_name] = {
                'type': 'multiclass_cat',
                'original_unique': unique_values,
                'n_levels': n_levels,
                'reference_level': reference_level,
                'encoding_rule': f"One-hot (k-1): reference='{reference_level}' (freq={value_counts[reference_level]}), creates {len(non_reference_levels)} columns",
                'dummy_columns': dummy_columns,
                'encoding_map': encoding_map,
                'value_counts': value_counts
            }

    # Apply multiclass transformations
    # Remove original multiclass columns
    if columns_to_remove:
        transformed_df = transformed_df.drop(columns=columns_to_remove)

    # Add new dummy columns
    if new_columns_data:
        for col_name, col_data in new_columns_data.items():
            transformed_df[col_name] = col_data

    return transformed_df, encoding_metadata


def column_doe_coding(data, col_range, reference_levels=None):
    """
    Automatic DoE (Design of Experiments) Coding - Unified Solution

    Intelligently encodes all column types:
    1. 2-level (numeric or categorical) → [-1, +1]
    2. 3+ levels numeric only → [-1, ..., +1] range
    3. 3+ levels categorical → Dummy coding (k-1) with manual reference level

    Parameters
    ----------
    data : pd.DataFrame
        Input dataframe
    col_range : tuple (start, end)
        Column range to encode
    reference_levels : dict, optional
        Manual reference levels for categorical multiclass
        Format: {'col_name': 'reference_value'}
        If None, uses highest frequency (auto-suggest)

    Returns
    -------
    tuple
        (transformed_df, encoding_metadata, multiclass_cols_dict)
        - transformed_df: DataFrame with encoded columns
        - encoding_metadata: Complete encoding information for each column
        - multiclass_cols_dict: Info about multiclass columns for UI suggestions

    Examples
    --------
    >>> # 2-level numeric
    >>> df = pd.DataFrame({'distance': [0, 10, 0, 10]})
    >>> result, meta, info = column_doe_coding(df, (0, 1))
    >>> result['distance'].tolist()
    [-1.0, 1.0, -1.0, 1.0]

    >>> # 2-level categorical
    >>> df = pd.DataFrame({'speed': ['high', 'low', 'high', 'low']})
    >>> result, meta, info = column_doe_coding(df, (0, 1))
    >>> result['speed'].tolist()
    [1.0, -1.0, 1.0, -1.0]

    >>> # 3+ level numeric
    >>> df = pd.DataFrame({'temp': [100, 150, 200, 100, 200]})
    >>> result, meta, info = column_doe_coding(df, (0, 1))
    >>> result['temp'].tolist()
    [-1.0, 0.0, 1.0, -1.0, 1.0]

    >>> # 3+ level categorical with manual reference
    >>> df = pd.DataFrame({'color': ['red', 'blue', 'green', 'red', 'blue', 'red']})
    >>> result, meta, info = column_doe_coding(df, (0, 1), reference_levels={'color': 'red'})
    >>> list(result.columns)
    ['color_blue', 'color_green']
    >>> meta['color']['reference_level']
    'red'
    """

    transformed_df = data.copy()
    encoding_metadata = {}
    multiclass_cols_info = {}  # Track multiclass for UI suggestion

    if reference_levels is None:
        reference_levels = {}

    # Get columns to process
    cols_to_process = data.columns[col_range[0]:col_range[1]]

    columns_to_remove = []
    new_columns_data = {}

    for col_name in cols_to_process:
        column = data[col_name]

        # Detect column type
        col_info = detect_column_type(column)
        dtype_detected = col_info['dtype_detected']
        n_levels = col_info['n_levels']
        unique_values = col_info['unique_values']
        value_counts = col_info['value_counts']
        is_numeric = col_info['is_numeric']

        # ═══════════════════════════════════════════════════════════════
        # CASE 1: 2-LEVEL NUMERIC → [-1, +1]
        # ═══════════════════════════════════════════════════════════════
        if is_numeric and n_levels == 2:
            min_val = float(unique_values[0])
            max_val = float(unique_values[1])

            # Apply formula: 2*(x - min)/(max - min) - 1
            if max_val != min_val:
                encoded = 2 * (column - min_val) / (max_val - min_val) - 1
            else:
                encoded = column.copy()

            transformed_df[col_name] = encoded

            encoding_metadata[col_name] = {
                'type': 'numeric_2level',
                'original_unique': unique_values,
                'n_levels': 2,
                'min': min_val,
                'max': max_val,
                'formula': f"2*(x - {min_val})/({max_val} - {min_val}) - 1",
                'encoding_rule': f"Numeric 2-level: [{min_val:.2f}, {max_val:.2f}] → [-1, +1]",
                'encoding_map': {min_val: -1.0, max_val: 1.0},
                'value_counts': value_counts
            }

        # ═══════════════════════════════════════════════════════════════
        # CASE 2: 2-LEVEL CATEGORICAL (alfanumerico) → [-1, +1]
        # ═══════════════════════════════════════════════════════════════
        elif not is_numeric and n_levels == 2:
            sorted_values = sorted(unique_values)
            low_val = sorted_values[0]
            high_val = sorted_values[1]

            # Create encoding map: alphabetical order
            encoding_map = {low_val: -1.0, high_val: 1.0}

            # Apply encoding
            encoded = column.map(encoding_map)
            transformed_df[col_name] = encoded

            encoding_metadata[col_name] = {
                'type': 'categorical_2level',
                'original_unique': unique_values,
                'n_levels': 2,
                'encoding_rule': f"Categorical 2-level: {low_val}→-1, {high_val}→+1",
                'encoding_map': encoding_map,
                'value_counts': value_counts
            }

        # ═══════════════════════════════════════════════════════════════
        # CASE 3: 3+ LEVELS NUMERIC ONLY → [-1, ..., +1] range
        # ═══════════════════════════════════════════════════════════════
        elif is_numeric and n_levels > 2:
            min_val = float(unique_values[0])
            max_val = float(unique_values[-1])

            # Apply formula: 2*(x - min)/(max - min) - 1
            if max_val != min_val:
                encoded = 2 * (column - min_val) / (max_val - min_val) - 1
            else:
                encoded = column.copy()

            transformed_df[col_name] = encoded

            encoding_metadata[col_name] = {
                'type': 'numeric_multiclass',
                'original_unique': unique_values,
                'n_levels': n_levels,
                'min': min_val,
                'max': max_val,
                'formula': f"2*(x - {min_val})/({max_val} - {min_val}) - 1",
                'encoding_rule': f"Numeric {n_levels}-level: [{min_val:.2f}, {max_val:.2f}] → [-1, ..., +1]",
                'value_counts': value_counts
            }

        # ═══════════════════════════════════════════════════════════════
        # CASE 4: 3+ LEVELS CATEGORICAL → DUMMY CODING (k-1) with reference
        # ═══════════════════════════════════════════════════════════════
        elif not is_numeric and n_levels > 2:
            # STEP A: Auto-suggest reference level (highest frequency)
            auto_suggested_ref = max(value_counts, key=value_counts.get)

            # STEP B: Use manual reference if provided, else use auto-suggest
            if col_name in reference_levels and reference_levels[col_name] in unique_values:
                reference_level = reference_levels[col_name]
            else:
                reference_level = auto_suggested_ref

            # Store info for UI suggestion
            multiclass_cols_info[col_name] = {
                'n_levels': n_levels,
                'unique_values': unique_values,
                'value_counts': value_counts,
                'auto_suggested_ref': auto_suggested_ref,
                'current_ref': reference_level
            }

            # STEP C: Create one-hot (k-1) dummy columns
            non_reference_levels = sorted([v for v in unique_values if v != reference_level])

            dummy_columns = []
            encoding_map = {reference_level: [0] * len(non_reference_levels)}

            for i, level in enumerate(non_reference_levels):
                dummy_col_name = f"{col_name}_{level}"
                dummy_columns.append(dummy_col_name)

                # Create dummy column: 1 where level matches, 0 elsewhere
                dummy_values = (column == level).astype(int)
                new_columns_data[dummy_col_name] = dummy_values

                # Create encoding pattern
                encoding_array = [0] * len(non_reference_levels)
                encoding_array[i] = 1
                encoding_map[level] = encoding_array

            # Mark original column for removal
            columns_to_remove.append(col_name)

            # Store metadata
            encoding_metadata[col_name] = {
                'type': 'categorical_multiclass',
                'original_unique': unique_values,
                'n_levels': n_levels,
                'reference_level': reference_level,
                'auto_suggested_ref': auto_suggested_ref,
                'encoding_rule': f"Dummy (k-1): ref='{reference_level}' (freq={value_counts[reference_level]}), {len(non_reference_levels)} cols",
                'dummy_columns': dummy_columns,
                'encoding_map': encoding_map,
                'value_counts': value_counts
            }

    # Apply multiclass transformations
    if columns_to_remove:
        transformed_df = transformed_df.drop(columns=columns_to_remove)

    if new_columns_data:
        for col_name, col_data in new_columns_data.items():
            transformed_df[col_name] = col_data

    return transformed_df, encoding_metadata, multiclass_cols_info


def column_centering(data, col_range):
    """Column centering (mean removal)"""
    M = data.iloc[:, col_range[0]:col_range[1]].copy()
    M_centered = M - M.mean(axis=0)
    return M_centered


def column_scaling(data, col_range):
    """Column scaling (unit variance)"""
    M = data.iloc[:, col_range[0]:col_range[1]].copy()
    M_scaled = M / M.std(axis=0, ddof=1)
    return M_scaled


def column_autoscale(data, col_range):
    """Column autoscaling (centering + scaling)"""
    M = data.iloc[:, col_range[0]:col_range[1]].copy()
    M_auto = (M - M.mean(axis=0)) / M.std(axis=0, ddof=1)
    return M_auto


def column_range_01(data, col_range):
    """Scale columns to [0,1] range"""
    M = data.iloc[:, col_range[0]:col_range[1]].copy()
    M_01 = (M - M.min(axis=0)) / (M.max(axis=0) - M.min(axis=0))
    return M_01


def column_range_11(data, col_range):
    """Scale columns to [-1,1] range (DoE coding)"""
    M = data.iloc[:, col_range[0]:col_range[1]].copy()
    M_11 = 2 * (M - M.min(axis=0)) / (M.max(axis=0) - M.min(axis=0)) - 1
    return M_11


def column_max100(data, col_range):
    """Scale column maximum to 100"""
    M = data.iloc[:, col_range[0]:col_range[1]].copy()
    M_max100 = (M / M.max(axis=0)) * 100
    return M_max100


def column_sum100(data, col_range):
    """Scale column sum to 100"""
    M = data.iloc[:, col_range[0]:col_range[1]].copy()
    M_sum100 = (M / M.sum(axis=0)) * 100
    return M_sum100


def column_length1(data, col_range):
    """Scale column length to 1"""
    M = data.iloc[:, col_range[0]:col_range[1]].copy()
    col_lengths = np.sqrt((M**2).sum(axis=0))
    M_l1 = M / col_lengths
    return M_l1


def column_log(data, col_range):
    """Log10 transformation with delta handling"""
    M = data.iloc[:, col_range[0]:col_range[1]].copy()

    if (M <= 0).any().any():
        min_val = M.min().min()
        delta = abs(min_val) + 1
        st.warning(f"Negative/zero values found. Adding delta: {delta}")
        M = M + delta

    M_log = np.log10(M)
    return M_log


def column_first_derivative(data, col_range):
    """First derivative by column"""
    M = data.iloc[:, col_range[0]:col_range[1]].copy()
    M_diff = M.diff(axis=0).iloc[1:, :]
    return M_diff


def column_second_derivative(data, col_range):
    """Second derivative by column"""
    M = data.iloc[:, col_range[0]:col_range[1]].copy()
    M_diff = M.diff(axis=0).diff(axis=0).iloc[2:, :]
    return M_diff


def moving_average_column(data, col_range, window):
    """Moving average by column"""
    M = data.iloc[:, col_range[0]:col_range[1]].copy()
    M_ma = M.rolling(window=window, axis=0, center=True).mean()
    return M_ma.dropna(axis=0)


def block_scaling(data, blocks_config):
    """Block scaling (autoscale + divide by sqrt(n_vars_in_block))"""
    transformed = data.copy()

    for block_name, col_range in blocks_config.items():
        M = data.iloc[:, col_range[0]:col_range[1]].copy()
        n_vars = M.shape[1]

        M_auto = (M - M.mean(axis=0)) / M.std(axis=0, ddof=1)
        M_block = M_auto / np.sqrt(n_vars)

        transformed.iloc[:, col_range[0]:col_range[1]] = M_block

    return transformed
