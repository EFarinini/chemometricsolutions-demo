"""
Column transformation functions
Transformations applied within each variable (along columns)
"""

import pandas as pd
import numpy as np
import streamlit as st


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
