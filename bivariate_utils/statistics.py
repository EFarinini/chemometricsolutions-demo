"""
Bivariate Statistical Analysis
Correlation, covariance, and statistical measures for bivariate analysis
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Tuple, Dict, Optional
import streamlit as st


@st.cache_data
def compute_correlation_matrix(
    data: pd.DataFrame,
    method: str = 'pearson'
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute correlation matrix with p-values

    Parameters
    ----------
    data : pd.DataFrame
        Input data with numeric variables
    method : str
        Correlation method: 'pearson', 'spearman', or 'kendall'

    Returns
    -------
    tuple
        (correlation_matrix, pvalue_matrix)
    """
    # Remove rows with NaN
    data_clean = data.dropna()

    if len(data_clean) < 2:
        raise ValueError("Need at least 2 complete observations for correlation")

    n_vars = data_clean.shape[1]

    # Initialize matrices
    corr_matrix = pd.DataFrame(
        np.zeros((n_vars, n_vars)),
        index=data_clean.columns,
        columns=data_clean.columns
    )
    pval_matrix = pd.DataFrame(
        np.zeros((n_vars, n_vars)),
        index=data_clean.columns,
        columns=data_clean.columns
    )

    # Compute correlations
    for i, col_i in enumerate(data_clean.columns):
        for j, col_j in enumerate(data_clean.columns):
            if i == j:
                corr_matrix.iloc[i, j] = 1.0
                pval_matrix.iloc[i, j] = 0.0
            else:
                if method == 'pearson':
                    corr, pval = stats.pearsonr(data_clean[col_i], data_clean[col_j])
                elif method == 'spearman':
                    corr, pval = stats.spearmanr(data_clean[col_i], data_clean[col_j])
                elif method == 'kendall':
                    corr, pval = stats.kendalltau(data_clean[col_i], data_clean[col_j])
                else:
                    raise ValueError(f"Unknown correlation method: {method}")

                corr_matrix.iloc[i, j] = corr
                pval_matrix.iloc[i, j] = pval

    return corr_matrix, pval_matrix


@st.cache_data
def compute_covariance_matrix(data: pd.DataFrame) -> pd.DataFrame:
    """
    Compute covariance matrix

    Parameters
    ----------
    data : pd.DataFrame
        Input data with numeric variables

    Returns
    -------
    pd.DataFrame
        Covariance matrix
    """
    # Remove rows with NaN
    data_clean = data.dropna()

    if len(data_clean) < 2:
        raise ValueError("Need at least 2 complete observations for covariance")

    # Compute covariance
    cov_matrix = data_clean.cov()

    return cov_matrix


def compute_spearman_correlation(
    data: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute Spearman rank correlation matrix with p-values

    Parameters
    ----------
    data : pd.DataFrame
        Input data with numeric variables

    Returns
    -------
    tuple
        (correlation_matrix, pvalue_matrix)
    """
    return compute_correlation_matrix(data, method='spearman')


def compute_kendall_correlation(
    data: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute Kendall's tau correlation matrix with p-values

    Parameters
    ----------
    data : pd.DataFrame
        Input data with numeric variables

    Returns
    -------
    tuple
        (correlation_matrix, pvalue_matrix)
    """
    return compute_correlation_matrix(data, method='kendall')


@st.cache_data
def get_correlation_summary(
    corr_matrix: pd.DataFrame,
    pval_matrix: pd.DataFrame,
    threshold: float = 0.05
) -> pd.DataFrame:
    """
    Get summary of significant correlations

    Parameters
    ----------
    corr_matrix : pd.DataFrame
        Correlation matrix
    pval_matrix : pd.DataFrame
        P-value matrix
    threshold : float
        Significance threshold (default 0.05)

    Returns
    -------
    pd.DataFrame
        Summary table with significant correlations
    """
    results = []

    n_vars = len(corr_matrix)
    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            var1 = corr_matrix.index[i]
            var2 = corr_matrix.columns[j]
            corr = corr_matrix.iloc[i, j]
            pval = pval_matrix.iloc[i, j]

            results.append({
                'Variable 1': var1,
                'Variable 2': var2,
                'Correlation': corr,
                'P-value': pval,
                'Significant': 'Yes' if pval < threshold else 'No'
            })

    summary_df = pd.DataFrame(results)
    summary_df = summary_df.sort_values('Correlation', key=abs, ascending=False)

    return summary_df


def compute_robust_correlation(
    data: pd.DataFrame,
    method: str = 'mcd'
) -> Tuple[pd.DataFrame, Dict]:
    """
    Compute robust correlation using Minimum Covariance Determinant (MCD)

    Parameters
    ----------
    data : pd.DataFrame
        Input data with numeric variables
    method : str
        Robust method ('mcd' for Minimum Covariance Determinant)

    Returns
    -------
    tuple
        (robust_correlation_matrix, diagnostics_dict)
    """
    from sklearn.covariance import MinCovDet

    # Remove rows with NaN
    data_clean = data.dropna()

    if len(data_clean) < data.shape[1] + 1:
        raise ValueError("Need more observations than variables for robust correlation")

    # Compute MCD
    mcd = MinCovDet(random_state=42)
    mcd.fit(data_clean)

    # Get robust covariance
    robust_cov = pd.DataFrame(
        mcd.covariance_,
        index=data_clean.columns,
        columns=data_clean.columns
    )

    # Convert to correlation
    std_devs = np.sqrt(np.diag(robust_cov))
    robust_corr = robust_cov / np.outer(std_devs, std_devs)

    # Diagnostics
    diagnostics = {
        'location': pd.Series(mcd.location_, index=data_clean.columns),
        'support': mcd.support_,
        'mahalanobis_distances': mcd.dist_,
        'n_outliers': len(data_clean) - mcd.support_.sum()
    }

    return robust_corr, diagnostics
