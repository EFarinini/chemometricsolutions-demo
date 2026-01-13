"""
Spectral Plot with GREEN SHADED BANDS
=====================================

Correct implementation based on:
- Leardi et al. Coffee_Barley paper (2012)
- gaplssp.m routine

Shows:
- Full spectrum in RED
- Selected BANDS in GREEN (shaded/hatched)
- BANDS are contiguous regions (not individual variables)

This is the CORRECT visualization!
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from typing import List, Optional, Tuple


def plot_spectrum_with_bands(
    X: np.ndarray,
    selected_indices: np.ndarray,
    column_names: List[str],
    wavelengths: Optional[np.ndarray] = None,
    title: str = "Spectral Data with Selected Bands",
    use_plotly: bool = True
) -> go.Figure:
    """
    Plot spectrum with selected BANDS highlighted in GREEN (shaded).

    CORRECT approach:
    - Identifies CONTIGUOUS BANDS of variables
    - Shades entire band regions in green
    - Shows on full spectrum

    Based on: Leardi et al. Coffee_Barley (2012), Fig. 4

    Parameters
    ----------
    X : np.ndarray
        Feature matrix (n_samples, n_variables)
    selected_indices : np.ndarray
        Indices of selected variables
    column_names : list
        Variable names
    wavelengths : np.ndarray, optional
        Wavelength values for x-axis
    title : str
        Plot title
    use_plotly : bool
        Use Plotly (interactive) or Matplotlib (publication-quality)

    Returns
    -------
    fig : plotly.graph_objects.Figure
        Plot with spectrum and shaded band regions
    """

    # Compute average spectrum
    avg_spectrum = np.mean(X, axis=0)
    n_vars = X.shape[1]

    # Create x-axis (wavelengths or indices)
    if wavelengths is None:
        x_axis = np.arange(n_vars)
        x_label = "Variable Index"
    else:
        x_axis = wavelengths
        x_label = "Wavenumber (cm⁻¹)"

    # Identify CONTIGUOUS BANDS
    bands = _identify_bands(selected_indices)

    # === PLOTLY VERSION (Interactive) ===
    fig = go.Figure()

    # Plot full spectrum in RED
    fig.add_trace(go.Scatter(
        x=x_axis,
        y=avg_spectrum,
        mode='lines',
        name='Spectrum',
        line=dict(color='red', width=2),
        fill='tozeroy',
        fillcolor='rgba(255, 100, 100, 0.2)',
        hovertemplate='<b>%{x}</b><br>Absorbance: %{y:.3f}<extra></extra>'
    ))

    # Add GREEN SHADED regions for each band
    for band_num, (start_idx, end_idx) in enumerate(bands):
        x_start = x_axis[start_idx]
        x_end = x_axis[end_idx]

        # Get spectrum values for this band
        band_spectrum = avg_spectrum[start_idx:end_idx+1]
        band_x = x_axis[start_idx:end_idx+1]

        # Add shaded band
        fig.add_trace(go.Scatter(
            x=np.concatenate([band_x, band_x[::-1]]),
            y=np.concatenate([band_spectrum, np.zeros_like(band_spectrum)]),
            fill='tozeroy',
            fillcolor='rgba(0, 200, 100, 0.5)',  # Green with transparency
            line=dict(color='rgba(0, 200, 100, 0)'),  # No border line
            name=f'Band {band_num+1}' if band_num == 0 else '',
            hovertemplate=f'<b>Band {band_num+1}</b><br>x: %{{x}}<extra></extra>',
            showlegend=(band_num == 0)  # Only show "Band" once in legend
        ))

        # Add vertical lines at band boundaries
        fig.add_vline(
            x=x_start, line_dash='dash', line_color='green', opacity=0.5
        )
        fig.add_vline(
            x=x_end, line_dash='dash', line_color='green', opacity=0.5
        )

    fig.update_layout(
        title={'text': title, 'x': 0.5, 'xanchor': 'center'},
        xaxis_title=x_label,
        yaxis_title='Absorbance',
        hovermode='x unified',
        template='plotly_white',
        height=600,
        legend=dict(x=0.02, y=0.98),
        margin=dict(l=80, r=50, t=80, b=80)
    )

    return fig


def _identify_bands(selected_indices: np.ndarray) -> List[Tuple[int, int]]:
    """
    Identify CONTIGUOUS BANDS from selected variable indices.

    A band is a contiguous group of selected variables.

    Parameters
    ----------
    selected_indices : np.ndarray
        Indices of selected variables

    Returns
    -------
    bands : list of (start_idx, end_idx) tuples
        Start and end indices of each band

    Example
    -------
    >>> selected = np.array([1, 2, 3, 8, 9, 15])
    >>> bands = _identify_bands(selected)
    >>> bands
    [(1, 3), (8, 9), (15, 15)]
    # Three bands:
    # Band 1: variables 1-3
    # Band 2: variables 8-9
    # Band 3: variable 15
    """
    if len(selected_indices) == 0:
        return []

    sorted_indices = np.sort(selected_indices)
    bands = []

    band_start = sorted_indices[0]
    band_end = sorted_indices[0]

    for i in range(1, len(sorted_indices)):
        if sorted_indices[i] == band_end + 1:
            # Continue current band
            band_end = sorted_indices[i]
        else:
            # Save current band and start new one
            bands.append((band_start, band_end))
            band_start = sorted_indices[i]
            band_end = sorted_indices[i]

    # Add last band
    bands.append((band_start, band_end))

    return bands


def create_bands_table(
    selected_indices: np.ndarray,
    column_names: List[str],
    wavelengths: Optional[np.ndarray] = None
) -> pd.DataFrame:
    """
    Create summary table of selected BANDS.

    Parameters
    ----------
    selected_indices : np.ndarray
        Indices of selected variables
    column_names : list
        Variable names
    wavelengths : np.ndarray, optional
        Wavelength values

    Returns
    -------
    df : pd.DataFrame
        Summary table with band information
    """

    bands = _identify_bands(selected_indices)

    summary_data = []

    for band_num, (start_idx, end_idx) in enumerate(bands):
        n_vars = end_idx - start_idx + 1

        if wavelengths is not None and len(wavelengths) > end_idx:
            wl_start = wavelengths[start_idx]
            wl_end = wavelengths[end_idx]
            wl_range = f"{wl_start:.1f}-{wl_end:.1f}"
        else:
            wl_range = f"{start_idx}-{end_idx}"

        summary_data.append({
            'Band': band_num + 1,
            'Start_Index': start_idx,
            'End_Index': end_idx,
            'N_Variables': n_vars,
            'Wavelength_Range': wl_range
        })

    return pd.DataFrame(summary_data)


def plot_multiple_runs_with_bands(
    X: np.ndarray,
    run_results: List[np.ndarray],
    run_labels: List[str],
    wavelengths: Optional[np.ndarray] = None,
    title: str = "Spectral Data: Multiple GA Runs with Selected Bands"
) -> go.Figure:
    """
    Plot multiple GA runs showing BANDS selected by each run.

    Each run shows which BANDS it selected.
    Bands that appear in MULTIPLE RUNS are more robust!

    Parameters
    ----------
    X : np.ndarray
        Feature matrix
    run_results : list of np.ndarray
        Selected indices from each GA run
    run_labels : list
        Labels for each run
    wavelengths : np.ndarray, optional
        Wavelength values
    title : str
        Plot title

    Returns
    -------
    fig : plotly.graph_objects.Figure
        Comparison plot
    """

    avg_spectrum = np.mean(X, axis=0)
    n_vars = X.shape[1]

    if wavelengths is None:
        x_axis = np.arange(n_vars)
        x_label = "Variable Index"
    else:
        x_axis = wavelengths
        x_label = "Wavenumber (cm⁻¹)"

    fig = go.Figure()

    # Plot spectrum
    fig.add_trace(go.Scatter(
        x=x_axis,
        y=avg_spectrum,
        mode='lines',
        name='Spectrum',
        line=dict(color='red', width=2),
        fill='tozeroy',
        fillcolor='rgba(255, 100, 100, 0.2)',
        hovertemplate='<b>%{x}</b><br>Absorbance: %{y:.3f}<extra></extra>'
    ))

    # Colors for different runs
    colors = [
        'rgba(0, 200, 100, 0.5)',
        'rgba(0, 100, 200, 0.5)',
        'rgba(200, 100, 0, 0.5)',
        'rgba(200, 0, 100, 0.5)',
        'rgba(100, 0, 200, 0.5)'
    ]

    # Add bands for each run
    for run_idx, (selected_indices, run_label) in enumerate(zip(run_results, run_labels)):
        bands = _identify_bands(selected_indices)

        color = colors[run_idx % len(colors)]

        for band_idx, (start_idx, end_idx) in enumerate(bands):
            x_start = x_axis[start_idx]
            x_end = x_axis[end_idx]

            band_spectrum = avg_spectrum[start_idx:end_idx+1]
            band_x = x_axis[start_idx:end_idx+1]

            fig.add_trace(go.Scatter(
                x=np.concatenate([band_x, band_x[::-1]]),
                y=np.concatenate([band_spectrum, np.zeros_like(band_spectrum)]),
                fill='tozeroy',
                fillcolor=color,
                line=dict(color='rgba(0,0,0,0)'),
                name=run_label,
                showlegend=(band_idx == 0),  # Show label once per run
                hovertemplate=f'<b>{run_label}</b><br>Band<extra></extra>'
            ))

    fig.update_layout(
        title={'text': title, 'x': 0.5, 'xanchor': 'center'},
        xaxis_title=x_label,
        yaxis_title='Absorbance',
        hovermode='x unified',
        template='plotly_white',
        height=600,
        legend=dict(x=0.02, y=0.98),
        margin=dict(l=80, r=50, t=80, b=80)
    )

    return fig
