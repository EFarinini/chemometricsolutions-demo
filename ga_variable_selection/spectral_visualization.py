"""
Spectral Plot Visualization for GA Variable Selection
======================================================

Displays the original spectrum with selected wavelength regions highlighted.
Useful for FT-IR, NIR, Raman, and other spectroscopic data.

Author: ChemometricSolutions
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from typing import List, Optional, Tuple
import warnings


def plot_spectral_regions(
    X: np.ndarray,
    selected_indices: np.ndarray,
    column_names: List[str],
    wavelengths: Optional[np.ndarray] = None,
    avg_spectrum: Optional[np.ndarray] = None,
    title: str = "Spectral Data with Selected Regions Highlighted"
) -> go.Figure:
    """
    Plot spectrum with selected wavelength regions highlighted.

    This is particularly useful for spectroscopic data (FT-IR, NIR, Raman, etc.)
    where visualization of selected wavelength regions is important for
    interpretation.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix (n_samples, n_variables)
    selected_indices : np.ndarray
        Indices of selected variables
    column_names : list
        Names/labels of variables (e.g., wavelengths as strings)
    wavelengths : np.ndarray, optional
        Actual wavelength values for x-axis
        If None, uses variable indices
    avg_spectrum : np.ndarray, optional
        Pre-computed average spectrum
        If None, computes from X (mean across samples)
    title : str
        Plot title

    Returns
    -------
    fig : plotly.graph_objects.Figure
        Interactive spectral plot with highlighted regions
    """

    # Compute average spectrum if not provided
    if avg_spectrum is None:
        avg_spectrum = np.mean(X, axis=0)

    n_vars = X.shape[1]

    # Create x-axis (wavelengths or indices)
    if wavelengths is None:
        x_axis = np.arange(n_vars)
        x_label = "Variable Index"
    else:
        if len(wavelengths) != n_vars:
            warnings.warn(f"Wavelength array length ({len(wavelengths)}) != n_variables ({n_vars})")
            x_axis = np.arange(n_vars)
            x_label = "Variable Index"
        else:
            x_axis = wavelengths
            x_label = "Wavelength (cm⁻¹)" if "cm" in str(wavelengths[0]) else "Wavelength"

    # Create figure
    fig = go.Figure()

    # Add full spectrum
    fig.add_trace(go.Scatter(
        x=x_axis,
        y=avg_spectrum,
        mode='lines',
        name='Full Spectrum',
        line=dict(color='red', width=2),
        hovertemplate='<b>%{x}</b><br>Intensity: %{y:.4f}<extra></extra>',
        fill='tozeroy',
        fillcolor='rgba(255, 0, 0, 0.3)'
    ))

    # Identify selected regions (contiguous groups)
    selected_regions = _identify_contiguous_regions(selected_indices)

    # Add highlighted regions as shaded areas
    for region_start, region_end in selected_regions:
        # Get x and y values for this region
        region_indices = np.arange(region_start, region_end + 1)
        region_x = x_axis[region_indices]
        region_y = avg_spectrum[region_indices]

        # Add shaded region
        fig.add_trace(go.Scatter(
            x=np.concatenate([region_x, region_x[::-1]]),
            y=np.concatenate([region_y, np.zeros_like(region_y)]),
            fill='tozeroy',
            fillcolor='rgba(0, 204, 150, 0.6)',  # Green with transparency
            line=dict(color='rgba(0, 204, 150, 0)'),  # Transparent line
            name='Selected Region',
            hoverinfo='skip',
            showlegend=(region_start == selected_regions[0][0])  # Only show once in legend
        ))

        # Add boundary lines for clarity
        fig.add_vline(
            x=region_x[0],
            line_dash='dash',
            line_color='green',
            opacity=0.7,
            annotation_text='',
            showlegend=False
        )
        fig.add_vline(
            x=region_x[-1],
            line_dash='dash',
            line_color='green',
            opacity=0.7,
            annotation_text='',
            showlegend=False
        )

    # Update layout
    fig.update_layout(
        title={
            'text': title,
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 14}
        },
        xaxis_title=x_label,
        yaxis_title='Intensity / Absorbance',
        hovermode='x unified',
        template='plotly_white',
        height=500,
        showlegend=True,
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='black',
            borderwidth=1
        )
    )

    return fig


def plot_spectral_regions_interactive(
    X: np.ndarray,
    selected_indices: np.ndarray,
    column_names: List[str],
    wavelengths: Optional[np.ndarray] = None,
    problem_type: str = 'pls',
    target_var_name: str = ''
) -> go.Figure:
    """
    Create interactive spectral plot with detailed information.

    Includes:
    - Original spectrum (red fill)
    - Selected regions highlighted (green hatching)
    - Region boundaries marked
    - Statistical information in hover text
    - Comparison to unselected regions

    Parameters
    ----------
    X : np.ndarray
        Feature matrix
    selected_indices : np.ndarray
        Indices of selected variables
    column_names : list
        Variable names
    wavelengths : np.ndarray, optional
        Wavelength values
    problem_type : str
        Type of problem (for labeling)
    target_var_name : str
        Name of target variable (for title)

    Returns
    -------
    fig : plotly.graph_objects.Figure
        Advanced interactive spectral plot
    """

    n_vars = X.shape[1]
    avg_spectrum = np.mean(X, axis=0)
    std_spectrum = np.std(X, axis=0)

    # Create x-axis
    if wavelengths is None:
        x_axis = np.arange(n_vars)
        x_label = "Variable Index"
    else:
        x_axis = wavelengths
        x_label = "Wavelength (cm⁻¹)"

    # Create figure
    fig = go.Figure()

    # Add spectrum with confidence band
    fig.add_trace(go.Scatter(
        x=x_axis,
        y=avg_spectrum + std_spectrum,
        fill=None,
        mode='lines',
        line_color='rgba(0,0,0,0)',
        showlegend=False,
        hoverinfo='skip',
        name='Upper Bound'
    ))

    fig.add_trace(go.Scatter(
        x=x_axis,
        y=avg_spectrum - std_spectrum,
        fill='tonexty',
        mode='lines',
        line_color='rgba(0,0,0,0)',
        name='±1 Std Dev',
        fillcolor='rgba(255, 200, 124, 0.3)',
        hoverinfo='skip'
    ))

    # Add mean spectrum
    fig.add_trace(go.Scatter(
        x=x_axis,
        y=avg_spectrum,
        mode='lines',
        name='Mean Spectrum',
        line=dict(color='red', width=2.5),
        fill='tozeroy',
        fillcolor='rgba(255, 0, 0, 0.2)',
        hovertemplate='<b>%{x}</b><br>Intensity: %{y:.4f}<extra></extra>'
    ))

    # Identify and highlight selected regions
    selected_regions = _identify_contiguous_regions(selected_indices)

    colors = ['rgb(0, 204, 150)', 'rgb(0, 150, 204)', 'rgb(150, 0, 204)', 'rgb(204, 150, 0)']

    for idx, (region_start, region_end) in enumerate(selected_regions):
        region_indices = np.arange(region_start, region_end + 1)
        region_x = x_axis[region_indices]
        region_y = avg_spectrum[region_indices]

        # Determine wavelength range label
        if isinstance(region_x[0], (int, np.integer)):
            region_label = f"Region {idx+1}: vars {region_start}-{region_end}"
        else:
            region_label = f"Region {idx+1}: {region_x[0]:.1f}-{region_x[-1]:.1f} cm⁻¹"

        # Add highlighted region
        fig.add_trace(go.Scatter(
            x=np.concatenate([region_x, region_x[::-1]]),
            y=np.concatenate([region_y, region_y[::-1]]),
            fill='toself',
            fillcolor=colors[idx % len(colors)],
            opacity=0.5,
            line=dict(color='rgba(0,0,0,0)'),
            name=region_label,
            hovertemplate='<b>%{x}</b><br>Intensity: %{y:.4f}<extra></extra>',
            showlegend=True
        ))

    # Add title
    title_text = f"Spectral Data: {problem_type.upper()}"
    if target_var_name:
        title_text += f" (Target: {target_var_name})"

    # Update layout
    fig.update_layout(
        title={
            'text': title_text,
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis_title=x_label,
        yaxis_title='Intensity / Absorbance',
        hovermode='x unified',
        template='plotly_white',
        height=600,
        showlegend=True,
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor='rgba(255, 255, 255, 0.9)',
            bordercolor='black',
            borderwidth=1,
            font=dict(size=10)
        ),
        margin=dict(l=80, r=50, t=80, b=80)
    )

    return fig


def _identify_contiguous_regions(selected_indices: np.ndarray) -> List[Tuple[int, int]]:
    """
    Identify contiguous groups of selected variable indices.

    Parameters
    ----------
    selected_indices : np.ndarray
        Indices of selected variables

    Returns
    -------
    regions : list of tuples
        List of (start_idx, end_idx) tuples for each contiguous region
    """
    if len(selected_indices) == 0:
        return []

    # Sort indices
    sorted_indices = np.sort(selected_indices)

    # Find gaps
    regions = []
    region_start = sorted_indices[0]
    region_end = sorted_indices[0]

    for i in range(1, len(sorted_indices)):
        if sorted_indices[i] == region_end + 1:
            # Continue current region
            region_end = sorted_indices[i]
        else:
            # Start new region
            regions.append((region_start, region_end))
            region_start = sorted_indices[i]
            region_end = sorted_indices[i]

    # Add last region
    regions.append((region_start, region_end))

    return regions


def create_spectral_summary(
    selected_indices: np.ndarray,
    column_names: List[str],
    wavelengths: Optional[np.ndarray] = None
) -> pd.DataFrame:
    """
    Create summary of selected spectral regions.

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
        Summary table with region information
    """

    regions = _identify_contiguous_regions(selected_indices)

    summary_data = []

    for idx, (start, end) in enumerate(regions):
        n_vars = end - start + 1

        if wavelengths is not None and len(wavelengths) > end:
            wl_start = wavelengths[start]
            wl_end = wavelengths[end]
            wl_range = f"{wl_start:.2f}-{wl_end:.2f}"
        else:
            wl_range = f"{start}-{end}"

        var_names = [column_names[i] for i in range(start, end + 1)]

        summary_data.append({
            'Region': idx + 1,
            'Start_Index': start,
            'End_Index': end,
            'N_Variables': n_vars,
            'Wavelength_Range': wl_range,
            'Variables': ', '.join(var_names) if n_vars <= 5 else f"{var_names[0]}, ..., {var_names[-1]}"
        })

    return pd.DataFrame(summary_data)


def plot_spectral_leardi_style(
    X: np.ndarray,
    selected_indices: np.ndarray,
    column_names: List[str],
    wavelengths: Optional[np.ndarray] = None,
    title: str = "Spectral Data - Selected Regions (Leardi Style)"
) -> go.Figure:
    """
    Plot spectrum with selected regions as horizontal lines below (Leardi's plotmore style).

    This is CLEARER than overlapping colored regions!
    Each selected region appears as a horizontal black line below the spectrum.

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

    Returns
    -------
    fig : plotly.graph_objects.Figure
        Interactive plot matching Leardi's plotmore visualization
    """

    # Compute average spectrum (like Leardi's plot(dataset','r'))
    avg_spectrum = np.mean(X, axis=0)
    n_vars = X.shape[1]

    # Create x-axis
    if wavelengths is None:
        x_axis = np.arange(n_vars)
        x_label = "Variable Index"
    else:
        x_axis = wavelengths
        x_label = "Wavelength (cm⁻¹)"

    # Get min/max for scaling (like Leardi's mi, ma, ra)
    mi = np.min(avg_spectrum)
    ma = np.max(avg_spectrum)
    ra = ma - mi

    # Create figure
    fig = go.Figure()

    # Plot spectrum in RED (like Leardi: plot(dataset','r'))
    fig.add_trace(go.Scatter(
        x=x_axis,
        y=avg_spectrum,
        mode='lines',
        name='Mean Spectrum',
        line=dict(color='red', width=2),
        fill='tozeroy',
        fillcolor='rgba(255, 100, 100, 0.3)',
        hovertemplate='<b>%{x}</b><br>Intensity: %{y:.4f}<extra></extra>'
    ))

    # Identify contiguous regions (auto-detect instead of manual input!)
    selected_regions = _identify_contiguous_regions(selected_indices)

    # Add horizontal lines for each region (like Leardi's plot([tot(i,ii)-.5 tot(i,ii)+.5],[lev lev],'k'))
    for region_num, (region_start, region_end) in enumerate(selected_regions):
        # Y position: below spectrum (like Leardi's lev = mi - (ra/20) * i)
        y_pos = mi - (ra / 20) * (region_num + 1)

        # X coordinates for this region
        region_x_start = x_axis[region_start]
        region_x_end = x_axis[region_end]

        # Region label
        if isinstance(region_x_start, (int, np.integer)):
            region_label = f"Region {region_num+1}: vars {region_start}-{region_end}"
        else:
            region_label = f"Region {region_num+1}: {region_x_start:.1f}-{region_x_end:.1f} cm⁻¹"

        # Add horizontal line (BLACK, like Leardi)
        fig.add_trace(go.Scatter(
            x=[region_x_start, region_x_end],
            y=[y_pos, y_pos],
            mode='lines',
            name=region_label,
            line=dict(color='black', width=3),
            hovertemplate=f'<b>{region_label}</b><br>x: %{{x}}<extra></extra>',
            showlegend=True
        ))

        # Add region number label
        fig.add_annotation(
            x=(region_x_start + region_x_end) / 2,
            y=y_pos,
            text=f"<b>{region_num + 1}</b>",
            showarrow=False,
            yshift=-15,
            font=dict(size=11, color='black')
        )

    # Set y-axis limits to show all regions (like Leardi's final YLim)
    y_min = mi - (ra / 20) * (len(selected_regions) + 1)
    y_max = ma + ra / 20

    # Update layout
    fig.update_layout(
        title={
            'text': title,
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 16}
        },
        xaxis_title=x_label,
        yaxis_title='Intensity / Absorbance',
        hovermode='closest',
        template='plotly_white',
        height=600,
        showlegend=True,
        yaxis=dict(
            range=[y_min, y_max]
        ),
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor='rgba(255, 255, 255, 0.9)',
            bordercolor='black',
            borderwidth=1,
            font=dict(size=9)
        ),
        margin=dict(l=80, r=50, t=100, b=80)
    )

    return fig


def plot_spectral_comparison(
    X_original: np.ndarray,
    X_selected: np.ndarray,
    wavelengths: Optional[np.ndarray] = None
) -> go.Figure:
    """
    Compare full spectrum vs selected variables subset.

    Parameters
    ----------
    X_original : np.ndarray
        Original feature matrix
    X_selected : np.ndarray
        Subset with only selected variables
    wavelengths : np.ndarray, optional
        Wavelength values for original spectrum

    Returns
    -------
    fig : plotly.graph_objects.Figure
        Comparison plot
    """

    avg_original = np.mean(X_original, axis=0)
    avg_selected_indices = np.where(np.sum(X_selected, axis=0) > 0)[0]

    if wavelengths is None:
        x_axis = np.arange(len(avg_original))
        x_label = "Variable Index"
    else:
        x_axis = wavelengths
        x_label = "Wavelength (cm⁻¹)"

    fig = go.Figure()

    # Original spectrum
    fig.add_trace(go.Scatter(
        x=x_axis,
        y=avg_original,
        mode='lines',
        name='Original Spectrum',
        line=dict(color='blue', width=2),
        opacity=0.7,
        hovertemplate='<b>%{x}</b><br>Intensity: %{y:.4f}<extra></extra>'
    ))

    # Highlight selected regions
    selected_regions = _identify_contiguous_regions(avg_selected_indices)

    for start, end in selected_regions:
        region_indices = np.arange(start, end + 1)
        region_x = x_axis[region_indices]
        region_y = avg_original[region_indices]

        fig.add_trace(go.Scatter(
            x=region_x,
            y=region_y,
            mode='lines',
            name='Selected Region' if start == selected_regions[0][0] else '',
            line=dict(color='green', width=4),
            hovertemplate='<b>%{x}</b><br>Intensity: %{y:.4f}<extra></extra>',
            showlegend=(start == selected_regions[0][0])
        ))

    fig.update_layout(
        title={
            'text': 'Spectral Regions: Original vs Selected',
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis_title=x_label,
        yaxis_title='Intensity / Absorbance',
        hovermode='x unified',
        template='plotly_white',
        height=500
    )

    return fig


def extract_wavelengths_from_columns(column_names: List[str]) -> Optional[np.ndarray]:
    """
    Attempt to extract wavelength values from column names.

    Tries various common formats:
    - Pure numbers: "1234.5", "1000"
    - With units: "1234.5 cm-1", "450 nm"
    - With prefixes: "X1234.5", "wl_1000"

    Parameters
    ----------
    column_names : list
        Column names that might contain wavelength information

    Returns
    -------
    wavelengths : np.ndarray or None
        Array of wavelength values if successfully extracted, None otherwise
    """
    wavelengths = []

    for name in column_names:
        try:
            # Try direct float conversion
            wl = float(name)
            wavelengths.append(wl)
        except ValueError:
            # Try removing common prefixes/suffixes
            import re

            # Pattern to extract numbers (including decimals)
            match = re.search(r'([\d.]+)', str(name))
            if match:
                try:
                    wl = float(match.group(1))
                    wavelengths.append(wl)
                except ValueError:
                    return None
            else:
                return None

    if len(wavelengths) == len(column_names):
        return np.array(wavelengths)
    else:
        return None


def is_spectroscopic_data(n_variables: int, column_names: List[str]) -> bool:
    """
    Heuristic to determine if data is likely spectroscopic.

    Parameters
    ----------
    n_variables : int
        Number of variables
    column_names : list
        Variable names

    Returns
    -------
    is_spectroscopic : bool
        True if data appears to be spectroscopic
    """
    # Check 1: Many variables (typical for spectra)
    if n_variables < 50:
        return False

    # Check 2: Column names are numeric or contain wavelength info
    try:
        wavelengths = extract_wavelengths_from_columns(column_names)
        if wavelengths is not None:
            # Check if wavelengths are monotonic (typical for spectra)
            diffs = np.diff(wavelengths)
            if np.all(diffs > 0) or np.all(diffs < 0):
                return True
    except:
        pass

    # Check 3: Many variables and numeric column names
    if n_variables > 100:
        numeric_count = 0
        for name in column_names[:10]:  # Check first 10
            try:
                float(name)
                numeric_count += 1
            except:
                pass

        if numeric_count >= 8:  # At least 80% numeric
            return True

    return False
