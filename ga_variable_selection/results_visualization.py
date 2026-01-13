"""
Results Visualization for GA Variable Selection
==============================================

Plotly-based interactive visualizations for genetic algorithm results.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Dict
from io import BytesIO


def plot_selection_frequency(
    selection_freq: np.ndarray,
    selected_indices: np.ndarray,
    column_names: List[str],
    top_n: int = 50
) -> go.Figure:
    """
    Plot variable selection frequency as bar chart in ORIGINAL variable order.

    This preserves spatial structure (e.g., spectral bands are contiguous).

    Parameters
    ----------
    selection_freq : np.ndarray
        Selection frequency for each variable
    selected_indices : np.ndarray
        Indices of variables in final model
    column_names : list
        Variable names
    top_n : int
        Unused (kept for compatibility). Shows all variables with freq > 0.

    Returns
    -------
    fig : plotly.graph_objects.Figure
        Interactive bar chart in original variable order
    """
    # DON'T SORT - keep original order!
    all_indices = np.arange(len(selection_freq))

    df = pd.DataFrame({
        'Variable': [column_names[i] for i in all_indices],
        'Selection_Frequency': selection_freq,
        'Selected': ['Final Model' if i in selected_indices else 'Not Selected'
                     for i in all_indices]
    })

    # Only show variables with non-zero frequency (optional: cleaner display)
    df = df[df['Selection_Frequency'] > 0]

    # Color mapping
    colors = {
        'Final Model': '#00CC96',  # Green
        'Not Selected': '#636EFA'   # Blue
    }

    fig = go.Figure()

    for status in ['Final Model', 'Not Selected']:
        mask = df['Selected'] == status
        fig.add_trace(go.Bar(
            x=df[mask]['Variable'],
            y=df[mask]['Selection_Frequency'],
            name=status,
            marker_color=colors[status],
            hovertemplate='<b>%{x}</b><br>' +
                         'Frequency: %{y:.1f}<br>' +
                         '<extra></extra>'
        ))

    fig.update_layout(
        title={
            'text': 'Variable Selection Frequency (Original Order)',
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis_title='Variable (Original Order)',
        yaxis_title='Selection Frequency',
        hovermode='closest',
        template='plotly_white',
        showlegend=True,
        height=500,
        barmode='group'  # Side-by-side for different groups
    )

    # Always rotate x-axis labels for better readability
    fig.update_xaxes(tickangle=-45)

    return fig


def plot_selection_frequency_spectral_order(
    consensus_frequency: np.ndarray,
    final_selected_variables: List[int],
    feature_names: List[str] = None,
    title: str = "Variable Selection Frequency (Spectral Order)"
) -> go.Figure:
    """
    Plot variable selection frequency for ALL variables in original spectral order.

    Shows which spectral regions contain selected vs not-selected variables.
    Displays a mixed/scattered pattern across the spectrum.

    Parameters
    ----------
    consensus_frequency : np.ndarray
        Selection frequency for each variable (0-5)
    final_selected_variables : list
        Indices of consensus-selected variables (≥3/5)
    feature_names : list, optional
        Original feature names (e.g., wavelengths/wavenumbers)
    title : str
        Plot title

    Returns
    -------
    fig : plotly.graph_objects.Figure
        Bar chart showing all variables in spectral order
    """

    n_vars = len(consensus_frequency)
    all_indices = np.arange(n_vars)

    # Create column names
    if feature_names is not None:
        col_names = [str(feature_names[i]) for i in all_indices]
    else:
        col_names = [f'Var_{i}' for i in all_indices]

    # Determine which variables are selected (consensus ≥3/5)
    selected_set = set(final_selected_variables)

    # Create color array: green for selected, blue for not selected
    colors = ['#00CC96' if i in selected_set else '#636EFA' for i in all_indices]

    # Create hover text with more info
    hover_text = []
    for i in all_indices:
        freq = int(consensus_frequency[i])
        status = "Selected (≥3/5)" if i in selected_set else "Not selected"
        hover_text.append(
            f"<b>{col_names[i]}</b><br>"
            f"Frequency: {freq}/5<br>"
            f"Status: {status}"
        )

    # Create bar chart
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=col_names,
        y=consensus_frequency,
        marker_color=colors,
        hovertemplate='%{hovertext}<extra></extra>',
        hovertext=hover_text,
        showlegend=False
    ))

    # Add custom legend entries
    fig.add_trace(go.Scatter(
        x=[None],
        y=[None],
        mode='markers',
        marker=dict(size=10, color='#00CC96', symbol='square'),
        name=f'Selected (≥3/5): {len(final_selected_variables)} vars',
        showlegend=True
    ))

    fig.add_trace(go.Scatter(
        x=[None],
        y=[None],
        mode='markers',
        marker=dict(size=10, color='#636EFA', symbol='square'),
        name=f'Not selected: {n_vars - len(final_selected_variables)} vars',
        showlegend=True
    ))

    # Update layout
    fig.update_layout(
        title={
            'text': title,
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis_title='Variable (Spectral Order)',
        yaxis_title='Selection Frequency (out of 5 runs)',
        yaxis=dict(
            range=[0, 5.5],
            tickmode='linear',
            tick0=0,
            dtick=1
        ),
        hovermode='closest',
        template='plotly_white',
        showlegend=True,
        legend=dict(
            x=1.02,
            y=1,
            xanchor='left',
            yanchor='top',
            bgcolor='rgba(255, 255, 255, 0.9)',
            bordercolor='black',
            borderwidth=1
        ),
        height=500,
        margin=dict(l=60, r=200, t=80, b=80)
    )

    # Rotate x-axis labels for readability
    fig.update_xaxes(tickangle=-45)

    return fig


def plot_fitness_evolution(
    fitness_history: List[Dict],
    show_runs: int = 10
) -> go.Figure:
    """
    Plot fitness score evolution over runs.

    Parameters
    ----------
    fitness_history : list of dict
        Fitness history from GA runs
    show_runs : int
        Number of individual runs to display

    Returns
    -------
    fig : plotly.graph_objects.Figure
        Interactive line chart
    """
    fig = go.Figure()

    # Extract data
    run_numbers = []
    best_fitnesses = []
    n_variables = []

    for entry in fitness_history:
        run_numbers.append(entry['run'])
        best_fitnesses.append(entry['best_fitness'])
        # GAPLSSP uses 'n_selected', not 'n_variables'
        n_variables.append(entry.get('n_selected', entry.get('n_variables', 0)))

    # NOTE: GAPLSSP does not provide per-generation history ('generation_history')
    # It only provides per-run best fitness values
    # Commenting out this section that tries to plot generation-by-generation evolution

    # if len(fitness_history) > show_runs:
    #     sample_indices = np.linspace(0, len(fitness_history) - 1, show_runs, dtype=int)
    # else:
    #     sample_indices = range(len(fitness_history))
    #
    # for idx in sample_indices:
    #     entry = fitness_history[idx]
    #     gen_history = entry['generation_history']  # ← This key doesn't exist in GAPLSSP!
    #
    #     fig.add_trace(go.Scatter(
    #         x=list(range(len(gen_history))),
    #         y=gen_history,
    #         mode='lines',
    #         name=f"Run {entry['run'] + 1}",
    #         line=dict(width=1),
    #         opacity=0.3,
    #         hovertemplate='Generation: %{x}<br>Fitness: %{y:.2f}<extra></extra>'
    #     ))

    # Plot best fitness across runs
    fig.add_trace(go.Scatter(
        x=run_numbers,
        y=best_fitnesses,
        mode='markers+lines',
        name='Best per Run',
        marker=dict(size=8, color='red'),
        line=dict(width=2, color='red'),
        hovertemplate='Run: %{x}<br>Best Fitness: %{y:.2f}<extra></extra>'
    ))

    # Add moving average
    if len(best_fitnesses) > 3:
        window = min(5, len(best_fitnesses) // 3)
        moving_avg = pd.Series(best_fitnesses).rolling(window, center=True).mean()

        fig.add_trace(go.Scatter(
            x=run_numbers,
            y=moving_avg,
            mode='lines',
            name=f'Moving Average ({window})',
            line=dict(width=3, color='orange', dash='dash'),
            hovertemplate='Average Fitness: %{y:.2f}<extra></extra>'
        ))

    fig.update_layout(
        title={
            'text': 'Fitness Score Evolution',
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis_title='Run / Generation',
        yaxis_title='Fitness Score',
        hovermode='closest',
        template='plotly_white',
        height=500
    )

    return fig


def plot_fitness_vs_nvars(
    fitness_history: List[Dict]
) -> go.Figure:
    """
    Plot fitness vs number of variables (Pareto front).

    Parameters
    ----------
    fitness_history : list of dict
        Fitness history from GA runs

    Returns
    -------
    fig : plotly.graph_objects.Figure
        Interactive scatter plot
    """
    # Extract data
    # GAPLSSP uses 'n_selected', not 'n_variables'
    n_variables = [entry.get('n_selected', entry.get('n_variables', 0)) for entry in fitness_history]
    best_fitnesses = [entry['best_fitness'] for entry in fitness_history]
    runs = [entry['run'] for entry in fitness_history]

    df = pd.DataFrame({
        'N_Variables': n_variables,
        'Fitness': best_fitnesses,
        'Run': runs
    })

    # Find Pareto front (non-dominated solutions)
    pareto_mask = []
    for i in range(len(df)):
        is_dominated = False
        for j in range(len(df)):
            if i != j:
                # Dominated if another solution has better fitness AND fewer variables
                if (df.iloc[j]['Fitness'] >= df.iloc[i]['Fitness'] and
                    df.iloc[j]['N_Variables'] <= df.iloc[i]['N_Variables'] and
                    (df.iloc[j]['Fitness'] > df.iloc[i]['Fitness'] or
                     df.iloc[j]['N_Variables'] < df.iloc[i]['N_Variables'])):
                    is_dominated = True
                    break
        pareto_mask.append(not is_dominated)

    df['Pareto'] = pareto_mask

    fig = go.Figure()

    # Non-Pareto points
    non_pareto = df[~df['Pareto']]
    if len(non_pareto) > 0:
        fig.add_trace(go.Scatter(
            x=non_pareto['N_Variables'],
            y=non_pareto['Fitness'],
            mode='markers',
            name='All Solutions',
            marker=dict(size=8, color='lightblue', opacity=0.6),
            hovertemplate='Variables: %{x}<br>Fitness: %{y:.2f}<extra></extra>'
        ))

    # Pareto front
    pareto = df[df['Pareto']].sort_values('N_Variables')
    if len(pareto) > 0:
        fig.add_trace(go.Scatter(
            x=pareto['N_Variables'],
            y=pareto['Fitness'],
            mode='markers+lines',
            name='Pareto Front',
            marker=dict(size=12, color='red', symbol='star'),
            line=dict(width=2, color='red', dash='dash'),
            hovertemplate='Variables: %{x}<br>Fitness: %{y:.2f}<extra></extra>'
        ))

    fig.update_layout(
        title={
            'text': 'Fitness vs Number of Variables (Pareto Front)',
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis_title='Number of Variables',
        yaxis_title='Fitness Score',
        hovermode='closest',
        template='plotly_white',
        height=500
    )

    return fig


def plot_population_heatmap(
    population_history: List[np.ndarray],
    column_names: List[str],
    max_generations: int = 50,
    max_vars: int = 100
) -> go.Figure:
    """
    Plot population evolution as heatmap.

    Shows which variables are selected across generations.

    Parameters
    ----------
    population_history : list of np.ndarray
        Population chromosomes over generations
    column_names : list
        Variable names
    max_generations : int
        Maximum generations to display
    max_vars : int
        Maximum variables to display

    Returns
    -------
    fig : plotly.graph_objects.Figure
        Interactive heatmap
    """
    # Sample generations if too many
    if len(population_history) > max_generations:
        sample_indices = np.linspace(
            0, len(population_history) - 1, max_generations, dtype=int
        )
        population_history = [population_history[i] for i in sample_indices]

    # Compute selection frequency per generation per variable
    n_vars = population_history[0].shape[1]

    # Sample variables if too many
    if n_vars > max_vars:
        # Select variables with highest overall frequency
        total_freq = np.sum([pop.sum(axis=0) for pop in population_history], axis=0)
        top_var_indices = np.argsort(total_freq)[-max_vars:]
        column_names = [column_names[i] for i in top_var_indices]
    else:
        top_var_indices = np.arange(n_vars)

    # Create heatmap data
    heatmap_data = []
    for pop in population_history:
        freq = pop[:, top_var_indices].sum(axis=0) / len(pop) * 100
        heatmap_data.append(freq)

    heatmap_data = np.array(heatmap_data)

    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data.T,
        x=[f'Gen {i}' for i in range(len(population_history))],
        y=column_names,
        colorscale='Viridis',
        hovertemplate='Generation: %{x}<br>Variable: %{y}<br>Selection: %{z:.1f}%<extra></extra>',
        colorbar=dict(title='Selection %')
    ))

    fig.update_layout(
        title={
            'text': 'Variable Selection Evolution Across Generations',
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis_title='Generation',
        yaxis_title='Variable',
        height=max(400, min(800, len(column_names) * 15)),
        template='plotly_white'
    )

    return fig


def create_results_summary_table(
    selected_vars: List[str],
    selection_freq: np.ndarray,
    column_names: List[str],
    best_fitness: float
) -> pd.DataFrame:
    """
    Create summary table of results.

    Parameters
    ----------
    selected_vars : list
        Names of selected variables
    selection_freq : np.ndarray
        Selection frequency for all variables
    column_names : list
        All variable names
    best_fitness : float
        Best fitness achieved

    Returns
    -------
    df : pd.DataFrame
        Summary table
    """
    # All variables sorted by frequency
    sorted_indices = np.argsort(selection_freq)[::-1]

    df = pd.DataFrame({
        'Variable': [column_names[i] for i in sorted_indices],
        'Selection_Frequency': selection_freq[sorted_indices],
        'Final_Model': ['✓' if column_names[i] in selected_vars else ''
                       for i in sorted_indices],
        'Frequency_%': (selection_freq[sorted_indices] /
                       max(selection_freq) * 100).round(1)
    })

    return df


def plot_cv_curve(
    n_vars_history: List[int],
    cv_scores_history: List[float]
) -> go.Figure:
    """
    Plot cross-validation score vs number of variables.

    Useful for identifying the "elbow point" - optimal number of variables.

    Parameters
    ----------
    n_vars_history : list
        Number of variables at each iteration
    cv_scores_history : list
        CV scores corresponding to each iteration

    Returns
    -------
    fig : plotly.graph_objects.Figure
        Interactive line chart
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=n_vars_history,
        y=cv_scores_history,
        mode='markers+lines',
        marker=dict(size=8, color='blue'),
        line=dict(width=2),
        hovertemplate='Variables: %{x}<br>CV Score: %{y:.2f}<extra></extra>'
    ))

    # Add trend line
    if len(n_vars_history) > 2:
        z = np.polyfit(n_vars_history, cv_scores_history, 2)
        p = np.poly1d(z)
        x_smooth = np.linspace(min(n_vars_history), max(n_vars_history), 100)
        y_smooth = p(x_smooth)

        fig.add_trace(go.Scatter(
            x=x_smooth,
            y=y_smooth,
            mode='lines',
            name='Trend',
            line=dict(width=2, color='red', dash='dash'),
            hoverinfo='skip'
        ))

    fig.update_layout(
        title={
            'text': 'Cross-Validation Score vs Number of Variables',
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis_title='Number of Variables',
        yaxis_title='CV Score',
        hovermode='closest',
        template='plotly_white',
        height=500
    )

    return fig


def plot_rmsecv_curve(
    n_vars_history: List[int],
    rmsecv_history: List[float]
) -> go.Figure:
    """
    Plot RMSECV (Root Mean Square Error - Cross Validation) vs number of variables.

    Shows how prediction error decreases as more variables are included.
    Error curve typically shows rapid decrease initially, then plateaus.
    This is Graph 3 from the Leardi paper.

    Parameters
    ----------
    n_vars_history : list or np.ndarray
        Number of variables at each step
        Example: [1, 2, 3, ..., 57, 100, 151]

    rmsecv_history : list or np.ndarray
        RMSECV values corresponding to each number of variables
        Example: [5.2, 4.1, 3.5, ..., 1.2, 1.0, 1.0]

    Returns
    -------
    fig : plotly.graph_objects.Figure
        Interactive line chart with markers

    Notes
    -----
    - RMSECV is calculated from cross-validation residuals
    - Lower values = better predictions
    - Curve typically shows exponential decay (rapid initial drop, then plateau)
    - Useful for identifying "elbow point" (diminishing returns)

    References
    ----------
    Leardi et al. (2002) - Variable selection using genetic algorithms
    """

    fig = go.Figure()

    # Add line trace
    fig.add_trace(go.Scatter(
        x=n_vars_history,
        y=rmsecv_history,
        mode='lines+markers',
        name='RMSECV',
        line=dict(
            color='#FF7F0E',      # Orange color (matches Leardi paper)
            width=2
        ),
        marker=dict(
            size=6,
            color='#FF7F0E'
        ),
        hovertemplate='<b>Variables: %{x}</b><br>' +
                     'RMSECV: %{y:.3f}<br>' +
                     '<extra></extra>'
    ))

    # Update layout
    fig.update_layout(
        title={
            'text': "RMSECV as a Function of the Number of Selected Variables",
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis_title='Number of Variables',
        yaxis_title='RMSECV',
        hovermode='x unified',
        template='plotly_white',
        height=500,
        margin=dict(l=80, r=50, t=80, b=80)
    )

    return fig


def plot_multirun_band_selection(
    X: np.ndarray,
    all_selected_variables: List[set],
    title: str = "Band Selection Pattern Across 5 Independent Runs"
) -> go.Figure:
    """
    Plot spectrum with band selections from all 5 runs overlaid.

    Shows which bands each run selected with different colors per run.
    Allows visual inspection of:
    - Which bands are consistently selected (robust = consensus)
    - Which bands vary across runs (less stable)

    Similar to Leardi's visualization showing multiple run patterns.

    Parameters
    ----------
    X : np.ndarray
        Spectral data (n_samples, n_variables)
    all_selected_variables : list of set
        Selected variable indices from each of 5 runs
        Example: [set([1, 2, 5]), set([1, 3, 5]), ...]
    title : str
        Plot title

    Returns
    -------
    fig : plotly.graph_objects.Figure
        Interactive plot with spectrum and run-specific band markers
    """

    # Compute mean spectrum
    mean_spectrum = np.mean(X, axis=0)
    n_vars = X.shape[1]
    x_axis = np.arange(n_vars)

    fig = go.Figure()

    # ────────────────────────────────────────────────────────
    # PART 1: Plot mean spectrum (RED background)
    # ────────────────────────────────────────────────────────

    fig.add_trace(go.Scatter(
        x=x_axis,
        y=mean_spectrum,
        mode='lines',
        name='Mean Spectrum',
        line=dict(color='red', width=2),
        fill='tozeroy',
        fillcolor='rgba(255, 100, 100, 0.10)',
        hovertemplate='<b>Var %{x}</b><br>Absorbance: %{y:.4f}<extra></extra>'
    ))

    # ────────────────────────────────────────────────────────
    # PART 2: Add markers for each run's selected variables
    # ────────────────────────────────────────────────────────

    # Define distinct colors for 5 runs (vibrant, distinguishable)
    run_colors = [
        '#2E8B57',  # Sea Green
        '#4169E1',  # Royal Blue
        '#FF8C00',  # Dark Orange
        '#9370DB',  # Medium Purple
        '#DC143C'   # Crimson
    ]

    run_labels = ['Run 1', 'Run 2', 'Run 3', 'Run 4', 'Run 5']

    # Get Y-axis range for positioning markers
    y_min = np.min(mean_spectrum)
    y_max = np.max(mean_spectrum)
    y_range = y_max - y_min

    # Each run gets its own horizontal level for markers
    marker_levels = [
        y_min - 0.03 * y_range,  # Run 1
        y_min - 0.06 * y_range,  # Run 2
        y_min - 0.09 * y_range,  # Run 3
        y_min - 0.12 * y_range,  # Run 4
        y_min - 0.15 * y_range   # Run 5
    ]

    for run_idx, (selected_vars, color, label, marker_y) in enumerate(
        zip(all_selected_variables, run_colors, run_labels, marker_levels)
    ):
        if len(selected_vars) == 0:
            continue

        selected_array = np.sort(list(selected_vars))

        # Add horizontal line showing selected variables as tick marks
        fig.add_trace(go.Scatter(
            x=selected_array,
            y=[marker_y] * len(selected_array),
            mode='markers',
            name=label,
            marker=dict(
                size=4,
                color=color,
                symbol='line-ns',  # Vertical tick marks
                line=dict(width=2)
            ),
            hovertemplate=f'<b>{label}</b><br>Variable: %{{x}}<extra></extra>'
        ))

    # ────────────────────────────────────────────────────────
    # PART 3: Overlay shaded regions showing each run's bands
    # (optional - adds transparency layers)
    # ────────────────────────────────────────────────────────

    for run_idx, (selected_vars, color) in enumerate(
        zip(all_selected_variables, run_colors)
    ):
        if len(selected_vars) == 0:
            continue

        selected_array = np.sort(list(selected_vars))

        # Group consecutive variables into contiguous bands
        bands = []
        if len(selected_array) > 0:
            band_start = selected_array[0]
            band_end = selected_array[0]

            for var_idx in selected_array[1:]:
                if var_idx == band_end + 1:
                    band_end = var_idx
                else:
                    bands.append((band_start, band_end))
                    band_start = var_idx
                    band_end = var_idx
            bands.append((band_start, band_end))

        # Add semi-transparent vertical bands
        for band_start, band_end in bands:
            fig.add_vrect(
                x0=band_start - 0.5,
                x1=band_end + 0.5,
                fillcolor=color,
                opacity=0.08,  # Very transparent
                layer='below',
                line_width=0
            )

    # ────────────────────────────────────────────────────────
    # PART 4: Count consensus variables (≥3/5 runs)
    # ────────────────────────────────────────────────────────

    var_counts = {}
    for selected_vars in all_selected_variables:
        for var_idx in selected_vars:
            var_counts[var_idx] = var_counts.get(var_idx, 0) + 1

    consensus_vars = [v for v, c in var_counts.items() if c >= 3]

    # ────────────────────────────────────────────────────────
    # PART 5: Layout and formatting
    # ────────────────────────────────────────────────────────

    fig.update_layout(
        title={
            'text': f"{title}<br><sub style='font-size:11px'>Consensus (≥3/5 runs): {len(consensus_vars)} variables</sub>",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 14, 'color': 'black'}
        },
        xaxis=dict(
            title='Variable Index (Wavelength / Wavenumber)',
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(200, 200, 200, 0.2)'
        ),
        yaxis=dict(
            title='Absorbance / Intensity',
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(200, 200, 200, 0.2)',
            range=[y_min - 0.20 * y_range, y_max + 0.05 * y_range]  # Extra space at bottom
        ),
        hovermode='x unified',
        template='plotly_white',
        height=600,
        showlegend=True,
        legend=dict(
            x=1.02,
            y=1,
            xanchor='left',
            yanchor='top',
            bgcolor='rgba(255, 255, 255, 0.9)',
            bordercolor='black',
            borderwidth=1
        ),
        margin=dict(l=60, r=200, t=100, b=60)
    )

    return fig


def plot_spectra_with_consensus_bands(
    X: np.ndarray,
    final_selected_variables: List[int],
    consensus_matrix: np.ndarray,
    title: str = "Spectral Data with Selected Bands (Consensus ≥3/5)"
) -> go.Figure:
    """
    Plot spectra with consensus-selected bands highlighted.

    Shows:
    - Faint individual spectra (light red)
    - Bold mean spectrum (red)
    - Green shaded regions for consensus bands
    - Bottom indicators showing per-run selections

    Parameters
    ----------
    X : np.ndarray
        Spectral data (samples × variables)
    final_selected_variables : list
        Variable indices selected by consensus (≥3/5)
    consensus_matrix : np.ndarray
        Selection pattern (variables × 5 runs)
    title : str
        Plot title

    Returns
    -------
    fig : plotly.graph_objects.Figure
        Plotly figure object
    """

    fig = go.Figure()

    # ───────────────────────────────────────────────────────────
    # PART 1: Add all spectra (faint, as background)
    # ───────────────────────────────────────────────────────────

    for sample_idx in range(min(X.shape[0], 100)):  # Limit to 100 for performance
        fig.add_trace(go.Scatter(
            x=np.arange(X.shape[1]),
            y=X[sample_idx],
            mode='lines',
            line=dict(color='rgba(255, 100, 100, 0.05)', width=0.5),
            showlegend=(sample_idx == 0),  # Only first one in legend
            name='Individual Spectra',
            hoverinfo='skip'
        ))

    # ───────────────────────────────────────────────────────────
    # PART 2: Add mean spectrum (bold red line)
    # ───────────────────────────────────────────────────────────

    mean_spectrum = np.mean(X, axis=0)
    fig.add_trace(go.Scatter(
        x=np.arange(X.shape[1]),
        y=mean_spectrum,
        mode='lines',
        line=dict(color='red', width=2),
        name='Mean Spectrum',
        hovertemplate='<b>Var %{x}</b><br>Mean: %{y:.4f}<extra></extra>'
    ))

    # ───────────────────────────────────────────────────────────
    # PART 3: Add green shaded regions for selected bands
    # ───────────────────────────────────────────────────────────

    # Group consecutive selected variables into bands
    selected_array = np.array(final_selected_variables, dtype=int)
    selected_array = np.sort(selected_array)  # Ensure sorted

    # Find contiguous bands
    bands = []
    if len(selected_array) > 0:
        current_band_start = selected_array[0]
        current_band_end = selected_array[0]

        for var_idx in selected_array[1:]:
            if var_idx == current_band_end + 1:
                # Consecutive, extend current band
                current_band_end = var_idx
            else:
                # Gap found, save current band and start new
                bands.append((current_band_start, current_band_end))
                current_band_start = var_idx
                current_band_end = var_idx

        # Don't forget last band
        bands.append((current_band_start, current_band_end))

    # Add shaded regions for each band
    band_legend_added = False
    for band_idx, (band_start, band_end) in enumerate(bands):
        # Green shaded rectangle
        fig.add_vrect(
            x0=band_start - 0.5,
            x1=band_end + 0.5,
            fillcolor='green',
            opacity=0.15,
            layer='below',
            line_width=2,
            line_color='green',
            line_dash='solid',
            annotation_text=f'Band {band_idx + 1}',
            annotation_position='top',
            annotation=dict(
                font_size=10,
                font_color='green'
            )
        )

        # Add to legend only once
        if not band_legend_added:
            fig.add_trace(go.Scatter(
                x=[None],
                y=[None],
                mode='markers',
                marker=dict(size=10, color='green', opacity=0.3, symbol='square'),
                showlegend=True,
                name=f'Consensus Bands ({len(bands)} total)',
                hoverinfo='skip'
            ))
            band_legend_added = True

    # ───────────────────────────────────────────────────────────
    # PART 4: Add bottom pattern showing per-run selections
    # (small vertical ticks at bottom showing which runs selected each var)
    # ───────────────────────────────────────────────────────────

    # Get Y-axis range for bottom annotation
    y_min = np.min(mean_spectrum)
    y_max = np.max(mean_spectrum)
    y_range = y_max - y_min

    # Create small tick marks at bottom for each run's selections
    run_colors = ['rgba(0, 0, 0, 0.3)', 'rgba(50, 50, 50, 0.3)',
                  'rgba(100, 100, 100, 0.3)', 'rgba(150, 150, 150, 0.3)',
                  'rgba(200, 200, 200, 0.3)']

    for run_idx in range(consensus_matrix.shape[1]):
        selected_in_run = np.where(consensus_matrix[:, run_idx] == 1)[0]

        if len(selected_in_run) > 0:
            # Create vertical tick marks for this run
            tick_y = y_min - (0.02 + run_idx * 0.015) * y_range

            fig.add_trace(go.Scatter(
                x=selected_in_run,
                y=[tick_y] * len(selected_in_run),
                mode='markers',
                marker=dict(
                    size=3,
                    color=run_colors[run_idx],
                    symbol='line-ns',
                    line=dict(width=1)
                ),
                showlegend=(run_idx == 0),
                name='Per-run selections',
                hovertemplate=f'<b>Run {run_idx + 1}</b><br>Variable: %{{x}}<extra></extra>'
            ))

    # ───────────────────────────────────────────────────────────
    # PART 5: Layout and formatting
    # ───────────────────────────────────────────────────────────

    fig.update_layout(
        title={
            'text': title,
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 16, 'color': 'black'}
        },
        xaxis=dict(
            title='Variable Index (Wavelength / Wavenumber)',
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(200, 200, 200, 0.2)'
        ),
        yaxis=dict(
            title='Absorbance / Intensity',
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(200, 200, 200, 0.2)',
            range=[y_min - 0.15 * y_range, y_max + 0.05 * y_range]  # Extra space at bottom
        ),
        hovermode='x unified',
        template='plotly_white',
        height=600,
        showlegend=True,
        legend=dict(
            x=1.02,
            y=1,
            xanchor='left',
            yanchor='top',
            bgcolor='rgba(255, 255, 255, 0.9)',
            bordercolor='black',
            borderwidth=1
        ),
        margin=dict(l=60, r=200, t=100, b=60)
    )

    return fig


def export_selected_dataset(
    X: np.ndarray,
    y: np.ndarray,
    final_selected_variables: List[int],
    consensus_matrix: np.ndarray,
    consensus_frequency: np.ndarray,
    feature_names: List[str] = None,
    y_column_name: str = "Target",
    dataset_name: str = "selected_bands"
) -> bytes:
    """
    Export dataset with selected bands to XLSX.

    Creates a multi-sheet Excel file with:
    - Sheet 1: Data (Y + selected X variables with ORIGINAL names)
    - Sheet 2: Band metadata (variables, frequencies, run info)
    - Sheet 3: Summary (statistics and methods)

    Structure matches ORIGINAL dataset:
    - First column: Y (with original column name!)
    - Following columns: Selected variable columns (in original order)

    Parameters
    ----------
    X : np.ndarray
        Original spectral data (samples × variables)
    y : np.ndarray
        Target variable (samples,)
    final_selected_variables : list
        Consensus-selected variable indices (≥3/5)
    consensus_matrix : np.ndarray
        Selection pattern (variables × 5 runs)
    consensus_frequency : np.ndarray
        Selection frequency per variable
    feature_names : list, optional
        Original feature column names (e.g., ['10000', '9992', ...])
    y_column_name : str
        Original name of Y column (e.g., "(w/w) of Bar")
        Default: "Target"
    dataset_name : str
        Name for output file (not used, kept for compatibility)

    Returns
    -------
    bytes
        XLSX file content (for Streamlit download)
    """

    # ───────────────────────────────────────────────────────────
    # PART 1: Sort selected variables by index
    # ───────────────────────────────────────────────────────────

    selected_vars_array = np.array(
        sorted(final_selected_variables),
        dtype=int
    )

    # ───────────────────────────────────────────────────────────
    # PART 2: Create main data sheet (Y FIRST, then selected X)
    # ───────────────────────────────────────────────────────────

    # Start with Y as FIRST column (use original name!)
    df_data = pd.DataFrame({
        y_column_name: y
    })

    # Add selected variable columns in original order
    for var_idx in selected_vars_array:
        # Determine column name
        if feature_names is not None and var_idx < len(feature_names):
            col_name = feature_names[var_idx]
        else:
            col_name = f'Var_{var_idx}'

        # Add column from original X
        df_data[col_name] = X[:, var_idx]

    # ───────────────────────────────────────────────────────────
    # PART 3: Create metadata sheet (band information)
    # ───────────────────────────────────────────────────────────

    # Create band info
    band_info = []
    for var_idx in selected_vars_array:
        freq = consensus_frequency[var_idx]
        selected_in_runs = [
            f"Run {r+1}"
            for r in range(5)
            if consensus_matrix[var_idx, r] == 1
        ]

        # Get feature name
        if feature_names is not None and var_idx < len(feature_names):
            col_name = feature_names[var_idx]
        else:
            col_name = f'Var_{var_idx}'

        band_info.append({
            'Original_Column': col_name,
            'Variable_Index': var_idx,
            'Consensus_Frequency': int(freq),
            'Selected_in_Runs': ', '.join(selected_in_runs),
            'Selection_Rate': f'{int(freq)}/5'
        })

    df_metadata = pd.DataFrame(band_info)

    # ───────────────────────────────────────────────────────────
    # PART 4: Create summary sheet
    # ───────────────────────────────────────────────────────────

    summary_data = {
        'Metric': [
            'Original Variables',
            'Selected Variables',
            'Reduction (%)',
            'Total Samples',
            'Consensus Threshold',
            'Method',
            'Y Column Name',
            'Structure'
        ],
        'Value': [
            X.shape[1],
            len(selected_vars_array),
            f'{(1 - len(selected_vars_array)/X.shape[1]) * 100:.1f}%',
            X.shape[0],
            '≥3/5 runs (60%)',
            'Leardi GAPLSSP (5 independent runs)',
            y_column_name,
            f'{y_column_name} | Selected Variables'
        ]
    }

    df_summary = pd.DataFrame(summary_data)

    # ───────────────────────────────────────────────────────────
    # PART 5: Write to XLSX with multiple sheets
    # ───────────────────────────────────────────────────────────

    output = BytesIO()

    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Sheet 1: Data
        df_data.to_excel(
            writer,
            sheet_name='Data',
            index=False
        )

        # Sheet 2: Band Metadata
        df_metadata.to_excel(
            writer,
            sheet_name='Band_Metadata',
            index=False
        )

        # Sheet 3: Summary
        df_summary.to_excel(
            writer,
            sheet_name='Summary',
            index=False
        )

    output.seek(0)
    return output.getvalue()
