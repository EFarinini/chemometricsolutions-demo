"""
PLS Diagnostic Plots Module - Extended
========================================

Comprehensive diagnostic visualizations for PLS regression:
- Loading plots (X and Y loadings)
- VIP (Variable Importance in Projection)
- Score plots (T vs U, with ellipse)
- Residual diagnostics
- Validation plots
- Model stability plots

References:
- Eriksson et al. (2006) - Multi- and Megavariate Data Analysis
- Wold et al. (2001) - PLS regression: a versatile tool
- Chong & Jun (2005) - Performance of some variable selection methods

Author: ChemoMetric Solutions
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Any, Optional, List, Tuple
import warnings


def plot_loading_plot(model: Dict[str, Any],
                      lv: int = 1,
                      lv_y: Optional[int] = 2,
                      n_features_highlight: int = 10,
                      feature_names: Optional[List[str]] = None) -> go.Figure:
    """
    Create loading plot (biplot) for PLS model.

    Shows relationship between X variables and latent variables.
    Variables far from origin are more important.

    Parameters
    ----------
    model : Dict
        Fitted PLS model
    lv : int
        First latent variable to plot (X-axis)
    lv_y : Optional[int]
        Second latent variable (Y-axis). If None, uses Y loading vs X loading
    n_features_highlight : int
        Number of top features to label
    feature_names : Optional[List[str]]
        Feature names for labeling

    Returns
    -------
    go.Figure
        Interactive loading plot

    Interpretation:
    - Distance from origin: importance of variable
    - Angle between variables: correlation
    - Position: contribution to LV
    """
    P = model['P']
    Q = model['Q']

    if feature_names is None:
        feature_names = model.get('feature_names', None)

    if lv > P.shape[1]:
        raise ValueError(f"LV {lv} exceeds model components {P.shape[1]}")

    lv_idx = lv - 1

    # X loadings
    p_x = P[:, lv_idx]

    if lv_y is None:
        # Y loading vs X loading (1-dimensional Y)
        q_y = Q[lv_idx] if len(Q.shape) == 1 else Q[lv_idx, 0]
        p_y = np.full_like(p_x, q_y)
        x_axis_label = f"X Loadings - LV{lv}"
        y_axis_label = "Y Loading"
    else:
        lv_y_idx = lv_y - 1
        p_y = P[:, lv_y_idx]
        x_axis_label = f"X Loadings - LV{lv}"
        y_axis_label = f"X Loadings - LV{lv_y}"

    # Feature names
    if feature_names is None:
        feature_names = [f"Var_{i+1}" for i in range(len(p_x))]

    # Calculate distances from origin
    distances = np.sqrt(p_x**2 + p_y**2)
    top_indices = np.argsort(distances)[-n_features_highlight:]

    # Create figure
    fig = go.Figure()

    # Plot all points
    fig.add_trace(go.Scatter(
        x=p_x,
        y=p_y,
        mode='markers',
        marker=dict(
            size=5,
            color=distances,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Distance"),
            line=dict(width=0.5, color='white')
        ),
        text=feature_names,
        hovertemplate='<b>%{text}</b><br>X: %{x:.3f}<br>Y: %{y:.3f}<extra></extra>',
        name='Variables'
    ))

    # Label top features
    for idx in top_indices:
        fig.add_annotation(
            x=p_x[idx],
            y=p_y[idx],
            text=feature_names[idx],
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=1,
            arrowcolor='rgba(0,0,0,0.3)',
            ax=30,
            ay=30,
            font=dict(size=9),
            bgcolor='rgba(255,255,255,0.7)',
            bordercolor='gray',
            borderwidth=0.5
        )

    # Add circle of correlation (unit circle)
    max_loading = max(np.max(np.abs(p_x)), np.max(np.abs(p_y)))
    if max_loading > 0.5:  # Only show if loadings are substantial
        theta = np.linspace(0, 2*np.pi, 100)
        circle_x = np.cos(theta)
        circle_y = np.sin(theta)
        fig.add_trace(go.Scatter(
            x=circle_x,
            y=circle_y,
            mode='lines',
            line=dict(color='rgba(0,0,0,0.3)', width=1, dash='dash'),
            name='Unit Circle',
            hoverinfo='skip'
        ))

    # Add reference lines
    fig.add_hline(y=0, line_dash="dash", line_color="rgba(0,0,0,0.2)")
    fig.add_vline(x=0, line_dash="dash", line_color="rgba(0,0,0,0.2)")

    fig.update_layout(
        title=f"Loading Plot: LV{lv} vs LV{lv_y if lv_y else 'Y'}",
        xaxis_title=x_axis_label,
        yaxis_title=y_axis_label,
        hovermode='closest',
        width=900,
        height=700,
        template='plotly_white',
        xaxis=dict(scaleanchor="y", scaleratio=1),
        yaxis=dict(scaleanchor="x", scaleratio=1)
    )

    return fig


def calculate_vip(model: Dict[str, Any]) -> np.ndarray:
    """
    Calculate Variable Importance in Projection (VIP) scores.

    VIP measures the contribution of each X variable across all PLS components.

    Formula:
    VIP_j = sqrt(p * sum_h(SS(b_h * w_jh)^2) / sum_h(SS(b_h * t_h)^2))

    where:
    - p: number of X variables
    - w_jh: weight of variable j in component h
    - b_h: regression coefficient for component h
    - t_h: scores for component h

    Parameters
    ----------
    model : Dict
        Fitted PLS model

    Returns
    -------
    np.ndarray
        VIP scores for each variable (n_features,)

    Interpretation:
    - VIP > 1.0: Important for model
    - VIP < 0.5: Can be removed
    - VIP 0.5-1.0: Borderline important
    """
    W = model['W']  # X weights (n_features, n_components)
    T = model['T']  # X scores (n_samples, n_components)
    Q = model['Q']  # y loadings (n_components,)

    n_features, n_components = W.shape
    n_samples = T.shape[0]

    # Calculate explained variance per component
    # SS(b_h * t_h) for each component
    ss_per_component = np.zeros(n_components)
    for h in range(n_components):
        ss_per_component[h] = np.sum(T[:, h]**2) * Q[h]**2

    total_ss = np.sum(ss_per_component)

    # Calculate VIP for each variable
    vip = np.zeros(n_features)
    for j in range(n_features):
        sum_contrib = 0
        for h in range(n_components):
            sum_contrib += ss_per_component[h] * (W[j, h]**2)
        vip[j] = np.sqrt(n_features * sum_contrib / (total_ss + 1e-10))

    return vip


def plot_vip(model: Dict[str, Any],
             threshold: float = 1.0,
             n_features_show: Optional[int] = None,
             feature_names: Optional[List[str]] = None) -> go.Figure:
    """
    Create Variable Importance in Projection (VIP) plot.

    Variables with VIP > threshold are important for predictions.

    Parameters
    ----------
    model : Dict
        Fitted PLS model
    threshold : float
        VIP threshold (default: 1.0)
    n_features_show : Optional[int]
        Show only top N features. If None, show all
    feature_names : Optional[List[str]]
        Feature names for labeling

    Returns
    -------
    go.Figure
        VIP bar plot

    Interpretation:
    - Higher VIP: more important variable
    - Threshold line: typical cutoff (VIP=1.0)
    - Above threshold: keep variable
    - Below threshold: consider removing
    """
    vip = calculate_vip(model)

    if feature_names is None:
        feature_names = model.get('feature_names', None)

    if feature_names is None:
        feature_names = [f"Var_{i+1}" for i in range(len(vip))]

    # Sort by VIP
    sorted_idx = np.argsort(vip)[::-1]
    vip_sorted = vip[sorted_idx]
    names_sorted = [feature_names[i] for i in sorted_idx]

    # Limit to top N
    if n_features_show is not None:
        vip_sorted = vip_sorted[:n_features_show]
        names_sorted = names_sorted[:n_features_show]

    # Color by threshold
    colors = ['#d62728' if v < threshold else '#2ca02c' for v in vip_sorted]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=names_sorted,      # ← X = variabili (orizzontale)
        y=vip_sorted,        # ← Y = VIP score (verticale)
        orientation='v',     # ← Orientation verticale
        marker=dict(color=colors, line=dict(width=0.5, color='white')),
        text=[f"{v:.2f}" for v in vip_sorted],
        textposition='auto',
        hovertemplate='<b>%{x}</b><br>VIP: %{y:.3f}<extra></extra>',
        name='VIP'
    ))

    # Add threshold line
    fig.add_hline(
        y=threshold,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Threshold: {threshold}",
        annotation_position="right"
    )

    fig.update_layout(
        title=f"Variable Importance in Projection (VIP) - {n_components} LV" if (n_components := model.get('n_components')) else "Variable Importance in Projection (VIP)",
        xaxis_title="Variable",
        yaxis_title="VIP Score",
        height=600,
        showlegend=False,
        template='plotly_white',
        hovermode='closest'
    )

    return fig


def plot_score_plot_with_ellipse(model: Dict[str, Any],
                                 lv1: int = 1,
                                 lv2: int = 2,
                                 confidence: float = 0.95,
                                 y_data: Optional[np.ndarray] = None,
                                 sample_names: Optional[List[str]] = None) -> go.Figure:
    """
    Create score plot (T scores) with confidence ellipse.

    Shows distribution of samples in latent variable space.
    Confidence ellipse identifies outliers/leverage samples.

    Parameters
    ----------
    model : Dict
        Fitted PLS model
    lv1, lv2 : int
        Latent variables to plot
    confidence : float
        Confidence level for ellipse (default: 0.95)
    y_data : Optional[np.ndarray]
        Response values to color by. If None, color by LV1
    sample_names : Optional[List[str]]
        Sample names for hover info

    Returns
    -------
    go.Figure
        Score plot with ellipse

    Interpretation:
    - Inside ellipse: typical samples
    - Outside ellipse: outliers/leverage samples
    - Cluster patterns: sample groups
    """
    T = model['T']

    lv1_idx = lv1 - 1
    lv2_idx = lv2 - 1

    if lv1_idx >= T.shape[1] or lv2_idx >= T.shape[1]:
        raise ValueError(f"Requested LV exceeds model components {T.shape[1]}")

    t1 = T[:, lv1_idx]
    t2 = T[:, lv2_idx]

    # Calculate confidence ellipse
    # Using chi-square distribution (standard for PLS score plots)
    n_samples = T.shape[0]

    cov_matrix = np.cov(np.column_stack([t1, t2]).T)
    mean = np.array([np.mean(t1), np.mean(t2)])

    # Eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))

    # Chi-square based scale (more standard for PLS)
    from scipy.stats import chi2
    p = 2  # 2D plot
    chi2_crit = chi2.ppf(confidence, p)
    scale = np.sqrt(chi2_crit * eigenvalues)

    # Ellipse
    theta = np.linspace(0, 2*np.pi, 100)
    ellipse_x_unit = np.cos(theta)
    ellipse_y_unit = np.sin(theta)

    # Rotate and scale ellipse
    rotation_matrix = np.array([
        [np.cos(np.radians(angle)), -np.sin(np.radians(angle))],
        [np.sin(np.radians(angle)), np.cos(np.radians(angle))]
    ])

    ellipse_points = np.column_stack([ellipse_x_unit * scale[0], ellipse_y_unit * scale[1]])
    ellipse_rotated = ellipse_points @ rotation_matrix.T
    ellipse_x = ellipse_rotated[:, 0] + mean[0]
    ellipse_y = ellipse_rotated[:, 1] + mean[1]

    # Create figure
    fig = go.Figure()

    # Color mapping
    if y_data is not None:
        color_vals = y_data
        colorscale = 'Viridis'
        colorbar_title = 'Response (y)'
    else:
        color_vals = t1
        colorscale = 'Blues'
        colorbar_title = f'LV{lv1}'

    # Sample names
    if sample_names is None:
        sample_names = [f"Sample {i+1}" for i in range(len(t1))]

    # Plot samples
    fig.add_trace(go.Scatter(
        x=t1,
        y=t2,
        mode='markers',
        marker=dict(
            size=8,
            color=color_vals,
            colorscale=colorscale,
            showscale=True,
            colorbar=dict(title=colorbar_title),
            line=dict(width=0.5, color='white')
        ),
        text=sample_names,
        hovertemplate='<b>%{text}</b><br>LV' + f'{lv1}' + ': %{x:.3f}<br>LV' + f'{lv2}' + ': %{y:.3f}<extra></extra>',
        name='Samples'
    ))

    # Add ellipse
    fig.add_trace(go.Scatter(
        x=ellipse_x,
        y=ellipse_y,
        mode='lines',
        line=dict(color='red', width=2, dash='dash'),
        name=f'{int(confidence*100)}% Confidence Ellipse',
        hoverinfo='skip'
    ))

    # Add center
    fig.add_trace(go.Scatter(
        x=[mean[0]],
        y=[mean[1]],
        mode='markers',
        marker=dict(size=10, color='red', symbol='x'),
        name='Center',
        hoverinfo='skip'
    ))

    # Reference lines
    fig.add_hline(y=0, line_dash="dash", line_color="rgba(0,0,0,0.2)")
    fig.add_vline(x=0, line_dash="dash", line_color="rgba(0,0,0,0.2)")

    fig.update_layout(
        title=f"Score Plot: LV{lv1} vs LV{lv2} (with {int(confidence*100)}% Confidence Ellipse)",
        xaxis_title=f"Latent Variable {lv1}",
        yaxis_title=f"Latent Variable {lv2}",
        hovermode='closest',
        width=700,           # ← Quadrato
        height=700,          # ← Quadrato
        template='plotly_white',
        xaxis=dict(
            scaleanchor="y",
            scaleratio=1,
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor='rgba(0,0,0,0.2)'
        ),
        yaxis=dict(
            scaleanchor="x",
            scaleratio=1,
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor='rgba(0,0,0,0.2)'
        ),
        legend=dict(
            x=0.02,              # ← Sinistra
            y=0.98,              # ← Alto
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='gray',
            borderwidth=1,
            font=dict(size=10)
        )
    )

    return fig


def plot_residuals_histogram(residuals: np.ndarray) -> go.Figure:
    """
    Create histogram of residuals with normality test.

    Checks if residuals are normally distributed (assumption for inference).

    Parameters
    ----------
    residuals : np.ndarray
        Residuals from prediction

    Returns
    -------
    go.Figure
        Histogram with fitted normal curve
    """
    from scipy.stats import norm, shapiro

    # Shapiro-Wilk test
    if len(residuals) >= 3:
        stat, p_value = shapiro(residuals)
    else:
        stat, p_value = np.nan, np.nan

    # Fit normal distribution
    mu = np.mean(residuals)
    sigma = np.std(residuals)
    x_norm = np.linspace(residuals.min(), residuals.max(), 100)
    y_norm = norm.pdf(x_norm, mu, sigma)

    fig = go.Figure()

    # Histogram
    fig.add_trace(go.Histogram(
        x=residuals,
        nbinsx=20,
        name='Residuals',
        marker=dict(color='skyblue', line=dict(color='navy', width=1)),
        histnorm=''
    ))

    # Normal curve (scaled to histogram)
    bin_width = (residuals.max() - residuals.min()) / 20
    y_norm_scaled = y_norm * len(residuals) * bin_width

    fig.add_trace(go.Scatter(
        x=x_norm,
        y=y_norm_scaled,
        mode='lines',
        name='Normal Distribution',
        line=dict(color='red', width=2)
    ))

    # Add text
    shapiro_text = f"Shapiro-Wilk p-value: {p_value:.4f}" if not np.isnan(p_value) else "N/A"
    normality_result = "✓ Normal" if (not np.isnan(p_value) and p_value > 0.05) else "✗ Non-normal"

    fig.add_annotation(
        text=f"Mean: {mu:.3f}<br>Std: {sigma:.3f}<br>{shapiro_text}<br>{normality_result}",
        xref="paper", yref="paper",
        x=0.98, y=0.97,
        showarrow=False,
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="black",
        borderwidth=1,
        xanchor="right",
        yanchor="top"
    )

    fig.update_layout(
        title="Distribution of Residuals (Normality Check)",
        xaxis_title="Residuals",
        yaxis_title="Frequency",
        hovermode='closest',
        template='plotly_white'
    )

    return fig


def plot_qq_plot(residuals: np.ndarray) -> go.Figure:
    """
    Create Q-Q plot for residuals.

    Compares residuals against theoretical normal distribution.
    Points on diagonal indicate normal distribution.

    Parameters
    ----------
    residuals : np.ndarray
        Residuals from prediction

    Returns
    -------
    go.Figure
        Q-Q plot
    """
    from scipy.stats import probplot

    (osm, osr), (slope, intercept, r) = probplot(residuals, dist="norm")

    fig = go.Figure()

    # Q-Q plot
    fig.add_trace(go.Scatter(
        x=osm,
        y=osr,
        mode='markers',
        marker=dict(size=6, color='blue'),
        name='Residuals',
        hovertemplate='Theoretical: %{x:.3f}<br>Observed: %{y:.3f}<extra></extra>'
    ))

    # Diagonal line (fitted)
    fig.add_trace(go.Scatter(
        x=osm,
        y=slope * osm + intercept,
        mode='lines',
        line=dict(color='red', width=2, dash='dash'),
        name='Fitted Line',
        hoverinfo='skip'
    ))

    # Add R² annotation
    fig.add_annotation(
        text=f"R² = {r**2:.4f}",
        xref="paper", yref="paper",
        x=0.02, y=0.98,
        showarrow=False,
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="black",
        borderwidth=1,
        xanchor="left",
        yanchor="top"
    )

    fig.update_layout(
        title="Q-Q Plot (Normality Assessment)",
        xaxis_title="Theoretical Quantiles",
        yaxis_title="Sample Quantiles",
        hovermode='closest',
        template='plotly_white',
        width=700,
        height=700,
        xaxis=dict(scaleanchor="y", scaleratio=1),
        yaxis=dict(scaleanchor="x", scaleratio=1)
    )

    return fig


def create_diagnostic_info_dict() -> Dict[str, str]:
    """
    Create dictionary with explanations for each diagnostic plot.

    Returns
    -------
    Dict[str, str]
        Explanations for diagnostics
    """
    return {
        'loading_plot': """
        **Loading Plot Explanation:**

        The loading plot shows the relationship between X variables and latent variables.

        - **X-axis/Y-axis**: Loadings on two latent variables (contributions)
        - **Distance from origin**: Importance of the variable
        - **Angle between vectors**: Correlation between variables
        - **Variables far from center**: Important for model
        - **Circle**: Unit circle reference (normalization)

        **Use for:**
        - Understanding which variables drive each latent variable
        - Identifying correlated variable groups
        - Validation of domain knowledge
        """,

        'vip': """
        **VIP (Variable Importance in Projection) Explanation:**

        VIP measures the contribution of each X variable to the PLS model predictions.

        **Interpretation:**
        - VIP > 1.0: Variable is important for predictions (KEEP)
        - VIP 0.5 - 1.0: Variable has borderline importance (CHECK)
        - VIP < 0.5: Variable contributes little (REMOVE)

        **Formula:** VIP = √[p × Σ(SS(b_h × w_jh)²) / Σ(SS(b_h × t_h)²)]

        **Use for:**
        - Variable selection
        - Feature importance ranking
        - Model simplification
        - Identifying key predictors
        """,

        'score_plot': """
        **Score Plot with Confidence Ellipse Explanation:**

        The score plot shows samples in latent variable space.

        - **X-axis/Y-axis**: Two latent variables (LV1, LV2)
        - **Each point**: One sample in model space
        - **Red ellipse**: Confidence region (e.g., 95%)
        - **Inside ellipse**: Normal/typical samples
        - **Outside ellipse**: Outliers or leverage samples

        **Use for:**
        - Detecting outliers
        - Identifying clusters
        - Checking model stability
        - Sample classification
        """,

        'residuals_histogram': """
        **Residuals Histogram Explanation:**

        Shows distribution of prediction errors (observed - predicted).

        - **Blue bars**: Frequency of residuals
        - **Red curve**: Theoretical normal distribution
        - **Shapiro-Wilk p-value**: Test of normality (p > 0.05 = normal)
        - **Mean**: Should be ≈ 0 (no bias)
        - **Std**: Prediction error magnitude

        **Ideal case:**
        - Symmetric distribution around 0
        - Bell-shaped curve (normal)
        - Shapiro-Wilk p > 0.05

        **Use for:**
        - Validating model assumptions
        - Checking for bias
        - Identifying systematic errors
        """,

        'qq_plot': """
        **Q-Q Plot Explanation:**

        Compares residuals to theoretical normal distribution.

        - **Blue points**: Sample residuals
        - **Red diagonal**: Theoretical normal line (y=x)
        - **Points on line**: Residuals follow normal distribution
        - **Points below line**: Negative deviation from normality
        - **Points above line**: Positive deviation from normality

        **Ideal case:**
        - All points on red diagonal line
        - No curved pattern
        - No outliers far from line

        **Deviations indicate:**
        - Heavy tails: Outliers present
        - S-shape: Different scales in tails
        - Systematic pattern: Model misspecification

        **Use for:**
        - Assessing normality assumption
        - Detecting outliers
        - Checking model fit quality
        """
    }


# Streamlit display helper
def display_diagnostic_with_explanation(fig, title: str, explanation_key: str) -> None:
    """
    Display diagnostic plot with expandable explanation.

    To use in Streamlit:
    ```python
    import streamlit as st
    from calibration_utils.pls_diagnostics import display_diagnostic_with_explanation

    fig = plot_loading_plot(model)
    display_diagnostic_with_explanation(fig, "Loading Plot", "loading_plot")
    ```
    """
    import streamlit as st

    st.markdown(f"### {title}")

    # Plot
    st.plotly_chart(fig, use_container_width=True)

    # Explanation
    explanations = create_diagnostic_info_dict()
    if explanation_key in explanations:
        with st.expander("📖 What does this plot mean?"):
            st.markdown(explanations[explanation_key])
