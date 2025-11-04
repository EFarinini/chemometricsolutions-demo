"""
Classification Visualization Functions
=======================================

Plotly-based visualization functions for classification including:
- Confusion matrix heatmaps
- Coomans plots for SIMCA/UNEQ
- Distance distribution plots
- Performance comparison charts
- Decision boundary visualization
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, Optional, List, Tuple
from .config import PLOT_COLORS, COOMANS_AXIS_LIMIT, COOMANS_THRESHOLD, COOMANS_CLIP_VALUE


def plot_confusion_matrix(
    cm: np.ndarray,
    classes: List[str],
    title: str = "Confusion Matrix",
    normalize: bool = False
) -> go.Figure:
    """
    Create an interactive confusion matrix heatmap.

    Parameters
    ----------
    cm : ndarray
        Confusion matrix
    classes : list of str
        Class labels
    title : str
        Plot title
    normalize : bool
        Whether to normalize by row (true class)

    Returns
    -------
    Figure
        Plotly figure
    """
    if normalize:
        cm_display = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        text_suffix = "%"
    else:
        cm_display = cm
        text_suffix = ""

    # Create text annotations
    text = [[f"{val:.1f}{text_suffix}" if normalize else f"{int(val)}"
             for val in row] for row in cm_display]

    fig = go.Figure(data=go.Heatmap(
        z=cm_display,
        x=[f"Predicted<br>{cls}" for cls in classes],
        y=[f"True {cls}" for cls in classes],
        text=text,
        texttemplate="%{text}",
        textfont={"size": 12},
        colorscale='Blues',
        showscale=True
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Predicted Class",
        yaxis_title="True Class",
        width=600,
        height=500,
        font=dict(size=11)
    )

    return fig


def plot_classification_metrics(
    metrics: Dict[str, any],
    title: str = "Classification Performance by Class"
) -> go.Figure:
    """
    Create bar chart of classification metrics per class.

    Parameters
    ----------
    metrics : dict
        Metrics dictionary from diagnostics
    title : str
        Plot title

    Returns
    -------
    Figure
        Plotly figure
    """
    class_metrics = metrics['class_metrics']
    classes = list(class_metrics.keys())

    # Extract metrics
    sensitivity = [class_metrics[cls]['sensitivity'] for cls in classes]
    specificity = [class_metrics[cls]['specificity'] for cls in classes]
    efficiency = [class_metrics[cls]['efficiency'] for cls in classes]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name='Sensitivity',
        x=classes,
        y=sensitivity,
        marker_color='rgb(55, 83, 109)'
    ))

    fig.add_trace(go.Bar(
        name='Specificity',
        x=classes,
        y=specificity,
        marker_color='rgb(26, 118, 255)'
    ))

    fig.add_trace(go.Bar(
        name='Efficiency',
        x=classes,
        y=efficiency,
        marker_color='rgb(50, 171, 96)'
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Class",
        yaxis_title="Percentage (%)",
        yaxis_range=[0, 105],
        barmode='group',
        width=800,
        height=500,
        legend=dict(x=0.01, y=0.99),
        font=dict(size=11)
    )

    return fig


def plot_coomans(
    dist_class1: np.ndarray,
    dist_class2: np.ndarray,
    y_true: np.ndarray,
    crit_dist1: float,
    crit_dist2: float,
    class_names: Optional[List[str]] = None,
    sample_names: Optional[List[str]] = None,
    title: str = 'Coomans Plot',
    fig_size: Tuple[int, int] = (700, 700),
    normalize: bool = False
) -> go.Figure:
    """
    Create Coomans plot for SIMCA/UNEQ two-class visualization.

    Plots distances to two class models with threshold lines. Optional normalization
    by critical distances. Square markers, with reference lines showing acceptance regions.

    Parameters
    ----------
    dist_class1 : ndarray of shape (n_samples,)
        Distances to first class model
    dist_class2 : ndarray of shape (n_samples,)
        Distances to second class model
    y_true : ndarray of shape (n_samples,)
        True class labels for coloring points
    crit_dist1 : float
        Critical distance (threshold) for class 1
    crit_dist2 : float
        Critical distance (threshold) for class 2
    class_names : list of str, optional, default=['Class 1', 'Class 2']
        Names of the two classes
    sample_names : list of str, optional
        Names/labels for samples (row names from DataFrame or 1-based indices).
        If None, uses numeric indices starting from 0
    title : str, default='Coomans Plot'
        Plot title
    fig_size : tuple of int, default=(700, 700)
        Figure size (width, height) in pixels
    normalize : bool, default=False
        If True, normalize distances by critical distances (d_norm = d/crit_d)
        and clamp values >6 to random [4,6]

    Returns
    -------
    Figure
        Plotly figure with square layout

    Notes
    -----
    - Normalize mode: d_norm = d/crit_d, crit_dist becomes 1.0
    - Axis range: [0, 6]
    - Lines: vertical (x=crit), horizontal (y=crit), diagonal (y=x)
    - Square markers, size=10
    - Hover shows sample index and distances

    Reference
    ---------
    uneq.m lines 409-445

    Examples
    --------
    >>> dist1 = np.random.rand(100) * 3
    >>> dist2 = np.random.rand(100) * 3
    >>> y = np.array([0]*50 + [1]*50)
    >>> fig = plot_coomans(dist1, dist2, y, 1.5, 1.5, normalize=True)
    >>> fig.show()
    """
    if class_names is None:
        class_names = ['Class 1', 'Class 2']

    # Copy arrays to avoid modifying originals
    d1 = dist_class1.copy()
    d2 = dist_class2.copy()

    # Ensure y_true is numpy array for proper masking
    if not isinstance(y_true, np.ndarray):
        y_true = np.array(y_true)

    # Normalize if requested
    if normalize:
        d1 = d1 / crit_dist1
        d2 = d2 / crit_dist2

        # Clamp values >6 to random [4, 6] for better visualization
        mask1 = d1 > 6
        if mask1.any():
            d1[mask1] = np.random.uniform(4, 6, size=mask1.sum())

        mask2 = d2 > 6
        if mask2.any():
            d2[mask2] = np.random.uniform(4, 6, size=mask2.sum())

        # Set critical distances to 1.0 after normalization
        crit_dist1_plot = 1.0
        crit_dist2_plot = 1.0
    else:
        crit_dist1_plot = crit_dist1
        crit_dist2_plot = crit_dist2

    # DEBUG
    import sys
    print(f"DEBUG plot_coomans INSIDE:", file=sys.stderr)
    print(f"  y_true type: {type(y_true)}, value: {y_true[:10]}", file=sys.stderr)
    print(f"  unique_classes: {np.unique(y_true)}", file=sys.stderr)
    print(f"  d1 shape: {d1.shape}, min={d1.min()}, max={d1.max()}", file=sys.stderr)
    print(f"  d2 shape: {d2.shape}, min={d2.min()}, max={d2.max()}", file=sys.stderr)

    # Create figure
    fig = go.Figure()

    # DEBUG After creating figure
    print(f"  Figure created", file=sys.stderr)

    # Plot points by true class
    unique_classes = np.unique(y_true)
    for i, cls in enumerate(unique_classes):
        mask = y_true == cls
        print(f"    Class {cls}: mask.sum()={mask.sum()}, indices={np.where(mask)[0][:3]}", file=sys.stderr)
        color = PLOT_COLORS[i % len(PLOT_COLORS)]

        # Create hover text with sample indices or names
        sample_indices = np.where(mask)[0]
        hover_text = []

        for local_idx in sample_indices:
            # local_idx are indices in the filtered arrays (d1, d2, y_true)
            # Map to sample_names if provided
            if sample_names is not None and local_idx < len(sample_names):
                sample_label = sample_names[local_idx]
            else:
                sample_label = str(local_idx)

            hover_text.append(
                f"Sample {sample_label}<br>"
                f"Dist to {class_names[0]}: {d1[local_idx]:.3f}<br>"
                f"Dist to {class_names[1]}: {d2[local_idx]:.3f}"
            )

        fig.add_trace(go.Scatter(
            x=d1[mask],
            y=d2[mask],
            mode='markers',
            name=f'Class {cls}',
            marker=dict(
                size=10,
                color=color,
                symbol='square',
                line=dict(width=1, color='white')
            ),
            text=hover_text,
            hovertemplate='%{text}<extra></extra>'
        ))

    # Add reference lines
    # Vertical line (x = crit_dist1)
    fig.add_vline(
        x=crit_dist1_plot,
        line_dash="dash",
        line_color="gray",
        annotation_text=f"{class_names[0]} threshold",
        annotation_position="top"
    )

    # Horizontal line (y = crit_dist2)
    fig.add_hline(
        y=crit_dist2_plot,
        line_dash="dash",
        line_color="gray",
        annotation_text=f"{class_names[1]} threshold",
        annotation_position="right"
    )

    # Diagonal line (y = x)
    fig.add_trace(go.Scatter(
        x=[0, 6],
        y=[0, 6],
        mode='lines',
        line=dict(color='gray', dash='dot', width=1),
        showlegend=False,
        hoverinfo='skip',
        name='Equal distance'
    ))

    # Update layout - SQUARE with equal axes
    axis_title_1 = (f"Normalized Distance to {class_names[0]}" if normalize
                    else f"Distance to {class_names[0]}")
    axis_title_2 = (f"Normalized Distance to {class_names[1]}" if normalize
                    else f"Distance to {class_names[1]}")

    # Calculate consistent axis range
    if normalize:
        axis_range = [0, 6]
    else:
        # Calculate max distance including critical thresholds
        max_dist = max(d1.max(), d2.max(), crit_dist1_plot, crit_dist2_plot) * 1.1
        axis_range = [0, max_dist]

    fig.update_layout(
        title=title,
        xaxis=dict(
            title=axis_title_1,
            range=axis_range,
            showgrid=True,
            zeroline=True,
            gridcolor='lightgray',
            scaleanchor="y",  # Force square aspect ratio
            scaleratio=1
        ),
        yaxis=dict(
            title=axis_title_2,
            range=axis_range,
            showgrid=True,
            zeroline=True,
            gridcolor='lightgray',
            scaleanchor="x",  # Enforce square aspect ratio from both axes
            scaleratio=1
        ),
        width=fig_size[0],
        height=fig_size[1],
        plot_bgcolor='white',
        showlegend=True,
        legend=dict(x=0.02, y=0.98, xanchor='left', yanchor='top'),
        font=dict(size=11)
    )

    # Adjust axis ranges for better visualization - SQUARE plot
    axis_limit = max(
        d1.max(), d2.max(),
        crit_dist1_plot * 1.2, crit_dist2_plot * 1.2
    )

    if normalize:
        axis_range = [0, 6]
    else:
        axis_range = [0, axis_limit]

    fig.update_xaxes(range=axis_range, scaleanchor="y", scaleratio=1)
    fig.update_yaxes(range=axis_range, scaleanchor="x", scaleratio=1)

    return fig


def plot_distance_distributions(
    distances: Dict[any, np.ndarray],
    y_true: np.ndarray,
    selected_class: any,
    threshold: Optional[float] = None,
    title: Optional[str] = None
) -> go.Figure:
    """
    Plot distribution of distances to a class model.

    Parameters
    ----------
    distances : dict
        Dictionary of distances
    y_true : ndarray
        True labels
    selected_class : any
        Class to plot distances for
    threshold : float, optional
        Acceptance threshold
    title : str, optional
        Plot title

    Returns
    -------
    Figure
        Plotly figure
    """
    if title is None:
        title = f"Distance Distribution for Class {selected_class}"

    dist_cls = distances[selected_class]

    fig = go.Figure()

    # Plot distribution for own class
    own_class_mask = y_true == selected_class
    fig.add_trace(go.Histogram(
        x=dist_cls[own_class_mask],
        name=f'Class {selected_class} (Own)',
        opacity=0.7,
        marker_color='blue',
        nbinsx=30
    ))

    # Plot distribution for other classes
    other_class_mask = ~own_class_mask
    if other_class_mask.sum() > 0:
        fig.add_trace(go.Histogram(
            x=dist_cls[other_class_mask],
            name=f'Other Classes',
            opacity=0.7,
            marker_color='red',
            nbinsx=30
        ))

    # Add threshold line
    if threshold is not None:
        fig.add_vline(x=threshold, line_dash="dash", line_color="green",
                      annotation_text="Threshold",
                      annotation_position="top right")

    fig.update_layout(
        title=title,
        xaxis_title="Distance",
        yaxis_title="Frequency",
        barmode='overlay',
        width=800,
        height=500,
        legend=dict(x=0.7, y=0.98),
        font=dict(size=11)
    )

    return fig


def plot_knn_performance(
    cv_results: Dict[int, Dict[str, any]],
    metric: str = 'accuracy'
) -> go.Figure:
    """
    Plot kNN performance vs k value.

    Parameters
    ----------
    cv_results : dict
        Results from cross_validate_knn
    metric : str
        Metric to plot

    Returns
    -------
    Figure
        Plotly figure
    """
    k_values = []
    metric_values = []

    for k, results in sorted(cv_results.items()):
        k_values.append(k)
        metric_values.append(results['metrics'][metric])

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=k_values,
        y=metric_values,
        mode='lines+markers',
        line=dict(color='rgb(55, 83, 109)', width=3),
        marker=dict(size=10, color='rgb(26, 118, 255)'),
        name=metric.replace('_', ' ').title()
    ))

    # Mark best k
    best_idx = np.argmax(metric_values)
    fig.add_trace(go.Scatter(
        x=[k_values[best_idx]],
        y=[metric_values[best_idx]],
        mode='markers',
        marker=dict(size=15, color='red', symbol='star'),
        name=f'Best k = {k_values[best_idx]}',
        showlegend=True
    ))

    fig.update_layout(
        title=f"kNN Performance: {metric.replace('_', ' ').title()} vs k",
        xaxis_title="Number of Neighbors (k)",
        yaxis_title=metric.replace('_', ' ').title() + " (%)",
        width=800,
        height=500,
        legend=dict(x=0.7, y=0.15),
        font=dict(size=11),
        xaxis=dict(dtick=1)
    )

    return fig


def plot_model_comparison(
    comparison_df: pd.DataFrame,
    title: str = "Model Performance Comparison"
) -> go.Figure:
    """
    Create grouped bar chart comparing multiple models.

    Parameters
    ----------
    comparison_df : DataFrame
        Comparison table from diagnostics.compare_models
    title : str
        Plot title

    Returns
    -------
    Figure
        Plotly figure
    """
    models = comparison_df['Model'].tolist()

    fig = go.Figure()

    metrics = ['Accuracy', 'Avg Sensitivity', 'Avg Specificity', 'Avg Efficiency']
    colors = ['rgb(55, 83, 109)', 'rgb(26, 118, 255)', 'rgb(50, 171, 96)', 'rgb(255, 150, 50)']

    for metric, color in zip(metrics, colors):
        if metric in comparison_df.columns:
            fig.add_trace(go.Bar(
                name=metric,
                x=models,
                y=comparison_df[metric],
                marker_color=color
            ))

    fig.update_layout(
        title=title,
        xaxis_title="Model",
        yaxis_title="Performance (%)",
        yaxis_range=[0, 105],
        barmode='group',
        width=900,
        height=500,
        legend=dict(x=0.01, y=0.99),
        font=dict(size=11)
    )

    return fig


def plot_decision_boundary_2d(
    X: np.ndarray,
    y: np.ndarray,
    model: any,
    predict_fn: callable,
    feature_indices: Tuple[int, int] = (0, 1),
    feature_names: Optional[List[str]] = None,
    resolution: int = 100,
    title: str = "Decision Boundary"
) -> go.Figure:
    """
    Plot decision boundary for 2D feature space.

    Parameters
    ----------
    X : ndarray
        Feature matrix
    y : ndarray
        True labels
    model : any
        Trained model
    predict_fn : callable
        Prediction function (model, X) -> predictions
    feature_indices : tuple of int
        Indices of two features to plot
    feature_names : list of str, optional
        Feature names
    resolution : int
        Grid resolution
    title : str
        Plot title

    Returns
    -------
    Figure
        Plotly figure
    """
    # Extract 2D features
    X_2d = X[:, feature_indices]
    feat_idx_1, feat_idx_2 = feature_indices

    if feature_names is None:
        feat_name_1 = f"Feature {feat_idx_1 + 1}"
        feat_name_2 = f"Feature {feat_idx_2 + 1}"
    else:
        feat_name_1 = feature_names[feat_idx_1]
        feat_name_2 = feature_names[feat_idx_2]

    # Create grid
    x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
    y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution)
    )

    # Reconstruct full feature space with mean values for other features
    grid_points = np.zeros((resolution * resolution, X.shape[1]))
    for i in range(X.shape[1]):
        if i == feat_idx_1:
            grid_points[:, i] = xx.ravel()
        elif i == feat_idx_2:
            grid_points[:, i] = yy.ravel()
        else:
            grid_points[:, i] = X[:, i].mean()

    # Predict on grid
    Z, _ = predict_fn(grid_points, model)
    Z = Z.reshape(xx.shape)

    # Convert labels to numeric for contour
    unique_classes = np.unique(y)
    class_to_num = {cls: i for i, cls in enumerate(unique_classes)}
    Z_numeric = np.array([[class_to_num[val] for val in row] for row in Z])

    fig = go.Figure()

    # Add contour for decision boundary
    fig.add_trace(go.Contour(
        x=xx[0],
        y=yy[:, 0],
        z=Z_numeric,
        colorscale='Viridis',
        opacity=0.3,
        showscale=False,
        hoverinfo='skip'
    ))

    # Add data points
    for i, cls in enumerate(unique_classes):
        mask = y == cls
        color = PLOT_COLORS[i % len(PLOT_COLORS)]

        fig.add_trace(go.Scatter(
            x=X_2d[mask, 0],
            y=X_2d[mask, 1],
            mode='markers',
            name=f'Class {cls}',
            marker=dict(
                size=8,
                color=color,
                symbol='circle',
                line=dict(width=1, color='white')
            )
        ))

    fig.update_layout(
        title=title,
        xaxis_title=feat_name_1,
        yaxis_title=feat_name_2,
        width=700,
        height=700,
        legend=dict(x=0.02, y=0.98),
        font=dict(size=11)
    )

    return fig


def plot_class_separation(
    X: np.ndarray,
    y: np.ndarray,
    feature_idx: int = 0,
    feature_name: Optional[str] = None,
    title: Optional[str] = None
) -> go.Figure:
    """
    Plot feature value distributions by class (violin plot).

    Parameters
    ----------
    X : ndarray
        Feature matrix
    y : ndarray
        Class labels
    feature_idx : int
        Feature index to plot
    feature_name : str, optional
        Feature name
    title : str, optional
        Plot title

    Returns
    -------
    Figure
        Plotly figure
    """
    if feature_name is None:
        feature_name = f"Feature {feature_idx + 1}"

    if title is None:
        title = f"Class Separation: {feature_name}"

    fig = go.Figure()

    unique_classes = np.unique(y)
    for cls in unique_classes:
        mask = y == cls
        fig.add_trace(go.Violin(
            y=X[mask, feature_idx],
            name=f'Class {cls}',
            box_visible=True,
            meanline_visible=True
        ))

    fig.update_layout(
        title=title,
        yaxis_title=feature_name,
        xaxis_title="Class",
        width=800,
        height=500,
        font=dict(size=11)
    )

    return fig


def plot_mahalanobis_distances(
    distances_matrix: np.ndarray,
    y_true: np.ndarray,
    class_names: Optional[List[str]] = None,
    title: str = "Mahalanobis Distances",
    fig_size: Tuple[int, int] = (1000, 600),
    colorscale: str = 'Viridis'
) -> go.Figure:
    """
    Create heatmap of Mahalanobis distances with samples sorted by class.

    Rows are sorted by class (same-class samples grouped together) with white
    horizontal lines at class boundaries. Useful for visualizing class separation.

    Parameters
    ----------
    distances_matrix : ndarray of shape (n_samples, n_classes)
        Distance matrix where distances_matrix[i, j] is distance from sample i to class j
    y_true : ndarray of shape (n_samples,)
        True class labels for each sample
    class_names : list of str, optional
        Class names for column labels. If None, uses y_true unique values
    title : str, default="Mahalanobis Distances"
        Plot title
    fig_size : tuple of int, default=(1000, 600)
        Figure size (width, height) in pixels
    colorscale : str, default='Viridis'
        Plotly colorscale name (e.g., 'Viridis', 'RdBu', 'Blues')

    Returns
    -------
    Figure
        Plotly figure with heatmap

    Notes
    -----
    - Samples are sorted by class for better visualization
    - White horizontal lines mark class boundaries
    - Colorbar shows distance scale (blue→red for small→large)
    - Hover shows sample index, class, and distance value

    Reference
    ---------
    CL_plot_mahalanobis.r

    Examples
    --------
    >>> distances = np.random.rand(100, 3)
    >>> y_true = np.array([0]*40 + [1]*30 + [2]*30)
    >>> fig = plot_mahalanobis_distances(distances, y_true)
    >>> fig.show()
    """
    # Get unique classes and sort order
    classes = np.unique(y_true)
    if class_names is None:
        class_names = [str(cls) for cls in classes]

    n_classes = len(classes)

    # Sort samples by class
    sort_indices = []
    class_boundaries = [0]  # Track where each class starts

    for cls in classes:
        cls_indices = np.where(y_true == cls)[0]
        sort_indices.extend(cls_indices)
        class_boundaries.append(len(sort_indices))

    sort_indices = np.array(sort_indices)

    # Sort distance matrix and labels
    distances_sorted = distances_matrix[sort_indices]
    y_sorted = y_true[sort_indices]

    # Create sample labels (with class info)
    sample_labels = [f"Sample {idx} (Class {y_true[idx]})"
                     for idx in sort_indices]

    # Create hover text
    hover_text = []
    for i, idx in enumerate(sort_indices):
        row_text = []
        for j, cls_name in enumerate(class_names):
            row_text.append(
                f"Sample: {idx}<br>"
                f"True Class: {y_true[idx]}<br>"
                f"Distance to {cls_name}: {distances_sorted[i, j]:.3f}"
            )
        hover_text.append(row_text)

    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=distances_sorted,
        x=class_names,
        y=list(range(len(sort_indices))),  # Use numeric indices for y-axis
        colorscale=colorscale,
        colorbar=dict(title="Distance"),
        hovertemplate='%{customdata}<extra></extra>',
        customdata=hover_text,
        showscale=True
    ))

    # Add white horizontal lines at class boundaries
    shapes = []
    for boundary in class_boundaries[1:-1]:  # Skip first (0) and last (n_samples)
        shapes.append(dict(
            type='line',
            x0=-0.5,
            x1=n_classes - 0.5,
            y0=boundary - 0.5,
            y1=boundary - 0.5,
            line=dict(color='white', width=3)
        ))

    # Add annotations for class regions on the left
    annotations = []
    for i, cls in enumerate(classes):
        start_idx = class_boundaries[i]
        end_idx = class_boundaries[i + 1]
        mid_idx = (start_idx + end_idx) / 2

        annotations.append(dict(
            x=-0.15,
            y=mid_idx,
            text=f"<b>Class {cls}</b>",
            xref='paper',
            yref='y',
            showarrow=False,
            font=dict(size=11, color='black'),
            xanchor='right'
        ))

    fig.update_layout(
        title=title,
        xaxis=dict(
            title="Target Class",
            side='top',
            tickfont=dict(size=11)
        ),
        yaxis=dict(
            title="Test Samples (sorted by true class)",
            autorange='reversed',  # Top to bottom
            showticklabels=False,  # Hide individual sample labels
            tickfont=dict(size=8)
        ),
        width=fig_size[0],
        height=fig_size[1],
        shapes=shapes,
        annotations=annotations,
        font=dict(size=11),
        plot_bgcolor='white'
    )

    return fig


def plot_knn_neighbors(
    X_test: np.ndarray,
    y_pred: np.ndarray,
    y_true: np.ndarray,
    k: int,
    class_colors: Optional[Dict] = None,
    title: str = 'kNN Classification',
    fig_size: Tuple[int, int] = (900, 700),
    compute_decision_regions: bool = False
) -> go.Figure:
    """
    Visualize kNN classification results in 2D space.

    If X_test has >2 features, applies PCA to reduce to 2D. Shows correctly
    classified samples as circles, misclassified as X markers. Optional decision
    region background.

    Parameters
    ----------
    X_test : ndarray of shape (n_samples, n_features)
        Test data
    y_pred : ndarray of shape (n_samples,)
        Predicted class labels
    y_true : ndarray of shape (n_samples,)
        True class labels
    k : int
        Number of neighbors used in kNN
    class_colors : dict, optional
        Mapping of class labels to colors. If None, uses default PLOT_COLORS
    title : str, default='kNN Classification'
        Plot title
    fig_size : tuple of int, default=(900, 700)
        Figure size (width, height) in pixels
    compute_decision_regions : bool, default=False
        If True, compute and show decision boundary as contour background
        (requires fit_knn and predict_knn functions - not implemented here)

    Returns
    -------
    Figure
        Plotly figure with 2D scatter plot

    Notes
    -----
    - If n_features > 2: applies PCA reduction to 2D
    - Correct predictions: circle markers
    - Misclassifications: X markers
    - Accuracy displayed in title annotation
    - Hover shows sample index, predicted and true class

    Reference
    ---------
    CL_plot_KNN.R

    Examples
    --------
    >>> from sklearn.decomposition import PCA
    >>> X = np.random.randn(100, 5)
    >>> y_pred = np.array([0]*50 + [1]*50)
    >>> y_true = np.array([0]*45 + [1]*5 + [0]*5 + [1]*45)
    >>> fig = plot_knn_neighbors(X, y_pred, y_true, k=3)
    >>> fig.show()
    """
    n_samples, n_features = X_test.shape

    # Reduce to 2D if needed
    if n_features > 2:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X_test)
        explained_var = pca.explained_variance_ratio_
        xlabel = f"PC1 ({explained_var[0]*100:.1f}% var)"
        ylabel = f"PC2 ({explained_var[1]*100:.1f}% var)"
    elif n_features == 2:
        X_2d = X_test
        xlabel = "Feature 1"
        ylabel = "Feature 2"
    else:
        raise ValueError("X_test must have at least 2 features")

    # Determine correct vs incorrect predictions
    correct_mask = y_pred == y_true
    incorrect_mask = ~correct_mask

    # Calculate accuracy
    accuracy = np.mean(correct_mask) * 100

    # Create figure
    fig = go.Figure()

    # Get unique classes
    unique_classes = np.unique(np.concatenate([y_pred, y_true]))

    # Use default colors if not provided
    if class_colors is None:
        class_colors = {cls: PLOT_COLORS[i % len(PLOT_COLORS)]
                       for i, cls in enumerate(unique_classes)}

    # Plot correct predictions (circles) by predicted class
    for cls in unique_classes:
        mask = correct_mask & (y_pred == cls)
        if mask.sum() > 0:
            sample_indices = np.where(mask)[0]
            hover_text = [
                f"Sample: {idx}<br>"
                f"Predicted: {y_pred[idx]}<br>"
                f"True: {y_true[idx]}"
                for idx in sample_indices
            ]

            fig.add_trace(go.Scatter(
                x=X_2d[mask, 0],
                y=X_2d[mask, 1],
                mode='markers',
                name=f'Class {cls} (Correct)',
                marker=dict(
                    size=10,
                    color=class_colors.get(cls, 'gray'),
                    symbol='circle',
                    line=dict(width=1, color='white')
                ),
                text=hover_text,
                hovertemplate='%{text}<extra></extra>'
            ))

    # Plot incorrect predictions (X markers) by predicted class
    for cls in unique_classes:
        mask = incorrect_mask & (y_pred == cls)
        if mask.sum() > 0:
            sample_indices = np.where(mask)[0]
            hover_text = [
                f"Sample: {idx}<br>"
                f"Predicted: {y_pred[idx]}<br>"
                f"True: {y_true[idx]} ❌"
                for idx in sample_indices
            ]

            fig.add_trace(go.Scatter(
                x=X_2d[mask, 0],
                y=X_2d[mask, 1],
                mode='markers',
                name=f'Class {cls} (Error)',
                marker=dict(
                    size=12,
                    color=class_colors.get(cls, 'gray'),
                    symbol='x',
                    line=dict(width=2)
                ),
                text=hover_text,
                hovertemplate='%{text}<extra></extra>'
            ))

    # Add text annotation for accuracy and k
    fig.add_annotation(
        xref='paper', yref='paper',
        x=0.02, y=0.98,
        text=f"<b>Accuracy: {accuracy:.1f}%<br>k = {k}</b>",
        showarrow=False,
        font=dict(size=14, color='black'),
        bgcolor='rgba(255, 255, 255, 0.8)',
        bordercolor='black',
        borderwidth=1,
        xanchor='left',
        yanchor='top'
    )

    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        width=fig_size[0],
        height=fig_size[1],
        plot_bgcolor='white',
        legend=dict(x=1.02, y=1, xanchor='left'),
        font=dict(size=11),
        showlegend=True
    )

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')

    return fig


def plot_classification_report(
    metrics_dict: Dict[str, any],
    class_names: List[str],
    title: str = 'Classification Report',
    fig_size: Tuple[int, int] = (800, 500),
    include_metrics: List[str] = None
) -> go.Figure:
    """
    Create heatmap visualization of classification metrics per class.

    Builds a matrix with classes as rows and metrics as columns, plus an
    average row. Uses diverging colorscale (red-yellow-green) centered at 50%.

    Parameters
    ----------
    metrics_dict : dict
        Dictionary containing classification metrics with keys:
        - 'sensitivity_per_class': dict mapping class to sensitivity (%)
        - 'specificity_per_class': dict mapping class to specificity (%)
        - 'precision_per_class': dict mapping class to precision (%)
        - 'f1_per_class': dict mapping class to F1-score (%)
        - 'accuracy': overall accuracy (%) - optional, for average row
    class_names : list of str
        List of class names (in order)
    title : str, default='Classification Report'
        Plot title
    fig_size : tuple of int, default=(800, 500)
        Figure size (width, height) in pixels
    include_metrics : list of str, default=['Sensitivity', 'Specificity', 'Precision', 'F1']
        Metrics to include in the report

    Returns
    -------
    Figure
        Plotly figure with heatmap

    Notes
    -----
    - Matrix rows: [classes] + ['Average']
    - Matrix values: 0-100 (percentages)
    - Colorscale: 'RdYlGn' (red-yellow-green) with zmid=50
    - Average row: mean of each metric across classes
    - Bold text for Average row
    - Cell text: formatted percentages

    Reference
    ---------
    Standard ML classification reporting

    Examples
    --------
    >>> metrics = {
    ...     'sensitivity_per_class': {0: 85.0, 1: 90.0, 2: 80.0},
    ...     'specificity_per_class': {0: 92.0, 1: 88.0, 2: 95.0},
    ...     'precision_per_class': {0: 87.0, 1: 91.0, 2: 82.0},
    ...     'f1_per_class': {0: 86.0, 1: 90.5, 2: 81.0},
    ...     'accuracy': 85.0
    ... }
    >>> fig = plot_classification_report(metrics, ['Class A', 'Class B', 'Class C'])
    >>> fig.show()
    """
    if include_metrics is None:
        include_metrics = ['Sensitivity', 'Specificity', 'Precision', 'F1']

    n_classes = len(class_names)
    n_metrics = len(include_metrics)

    # Mapping from metric names to dict keys
    metric_key_map = {
        'Sensitivity': 'sensitivity_per_class',
        'Specificity': 'specificity_per_class',
        'Precision': 'precision_per_class',
        'F1': 'f1_per_class'
    }

    # Build matrix: rows = classes + Average, cols = metrics
    matrix = np.zeros((n_classes + 1, n_metrics))

    for j, metric_name in enumerate(include_metrics):
        dict_key = metric_key_map.get(metric_name)
        if dict_key and dict_key in metrics_dict:
            metric_values = metrics_dict[dict_key]
            for i, cls in enumerate(class_names):
                matrix[i, j] = metric_values.get(cls, 0.0)

            # Average row: mean across classes
            matrix[n_classes, j] = np.mean(matrix[:n_classes, j])

    # Row labels: classes + Average
    row_labels = class_names + ['<b>Average</b>']

    # Create text annotations with bold for Average row
    text_annotations = []
    for i in range(n_classes + 1):
        row_text = []
        for j in range(n_metrics):
            val = matrix[i, j]
            if i == n_classes:  # Average row
                row_text.append(f"<b>{val:.1f}%</b>")
            else:
                row_text.append(f"{val:.1f}%")
        text_annotations.append(row_text)

    # Create hover text
    hover_text = []
    for i, row_label in enumerate(row_labels):
        row_hover = []
        for j, metric_name in enumerate(include_metrics):
            row_hover.append(
                f"Class: {row_label.replace('<b>', '').replace('</b>', '')}<br>"
                f"Metric: {metric_name}<br>"
                f"Value: {matrix[i, j]:.1f}%"
            )
        hover_text.append(row_hover)

    # Create heatmap with diverging colorscale
    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=include_metrics,
        y=row_labels,
        text=text_annotations,
        texttemplate='%{text}',
        textfont={"size": 12},
        colorscale='RdYlGn',  # Red-Yellow-Green
        zmid=50,  # Center diverging scale at 50%
        zmin=0,
        zmax=100,
        colorbar=dict(title="Percentage (%)", ticksuffix="%"),
        hovertemplate='%{customdata}<extra></extra>',
        customdata=hover_text,
        showscale=True
    ))

    fig.update_layout(
        title=title,
        xaxis=dict(
            title="Metrics",
            side='top',
            tickfont=dict(size=11)
        ),
        yaxis=dict(
            title="Classes",
            tickfont=dict(size=11)
        ),
        width=fig_size[0],
        height=fig_size[1],
        font=dict(size=11),
        plot_bgcolor='white'
    )

    return fig
