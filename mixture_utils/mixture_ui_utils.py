"""
Mixture Design UI Utilities
Streamlit UI helper components for mixture design interface
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go


def show_constraint_editor_ui():
    """
    Show constraint editor UI (stub implementation)

    Returns:
        list of constraint dicts
    """
    st.markdown("### ⚙️ Constraint Editor")
    st.info("Constraint editor UI - to be fully implemented")

    # Placeholder
    constraints = []

    return constraints


def show_pseudo_component_ui():
    """
    Show pseudo-component transformation UI (stub implementation)

    Returns:
        dict with pseudo-component config
    """
    st.markdown("### 🔄 Pseudo-Component Transformation")
    st.info("Pseudo-component UI - to be fully implemented")

    # Placeholder
    pseudo_config = {}

    return pseudo_config


def show_design_selection_ui():
    """
    Show design selection UI (stub implementation)

    Returns:
        selected design type
    """
    st.markdown("### 🎯 Design Selection")

    design_type = st.radio(
        "Design Type",
        ["Standard Simplex Centroid", "D-Optimal"],
        key="design_type_selector"
    )

    return design_type


def show_model_formula_builder():
    """
    Show model formula builder UI (stub implementation)

    Returns:
        model formula string
    """
    st.markdown("### 📐 Model Formula")

    degree = st.radio(
        "Scheffe Polynomial Degree",
        ["linear", "quadratic", "cubic"],
        index=1,
        key="model_degree_selector"
    )

    if degree == "linear":
        st.info("Linear model: Y = β₁X₁ + β₂X₂ + β₃X₃")
    elif degree == "quadratic":
        st.info("Quadratic model: Linear + β₁₂X₁X₂ + β₁₃X₁X₃ + β₂₃X₂X₃")
    else:
        st.info("Cubic model: Quadratic + β₁₂₃X₁X₂X₃")

    return degree


def show_design_summary_ui(mixture_design_matrix, pseudo_component_config=None, constraints=None):
    """
    Show design summary UI

    Args:
        mixture_design_matrix: design DataFrame
        pseudo_component_config: optional config
        constraints: optional constraint list
    """
    st.markdown("### 📊 Design Summary")

    if mixture_design_matrix is not None:
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Experiments", len(mixture_design_matrix))

        with col2:
            st.metric("Components", mixture_design_matrix.shape[1])

        with col3:
            # Verify sums
            row_sums = mixture_design_matrix.sum(axis=1)
            all_valid = np.allclose(row_sums, 1.0, atol=1e-10)
            st.metric("Valid", "✓" if all_valid else "✗")

        st.dataframe(mixture_design_matrix, use_container_width=True)

    else:
        st.info("No design matrix available")


def plot_ternary_design(design_matrix, design_type='simplex_centroid',
                        response_values=None, constraints=None):
    """
    Plot mixture design on ternary plot (3 components only)

    Args:
        design_matrix: pd.DataFrame with 3 components
        design_type: 'simplex_centroid' or 'd_optimal'
        response_values: optional array of responses for coloring
        constraints: optional constraint info

    Returns:
        plotly Figure
    """
    if design_matrix.shape[1] != 3:
        raise ValueError(f"Ternary plot requires 3 components, got {design_matrix.shape[1]}")

    component_names = design_matrix.columns.tolist()

    # Create ternary plot
    fig = go.Figure()

    # Determine marker properties
    if response_values is not None:
        marker_color = response_values
        colorbar = dict(title="Response")
        showscale = True
    else:
        marker_color = 'blue'
        colorbar = None
        showscale = False

    # Add experimental points
    hover_text = []
    for idx, row in design_matrix.iterrows():
        text = f"<b>Point {idx+1}</b><br>"
        for comp in component_names:
            text += f"{comp}: {row[comp]:.4f}<br>"
        if response_values is not None and idx < len(response_values):
            text += f"Response: {response_values[idx]:.4f}"
        hover_text.append(text)

    fig.add_trace(go.Scatterternary(
        a=design_matrix.iloc[:, 0],
        b=design_matrix.iloc[:, 1],
        c=design_matrix.iloc[:, 2],
        mode='markers+text',
        marker=dict(
            size=12,
            color=marker_color,
            colorscale='Viridis' if showscale else None,
            showscale=showscale,
            colorbar=colorbar,
            line=dict(width=2, color='white')
        ),
        text=[f"{i+1}" for i in range(len(design_matrix))],
        textposition='top center',
        textfont=dict(size=10, color='black'),
        hovertext=hover_text,
        hoverinfo='text',
        name='Design Points'
    ))

    # Title based on design type
    title = f'Mixture Design - {design_type.replace("_", " ").title()}'

    # Note: Plotly ternary axes do NOT support 'max' property
    # The max is automatically determined from sum=1
    fig.update_layout(
        title=title,
        ternary=dict(
            sum=1,
            aaxis=dict(
                title=component_names[0],
                min=0,
                tickformat='.2f',
                gridcolor='lightgray',
                showline=True,
                linewidth=2,
                linecolor='black'
            ),
            baxis=dict(
                title=component_names[1],
                min=0,
                tickformat='.2f',
                gridcolor='lightgray',
                showline=True,
                linewidth=2,
                linecolor='black'
            ),
            caxis=dict(
                title=component_names[2],
                min=0,
                tickformat='.2f',
                gridcolor='lightgray',
                showline=True,
                linewidth=2,
                linecolor='black'
            ),
            bgcolor='rgba(240, 240, 240, 0.5)'
        ),
        height=700,
        showlegend=True,
        font=dict(size=12)
    )

    return fig


def validate_mixture_input(mixture_dict, tolerance=1e-6):
    """
    Validate mixture composition

    Args:
        mixture_dict: dict like {'X1': 0.5, 'X2': 0.3, 'X3': 0.2}
        tolerance: numerical tolerance

    Returns:
        tuple: (is_valid: bool, message: str)
    """
    # Check sum equals 1
    total = sum(mixture_dict.values())

    if abs(total - 1.0) > tolerance:
        return False, f"Mixture components sum to {total:.6f}, must sum to 1.0"

    # Check all values in [0, 1]
    for comp, val in mixture_dict.items():
        if val < -tolerance:
            return False, f"Component {comp} has negative value: {val:.6f}"
        if val > 1.0 + tolerance:
            return False, f"Component {comp} exceeds 1.0: {val:.6f}"

    return True, "Valid mixture"
