"""
Multi-DOE Pareto Optimization Module
Multi-criteria optimization across multiple response variables

This module provides multi-criteria decision making functionality,
allowing users to optimize multiple responses simultaneously with weighted objectives.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import existing utilities
from .surface_analysis import create_prediction_matrix


def generate_search_grid(search_ranges, grid_res):
    """
    Create X grid of points to evaluate

    Args:
        search_ranges (dict): {var_name: (min, max)}
        grid_res (int): Number of points per dimension

    Returns:
        ndarray: Grid of X points (n_points × n_vars)
    """
    var_names = list(search_ranges.keys())
    n_vars = len(var_names)

    # Create 1D grids for each variable
    grids_1d = []
    for var_name in var_names:
        min_val, max_val = search_ranges[var_name]
        grid_1d = np.linspace(min_val, max_val, grid_res)
        grids_1d.append(grid_1d)

    # Create full grid using meshgrid
    grids_nd = np.meshgrid(*grids_1d, indexing='ij')

    # Flatten and stack
    n_points = grid_res ** n_vars
    X_grid = np.column_stack([grid.flatten() for grid in grids_nd])

    return X_grid


def calculate_weighted_objective(predictions_dict, weights, directions):
    """
    Calculate normalized weighted objective for all points

    Args:
        predictions_dict (dict): {y_name: predictions_array}
        weights (dict): {y_name: weight_value}
        directions (dict): {y_name: 'maximize' or 'minimize'}

    Returns:
        ndarray: Weighted objective values for each point
    """
    n_points = len(next(iter(predictions_dict.values())))
    weighted_scores = np.zeros(n_points)

    for y_name, predictions in predictions_dict.items():
        weight = weights.get(y_name, 0)
        direction = directions.get(y_name, 'maximize')

        # Normalize predictions to [0, 1]
        pred_min = predictions.min()
        pred_max = predictions.max()

        if pred_max - pred_min > 1e-10:  # Avoid division by zero
            normalized = (predictions - pred_min) / (pred_max - pred_min)
        else:
            normalized = np.ones_like(predictions) * 0.5

        # Invert if minimizing
        if direction == 'minimize':
            normalized = 1 - normalized

        # Add weighted contribution
        weighted_scores += weight * normalized

    return weighted_scores


def find_pareto_frontier(predictions_dict, directions, max_points=50):
    """
    Find Pareto-optimal points (non-dominated solutions)

    Args:
        predictions_dict (dict): {y_name: predictions_array}
        directions (dict): {y_name: 'maximize' or 'minimize'}
        max_points (int): Maximum number of Pareto points to return

    Returns:
        ndarray: Indices of Pareto-optimal points
    """
    n_points = len(next(iter(predictions_dict.values())))
    y_names = list(predictions_dict.keys())

    # Create objectives matrix (n_points × n_objectives)
    objectives = np.column_stack([predictions_dict[y] for y in y_names])

    # Convert to maximization problem (flip minimization objectives)
    for i, y_name in enumerate(y_names):
        if directions[y_name] == 'minimize':
            objectives[:, i] = -objectives[:, i]

    # Find Pareto frontier
    pareto_indices = []

    for i in range(n_points):
        is_dominated = False
        for j in range(n_points):
            if i == j:
                continue

            # Check if j dominates i
            # j dominates i if: j >= i in all objectives AND j > i in at least one
            if np.all(objectives[j] >= objectives[i]) and np.any(objectives[j] > objectives[i]):
                is_dominated = True
                break

        if not is_dominated:
            pareto_indices.append(i)

        # Limit number of Pareto points
        if len(pareto_indices) >= max_points:
            break

    return np.array(pareto_indices)


def show_pareto_ui_multidoe(models_dict, x_vars, y_vars):
    """
    Multi-criteria optimization UI for Multi-DOE

    Args:
        models_dict (dict): Dictionary {y_name: model_result}
        x_vars (list): List of X variable names
        y_vars (list): List of Y variable names
    """
    st.markdown("## 🎯 Multi-Criteria Optimization")
    st.info("""
    Find optimal experimental conditions across multiple response criteria.
    Define weights and optimization direction for each response, then search the design space.
    """)

    # ========================================================================
    # SECTION 1: OBJECTIVE CONFIGURATION
    # ========================================================================
    st.markdown("### 📊 Response Weights")
    st.info("Specify relative importance of each response (will be normalized to 100%)")

    # Weight sliders
    weights = {}
    weight_cols = st.columns(min(3, len(y_vars)))

    for i, y_var in enumerate(y_vars):
        with weight_cols[i % len(weight_cols)]:
            weight = st.slider(
                f"{y_var} importance:",
                0, 100, 30,
                help=f"Weight for {y_var}",
                key=f"multidoe_pareto_weight_{y_var}"
            )
            weights[y_var] = weight

    # Normalize weights
    total_weight = sum(weights.values())
    if total_weight > 0:
        normalized_weights = {k: v/total_weight for k, v in weights.items()}

        st.markdown("#### Normalized Weights (%)")
        weight_display = pd.DataFrame({
            'Response': list(normalized_weights.keys()),
            'Weight (%)': [f"{v*100:.1f}" for v in normalized_weights.values()]
        })
        st.dataframe(weight_display, use_container_width=True, hide_index=True)
    else:
        st.warning("⚠️ All weights are zero. Please set at least one weight > 0")
        return

    # ========================================================================
    # SECTION 2: OPTIMIZATION DIRECTIONS
    # ========================================================================
    st.markdown("---")
    st.markdown("### ⬆️/⬇️ Optimization Direction")

    directions = {}
    dir_cols = st.columns(min(3, len(y_vars)))

    for i, y_var in enumerate(y_vars):
        with dir_cols[i % len(dir_cols)]:
            direction = st.radio(
                f"{y_var}:",
                ["Maximize", "Minimize"],
                key=f"multidoe_pareto_direction_{y_var}",
                horizontal=True
            )
            directions[y_var] = "maximize" if direction == "Maximize" else "minimize"

    # ========================================================================
    # SECTION 3: SEARCH SPACE DEFINITION
    # ========================================================================
    st.markdown("---")
    st.markdown("### 🔍 Search Space")

    search_ranges = {}
    range_cols = st.columns(len(x_vars))

    for i, x_var in enumerate(x_vars):
        with range_cols[i]:
            st.markdown(f"**{x_var}**")
            col_min, col_max = st.columns(2)
            with col_min:
                min_val = st.number_input(
                    "Min:",
                    value=-1.0,
                    step=0.1,
                    format="%.2f",
                    key=f"multidoe_pareto_min_{x_var}"
                )
            with col_max:
                max_val = st.number_input(
                    "Max:",
                    value=1.0,
                    step=0.1,
                    format="%.2f",
                    key=f"multidoe_pareto_max_{x_var}"
                )
            search_ranges[x_var] = (min_val, max_val)

    # Grid resolution
    grid_res = st.slider(
        "Grid resolution (points per variable):",
        5, 30, 10,
        help="Higher = more points, slower computation",
        key="multidoe_pareto_grid_res"
    )

    n_total_points = grid_res ** len(x_vars)
    st.metric("Total evaluation points", n_total_points)

    if n_total_points > 10000:
        st.warning(f"⚠️ High number of points ({n_total_points}). This may be slow.")

    # ========================================================================
    # SECTION 4: OPTIMIZATION EXECUTION
    # ========================================================================
    st.markdown("---")

    if st.button("🚀 Find Optimal Point(s)", type="primary", key="multidoe_pareto_optimize"):
        try:
            with st.spinner(f"Optimizing across {n_total_points} points..."):
                # Generate X grid
                X_grid = generate_search_grid(search_ranges, grid_res)

                # Predict all Y for all grid points
                predictions_dict = {}
                for y_name, model in models_dict.items():
                    if 'error' in model:
                        st.warning(f"⚠️ Skipping {y_name} due to model error")
                        continue

                    # Get coefficient names
                    coef_names = model['coefficients'].index.tolist()

                    # Create model matrix
                    X_model = create_prediction_matrix(X_grid, x_vars, coef_names)

                    # Calculate predictions
                    y_pred = X_model @ model['coefficients'].values
                    predictions_dict[y_name] = y_pred

                # Calculate weighted objective for each point
                weighted_objectives = calculate_weighted_objective(
                    predictions_dict, normalized_weights, directions
                )

                # Find best point (highest weighted objective)
                best_idx = np.argmax(weighted_objectives)
                best_x = X_grid[best_idx]
                best_score = weighted_objectives[best_idx]

                # Find Pareto frontier (optional)
                pareto_indices = find_pareto_frontier(predictions_dict, directions, max_points=20)

            st.success("✅ Optimization complete!")

            # ================================================================
            # DISPLAY: OPTIMAL POINT
            # ================================================================
            st.markdown("### 🎯 Optimal Point (Weighted)")

            col_opt1, col_opt2 = st.columns([2, 1])

            with col_opt1:
                st.markdown("**Optimal X Coordinates:**")
                optimal_df = pd.DataFrame({
                    'Variable': x_vars,
                    'Value': best_x
                })
                st.dataframe(optimal_df, use_container_width=True, hide_index=True)

            with col_opt2:
                st.metric("Weighted Score", f"{best_score:.4f}")

            # Show predicted Y values at optimal point
            st.markdown("**Predicted Responses at Optimal Point:**")

            optimal_y_cols = st.columns(len(y_vars))
            for i, y_name in enumerate(y_vars):
                with optimal_y_cols[i]:
                    if y_name in predictions_dict:
                        y_pred_optimal = predictions_dict[y_name][best_idx]
                        direction_str = "⬆️ MAX" if directions[y_name] == 'maximize' else "⬇️ MIN"

                        st.metric(
                            f"{y_name}",
                            f"{y_pred_optimal:.4f}",
                            delta=direction_str
                        )

            # ================================================================
            # DISPLAY: PARETO FRONTIER
            # ================================================================
            if len(pareto_indices) > 1:
                st.markdown("---")
                st.markdown("### 📊 Pareto Frontier (Trade-off Points)")
                st.info(f"Found {len(pareto_indices)} Pareto-optimal points (non-dominated solutions)")

                # Create Pareto table
                pareto_data = []
                for idx in pareto_indices:
                    row_data = {'Index': idx}
                    # Add X values
                    for j, x_var in enumerate(x_vars):
                        row_data[x_var] = X_grid[idx, j]
                    # Add Y predictions
                    for y_name in y_vars:
                        if y_name in predictions_dict:
                            row_data[y_name] = predictions_dict[y_name][idx]
                    pareto_data.append(row_data)

                pareto_df = pd.DataFrame(pareto_data)
                st.dataframe(pareto_df, use_container_width=True, hide_index=True)

                # 2D scatter of Pareto frontier (if exactly 2 Y vars)
                if len(y_vars) == 2:
                    y1_name = y_vars[0]
                    y2_name = y_vars[1]

                    fig = go.Figure()

                    # Plot all points
                    fig.add_trace(go.Scatter(
                        x=predictions_dict[y1_name],
                        y=predictions_dict[y2_name],
                        mode='markers',
                        marker=dict(size=4, color='lightgray', opacity=0.5),
                        name='All points',
                        hoverinfo='skip'
                    ))

                    # Plot Pareto points
                    fig.add_trace(go.Scatter(
                        x=predictions_dict[y1_name][pareto_indices],
                        y=predictions_dict[y2_name][pareto_indices],
                        mode='markers',
                        marker=dict(size=10, color='red', symbol='star'),
                        name='Pareto frontier',
                        text=[f"Point {i}" for i in pareto_indices],
                        hovertemplate=f'<b>%{{text}}</b><br>{y1_name}: %{{x:.4f}}<br>{y2_name}: %{{y:.4f}}<extra></extra>'
                    ))

                    # Highlight best point
                    fig.add_trace(go.Scatter(
                        x=[predictions_dict[y1_name][best_idx]],
                        y=[predictions_dict[y2_name][best_idx]],
                        mode='markers',
                        marker=dict(size=15, color='green', symbol='star'),
                        name='Best (weighted)',
                        hovertemplate=f'<b>Best point</b><br>{y1_name}: %{{x:.4f}}<br>{y2_name}: %{{y:.4f}}<extra></extra>'
                    ))

                    fig.update_layout(
                        title="Pareto Frontier (2D)",
                        xaxis_title=y1_name,
                        yaxis_title=y2_name,
                        height=500
                    )

                    st.plotly_chart(fig, use_container_width=True)

                # 3D scatter of Pareto frontier (if exactly 3 Y vars)
                elif len(y_vars) == 3:
                    y1_name = y_vars[0]
                    y2_name = y_vars[1]
                    y3_name = y_vars[2]

                    fig = go.Figure(data=[
                        go.Scatter3d(
                            x=predictions_dict[y1_name][pareto_indices],
                            y=predictions_dict[y2_name][pareto_indices],
                            z=predictions_dict[y3_name][pareto_indices],
                            mode='markers',
                            marker=dict(size=8, color='red'),
                            text=[f"Point {i}" for i in pareto_indices],
                            hovertemplate=f'<b>%{{text}}</b><br>{y1_name}: %{{x:.4f}}<br>{y2_name}: %{{y:.4f}}<br>{y3_name}: %{{z:.4f}}<extra></extra>'
                        )
                    ])

                    fig.update_layout(
                        title="Pareto Frontier (3D)",
                        scene=dict(
                            xaxis_title=y1_name,
                            yaxis_title=y2_name,
                            zaxis_title=y3_name
                        ),
                        height=500
                    )

                    st.plotly_chart(fig, use_container_width=True)

            # Store results in session state
            st.session_state.multidoe_pareto_results = {
                'best_x': best_x,
                'best_score': best_score,
                'best_y_predictions': {y: predictions_dict[y][best_idx] for y in predictions_dict},
                'pareto_indices': pareto_indices,
                'X_grid': X_grid,
                'predictions_dict': predictions_dict
            }

        except Exception as e:
            st.error(f"❌ Optimization failed: {str(e)}")
            import traceback
            with st.expander("🐛 Error details"):
                st.code(traceback.format_exc())
