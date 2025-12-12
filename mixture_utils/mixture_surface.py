"""
Mixture Response Surface Visualization Module
Ultra-smooth ternary surface using high-density grid
No visible dots - professional scientific appearance
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go


def show_mixture_surface_ui(model_results, mixture_design_matrix=None, data=None):
    """
    Display ultra-smooth ternary surface with high-density grid
    Correct orientation: x1 bottom-left, x2 top, x3 bottom-right
    """
    st.markdown("## 📊 Response Surface Visualization")

    n_components = model_results['n_components']

    if n_components != 3:
        st.warning(f"⚠️ Ternary plots require exactly 3 components. You have {n_components}.")
        st.info("For other dimensions: use 2D contour slices.")
        return

    st.info("**Ultra-Smooth Ternary Surface:** High-resolution response visualization")

    # Grid resolution (start higher)
    col1, col2 = st.columns([3, 1])

    with col1:
        st.markdown("### ⚙️ Plot Settings")

    with col2:
        n_points = st.slider(
            "Grid Resolution",
            min_value=40,
            max_value=100,
            value=70,  # ← Higher default
            step=5,
            key="ternary_surface_resolution"
        )

    st.markdown("---")

    # Create ternary grid
    with st.spinner(f"Generating ternary surface ({n_points}² grid)..."):
        try:
            # Create barycentric grid
            step = 1.0 / n_points
            a_list = []  # X1
            b_list = []  # X2
            c_list = []  # X3
            responses_list = []

            component_names = model_results['component_names']

            for i in range(n_points + 1):
                for j in range(n_points + 1 - i):
                    x1 = i * step
                    x2 = j * step
                    x3 = 1.0 - x1 - x2

                    if x3 >= -1e-10:
                        # Predict response
                        from mixture_utils.mixture_computation import scheffe_polynomial_prediction

                        pred_dict = scheffe_polynomial_prediction(
                            model_results,
                            {component_names[0]: x1, component_names[1]: x2, component_names[2]: x3}
                        )
                        response = pred_dict['predicted_value']

                        a_list.append(x1)
                        b_list.append(x2)
                        c_list.append(x3)
                        responses_list.append(response)

            responses_array = np.array(responses_list)

            # ═══════════════════════════════════════════════════════════════════════════
            # SAVE GRID DATA FOR COMPONENT EFFECTS EXTRACTION
            # ═══════════════════════════════════════════════════════════════════════════

            st.session_state.ternary_grid_data = {
                'pc1': a_list,
                'pc2': b_list,
                'pc3': c_list,
                'responses': responses_list,
                'component_names': component_names,
                'n_points': n_points
            }

            # ═══════════════════════════════════════════════════════════════════════════
            # CREATE ULTRA-SMOOTH TERNARY SURFACE
            # ═══════════════════════════════════════════════════════════════════════════

            fig = go.Figure()

            # High-density scatter creates pixel-perfect smooth appearance
            fig.add_trace(go.Scatterternary(
                a=b_list,  # X2 → top
                b=a_list,  # X1 → bottom-left
                c=c_list,  # X3 → bottom-right
                mode='markers',
                marker=dict(
                    size=20,  # ← Invisible dots
                    color=responses_array,
                    colorscale='Viridis',  # Yellow = high, Blue = low
                    showscale=True,
                    colorbar=dict(
                        title="Response<br>Value",
                        thickness=15,
                        len=0.65,
                        tickformat='.2f',
                        x=1.02
                    ),
                    line=dict(width=0),  # No borders
                    opacity=1.0  # ← Full opacity for seamless coverage
                ),
                text=[f"<b>Response:</b> {r:.4f}" for r in responses_array],
                hovertemplate='<b>Mixture Composition</b><br>' +
                             f'{component_names[0]}: %{{b:.4f}}<br>' +
                             f'{component_names[1]}: %{{a:.4f}}<br>' +
                             f'{component_names[2]}: %{{c:.4f}}<br>' +
                             '%{text}<extra></extra>',
                name='Response Surface'
            ))

            # ═══════════════════════════════════════════════════════════════════════════
            # LAYOUT CONFIGURATION
            # ═══════════════════════════════════════════════════════════════════════════

            fig.update_layout(
                title={
                    'text': 'Mixture Response Surface - Ternary Plot',
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 15, 'color': '#333'}
                },

                ternary=dict(
                    sum=1,
                    aaxis=dict(
                        title=f'<b>{component_names[1]}</b>',
                        min=0,
                        tickformat='.2f',
                        showgrid=True,
                        gridcolor='rgba(200, 200, 200, 0.3)',
                        gridwidth=0.5,
                        showline=True,
                        linewidth=2,
                        linecolor='black'
                    ),
                    baxis=dict(
                        title=f'<b>{component_names[0]}</b>',
                        min=0,
                        tickformat='.2f',
                        showgrid=True,
                        gridcolor='rgba(200, 200, 200, 0.3)',
                        gridwidth=0.5,
                        showline=True,
                        linewidth=2,
                        linecolor='black'
                    ),
                    caxis=dict(
                        title=f'<b>{component_names[2]}</b>',
                        min=0,
                        tickformat='.2f',
                        showgrid=True,
                        gridcolor='rgba(200, 200, 200, 0.3)',
                        gridwidth=0.5,
                        showline=True,
                        linewidth=2,
                        linecolor='black'
                    ),
                    bgcolor='rgba(255, 255, 255, 0.1)'
                ),

                height=650,  # ← Ridotto
                width=750,   # ← Ridotto
                showlegend=False,
                font=dict(family="Arial, sans-serif", size=11),
                margin=dict(l=80, r=130, t=80, b=80),
                hovermode='closest',
                paper_bgcolor='white',
                plot_bgcolor='white'
            )

            st.plotly_chart(fig, use_container_width=True)  # ← CHANGE: True (centered)

            st.success(f"✅ Ultra-smooth ternary surface generated ({len(responses_array):,} points)!")

        except Exception as e:
            st.error(f"❌ Surface generation failed: {str(e)}")
            import traceback
            with st.expander("🐛 Error details"):
                st.code(traceback.format_exc())

    st.markdown("---")

    # ═══════════════════════════════════════════════════════════════════════════
    # COMPONENT EFFECTS
    # ═══════════════════════════════════════════════════════════════════════════

    st.markdown("## 📊 Component Effects")

    st.info("""
    **Component Effect Analysis:**

    - **Top:** Effect magnitude (Pure component - Opposite edge)
    - **Bottom:** Response profiles along each component axis
    """)

    try:
        # Get grid data from session_state
        grid_data = st.session_state.get('ternary_grid_data', None)

        if grid_data is None:
            st.warning("⚠️ No ternary grid data available. Please generate the response surface first (above).")
        else:
            from mixture_utils.mixture_computation import extract_component_trajectories

            # Extract trajectories from grid
            trajectories = extract_component_trajectories(grid_data, tolerance=0.02, downsample=3)

            # ═══════════════════════════════════════════════════════════════════════════
            # SECTION 1: Effect Magnitude Bar Chart
            # ═══════════════════════════════════════════════════════════════════════════

            st.markdown("### 📉 Component Effects Magnitude")

            fig_effects = plot_component_effects_magnitude(
                trajectories,
                model_results['component_names']
            )

            st.plotly_chart(fig_effects, use_container_width=True, key="response_surface_effects_magnitude")

            # ═══════════════════════════════════════════════════════════════════════════
            # SECTION 2: Response Profiles
            # ═══════════════════════════════════════════════════════════════════════════

            st.markdown("### 📊 Component Effects: Response Profiles")

            # Plot response profiles
            fig_grid_trajectories = plot_component_trajectories_from_grid(
                trajectories,
                model_results['component_names']
            )

            st.plotly_chart(fig_grid_trajectories, use_container_width=True, key="response_surface_trajectories")

            # ═══════════════════════════════════════════════════════════════════════════
            # SECTION 3: Summary Table
            # ═══════════════════════════════════════════════════════════════════════════

            st.markdown("### 📋 Trajectory Summary (from Grid)")

            grid_summary_data = []
            for comp_name in model_results['component_names']:
                traj = trajectories[comp_name]
                grid_summary_data.append({
                    'Component': comp_name,
                    'Effect (Δ)': traj['effect_magnitude'],
                    'Max Response': traj['max_response'],
                    'Min Response': traj['min_response'],
                    'Max at x': f"{traj['max_at_t']:.3f}",
                    'Curvature': traj['curvature'],
                    'n_points': traj['n_points']
                })

            grid_summary_df = pd.DataFrame(grid_summary_data)
            st.dataframe(
                grid_summary_df.style.format({
                    'Effect (Δ)': '{:.4f}',
                    'Max Response': '{:.4f}',
                    'Min Response': '{:.4f}',
                    'Curvature': '{:.4f}'
                }),
                use_container_width=True,
                hide_index=True
            )

    except Exception as e:
        st.error(f"❌ Trajectory extraction failed: {str(e)}")
        import traceback
        with st.expander("🐛 Error details"):
            st.code(traceback.format_exc())

    st.markdown("---")


def plot_component_trajectories_from_grid(trajectories, component_names):
    """
    Plot 3 side-by-side component trajectories extracted from ternary grid.

    Args:
        trajectories: dict from extract_component_trajectories()
        component_names: list of component names

    Returns:
        plotly Figure with 3 subplots
    """
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    n_components = len(component_names)

    fig = make_subplots(
        rows=1, cols=n_components,
        subplot_titles=[f"{cn} Trajectory (from Grid)" for cn in component_names],
        specs=[[{'secondary_y': False}] * n_components],
        horizontal_spacing=0.12
    )

    for col_idx, comp_name in enumerate(component_names, 1):
        traj = trajectories[comp_name]

        x_vals = traj['x_values']
        y_vals = traj['y_values']

        # Response curve (blue)
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=y_vals,
                mode='lines+markers',
                name=f'{comp_name}',
                line=dict(color='#1f77b4', width=3),
                marker=dict(size=6, color='#1f77b4'),
                hovertemplate=f'<b>{comp_name}</b><br>x: %{{x:.3f}}<br>Y: %{{y:.4f}}<extra></extra>',
                showlegend=False
            ),
            row=1, col=col_idx
        )

        # Mark maximum
        max_idx = traj['max_index']
        max_x = x_vals[max_idx]
        max_y = traj['max_response']

        # ═════════════════════════════════════════════════════════════════════
        # Add vertical dashed line at maximum with LABEL at MID-SEGMENT
        # ═════════════════════════════════════════════════════════════════════

        # Get y-axis range to position label at mid-segment
        y_min = np.min(y_vals)
        y_max = np.max(y_vals)
        y_range = y_max - y_min
        y_mid = y_min + (y_range * 0.5)  # ← Position at 50% of range

        # Add vertical dashed line (no annotation here)
        fig.add_vline(
            x=max_x,
            line_dash="dash",
            line_color="#2ca02c",
            line_width=2,
            row=1, col=col_idx
        )

        # Add text annotation separately (at mid-segment, not at top)
        fig.add_annotation(
            x=max_x,
            y=y_mid,  # ← MID-SEGMENT position
            text=f"<b>{max_y:.3f}</b>",
            showarrow=False,
            xanchor="left",
            yanchor="middle",
            bgcolor="white",
            bordercolor="#2ca02c",
            borderwidth=1,
            borderpad=4,
            font=dict(color="#2ca02c", size=11),
            row=1,
            col=col_idx
        )

        # Update axes
        fig.update_xaxes(
            title_text=f"{comp_name}: 0→1",
            row=1, col=col_idx
        )

        if col_idx == 1:
            fig.update_yaxes(title_text="Response (Y)", row=1, col=1)

    fig.update_layout(
        title="Component Effects: Response Profiles",
        height=500,
        width=1400,
        hovermode='x unified',
        showlegend=False,
        template='plotly_white',
        paper_bgcolor='white'
    )

    return fig


def plot_component_effects_magnitude(trajectories, component_names):
    """
    Plot bar chart of component effect magnitudes.

    Effect = Y(pure component) - Y(opposite edge)

    Args:
        trajectories: dict from extract_component_trajectories()
        component_names: list of component names

    Returns:
        plotly Figure with bar chart
    """
    import plotly.graph_objects as go

    effects = []
    colors = []

    for comp_name in component_names:
        traj = trajectories[comp_name]
        effect = traj['effect_magnitude']
        effects.append(effect)

        # Green if positive, Red if negative
        colors.append('#2ca02c' if effect >= 0 else '#d62728')

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=component_names,
        y=effects,
        marker=dict(
            color=colors,
            line=dict(color='black', width=2)
        ),
        text=[f'{e:+.4f}' for e in effects],
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>Effect: %{y:.4f}<extra></extra>',
        showlegend=False
    ))

    # Add zero line
    fig.add_hline(
        y=0,
        line_dash="solid",
        line_color="black",
        line_width=2
    )

    fig.update_layout(
        title="Component Effects: Pure Component vs. Opposite Edge",
        xaxis_title="Component",
        yaxis_title="Effect (Pure - Opposite Midpoint)",
        height=400,
        template='plotly_white',
        hovermode='x unified'
    )

    return fig



