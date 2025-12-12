"""
Mixture Predictions and Component Effects Module
"""

import streamlit as st
import numpy as np
import pandas as pd
from mixture_utils.mixture_computation import scheffe_polynomial_prediction, calculate_component_effects


def predict_custom_mixture(model_results, mixture_input):
    """
    Wrapper for prediction on custom mixture

    Returns dict with prediction details
    """
    return scheffe_polynomial_prediction(model_results, mixture_input)


def show_mixture_predictions_ui(model_results, mixture_design_matrix, data):
    """
    Display predictions and component effects UI

    Args:
        model_results: fitted mixture model
        mixture_design_matrix: original design
        data: full dataset
    """
    st.markdown("## 🎯 Predictions & Component Effects")

    component_names = model_results['component_names']
    n_components = len(component_names)

    # Section 1: Interactive Prediction
    st.markdown("### 🔮 Predict Custom Mixture")

    st.info("Enter mixture composition (values must sum to 1.0)")

    # Sliders for each component
    mixture_values = {}
    remaining = 1.0

    for i, comp in enumerate(component_names[:-1]):
        max_val = min(1.0, remaining)
        val = st.slider(
            f"{comp}",
            0.0,
            max_val,
            max_val / (n_components - i),
            0.01,
            key=f"mixture_slider_{comp}"
        )
        mixture_values[comp] = val
        remaining -= val

    # Last component auto-calculated
    last_comp = component_names[-1]
    mixture_values[last_comp] = max(0.0, remaining)

    st.write(f"**{last_comp}:** {mixture_values[last_comp]:.4f} (auto-calculated)")

    # Validate sum
    total = sum(mixture_values.values())
    if abs(total - 1.0) < 1e-6:
        st.success(f"✓ Valid mixture (sum = {total:.6f})")

        # Make prediction
        pred_result = predict_custom_mixture(model_results, mixture_values)

        st.markdown("---")
        st.markdown("### 📊 Prediction Result")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Predicted Response", f"{pred_result['predicted_value']:.4f}")

        with col2:
            st.metric("Standard Error", f"{pred_result['se']:.4f}")

        with col3:
            ci_width = pred_result['ci_upper'] - pred_result['ci_lower']
            st.metric("95% CI Width", f"{ci_width:.4f}")

        st.markdown(f"**95% Confidence Interval:** [{pred_result['ci_lower']:.4f}, {pred_result['ci_upper']:.4f}]")

    else:
        st.error(f"❌ Invalid mixture (sum = {total:.6f}, must = 1.0)")

    st.markdown("---")

    # ═══════════════════════════════════════════════════════════════════════════
    # Section 2: Component Effects Analysis (same as Response Surface tab)
    # ═══════════════════════════════════════════════════════════════════════════

    st.markdown("## 📈 Component Effects")

    st.info("""
    **Component Effect Analysis:**

    - **Top:** Effect magnitude (Pure component - Opposite edge)
    - **Bottom:** Response profiles along each component axis
    """)

    try:
        # Get grid data from session_state
        grid_data = st.session_state.get('ternary_grid_data', None)

        if grid_data is None:
            st.warning("⚠️ No ternary grid data available. Please generate the response surface first (Tab: Response Surface).")
        else:
            from mixture_utils.mixture_computation import extract_component_trajectories
            from mixture_utils.mixture_surface import plot_component_effects_magnitude, plot_component_trajectories_from_grid

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

            st.plotly_chart(fig_effects, use_container_width=True, key="predictions_effects_magnitude")

            # ═══════════════════════════════════════════════════════════════════════════
            # SECTION 2: Response Profiles
            # ═══════════════════════════════════════════════════════════════════════════

            st.markdown("### 📊 Component Effects: Response Profiles")

            # Plot response profiles
            fig_grid_trajectories = plot_component_trajectories_from_grid(
                trajectories,
                model_results['component_names']
            )

            st.plotly_chart(fig_grid_trajectories, use_container_width=True, key="predictions_trajectories")

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
        st.error(f"❌ Component effects analysis failed: {str(e)}")
        import traceback
        with st.expander("🐛 Error details"):
            st.code(traceback.format_exc())
