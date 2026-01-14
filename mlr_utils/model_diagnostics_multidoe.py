"""
FIXED VERSION: model_diagnostics_multidoe.py
Multi-DOE Model Diagnostics Module - STANDALONE (no session_state dependencies)

PROBLEM: Original version called show_model_diagnostics_ui() from model_diagnostics.py
         which hardcoded references to st.session_state.mlr_y_var

SOLUTION: Complete rewrite with standalone implementation
         No dependencies on MLR/DOE session variables
         All plots use the y_name parameter passed to this function
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy import stats


def show_model_diagnostics_ui_multidoe(model_results, X, y, y_name="Response"):
    """
    Show diagnostics for a single multi-DOE model
    STANDALONE VERSION - no session_state dependencies

    Args:
        model_results (dict): Model results from fit_multidoe_model()
        X (DataFrame): DataFrame of predictors
        y (Series): Series of response values
        y_name (str): Name of response variable
    """
    # Display header
    st.markdown(f"## Diagnostics for Response: **{y_name}**")

    # Display key metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        r2 = model_results.get('r_squared', np.nan)
        st.metric("R²", f"{r2:.4f}" if not np.isnan(r2) else "N/A")

    with col2:
        rmse = model_results.get('rmse', np.nan)
        st.metric("RMSE", f"{rmse:.4f}" if not np.isnan(rmse) else "N/A")

    with col3:
        q2 = model_results.get('q2', np.nan)
        st.metric("Q²", f"{q2:.4f}" if not np.isnan(q2) else "N/A")

    with col4:
        dof = model_results.get('dof', np.nan)
        st.metric("DOF", f"{int(dof)}" if not np.isnan(dof) else "N/A")

    st.markdown("---")

    # ========================================================================
    # SECTION 1: RESIDUALS PLOT
    # ========================================================================
    st.markdown("### Residual Analysis")

    if 'residuals' in model_results and 'y_pred' in model_results:
        residuals = model_results['residuals']
        y_pred = model_results['y_pred']

        # Create residual plot
        fig_residuals = go.Figure()

        fig_residuals.add_trace(go.Scatter(
            x=y_pred,
            y=residuals,
            mode='markers',
            marker=dict(size=8, color='blue', opacity=0.6),
            name='Residuals',
            hovertemplate=f'Fitted: %{{x:.3f}}<br>Residual: %{{y:.3f}}<extra></extra>'
        ))

        # Add zero line
        fig_residuals.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="Zero")

        fig_residuals.update_layout(
            title=f"Residual Plot - {y_name}",
            xaxis_title="Fitted Values",
            yaxis_title="Residuals",
            height=400,
            hovermode='closest'
        )

        st.plotly_chart(fig_residuals, use_container_width=True)

    else:
        st.info("Residuals not available in model results")

    # ========================================================================
    # SECTION 1B: RESIDUALS VS TIME SEQUENCE
    # ========================================================================
    st.markdown("### ⏱️ Residuals vs Experiment Number (Time Sequence)")

    if 'residuals' in model_results:
        residuals = model_results['residuals']

        # Get sample names from y.index if available, otherwise use generic names
        if hasattr(y, 'index'):
            sample_names = y.index.tolist()
        else:
            sample_names = [f"Exp_{i+1}" for i in range(len(residuals))]

        # Experiment numbers (0-indexed for plotting)
        exp_numbers = np.arange(len(residuals))

        # Calculate statistics for reference lines
        residuals_std = np.std(residuals)

        # Create Plotly figure
        fig_time = go.Figure()

        # Add scatter + line trace
        fig_time.add_trace(go.Scatter(
            x=exp_numbers,
            y=residuals,
            mode='markers+lines',
            marker=dict(
                size=8,
                color=residuals,  # Color by residual value
                colorscale='RdBu',  # Red (negative) to Blue (positive)
                showscale=False,  # Hide colorbar
                line=dict(color='darkblue', width=1)  # Dark blue marker border
            ),
            line=dict(
                color='rgba(100, 100, 200, 0.3)',  # Semi-transparent blue-gray
                width=1
            ),
            name='Residuals',
            # Custom hover data: Exp #N: SampleName + Residual value
            customdata=[[i+1, name] for i, name in enumerate(sample_names)],
            hovertemplate='<b>Exp #%{customdata[0]}: %{customdata[1]}</b><br>Residual: %{y:.4f}<extra></extra>'
        ))

        # Add reference lines (without annotations - we'll add them separately)

        # Zero line (perfect fit) - Red dashed
        fig_time.add_hline(
            y=0,
            line_dash="dash",
            line_color="red",
            line_width=2
        )

        # +1 Std Dev line - Orange dotted
        fig_time.add_hline(
            y=residuals_std,
            line_dash="dot",
            line_color="orange",
            line_width=1.5
        )

        # -1 Std Dev line - Orange dotted
        fig_time.add_hline(
            y=-residuals_std,
            line_dash="dot",
            line_color="orange",
            line_width=1.5
        )

        # Reference lines are self-explanatory - no legend needed

        # Update layout
        fig_time.update_layout(
            title=f"Residuals vs Experiment Number (Time Sequence) - {y_name}",
            xaxis_title="Experiment Number",
            yaxis_title="Residual Value",
            hovermode='closest',
            height=500,
            template='plotly_white',
            showlegend=False,
            # Adjust tick density for small datasets
            xaxis=dict(
                dtick=1 if len(residuals) < 20 else None  # Show every tick for small datasets
            )
        )

        # Display plot
        st.plotly_chart(fig_time, use_container_width=True)

        # Display statistics in columns
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Mean Residual", f"{np.mean(residuals):.6f}")
        with col2:
            st.metric("Std Dev", f"{np.std(residuals):.6f}")
        with col3:
            st.metric("Max |Residual|", f"{np.max(np.abs(residuals)):.6f}")
        with col4:
            st.metric("N Experiments", len(residuals))

        # Interpretation guide
        with st.expander("📖 Interpretation Guide"):
            col_good, col_warning = st.columns(2)

            with col_good:
                st.markdown("**✅ GOOD Signs:**")
                st.markdown("""
                - Random scatter around zero
                - No systematic pattern or trend
                - Values roughly equidistributed above/below zero
                - Most points within ±1 std dev band
                - No clustering or grouping
                """)

            with col_warning:
                st.markdown("**⚠️ WARNING Signs:**")
                st.markdown("""
                - **Trend:** Increasing or decreasing pattern
                - **Cycles:** Periodic/sinusoidal patterns
                - **Outliers:** Isolated extreme values
                - **Variance change:** Spread increases/decreases over time
                - **Clustering:** Groups of similar residuals
                """)

            st.info("""
            **Why this matters:**
            - Patterns suggest missing time-dependent variables
            - Trends may indicate instrument drift or process changes
            - Cycles suggest periodic disturbances
            - Clustering may indicate batch effects or grouping factors
            """)

    else:
        st.info("Residuals not available for time sequence plot")

    st.markdown("---")

    # ========================================================================
    # SECTION 2: Q-Q PLOT
    # ========================================================================
    st.markdown("### Q-Q Plot (Normality Check)")

    if 'residuals' in model_results:
        residuals = model_results['residuals']
        residuals_std = (residuals - np.mean(residuals)) / np.std(residuals)

        # Calculate theoretical quantiles
        theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(residuals)))
        sample_quantiles = np.sort(residuals_std)

        fig_qq = go.Figure()

        fig_qq.add_trace(go.Scatter(
            x=theoretical_quantiles,
            y=sample_quantiles,
            mode='markers',
            marker=dict(size=8, color='blue', opacity=0.6),
            name='Data',
            hovertemplate='Theoretical: %{x:.3f}<br>Sample: %{y:.3f}<extra></extra>'
        ))

        # Add diagonal line (perfect normal distribution)
        min_val = min(theoretical_quantiles.min(), sample_quantiles.min())
        max_val = max(theoretical_quantiles.max(), sample_quantiles.max())
        fig_qq.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Perfect Normal',
            line=dict(color='red', dash='dash')
        ))

        fig_qq.update_layout(
            title=f"Q-Q Plot - {y_name}",
            xaxis_title="Theoretical Quantiles",
            yaxis_title="Sample Quantiles",
            height=400,
            hovermode='closest'
        )

        st.plotly_chart(fig_qq, use_container_width=True)

    else:
        st.info("Residuals not available for Q-Q plot")

    # ========================================================================
    # SECTION 3: FITTED VS OBSERVED
    # ========================================================================
    st.markdown("### Fitted vs Observed Values")

    if 'y_pred' in model_results:
        y_pred = model_results['y_pred']
        y_actual = y.values if hasattr(y, 'values') else y

        fig_fitted = go.Figure()

        fig_fitted.add_trace(go.Scatter(
            x=y_actual,
            y=y_pred,
            mode='markers',
            marker=dict(size=8, color='blue', opacity=0.6),
            name='Predictions',
            hovertemplate='Observed: %{x:.3f}<br>Fitted: %{y:.3f}<extra></extra>'
        ))

        # Add perfect fit line
        min_val = min(y_actual.min(), y_pred.min())
        max_val = max(y_actual.max(), y_pred.max())
        fig_fitted.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Perfect Fit',
            line=dict(color='red', dash='dash')
        ))

        fig_fitted.update_layout(
            title=f"Fitted vs Observed - {y_name}",
            xaxis_title="Observed Values",
            yaxis_title="Fitted Values",
            height=400,
            hovermode='closest'
        )

        st.plotly_chart(fig_fitted, use_container_width=True)

    else:
        st.info("Fitted values not available")

    # ========================================================================
    # SECTION 4: LEVERAGE PLOT
    # ========================================================================
    st.markdown("### Leverage Analysis")

    if 'leverage' in model_results:
        leverage = model_results['leverage']

        fig_leverage = go.Figure()

        fig_leverage.add_trace(go.Scatter(
            x=np.arange(len(leverage)),
            y=leverage,
            mode='markers',
            marker=dict(size=8, color='green', opacity=0.6),
            name='Leverage',
            hovertemplate='Sample: %{x}<br>Leverage: %{y:.4f}<extra></extra>'
        ))

        # Add threshold line (2*p/n where p is number of parameters)
        n_samples = len(leverage)
        n_params = model_results.get('n_features', X.shape[1]) + 1
        threshold = 2 * n_params / n_samples

        fig_leverage.add_hline(y=threshold, line_dash="dash", line_color="red",
                              annotation_text=f"Threshold: {threshold:.4f}")

        fig_leverage.update_layout(
            title=f"Leverage Plot - {y_name}",
            xaxis_title="Sample Index",
            yaxis_title="Leverage",
            height=400,
            hovermode='closest'
        )

        st.plotly_chart(fig_leverage, use_container_width=True)

        # Highlight high leverage points
        high_leverage_indices = np.where(leverage > threshold)[0]
        if len(high_leverage_indices) > 0:
            st.warning(f"⚠️ {len(high_leverage_indices)} high leverage points detected: {list(high_leverage_indices)}")

    else:
        st.info("Leverage values not available in model results")

    # ========================================================================
    # SECTION 5: VIF (Variance Inflation Factor)
    # ========================================================================
    st.markdown("### Variance Inflation Factor (VIF)")

    if 'vif' in model_results:
        vif_dict = model_results['vif']

        # Filter out Intercept and NaN values
        vif_filtered = {k: v for k, v in vif_dict.items()
                       if 'Intercept' not in str(k) and not np.isnan(v)}

        if len(vif_filtered) > 0:
            vif_df = pd.DataFrame(list(vif_filtered.items()), columns=['Variable', 'VIF'])
            vif_df = vif_df.sort_values('VIF', ascending=False)

            # Create bar chart
            fig_vif = go.Figure()

            fig_vif.add_trace(go.Bar(
                x=vif_df['Variable'],
                y=vif_df['VIF'],
                marker=dict(
                    color=vif_df['VIF'],
                    colorscale='Reds',
                    showscale=False
                ),
                hovertemplate='%{x}<br>VIF: %{y:.2f}<extra></extra>'
            ))

            # Add threshold line (VIF = 10)
            fig_vif.add_hline(y=10, line_dash="dash", line_color="red",
                             annotation_text="VIF=10 (threshold)")

            fig_vif.update_layout(
                title=f"Variance Inflation Factor - {y_name}",
                xaxis_title="Variable",
                yaxis_title="VIF",
                height=400,
                hovermode='x unified'
            )

            st.plotly_chart(fig_vif, use_container_width=True)

            # Display table
            st.markdown("#### VIF Values Table")
            st.dataframe(vif_df, use_container_width=True, hide_index=True)

            # Interpretation
            high_vif = vif_df[vif_df['VIF'] > 10]
            if len(high_vif) > 0:
                st.warning(f"⚠️ High VIF (>10) detected for: {', '.join(high_vif['Variable'].values)}")

        else:
            st.info("No VIF values available")

    else:
        st.info("VIF analysis not available")

    # ========================================================================
    # SECTION 6: MODEL SUMMARY TABLE
    # ========================================================================
    st.markdown("### Model Summary")

    summary_data = {
        'Statistic': [
            'R² (Coefficient of Determination)',
            'Adjusted R²',
            'RMSE (Root Mean Squared Error)',
            'Q² (Cross-validation)',
            'RMSECV (CV Error)',
            'DOF (Degrees of Freedom)',
            'Number of Samples',
            'Number of Parameters',
            'Model Status'
        ],
        'Value': [
            f"{model_results.get('r_squared', np.nan):.6f}",
            f"{model_results.get('adj_r_squared', np.nan):.6f}",
            f"{model_results.get('rmse', np.nan):.6f}",
            f"{model_results.get('q2', np.nan):.6f}",
            f"{model_results.get('rmsecv', np.nan):.6f}",
            f"{model_results.get('dof', np.nan):.0f}",
            f"{len(y)}",
            f"{model_results.get('n_features', X.shape[1]) + 1}",
            "✅ OK" if model_results.get('dof', 0) > 0 else "⚠️ Saturated"
        ]
    }

    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)
