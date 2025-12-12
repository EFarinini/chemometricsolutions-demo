"""
Mixture Model Diagnostics Module
Diagnostic plots and statistics for Scheffe polynomial models
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats


def calculate_mixture_leverage(model_results):
    """
    Calculate leverage values for mixture model

    Returns:
        pd.DataFrame with leverage analysis
    """
    leverage = model_results['leverage']
    n_samples = model_results['n_samples']
    n_features = model_results['n_features']

    # High leverage threshold: 2p/n
    threshold = 2 * n_features / n_samples

    leverage_df = pd.DataFrame({
        'Experiment': range(1, n_samples + 1),
        'Leverage': leverage,
        'IsHighLeverage': leverage > threshold,
        'Threshold': threshold
    })

    return leverage_df


def show_mixture_diagnostics_ui(model_results):
    """
    Display mixture model diagnostics.

    When DOF = 0: show ONLY warning message (no diagnostics)
    When DOF > 0: show FULL diagnostics
    """
    dof = model_results.get('dof', 1)
    feature_names = model_results.get('feature_names', [])

    # ═══════════════════════════════════════════════════════════════════════════
    # EARLY EXIT: DOF ≤ 0 → Show warning and RETURN (don't show diagnostics)
    # ═══════════════════════════════════════════════════════════════════════════

    if dof <= 0:
        st.info("""
⚠️ **Saturated Model (DOF = 0)**

Your design perfectly matches the model complexity. All parameters are exactly identified.

✅ **The model IS valid for:**
- Predictions on new mixtures
- Sensitivity analysis
- Optimization

❌ **NOT available:**
- Diagnostics (model is saturated)
- R², Q², RMSE (indeterminate)
- p-values, Confidence Intervals
- Diagnostic plots

**Proceed to the next Tab for predictions and response surface visualization.**
        """)

        return  # ← Just return, don't show diagnostics

    # ═══════════════════════════════════════════════════════════════════════════
    # DOF > 0: Show FULL diagnostics
    # ═══════════════════════════════════════════════════════════════════════════

    st.markdown("## 📊 Model Diagnostics")

    # Display model type
    degree = model_results.get('degree', 'unknown')
    model_name_map = {
        'linear': 'Linear Scheffe Model',
        'reduced_cubic': 'Reduced Cubic Scheffe Model (Linear + Ternary)',
        'quadratic': 'Quadratic Scheffe Model',
        'cubic': 'Full Cubic Scheffe Model'
    }
    model_name = model_name_map.get(degree, f'{degree.upper()} Scheffe Model')

    st.info(f"📋 **Model:** {model_name}")
    st.success("✓ Model fitted with degrees of freedom for full diagnostics")

    # ═ Model Solution ═
    st.markdown("### 📋 Model Solution")

    # Dispersion Matrix
    st.markdown("**Dispersion Matrix (X'X)⁻¹**")
    disp_matrix = model_results.get('XtX_inv', None)
    if disp_matrix is not None:
        disp_df = pd.DataFrame(
            disp_matrix,
            index=feature_names,
            columns=feature_names
        )
        st.dataframe(disp_df.style.format('{:.4f}'), use_container_width=True)

    # Trace
    trace = np.trace(disp_matrix) if disp_matrix is not None else 0
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Trace", f"{trace:.4f}")

    # Leverage
    st.markdown("**Leverage of Experimental Points**")
    leverage = model_results.get('leverage', np.array([]))

    # ← FIX: Check array length properly
    if isinstance(leverage, np.ndarray) and len(leverage) > 0:
        lev_df = pd.DataFrame({
            'Experiment': range(1, len(leverage) + 1),
            'Leverage': leverage
        })
        st.dataframe(lev_df.style.format({'Leverage': '{:.4f}'}), use_container_width=True)

        # ← FIX: Use np.max() instead of max() for arrays
        max_lev = np.max(leverage)
        avg_lev = np.mean(leverage)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Max Leverage", f"{max_lev:.4f}")
        with col2:
            st.metric("Avg Leverage", f"{avg_lev:.4f}")
    else:
        st.warning("No leverage data available")

    st.markdown("---")

    # ═ Model Fit Quality ═
    st.markdown("### 📊 Model Fit Quality")

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("R²", f"{model_results['r_squared']:.6f}")
    with col2:
        st.metric("Q² (CV)", f"{model_results.get('q2', np.nan):.6f}")
    with col3:
        st.metric("RMSE", f"{model_results['rmse']:.6f}")
    with col4:
        st.metric("RMSECV", f"{model_results.get('rmsecv', np.nan):.6f}")
    with col5:
        st.metric("DOF", dof)

    st.markdown("---")

    # ═ Coefficients with SE/CI/p-value ═
    st.markdown("### 📈 Scheffe Polynomial Coefficients")

    # Identify linear terms (pure components) - no significance testing
    linear_terms = {term for term in feature_names if '*' not in term}

    # Get coefficient statistics
    se_coef = model_results.get('se_coef', [np.nan] * len(feature_names))
    ci_upper = model_results.get('ci_upper', [np.nan] * len(feature_names))
    p_values = model_results.get('p_values', [np.nan] * len(feature_names))

    # Build significance markers (only for interaction terms)
    sig_marks = []
    for term, p_val in zip(feature_names, p_values):
        if term in linear_terms:
            sig_marks.append('')  # No marker for pure components
        else:
            if not np.isnan(p_val):
                if p_val < 0.001:
                    sig_marks.append('***')
                elif p_val < 0.01:
                    sig_marks.append('**')
                elif p_val < 0.05:
                    sig_marks.append('*')
                else:
                    sig_marks.append('')
            else:
                sig_marks.append('')

    coef_df = pd.DataFrame({
        'Term': feature_names,
        'Coefficient': model_results['coefficients'],
        'Std.dev.': se_coef,
        'Conf.Int. (±)': [f"{ci - coef:.6f}" if not np.isnan(ci) else "N/A"
                          for ci, coef in zip(ci_upper, model_results['coefficients'])],
        'p-value': p_values,
        'Sig.': sig_marks
    })

    st.dataframe(coef_df.style.format({
        'Coefficient': '{:.6f}',
        'Std.dev.': '{:.6f}',
        'p-value': '{:.6f}'
    }), use_container_width=True, hide_index=True)

    st.caption("""
**Significance codes:** *** p<0.001, ** p<0.01, * p<0.05

**Note:** Pure components (linear terms) have no significance markers - they are constrained by the mixture property (Σ = 1.0).
Only interaction terms can be tested for statistical significance.
    """)

    st.markdown("---")

    # ═ Fitted Values & Residuals ═
    st.markdown("### 📊 Fitted Values & Residuals")

    fitted_df = pd.DataFrame({
        'Exp': range(1, len(model_results['y']) + 1),
        'Observed': model_results['y'],
        'Predicted': model_results['y_pred'],
        'Residuals': model_results['residuals']
    })

    st.dataframe(fitted_df.style.format({
        'Observed': '{:.6f}',
        'Predicted': '{:.6f}',
        'Residuals': '{:.6f}'
    }), use_container_width=True, hide_index=True, height=min(300, (len(fitted_df) + 1) * 35 + 3))

    st.markdown("---")

    # ═ Cross-Validation ═
    st.markdown("### 🔄 Cross-Validation (Leave-One-Out)")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("RMSECV", f"{model_results.get('rmsecv', np.nan):.6f}")
    with col2:
        st.metric("Q²", f"{model_results.get('q2', np.nan):.6f}")
    with col3:
        r2 = model_results['r_squared']
        q2 = model_results.get('q2', np.nan)
        if not np.isnan(q2):
            gap = abs(r2 - q2)
            st.metric("R² - Q² Gap", f"{gap:.4f}",
                     delta="Good" if gap < 0.1 else "Check" if gap < 0.2 else "Overfitting")
        else:
            st.metric("R² - Q² Gap", "N/A")

    # CV predictions table (optional expander)
    if 'cv_predictions' in model_results and len(model_results['cv_predictions']) > 0:
        cv_residuals = model_results.get('cv_residuals', [])
        cv_table = pd.DataFrame({
            'Exp': range(1, len(model_results['y']) + 1),
            'Observed': model_results['y'],
            'CV Predicted': model_results['cv_predictions'],
            'CV Residual': cv_residuals
        })

        with st.expander("📊 View CV Predictions", expanded=False):
            st.dataframe(cv_table.style.format({
                'Observed': '{:.6f}',
                'CV Predicted': '{:.6f}',
                'CV Residual': '{:.6f}'
            }), use_container_width=True, hide_index=True)

    st.markdown("---")

    # ═ Diagnostic Plots ═
    st.markdown("### 📈 Diagnostic Plots")

    residuals = model_results['residuals']
    y_pred = model_results['y_pred']

    col_plot1, col_plot2 = st.columns(2)

    with col_plot1:
        # Residuals vs Fitted
        fig_res = go.Figure()
        fig_res.add_trace(go.Scatter(
            x=y_pred,
            y=residuals,
            mode='markers',
            marker=dict(size=10, color='blue', opacity=0.6),
            text=[f"Obs {i+1}" for i in range(len(residuals))],
            hovertemplate='Fitted: %{x:.4f}<br>Residual: %{y:.4f}<br>%{text}<extra></extra>'
        ))
        fig_res.add_hline(y=0, line_dash="dash", line_color="red", line_width=2)
        fig_res.update_layout(
            title="Residuals vs Fitted Values",
            xaxis_title="Fitted Values",
            yaxis_title="Residuals",
            height=400,
            hovermode='closest'
        )
        st.plotly_chart(fig_res, use_container_width=True)

    with col_plot2:
        # Leverage plot
        n_features = model_results['n_features']
        n_samples = len(leverage) if isinstance(leverage, np.ndarray) else 0
        threshold = 2 * n_features / n_samples if n_samples > 0 else 1

        if isinstance(leverage, np.ndarray) and len(leverage) > 0:
            fig_lev = go.Figure()
            fig_lev.add_trace(go.Scatter(
                x=list(range(1, len(leverage) + 1)),
                y=leverage,
                mode='markers',
                marker=dict(
                    size=10,
                    color=['red' if lev > threshold else 'blue' for lev in leverage],
                    opacity=0.6
                ),
                text=[f"Exp {i+1}: {lev:.4f}" for i, lev in enumerate(leverage)],
                hovertemplate='%{text}<extra></extra>'
            ))
            fig_lev.add_hline(y=threshold, line_dash="dash", line_color="red", line_width=2,
                             annotation_text=f"Threshold ({threshold:.4f})")
            fig_lev.update_layout(
                title="Leverage Plot",
                xaxis_title="Observation",
                yaxis_title="Leverage",
                height=400,
                hovermode='closest'
            )
            st.plotly_chart(fig_lev, use_container_width=True)

    # Q-Q plot for normality
    from scipy.stats import probplot

    qq_data = probplot(residuals, dist="norm")

    fig_qq = go.Figure()
    fig_qq.add_trace(go.Scatter(
        x=qq_data[0][0],
        y=qq_data[0][1],
        mode='markers',
        marker=dict(size=8, color='blue', opacity=0.6),
        name='Data'
    ))
    fig_qq.add_trace(go.Scatter(
        x=qq_data[0][0],
        y=qq_data[1][1] + qq_data[1][0] * qq_data[0][0],
        mode='lines',
        line=dict(color='red', dash='dash', width=2),
        name='Normal Reference'
    ))
    fig_qq.update_layout(
        title="Q-Q Plot (Normality Check)",
        xaxis_title="Theoretical Quantiles",
        yaxis_title="Sample Quantiles",
        height=400,
        hovermode='closest'
    )
    st.plotly_chart(fig_qq, use_container_width=True)

    st.info("✅ **Model diagnostics complete.** Proceed to Tab 3 for response surface visualization.")
