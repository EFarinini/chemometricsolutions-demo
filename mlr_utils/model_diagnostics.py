"""
MLR Model Diagnostics UI
Equivalent to DOE diagnostic plots (DOE_experimental_fitted.r, DOE_residuals_fitting.r, etc.)
Interactive diagnostic plots for model evaluation
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go


def show_model_diagnostics_ui():
    """
    Display the MLR Model Diagnostics UI with various diagnostic plots

    Requires model results to be stored in st.session_state.mlr_model

    GENERIC IMPLEMENTATION:
    - ALWAYS shows: VIF, Leverage, Correlation matrix (independent of DoF)
    - CONDITIONAL: R², RMSE, residual plots (require DoF > 0)
    - Works with any design: screening, factorial, custom, saturated, etc.
    """
    st.markdown("## 📊 Model Diagnostics")
    st.markdown("*Equivalent to DOE diagnostic plots*")

    if 'mlr_model' not in st.session_state:
        st.warning("⚠️ No MLR model fitted. Please fit a model first.")
        return

    model_results = st.session_state.mlr_model

    # ===== CHECK DEGREES OF FREEDOM =====
    has_residual_diagnostics = 'dof' in model_results and model_results['dof'] > 0

    if not has_residual_diagnostics:
        st.warning("⚠️ **Saturated model** (samples = parameters). Limited diagnostics available.")
        st.info("""
        **Your model has:**
        - Samples: {n_samples}
        - Parameters: {n_features}
        - Degrees of freedom: {dof}

        **Available diagnostics** (independent of DoF):
        - VIF (multicollinearity)
        - Leverage (influential points)
        - Coefficients display

        **Unavailable** (require DoF > 0):
        - R², RMSE (no residual variance)
        - Residual plots
        - Statistical tests (p-values, t-stats)
        """.format(
            n_samples=model_results.get('n_samples', 'N/A'),
            n_features=model_results.get('n_features', 'N/A'),
            dof=model_results.get('dof', 'N/A')
        ))

    # Build diagnostic options based on available data
    diagnostic_options = []

    # ALWAYS available (independent of DoF)
    diagnostic_options.extend([
        "🎯 Leverage Plot",
        "📐 Coefficients Bar Plot",
        "🔢 VIF & Multicollinearity"
    ])

    # CONDITIONAL: Only if DoF > 0
    if has_residual_diagnostics:
        diagnostic_options.extend([
            "📈 Experimental vs Fitted",
            "📉 Residuals vs Fitted"
        ])

    # CONDITIONAL: Only if CV available
    if 'cv_predictions' in model_results:
        diagnostic_options.extend([
            "🔄 Experimental vs CV Predicted",
            "📊 CV Residuals"
        ])

    # Diagnostic plot selector
    diagnostic_type = st.selectbox(
        "Select diagnostic plot:",
        diagnostic_options
    )

    # Display the selected diagnostic plot
    if diagnostic_type == "📈 Experimental vs Fitted":
        _plot_experimental_vs_fitted(model_results)

    elif diagnostic_type == "📉 Residuals vs Fitted":
        _plot_residuals_vs_fitted(model_results)

    elif diagnostic_type == "🔄 Experimental vs CV Predicted":
        _plot_experimental_vs_cv(model_results)

    elif diagnostic_type == "📊 CV Residuals":
        _plot_cv_residuals(model_results)

    elif diagnostic_type == "🎯 Leverage Plot":
        _plot_leverage(model_results)

    elif diagnostic_type == "📐 Coefficients Bar Plot":
        _plot_coefficients_bar(model_results)

    elif diagnostic_type == "🔢 VIF & Multicollinearity":
        _display_vif_multicollinearity(model_results)


def _plot_experimental_vs_fitted(model_results):
    """
    Plot experimental vs fitted values
    Equivalent to DOE_experimental_fitted.r
    """
    st.markdown("### 📈 Experimental vs Fitted Values")

    y_exp = model_results['y'].values
    y_pred = model_results['y_pred']

    # Calculate limits for 1:1 line
    min_val = min(y_exp.min(), y_pred.min())
    max_val = max(y_exp.max(), y_pred.max())
    margin = (max_val - min_val) * 0.05
    limits = [min_val - margin, max_val + margin]

    fig = go.Figure()

    # Add points with sample numbers
    fig.add_trace(go.Scatter(
        x=y_exp,
        y=y_pred,
        mode='markers+text',
        text=[str(i+1) for i in range(len(y_exp))],
        textposition="top center",
        marker=dict(size=8, color='red'),
        name='Samples'
    ))

    # Add 1:1 line
    fig.add_trace(go.Scatter(
        x=limits,
        y=limits,
        mode='lines',
        line=dict(color='green', dash='solid'),
        name='1:1 line'
    ))

    fig.update_layout(
        title=f"Experimental vs Fitted - {st.session_state.mlr_y_var}",
        xaxis_title="Experimental Value",
        yaxis_title="Fitted Value",
        height=600,
        width=600,
        xaxis=dict(range=limits, scaleanchor="y", scaleratio=1),
        yaxis=dict(range=limits),
        showlegend=True
    )

    st.plotly_chart(fig, use_container_width=True)

    # Statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("R²", f"{model_results['r_squared']:.4f}")
    with col2:
        st.metric("RMSE", f"{model_results['rmse']:.4f}")
    with col3:
        correlation = np.corrcoef(y_exp, y_pred)[0, 1]
        st.metric("Correlation", f"{correlation:.4f}")


def _plot_residuals_vs_fitted(model_results):
    """
    Plot residuals vs fitted values
    Equivalent to DOE_residuals_fitting.r
    """
    st.markdown("### 📉 Residuals vs Fitted Values")

    y_pred = model_results['y_pred']
    residuals = model_results['residuals']

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=y_pred,
        y=residuals,
        mode='markers',
        marker=dict(size=8, color='red'),
        name='Residuals'
    ))

    # Add zero line
    fig.add_hline(y=0, line_dash="dash", line_color="green")

    fig.update_layout(
        title=f"Residuals vs Fitted - {st.session_state.mlr_y_var}",
        xaxis_title="Fitted Value",
        yaxis_title="Residual",
        height=500,
        showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True)

    # Display statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Mean Residual", f"{residuals.mean():.6f}")
    with col2:
        st.metric("Std Residual", f"{residuals.std():.4f}")
    with col3:
        st.metric("Max |Residual|", f"{np.abs(residuals).max():.4f}")


def _plot_experimental_vs_cv(model_results):
    """Plot experimental vs cross-validation predicted values"""
    st.markdown("### 🔄 Experimental vs CV Predicted Values")

    if 'cv_predictions' not in model_results:
        st.warning("⚠️ No cross-validation results available. Run model with CV enabled.")
        return

    y_exp = model_results['y'].values
    y_cv = model_results['cv_predictions']

    # Calculate limits for 1:1 line
    min_val = min(y_exp.min(), y_cv.min())
    max_val = max(y_exp.max(), y_cv.max())
    margin = (max_val - min_val) * 0.05
    limits = [min_val - margin, max_val + margin]

    fig = go.Figure()

    # Add points with sample numbers
    fig.add_trace(go.Scatter(
        x=y_exp,
        y=y_cv,
        mode='markers+text',
        text=[str(i+1) for i in range(len(y_exp))],
        textposition="top center",
        marker=dict(size=8, color='blue'),
        name='Samples'
    ))

    # Add 1:1 line
    fig.add_trace(go.Scatter(
        x=limits,
        y=limits,
        mode='lines',
        line=dict(color='green', dash='solid'),
        name='1:1 line'
    ))

    fig.update_layout(
        title=f"Experimental vs CV Predicted - {st.session_state.mlr_y_var}",
        xaxis_title="Experimental Value",
        yaxis_title="CV Predicted Value",
        height=600,
        width=600,
        xaxis=dict(range=limits, scaleanchor="y", scaleratio=1),
        yaxis=dict(range=limits),
        showlegend=True
    )

    st.plotly_chart(fig, use_container_width=True)

    # Statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Q²", f"{model_results['q2']:.4f}")
    with col2:
        st.metric("RMSECV", f"{model_results['rmsecv']:.4f}")
    with col3:
        correlation = np.corrcoef(y_exp, y_cv)[0, 1]
        st.metric("Correlation", f"{correlation:.4f}")


def _plot_cv_residuals(model_results):
    """Plot cross-validation residuals"""
    st.markdown("### 📊 CV Residuals")

    if 'cv_residuals' not in model_results:
        st.warning("⚠️ No cross-validation results available. Run model with CV enabled.")
        return

    cv_residuals = model_results['cv_residuals']
    sample_numbers = list(range(1, len(cv_residuals) + 1))

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=sample_numbers,
        y=cv_residuals,
        mode='markers',
        marker=dict(size=8, color='blue'),
        name='CV Residuals'
    ))

    # Add zero line
    fig.add_hline(y=0, line_dash="dash", line_color="green")

    fig.update_layout(
        title=f"CV Residuals - {st.session_state.mlr_y_var}",
        xaxis_title="Sample Number",
        yaxis_title="CV Residual",
        height=500,
        showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True)

    # Display statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Mean CV Residual", f"{cv_residuals.mean():.6f}")
    with col2:
        st.metric("Std CV Residual", f"{cv_residuals.std():.4f}")
    with col3:
        st.metric("Max |CV Residual|", f"{np.abs(cv_residuals).max():.4f}")


def _plot_leverage(model_results):
    """Plot leverage (hat values) for each sample"""
    st.markdown("### 🎯 Leverage Plot")

    leverage = model_results['leverage']
    sample_numbers = list(range(1, len(leverage) + 1))

    # Calculate critical leverage threshold
    n_samples = model_results['n_samples']
    n_features = model_results['n_features']
    critical_leverage = 2 * n_features / n_samples

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=sample_numbers,
        y=leverage,
        mode='markers+text',
        text=[str(i) for i in sample_numbers],
        textposition="top center",
        marker=dict(size=8, color='red'),
        name='Leverage'
    ))

    # Add critical leverage line
    fig.add_hline(
        y=critical_leverage,
        line_dash="dash",
        line_color="orange",
        annotation_text=f"Critical leverage: {critical_leverage:.4f}",
        annotation_position="right"
    )

    fig.update_layout(
        title=f"Leverage Plot - {st.session_state.mlr_y_var}",
        xaxis_title="Sample Number",
        yaxis_title="Leverage",
        height=500,
        showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True)

    # Identify high leverage points
    high_leverage = [i for i, lev in enumerate(leverage, 1) if lev > critical_leverage]

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Max Leverage", f"{leverage.max():.4f}")
    with col2:
        st.metric("Mean Leverage", f"{leverage.mean():.4f}")
    with col3:
        st.metric("High Leverage Points", len(high_leverage))

    if high_leverage:
        st.warning(f"⚠️ High leverage points detected: {', '.join(map(str, high_leverage))}")
        st.info("""
        **High leverage points** are samples with unusual predictor values.
        They have a strong influence on the model fit and should be examined carefully.
        """)
    else:
        st.success("✅ No high leverage points detected")


def _plot_coefficients_bar(model_results):
    """Plot model coefficients as bar chart"""
    st.markdown("### 📐 Model Coefficients")

    coefficients = model_results['coefficients']

    # Filter out intercept term
    coef_no_intercept = coefficients[coefficients.index != 'Intercept']
    coef_names = coef_no_intercept.index.tolist()

    if len(coef_names) == 0:
        st.warning("No coefficients to plot (model contains only intercept)")
        return

    # Determine colors based on coefficient type
    colors = []
    for name in coef_names:
        if '*' in name:
            colors.append('green')  # Interactions
        elif '^2' in name:
            colors.append('cyan')  # Quadratic
        else:
            colors.append('red')  # Linear

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=coef_names,
        y=coef_no_intercept.values,
        marker_color=colors,
        name='Coefficients'
    ))

    # Add error bars if available
    if 'ci_lower' in model_results:
        ci_lower = model_results['ci_lower'][coef_no_intercept.index].values
        ci_upper = model_results['ci_upper'][coef_no_intercept.index].values

        for i, name in enumerate(coef_names):
            fig.add_trace(go.Scatter(
                x=[name, name],
                y=[ci_lower[i], ci_upper[i]],
                mode='lines',
                line=dict(color='black', width=2),
                showlegend=False
            ))

    # Add significance markers
    if 'p_values' in model_results:
        p_values = model_results['p_values'][coef_no_intercept.index].values
        for i, (name, coef, p) in enumerate(zip(coef_names, coef_no_intercept.values, p_values)):
            if p <= 0.001:
                fig.add_annotation(x=name, y=coef, text='***', showarrow=False, font=dict(size=16))
            elif p <= 0.01:
                fig.add_annotation(x=name, y=coef, text='**', showarrow=False, font=dict(size=16))
            elif p <= 0.05:
                fig.add_annotation(x=name, y=coef, text='*', showarrow=False, font=dict(size=16))

    fig.update_layout(
        title=f"Coefficients - {st.session_state.mlr_y_var} (excluding intercept)",
        xaxis_title="Term",
        yaxis_title="Coefficient Value",
        height=600,
        xaxis={'tickangle': 45}
    )

    st.plotly_chart(fig, use_container_width=True)

    st.info("Red=Linear, Green=Interactions, Cyan=Quadratic")
    st.info("Significance: *** p≤0.001, ** p≤0.01, * p≤0.05")


def _display_vif_multicollinearity(model_results):
    """
    Display VIF and multicollinearity diagnostics

    ALWAYS AVAILABLE - Independent of DoF:
    - VIF calculated from X matrix structure only
    - Does not require residuals or statistical tests
    - Formula: VIF_j = sum(X_centered_j^2) * diag(XtX_inv)_j
    """
    st.markdown("### 🔢 VIF & Multicollinearity Analysis")

    st.info("""
    **Variance Inflation Factors (VIF)** measure multicollinearity among predictors.
    - VIF calculated from X matrix structure only (independent of residuals/DoF)
    - High VIF indicates predictor is highly correlated with other predictors
    """)

    # ===== VIF DISPLAY =====
    if 'vif' in model_results and model_results['vif'] is not None:
        st.markdown("#### Variance Inflation Factors (VIF)")

        vif_df = model_results['vif'].to_frame('VIF')
        # Remove intercept and NaN values
        vif_df_clean = vif_df[~vif_df.index.str.contains('Intercept', case=False, na=False)]
        vif_df_clean = vif_df_clean.dropna()

        if not vif_df_clean.empty:
            def interpret_vif(vif_val):
                if vif_val <= 1:
                    return "✅ No correlation"
                elif vif_val <= 2:
                    return "✅ OK"
                elif vif_val <= 4:
                    return "⚠️ Good"
                elif vif_val <= 8:
                    return "⚠️ Acceptable"
                else:
                    return "❌ High multicollinearity"

            vif_df_clean['Interpretation'] = vif_df_clean['VIF'].apply(interpret_vif)

            # Sort by VIF descending to show problematic terms first
            vif_df_clean = vif_df_clean.sort_values('VIF', ascending=False)

            st.dataframe(vif_df_clean.round(4), use_container_width=True)

            # Summary statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Max VIF", f"{vif_df_clean['VIF'].max():.2f}")
            with col2:
                st.metric("Mean VIF", f"{vif_df_clean['VIF'].mean():.2f}")
            with col3:
                problematic = (vif_df_clean['VIF'] > 8).sum()
                st.metric("VIF > 8", problematic)

            st.info("""
            **VIF Interpretation:**
            - VIF = 1: No correlation with other predictors
            - VIF < 2: Low multicollinearity (OK)
            - VIF < 4: Moderate multicollinearity (Good)
            - VIF < 8: High multicollinearity (Acceptable)
            - VIF > 8: Very high multicollinearity (Problematic)

            **High VIF indicates:**
            - Predictor is highly correlated with other predictors
            - Coefficient estimates may be unstable
            - Consider removing or combining correlated predictors
            """)

            # Highlight problematic VIFs
            if problematic > 0:
                st.warning(f"⚠️ {problematic} predictor(s) with VIF > 8 detected!")
                problematic_terms = vif_df_clean[vif_df_clean['VIF'] > 8].index.tolist()
                st.write(f"**Problematic terms:** {', '.join(problematic_terms)}")
            else:
                st.success("✅ No severe multicollinearity detected (all VIF ≤ 8)")
        else:
            st.info("VIF not applicable for this model (single predictor or no variation)")
    else:
        st.warning("⚠️ VIF not calculated for this model")

    # ===== CORRELATION MATRIX =====
    if 'X' in model_results:
        st.markdown("---")
        st.markdown("#### Predictor Correlation Matrix")

        try:
            X_df = model_results['X']
            # Remove intercept column if present
            X_no_intercept = X_df[[col for col in X_df.columns if col.lower() != 'intercept']]

            if not X_no_intercept.empty and len(X_no_intercept.columns) > 1:
                # Calculate correlation matrix
                corr_matrix = X_no_intercept.corr()

                # Display as heatmap using plotly
                import plotly.graph_objects as go

                fig = go.Figure(data=go.Heatmap(
                    z=corr_matrix.values,
                    x=corr_matrix.columns,
                    y=corr_matrix.columns,
                    colorscale='RdBu_r',
                    zmid=0,
                    zmin=-1,
                    zmax=1,
                    text=corr_matrix.values,
                    texttemplate='%{text:.2f}',
                    textfont={"size": 10},
                    colorbar=dict(title="Correlation")
                ))

                fig.update_layout(
                    title="Predictor Correlation Matrix",
                    xaxis_title="Predictors",
                    yaxis_title="Predictors",
                    height=max(400, len(corr_matrix.columns) * 40),
                    xaxis={'tickangle': 45}
                )

                st.plotly_chart(fig, use_container_width=True)

                # Find high correlations
                high_corr_threshold = 0.8
                high_corr_pairs = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        corr_val = abs(corr_matrix.iloc[i, j])
                        if corr_val > high_corr_threshold:
                            high_corr_pairs.append((
                                corr_matrix.columns[i],
                                corr_matrix.columns[j],
                                corr_val
                            ))

                if high_corr_pairs:
                    st.warning(f"⚠️ {len(high_corr_pairs)} pair(s) with |correlation| > {high_corr_threshold}")
                    with st.expander("Show high correlation pairs"):
                        for var1, var2, corr_val in high_corr_pairs:
                            st.write(f"- **{var1}** ↔ **{var2}**: {corr_val:.3f}")
                else:
                    st.success(f"✅ No extreme correlations (|r| > {high_corr_threshold}) detected")

                st.info("""
                **Correlation Matrix Interpretation:**
                - Values near +1 or -1 indicate strong linear relationships
                - Values near 0 indicate weak relationships
                - High correlations (|r| > 0.8) may cause multicollinearity issues
                """)
            else:
                st.info("Correlation matrix not applicable (single predictor)")
        except Exception as e:
            st.warning(f"Could not display correlation matrix: {str(e)}")
