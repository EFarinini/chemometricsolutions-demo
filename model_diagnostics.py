"""
MLR Model Diagnostics - Core Functions and UI
Equivalent to DOE diagnostic plots (DOE_experimental_fitted.r, DOE_residuals_fitting.r, etc.)
Interactive diagnostic plots for model evaluation

This module contains:
1. Core diagnostic functions (calculate_vif, check_model_saturated)
2. UI display functions (show_model_diagnostics_ui)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Import core computation functions
try:
    from mlr_utils.model_computation import statistical_summary, fit_mlr_model
except ImportError:
    # Fallback if module structure differs
    try:
        from model_computation import statistical_summary, fit_mlr_model
    except ImportError:
        st.warning("⚠️ Could not import core computation functions")


# ============================================================================
# CORE DIAGNOSTIC FUNCTIONS
# ============================================================================

def calculate_vif(X):
    """
    Calculate VIF (Variance Inflation Factors) - ALWAYS works, independent of DoF

    VIF measures multicollinearity among predictors.
    - VIF = 1: No correlation
    - VIF < 2: OK
    - VIF < 4: Good
    - VIF < 8: Acceptable
    - VIF > 8: High multicollinearity (problematic)

    FORMULA (from R implementation):
    VIF_j = sum(X_centered_j^2) * diag((X'X)^-1)_j

    Args:
        X: DataFrame with model matrix (includes intercept, interactions, quadratics)

    Returns:
        pd.Series with VIF values (NaN for intercept)
    """
    print(f"\n[DEBUG calculate_vif]")
    print(f"  Input X shape: {X.shape}")

    # Input validation
    if not isinstance(X, pd.DataFrame):
        raise ValueError("X must be a pandas DataFrame")

    if X.empty:
        raise ValueError("X DataFrame is empty")

    X_mat = X.values
    n_samples, n_features = X_mat.shape

    print(f"  n_samples: {n_samples}, n_features: {n_features}")

    if n_features < 2:
        print(f"  Single feature - VIF not applicable")
        return pd.Series([np.nan], index=X.columns)

    try:
        # Compute (X'X)^-1
        XtX = X_mat.T @ X_mat
        XtX_inv = np.linalg.inv(XtX)

        # Center the X matrix (subtract column means)
        X_centered = X_mat - X_mat.mean(axis=0)

        vif = []
        for i in range(n_features):
            if X.columns[i] == 'Intercept' or 'intercept' in X.columns[i].lower():
                vif.append(np.nan)
            else:
                # Formula: sum(X_centered_i^2) * diag(XtX_inv)_i
                ss_centered = np.sum(X_centered[:, i]**2)
                vif_value = ss_centered * XtX_inv[i, i]
                vif.append(vif_value)

        vif_series = pd.Series(vif, index=X.columns)
        print(f"  VIF computed successfully")
        print(f"  Max VIF (excluding intercept): {vif_series.dropna().max():.2f}")

        return vif_series

    except np.linalg.LinAlgError as e:
        print(f"  ERROR: Cannot compute VIF - matrix is singular: {e}")
        return pd.Series([np.nan] * n_features, index=X.columns)
    except Exception as e:
        print(f"  ERROR: VIF calculation failed: {e}")
        return pd.Series([np.nan] * n_features, index=X.columns)


def check_model_saturated(model_results):
    """
    Check if model is saturated (DoF <= 0)

    SATURATED MODEL:
    - Number of samples = Number of parameters
    - Perfect fit to training data (no residual variance)
    - Cannot estimate statistical tests (p-values, t-stats)
    - Cannot compute R², RMSE
    - CAN still compute: VIF, Leverage, Coefficients

    Args:
        model_results: dict from fit_mlr_model()

    Returns:
        bool: True if saturated (DoF <= 0), False otherwise
    """
    print(f"\n[DEBUG check_model_saturated]")

    # Check if DoF key exists
    if 'dof' not in model_results:
        print(f"  WARNING: 'dof' key not found in model_results")
        return True  # Assume saturated if DoF not available

    dof = model_results['dof']
    print(f"  DoF = {dof}")

    is_saturated = dof <= 0

    if is_saturated:
        print(f"  MODEL IS SATURATED (DoF <= 0)")
        print(f"    - Cannot compute: Adjusted R², RMSE, residual plots, p-values")
        print(f"    - Can compute: VIF, Leverage, Coefficients")
    else:
        print(f"  Model has adequate DoF (DoF > 0)")

    return is_saturated


# ============================================================================
# UI DISPLAY FUNCTIONS
# ============================================================================


def show_model_diagnostics_ui(model_results=None, X=None, y=None):
    """
    Display the MLR Model Diagnostics UI with various diagnostic plots

    GENERIC IMPLEMENTATION:
    - ALWAYS shows: VIF, Leverage, Correlation matrix (independent of DoF)
    - CONDITIONAL: R², RMSE, residual plots (require DoF > 0)
    - Works with any design: screening, factorial, custom, saturated, etc.

    LOGIC:
    - IF saturated (DoF <= 0): show warning, display only VIF/Leverage/Correlation
    - ELSE: show R², RMSE, residual plots, Q², RMSECV

    Args:
        model_results: dict from fit_mlr_model() (optional, uses session_state if None)
        X: original predictor DataFrame (optional)
        y: original response Series (optional)
    """
    st.markdown("## 📊 Model Diagnostics")
    st.markdown("*Equivalent to DOE diagnostic plots*")

    # Get model results from session state if not provided
    if model_results is None:
        if 'mlr_model' not in st.session_state:
            st.warning("⚠️ No MLR model fitted. Please fit a model first.")
            return
        model_results = st.session_state.mlr_model

    # ===== CHECK IF MODEL IS SATURATED =====
    is_saturated = check_model_saturated(model_results)

    if is_saturated:
        st.error("⚠️ **SATURATED MODEL** (DoF ≤ 0)")
        st.warning("""
        **Your model has:**
        - Samples: {n_samples}
        - Parameters: {n_features}
        - Degrees of freedom: {dof}

        A saturated model has **no residual variance** because the number of parameters equals (or exceeds) the number of samples.
        This means the model fits the training data perfectly, but statistical inference is not possible.

        **✅ Available diagnostics** (independent of DoF):
        - ✅ VIF (Variance Inflation Factors) - multicollinearity check
        - ✅ Leverage (Hat values) - influential points
        - ✅ Correlation Matrix - predictor correlations
        - ✅ Coefficients display

        **❌ Unavailable** (require DoF > 0):
        - ❌ Adjusted R², RMSE (no residual variance to estimate)
        - ❌ Residual plots (residuals are zero by definition)
        - ❌ Statistical tests (p-values, confidence intervals)
        - ❌ Cross-validation

        **Recommendation**: Collect more samples or reduce model complexity (remove terms).
        """.format(
            n_samples=model_results.get('n_samples', 'N/A'),
            n_features=model_results.get('n_features', 'N/A'),
            dof=model_results.get('dof', 'N/A')
        ))
    else:
        # Model has adequate DoF
        st.success(f"✅ Model has adequate degrees of freedom (DoF = {model_results.get('dof', 'N/A')})")

        # Display summary stats if available
        if 'r_squared' in model_results and 'rmse' in model_results:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Adjusted R²", f"{model_results['r_squared']:.4f}")
            with col2:
                st.metric("RMSE", f"{model_results['rmse']:.4f}")
            with col3:
                if 'q2' in model_results:
                    st.metric("Q² (CV)", f"{model_results['q2']:.4f}")
                else:
                    st.metric("DoF", model_results['dof'])

    st.markdown("---")

    # Build diagnostic options based on available data and saturation status
    diagnostic_options = []

    # ALWAYS available (independent of DoF)
    diagnostic_options.extend([
        "🔢 VIF & Multicollinearity",
        "🎯 Leverage Plot",
        "📐 Coefficients Bar Plot"
    ])

    # CONDITIONAL: Only if NOT saturated (DoF > 0)
    if not is_saturated:
        if 'r_squared' in model_results and 'rmse' in model_results:
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

    REQUIRES:
    - y, y_pred (predictions)
    - r_squared, rmse (optional for display)
    """
    st.markdown("### 📈 Experimental vs Fitted Values")

    # Defensive checks
    if 'y' not in model_results or 'y_pred' not in model_results:
        st.error("❌ Missing required data: 'y' or 'y_pred' not in model results")
        return

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

    # Statistics (defensive - only show if available)
    col1, col2, col3 = st.columns(3)
    with col1:
        if 'r_squared' in model_results:
            st.metric("Adjusted R²", f"{model_results['r_squared']:.4f}")
        else:
            st.metric("Adjusted R²", "N/A")
    with col2:
        if 'rmse' in model_results:
            st.metric("RMSE", f"{model_results['rmse']:.4f}")
        else:
            st.metric("RMSE", "N/A")
    with col3:
        correlation = np.corrcoef(y_exp, y_pred)[0, 1]
        st.metric("Correlation", f"{correlation:.4f}")


def _plot_residuals_vs_fitted(model_results):
    """
    Plot residuals vs fitted values
    Equivalent to DOE_residuals_fitting.r

    REQUIRES:
    - y_pred (predictions)
    - residuals
    """
    st.markdown("### 📉 Residuals vs Fitted Values")

    # Defensive checks
    if 'y_pred' not in model_results or 'residuals' not in model_results:
        st.error("❌ Missing required data: 'y_pred' or 'residuals' not in model results")
        return

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
    """
    Plot leverage (hat values) for each sample

    ALWAYS AVAILABLE - independent of DoF
    Leverage shows influential points in the design space

    REQUIRES:
    - leverage (hat values)
    - n_samples, n_features
    """
    st.markdown("### 🎯 Leverage Plot")

    # Defensive checks
    if 'leverage' not in model_results or model_results['leverage'] is None:
        st.error("❌ Leverage values not available in model results")
        return

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
    """
    Plot model coefficients as bar chart with proper styling and significance markers.

    STYLING RULES:
    - RED: Linear terms (no operators)
    - GREEN: Single interactions (single * or ^)
    - CYAN: Double interactions (multiple *) or quadratic terms (^2)
    - Black borders with line width=1
    - Error bars: asymmetric confidence intervals
    - Significance markers: *, **, *** positioned above/below bars

    REQUIRES:
    - coefficients
    - ci_lower, ci_upper (if DoF > 0)
    - p_values (if DoF > 0)
    """
    st.markdown("### 📐 Model Coefficients")

    # Defensive checks
    if 'coefficients' not in model_results or model_results['coefficients'] is None:
        st.error("❌ Coefficients not available in model results")
        return

    coefficients = model_results['coefficients']

    # Filter out intercept term
    coef_no_intercept = coefficients[coefficients.index != 'Intercept']
    coef_names = coef_no_intercept.index.tolist()

    if len(coef_names) == 0:
        st.warning("No coefficients to plot (model contains only intercept)")
        return

    # Determine colors based on coefficient type
    # RED: linear (no operators)
    # GREEN: single interaction (exactly one *)
    # CYAN: double interactions (multiple *) or quadratic (^2)
    colors = []
    for name in coef_names:
        asterisk_count = name.count('*')

        if asterisk_count > 1:
            # Multiple asterisks: double interaction
            colors.append('cyan')
        elif asterisk_count == 1:
            # Single asterisk: simple interaction
            colors.append('green')
        elif '^2' in name or '^' in name:
            # Quadratic term
            colors.append('cyan')
        else:
            # Linear term
            colors.append('red')

    # Create figure
    fig = go.Figure()

    # Add bar trace
    fig.add_trace(go.Bar(
        x=coef_names,
        y=coef_no_intercept.values,
        marker_color=colors,
        marker_line_color='black',      # Black border
        marker_line_width=1,            # Border width
        name='Coefficients',
        showlegend=False
    ))

    # Add error bars if confidence intervals available
    if 'ci_lower' in model_results and 'ci_upper' in model_results:
        ci_lower = model_results['ci_lower'][coef_no_intercept.index].values
        ci_upper = model_results['ci_upper'][coef_no_intercept.index].values

        error_minus = coef_no_intercept.values - ci_lower
        error_plus = ci_upper - coef_no_intercept.values

        fig.update_traces(
            error_y=dict(
                type='data',
                symmetric=False,        # Asymmetric error bars
                array=error_plus,
                arrayminus=error_minus,
                color='black',
                thickness=2,
                width=4
            )
        )

    # Add significance markers (*, **, ***)
    if 'p_values' in model_results:
        p_values = model_results['p_values'][coef_no_intercept.index].values

        for i, (name, coef, p) in enumerate(zip(coef_names, coef_no_intercept.values, p_values)):
            # Determine significance level and marker text
            if p <= 0.001:
                sig_text = '***'
            elif p <= 0.01:
                sig_text = '**'
            elif p <= 0.05:
                sig_text = '*'
            else:
                sig_text = None

            # Position marker above or below bar depending on coefficient sign
            if sig_text:
                y_pos = coef
                # Calculate offset as 5% of absolute coefficient value, minimum 0.01
                y_offset = max(abs(coef) * 0.05, 0.01) if coef >= 0 else -max(abs(coef) * 0.05, 0.01)

                fig.add_annotation(
                    x=name,
                    y=y_pos + y_offset,
                    text=sig_text,
                    showarrow=False,
                    font=dict(size=20, color='black'),
                    yshift=10 if coef >= 0 else -10
                )

    # Update layout
    fig.update_layout(
        title=f"Coefficients - {st.session_state.get('mlr_y_var', 'Response')} (excluding intercept)",
        xaxis_title="Term",
        yaxis_title="Coefficient Value",
        height=600,
        xaxis={'tickangle': 45},
        hovermode='x unified',
        showlegend=False,
        yaxis=dict(zeroline=True, zerolinewidth=2, zerolinecolor='gray')
    )

    st.plotly_chart(fig, use_container_width=True)

    # Display legend
    st.markdown("""
    **Color legend:**
    - Red = Linear terms
    - Green = Two-term interactions
    - Cyan = Quadratic terms
    """)
    st.info("Significance markers: *** p≤0.001, ** p≤0.01, * p≤0.05")

    # ===== FITTED MODEL FORMULA WITH UNCERTAINTY-BASED DECIMALS =====
    st.markdown("---")
    st.markdown("#### Fitted Model Formula")

    import math

    # Get response variable name
    y_var = st.session_state.get('mlr_y_var', 'Response')

    # Get intercept
    intercept = model_results['coefficients'].get('Intercept', 0)

    # Helper function: determine decimals from uncertainty
    def get_decimals_from_uncertainty(coef_value, ci_lower, ci_upper):
        """
        Determine decimal places based on confidence interval width.

        Rule: Show coefficient decimals = where CI uncertainty starts

        Examples:
        - CI: [0.02, 0.07] → width 0.05 (1st decimal) → 1 decimal
        - CI: [0.039, 0.051] → width 0.012 (2nd decimal) → 2 decimals
        - CI: [0.0440, 0.0462] → width 0.0022 (3rd decimal) → 3 decimals
        """
        # Calculate uncertainty half-width
        uncertainty_width = ci_upper - ci_lower
        uncertainty_half_width = uncertainty_width / 2

        # If no uncertainty, use 3 decimals
        if uncertainty_half_width == 0:
            return 3

        # Find order of magnitude of uncertainty
        try:
            # Get the order of magnitude
            magnitude = math.floor(math.log10(abs(uncertainty_half_width)))

            # Convert magnitude to decimal places
            # magnitude = -1 (0.1) → 1 decimal
            # magnitude = -2 (0.01) → 2 decimals
            # magnitude = -3 (0.001) → 3 decimals
            decimals = -magnitude

            # Cap at range [1, 3]
            decimals = max(1, min(3, decimals))

        except (ValueError, OverflowError):
            # Fallback to 2 decimals if calculation fails
            decimals = 2

        return decimals

    # Build fitted formula with adaptive decimals
    formula_parts = []

    # Format intercept (1-3 decimals based on its uncertainty)
    if 'ci_lower' in model_results and 'ci_upper' in model_results:
        ci_lower_intercept = model_results['ci_lower'].get('Intercept', intercept)
        ci_upper_intercept = model_results['ci_upper'].get('Intercept', intercept)
        decimals_intercept = get_decimals_from_uncertainty(intercept, ci_lower_intercept, ci_upper_intercept)
    else:
        decimals_intercept = 2  # Default if no CI available

    # Format all coefficients with adaptive decimals
    coef_formatted = {}
    all_decimals = []

    for name in coef_names:
        coef_value = coef_no_intercept[name]

        # Get decimals from uncertainty if available
        if 'ci_lower' in model_results and 'ci_upper' in model_results:
            try:
                ci_lower = model_results['ci_lower'][name]
                ci_upper = model_results['ci_upper'][name]
                decimals = get_decimals_from_uncertainty(coef_value, ci_lower, ci_upper)
            except (KeyError, TypeError):
                decimals = 2  # Fallback
        else:
            decimals = 2  # Default if no CI available

        coef_formatted[name] = (coef_value, decimals)
        all_decimals.append(decimals)

    # Special rule: If many coefficients round to zero, increase precision
    # Try formatting with current decimals
    test_formatted = []
    for name in coef_names:
        coef_value, decimals = coef_formatted[name]
        test_str = f"{coef_value:.{decimals}f}"
        test_formatted.append(test_str)

    # Count how many are "0.0" or "-0.0" or "0.00" etc
    zero_count = sum(1 for s in test_formatted if float(s) == 0.0)

    # If more than 50% are zeros, add 1 decimal to all (but cap at 3)
    if len(test_formatted) > 0 and zero_count / len(test_formatted) > 0.5:
        for name in coef_names:
            coef_value, decimals = coef_formatted[name]
            # Increase decimals by 1, cap at 3
            coef_formatted[name] = (coef_value, min(3, decimals + 1))
        decimals_intercept = min(3, decimals_intercept + 1)

    # Build formula string
    intercept_str = f"{intercept:.{decimals_intercept}f}"
    formula_parts.append(f"{y_var} = {intercept_str}")

    for name in coef_names:
        coef_value, decimals = coef_formatted[name]
        coef_str = f"{coef_value:+.{decimals}f}"
        formula_parts.append(f"{coef_str}·{name}")

    fitted_formula = " ".join(formula_parts)

    # Display in code block
    st.code(fitted_formula, language="text")

    # Optional: Show precision info
    if 'ci_lower' in model_results:
        st.info("📊 **Coefficient precision:** Decimal places based on confidence interval width")

    # Copy to clipboard button
    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        if st.button("📋 Copy to clipboard", key="copy_fitted_formula_tab2"):
            st.write("""
            <script>
            var text = `""" + fitted_formula.replace("`", "\\`") + """`
            navigator.clipboard.writeText(text);
            </script>
            """, unsafe_allow_html=True)
            st.success("✓ Formula copied!")


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
