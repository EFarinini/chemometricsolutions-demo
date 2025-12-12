"""
Mixture Model Computation Module
Scheffe Polynomial model fitting and statistical inference
Equivalent to R script: DOE_model_computation_mixt.r
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import LeaveOneOut
import warnings


# ============================================================================
# MIXTURE DESIGN DETECTION AND PREPROCESSING
# ============================================================================

def detect_mixture_design(design_df):
    """
    Auto-detect if DataFrame represents a mixture design.

    A mixture design has:
    - All numeric columns sum to ≈ 1.0 (within ±0.01 tolerance)
    - All values in [0, 1]

    Args:
        design_df: pd.DataFrame with experimental data

    Returns:
        dict with keys:
        - 'is_mixture': bool - True if mixture design detected
        - 'n_components': int - Number of components (if mixture)
        - 'component_names': list - Component column names (if mixture)
        - 'reason': str - Explanation of detection result

    Example:
        >>> df = pd.DataFrame({'X1': [0.5, 0.33], 'X2': [0.3, 0.33], 'X3': [0.2, 0.34]})
        >>> result = detect_mixture_design(df)
        >>> result['is_mixture']
        True
    """
    # Check if DataFrame is empty
    if design_df is None or len(design_df) == 0:
        return {
            'is_mixture': False,
            'n_components': 0,
            'component_names': [],
            'reason': 'Empty dataset'
        }

    # Get numeric columns only
    numeric_cols = design_df.select_dtypes(include=[np.number]).columns.tolist()

    if len(numeric_cols) < 2:
        return {
            'is_mixture': False,
            'n_components': 0,
            'component_names': [],
            'reason': f'Need at least 2 numeric columns for mixture design (found {len(numeric_cols)})'
        }

    # Extract numeric data
    numeric_data = design_df[numeric_cols].values

    # Check 1: All values in [0, 1]
    if not np.all((numeric_data >= 0) & (numeric_data <= 1)):
        out_of_range = np.sum((numeric_data < 0) | (numeric_data > 1))
        return {
            'is_mixture': False,
            'n_components': 0,
            'component_names': [],
            'reason': f'{out_of_range} values outside [0,1] range (mixture components must be proportions)'
        }

    # Check 2: Row sums ≈ 1.0 (tolerance ±0.01)
    row_sums = numeric_data.sum(axis=1)
    tolerance = 0.01

    if not np.allclose(row_sums, 1.0, atol=tolerance):
        max_deviation = np.max(np.abs(row_sums - 1.0))
        return {
            'is_mixture': False,
            'n_components': 0,
            'component_names': [],
            'reason': f'Row sums do not equal 1.0 (max deviation: {max_deviation:.4f}, tolerance: {tolerance})'
        }

    # ✅ All checks passed - this is a mixture design!
    return {
        'is_mixture': True,
        'n_components': len(numeric_cols),
        'component_names': numeric_cols,
        'reason': f'Valid mixture design: {len(numeric_cols)} components, all values in [0,1], row sums = 1.0'
    }


def apply_mixture_transformation(mixture_df, component_names):
    """
    Transform real mixture compositions to pseudo-component coordinates.

    **Pseudo-Component Transformation:**
    Rescales each component to [0, 1] based on its range in the design:

    Formula: PseudoCompi = (Xi - min_i) / (max_i - min_i)

    Where:
    - Xi: real value of component i
    - min_i: minimum value of component i in the design
    - max_i: maximum value of component i in the design

    This transformation maps the constrained mixture simplex to an
    orthogonal [0, 1]^n coordinate system, which:
    - Makes vertices pure components: (1, 0, 0), (0, 1, 0), (0, 0, 1)
    - Simplifies model interpretation
    - Enables orthogonal analysis

    Args:
        mixture_df: pd.DataFrame with real composition values (rows sum to 1.0)
        component_names: list of component column names

    Returns:
        pd.DataFrame with PseudoComp1, PseudoComp2, PseudoComp3... columns

    Example:
        >>> df = pd.DataFrame({
        ...     'Parmesan': [0.40, 0.20, 0.20],
        ...     'Bread': [0.30, 0.50, 0.30],
        ...     'Eggs': [0.30, 0.30, 0.50]
        ... })
        >>> pseudo = apply_mixture_transformation(df, ['Parmesan', 'Bread', 'Eggs'])
        >>> pseudo.columns.tolist()
        ['PseudoComp1', 'PseudoComp2', 'PseudoComp3']
        >>> pseudo.iloc[0].values  # First row: (0.40, 0.30, 0.30)
        array([1., 0., 0.])  # Transformed: (1, 0, 0) - pure Parmesan

    Note:
        This is NOT Principal Component Analysis (PCA)!
        "PseudoComp" refers to the mixture design concept of
        pseudo-components, not principal components from PCA.
    """
    import numpy as np

    # Validate inputs
    if mixture_df is None or len(mixture_df) == 0:
        raise ValueError("mixture_df is empty")

    if not component_names:
        raise ValueError("component_names is empty")

    # Extract component data
    component_data = mixture_df[component_names]

    # Calculate min and max for each component
    min_vals = component_data.min()
    max_vals = component_data.max()
    range_vals = max_vals - min_vals

    # Handle case where range is 0 (constant component)
    # If a component is constant, set its pseudo-component value to 0.5
    range_vals[range_vals == 0] = 1.0
    const_components = (max_vals == min_vals)

    # Create pseudo-component DataFrame
    pseudo_df = pd.DataFrame(index=mixture_df.index)

    for i, comp_name in enumerate(component_names):
        pseudo_name = f'PseudoComp{i + 1}'

        if const_components[comp_name]:
            # Constant component: set to 0.5 (middle of [0, 1])
            pseudo_df[pseudo_name] = 0.5
        else:
            # Apply transformation: (Xi - min) / (max - min)
            pseudo_df[pseudo_name] = (
                (component_data[comp_name] - min_vals[comp_name]) /
                range_vals[comp_name]
            )

    return pseudo_df


# ============================================================================
# SCHEFFE POLYNOMIAL FEATURE MATRIX CREATION
# ============================================================================

def create_scheffe_polynomial_matrix(X, degree='quadratic'):
    """
    Create Scheffe polynomial feature matrix for mixture models

    Args:
        X: pd.DataFrame with component columns (must sum to 1)
        degree: 'linear', 'reduced_cubic', 'quadratic', 'cubic'

    Returns:
        pd.DataFrame with Scheffe polynomial features

    Scheffe Polynomial Features:
    - LINEAR: X1, X2, X3, ... (NO INTERCEPT!)
    - REDUCED_CUBIC: Linear + X1*X2*X3 (skip binary interactions)
    - QUADRATIC: Linear + X1*X2, X1*X3, X2*X3 (binary interactions)
    - CUBIC (FULL): Quadratic + X1*X2*X3 (all interactions)

    KEY DIFFERENCE from standard polynomials:
    NO CONSTANT TERM (b0) because sum(X) = 1 always!
    The mixture constraint makes the intercept unidentifiable.

    Mathematical Background:
        In standard regression: y = β₀ + β₁X₁ + β₂X₂ + ...
        In mixture models: y = β₁X₁ + β₂X₂ + ... (no β₀)
        because there's no feasible point where all Xᵢ = 0
    """
    if isinstance(X, pd.DataFrame):
        X_array = X.values
        component_names = X.columns.tolist()
    else:
        X_array = np.array(X)
        component_names = [f"X{i+1}" for i in range(X_array.shape[1])]

    n_samples, n_components = X_array.shape

    # Validate mixture constraint
    row_sums = X_array.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-6):
        max_dev = np.max(np.abs(row_sums - 1.0))
        warnings.warn(f"Input rows do not sum to 1.0 (max deviation: {max_dev:.6f})")

    # Feature list
    features = []
    feature_names = []

    # LINEAR TERMS (always included)
    for i in range(n_components):
        features.append(X_array[:, i])
        feature_names.append(component_names[i])

    # ═══════════════════════════════════════════════════════════════════════════
    # QUADRATIC TERMS (2-way interactions)
    # ═══════════════════════════════════════════════════════════════════════════
    if degree in ['quadratic', 'cubic']:
        for i in range(n_components):
            for j in range(i+1, n_components):
                # Interaction term Xi*Xj
                features.append(X_array[:, i] * X_array[:, j])
                feature_names.append(f"{component_names[i]}*{component_names[j]}")

    # ═══════════════════════════════════════════════════════════════════════════
    # CUBIC TERMS (3-way interactions)
    # ═══════════════════════════════════════════════════════════════════════════
    if degree in ['cubic', 'reduced_cubic']:
        for i in range(n_components):
            for j in range(i+1, n_components):
                for k in range(j+1, n_components):
                    # Three-way interaction Xi*Xj*Xk
                    features.append(X_array[:, i] * X_array[:, j] * X_array[:, k])
                    feature_names.append(f"{component_names[i]}*{component_names[j]}*{component_names[k]}")

    # Create feature matrix
    X_features = np.column_stack(features)

    return pd.DataFrame(X_features, columns=feature_names)


# ============================================================================
# MIXTURE MODEL FITTING
# ============================================================================

def fit_mixture_model(X, y, degree='quadratic'):
    """
    Fit Scheffe polynomial mixture model using least squares

    Args:
        X: pd.DataFrame with component values (n_samples × n_components)
        y: np.array or pd.Series with response values
        degree: 'linear', 'quadratic', 'cubic'

    Returns:
        dict with comprehensive model results

    Model Results Dictionary:
    {
        'coefficients': Scheffe polynomial coefficients,
        'X': Augmented feature matrix,
        'y': Response variable,
        'y_pred': Fitted values,
        'residuals': y - y_pred,
        'r_squared': R²,
        'rmse': Root Mean Square Error,
        'dof': Degrees of freedom,
        'n_samples', 'n_features', 'n_components',
        'component_names': ['X1', 'X2', ...],
        'model_type': 'Scheffe Polynomial',
        'degree': degree,
        'XtX_inv': (X'X)^-1 for inference,
        'cv_predictions': Leave-One-Out CV predictions,
        'q2': Cross-validation R²,
        'rmsecv': Cross-validation RMSE,
        'se_coef': Standard errors of coefficients,
        't_stats': t-statistics,
        'p_values': p-values for coefficients,
        'ci_lower': 95% CI lower bounds,
        'ci_upper': 95% CI upper bounds,
        'leverage': Diagonal of hat matrix,
        'feature_names': ['X1', 'X1*X2', ...],
        'vif': Variance Inflation Factors (if computable)
    }
    """
    # Prepare data
    if isinstance(X, pd.DataFrame):
        component_names = X.columns.tolist()
    else:
        X = pd.DataFrame(X, columns=[f"X{i+1}" for i in range(X.shape[1])])
        component_names = X.columns.tolist()

    if isinstance(y, pd.Series):
        y = y.values
    else:
        y = np.array(y).flatten()

    n_samples = len(y)
    n_components = len(component_names)

    # Create Scheffe polynomial feature matrix
    X_features = create_scheffe_polynomial_matrix(X, degree=degree)

    n_features = X_features.shape[1]
    feature_names = X_features.columns.tolist()

    # Convert to numpy for computation
    X_mat = X_features.values

    # Degrees of freedom
    dof = n_samples - n_features

    # ✓ ALLOW DOF = 0 (saturated model is OK for mixture designs)
    # Only block if truly over-parameterized (DOF < 0)
    if dof < 0:
        raise ValueError(
            f"Over-parameterized model: {n_samples} samples, {n_features} parameters. "
            f"Need at least {n_features} samples. Use simpler model (lower degree)."
        )

    # Fit model using least squares (X'X)^-1 X'y
    try:
        XtX = X_mat.T @ X_mat
        XtX_inv = np.linalg.inv(XtX)
        Xty = X_mat.T @ y

        coefficients = XtX_inv @ Xty

    except np.linalg.LinAlgError:
        raise ValueError(
            "Singular matrix encountered. Design matrix may be rank-deficient. "
            "Try reducing model complexity or using D-optimal design."
        )

    # Fitted values and residuals
    y_pred = X_mat @ coefficients
    residuals = y - y_pred

    # R² and RMSE
    r_squared = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))

    # Adjusted R²
    adj_r_squared = 1 - (1 - r_squared) * (n_samples - 1) / dof if dof > 0 else np.nan

    # Residual standard error
    if dof > 0:
        sigma_squared = np.sum(residuals**2) / dof
        sigma = np.sqrt(sigma_squared)
    else:
        sigma = np.nan
        sigma_squared = np.nan

    # Standard errors of coefficients
    se_coef = np.sqrt(np.diag(XtX_inv) * sigma_squared) if dof > 0 else np.full(n_features, np.nan)

    # t-statistics and p-values
    if dof > 0 and not np.any(np.isnan(se_coef)):
        t_stats = coefficients / se_coef
        p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), dof))
    else:
        t_stats = np.full(n_features, np.nan)
        p_values = np.full(n_features, np.nan)

    # 95% Confidence intervals
    if dof > 0:
        t_critical = stats.t.ppf(0.975, dof)
        ci_lower = coefficients - t_critical * se_coef
        ci_upper = coefficients + t_critical * se_coef
    else:
        ci_lower = np.full(n_features, np.nan)
        ci_upper = np.full(n_features, np.nan)

    # Leverage (diagonal of hat matrix)
    H = X_mat @ XtX_inv @ X_mat.T
    leverage = np.diag(H)

    # Cross-validation (Leave-One-Out)
    # ⚠️ SKIP CV when DOF = 0 (saturated model - no residual variance)
    if dof <= 0:
        warnings.warn(
            "Saturated model (DOF ≤ 0): Cross-validation skipped. "
            "No residual variance to estimate. Add replicate experiments to enable CV."
        )
        cv_predictions = np.full(n_samples, np.nan)
        cv_residuals = np.full(n_samples, np.nan)
        q2 = np.nan
        rmsecv = np.nan
    else:
        try:
            loo = LeaveOneOut()
            cv_predictions = np.zeros(n_samples)

            for train_idx, test_idx in loo.split(X_mat):
                X_train, X_test = X_mat[train_idx], X_mat[test_idx]
                y_train = y[train_idx]

                # Fit on training set
                try:
                    XtX_train = X_train.T @ X_train
                    XtX_inv_train = np.linalg.inv(XtX_train)
                    coef_train = XtX_inv_train @ (X_train.T @ y_train)

                    # Predict on test set
                    cv_predictions[test_idx] = X_test @ coef_train
                except:
                    cv_predictions[test_idx] = np.nan

            # Q² (cross-validation R²)
            cv_residuals = y - cv_predictions
            ss_res_cv = np.sum(cv_residuals**2)
            ss_tot = np.sum((y - np.mean(y))**2)
            q2 = 1 - (ss_res_cv / ss_tot) if ss_tot > 0 else np.nan

            rmsecv = np.sqrt(mean_squared_error(y, cv_predictions))

        except Exception as e:
            warnings.warn(f"Cross-validation failed: {str(e)}")
            cv_predictions = np.full(n_samples, np.nan)
            cv_residuals = np.full(n_samples, np.nan)
            q2 = np.nan
            rmsecv = np.nan

    # VIF (Variance Inflation Factors) - optional, may fail for mixture models
    try:
        vif_values = {}
        for i in range(n_features):
            # VIF_i = 1 / (1 - R²_i)
            # where R²_i is from regressing X_i on all other X_j
            if n_features > 1:
                X_others = np.delete(X_mat, i, axis=1)
                X_i = X_mat[:, i]

                # Regression
                try:
                    beta = np.linalg.lstsq(X_others, X_i, rcond=None)[0]
                    X_i_pred = X_others @ beta
                    r2_i = r2_score(X_i, X_i_pred)

                    vif = 1 / (1 - r2_i) if r2_i < 0.9999 else np.inf
                except:
                    vif = np.nan
            else:
                vif = 1.0

            vif_values[feature_names[i]] = vif

        vif_series = pd.Series(vif_values)

    except Exception as e:
        warnings.warn(f"VIF computation failed: {str(e)}")
        vif_series = pd.Series({name: np.nan for name in feature_names})

    # Compile model results
    model_results = {
        'coefficients': coefficients,
        'X': X_features,  # Feature DataFrame
        'y': y,
        'y_pred': y_pred,
        'residuals': residuals,
        'r_squared': r_squared,
        'adj_r_squared': adj_r_squared,
        'rmse': rmse,
        'sigma': sigma,
        'dof': dof,
        'n_samples': n_samples,
        'n_features': n_features,
        'n_components': n_components,
        'component_names': component_names,
        'model_type': 'Scheffe Polynomial',
        'degree': degree,
        'XtX_inv': XtX_inv,
        'cv_predictions': cv_predictions,
        'cv_residuals': cv_residuals,
        'q2': q2,
        'rmsecv': rmsecv,
        'se_coef': se_coef,
        't_stats': t_stats,
        'p_values': p_values,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'leverage': leverage,
        'feature_names': feature_names,
        'vif': vif_series,
        'original_X': X,  # Store original component matrix
        'is_saturated': (dof <= 0)  # ✅ Flag for saturated model
    }

    return model_results


# ============================================================================
# STATISTICAL SUMMARY
# ============================================================================

def statistical_summary(model_results):
    """
    Generate statistical summary DataFrame for mixture model

    Args:
        model_results: dict from fit_mixture_model

    Returns:
        pd.DataFrame with key statistics
    """
    summary_data = {
        'Statistic': [
            'Model Type',
            'Polynomial Degree',
            'Number of Samples',
            'Number of Components',
            'Number of Model Terms',
            'Degrees of Freedom',
            'R²',
            'Adjusted R²',
            'Q² (Cross-Validation R²)',
            'RMSE',
            'RMSECV',
            'Residual Std Error (σ)',
            'Model Significance'
        ],
        'Value': [
            model_results['model_type'],
            model_results['degree'],
            model_results['n_samples'],
            model_results['n_components'],
            model_results['n_features'],
            model_results['dof'],
            f"{model_results['r_squared']:.6f}",
            f"{model_results['adj_r_squared']:.6f}",
            f"{model_results.get('q2', np.nan):.6f}",
            f"{model_results['rmse']:.6f}",
            f"{model_results.get('rmsecv', np.nan):.6f}",
            f"{model_results.get('sigma', np.nan):.6f}",
            "See F-test below"
        ]
    }

    return pd.DataFrame(summary_data)


# ============================================================================
# PREDICTION ON NEW MIXTURES
# ============================================================================

def scheffe_polynomial_prediction(model_results, mixture_point):
    """
    Predict response for a new mixture composition

    Args:
        model_results: dict from fit_mixture_model
        mixture_point: dict like {'X1': 0.5, 'X2': 0.3, 'X3': 0.2}
                      OR pd.DataFrame with one or more rows

    Returns:
        dict with:
        {
            'predicted_value': float or array,
            'se': standard error(s),
            'ci_lower': 95% CI lower bound(s),
            'ci_upper': 95% CI upper bound(s),
            'is_valid_mixture': bool or array
        }

    Logic:
        1. Create feature vector using same Scheffe polynomial
        2. y_pred = X_new @ coefficients
        3. SE_pred = σ * sqrt(X_new @ (X'X)^-1 @ X_new')
        4. CI = y_pred ± t_critical * SE_pred
    """
    # Parse mixture point
    if isinstance(mixture_point, dict):
        # Single point as dict
        component_names = model_results['component_names']

        # Validate keys
        if not all(comp in mixture_point for comp in component_names):
            raise ValueError(f"mixture_point must contain all components: {component_names}")

        # Create DataFrame
        mixture_df = pd.DataFrame([mixture_point])[component_names]

    elif isinstance(mixture_point, pd.DataFrame):
        # Multiple points as DataFrame
        mixture_df = mixture_point

    else:
        raise ValueError("mixture_point must be dict or DataFrame")

    # Validate mixture constraint
    row_sums = mixture_df.sum(axis=1).values
    is_valid_mixture = np.isclose(row_sums, 1.0, atol=1e-6)

    if not np.all(is_valid_mixture):
        warnings.warn("Some mixture points do not sum to 1.0")

    # Create Scheffe polynomial features
    X_new_features = create_scheffe_polynomial_matrix(mixture_df, degree=model_results['degree'])
    X_new = X_new_features.values

    # Prediction
    coefficients = model_results['coefficients']
    y_pred = X_new @ coefficients

    # Standard error of prediction
    sigma = model_results.get('sigma', np.nan)
    XtX_inv = model_results['XtX_inv']
    dof = model_results['dof']

    # SE = σ * sqrt(X_new (X'X)^-1 X_new')
    if not np.isnan(sigma) and dof > 0:
        var_pred = np.diag(X_new @ XtX_inv @ X_new.T) * (sigma**2)
        se = np.sqrt(var_pred)

        # 95% CI
        t_critical = stats.t.ppf(0.975, dof)
        ci_lower = y_pred - t_critical * se
        ci_upper = y_pred + t_critical * se
    else:
        se = np.full_like(y_pred, np.nan)
        ci_lower = np.full_like(y_pred, np.nan)
        ci_upper = np.full_like(y_pred, np.nan)

    # Return
    if len(y_pred) == 1:
        # Single prediction
        return {
            'predicted_value': y_pred[0],
            'se': se[0],
            'ci_lower': ci_lower[0],
            'ci_upper': ci_upper[0],
            'is_valid_mixture': is_valid_mixture[0]
        }
    else:
        # Multiple predictions
        return {
            'predicted_value': y_pred,
            'se': se,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'is_valid_mixture': is_valid_mixture
        }


# ============================================================================
# COMPONENT EFFECTS ANALYSIS
# ============================================================================

def calculate_component_effects(model_results, baseline_mixture=None):
    """
    Calculate effect of each component on response

    Args:
        model_results: dict from mixture model
        baseline_mixture: dict with starting mixture (default: equal proportions)

    Returns:
        pd.DataFrame with component effects

    Theory from Mixture Design:
        Component effect = change in response from reference to pure component
        Reference: equal mixture or user-specified baseline

    For 3-component mixture:
        Effect of X1 = y(1, 0, 0) - y(0, 0.5, 0.5)
        (pure X1 vs. equal mixture of X2 and X3)
    """
    component_names = model_results['component_names']
    n_components = model_results['n_components']

    # Default baseline: equal proportions
    if baseline_mixture is None:
        baseline_mixture = {comp: 1.0 / n_components for comp in component_names}

    # Predict at baseline
    baseline_pred = scheffe_polynomial_prediction(model_results, baseline_mixture)
    baseline_response = baseline_pred['predicted_value']

    # Calculate effect for each component
    effects = []

    for comp in component_names:
        # Create pure component mixture
        pure_mixture = {c: 0.0 for c in component_names}
        pure_mixture[comp] = 1.0

        # Predict at pure component
        pure_pred = scheffe_polynomial_prediction(model_results, pure_mixture)
        pure_response = pure_pred['predicted_value']

        # Effect
        effect = pure_response - baseline_response

        # Determine direction
        if abs(effect) < 1e-6:
            direction = 'neutral'
        elif effect > 0:
            direction = 'positive'
        else:
            direction = 'negative'

        effects.append({
            'Component': comp,
            'Baseline_Value': baseline_mixture[comp],
            'Pure_Response': pure_response,
            'Baseline_Response': baseline_response,
            'Effect': effect,
            'Direction': direction,
            'Significance': 'TBD'  # Would require hypothesis test
        })

    effects_df = pd.DataFrame(effects)

    return effects_df


# ============================================================================
# MODEL ADEQUACY TESTS
# ============================================================================

def check_mixture_model_adequacy(model_results, alpha=0.05):
    """
    Test overall model significance and adequacy

    Args:
        model_results: dict from mixture model
        alpha: significance level

    Returns:
        dict with test results:
        {
            'overall_significant': bool,
            'f_statistic': float,
            'f_pvalue': float,
            'lacks_fit': bool (if replicates available),
            'residuals_normal': bool (Shapiro-Wilk test)
        }

    Tests:
        1. Overall F-test: H0: all β = 0
        2. Lack-of-fit test (if replicates exist)
        3. Normality of residuals
    """
    n_samples = model_results['n_samples']
    n_features = model_results['n_features']
    dof = model_results['dof']

    r_squared = model_results['r_squared']
    residuals = model_results['residuals']

    # 1. Overall F-test
    # F = (R² / p) / ((1 - R²) / (n - p - 1))
    # where p = n_features (no intercept in Scheffe polynomial)

    if dof > 0 and r_squared < 1.0:
        f_statistic = (r_squared / n_features) / ((1 - r_squared) / dof)
        f_pvalue = 1 - stats.f.cdf(f_statistic, n_features, dof)
        overall_significant = f_pvalue < alpha
    else:
        f_statistic = np.nan
        f_pvalue = np.nan
        overall_significant = False

    # 2. Normality test (Shapiro-Wilk)
    if n_samples >= 3:
        try:
            shapiro_stat, shapiro_p = stats.shapiro(residuals)
            residuals_normal = shapiro_p >= alpha
        except:
            shapiro_stat = np.nan
            shapiro_p = np.nan
            residuals_normal = None
    else:
        shapiro_stat = np.nan
        shapiro_p = np.nan
        residuals_normal = None

    # 3. Lack-of-fit test (requires replicate detection - placeholder)
    # This would require comparing pure error (from replicates) vs model error
    lacks_fit = None  # Not implemented yet

    return {
        'overall_significant': overall_significant,
        'f_statistic': f_statistic,
        'f_pvalue': f_pvalue,
        'lacks_fit': lacks_fit,
        'residuals_normal': residuals_normal,
        'shapiro_statistic': shapiro_stat,
        'shapiro_pvalue': shapiro_p
    }


# ============================================================================
# COMPONENT EFFECTS TRAJECTORY ANALYSIS
# ============================================================================

def calculate_component_effects_with_trajectory(model_results, mixture_design_matrix=None, n_steps=20):
    """
    Calculate component effects with full trajectory from opposite edge to pure component.

    **Concept:**
    Effect is not just the difference (pure - opposite), but the ENTIRE PATH.

    For each component i:
    - Start at opposite edge: Xi=0, all others equal (e.g., (0, 0.5, 0.5) for 3 components)
    - End at pure component: Xi=1, all others zero (e.g., (1, 0, 0))
    - Trajectory shows response along this path

    **Example:**
    Component X1 trajectory from (0, 0.5, 0.5) to (1, 0, 0):
    - t=0.0: (0.0, 0.5, 0.5) → Y = 50
    - t=0.2: (0.2, 0.4, 0.4) → Y = 75  (rising)
    - t=0.5: (0.5, 0.25, 0.25) → Y = 100 (peak!)
    - t=0.8: (0.8, 0.1, 0.1) → Y = 75  (falling)
    - t=1.0: (1.0, 0.0, 0.0) → Y = 50

    Simple effect: 50 - 50 = 0 ❌ (misleading)
    True trajectory: Shows cubic dome with max at center ✅

    **Formula:**
    For component i and parameter t ∈ [0, 1]:
        Xi(t) = t                    (grows from 0 to 1)
        Xj(t) = (1-t) / (n-1)        for j ≠ i (equally distributed)

    Verification: Xi + sum(Xj) = t + (n-1)*(1-t)/(n-1) = t + (1-t) = 1 ✓

    Args:
        model_results: Fitted mixture model from fit_mixture_model()
        mixture_design_matrix: Original design matrix (optional, not used currently)
        n_steps: Number of steps along trajectory (default 20)

    Returns:
        dict with one entry per component:
        {
            'ComponentName': {
                'effect_magnitude': float,              # Simple difference (pure - opposite)
                'trajectory_df': pd.DataFrame,          # Full trajectory data
                'responses': list[float],               # Response at each step
                'max_response': float,                  # Maximum along path
                'min_response': float,                  # Minimum along path
                'max_at_step': int,                     # Index where max occurs
                'max_value': float,                     # Maximum value
                't_values': list[float],                # Parameter values
                'compositions': list[dict]              # Mixture compositions
            }
        }

    Usage:
        >>> effect_results = calculate_component_effects_with_trajectory(model_results, n_steps=25)
        >>> pc1_trajectory = effect_results['PseudoComp1']['trajectory_df']
        >>> max_response = effect_results['PseudoComp1']['max_value']
    """
    component_names = model_results['component_names']
    n_components = len(component_names)

    results = {}

    for i, comp_name in enumerate(component_names):
        trajectory_data = {
            't': [],                    # Parameter from 0 to 1
            'step': [],                 # Step number
            'composition': [],          # Full composition dict
            'response': [],             # Predicted response
            'gradient': []              # Local gradient (dY/dt)
        }

        responses = []
        t_values = []
        compositions = []

        # Parametrize from opposite edge (t=0) to pure component (t=1)
        for step_idx, t in enumerate(np.linspace(0, 1, n_steps)):
            # Build composition
            composition = {}
            composition[comp_name] = t  # This component goes 0→1

            # Other components: equal distribution of (1-t)
            other_comps = [c for c in component_names if c != comp_name]
            for other_comp in other_comps:
                composition[other_comp] = (1 - t) / (n_components - 1)

            # Predict response
            pred = scheffe_polynomial_prediction(model_results, composition)
            response = pred['predicted_value']

            trajectory_data['t'].append(t)
            trajectory_data['step'].append(step_idx + 1)
            trajectory_data['composition'].append(composition)
            trajectory_data['response'].append(response)

            responses.append(response)
            t_values.append(t)
            compositions.append(composition)

        # Calculate local gradients (finite differences)
        gradients = [0]  # First point: no gradient
        for k in range(1, len(responses)):
            dt = t_values[k] - t_values[k-1]
            dY = responses[k] - responses[k-1]
            gradient = dY / dt if dt > 0 else 0
            gradients.append(gradient)

        trajectory_data['gradient'] = gradients

        # Calculate simple effect magnitude
        effect_magnitude = responses[-1] - responses[0]  # pure - opposite

        # Find maximum and minimum
        max_response = max(responses)
        min_response = min(responses)
        max_at_step = int(np.argmax(responses))

        # ✅ FIX: Calculate Max at t (normalized value in [0, 1])
        max_t_value = t_values[max_at_step]

        # ✅ FIX: Curvature = max response - highest endpoint
        endpoint_start = responses[0]
        endpoint_end = responses[-1]
        max_endpoint = max(endpoint_start, endpoint_end)
        curvature = max_response - max_endpoint

        # Store results
        results[comp_name] = {
            'effect_magnitude': effect_magnitude,
            'trajectory_df': pd.DataFrame(trajectory_data),
            'responses': responses,
            'max_response': max_response,
            'min_response': min_response,
            'max_at_step': max_at_step,              # Index (for vline in plot)
            'max_t_value': max_t_value,              # ✅ NEW: Normalized t value [0,1]
            'max_value': max_response,
            't_values': t_values,
            'compositions': compositions,
            'curvature': curvature,                  # ✅ FIXED: max - max(endpoints)
            'endpoint_start': endpoint_start,        # ✅ NEW: Start endpoint value
            'endpoint_end': endpoint_end             # ✅ NEW: End endpoint value
        }

    return results


# ============================================================================
# COMPONENT EFFECTS PROFILES (GRID-BASED)
# ============================================================================

def calculate_component_effects_profiles(model_results, grid_resolution=30):
    """
    Calculate component effect profiles by varying each component from 0 to 1.

    **Concept:**
    For each component, vary it from 0 to 1 while keeping others FIXED at their
    central value in pseudo-component space.

    **For 3 components, central value = 0.5:**

    PC1 Profile:
        PC1: [0, 1/30, 2/30, ..., 1]  (grid_resolution + 1 points)
        PC2: 0.5 (fixed)
        PC3: 0.5 (fixed)
        Y = [Y(0, 0.5, 0.5), Y(1/30, 0.5, 0.5), ..., Y(1, 0.5, 0.5)]

    PC2 Profile:
        PC1: 0.5 (fixed)
        PC2: [0, 1/30, 2/30, ..., 1]
        PC3: 0.5 (fixed)
        Y = [Y(0.5, 0, 0.5), Y(0.5, 1/30, 0.5), ..., Y(0.5, 1, 0.5)]

    PC3 Profile:
        PC1: 0.5 (fixed)
        PC2: 0.5 (fixed)
        PC3: [0, 1/30, 2/30, ..., 1]
        Y = [Y(0.5, 0.5, 0), Y(0.5, 0.5, 1/30), ..., Y(0.5, 0.5, 1)]

    **Why profiles are DIFFERENT:**
    Each profile starts from a different point in mixture space:
    - PC1 starts at (0, 0.5, 0.5)
    - PC2 starts at (0.5, 0, 0.5)
    - PC3 starts at (0.5, 0.5, 0)

    These are DIFFERENT compositions → DIFFERENT starting Y values!

    Args:
        model_results: Fitted mixture model from fit_mixture_model()
        grid_resolution: Number of points (default 30)

    Returns:
        dict with one entry per component:
        {
            'ComponentName': {
                'x_values': array,           # [0, 1/30, 2/30, ..., 1]
                'y_values': array,           # [Y0, Y1, ..., Y30]
                'effect_magnitude': float,   # Y(1) - Y(0)
                'max_response': float,       # max(Y)
                'min_response': float,       # min(Y)
                'max_at_t': float,          # x-value where max occurs [0,1]
                'max_index': int,           # Index where max occurs
                'curvature': float,         # max - max(endpoints)
                'endpoint_start': float,    # Y(0)
                'endpoint_end': float,      # Y(1)
                'gradient': array           # dY/dx
            }
        }

    Usage:
        >>> profiles = calculate_component_effects_profiles(model_results, grid_resolution=30)
        >>> pc1_profile = profiles['PseudoComp1']
        >>> pc1_x = pc1_profile['x_values']
        >>> pc1_y = pc1_profile['y_values']
    """
    component_names = model_results['component_names']
    n_components = len(component_names)

    # For 3 components: central value = 0.5
    # For n components: central value = 1/(n-1)
    central_value = 1.0 / (n_components - 1) if n_components > 1 else 0.5

    results = {}

    # For each component
    for comp_idx, comp_name in enumerate(component_names):

        # Create x-values: 0 to 1 with grid_resolution + 1 points
        x_values = np.linspace(0, 1, grid_resolution + 1)
        y_values = []

        # For each x-value, predict response
        for x_val in x_values:
            # Build composition dict
            composition = {}
            composition[comp_name] = x_val

            # Other components = central_value (fixed!)
            for other_idx, other_comp in enumerate(component_names):
                if other_idx != comp_idx:
                    composition[other_comp] = central_value

            # Predict
            pred = scheffe_polynomial_prediction(model_results, composition)
            y_values.append(pred['predicted_value'])

        y_array = np.array(y_values)

        # Calculate effect metrics
        effect_magnitude = y_array[-1] - y_array[0]  # Response at pure - response at 0
        max_idx = np.argmax(y_array)
        max_t_value = x_values[max_idx]
        max_response = y_array[max_idx]
        min_response = np.min(y_array)

        endpoint_start = y_array[0]
        endpoint_end = y_array[-1]
        max_endpoint = max(endpoint_start, endpoint_end)
        curvature = max_response - max_endpoint

        # Calculate gradient using numpy gradient
        gradient = np.gradient(y_values, x_values)

        results[comp_name] = {
            'x_values': x_values,           # [0, 1/30, 2/30, ..., 1]
            'y_values': y_values,           # [Y0, Y1, Y2, ..., Y30]
            'effect_magnitude': effect_magnitude,
            'max_response': max_response,
            'min_response': min_response,
            'max_at_t': max_t_value,       # Normalized x-value [0,1]
            'max_index': max_idx,           # Index for plotting
            'curvature': curvature,
            'endpoint_start': endpoint_start,
            'endpoint_end': endpoint_end,
            'gradient': gradient            # dY/dx
        }

    return results


# ============================================================================
# EXTRACT TRAJECTORIES FROM TERNARY GRID
# ============================================================================

def extract_component_trajectories(grid_data, tolerance=0.01, downsample=3):
    """
    Extract component effect trajectories from the ternary grid.

    **Concept:**
    Instead of recalculating predictions, extract existing points from the
    ternary surface grid that lie on component effect trajectories.

    **For each component:**
    Find grid points where the component varies from 0 to 1 while
    others remain at their central value (0.5).

    **Trajectory 1 (PC1):** From (0, 0.5, 0.5) to (1, 0, 0)
    - Select points where: PC2 ≈ 0.5*(1-PC1) AND PC3 ≈ 0.5*(1-PC1)

    **Trajectory 2 (PC2):** From (0.5, 0, 0.5) to (0, 1, 0)
    - Select points where: PC1 ≈ 0.5*(1-PC2) AND PC3 ≈ 0.5*(1-PC2)

    **Trajectory 3 (PC3):** From (0.5, 0.5, 0) to (0, 0, 1)
    - Select points where: PC1 ≈ 0.5*(1-PC3) AND PC2 ≈ 0.5*(1-PC3)

    Args:
        grid_data: dict with keys:
            - 'pc1': list of PC1 values
            - 'pc2': list of PC2 values
            - 'pc3': list of PC3 values
            - 'responses': list of Y values
            - 'component_names': list of component names
            - 'n_points': grid resolution
        tolerance: matching tolerance (default 0.01)
        downsample: take every N-th point for smoother curves (default 3)
            - downsample=1: all points (may be noisy)
            - downsample=3: every 3rd point (~30% of points, smoother)
            - downsample=4: every 4th point (~25% of points, very smooth)

    Returns:
        dict with trajectories for each component:
        {
            'ComponentName': {
                'x_values': [0, 0.1, ..., 1],      # Component values
                'y_values': [Y0, Y1, ..., Yn],     # Response values
                'gradient': [dY/dx values],        # Gradients
                'effect_magnitude': float,         # Y(1) - Y(0)
                'max_response': float,
                'min_response': float,
                'max_at_t': float,                 # x where max occurs
                'max_index': int,
                'curvature': float,
                'endpoint_start': float,
                'endpoint_end': float,
                'n_points': int                     # Number of points extracted
            }
        }

    Usage:
        >>> grid_data = st.session_state.ternary_grid_data
        >>> trajectories = extract_component_trajectories(grid_data, tolerance=0.01, downsample=3)
        >>> pc1_traj = trajectories['PseudoComp1']
    """
    pc1_list = grid_data['pc1']
    pc2_list = grid_data['pc2']
    pc3_list = grid_data['pc3']
    responses_list = grid_data['responses']
    component_names = grid_data['component_names']

    trajectories = {}

    # ═══════════════════════════════════════════════════════════════════════════
    # TRAJECTORY 1: PC1 from 0 to 1 (PC2, PC3 at 0.5*(1-PC1))
    # ═══════════════════════════════════════════════════════════════════════════

    pc1_traj = []
    for pc1, pc2, pc3, y in zip(pc1_list, pc2_list, pc3_list, responses_list):
        # Check if point is on the trajectory: PC2 ≈ 0.5*(1-PC1), PC3 ≈ 0.5*(1-PC1)
        expected_pc2 = 0.5 * (1 - pc1)
        expected_pc3 = 0.5 * (1 - pc1)

        if abs(pc2 - expected_pc2) < tolerance and abs(pc3 - expected_pc3) < tolerance:
            pc1_traj.append((pc1, y))

    # Sort by PC1 value
    pc1_traj = sorted(pc1_traj, key=lambda x: x[0])

    # Downsample: take every N-th point for smoother curves
    pc1_traj = pc1_traj[::downsample]

    x_vals = np.array([p[0] for p in pc1_traj])
    y_vals = np.array([p[1] for p in pc1_traj])

    # Calculate metrics
    grad_vals = np.gradient(y_vals, x_vals)
    effect_magnitude = y_vals[-1] - y_vals[0]
    max_idx = np.argmax(y_vals)
    max_response = y_vals[max_idx]
    min_response = np.min(y_vals)
    max_at_t = x_vals[max_idx]
    endpoint_start = y_vals[0]
    endpoint_end = y_vals[-1]
    max_endpoint = max(endpoint_start, endpoint_end)
    curvature = max_response - max_endpoint

    trajectories[component_names[0]] = {
        'x_values': x_vals,
        'y_values': y_vals,
        'gradient': grad_vals,
        'effect_magnitude': effect_magnitude,
        'max_response': max_response,
        'min_response': min_response,
        'max_at_t': max_at_t,
        'max_index': max_idx,
        'curvature': curvature,
        'endpoint_start': endpoint_start,
        'endpoint_end': endpoint_end,
        'n_points': len(y_vals)
    }

    # ═══════════════════════════════════════════════════════════════════════════
    # TRAJECTORY 2: PC2 from 0 to 1 (PC1, PC3 at 0.5*(1-PC2))
    # ═══════════════════════════════════════════════════════════════════════════

    pc2_traj = []
    for pc1, pc2, pc3, y in zip(pc1_list, pc2_list, pc3_list, responses_list):
        # Check if point is on the trajectory: PC1 ≈ 0.5*(1-PC2), PC3 ≈ 0.5*(1-PC2)
        expected_pc1 = 0.5 * (1 - pc2)
        expected_pc3 = 0.5 * (1 - pc2)

        if abs(pc1 - expected_pc1) < tolerance and abs(pc3 - expected_pc3) < tolerance:
            pc2_traj.append((pc2, y))

    # Sort by PC2 value
    pc2_traj = sorted(pc2_traj, key=lambda x: x[0])

    # Downsample: take every N-th point for smoother curves
    pc2_traj = pc2_traj[::downsample]

    x_vals = np.array([p[0] for p in pc2_traj])
    y_vals = np.array([p[1] for p in pc2_traj])

    # Calculate metrics
    grad_vals = np.gradient(y_vals, x_vals)
    effect_magnitude = y_vals[-1] - y_vals[0]
    max_idx = np.argmax(y_vals)
    max_response = y_vals[max_idx]
    min_response = np.min(y_vals)
    max_at_t = x_vals[max_idx]
    endpoint_start = y_vals[0]
    endpoint_end = y_vals[-1]
    max_endpoint = max(endpoint_start, endpoint_end)
    curvature = max_response - max_endpoint

    trajectories[component_names[1]] = {
        'x_values': x_vals,
        'y_values': y_vals,
        'gradient': grad_vals,
        'effect_magnitude': effect_magnitude,
        'max_response': max_response,
        'min_response': min_response,
        'max_at_t': max_at_t,
        'max_index': max_idx,
        'curvature': curvature,
        'endpoint_start': endpoint_start,
        'endpoint_end': endpoint_end,
        'n_points': len(y_vals)
    }

    # ═══════════════════════════════════════════════════════════════════════════
    # TRAJECTORY 3: PC3 from 0 to 1 (PC1, PC2 at 0.5*(1-PC3))
    # ═══════════════════════════════════════════════════════════════════════════

    pc3_traj = []
    for pc1, pc2, pc3, y in zip(pc1_list, pc2_list, pc3_list, responses_list):
        # Check if point is on the trajectory: PC1 ≈ 0.5*(1-PC3), PC2 ≈ 0.5*(1-PC3)
        expected_pc1 = 0.5 * (1 - pc3)
        expected_pc2 = 0.5 * (1 - pc3)

        if abs(pc1 - expected_pc1) < tolerance and abs(pc2 - expected_pc2) < tolerance:
            pc3_traj.append((pc3, y))

    # Sort by PC3 value
    pc3_traj = sorted(pc3_traj, key=lambda x: x[0])

    # Downsample: take every N-th point for smoother curves
    pc3_traj = pc3_traj[::downsample]

    x_vals = np.array([p[0] for p in pc3_traj])
    y_vals = np.array([p[1] for p in pc3_traj])

    # Calculate metrics
    grad_vals = np.gradient(y_vals, x_vals)
    effect_magnitude = y_vals[-1] - y_vals[0]
    max_idx = np.argmax(y_vals)
    max_response = y_vals[max_idx]
    min_response = np.min(y_vals)
    max_at_t = x_vals[max_idx]
    endpoint_start = y_vals[0]
    endpoint_end = y_vals[-1]
    max_endpoint = max(endpoint_start, endpoint_end)
    curvature = max_response - max_endpoint

    trajectories[component_names[2]] = {
        'x_values': x_vals,
        'y_values': y_vals,
        'gradient': grad_vals,
        'effect_magnitude': effect_magnitude,
        'max_response': max_response,
        'min_response': min_response,
        'max_at_t': max_at_t,
        'max_index': max_idx,
        'curvature': curvature,
        'endpoint_start': endpoint_start,
        'endpoint_end': endpoint_end,
        'n_points': len(y_vals)
    }

    return trajectories
