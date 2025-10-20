"""
CAT MLR/DOE Analysis Page
Complete Design of Experiments and Multiple Linear Regression suite
Equivalent to DOE_* R scripts
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut
from scipy import stats
from scipy.spatial import ConvexHull
import itertools
from io import BytesIO

# Try to import openpyxl for Excel export
try:
    import openpyxl
    EXCEL_AVAILABLE = True
except ImportError:
    EXCEL_AVAILABLE = False

def generate_candidate_points(variables_config):
    """
    Generate candidate points for experimental design
    Equivalent to DOE_candidate_points.r
    
    Args:
        variables_config: dict with variable names as keys and levels as values
    
    Returns:
        DataFrame with all combinations of factor levels
    """
    levels_dict = {}
    
    for var_name, levels in variables_config.items():
        if isinstance(levels, str):
            # Parse string like "10,15,20"
            levels_dict[var_name] = [float(x.strip()) for x in levels.split(',')]
        else:
            levels_dict[var_name] = levels
    
    # Generate full factorial design
    keys = list(levels_dict.keys())
    values = list(levels_dict.values())
    
    combinations = list(itertools.product(*values))
    
    candidate_points = pd.DataFrame(combinations, columns=keys)
    
    return candidate_points

def create_model_matrix(X, include_intercept=True, include_interactions=True, 
                       include_quadratic=True, interaction_matrix=None):
    """
    Create model matrix with interactions and quadratic terms
    Equivalent to model matrix creation in DOE_model_computation.r
    
    Args:
        X: DataFrame with predictor variables
        include_intercept: bool, include intercept term
        include_interactions: bool, include two-term interactions
        include_quadratic: bool, include quadratic terms
        interaction_matrix: optional matrix specifying which interactions to include
    
    Returns:
        DataFrame with expanded model matrix and list of term names
    """
    n_vars = X.shape[1]
    var_names = X.columns.tolist()
    
    # Start with linear terms
    model_matrix = X.copy()
    term_names = var_names.copy()
    
    # Add interactions
    if include_interactions:
        for i in range(n_vars):
            for j in range(i+1, n_vars):
                # Check if this interaction should be included
                if interaction_matrix is not None:
                    if not interaction_matrix.iloc[i, j]:
                        continue
                
                interaction = X.iloc[:, i] * X.iloc[:, j]
                interaction_name = f"{var_names[i]}*{var_names[j]}"
                model_matrix[interaction_name] = interaction
                term_names.append(interaction_name)
    
    # Add quadratic terms
    if include_quadratic:
        for i in range(n_vars):
            # Check if quadratic should be included
            if interaction_matrix is not None:
                if not interaction_matrix.iloc[i, i]:
                    continue
            
            quadratic = X.iloc[:, i] ** 2
            quadratic_name = f"{var_names[i]}^2"
            model_matrix[quadratic_name] = quadratic
            term_names.append(quadratic_name)
    
    # Add intercept
    if include_intercept:
        model_matrix.insert(0, 'Intercept', 1.0)
        term_names.insert(0, 'Intercept')
    
    return model_matrix, term_names

def fit_mlr_model(X, y, return_diagnostics=True):
    """
    Fit Multiple Linear Regression model with full diagnostics
    Equivalent to DOE_model_computation.r
    
    Args:
        X: model matrix (with interactions, quadratic terms, intercept)
        y: response variable
        return_diagnostics: bool, compute full diagnostics
    
    Returns:
        dict with model results and diagnostics
    """
    # Convert to numpy arrays
    X_mat = X.values
    y_vec = y.values
    
    n_samples, n_features = X_mat.shape
    
    # Check rank
    rank = np.linalg.matrix_rank(X_mat)
    if rank < n_features:
        st.error(f"⚠️ Model matrix is rank deficient! Rank={rank}, Features={n_features}")
        return None
    
    # Compute coefficients: b = (X'X)^-1 X'y
    XtX = X_mat.T @ X_mat
    XtX_inv = np.linalg.inv(XtX)
    Xty = X_mat.T @ y_vec
    coefficients = XtX_inv @ Xty
    
    # Predictions
    y_pred = X_mat @ coefficients
    
    # Residuals
    residuals = y_vec - y_pred
    
    # Degrees of freedom
    dof = n_samples - n_features
    
    # Results dictionary
    results = {
        'coefficients': pd.Series(coefficients, index=X.columns),
        'y_pred': y_pred,
        'residuals': residuals,
        'dof': dof,
        'n_samples': n_samples,
        'n_features': n_features,
        'X': X,
        'y': y,
        'XtX_inv': XtX_inv
    }
    
    if dof > 0:
        # Variance of residuals
        rss = np.sum(residuals**2)
        var_res = rss / dof
        rmse = np.sqrt(var_res)
        
        # Variance of Y
        var_y = np.var(y_vec, ddof=1)
        
        # R-squared
        r_squared = 1 - (var_res / var_y)
        
        # Standard errors of coefficients
        var_coef = var_res * np.diag(XtX_inv)
        se_coef = np.sqrt(var_coef)
        
        # t-statistics
        t_stats = coefficients / se_coef
        
        # p-values
        p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), dof))
        
        # Confidence intervals
        t_critical = stats.t.ppf(0.975, dof)
        ci_lower = coefficients - t_critical * se_coef
        ci_upper = coefficients + t_critical * se_coef
        
        results.update({
            'rmse': rmse,
            'var_res': var_res,
            'var_y': var_y,
            'r_squared': r_squared,
            'se_coef': pd.Series(se_coef, index=X.columns),
            't_stats': pd.Series(t_stats, index=X.columns),
            'p_values': pd.Series(p_values, index=X.columns),
            'ci_lower': pd.Series(ci_lower, index=X.columns),
            'ci_upper': pd.Series(ci_upper, index=X.columns)
        })
        
        # Cross-validation (Leave-One-Out)
        if return_diagnostics and n_samples <= 100:  # Only for reasonable sample sizes
            cv_predictions = np.zeros(n_samples)
            
            for i in range(n_samples):
                # Remove sample i
                X_cv = np.delete(X_mat, i, axis=0)
                y_cv = np.delete(y_vec, i)
                
                # Fit model without sample i
                XtX_cv = X_cv.T @ X_cv
                XtX_cv_inv = np.linalg.inv(XtX_cv)
                coef_cv = XtX_cv_inv @ (X_cv.T @ y_cv)
                
                # Predict sample i
                cv_predictions[i] = X_mat[i, :] @ coef_cv
            
            cv_residuals = y_vec - cv_predictions
            rss_cv = np.sum(cv_residuals**2)
            rmsecv = np.sqrt(rss_cv / n_samples)
            q2 = 1 - (rss_cv / (var_y * n_samples))
            
            results.update({
                'cv_predictions': cv_predictions,
                'cv_residuals': cv_residuals,
                'rmsecv': rmsecv,
                'q2': q2
            })
        
        # Leverage (hat matrix diagonal)
        leverage = np.diag(X_mat @ XtX_inv @ X_mat.T)
        results['leverage'] = leverage
        
        # VIF (Variance Inflation Factors) - FORMULA DA R
        if n_features > 1:
            vif = []
            
            # Center the X matrix (subtract column means) - equivalente a xcc in R
            X_centered = X_mat - X_mat.mean(axis=0)
            
            for i in range(n_features):
                if X.columns[i] == 'Intercept':
                    vif.append(np.nan)
                else:
                    # Formula R: sum(X_centered_i^2) * diag(XtX_inv)_i
                    ss_centered = np.sum(X_centered[:, i]**2)
                    vif_value = ss_centered * XtX_inv[i, i]
                    vif.append(vif_value)
            
            results['vif'] = pd.Series(vif, index=X.columns)
    
    return results

def predict_new_points(model_results, X_new):
    """
    Predict response for new experimental points
    Equivalent to DOE_prediction.r
    
    Args:
        model_results: dict from fit_mlr_model
        X_new: DataFrame with new predictor values (same structure as training X)
    
    Returns:
        DataFrame with predictions, confidence intervals, leverage
    """
    X_mat = X_new.values
    coefficients = model_results['coefficients'].values
    
    # Predictions
    y_pred = X_mat @ coefficients
    
    # Leverage for new points
    XtX_inv = model_results['XtX_inv']
    leverage_new = np.diag(X_mat @ XtX_inv @ X_mat.T)
    
    # Confidence intervals
    if 'rmse' in model_results:
        rmse = model_results['rmse']
        dof = model_results['dof']
        t_critical = stats.t.ppf(0.975, dof)
        
        se_pred = rmse * np.sqrt(leverage_new)
        ci_lower = y_pred - t_critical * se_pred
        ci_upper = y_pred + t_critical * se_pred
        
        predictions_df = pd.DataFrame({
            'Predicted': y_pred,
            'Leverage': leverage_new,
            'CI_Lower': ci_lower,
            'CI_Upper': ci_upper,
            'SE_Pred': se_pred
        }, index=X_new.index)
    else:
        predictions_df = pd.DataFrame({
            'Predicted': y_pred,
            'Leverage': leverage_new
        }, index=X_new.index)
    
    return predictions_df

def detect_replicates(X_data, y_data, tolerance=1e-10):
    """
    Detect experimental replicates in the design matrix
    Returns None if no replicates found
    """
    n_samples = len(X_data)
    X_values = X_data.values
    y_values = y_data.values
    
    replicate_groups = []
    used_indices = set()
    
    for i in range(n_samples):
        if i in used_indices:
            continue
            
        group = [i]
        used_indices.add(i)
        
        for j in range(i + 1, n_samples):
            if j in used_indices:
                continue
            
            if np.allclose(X_values[i], X_values[j], atol=tolerance):
                group.append(j)
                used_indices.add(j)
        
        if len(group) > 1:
            replicate_groups.append(group)
    
    if not replicate_groups:
        return None
    
    # Calculate pooled standard deviation
    variance_sum = 0
    dof_sum = 0
    group_stats = []
    
    for group in replicate_groups:
        y_group = y_values[group]
        n_reps = len(group)
        mean_y = np.mean(y_group)
        
        if n_reps > 1:
            var_y = np.var(y_group, ddof=1)
            std_y = np.sqrt(var_y)
            dof = n_reps - 1
            
            variance_sum += var_y * dof
            dof_sum += dof
            
            group_stats.append({
                'indices': group,
                'n_replicates': n_reps,
                'mean': mean_y,
                'std': std_y,
                'variance': var_y,
                'dof': dof
            })
    
    if dof_sum > 0:
        pooled_variance = variance_sum / dof_sum
        pooled_std = np.sqrt(pooled_variance)
    else:
        return None
    
    return {
        'n_replicate_groups': len(replicate_groups),
        'total_replicates': sum(len(g) for g in replicate_groups),
        'group_stats': group_stats,
        'pooled_std': pooled_std,
        'pooled_variance': pooled_variance,
        'pooled_dof': dof_sum
    }


def calculate_lack_of_fit(y_data, y_pred, replicate_info, n_parameters):
    """
    Calculate Lack of Fit test comparing model error to pure experimental error
    
    Returns:
        dict with LOF statistics and F-test results
    """
    if replicate_info is None:
        return None
    
    n_total = len(y_data)
    
    # Pure error (from replicates)
    pure_error_var = replicate_info['pooled_variance']
    pure_error_dof = replicate_info['pooled_dof']
    
    # Total residual error
    residuals = y_data.values - y_pred
    rss_total = np.sum(residuals**2)
    dof_total = n_total - n_parameters
    
    # Lack of fit error = Total error - Pure error
    pure_error_ss = pure_error_var * pure_error_dof
    lof_ss = rss_total - pure_error_ss
    
    # Number of distinct experimental points
    n_distinct = replicate_info['n_replicate_groups'] + (n_total - replicate_info['total_replicates'])
    lof_dof = n_distinct - n_parameters
    
    if lof_dof <= 0:
        return None
    
    # Mean squares
    lof_ms = lof_ss / lof_dof
    pure_error_ms = pure_error_var
    
    # F-test for Lack of Fit
    f_statistic = lof_ms / pure_error_ms
    f_critical_95 = stats.f.ppf(0.95, lof_dof, pure_error_dof)
    f_critical_99 = stats.f.ppf(0.99, lof_dof, pure_error_dof)
    p_value = 1 - stats.f.cdf(f_statistic, lof_dof, pure_error_dof)
    
    return {
        'lof_ss': lof_ss,
        'lof_dof': lof_dof,
        'lof_ms': lof_ms,
        'pure_error_ss': pure_error_ss,
        'pure_error_dof': pure_error_dof,
        'pure_error_ms': pure_error_ms,
        'f_statistic': f_statistic,
        'f_critical_95': f_critical_95,
        'f_critical_99': f_critical_99,
        'p_value': p_value
    }

def detect_central_points(X_data, tolerance=1e-6):
    """
    Detect central points in the design matrix 
    A central point has ALL variables at their central value:
    - For coded variables: ALL must be 0
    - For natural variables: ALL must be at the midpoint of their range
    
    Args:
        X_data: DataFrame with predictor variables
        tolerance: tolerance for considering values as zero/central
    
    Returns:
        list of indices of central points
    """
    central_indices = []
    X_values = X_data.values
    
    for i in range(len(X_data)):
        # Method 1: Check for rows with ALL zeros (coded variables)
        if np.allclose(X_values[i], 0, atol=tolerance):
            central_indices.append(i)
            continue
        
        # Method 2: Check if ALL values are at the midpoint of their respective ranges
        is_central = True
        for j in range(X_data.shape[1]):
            col_values = X_values[:, j]
            unique_vals = np.unique(col_values)
            
            # Skip if only one unique value (no variation in this column)
            if len(unique_vals) <= 1:
                continue
                
            # For coded variables (-1, 0, 1), center MUST be 0
            if set(unique_vals).issubset({-1, 0, 1}):
                if not np.isclose(X_values[i, j], 0, atol=tolerance):
                    is_central = False
                    break
            else:
                # For natural variables, center MUST be at midpoint
                min_val = col_values.min()
                max_val = col_values.max()
                mid_val = (min_val + max_val) / 2
                
                if not np.isclose(X_values[i, j], mid_val, atol=tolerance):
                    is_central = False
                    break
        
        # Only add if ALL columns passed the central point test
        if is_central:
            central_indices.append(i)
    
    return central_indices


def detect_coded_matrix(data):
    """
    Detect if the data matrix contains coded variables (typically -1, 0, +1)
    Returns info about coding status and suggestions
    """
    numeric_data = data.select_dtypes(include=[np.number])
    
    if numeric_data.empty:
        return None
    
    coded_indicators = {
        'likely_coded': False,
        'columns_analysis': {},
        'suggestion': None
    }
    
    coded_columns = 0
    
    for col in numeric_data.columns:
        col_data = numeric_data[col].dropna()
        unique_vals = sorted(col_data.unique())
        
        analysis = {
            'unique_values': unique_vals,
            'n_unique': len(unique_vals),
            'range': col_data.max() - col_data.min() if len(col_data) > 0 else 0,
            'is_likely_coded': False
        }
        
        # Check for typical coded patterns
        if len(unique_vals) <= 5:  # Few levels suggest factorial design
            # Common coded patterns
            if set(unique_vals).issubset({-1, 0, 1}):
                analysis['is_likely_coded'] = True
                analysis['coding_type'] = 'Standard coded (-1, 0, +1)'
                coded_columns += 1
            elif set(unique_vals).issubset({-1, 1}):
                analysis['is_likely_coded'] = True
                analysis['coding_type'] = 'Two-level coded (-1, +1)'
                coded_columns += 1
            elif len(unique_vals) == 2 and analysis['range'] == 1:
                analysis['is_likely_coded'] = True
                analysis['coding_type'] = 'Binary coded (0, 1)'
                coded_columns += 1
            elif len(unique_vals) <= 3 and all(abs(v) <= 2 for v in unique_vals):
                analysis['is_likely_coded'] = True
                analysis['coding_type'] = 'Possible coded values'
                coded_columns += 1
        
        coded_indicators['columns_analysis'][col] = analysis
    
    # Overall assessment
    if coded_columns >= len(numeric_data.columns) * 0.7:  # 70% of columns appear coded
        coded_indicators['likely_coded'] = True
        coded_indicators['suggestion'] = 'transformation_menu'
    
    return coded_indicators

def show():
    """Display the MLR/DOE Analysis page"""
    
    st.markdown("# 🧪 Multiple Linear Regression & Design of Experiments")
    st.markdown("*Complete MLR/DOE analysis suite equivalent to DOE_* R scripts*")
    
    if 'current_data' not in st.session_state:
        st.warning("⚠️ No data loaded. Please go to Data Handling to load your dataset first.")
        return
    
    data = st.session_state.current_data
    
    # Detect coded matrix
    coded_info = detect_coded_matrix(data)
    if coded_info and coded_info['likely_coded']:
        st.info("🔍 **Coded Data Matrix Detected** - Variables appear to be coded (-1, 0, +1). For natural units transformation, go to **Menu → Transformation**.")
        
        with st.expander("📋 Coding Analysis Details"):
            for col, analysis in coded_info['columns_analysis'].items():
                if analysis['is_likely_coded']:
                    st.write(f"**{col}**: {analysis.get('coding_type', 'Unknown coding')} - Values: {analysis['unique_values']}")

    # Main tabs - FIXED SYNTAX ERROR
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "🔧 Model Computation", 
        "📊 Model Diagnostics",
        "📈 Response Surface",
        "🎨 Confidence Intervals",
        "🔮 Predictions",
        "🎯 Generate Matrix",
        "💾 Extract & Export"
    ])
    
    # ===== MODEL COMPUTATION TAB =====
    with tab1:
        st.markdown("## 🔧 MLR Model Computation")
        st.markdown("*Equivalent to DOE_model_computation.r*")
        
        # DATA PREVIEW SECTION
        st.markdown("### 👁️ Data Preview")
        with st.expander("Show current dataset", expanded=True):
            # Full scrollable dataframe
            st.dataframe(data, use_container_width=True, height=400)
            
            col_info1, col_info2, col_info3 = st.columns(3)
            with col_info1:
                st.metric("Total Samples", data.shape[0])
            with col_info2:
                st.metric("Total Variables", data.shape[1])
            with col_info3:
                numeric_cols_count = len(data.select_dtypes(include=[np.number]).columns)
                st.metric("Numeric Variables", numeric_cols_count)
        
        st.markdown("---")
        
        # Variable and sample selection
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 Variable Selection")
            
            numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
            
            if not numeric_columns:
                st.error("❌ No numeric columns found!")
                return
            
            # X variables
            x_vars = st.multiselect(
                "Select X variables (predictors):",
                numeric_columns,
                default=numeric_columns[:min(3, len(numeric_columns))],
                key="mlr_x_vars_widget"
            )
            
            # Y variable
            remaining_cols = [col for col in numeric_columns if col not in x_vars]
            if remaining_cols:
                y_var = st.selectbox("Select Y variable (response):", remaining_cols, key="mlr_y_var_widget")
            else:
                st.warning("⚠️ Select at least one X variable")
                return
            
            # Show selected variables info - FIX FOR TYPEERROR
            if x_vars and y_var:
                # Ensure x_vars are strings
                x_vars_str = [str(var) for var in x_vars]
                st.info(f"Model: {y_var} ~ {' + '.join(x_vars_str)}")
        
        with col2:
            st.markdown("### 🎯 Sample Selection")
            
            # Sample selection options
            sample_selection_mode = st.radio(
                "Select samples:",
                ["Use all samples", "Select by index", "Select by range"],
                key="sample_selection_mode"
            )
            
            if sample_selection_mode == "Use all samples":
                selected_samples = data.index.tolist()
                st.success(f"Using all {len(selected_samples)} samples")
                
            elif sample_selection_mode == "Select by index":
                sample_input = st.text_input(
                    "Enter sample indices (1-based, comma-separated or ranges):",
                    value=f"1-{data.shape[0]}",
                    help="Examples: 1,2,5-10,15 or 1-20"
                )
                
                try:
                    selected_indices = []
                    for part in sample_input.split(','):
                        part = part.strip()
                        if '-' in part:
                            start, end = map(int, part.split('-'))
                            selected_indices.extend(range(start-1, end))
                        else:
                            selected_indices.append(int(part)-1)
                    
                    selected_indices = sorted(list(set(selected_indices)))
                    valid_indices = [i for i in selected_indices if 0 <= i < len(data)]
                    selected_samples = data.index[valid_indices].tolist()
                    
                    st.success(f"Selected {len(selected_samples)} samples")
                    
                except Exception as e:
                    st.error(f"Invalid format: {e}")
                    selected_samples = data.index.tolist()
                    
            else:  # Select by range
                col_range1, col_range2 = st.columns(2)
                with col_range1:
                    start_idx = st.number_input("From sample:", 1, len(data), 1)
                with col_range2:
                    end_idx = st.number_input("To sample:", start_idx, len(data), len(data))
                
                selected_samples = data.index[start_idx-1:end_idx].tolist()
                st.success(f"Selected {len(selected_samples)} samples (rows {start_idx}-{end_idx})")
            
            # Show selected samples preview
            if len(selected_samples) < len(data):
                with st.expander("Preview selected samples"):
                    st.dataframe(data.loc[selected_samples].head(10), use_container_width=True)
        
        st.markdown("---")
        
        # Model Settings
        st.markdown("### ⚙️ Model Settings")

        include_intercept = st.checkbox("Include intercept", value=True)
        include_interactions = st.checkbox("Include two-term interactions", value=True)
        include_quadratic = st.checkbox("Include quadratic terms", value=True)
        exclude_central_points = st.checkbox("Exclude central points (0,0,0...)", value=False,
                                            help="Central points are typically used only for validation in factorial designs")

        with st.expander("🔧 Advanced Options"):
            run_cv = st.checkbox("Run cross-validation", value=True, 
                                help="Leave-one-out CV (only for n≤100)")
            
            custom_interactions = st.checkbox("Customize interaction matrix", value=False)
        
        # Custom interaction matrix
        interaction_matrix = None
        if custom_interactions and include_interactions:
            st.markdown("### 🎛️ Interaction Matrix")
            st.info("Check boxes to include specific interactions/quadratic terms")
            
            # Create interaction matrix
            n_vars = len(x_vars)
            interaction_df = pd.DataFrame(
                np.ones((n_vars, n_vars), dtype=int),
                index=x_vars,
                columns=x_vars
            )
            
            # Make lower triangle empty (only upper triangle for interactions)
            for i in range(n_vars):
                for j in range(i):
                    interaction_df.iloc[i, j] = 0
            
            # Editable matrix
            edited_matrix = st.data_editor(
                interaction_df,
                use_container_width=True,
                hide_index=False
            )
            
            interaction_matrix = edited_matrix
        
        # Fit model button
        if st.button("🚀 Fit MLR Model", type="primary"):
            try:
                # DEBUG - Aggiungi queste righe
                st.write("DEBUG - Selected X vars:", x_vars)
                st.write("DEBUG - Selected Y var:", y_var)
                st.write("DEBUG - Data shape:", data.shape)
                
                # Prepare data with selected samples
                X_data = data.loc[selected_samples, x_vars].copy()
                y_data = data.loc[selected_samples, y_var].copy()
                
                # DEBUG - Aggiungi queste righe
                st.write("DEBUG - X_data shape:", X_data.shape)
                st.write("DEBUG - X_data columns:", X_data.columns.tolist())
                st.write("DEBUG - y_data shape:", y_data.shape)
                st.write("DEBUG - X_data preview:")
                st.dataframe(X_data.head())
                
                # Remove missing values
                valid_idx = ~(X_data.isnull().any(axis=1) | y_data.isnull())
                X_data = X_data[valid_idx]
                y_data = y_data[valid_idx]
                
                # DEBUG - Aggiungi questa riga
                st.write("DEBUG - After removing missing values - X_data shape:", X_data.shape)
                
                if len(X_data) < len(x_vars) + 1:
                    st.error("❌ Not enough samples for model fitting!")
                    return
                
                st.info(f"ℹ️ Using {len(X_data)} samples after removing missing values")
                
                # Detect and optionally exclude central points
                central_points = detect_central_points(X_data)

                if central_points:
                    st.info(f"🎯 Detected {len(central_points)} central point(s) at indices: {[i+1 for i in central_points]}")
                    
                    if exclude_central_points:
                        # Store original indices before filtering
                        central_samples_original = X_data.index[central_points].tolist()
                        
                        # Remove central points from modeling data
                        X_data = X_data.drop(X_data.index[central_points])
                        y_data = y_data.drop(y_data.index[central_points])
                        
                        st.warning(f"⚠️ Excluded {len(central_points)} central point(s) from model fitting")
                        st.info(f"ℹ️ Using {len(X_data)} samples for model (excluding central points)")
                        
                        # Store excluded central points for later validation
                        st.session_state.mlr_central_points = {
                            'X': data.loc[central_samples_original, x_vars],
                            'y': data.loc[central_samples_original, y_var],
                            'indices': central_samples_original
                        }
                    else:
                        st.info("ℹ️ Central points included in the model")

                # Create model matrix
                with st.spinner("Creating model matrix..."):
                    X_model, term_names = create_model_matrix(
                        X_data,
                        include_intercept=include_intercept,
                        include_interactions=include_interactions,
                        include_quadratic=include_quadratic,
                        interaction_matrix=interaction_matrix
                    )

                st.success(f"✅ Model matrix created: {X_model.shape[0]} × {X_model.shape[1]}")

                # Fit model
                with st.spinner("Fitting MLR model..."):
                    model_results = fit_mlr_model(X_model, y_data, return_diagnostics=run_cv)

                if model_results is None:
                    return

                # Store results
                st.session_state.mlr_model = model_results
                st.session_state.mlr_y_var = y_var
                st.session_state.mlr_x_vars = x_vars

                st.success("✅ MLR model fitted successfully!")
                
                # Show number of experiments used for fitting
                st.info(f"📊 **Model fitted using {model_results['n_samples']} experiments** (after excluding central points if selected)")

                # Model Quality Summary - SIMPLIFIED
                st.markdown("### 📈 Model Quality Summary")
                
                summary_col1, summary_col2 = st.columns(2)
                
                with summary_col1:
                    if 'r_squared' in model_results:
                        var_explained_pct = model_results['r_squared'] * 100
                        st.metric("% Explained Variance (R²)", f"{var_explained_pct:.2f}%")
                
                with summary_col2:
                    if 'rmse' in model_results:
                        st.metric("Std Dev of Residuals (RMSE)", f"{model_results['rmse']:.4f}")

                # ========== EXPERIMENTAL VARIABILITY ANALYSIS ==========

                # ALWAYS use ALL original data (including central points) for experimental variability calculation
                # This gives the true experimental error regardless of modeling choices
                all_X_data = data.loc[selected_samples, x_vars].copy()
                all_y_data = data.loc[selected_samples, y_var].copy()
                all_valid_idx = ~(all_X_data.isnull().any(axis=1) | all_y_data.isnull())
                all_X_data = all_X_data[all_valid_idx]
                all_y_data = all_y_data[all_valid_idx]

                replicate_info_full = detect_replicates(all_X_data, all_y_data)
                st.write("DEBUG - replicate_info_full:", replicate_info_full)

                if replicate_info_full:
                    st.markdown("### 🔬 Experimental Variability (Pure Error)")
                    st.info("""
                    **Pure experimental error** estimated from replicate measurements 
                    (including ALL points - central points always included for experimental error calculation). 
                    This represents the baseline measurement variability.
                    """)
                    
                    rep_col1, rep_col2, rep_col3, rep_col4 = st.columns(4)
                    
                    with rep_col1:
                        st.metric("Replicate Groups", replicate_info_full['n_replicate_groups'])
                    with rep_col2:
                        st.metric("Total Replicates", replicate_info_full['total_replicates'])
                    with rep_col3:
                        st.metric("Pooled Std Dev (σ_exp)", f"{replicate_info_full['pooled_std']:.4f}")
                    with rep_col4:
                        st.metric("Pure Error DOF", replicate_info_full['pooled_dof'])
                    
                    with st.expander("📋 Replicate Groups Details"):
                        rep_data = []
                        for i, group in enumerate(replicate_info_full['group_stats'], 1):
                            rep_data.append({
                                'Group': i,
                                'Samples': ', '.join([str(idx+1) for idx in group['indices']]),
                                'N': group['n_replicates'],
                                'Mean Y': f"{group['mean']:.4f}",
                                'Std Dev': f"{group['std']:.4f}",
                                'Variance': f"{group['variance']:.6f}",
                                'DOF': group['dof']
                            })
                        
                        rep_df = pd.DataFrame(rep_data)
                        st.dataframe(rep_df, )
                        
                        st.markdown(f"""
                        **Pooled Standard Deviation Formula:**
                        
                        σ_pooled = √[Σ(s²ᵢ × dfᵢ) / Σ(dfᵢ)]
                        
                        Where s²ᵢ is the variance of group i and dfᵢ is its degrees of freedom.
                        
                        **Result:** σ_exp = {replicate_info_full['pooled_std']:.4f} 
                        (from {replicate_info_full['pooled_dof']} degrees of freedom)
                        """)
                    
                    # ========== STATISTICAL TESTS ==========
                    
                    st.markdown("---")
                    st.markdown("### 📊 Statistical Analysis of Model Quality")
                    
                    # 1. Compare DoE variability vs experimental variability (F-test)
                    st.markdown("#### 1️⃣ DoE Factor Variability vs Experimental Variability")
                    
                    if 'var_y' in model_results:
                        # CORRECTED: Use only DoE data (excluding central points if excluded from model)
                        if central_points and exclude_central_points:
                            # Use only the modeling data (DoE points without central points)
                            doe_y_data = y_data  # This already excludes central points
                            var_y_doe = np.var(doe_y_data, ddof=1)
                            dof_y_doe = len(doe_y_data) - 1
                            
                            st.info(f"""
                            **DoE Variability**: Calculated from {len(doe_y_data)} DoE experimental points 
                            (central points excluded as they don't contribute to factor-induced variation).
                            """)
                        else:
                            # Use all data if central points are included in model
                            var_y_doe = model_results['var_y']
                            dof_y_doe = len(all_y_data) - 1
                            
                            st.info("""
                            **DoE Variability**: Calculated from all experimental points 
                            (central points included in model).
                            """)
                        
                        # F-test: σ²_DoE / σ²_exp
                        f_global = var_y_doe / replicate_info_full['pooled_variance']
                        f_crit_global = stats.f.ppf(0.95, dof_y_doe, replicate_info_full['pooled_dof'])
                        p_global = 1 - stats.f.cdf(f_global, dof_y_doe, replicate_info_full['pooled_dof'])
                        
                        test_col1, test_col2, test_col3 = st.columns(3)
                        
                        with test_col1:
                            st.metric("DoE Variance (σ²_DoE)", f"{var_y_doe:.6f}")
                            st.metric("DOF", dof_y_doe)
                        
                        with test_col2:
                            st.metric("Experimental Variance (σ²_exp)", f"{replicate_info_full['pooled_variance']:.6f}")
                            st.metric("DOF", replicate_info_full['pooled_dof'])
                        
                        with test_col3:
                            st.metric("F-statistic", f"{f_global:.2f}")
                            st.metric("p-value", f"{p_global:.4f}")
                        
                        if p_global < 0.05:
                            st.success(f"✅ DoE factors induce significant variation in response (p={p_global:.4f})")
                            st.info("The experimental factors have meaningful effects on the response variable.")
                        else:
                            st.warning(f"⚠️ DoE factor effects not significantly different from experimental noise (p={p_global:.4f})")
                            st.info("The factors may have weak effects or the experimental error is too large.")
                        
                        # Show variance ratio for interpretation
                        variance_ratio = var_y_doe / replicate_info_full['pooled_variance']
                        st.markdown(f"""
                        **Variance Ratio**: σ²_DoE / σ²_exp = {variance_ratio:.2f}
                        
                        - Ratio > 4: Strong factor effects
                        - Ratio 2-4: Moderate factor effects  
                        - Ratio < 2: Weak factor effects
                        """)
                    
                    # 2. Lack of Fit test (residual error vs experimental error)
                    st.markdown("---")
                    st.markdown("#### 2️⃣ Lack of Fit Test (Model Adequacy)")
                    st.info("""
                    **F-test**: Compares model residual variance vs pure experimental variance.
                    - H₀: Model is adequate (σ²_model = σ²_exp)
                    - H₁: Significant lack of fit (σ²_model > σ²_exp)
                    """)
                    
                    if 'rmse' in model_results:
                        lof_col1, lof_col2, lof_col3 = st.columns(3)
                        
                        with lof_col1:
                            st.metric("Model RMSE", f"{model_results['rmse']:.4f}")
                            st.caption(f"Variance: {model_results['var_res']:.6f}")
                            st.caption(f"DOF: {model_results['dof']}")
                        
                        with lof_col2:
                            st.metric("Experimental Std Dev", f"{replicate_info_full['pooled_std']:.4f}")
                            st.caption(f"Variance: {replicate_info_full['pooled_variance']:.6f}")
                            st.caption(f"DOF: {replicate_info_full['pooled_dof']}")
                        
                        with lof_col3:
                            # F = variance_model / variance_exp
                            f_lof = model_results['var_res'] / replicate_info_full['pooled_variance']
                            f_crit = stats.f.ppf(0.95, model_results['dof'], replicate_info_full['pooled_dof'])
                            p_lof = 1 - stats.f.cdf(f_lof, model_results['dof'], replicate_info_full['pooled_dof'])
                            
                            st.metric("F-statistic", f"{f_lof:.2f}")
                            st.caption(f"F-crit (95%): {f_crit:.2f}")
                            st.caption(f"p-value: {p_lof:.4f}")
                        
                        # Unified interpretation with ratio
                        ratio = model_results['rmse'] / replicate_info_full['pooled_std']
                        
                        st.markdown("---")
                        
                        result_col1, result_col2 = st.columns([1, 3])
                        
                        with result_col1:
                            st.metric("RMSE / σ_exp", f"{ratio:.2f}")
                        
                        with result_col2:
                            if p_lof > 0.05:
                                st.success(f"✅ No significant Lack of Fit (p={p_lof:.4f})")
                                if ratio < 1.2:
                                    st.info("🎯 Model error ≈ experimental error - excellent fit")
                                elif ratio < 2.0:
                                    st.info("✅ Model error is reasonable")
                                else:
                                    st.warning("⚠️ Model error exceeds experimental error despite non-significant test")
                            else:
                                st.error(f"❌ Significant Lack of Fit detected (p={p_lof:.4f})")
                                st.warning("""
                                **Model inadequate!** Consider:
                                - Adding missing interaction or quadratic terms
                                - Checking for outliers or influential points
                                - Data transformations (log, sqrt, etc.)
                                - Verifying model assumptions
                                """)
                    else:
                        st.warning("Insufficient data for Lack of Fit test")
                else:
                    st.info("No replicates detected - skipping replicate analysis")
                # Central points validation section (if excluded from model)
                if central_points and exclude_central_points:
                    st.markdown("---")
                    st.markdown("### 🎯 Central Points Validation")
                    
                    st.info(f"""
                    **{len(central_points)} central point(s)** excluded from model fitting - reserved for validation.
                    These points assess model adequacy and curvature effects at the center of the experimental domain.
                    """)
                    
                    if 'mlr_central_points' in st.session_state:
                        central_X = st.session_state.mlr_central_points['X']
                        central_y = st.session_state.mlr_central_points['y']
                        
                        # Calculate central point statistics
                        central_mean = central_y.mean()
                        central_std = central_y.std(ddof=1) if len(central_y) > 1 else 0
                        
                        central_stats_col1, central_stats_col2, central_stats_col3 = st.columns(3)
                        
                        with central_stats_col1:
                            st.metric("Central Points Count", len(central_y))
                        with central_stats_col2:
                            st.metric("Mean Response", f"{central_mean:.4f}")
                        with central_stats_col3:
                            if len(central_y) > 1:
                                st.metric("Std Dev", f"{central_std:.4f}")
                            else:
                                st.metric("Std Dev", "N/A (single point)")
                        
                        with st.expander("📋 Central Points Details"):
                            central_display = pd.DataFrame({
                                'Sample': [str(idx) for idx in st.session_state.mlr_central_points['indices']],
                                'Observed Y': central_y.values
                            })
                            
                            for col in central_X.columns:
                                central_display[col] = central_X[col].values
                            
                            st.dataframe(central_display, use_container_width=True)
                            
                        st.info("""
                        **Central Point Validation**: Use these points for model validation in the Predictions tab.
                        They help assess curvature and lack of fit at the experimental center.
                        """)

                # Check for experimental replicates (on the data used for modeling)
                replicate_info = detect_replicates(X_data, y_data)

                if replicate_info:
                    st.markdown("### 🔬 Experimental Replicates in Model Data")
                    
                    rep_col1, rep_col2, rep_col3, rep_col4 = st.columns(4)
                    
                    with rep_col1:
                        st.metric("Replicate Groups", replicate_info['n_replicate_groups'])
                    with rep_col2:
                        st.metric("Total Replicates", replicate_info['total_replicates'])
                    with rep_col3:
                        st.metric("Pooled Std Dev", f"{replicate_info['pooled_std']:.4f}")
                    with rep_col4:
                        st.metric("Replicate DOF", replicate_info['pooled_dof'])
                    
                    with st.expander("📋 Model Data Replicate Groups Details"):
                        rep_data = []
                        for i, group in enumerate(replicate_info['group_stats'], 1):
                            rep_data.append({
                                'Group': i,
                                'Samples': ', '.join([str(idx+1) for idx in group['indices']]),
                                'N': group['n_replicates'],
                                'Mean Y': f"{group['mean']:.4f}",
                                'Std Dev': f"{group['std']:.4f}",
                                'DOF': group['dof']
                            })
                        
                        rep_df = pd.DataFrame(rep_data)
                        st.dataframe(rep_df, use_container_width=True)
                    
                    st.info(f"""
                    **Model Data Experimental Error** = {replicate_info['pooled_std']:.4f} 
                    (from {replicate_info['pooled_dof']} degrees of freedom)
                    
                    This represents the experimental error in the data actually used for modeling.
                    """)
                    
                    # Compare model replicates vs full replicates
                    if replicate_info['pooled_std'] != replicate_info_full['pooled_std']:
                        st.warning(f"""
                        **Note**: Model data experimental error ({replicate_info['pooled_std']:.4f}) differs from 
                        full dataset experimental error ({replicate_info_full['pooled_std']:.4f}).
                        This occurs when central point replicates are excluded from modeling.
                        """)
                else:
                    st.info("No experimental replicates detected in the model data.")

                # ========== SUMMARY SECTION (MOVED OUT AND FIXED) ==========
                
                st.markdown("---")
                st.markdown("### 📋 Statistical Analysis Summary")
                
                # Initialize variables with defaults to avoid NameError
                var_y_doe = model_results.get('var_y', 0)
                dof_y_doe = len(all_y_data) - 1
                p_global = 1.0
                central_mean = 0
                
                # Recalculate DoE variance if central points were excluded
                if central_points and exclude_central_points:
                    var_y_doe = np.var(y_data, ddof=1)
                    dof_y_doe = len(y_data) - 1
                
                # Recalculate p_global if replicates exist
                if replicate_info_full:
                    f_global = var_y_doe / replicate_info_full['pooled_variance']
                    p_global = 1 - stats.f.cdf(f_global, dof_y_doe, replicate_info_full['pooled_dof'])
                
                # Calculate central point mean if they exist
                if central_points and exclude_central_points and 'mlr_central_points' in st.session_state:
                    central_mean = st.session_state.mlr_central_points['y'].mean()
                
                summary_text = f"""
                **Experimental Design Quality:**
                
                📊 **Data Structure:**
                - Total samples: {len(all_y_data)}
                - Model samples: {len(y_data)}
                - Replicate groups: {replicate_info_full['n_replicate_groups'] if replicate_info_full else 0}
                - Central points: {len(central_points) if central_points else 0}
                
                🔬 **Experimental Error:**
                - Pure error: σ_exp = {replicate_info_full['pooled_std']:.4f} (DOF = {replicate_info_full['pooled_dof']}) {'' if replicate_info_full else '(No replicates detected)'}
                - Model error: RMSE = {model_results['rmse']:.4f} (DOF = {model_results['dof']})
                - Error ratio: {model_results['rmse']/replicate_info_full['pooled_std']:.2f} {'' if replicate_info_full else '(Cannot calculate - no replicates)'}
                
                📈 **Factor Effects:**
                - DoE variance: σ²_DoE = {var_y_doe:.6f} (DOF = {dof_y_doe})
                - Effect significance: p = {p_global:.4f}
                - Variance amplification: {var_y_doe/replicate_info_full['pooled_variance']:.1f}× {'' if replicate_info_full else '(Cannot calculate - no replicates)'}
                """
                
                if central_points and exclude_central_points:
                    summary_text += f"""
                    
                🎯 **Central Points:**
                - Excluded from model: {len(central_points)} points
                - Reserved for validation
                - Mean response: {central_mean:.4f}
                    """
                
                st.info(summary_text)


                # Display results - EQUIVALENT TO R OUTPUT
                st.markdown("### 📋 Model Summary")
                
                # Dispersion Matrix (XtX)^-1
                st.markdown("#### Dispersion Matrix")
                dispersion_df = pd.DataFrame(
                    model_results['XtX_inv'],
                    index=model_results['X'].columns,
                    columns=model_results['X'].columns
                )
                st.dataframe(dispersion_df.round(4), use_container_width=True)
                
                trace = np.trace(model_results['XtX_inv'])
                st.info(f"**Trace of Dispersion Matrix:** {trace:.4f}")
                
                # VIF (Variance Inflation Factors)
                if 'vif' in model_results:
                    st.markdown("#### Variance Inflation Factors (VIF)")
                    vif_df = model_results['vif'].to_frame('VIF')
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
                        st.dataframe(vif_df_clean.round(4), use_container_width=True)
                        
                        st.info("""
                        **VIF Interpretation:**
                        - VIF = 1: No correlation
                        - VIF < 2: OK
                        - VIF < 4: Good
                        - VIF < 8: Acceptable
                        - VIF > 8: High multicollinearity (problematic)
                        """)
                    else:
                        st.info("VIF not applicable for this model")
                
                # Leverage
                st.markdown("#### Leverage of Experimental Points")
                leverage_series = pd.Series(model_results['leverage'], index=range(1, len(model_results['leverage'])+1))
                st.dataframe(leverage_series.to_frame('Leverage').T.round(4), use_container_width=True)
                st.info(f"**Maximum Leverage:** {model_results['leverage'].max():.4f}")

                # Comparison between model error and experimental error
                if replicate_info and 'rmse' in model_results:
                    st.markdown("#### 🎯 Model vs Experimental Error Comparison")
                    
                    comparison_col1, comparison_col2, comparison_col3 = st.columns(3)
                    
                    with comparison_col1:
                        st.metric("Model RMSE", f"{model_results['rmse']:.4f}")
                    
                    with comparison_col2:
                        st.metric("Experimental Std Dev", f"{replicate_info['pooled_std']:.4f}")
                    
                    with comparison_col3:
                        ratio = model_results['rmse'] / replicate_info['pooled_std']
                        st.metric("RMSE / Exp. Std Dev", f"{ratio:.2f}")
                    
                    if ratio < 1.2:
                        st.success("✅ Model error is close to experimental error - excellent fit!")
                    elif ratio < 2.0:
                        st.info("ℹ️ Model error is reasonable compared to experimental error")
                    else:
                        st.warning("⚠️ Model error significantly exceeds experimental error - consider additional terms or transformation")

                # Coefficients table
                st.markdown("### 📊 Model Coefficients")
                
                coef_df = pd.DataFrame({'Coefficient': model_results['coefficients']})
                
                if 'se_coef' in model_results:
                    coef_df['Std. Error'] = model_results['se_coef']
                    coef_df['t-statistic'] = model_results['t_stats']
                    coef_df['p-value'] = model_results['p_values']
                    coef_df['CI Lower'] = model_results['ci_lower']
                    coef_df['CI Upper'] = model_results['ci_upper']
                    
                    def add_stars(p):
                        if p <= 0.001:
                            return '***'
                        elif p <= 0.01:
                            return '**'
                        elif p <= 0.05:
                            return '*'
                        else:
                            return ''
                    
                    coef_df['Sig.'] = coef_df['p-value'].apply(add_stars)

                st.dataframe(coef_df.round(4), use_container_width=True)
                st.info("Significance codes: *** p≤0.001, ** p≤0.01, * p≤0.05")

                # Coefficients bar plot
                st.markdown("#### Coefficients Bar Plot")
                
                coefficients = model_results['coefficients']
                coef_no_intercept = coefficients[coefficients.index != 'Intercept']
                coef_names = coef_no_intercept.index.tolist()
                
                if len(coef_names) == 0:
                    st.warning("No coefficients to plot (model contains only intercept)")
                else:
                    colors = []
                    for name in coef_names:
                        if '*' in name:
                            n_asterisks = name.count('*')
                            colors.append('cyan' if n_asterisks > 1 else 'green')
                        elif '^2' in name or '^' in name:
                            colors.append('cyan')
                        else:
                            colors.append('red')
                    
                    fig = go.Figure()
                    
                    fig.add_trace(go.Bar(
                        x=coef_names,
                        y=coef_no_intercept.values,
                        marker_color=colors,
                        marker_line_color='black',
                        marker_line_width=1,
                        name='Coefficients',
                        showlegend=False
                    ))
                    
                    if 'ci_lower' in model_results and 'ci_upper' in model_results:
                        ci_lower = model_results['ci_lower'][coef_no_intercept.index].values
                        ci_upper = model_results['ci_upper'][coef_no_intercept.index].values
                        
                        error_minus = coef_no_intercept.values - ci_lower
                        error_plus = ci_upper - coef_no_intercept.values
                        
                        fig.update_traces(
                            error_y=dict(
                                type='data',
                                symmetric=False,
                                array=error_plus,
                                arrayminus=error_minus,
                                color='black',
                                thickness=2,
                                width=4
                            )
                        )
                    
                    if 'p_values' in model_results:
                        p_values = model_results['p_values'][coef_no_intercept.index].values
                        for i, (name, coef, p) in enumerate(zip(coef_names, coef_no_intercept.values, p_values)):
                            y_pos = coef
                            y_offset = max(abs(coef) * 0.05, 0.01) if coef >= 0 else -max(abs(coef) * 0.05, 0.01)
                            
                            if p <= 0.001:
                                sig_text = '***'
                            elif p <= 0.01:
                                sig_text = '**'
                            elif p <= 0.05:
                                sig_text = '*'
                            else:
                                sig_text = None
                            
                            if sig_text:
                                fig.add_annotation(
                                    x=name, y=y_pos + y_offset,
                                    text=sig_text,
                                    showarrow=False,
                                    font=dict(size=20, color='black'),
                                    yshift=10 if coef >= 0 else -10
                                )
                    
                    fig.update_layout(
                        title=f"Coefficients - {y_var}",
                        xaxis_title="Term",
                        yaxis_title="Coefficient Value",
                        height=500,
                        xaxis={'tickangle': 45},
                        showlegend=False,
                        yaxis=dict(zeroline=True, zerolinewidth=2, zerolinecolor='gray')
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown("""
                    **Color legend:** 
                    - Red = Linear terms
                    - Green = Two-term interactions
                    - Cyan = Quadratic terms
                    """)
                    
                    st.info("Significance markers: *** p≤0.001, ** p≤0.01, * p≤0.05")

                # Cross-validation results
                if 'q2' in model_results:
                    st.markdown("### 🔄 Cross-Validation Results")
                    
                    cv_col1, cv_col2 = st.columns(2)
                    with cv_col1:
                        st.metric("RMSECV", f"{model_results['rmsecv']:.4f}")
                    with cv_col2:
                        st.metric("Q² (LOO-CV)", f"{model_results['q2']:.4f}")

            except Exception as e:
                st.error(f"❌ Error fitting model: {str(e)}")
                import traceback
                if st.checkbox("Show debug info"):
                    st.code(traceback.format_exc())

    # ===== MODEL DIAGNOSTICS TAB =====
    with tab2:
        st.markdown("## 📊 Model Diagnostics")
        st.markdown("*Equivalent to DOE diagnostic plots*")
        
        if 'mlr_model' not in st.session_state:
            st.warning("⚠️ No MLR model fitted. Please fit a model first.")
        else:
            model_results = st.session_state.mlr_model
            
            diagnostic_type = st.selectbox(
                "Select diagnostic plot:",
                [
                    "📈 Experimental vs Fitted",
                    "📉 Residuals vs Fitted",
                    "🔄 Experimental vs CV Predicted",
                    "📊 CV Residuals",
                    "🎯 Leverage Plot",
                    "📐 Coefficients Bar Plot"
                ]
            )
            
            if diagnostic_type == "📈 Experimental vs Fitted":
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
            
            elif diagnostic_type == "📉 Residuals vs Fitted":
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
            
            elif diagnostic_type == "📐 Coefficients Bar Plot":
                st.markdown("### 📐 Model Coefficients")
                
                coefficients = model_results['coefficients']
                
                # Filter out intercept term
                coef_no_intercept = coefficients[coefficients.index != 'Intercept']
                coef_names = coef_no_intercept.index.tolist()
                
                if len(coef_names) == 0:
                    st.warning("No coefficients to plot (model contains only intercept)")
                else:
                    # Determine colors based on coefficient type (excluding intercept)
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
                    
                    # Add error bars if available (excluding intercept)
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
                    
                    # Add significance markers (excluding intercept)
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

    # ===== RESPONSE SURFACE TAB =====
    with tab3:
        st.markdown("## 📈 Response Surface")
        st.markdown("*Equivalent to DOE_response_surface.r*")
        
        if 'mlr_model' not in st.session_state:
            st.warning("⚠️ No MLR model fitted.")
        else:
            st.info("🚧 Advanced response surface analysis available in professional versions")
            st.markdown("""
            **Professional DoE features include:**
            
            📈 **Response Surface Methods:**
            - Interactive 3D surface plots
            - Contour optimization maps
            - Multiple response optimization

            
            🎯 **Advanced Designs:**
            - D-optimal designs
            - Custom experimental strategies
            
            🔍 **Analysis Tools:**
            - Monte Carlo simulations
            - Robust parameter design
            - Mixture designs
            - Split-plot designs
            
            📞 **Contact for full DoE capabilities:** [chemometricsolutions.com](https://chemometricsolutions.com)
            """)

    # ===== CONFIDENCE INTERVALS TAB =====
    with tab4:
        st.markdown("## 🎨 Confidence Intervals")
        st.markdown("*Equivalent to DOE_CI_surface.r*")
        
        if 'mlr_model' not in st.session_state:
            st.warning("⚠️ No MLR model fitted.")
        else:
            st.info("🚧 Confidence interval visualization will be implemented")
            st.markdown("Features planned:")
            st.markdown("- Prediction confidence interval surface")
            st.markdown("- Experimental confidence interval surface")
            st.markdown("- Leverage surface")

    # ===== PREDICTIONS TAB =====
    with tab5:
        st.markdown("## 🔮 Predictions for New Points")
        st.markdown("*Equivalent to DOE_prediction.r*")
        
        if 'mlr_model' not in st.session_state:
            st.warning("⚠️ No MLR model fitted.")
        else:
            st.info("🚧 Prediction interface will be implemented")
            st.markdown("Features planned:")
            st.markdown("- Manual input of new experimental conditions")
            st.markdown("- Batch prediction from file")
            st.markdown("- Confidence intervals for predictions")
            st.markdown("- Leverage analysis for new points")

    # ===== GENERATE MATRIX TAB (NOW CORRECTLY IN TAB6) =====
    with tab6:
        st.markdown("## 🎯 Generate Candidate Points")
        st.markdown("*Equivalent to DOE_candidate_points.r*")
        
        st.info("Create a full factorial design by specifying factor levels")
        
        n_variables = st.number_input("Number of variables:", min_value=2, max_value=10, value=3)
        
        variables_config = {}
        
        st.markdown("### Variable Configuration")
        
        for i in range(n_variables):
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col1:
                var_name = st.text_input(f"Variable {i+1} name:", value=f"X{i+1}", key=f"var_name_{i}")
            
            with col2:
                var_type = st.selectbox(f"Type:", ["Numeric", "Categorical"], key=f"var_type_{i}")
            
            with col3:
                if var_type == "Numeric":
                    levels_input = st.text_input(f"Levels:", value="0,1,2", key=f"var_levels_{i}",
                                                help="Comma-separated numeric values")
                else:
                    levels_input = st.text_input(f"Levels:", value="Low,Medium,High", key=f"var_levels_{i}",
                                                help="Comma-separated text values")
            
            variables_config[var_name] = levels_input
        
        if st.button("🚀 Generate Candidate Points", type="primary"):
            try:
                candidate_points = generate_candidate_points(variables_config)
                
                st.success(f"✅ Generated {len(candidate_points)} candidate points")
                
                # Store in session state
                st.session_state.candidate_points = candidate_points
                
                # Display
                st.markdown("### Generated Design Matrix")
                st.dataframe(candidate_points, use_container_width=True)
                
                # Statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Points", len(candidate_points))
                with col2:
                    st.metric("Variables", len(variables_config))
                with col3:
                    memory_kb = candidate_points.memory_usage(deep=True).sum() / 1024
                    st.metric("Size (KB)", f"{memory_kb:.1f}")
                
                # Export option
                csv_data = candidate_points.to_csv(index=False)
                st.download_button(
                    "💾 Download Candidate Points CSV",
                    csv_data,
                    "candidate_points.csv",
                    "text/csv"
                )
                
            except Exception as e:
                st.error(f"❌ Error generating candidate points: {str(e)}")

    # ===== EXTRACT & EXPORT TAB =====
    with tab7:
        st.markdown("## 💾 Extract & Export")
        st.markdown("*Equivalent to DOE_extract.r*")
        
        if 'mlr_model' not in st.session_state:
            st.warning("⚠️ No MLR model fitted.")
        else:
            model_results = st.session_state.mlr_model
            
            st.markdown("### 📊 Available Data for Export")
            
            export_options = {
                "📐 Coefficients": model_results['coefficients'].to_frame('Coefficient'),
                "📈 Fitted Values": pd.DataFrame({'Fitted': model_results['y_pred']}),
                "📉 Residuals": pd.DataFrame({'Residuals': model_results['residuals']}),
            }
            
            if 'cv_predictions' in model_results:
                export_options["🔄 CV Predictions"] = pd.DataFrame({'CV_Predicted': model_results['cv_predictions']})
                export_options["📊 CV Residuals"] = pd.DataFrame({'CV_Residuals': model_results['cv_residuals']})
            
            if 'XtX_inv' in model_results:
                export_options["🔢 Dispersion Matrix"] = pd.DataFrame(
                    model_results['XtX_inv'],
                    index=model_results['X'].columns,
                    columns=model_results['X'].columns
                )
            
            for name, df in export_options.items():
                with st.expander(f"{name} ({df.shape[0]}×{df.shape[1]})"):
                    st.dataframe(df, use_container_width=True)
                    
                    csv_data = df.to_csv(index=True)
                    clean_name = name.replace("📐 ", "").replace("📈 ", "").replace("📉 ", "").replace("🔄 ", "").replace("📊 ", "").replace("🔢 ", "").replace(" ", "_")
                    
                    st.download_button(
                        f"💾 Download {name} as CSV",
                        csv_data,
                        f"MLR_{clean_name}.csv",
                        "text/csv",
                        key=f"download_{clean_name}"
                    )
            
            # Model summary export
            if st.button("📦 Export Complete MLR Analysis"):
                try:
                    excel_buffer = BytesIO()
                    
                    with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                        # Coefficients
                        coef_export = pd.DataFrame({
                            'Coefficient': model_results['coefficients'],
                            'Std_Error': model_results.get('se_coef', [np.nan]*len(model_results['coefficients'])),
                            'p_value': model_results.get('p_values', [np.nan]*len(model_results['coefficients']))
                        })
                        coef_export.to_excel(writer, sheet_name='Coefficients', index=True)
                        
                        # Fitted values
                        fitted_export = pd.DataFrame({
                            'Experimental': model_results['y'].values,
                            'Fitted': model_results['y_pred'],
                            'Residuals': model_results['residuals']
                        })
                        fitted_export.to_excel(writer, sheet_name='Fitted_Values', index=True)
                        
                        # Model summary
                        summary_data = {
                            'Metric': ['N_Samples', 'N_Features', 'DOF', 'R_Squared', 'RMSE'],
                            'Value': [
                                model_results['n_samples'],
                                model_results['n_features'],
                                model_results['dof'],
                                model_results.get('r_squared', np.nan),
                                model_results.get('rmse', np.nan)
                            ]
                        }
                        summary_df = pd.DataFrame(summary_data)
                        summary_df.to_excel(writer, sheet_name='Summary', index=False)
                    
                    excel_buffer.seek(0)
                    
                    st.download_button(
                        "📄 Download Complete MLR Analysis (XLSX)",
                        excel_buffer.getvalue(),
                        "Complete_MLR_Analysis.xlsx",
                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                    
                    st.success("✅ Complete MLR analysis ready for download!")
                    
                except Exception as e:
                    st.error(f"Excel export failed: {str(e)}")
                    st.info("Individual CSV exports are still available above")
