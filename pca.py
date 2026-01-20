"""
PCA Analysis Module - ChemometricSolutions

Principal Component Analysis interface for Streamlit application.
Pure NIPALS implementation with NO sklearn dependencies.

Author: ChemometricSolutions
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Optional, Dict, Any
from pathlib import Path
import io

# Import PCA calculation functions (REQUIRED - from pca_utils)
from pca_utils.pca_calculations import compute_pca, varimax_rotation, varimax_with_scores

# Try to import plotting and statistics modules (optional - from pca_utils)
try:
    from pca_utils.pca_plots import (
        plot_scores, plot_loadings, plot_scree,
        plot_cumulative_variance, plot_biplot, plot_loadings_line,
        plot_loadings_line_antiderivative, plot_loadings_antiderivative,
        add_convex_hulls, add_sample_trajectory_lines, plot_line_scores,
        plot_loadings_scores_side_by_side
    )
    PLOTS_AVAILABLE = True
except ImportError as e:
    st.error(f"⚠️ Could not import plotting functions: {e}")
    PLOTS_AVAILABLE = False
except Exception as e:
    st.error(f"⚠️ Unexpected error importing plotting functions: {e}")
    PLOTS_AVAILABLE = False

try:
    from pca_utils.pca_statistics import (
        calculate_hotelling_t2, calculate_q_residuals, calculate_contributions
    )
    STATS_AVAILABLE = True
except ImportError:
    STATS_AVAILABLE = False

try:
    from color_utils import (
        get_unified_color_schemes, create_categorical_color_map,
        is_quantitative_variable
    )
    COLORS_AVAILABLE = True
except ImportError:
    COLORS_AVAILABLE = False

# Try to import contribution analysis functions from pca_monitoring_page
try:
    from pca_monitoring_page import (
        calculate_all_contributions,
        create_contribution_plot_all_vars
    )
    CONTRIB_FUNCS_AVAILABLE = True
except ImportError:
    CONTRIB_FUNCS_AVAILABLE = False

# Try to import missing data reconstruction functions
try:
    from pca_utils.missing_data_reconstruction import (
        count_missing_values,
        reconstruct_missing_data,
        get_reconstruction_info
    )
    MISSING_DATA_AVAILABLE = True
except ImportError:
    MISSING_DATA_AVAILABLE = False


# ============================================================================
# HELPER FUNCTIONS FOR FLEXIBLE CORRELATION ANALYSIS
# ============================================================================

def get_top_contributors(contrib_vector, variables, n_top=10):
    """
    Get top contributors ranked by absolute contribution magnitude.

    Parameters
    ----------
    contrib_vector : array-like
        Contribution vector (T² or Q contributions)
    variables : list
        Variable names
    n_top : int
        Number of top contributors to return

    Returns
    -------
    list of tuples
        [(var_name, contribution_value), ...] sorted by |contribution| descending
    """
    contrib_abs = np.abs(contrib_vector)
    top_indices = np.argsort(contrib_abs)[::-1][:n_top]
    return [(variables[i], contrib_abs[i]) for i in top_indices]


def get_top_correlated(X, var_idx, variables, n_top=10):
    """
    Get variables most correlated to a given variable.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_variables)
        Data matrix
    var_idx : int
        Index of the variable to correlate with
    variables : list
        Variable names
    n_top : int
        Number of top correlated variables to return

    Returns
    -------
    list of tuples
        [(var_name, correlation), ...] sorted by |correlation| descending
    """
    correlations = {}
    for i, var in enumerate(variables):
        if i != var_idx:
            corr = np.corrcoef(X[:, var_idx], X[:, i])[0, 1]
            # Handle NaN correlations (e.g., constant columns)
            if not np.isnan(corr):
                correlations[var] = corr

    # Sort by absolute correlation value (descending)
    sorted_corr = sorted(correlations.items(),
                        key=lambda x: abs(x[1]),
                        reverse=True)[:n_top]
    return sorted_corr


def create_correlation_scatter(X_train, X_sample, var1_idx, var2_idx,
                               var1_name, var2_name, correlation_val, sample_idx):
    """
    Create flexible correlation scatter plot showing training (grey), sample (red star).

    Parameters
    ----------
    X_train : array-like, shape (n_samples, n_variables)
        Training data
    X_sample : array-like, shape (n_variables,)
        Single sample (outlier point marked as red star)
    var1_idx : int
        Column index for X-axis variable
    var2_idx : int
        Column index for Y-axis variable
    var1_name : str
        Name of X-axis variable
    var2_name : str
        Name of Y-axis variable
    correlation_val : float
        Pearson correlation coefficient (for title)
    sample_idx : int
        Sample index (for legend)

    Returns
    -------
    plotly.graph_objects.Figure
    """
    # Calculate axis ranges including BOTH training data AND sample point
    sample_x = X_sample[var1_idx]
    sample_y = X_sample[var2_idx]

    # Include sample in axis calculation (so star never disappears)
    x_min = min(X_train[:, var1_idx].min(), sample_x)
    x_max = max(X_train[:, var1_idx].max(), sample_x)
    y_min = min(X_train[:, var2_idx].min(), sample_y)
    y_max = max(X_train[:, var2_idx].max(), sample_y)

    x_padding = (x_max - x_min) * 0.05
    y_padding = (y_max - y_min) * 0.05

    x_range = [x_min - x_padding, x_max + x_padding]
    y_range = [y_min - y_padding, y_max + y_padding]

    # Create figure
    fig = go.Figure()

    # Training set (grey, sampled for performance if dataset is large)
    n_plot = min(1000, X_train.shape[0])
    sample_indices = np.random.choice(X_train.shape[0], n_plot, replace=False)

    fig.add_trace(go.Scatter(
        x=X_train[sample_indices, var1_idx],
        y=X_train[sample_indices, var2_idx],
        mode='markers',
        name='Training',
        marker=dict(color='darkgrey', size=5, opacity=0.7),
        hovertemplate=f'{var1_name}: %{{x:.2f}}<br>{var2_name}: %{{y:.2f}}<extra></extra>'
    ))

    # Selected sample (red star)
    fig.add_trace(go.Scatter(
        x=[X_sample[var1_idx]],
        y=[X_sample[var2_idx]],
        mode='markers',
        name=f'Sample {sample_idx+1}',
        marker=dict(color='red', size=15, symbol='star'),
        hovertemplate=f'{var1_name}: %{{x:.2f}}<br>{var2_name}: %{{y:.2f}}<br>Sample {sample_idx+1}<extra></extra>'
    ))

    # Get unified color scheme
    color_scheme = get_unified_color_schemes() if COLORS_AVAILABLE else {
        'background': 'white',
        'paper': 'white',
        'text': 'black',
        'grid': 'lightgray'
    }

    # Update layout with auto-scaled axes (including sample point)
    fig.update_layout(
        title=f'{var1_name} vs {var2_name}<br><sub>Correlation (training): r = {correlation_val:.3f}</sub>',
        xaxis_title=var1_name,
        yaxis_title=var2_name,
        plot_bgcolor=color_scheme['background'],
        paper_bgcolor=color_scheme['paper'],
        font=dict(color=color_scheme['text']),
        hovermode='closest',
        width=600,
        height=550,
        showlegend=True,
        xaxis=dict(
            showgrid=True,
            gridcolor=color_scheme['grid'],
            range=x_range,
            autorange=False
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor=color_scheme['grid'],
            range=y_range,
            autorange=False
        )
    )

    return fig


def show():
    """
    Main PCA Analysis page.
    
    Displays a multi-tab interface for complete PCA workflow:
    1. Model Computation
    2. Variance Plots
    3. Loading Plots
    4. Score Plots
    5. Interpretation
    6. Advanced Diagnostics
    7. Extract & Export
    """
    st.markdown("# 🎯 Principal Component Analysis (PCA)")
    st.markdown("*Pure NIPALS implementation - No sklearn dependencies*")
    
    # Check if data is loaded
    if 'current_data' not in st.session_state:
        st.warning("⚠️ No data loaded. Please go to **Data Handling** to load your dataset first.")
        return
    
    data = st.session_state.current_data

    # === COMPLETE DATASET VALIDATION ===
    # Check 1: Empty dataset (no samples)
    if len(data) == 0:
        st.error("❌ Dataset is empty - no samples available!")
        st.info("📊 Possible causes: All samples removed by lasso selection / Empty file loaded")
        st.info("📊 Action: Go back to Data Handling or reload original dataset")
        return

    # Check 2: No columns
    if len(data.columns) == 0:
        st.error("❌ Dataset has no columns!")
        return

    # Check 3: No numeric columns
    if data.select_dtypes(include=[np.number]).shape[1] == 0:
        st.error("❌ No numeric columns found - PCA cannot run on non-numeric data!")
        st.info("📊 Action: Go back to Data Handling and ensure numeric variables are included")
        return

    st.divider()

    # Create tabs
    tabs = st.tabs([
        "🔧 Model Computation",
        "📊 Variance Plots",
        "📈 Loadings Plots",
        "🎯 Score Plots",
        "📝 Interpretation",
        "🔬 Advanced Diagnostics",
        "💾 Extract & Export",
        "🔄 Missing Data Reconstruction"
    ])
    
    # TAB 1: Model Computation
    with tabs[0]:
        _show_model_computation_tab(data)
    
    # TAB 2: Variance Plots
    with tabs[1]:
        _show_variance_plots_tab()
    
    # TAB 3: Loadings Plots
    with tabs[2]:
        _show_loadings_plots_tab()
    
    # TAB 4: Score Plots
    with tabs[3]:
        _show_score_plots_tab()
    
    # TAB 5: Interpretation
    with tabs[4]:
        _show_interpretation_tab()
    
    # TAB 6: Advanced Diagnostics
    with tabs[5]:
        _show_advanced_diagnostics_tab()
    
    # TAB 7: Extract & Export
    with tabs[6]:
        _show_export_tab()

    # TAB 8: Missing Data Reconstruction
    with tabs[7]:
        _show_missing_data_reconstruction_tab()


# ============================================================================
# TAB 1: MODEL COMPUTATION
# ============================================================================

def _show_model_computation_tab(data: pd.DataFrame):
    """
    Display the Model Computation tab.

    Allows users to:
    - Select dataset from workspace (training/test splits or current data)
    - Select variables and samples
    - Configure preprocessing (centering, scaling)
    - Choose number of components
    - Compute PCA model
    - Apply Varimax rotation (optional)
    """
    st.markdown("## 🔧 PCA Model Computation")
   # st.markdown("*Equivalent to R PCA_model_PCA.r*")

    # === SECTION 0: SELECT DATASET FROM WORKSPACE ===
    from workspace_utils import display_workspace_dataset_selector

    st.markdown("### 📊 Select Dataset from Workspace")
    st.info("Choose dataset for PCA model computation")

    # USE the same function that works in pca_monitoring!
    result = display_workspace_dataset_selector(
        label="Select dataset:",
        key="pca_dataset_select"
    )

    if result:
        dataset_name, data = result

        # UPDATE session state
        st.session_state.current_data = data
        st.session_state.dataset_name = dataset_name

        # Display info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📊 Samples", len(data))
        with col2:
            st.metric("📈 Variables", len(data.columns))
        with col3:
            st.metric("🔢 Numeric", data.select_dtypes(include=[np.number]).shape[1])

        st.success(f"✅ Loaded: {dataset_name}")
    else:
        st.warning("No dataset selected")
        return

    st.divider()

    # === DATASET OVERVIEW ===
    st.markdown("### 📊 Dataset Overview")

    total_cols = len(data.columns)
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    non_numeric_cols = data.select_dtypes(exclude=[np.number]).columns.tolist()

    col1, col2 = st.columns(2)
    with col1:
        st.info(f"📋 **Dataset**: {len(data)} samples, {total_cols} columns")
    with col2:
        st.info(f"🔢 **Numeric**: {len(numeric_cols)} variables | **Non-numeric**: {len(non_numeric_cols)}")

    # Validate minimum data requirements
    if len(data) == 0:
        st.error("❌ Dataset is empty - no samples available")
        st.info("💡 Please load data or check your lasso selection")
        return

    if len(data.columns) == 0:
        st.error("❌ Dataset has no columns")
        return

    if len(numeric_cols) == 0:
        st.error("❌ No numeric columns found in dataset")
        st.info("💡 PCA requires numeric variables")
        return

    if len(numeric_cols) > 100:
        st.success(f"🔬 **Spectral data detected**: {len(numeric_cols)} variables (likely NIR/spectroscopy)")

    # === VARIABLE/SAMPLE SELECTION ===
    st.markdown("### 🎯 Variable & Sample Selection")
    
    col1, col2 = st.columns(2)
    with col1:
        first_col = st.number_input(
            "First column (1-based):", 
            min_value=1, 
            max_value=len(data.columns),
            value=2 if len(non_numeric_cols) > 0 else 1,
            help="Start from column 2 to skip ID/metadata column"
        )
        first_row = st.number_input(
            "First sample (1-based):",
            min_value=1,
            max_value=len(data),
            value=1
        )
    
    with col2:
        last_col = st.number_input(
            "Last column (1-based):",
            min_value=1,
            max_value=len(data.columns),
            value=len(data.columns)
        )
        last_row = st.number_input(
            "Last sample (1-based):",
            min_value=1,
            max_value=len(data),
            value=len(data)
        )
    
    # Select data subset
    selected_data = data.iloc[first_row-1:last_row, first_col-1:last_col]
    n_vars = last_col - first_col + 1
    n_samples = last_row - first_row + 1
    
    st.info(f"📊 Selected: **{n_samples} samples** × **{n_vars} variables**")
    
    # Preview selected data
    with st.expander("👁️ Preview Selected Data"):
        st.dataframe(selected_data.head(10), use_container_width=True)
    
    # Filter to numeric columns only
    numeric_data = selected_data.select_dtypes(include=[np.number])
    
    if len(numeric_data.columns) == 0:
        st.error("❌ No numeric columns in selected range! Please adjust column selection.")
        return
    
    st.success(f"✅ Will analyze {len(numeric_data.columns)} numeric variables")
    
    # === PREPROCESSING ===
    st.markdown("### ⚙️ Preprocessing Options")
    
    col1, col2 = st.columns(2)
    with col1:
        center_data = st.checkbox(
            "**Center data** (mean-centering)",
            value=True,
            help="Subtract column means - standard for PCA"
        )
    with col2:
        scale_data = st.checkbox(
            "**Scale data** (unit variance)",
            value=True,
            help="Divide by std dev - recommended for variables with different units"
        )
    
    if center_data and scale_data:
        st.info("📌 **Autoscaling** enabled (centering + scaling = correlation matrix PCA)")
    elif center_data:
        st.info("📌 **Mean-centering** only (covariance matrix PCA)")
    else:
        st.warning("⚠️ No preprocessing - analyzing raw data (unusual for PCA)")
    
    # === NUMBER OF COMPONENTS ===
    st.markdown("### 📊 Number of Components")

    max_components = min(len(numeric_data), len(numeric_data.columns)) - 1
    default_n = min(10, max_components)

    n_components_display = st.slider(
        "Number of components to display/analyze:",
        min_value=2,
        max_value=min(20, max_components),
        value=default_n,
        help="Components to display in plots and analysis (internally computes all for mathematical correctness)"
    )

    st.info(f"🚀 Will compute **{n_components_display}** components using NIPALS algorithm")

    # === CHECK FOR MISSING VALUES ===
    has_missing = numeric_data.isnull().any().any()
    if has_missing:
        n_missing = numeric_data.isnull().sum().sum()
        total_values = numeric_data.shape[0] * numeric_data.shape[1]
        pct_missing = (n_missing / total_values) * 100
        st.warning(f"⚠️ Dataset contains **{n_missing:,}** missing values ({pct_missing:.2f}%)")
        st.info("✅ NIPALS algorithm handles missing values natively (no imputation needed)")

    # === COMPUTE PCA BUTTON ===
    st.markdown("---")

    if st.button("🚀 Compute PCA Model", type="primary", use_container_width=True):
        try:
            with st.spinner("Computing PCA with NIPALS algorithm..."):
                import time
                start_time = time.time()

                # Compute ONLY the requested number of components
                # No need to compute all - NIPALS is efficient with partial computation
                pca_dict_display = compute_pca(
                    X=numeric_data,
                    n_components=n_components_display,  # ← ONLY what user requested
                    center=center_data,
                    scale=scale_data
                )

                elapsed = time.time() - start_time

                # Preprocess data for variance_explained calculation
                # (NIPALS does this internally, but we need it for R formula)
                X_preprocessed = numeric_data.copy()
                if center_data:
                    X_preprocessed = X_preprocessed - pca_dict_display['means']
                if scale_data:
                    X_preprocessed = X_preprocessed / pca_dict_display['stds']

                # Store results in session state
                st.session_state['pca_results'] = {
                    **pca_dict_display,
                    'method': 'Standard PCA',
                    'selected_vars': numeric_data.columns.tolist(),
                    'computation_time': elapsed,
                    'varimax_applied': False,
                    'original_data': numeric_data,  # Raw data
                    'X_preprocessed': X_preprocessed  # For variance_explained calculation
                }

                # For backward compatibility with code below
                pca_dict = pca_dict_display
                n_components = n_components_display

                st.success(f"✅ PCA computation completed in {elapsed:.2f} seconds!")

                # Display results summary
                st.markdown("### 📊 PCA Results Summary")

                # Create metrics row
                metric_cols = st.columns(4)
                with metric_cols[0]:
                    st.metric("Algorithm", pca_dict['algorithm'])
                with metric_cols[1]:
                    st.metric("Components", n_components)
                with metric_cols[2]:
                    variance_1 = pca_dict['explained_variance_ratio'][0] * 100
                    st.metric("PC1 Variance", f"{variance_1:.1f}%")
                with metric_cols[3]:
                    total_var = pca_dict['cumulative_variance'][-1] * 100
                    st.metric("Total Variance", f"{total_var:.1f}%")
                
                # Variance table
                st.markdown("#### 📈 Variance Explained per Component")
                
                variance_df = pd.DataFrame({
                    'Component': [f'PC{i+1}' for i in range(n_components)],
                    'Eigenvalue': pca_dict['eigenvalues'][:n_components],
                    'Variance %': pca_dict['explained_variance_ratio'][:n_components] * 100,
                    'Cumulative %': pca_dict['cumulative_variance'][:n_components] * 100,
                    'Iterations': pca_dict['n_iterations'][:n_components]
                })
                
                st.dataframe(
                    variance_df.style.format({
                        'Eigenvalue': '{:.3f}',
                        'Variance %': '{:.2f}',
                        'Cumulative %': '{:.2f}',
                        'Iterations': '{:.0f}'
                    }),
                    use_container_width=True,
                    hide_index=True
                )

                # === SCREE PLOT: Guide component selection ===
                # Only show if data has missing values (reconstruction context)
                if has_missing:
                    st.markdown("---")
                    st.markdown("### 📊 Scree Plot - Choose Components")

                    # Use simple Scree Plot from Tab 2 (consistent!)
                    # Note: plot_scree is imported at module level (line 25)
                    component_labels = [f'PC{i+1}' for i in range(len(pca_dict['explained_variance_ratio']))]
                    fig = plot_scree(
                        pca_dict['explained_variance_ratio'],
                        is_varimax=False,
                        component_labels=component_labels
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    # === EXPLAINED VARIANCE TABLE ===
                    st.markdown("---")
                    st.markdown("#### 📋 Explained Variance by Component (Descending)")

                    # Create variance table
                    explained_var_pct = pca_dict['explained_variance_ratio'] * 100
                    cumulative_var_pct = pca_dict['cumulative_variance'] * 100

                    # Build DataFrame
                    variance_table = pd.DataFrame({
                        'Component': [f'PC{i+1}' for i in range(len(explained_var_pct))],
                        'Explained Variance (%)': [f'{var:.2f}%' for var in explained_var_pct],
                        'Cumulative Variance (%)': [f'{cum:.2f}%' for cum in cumulative_var_pct]
                    })

                    # Display as formatted table
                    st.dataframe(
                        variance_table,
                        use_container_width=True,
                        hide_index=True
                    )

                    # Find elbow point (80% cumulative variance)
                    cumulative_pct = pca_dict['cumulative_variance'] * 100
                    elbow_idx = 0
                    for i, cum_var in enumerate(cumulative_pct):
                        if cum_var >= 80:
                            elbow_idx = i
                            break

                    # Fallback: if no elbow found, use last component
                    if elbow_idx == 0 and cumulative_pct[-1] < 80:
                        elbow_idx = len(cumulative_pct) - 1

                    # Bounds check
                    elbow_idx = min(elbow_idx, len(cumulative_pct) - 1)

                    # Show key metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(
                            "🎯 Elbow at PC",
                            f"{elbow_idx + 1}",
                            f"{cumulative_pct[elbow_idx]:.1f}% variance"
                        )
                    with col2:
                        st.metric(
                            "📊 Total Components",
                            len(component_labels),
                            f"{cumulative_pct[-1]:.1f}% variance"
                        )
                    with col3:
                        st.metric(
                            "💡 Recommended",
                            f"{elbow_idx + 1}",
                            "for reconstruction"
                        )

                    # Store suggested components
                    st.session_state['suggested_components'] = elbow_idx + 1

                    st.info("👉 Use recommended components for data reconstruction below")

        except Exception as e:
            st.error(f"❌ PCA computation failed: {str(e)}")
            st.exception(e)

    # === PCA RECONSTRUCTION: Fill NaN using PCA model ===
    # Only show if PCA was computed AND data has missing values
    if 'pca_results' in st.session_state and numeric_data.isnull().any().any():
        pca_dict = st.session_state['pca_results']

        if 'scores' in pca_dict and pca_dict is not None:
            st.markdown("---")
            st.markdown("### 🔧 Reconstruct Missing Values Using PCA Model")

            # Get suggested components from Scree Plot (stored in session state)
            suggested_components = st.session_state.get('suggested_components', 5)

            st.info(f"""
            **Based on the Scree Plot above:**
            - **Suggested components**: {suggested_components} (≥80% cumulative variance)
            - **Adjust slider** if you want more/fewer components
            - **More components** = better reconstruction (may overfit)
            - **Fewer components** = smoother estimates

            **Formula**: X_reconstructed = Scores @ Loadings.T
            """)

            # User chooses how many components for reconstruction
            col1, col2 = st.columns([3, 1])

            with col1:
                n_components_recon = st.slider(
                    "🔢 Components for reconstruction:",
                    min_value=1,
                    max_value=pca_dict['n_components'],
                    value=min(suggested_components, pca_dict['n_components']),
                    help=f"Based on Scree Plot elbow point (suggested: {suggested_components})",
                    key="n_comp_recon_slider"
                )

            with col2:
                st.markdown("")
                st.markdown("")
                recon_button = st.button(
                    "🔄 Reconstruct",
                    use_container_width=True,
                    key="recon_btn_main"
                )

            if recon_button:
                try:
                    with st.spinner("🔄 Reconstructing missing values using NIPALS algorithm..."):
                        from pca_utils.missing_data_reconstruction import (
                            reconstruct_missing_data,
                            get_reconstruction_info
                        )

                        # Perform NIPALS reconstruction
                        X_reconstructed, recon_info_dict = reconstruct_missing_data(
                            X=numeric_data,
                            n_components=n_components_recon,
                            max_iter=1000,
                            tol=1e-6,
                            center=pca_dict.get('centering', True),
                            scale=pca_dict.get('scaling', False)
                        )

                        # Get additional reconstruction statistics
                        recon_info = get_reconstruction_info(numeric_data, X_reconstructed)

                        # Display reconstruction summary
                        st.markdown("#### 📊 Reconstruction Summary:")

                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric(
                                "Missing Before",
                                f"{recon_info['n_missing_before']:,}",
                                delta=-recon_info['n_filled']
                            )
                        with col2:
                            st.metric(
                                "Missing After",
                                f"{recon_info['n_missing_after']:,}"
                            )
                        with col3:
                            st.metric(
                                "Values Filled",
                                f"{recon_info['n_filled']:,}"
                            )
                        with col4:
                            fill_rate = (recon_info['n_filled'] / recon_info['n_missing_before'] * 100) if recon_info['n_missing_before'] > 0 else 0
                            st.metric(
                                "Fill Rate",
                                f"{fill_rate:.1f}%"
                            )

                        # Statistics of reconstructed values
                        st.markdown("#### 📈 Statistics of Reconstructed Values:")
                        stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
                        with stat_col1:
                            st.metric(
                                "Mean",
                                f"{recon_info['filled_mean']:.4f}"
                            )
                        with stat_col2:
                            st.metric(
                                "Std Dev",
                                f"{recon_info['filled_std']:.4f}"
                            )
                        with stat_col3:
                            st.metric(
                                "Min",
                                f"{recon_info['filled_min']:.4f}"
                            )
                        with stat_col4:
                            st.metric(
                                "Max",
                                f"{recon_info['filled_max']:.4f}"
                            )

                        # NIPALS Algorithm Results
                        st.markdown("#### 🔬 NIPALS Algorithm Results:")
                        nipals_col1, nipals_col2, nipals_col3 = st.columns(3)
                        with nipals_col1:
                            st.metric(
                                "Variance Explained",
                                f"{recon_info_dict['total_variance_explained']:.1f}%"
                            )
                        with nipals_col2:
                            converged_status = "✅ Yes" if recon_info_dict['converged'] else "⚠️ No"
                            st.metric(
                                "Converged",
                                converged_status
                            )
                        with nipals_col3:
                            st.metric(
                                "Iterations",
                                recon_info_dict['n_iterations']
                            )

                        # Show convergence warning if needed
                        if not recon_info_dict['converged']:
                            st.warning("⚠️ NIPALS did not fully converge. Consider increasing max iterations or reducing components.")

                        # === FIX #1: PRESERVE METADATA COLUMNS ===
                        # Check if original data has non-numeric columns (metadata)
                        original_data = st.session_state.current_data
                        numeric_cols_list = original_data.select_dtypes(include=[np.number]).columns.tolist()
                        non_numeric_cols_list = [col for col in original_data.columns if col not in numeric_cols_list]

                        # If metadata exists, add it back to reconstructed data
                        if non_numeric_cols_list:
                            X_recon_with_meta = X_reconstructed.copy()

                            for col in non_numeric_cols_list:
                                if col in original_data.columns:
                                    X_recon_with_meta[col] = original_data[col].values

                            # Reorder: metadata first, then numeric
                            cols_order = non_numeric_cols_list + numeric_cols_list
                            X_recon_final = X_recon_with_meta[cols_order]

                            # Update X_reconstructed with metadata
                            X_reconstructed = X_recon_final

                        # Download section
                        st.markdown("#### 📥 Download Reconstructed Data:")

                        # Create Excel file in memory (AFTER metadata preservation)
                        from io import BytesIO

                        buffer = BytesIO()
                        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                            X_reconstructed.to_excel(
                                writer,
                                sheet_name='Data',
                                index=True
                            )

                            # Add reconstruction info to second sheet
                            info_df = pd.DataFrame({
                                'Metric': [
                                    'Algorithm',
                                    'Missing Values (Before)',
                                    'Missing Values (After)',
                                    'Values Filled',
                                    'Fill Rate (%)',
                                    'Components Used',
                                    'Variance Explained (%)',
                                    'NIPALS Converged',
                                    'NIPALS Iterations',
                                    'Centering Applied',
                                    'Scaling Applied',
                                    'Reconstruction Date',
                                    'Mean of Filled',
                                    'Std Dev of Filled',
                                    'Min of Filled',
                                    'Max of Filled',
                                    'Metadata Columns Preserved'
                                ],
                                'Value': [
                                    'NIPALS (Nonlinear Iterative Partial Least Squares)',
                                    recon_info['n_missing_before'],
                                    recon_info['n_missing_after'],
                                    recon_info['n_filled'],
                                    f"{(recon_info['n_filled'] / recon_info['n_missing_before'] * 100) if recon_info['n_missing_before'] > 0 else 0:.2f}",
                                    n_components_recon,
                                    f"{recon_info_dict['total_variance_explained']:.2f}",
                                    'Yes' if recon_info_dict['converged'] else 'No',
                                    recon_info_dict['n_iterations'],
                                    'Yes' if recon_info_dict['center'] else 'No',
                                    'Yes' if recon_info_dict['scale'] else 'No',
                                    pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                                    f"{recon_info['filled_mean']:.4f}",
                                    f"{recon_info['filled_std']:.4f}",
                                    f"{recon_info['filled_min']:.4f}",
                                    f"{recon_info['filled_max']:.4f}",
                                    ', '.join(non_numeric_cols_list) if non_numeric_cols_list else 'None'
                                ]
                            })
                            info_df.to_excel(
                                writer,
                                sheet_name='Reconstruction Info',
                                index=False
                            )

                        buffer.seek(0)

                        st.download_button(
                            label="📥 Download as Excel (.xlsx)",
                            data=buffer.getvalue(),
                            file_name=f"data_reconstructed_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True,
                            help="Download reconstructed data as Excel file (WITH metadata columns)",
                            key="download_recon_button"
                        )

                        st.divider()

                        # === FIX #2: LOAD TO WORKSPACE BUTTON ===
                        st.markdown("#### 📂 Load Reconstructed Data to Workspace")
                        st.markdown("*Make reconstructed data available for other analyses*")

                        if st.button(
                            "📥 Load to Workspace",
                            key="load_reconstructed_workspace_tab1",
                            use_container_width=True,
                            type="primary"
                        ):
                            try:
                                if 'split_datasets' not in st.session_state:
                                    st.session_state.split_datasets = {}

                                # Create workspace entry
                                workspace_name = f"{st.session_state.get('dataset_name', 'data')}_reconstructed_{pd.Timestamp.now().strftime('%H%M%S')}"

                                st.session_state.split_datasets[workspace_name] = {
                                    'data': X_reconstructed,
                                    'type': 'Reconstructed',
                                    'parent': st.session_state.get('dataset_name', 'Original'),
                                    'n_samples': len(X_reconstructed),
                                    'creation_time': pd.Timestamp.now(),
                                    'description': f'Missing data reconstructed - {recon_info["n_filled"]} values filled'
                                }

                                st.success(f"✅ Data loaded to workspace as: **{workspace_name}**")
                                st.info("🔄 You can now use this dataset in other analyses")

                            except Exception as e:
                                st.error(f"❌ Failed to load: {str(e)}")
                                import traceback
                                with st.expander("🔍 Debug Info"):
                                    st.code(traceback.format_exc())

                        st.divider()

                        # === FIX #3: ENHANCED PREVIEW WITH METADATA INFO ===
                        st.markdown("#### 👁️ Data Preview")

                        # Show metadata info if it exists
                        if non_numeric_cols_list:
                            st.markdown(f"**Showing reconstructed data with {len(non_numeric_cols_list)} metadata column(s)**")
                            st.caption(f"Metadata: {', '.join(non_numeric_cols_list)}")
                        else:
                            st.markdown("**Showing reconstructed data**")

                        preview_option = st.radio(
                            "Select data to preview:",
                            ["Reconstructed Data", "Original Data (with NaN)", "Comparison"],
                            horizontal=True,
                            key="preview_option_missing_tab1"
                        )

                        if preview_option == "Reconstructed Data":
                            st.dataframe(X_reconstructed, use_container_width=True, height=300)
                            st.caption(f"Rows: {len(X_reconstructed)} | Columns: {len(X_reconstructed.columns)}")

                        elif preview_option == "Original Data (with NaN)":
                            st.dataframe(original_data, use_container_width=True, height=300)
                            st.caption(f"Rows: {len(original_data)} | Columns: {len(original_data.columns)}")

                        else:  # Comparison
                            st.markdown("**Numeric columns comparison (first 10 rows)**")
                            col_a, col_b = st.columns(2)

                            with col_a:
                                st.markdown("*Original (with NaN)*")
                                st.dataframe(original_data[numeric_cols_list].head(10), use_container_width=True)

                            with col_b:
                                st.markdown("*Reconstructed*")
                                st.dataframe(X_reconstructed[numeric_cols_list].head(10), use_container_width=True)

                        # Success message
                        st.success(
                            f"✅ Reconstruction complete!\n\n"
                            f"• {recon_info['n_filled']:,} missing values filled\n"
                            f"• Using {n_components_recon} PCA components\n"
                            f"• Metadata columns preserved: {len(non_numeric_cols_list)}\n"
                            f"• Ready for workspace or download"
                        )

                except Exception as e:
                    st.error(f"❌ Reconstruction failed: {str(e)}")
                    st.info("💡 Try with fewer components or check data format")

    # === VARIMAX ROTATION (OPTIONAL) ===
    if 'pca_results' in st.session_state:
        st.markdown("---")
        st.markdown("### 🔄 Varimax Rotation (Optional)")
        
        with st.expander("ℹ️ What is Varimax Rotation?"):
            st.markdown("""
            **Varimax rotation** simplifies the loading structure by rotating the 
            principal components to maximize the variance of squared loadings.
            
            **Benefits:**
            - Easier interpretation (each variable loads highly on fewer factors)
            - Clearer factor structure
            - Orthogonal rotation (factors remain uncorrelated)
            
            **When to use:**
            - When you need simpler interpretation
            - For factor analysis applications
            - When you have many variables
            """)
        
        apply_varimax = st.checkbox("Apply Varimax Rotation", value=False)
        
        if apply_varimax:
            pca_results = st.session_state['pca_results']
            loadings = pca_results['loadings']

            # Display scree plot to help choose number of factors
            st.markdown("### 📊 Scree Plot - Choose Components to Rotate")
            st.info("💡 All computed PCs shown below. Use the elbow point to decide how many components to rotate.")

            # Display scree plot of all n_components from PCA
            fig_scree = plot_scree(
                pca_results['explained_variance_ratio'],
                component_labels=pca_results['loadings'].columns.tolist()
            )
            st.plotly_chart(fig_scree, use_container_width=True, key="scree_before_varimax")

            # Select number of factors to rotate
            n_components = len(loadings.columns)
            if n_components > 2:
                n_factors = st.slider(
                    "Number of factors for rotation:",
                    min_value=2,
                    max_value=n_components,
                    value=min(3, n_components),
                    help="Typically rotate 2-5 factors for best interpretation"
                )
            else:
                st.info(f"⚠️ Only {n_components} components computed. Rotating all {n_components} components.")
                n_factors = n_components

            col1, col2 = st.columns(2)
            with col1:
                cumvar = pca_results['cumulative_variance'][n_factors-1] * 100
                st.metric("Cumulative Variance", f"{cumvar:.1f}%")
            with col2:
                st.metric("Factors to Rotate", n_factors)
            
            if st.button("🔄 Apply Varimax Rotation", type="secondary"):
                try:
                    with st.spinner("Applying Varimax rotation..."):
                        # Extract loadings and scores for rotation
                        loadings_subset = loadings.iloc[:, :n_factors]
                        scores_subset = pca_results['scores'].iloc[:, :n_factors]

                        # Prepare preprocessed data (CRITICAL!)
                        X_data = pca_results['original_data'].copy()
                        if pca_results['centering']:
                            X_data = X_data - pca_results['means']
                        if pca_results['scaling']:
                            X_data = X_data / pca_results['stds']

                        # Apply rotation with variance recalculation (MATLAB-aligned)
                        varimax_result = varimax_with_scores(
                            X=X_data,
                            loadings=loadings_subset,
                            scores=scores_subset
                        )

                        # Extract results
                        rotated_loadings = varimax_result['loadings_rotated']
                        rotated_scores = varimax_result['scores_rotated']
                        variance_rotated = varimax_result['variance_rotated']  # NEW!
                        variance_cumulative = varimax_result['variance_cumulative']  # NEW!
                        iterations = varimax_result['iterations']

                        # Rename to Factor instead of PC
                        factor_names = [f'Factor{i+1}' for i in range(n_factors)]
                        rotated_loadings.columns = factor_names
                        rotated_scores.columns = factor_names

                        # CRITICAL: Update eigenvalues and variance ratios from Varimax results
                        # These are DIFFERENT from original PCA after rotation!
                        eigenvalues_rotated = variance_rotated * varimax_result['vartot'] / 100

                        # Update session state with COMPLETE Varimax results
                        st.session_state['pca_results'].update({
                            'method': 'Varimax Rotation',
                            'scores': rotated_scores,
                            'loadings': rotated_loadings,
                            'eigenvalues': eigenvalues_rotated,  # NEW!
                            'explained_variance': eigenvalues_rotated,  # NEW!
                            'explained_variance_ratio': variance_rotated / 100,  # NEW! (as ratio 0-1)
                            'cumulative_variance': variance_cumulative / 100,  # NEW! (as ratio 0-1)
                            'varimax_applied': True,
                            'varimax_iterations': iterations,
                            'n_components': n_factors
                        })

                        st.success(f"✅ Varimax rotation completed in {iterations} iterations!")
                        st.info("♻️ Scores, loadings, AND variance recalculated. Check other tabs to see rotated results.")
                        st.rerun()

                except Exception as e:
                    st.error(f"❌ Varimax rotation failed: {str(e)}")
                    import traceback
                    st.error(f"Details: {traceback.format_exc()}")


# ============================================================================
# TAB 2: VARIANCE PLOTS (to be updated - from pca_OLD.py: copy contents)
# ============================================================================

def _show_variance_plots_tab():
    """Display advanced variance plots with multiple visualization options."""
    st.markdown("## 📊 Variance Plots")
    #st.markdown("*Equivalent to R PCA_variance_plot.r and PCA_cumulative_var_plot.r*")

    if 'pca_results' not in st.session_state:
        st.warning("⚠️ No PCA model computed. Please compute a model first.")
        return

    pca_results = st.session_state['pca_results']
    is_varimax = pca_results.get('varimax_applied', False)

    plot_type = st.selectbox(
        "Select variance plot:",
        ["📈 Scree Plot", "📊 Cumulative Variance", "🎯 Individual Variable Contribution"]
    )

    if plot_type == "📈 Scree Plot":
        title_suffix = " (Varimax Factors)" if is_varimax else " (Principal Components)"
        st.markdown(f"### 📈 Scree Plot{title_suffix}")

        # Use plot_scree from pca_utils.pca_plots
        component_labels = pca_results['loadings'].columns.tolist()
        fig = plot_scree(
            pca_results['explained_variance_ratio'],
            is_varimax=is_varimax,
            component_labels=component_labels
        )

        st.plotly_chart(fig, use_container_width=True, key="scree_plot")

        # === EXPLAINED VARIANCE TABLE ===
        st.markdown("---")
        st.markdown("#### 📋 Variance Explained by Component (Descending)")

        # Get variance data
        explained_var_pct = pca_results['explained_variance_ratio'] * 100
        cumulative_var_pct = pca_results['cumulative_variance'] * 100

        # Build DataFrame
        variance_table = pd.DataFrame({
            'Component': component_labels,
            'Explained Variance (%)': [f'{var:.2f}%' for var in explained_var_pct],
            'Cumulative Variance (%)': [f'{cum:.2f}%' for cum in cumulative_var_pct]
        })

        # Display table
        st.dataframe(
            variance_table,
            use_container_width=True,
            hide_index=True
        )

    elif plot_type == "📊 Cumulative Variance":
        title_suffix = " (Varimax)" if is_varimax else ""
        st.markdown(f"### 📊 Cumulative Variance Plot{title_suffix}")

        # Use plot_cumulative_variance from pca_utils.pca_plots
        component_labels = pca_results['loadings'].columns.tolist()
        fig = plot_cumulative_variance(
            pca_results['cumulative_variance'],
            is_varimax=is_varimax,
            component_labels=component_labels,
            reference_lines=[80, 95]
        )

        st.plotly_chart(fig, use_container_width=True, key="cumulative_plot")

    elif plot_type == "🎯 Individual Variable Contribution":
        comp_label = "Factor" if is_varimax else "PC"
        st.markdown(f"### 🎯 Variance of Each Variable Explained")
        st.markdown("*Fraction of each variable's variance explained by selected components*")
        st.markdown("*Equivalent to R chemometrics::pcaVarexpl()*")

        # Step 1: Select significant components
        st.markdown("#### Step 1: Select Number of Significant Components")
        st.info("📊 Use the Scree Plot above to identify the number of significant components")

        max_components = len(pca_results['loadings'].columns)
        n_significant = st.number_input(
            f"Number of significant {comp_label.lower()}s (from Scree Plot analysis):",
            min_value=1,
            max_value=max_components,
            value=2,
            help="Look at the Scree Plot to identify where the curve 'breaks' or levels off"
        )

        # Step 2: Calculate variance explained per variable
        st.markdown(f"#### Step 2: Explained Variance by Variable")

        # Add sorting checkbox
        sort_by_importance = st.checkbox(
            "Sort variables by importance (highest to lowest)",
            value=False,
            key="sort_variables_by_importance",
            help="When checked, variables are sorted by variance explained (descending). When unchecked, original variable order is preserved."
        )

        # Import from pca_utils.pca_statistics
        from pca_utils.pca_statistics import calculate_variable_variance_explained

        # Get the preprocessed data from PCA results
        X_preprocessed = pca_results.get('X_preprocessed')

        if X_preprocessed is None:
            st.error("❌ Preprocessed data not available. Please recompute PCA model.")
            return

        # Get scores and loadings
        scores = pca_results['scores']
        loadings = pca_results['loadings']

        # Calculate variance explained using EXACT R formula:
        # varexpl = 1 - Σ(residuals²) / Σ(X²)
        # where residuals = X - T[1:a] × P[1:a]ᵀ
        var_expl_df = calculate_variable_variance_explained(
            X_preprocessed=X_preprocessed,
            scores=scores,
            loadings=loadings,
            n_components=n_significant
        )

        # Sort by importance if requested
        if sort_by_importance:
            var_expl_df = var_expl_df.sort_values('Variance_Explained_Ratio', ascending=False).reset_index(drop=True)

        # Create bar plot
        fig = go.Figure()

        # Use ratio as-is (0-1.0 scale, matching R-CAT)
        fig.add_trace(go.Bar(
            x=var_expl_df['Variable'],
            y=var_expl_df['Variance_Explained_Ratio'],
            name='Variance Explained',
            marker=dict(
                color=var_expl_df['Variance_Explained_Ratio'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Variance Ratio")
            ),
            text=[f'{val:.3f}' for val in var_expl_df['Variance_Explained_Ratio']],
            textposition='outside'
        ))

        # Update title to show sort mode
        sort_mode_suffix = " - Sorted by Importance" if sort_by_importance else " - Original Order"

        fig.update_layout(
            title=f"Variance of Each Variable Explained by {n_significant} Significant {comp_label.lower()}s{sort_mode_suffix}",
            xaxis_title="Variables",
            yaxis_title="Variance Explained (0-1.0)",
            height=600,
            xaxis={'tickangle': 45},
            showlegend=False,
            yaxis=dict(range=[0, max(1.0, var_expl_df['Variance_Explained_Ratio'].max() * 1.1)])
        )

        st.plotly_chart(fig, use_container_width=True, key="variable_variance_plot")

        # Step 3: Detailed table
        st.markdown("#### Step 3: Detailed Results Table")

        st.dataframe(
            var_expl_df.round(2),
            use_container_width=True,
            hide_index=True
        )

        # Interpretation
        st.markdown("#### 📋 Interpretation")

        top_vars = var_expl_df.head(3)
        # Convert ratio (0-1) to percentage (0-100)
        avg_explained = var_expl_df['Variance_Explained_Ratio'].mean() * 100

        st.success(f"""
        **Key Findings (using {n_significant} significant {comp_label.lower()}s):**

        - **Average variance explained per variable**: {avg_explained:.1f}%
        - **Top 3 best-explained variables**:
          1. {top_vars.iloc[0]['Variable']}: {top_vars.iloc[0]['Variance_Explained_Ratio']*100:.1f}%
          2. {top_vars.iloc[1]['Variable']}: {top_vars.iloc[1]['Variance_Explained_Ratio']*100:.1f}%
          3. {top_vars.iloc[2]['Variable']}: {top_vars.iloc[2]['Variance_Explained_Ratio']*100:.1f}%

        - **Variables are better represented when their variance is > 80%**
        """)

        # Show variables with low representation
        low_var_threshold = 0.50  # 50% as ratio (0-1)
        low_explained = var_expl_df[var_expl_df['Variance_Explained_Ratio'] < low_var_threshold]

        if len(low_explained) > 0:
            st.warning(f"""
            ⚠️ **Variables with low representation** (< {low_var_threshold*100:.0f}%):
            {', '.join(map(str, low_explained['Variable'].tolist()))}

            
            """)


# ============================================================================
# TAB 3: LOADINGS PLOTS (to be updated - from pca_OLD.py: copy contents)
# ============================================================================

def _show_loadings_plots_tab():
    """Display loadings plots with multiple visualization options."""
    st.markdown("## 📈 Loading Plots")
    #st.markdown("*Equivalent to R PCA_plots_loadings.r*")

    if 'pca_results' not in st.session_state:
        st.warning("⚠️ Please compute PCA in **Model Computation** tab first")
        return

    pca_results = st.session_state['pca_results']
    loadings = pca_results['loadings']
    is_varimax = pca_results.get('varimax_applied', False)

    title_suffix = " (Varimax Factors)" if is_varimax else ""

    if is_varimax:
        st.info("🔄 Displaying Varimax-rotated factor loadings")

    # === PLOT TYPE SELECTION ===
    loading_plot_type = st.selectbox(
        "Select loading plot type:",
        ["📊 Loading Scatter Plot", "📈 Loading Line Plot", "🔝 Top Variables"]
    )

    # === PC/FACTOR SELECTION (for scatter and line plots) ===
    if loading_plot_type != "🔝 Top Variables":
        col1, col2 = st.columns(2)
        with col1:
            pc_x = st.selectbox("X-axis:", loadings.columns, index=0, key='load_x')
        with col2:
            pc_y_idx = 1 if len(loadings.columns) > 1 else 0
            pc_y = st.selectbox("Y-axis:", loadings.columns, index=pc_y_idx, key='load_y')

    # === LOADING SCATTER PLOT ===
    if loading_plot_type == "📊 Loading Scatter Plot":
        st.markdown(f"### 📊 Loading Scatter Plot{title_suffix}")

        # === NEW: ANTIDERIVATIVE OPTION ===
        use_antiderivative = st.checkbox(
            "📊 Show Antiderivative (for derivative-preprocessed data) by P. Oliveri et al. / Analytica Chimica Acta 1058 (2019) 9-17 DOI: https://doi.org/10.1016/j.aca.2018.10.055",
            value=False,
            key="scatter_antideriv_checkbox",
            help="Enable if data was preprocessed with row derivative transformation"
        )

        derivative_order = 1
        if use_antiderivative:
            derivative_order = st.radio(
                "Derivative order:",
                [1, 2],
                horizontal=True,
                key="scatter_deriv_order",
                help="Select the order of derivative applied to original data"
            )

        # === NEW: VARIABLE BLOCK COLORING OPTION ===
        st.markdown("###### Optional: Color by Contiguous Variable Blocks")

        use_block_colors = st.checkbox(
            "Color loading points by variable blocks",
            value=False,
            key="use_block_colors_loadings",
            help="Assign variables to blocks and color them differently in the loading plot"
        )

        variable_blocks = None
        if use_block_colors:
            st.info("💡 Define variable blocks by index ranges (1-based, inclusive). Example: Block 1: variables 1-4, Block 2: variables 5-8")

            # Initialize session state for block values
            if 'num_blocks_state_loadings' not in st.session_state:
                st.session_state.num_blocks_state_loadings = 3

            # Number of blocks (with change detection)
            num_blocks = st.number_input(
                "Number of blocks:",
                min_value=1,
                max_value=10,
                value=st.session_state.num_blocks_state_loadings,
                key="num_blocks_input_loadings"
            )

            # Reset values if num_blocks changed
            if num_blocks != st.session_state.num_blocks_state_loadings:
                st.session_state.num_blocks_state_loadings = num_blocks
                # Clear old values from session state
                for i in range(11):  # Clear up to 11 blocks (max possible + 1)
                    if f'block_start_loadings_{i}' in st.session_state:
                        del st.session_state[f'block_start_loadings_{i}']
                    if f'block_end_loadings_{i}' in st.session_state:
                        del st.session_state[f'block_end_loadings_{i}']
                    if f'block_name_loadings_{i}' in st.session_state:
                        del st.session_state[f'block_name_loadings_{i}']
                st.rerun()

            variable_blocks = {}
            cols = st.columns(min(num_blocks, 3))  # Max 3 columns per row

            # Calculate total variables
            n_vars = len(loadings)

            for i in range(num_blocks):
                with cols[i % 3]:
                    st.write(f"**Block {i+1}**")

                    # Block name
                    block_name = st.text_input(
                        f"Block {i+1} name:",
                        value=f"Block {i+1}",
                        key=f"block_name_loadings_{i}"
                    )

                    # Initialize defaults based on block number
                    default_start = 1 + i * (n_vars // num_blocks)
                    default_end = min((i + 1) * (n_vars // num_blocks), n_vars)

                    col_start, col_end = st.columns(2)
                    with col_start:
                        # Start index
                        start = st.number_input(
                            f"Start:",
                            min_value=1,
                            max_value=n_vars,
                            value=default_start,
                            key=f"block_start_loadings_{i}"
                        )
                    with col_end:
                        # End index - SAFE calculation
                        end = st.number_input(
                            f"End:",
                            min_value=start,  # Ensures min_value is always ≤ value
                            max_value=n_vars,
                            value=max(start, default_end),  # Ensure value ≥ min_value
                            key=f"block_end_loadings_{i}"
                        )

                    variable_blocks[block_name] = (start, end)

        if pc_x != pc_y:
            if use_antiderivative:
                # Plot antiderivative scatter
                fig = plot_loadings_antiderivative(
                    loadings,
                    pc_x,
                    pc_y,
                    pca_results['explained_variance_ratio'],
                    derivative_order=derivative_order,
                    is_varimax=is_varimax,
                    color_by_magnitude=is_varimax
                )
            else:
                # Plot normal scatter
                fig = plot_loadings(
                    loadings,
                    pc_x,
                    pc_y,
                    pca_results['explained_variance_ratio'],
                    is_varimax=is_varimax,
                    color_by_magnitude=is_varimax,
                    variable_blocks=variable_blocks if use_block_colors else None
                )

            st.plotly_chart(fig, use_container_width=True, key="loadings_scatter")

            if is_varimax:
                st.info("💡 In Varimax rotation, variables should load highly on few factors (simple structure)")

            # Display variance metrics
            pc_x_idx = list(loadings.columns).index(pc_x)
            pc_y_idx = list(loadings.columns).index(pc_y)
            var_x = pca_results['explained_variance_ratio'][pc_x_idx] * 100
            var_y = pca_results['explained_variance_ratio'][pc_y_idx] * 100
            var_total = var_x + var_y

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(f"{pc_x} Variance", f"{var_x:.1f}%")
            with col2:
                st.metric(f"{pc_y} Variance", f"{var_y:.1f}%")
            with col3:
                st.metric("Combined Variance", f"{var_total:.1f}%")
        else:
            st.warning("⚠️ X and Y axes must be different components")

    # === LOADING LINE PLOT ===
    elif loading_plot_type == "📈 Loading Line Plot":
        st.markdown(f"### 📈 Loading Line Plot{title_suffix}")

        selected_comps = st.multiselect(
            f"Select {'factors' if is_varimax else 'components'} to display:",
            loadings.columns.tolist(),
            default=loadings.columns[:min(3, len(loadings.columns))].tolist(),
            key="loading_line_components"
        )

        # === NEW: ANTIDERIVATIVE OPTION ===
        use_antiderivative = st.checkbox(
            "📊 Show Antiderivative (for derivative-preprocessed data)",
            value=False,
            help="Enable if data was preprocessed with row derivative transformation"
        )

        derivative_order = 1
        if use_antiderivative:
            derivative_order = st.radio(
                "Derivative order:",
                [1, 2],
                horizontal=True,
                help="Select the order of derivative applied to original data"
            )

        if selected_comps:
            if use_antiderivative:
                # Plot antiderivative
                fig = plot_loadings_line_antiderivative(
                    loadings,
                    selected_comps,
                    derivative_order=derivative_order,
                    is_varimax=is_varimax
                )
            else:
                # Plot normal loadings
                fig = plot_loadings_line(
                    loadings,
                    selected_comps,
                    is_varimax=is_varimax
                )

            st.plotly_chart(fig, use_container_width=True, key="loadings_line")
        else:
            st.warning("⚠️ Please select at least one component to display")

    # === TOP CONTRIBUTING VARIABLES ===
    elif loading_plot_type == "🔝 Top Variables":
        st.markdown(f"### 🔝 Top Contributing Variables per Component{title_suffix}")

        n_top = st.slider("Number of top variables to show:", 5, 20, 10)

        for col_name in loadings.columns:
            with st.expander(f"📊 {col_name} - Top {n_top} Variables"):
                load_values = loadings[col_name]

                # Get top positive and negative loadings
                abs_loadings = load_values.abs().sort_values(ascending=False)
                top_vars = abs_loadings.head(n_top)

                # Create DataFrame with signed loadings
                top_df = pd.DataFrame({
                    'Variable': top_vars.index,
                    'Loading': [load_values[var] for var in top_vars.index],
                    'Abs Loading': top_vars.values
                })

                # Display with color coding
                st.dataframe(
                    top_df.style.format({
                        'Loading': '{:.4f}',
                        'Abs Loading': '{:.4f}'
                    }).background_gradient(subset=['Loading'], cmap='RdBu_r', vmin=-1, vmax=1),
                    use_container_width=True,
                    hide_index=True
                )

                # Bar chart
                fig_bar = go.Figure()
                fig_bar.add_trace(go.Bar(
                    x=top_df['Loading'],
                    y=top_df['Variable'],
                    orientation='h',
                    marker_color=np.where(top_df['Loading'] > 0, '#2ca02c', '#d62728')
                ))

                fig_bar.update_layout(
                    title=f"Top {n_top} Variables for {col_name}",
                    xaxis_title="Loading",
                    yaxis_title="Variable",
                    yaxis=dict(autorange="reversed"),
                    height=400,
                    template='plotly_white'
                )

                st.plotly_chart(fig_bar, use_container_width=True, key=f"top_vars_{col_name}")

    # === VARIABLE CONTRIBUTIONS BY COMPONENT ===
    st.divider()
    st.markdown("### 📋 Variable Contributions by Component")

    # Get loadings
    loadings = pca_results['loadings']
    n_comp_display = min(4, loadings.shape[1])  # Show top 4 PCs

    # CREATE tabs for each PC
    comp_tabs = st.tabs([f"PC{i+1}" for i in range(n_comp_display)])

    for pc_idx, comp_tab in enumerate(comp_tabs):
        with comp_tab:
            st.markdown(f"### **PC{pc_idx+1}** - Variable Loadings")

            # Get loadings for this PC
            pc_loadings = loadings.iloc[:, pc_idx]

            # Explained variance
            exp_var = pca_results['explained_variance_ratio'][pc_idx] * 100
            st.metric(f"Explained Variance", f"{exp_var:.2f}%")

            # POSITIVE loadings (sorted descending)
            positive_mask = pc_loadings >= 0
            positive_loadings = pc_loadings[positive_mask].sort_values(ascending=False)

            # NEGATIVE loadings (sorted ascending, i.e., most negative first)
            negative_mask = pc_loadings < 0
            negative_loadings = pc_loadings[negative_mask].sort_values(ascending=True)

            # Display POSITIVE
            if len(positive_loadings) > 0:
                st.markdown("#### 🔼 Positive Loadings")

                pos_df = pd.DataFrame({
                    'Variable': positive_loadings.index,
                    'Loading': positive_loadings.values
                })

                # Color code: green for positive
                pos_df['Contribution %'] = (abs(pos_df['Loading']) / abs(pc_loadings).max() * 100).round(1)

                st.dataframe(
                    pos_df.style.format({'Loading': '{:.4f}', 'Contribution %': '{:.1f}%'}),
                    use_container_width=True,
                    hide_index=True
                )

            # Display NEGATIVE
            if len(negative_loadings) > 0:
                st.markdown("#### 🔽 Negative Loadings")

                neg_df = pd.DataFrame({
                    'Variable': negative_loadings.index,
                    'Loading': negative_loadings.values
                })

                # Color code: red for negative
                neg_df['Contribution %'] = (abs(neg_df['Loading']) / abs(pc_loadings).max() * 100).round(1)

                st.dataframe(
                    neg_df.style.format({'Loading': '{:.4f}', 'Contribution %': '{:.1f}%'}),
                    use_container_width=True,
                    hide_index=True
                )


# ============================================================================
# TAB 4: SCORE PLOTS (to be updated - from pca_OLD.py: copy contents)
# ============================================================================

def _show_score_plots_tab():
    """Display score plots (2D and 3D) with color-by options."""
    st.markdown("## 🎯 Score Plots")
    #st.markdown("*Equivalent to R PCA_plots_scores.r*")

    if 'pca_results' not in st.session_state:
        st.warning("⚠️ Please compute PCA in **Model Computation** tab first")
        return

    pca_results = st.session_state['pca_results']
    scores = pca_results['scores']
    is_varimax = pca_results.get('varimax_applied', False)

    # Get original data for color-by options
    data = st.session_state.get('current_data', None)

    title_suffix = " (Varimax)" if is_varimax else ""

    if is_varimax:
        st.info("🔄 Displaying Varimax-rotated factor scores")

    # === PLOT TYPE SELECTION ===
    plot_type = st.radio("Plot type:", ["2D Scatter", "3D Scatter"], horizontal=True)

    if plot_type == "2D Scatter":
        # === 2D SCORE PLOT ===
        col1, col2 = st.columns(2)
        with col1:
            pc_x = st.selectbox("X-axis:", scores.columns, index=0, key='score_x')
        with col2:
            pc_y_idx = 1 if len(scores.columns) > 1 else 0
            pc_y = st.selectbox("Y-axis:", scores.columns, index=pc_y_idx, key='score_y')

        # Get variance for axis labels
        var_x = pca_results['explained_variance_ratio'][scores.columns.get_loc(pc_x)] * 100
        var_y = pca_results['explained_variance_ratio'][scores.columns.get_loc(pc_y)] * 100

        # === COLOR AND DISPLAY OPTIONS ===
        col3, col4 = st.columns(2)
        with col3:
            # Show ALL columns plus Index option
            color_options = ["None"]
            if data is not None:
                color_options.extend(list(data.columns))
            color_options.append("Index")

            color_by = st.selectbox("Color points by:", color_options, key='color_by_2d')

        with col4:
            # Label options: None, Index, or any column
            label_options = ["Index", "None"]
            if data is not None:
                label_options.extend(list(data.columns))

            show_labels_from = st.selectbox("Show labels:", label_options, key='show_labels_2d')

        # Optional: show convex hulls for categorical color variables
        if color_by != "None":
            col_hull1, col_hull2 = st.columns(2)
            with col_hull1:
                show_convex_hull = st.checkbox("Show convex hulls (categorical)", value=False, key='show_hull_2d')
            with col_hull2:
                hull_opacity = st.slider(
                    "Hull opacity:",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.2,
                    step=0.05,
                    key='hull_opacity_2d'
                )
        else:
            show_convex_hull = False
            hull_opacity = 0.2

        # === TIER 2: Trajectory Lines Strategy ===
        st.markdown("---")
        st.markdown("**🎯 Sample Trajectory Lines**")
        st.caption("Connect samples in order to visualize temporal or sequential progression")

        col_traj1, col_traj2, col_traj3 = st.columns(3)
        with col_traj1:
            trajectory_strategy = st.selectbox(
                "Trajectory strategy:",
                ["None", "Sequential", "Categorical"],
                help="None: no lines | Sequential: one line through all samples | Categorical: separate lines per group"
            )

        with col_traj2:
            trajectory_width = st.slider(
                "Line width:",
                min_value=1,
                max_value=5,
                value=2,
                step=1,
                key='trajectory_width_2d'
            )

        with col_traj3:
            trajectory_opacity = st.slider(
                "Line opacity:",
                min_value=0.0,
                max_value=1.0,
                value=0.6,
                step=0.05,
                key='trajectory_opacity_2d'
            )

        # === Marker Size Control ===
        st.markdown("##### ⚫ Marker Size")
        marker_size = st.slider(
            "Point size:",
            min_value=1,
            max_value=20,
            value=5,
            step=1,
            key="marker_size_2d"
        )

        # === Trajectory Grouping ===
        # For Categorical trajectory, use the same variable as "Color points by"
        trajectory_groupby_column = None
        trajectory_metadata_for_coloring = None

        if trajectory_strategy == "Categorical":
            if color_by != "None" and color_by != "Index":
                # Use the same variable as point coloring
                if data is not None and color_by in data.columns:
                    try:
                        trajectory_groupby_column = data.loc[scores.index, color_by]
                        trajectory_metadata_for_coloring = trajectory_groupby_column
                        st.info(f"✅ Trajectory lines grouped by: **{color_by}** (same as point colors)")
                    except Exception as e:
                        st.warning(f"⚠️ Could not use '{color_by}' for trajectory: {str(e)}")
                        trajectory_strategy = "None"
                else:
                    st.warning("⚠️ Selected variable is not available in data")
                    trajectory_strategy = "None"
            else:
                st.warning("⚠️ Please select a categorical variable in 'Color points by' to use Categorical trajectories")
                trajectory_strategy = "None"

        # === TIER 3: Trajectory Coloring Options ===
        trajectory_color_variable = None
        trajectory_color_by_index = True  # Default for Sequential
        trajectory_color_vector = None

        if trajectory_strategy == "Sequential":
            # ========================================
            # SEQUENTIAL: Show gradient coloring
            # ========================================
            st.markdown("---")
            st.markdown("**🎨 Trajectory Line Coloring (Sequential)**")
            st.caption("Colors flow from BLUE (first point) → RED (last point)")

            # Get numeric columns for coloring options
            numeric_cols = []
            if data is not None:
                numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()

            # Color by selector
            color_trajectory_by = st.selectbox(
                "Color gradient by:",
                ["Sample Sequence (1→N)"] + numeric_cols,
                key="color_trajectory_by_2d",
                help="Sample Sequence: BLUE(1st)→RED(last) | Numeric Variable: BLUE(min)→RED(max)"
            )

            # Determine coloring strategy
            if color_trajectory_by == "Sample Sequence (1→N)":
                trajectory_color_by_index = True
                trajectory_color_variable = None
                st.info("💡 Gradient: Sample sequence (1→N) | BLUE=first point, RED=last point")
            else:
                trajectory_color_variable = color_trajectory_by
                trajectory_color_by_index = False
                st.info(f"💡 Gradient: {color_trajectory_by} variable | BLUE=minimum, RED=maximum")

        elif trajectory_strategy == "Categorical":
            # ========================================
            # CATEGORICAL: Show two MUTUALLY EXCLUSIVE options
            # ========================================
            st.markdown("---")
            st.markdown("**🎨 Trajectory Coloring (Categorical)**")

            # === CHECKBOX: Choose between Option 1 or Option 2 ===
            use_custom_color = st.checkbox(
                "Apply coloring to specific batch only",
                value=False,
                key="use_custom_trajectory_color_2d",
                help="Highlight one batch (BRIGHT gradient BLUE→RED) while dimming others (BACKGROUND gray)"
            )

            if use_custom_color and trajectory_groupby_column is not None:
                # ========================================
                # OPTION 2️⃣ ACTIVE: Independent batch highlighting
                # ========================================
                unique_categories = sorted(trajectory_groupby_column.dropna().unique())

                if len(unique_categories) > 0:
                    selected_batch_for_color = st.selectbox(
                        "Apply coloring to batch:",
                        unique_categories,
                        key="batch_for_color_2d"
                    )

                    st.info(f"✨ **{selected_batch_for_color}** will be BRIGHT (BLUE→RED gradient)\nOther batches will be DIM (gray background)")

                    # *** KEY: Use sequential_bright mode for gradient coloring ***
                    trajectory_color_vector = {
                        'category': selected_batch_for_color,
                        'variable': None,  # Sequential index, no numeric variable
                        'mode': 'sequential_bright',  # ← Use sequential gradient, NOT category colors
                        'values': None,
                        'min': None,
                        'max': None
                    }

                    # *** DISABLE Option 1 when Option 2 is active ***
                    trajectory_metadata_for_coloring = None  # No category metadata
                else:
                    trajectory_color_vector = None
                    trajectory_metadata_for_coloring = None

            else:
                # ========================================
                # OPTION 1️⃣ ACTIVE (DEFAULT): Color by category
                # ========================================
                st.markdown("**Option 1️⃣: Color by Category** (Default)")
                st.caption(f"Trajectory lines colored by category: **{color_by}**")
                st.info(f"✅ Each {color_by} category gets a distinct color")

                trajectory_color_vector = None  # No batch highlighting
                # Keep trajectory_metadata_for_coloring as set earlier (line 1513)

            # Default coloring: by index (for backward compatibility)
            trajectory_color_by_index = True
            trajectory_color_variable = None

        # === BLOCK COLORING BY SAMPLE ASSIGNMENT (CORRECT) ===
        st.markdown("##### Optional: Color by Sample Block")
        st.caption("Assign samples to blocks and color them accordingly")

        use_score_block_colors = st.checkbox(
            "Color score points by sample block",
            value=False,
            help="Assign samples to blocks and color each sample by its block.",
            key="use_score_block_colors"
        )

        score_sample_blocks = None
        if use_score_block_colors:
            st.info("📌 Assign samples to blocks by sample index range. Each sample will be colored by its assigned block.")

            # Get number of samples
            n_samples = len(scores)
            st.caption(f"📊 Dataset has {n_samples} samples")

            # Initialize session state
            if 'score_num_blocks_state' not in st.session_state:
                st.session_state.score_num_blocks_state = 2

            num_score_blocks = st.number_input(
                "Number of sample blocks:",
                min_value=1,
                max_value=5,
                value=st.session_state.score_num_blocks_state,
                key="score_num_blocks_input"
            )

            # Reset on change
            if num_score_blocks != st.session_state.score_num_blocks_state:
                st.session_state.score_num_blocks_state = num_score_blocks
                for i in range(6):
                    if f'score_block_start_{i}' in st.session_state:
                        del st.session_state[f'score_block_start_{i}']
                    if f'score_block_end_{i}' in st.session_state:
                        del st.session_state[f'score_block_end_{i}']
                    if f'score_block_name_{i}' in st.session_state:
                        del st.session_state[f'score_block_name_{i}']
                st.rerun()

            score_sample_blocks = {}
            cols = st.columns(num_score_blocks)

            for i in range(num_score_blocks):
                with cols[i]:
                    st.write(f"**Block {i+1}**")

                    # Block name
                    block_name = st.text_input(
                        f"Name:",
                        value=f"Block {i+1}",
                        key=f"score_block_name_{i}"
                    )

                    # Calculate default ranges based on SAMPLES, not variables
                    default_start = 1 + i * (n_samples // num_score_blocks)
                    default_end = min((i + 1) * (n_samples // num_score_blocks), n_samples)

                    # Safety checks
                    if default_start > n_samples:
                        default_start = n_samples
                    if default_end > n_samples:
                        default_end = n_samples
                    if default_end < default_start:
                        default_end = default_start

                    # Start sample index
                    start = st.number_input(
                        f"Start sample:",
                        min_value=1,
                        max_value=n_samples,
                        value=default_start,
                        key=f"score_block_start_{i}"
                    )

                    # End sample index
                    end = st.number_input(
                        f"End sample:",
                        min_value=start,
                        max_value=n_samples,
                        value=max(start, default_end),
                        key=f"score_block_end_{i}"
                    )

                    score_sample_blocks[block_name] = (start, end)

        # === ASSIGN SAMPLES TO BLOCKS (SIMPLE INDEX-BASED) ===
        if use_score_block_colors and score_sample_blocks is not None:
            # Create assignment for each sample (1-based indexing)
            sample_block_assignment = {}

            for sample_idx in range(1, len(scores) + 1):
                # Find which block this sample belongs to
                assigned_block = None
                for block_name, (start, end) in score_sample_blocks.items():
                    if start <= sample_idx <= end:
                        assigned_block = block_name
                        break

                if assigned_block is None:
                    assigned_block = "Unassigned"

                sample_block_assignment[sample_idx] = assigned_block

            # Create color map
            block_names = list(score_sample_blocks.keys())
            if COLORS_AVAILABLE:
                score_block_color_map = create_categorical_color_map(block_names)

            # Create Series aligned with scores index
            color_data = pd.Series(
                [sample_block_assignment.get(i, "Unassigned") for i in range(1, len(scores) + 1)],
                index=scores.index,
                name="Sample Block"
            )
            color_by = "Sample Block"

            # === SHOW RESULTS TABLE ===
            st.markdown("---")
            st.subheader("📊 Sample Block Assignment")
            st.caption("Each sample is assigned to a block by its index")

            results_df = pd.DataFrame({
                'Sample': range(1, len(scores) + 1),
                'Block': color_data.values
            })
            st.dataframe(results_df, hide_index=True, use_container_width=True)
        else:
            # Prepare color data and text labels (ORIGINAL LOGIC)
            color_data = None
            if color_by != "None":
                if color_by == "Index":
                    color_data = pd.Series(range(len(scores)), index=scores.index, name="Row Index")
                elif data is not None:
                    try:
                        color_data = data.loc[scores.index, color_by]
                    except:
                        st.warning(f"⚠️ Could not align color variable '{color_by}' with scores")
                        color_data = None

        # Prepare text labels - show sample name + color variable value
        text_param = None
        if show_labels_from != "None":
            # Start with sample names/indices
            if show_labels_from == "Index":
                base_labels = [str(idx) for idx in scores.index]
            elif data is not None:
                try:
                    # Ensure column exists and has matching index
                    if show_labels_from in data.columns:
                        # Get values, handle NaN
                        col_values = data[show_labels_from].reindex(scores.index)
                        base_labels = [str(val) if pd.notna(val) else str(idx)
                                      for idx, val in zip(scores.index, col_values)]
                    else:
                        # Column not found - fallback to index
                        st.warning(f"⚠️ Column '{show_labels_from}' not found in data")
                        base_labels = [str(idx) for idx in scores.index]
                except Exception as e:
                    # Debug: show what went wrong
                    st.warning(f"⚠️ Could not read labels from '{show_labels_from}': {str(e)}")
                    base_labels = [str(idx) for idx in scores.index]
            else:
                base_labels = [str(idx) for idx in scores.index]

            # Show only the label values, no color_by info
            text_param = base_labels

        # Debug output for label verification
        if text_param and len(text_param) > 0:
            st.caption(f"📋 Sample labels: {text_param[0]} (+ {len(text_param)-1} more)")

        # Calculate total variance
        var_total = var_x + var_y

        # Create plot using px.scatter with color logic from pca_OLD.py
        color_discrete_map = None  # Initialize

        # === SPECIAL HANDLING FOR SAMPLE BLOCK COLORING ===
        if use_score_block_colors and color_by == "Sample Block":
            # Use the pre-calculated block color map
            color_discrete_map = score_block_color_map

            fig = px.scatter(
                x=scores[pc_x],
                y=scores[pc_y],
                color=color_data,
                color_discrete_map=color_discrete_map,
                text=text_param,
                title=f"Scores: {pc_x} vs {pc_y} (colored by Sample Block){title_suffix}<br>Total Explained Variance: {var_total:.1f}%",
                labels={
                    'x': f'{pc_x} ({var_x:.1f}%)',
                    'y': f'{pc_y} ({var_y:.1f}%)',
                    'color': 'Sample Block'
                }
            )
            # CRITICAL: Enable legend for block coloring
            fig.update_traces(showlegend=True)

        elif color_by == "None":
            fig = px.scatter(
                x=scores[pc_x], y=scores[pc_y], text=text_param,
                title=f"Scores: {pc_x} vs {pc_y}{title_suffix}<br>Total Explained Variance: {var_total:.1f}%",
                labels={'x': f'{pc_x} ({var_x:.1f}%)', 'y': f'{pc_y} ({var_y:.1f}%)'}
            )
        else:
            # Check if numeric and quantitative
            if (color_by != "None" and color_by != "Index" and
                hasattr(color_data, 'dtype') and pd.api.types.is_numeric_dtype(color_data)):

                if COLORS_AVAILABLE and is_quantitative_variable(color_data):
                    # Quantitative: use blue-to-red color scale
                    # Format: [(0, 'blue'), (0.5, 'purple'), (1, 'red')]
                    color_palette = [(0.0, 'rgb(0, 0, 255)'), (0.5, 'rgb(128, 0, 128)'), (1.0, 'rgb(255, 0, 0)')]

                    fig = px.scatter(
                        x=scores[pc_x], y=scores[pc_y], color=color_data, text=text_param,
                        title=f"Scores: {pc_x} vs {pc_y} (colored by {color_by}){title_suffix}<br>Total Explained Variance: {var_total:.1f}%",
                        labels={'x': f'{pc_x} ({var_x:.1f}%)', 'y': f'{pc_y} ({var_y:.1f}%)', 'color': color_by},
                        color_continuous_scale=color_palette
                    )
                else:
                    # Discrete numeric: use categorical color map
                    color_data_series = pd.Series(color_data)
                    unique_values = color_data_series.dropna().unique()
                    if COLORS_AVAILABLE:
                        color_discrete_map = create_categorical_color_map(unique_values)

                    fig = px.scatter(
                        x=scores[pc_x], y=scores[pc_y], color=color_data, text=text_param,
                        title=f"Scores: {pc_x} vs {pc_y} (colored by {color_by}){title_suffix}<br>Total Explained Variance: {var_total:.1f}%",
                        labels={'x': f'{pc_x} ({var_x:.1f}%)', 'y': f'{pc_y} ({var_y:.1f}%)', 'color': color_by},
                        color_discrete_map=color_discrete_map
                    )
            else:
                # Categorical (default for Index and string data)
                color_data_series = pd.Series(color_data)
                unique_values = color_data_series.dropna().unique()
                if COLORS_AVAILABLE:
                    color_discrete_map = create_categorical_color_map(unique_values)

                fig = px.scatter(
                    x=scores[pc_x], y=scores[pc_y], color=color_data, text=text_param,
                    title=f"Scores: {pc_x} vs {pc_y} (colored by {color_by}){title_suffix}<br>Total Explained Variance: {var_total:.1f}%",
                    labels={'x': f'{pc_x} ({var_x:.1f}%)', 'y': f'{pc_y} ({var_y:.1f}%)', 'color': color_by},
                    color_discrete_map=color_discrete_map
                )

        # Add convex hulls (only for categorical variables)
        if (color_by != "None" and show_convex_hull and
            not (hasattr(color_data, 'dtype') and pd.api.types.is_numeric_dtype(color_data) and
                 COLORS_AVAILABLE and is_quantitative_variable(color_data))):
            try:
                if PLOTS_AVAILABLE:
                    fig = add_convex_hulls(fig, scores, pc_x, pc_y, color_data, color_discrete_map, hull_opacity)
            except Exception as e:
                st.warning(f"⚠️ Could not add convex hulls: {str(e)}")

        # === Apply marker size BEFORE adding trajectory (to exclude arrow markers) ===
        # This ensures only scatter points are resized, not arrow markers
        # Update scatter traces created by px.scatter (these are the main data points)
        # They have mode='markers' or 'markers+text' and come BEFORE trajectory/hull traces
        for i, trace in enumerate(fig.data):
            # Main scatter points are created by px.scatter and appear first in fig.data
            # They have markers and are NOT lines (convex hull has mode='lines')
            # They are NOT arrows (arrows are added later in trajectory)
            if hasattr(trace, 'mode') and trace.mode in ['markers', 'markers+text']:
                # This is a main scatter point trace
                trace.marker.size = marker_size
                # Ensure scatter points are visible (not hidden)
                trace.visible = True

        # Add sample trajectory lines with gradient coloring
        # IMPORTANT: Lines use their own coloring (sequential index or variable)
        # They are INDEPENDENT from point colors (color_discrete_map NOT passed)
        if trajectory_strategy != "None" and PLOTS_AVAILABLE:
            try:
                fig = add_sample_trajectory_lines(
                    fig=fig,
                    scores=scores,
                    pc_x=pc_x,
                    pc_y=pc_y,
                    line_strategy=trajectory_strategy.lower(),
                    groupby_column=trajectory_groupby_column,
                    # color_discrete_map NOT passed - lines are independent!
                    line_width=trajectory_width,
                    line_opacity=trajectory_opacity,
                    color_by_index=trajectory_color_by_index,
                    color_variable=trajectory_color_variable,
                    original_data=data,
                    trajectory_color_vector=trajectory_color_vector,
                    # === NEW PARAMETERS FOR CATEGORICAL TRAJECTORY COLORING ===
                    metadata_column=trajectory_metadata_for_coloring,
                    use_category_colors=(trajectory_strategy == "Categorical" and trajectory_metadata_for_coloring is not None),
                    show_trajectory_arrow=True
                )
            except Exception as e:
                st.warning(f"⚠️ Could not add trajectory lines: {str(e)}")

        # [NEW] Selective batch highlighting: Dim non-selected batches
        # If batch-specific coloring is active, make selected batch PROMINENT and dim others
        if trajectory_color_vector and trajectory_groupby_column is not None:
            selected_batch = trajectory_color_vector['category']

            # Update opacity for each batch's scatter points
            # Get unique batches from the groupby column
            unique_batches = trajectory_groupby_column.dropna().unique()

            for batch_name in unique_batches:
                if batch_name != selected_batch:
                    # Dim non-selected batches to 0.15 (very faint background)
                    # This makes them almost invisible, highlighting the selected batch
                    try:
                        # Update traces that match this batch name
                        # Note: Plotly scatter points have the category as their name
                        fig.update_traces(
                            opacity=0.15,
                            selector=dict(name=str(batch_name))
                        )
                    except:
                        pass  # Skip if selector doesn't match

        # === FINAL CHECK: Ensure scatter points remain visible ===
        # After all trajectory operations, verify scatter points are not hidden
        for trace in fig.data:
            # Identify main scatter points (mode='markers' or 'markers+text')
            if hasattr(trace, 'mode') and trace.mode in ['markers', 'markers+text']:
                # Scatter points must be visible unless explicitly dimmed by batch highlighting
                if trace.visible is None or trace.visible == False:
                    trace.visible = True

        # Update text position
        if show_labels_from != "None":
            fig.update_traces(textposition="top center")

        # EQUAL AXES SCALE (from pca_OLD)
        x_range = [scores[pc_x].min(), scores[pc_x].max()]
        y_range = [scores[pc_y].min(), scores[pc_y].max()]
        max_abs_range = max(abs(min(x_range + y_range)), abs(max(x_range + y_range)))
        axis_range = [-max_abs_range * 1.1, max_abs_range * 1.1]

        # ADD ZERO LINES (gray dashed)
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.7)
        fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.7)

        # UPDATE LAYOUT with equal scale and lasso selection - COMPACT LEGEND NO BORDER
        fig.update_layout(
            # TITLE: centered and compact
            title=dict(
                text=fig.layout.title.text,  # Keep existing title
                x=0.5,
                xanchor='center',
                y=0.98,
                yanchor='top',
                font=dict(size=13)
            ),

            # DIMENSIONS: compact height, full width
            height=550,
            width=None,  # Let it use container width

            # MARGINS: reduce whitespace
            margin=dict(l=60, r=120, t=70, b=60),

            # TEMPLATE and INTERACTION
            template='plotly_white',
            dragmode='zoom',

            # EQUAL ASPECT RATIO
            xaxis=dict(
                range=axis_range,
                scaleanchor="y",
                scaleratio=1,
                constrain="domain",
                title=dict(font=dict(size=11))
            ),
            yaxis=dict(
                range=axis_range,
                scaleanchor="x",
                scaleratio=1,
                constrain="domain",
                title=dict(font=dict(size=11))
            ),

            # LEGEND: NO BORDER, closer to edges (ULTRA COMPACT!)
            legend=dict(
                x=0.99,           # Very close to right edge
                y=0.99,           # Very close to top edge
                xanchor='right',
                yanchor='top',
                bgcolor='rgba(255, 255, 255, 0.9)',  # Slightly opaque white
                borderwidth=0,     # NO BORDER (removes border completely)
                font=dict(size=10)
            ),

            # HOVER: compact mode
            hovermode='closest'
        )

        # === SPECIAL LEGEND CONFIGURATION FOR BLOCK COLORING ===
        if use_score_block_colors and color_by == "Sample Block":
            fig.update_layout(
                showlegend=True,
                legend=dict(
                    x=0.99,
                    y=0.99,
                    xanchor='right',
                    yanchor='top',
                    bgcolor='rgba(255, 255, 255, 0.9)',
                    borderwidth=1,
                    bordercolor='rgba(0, 0, 0, 0.2)',
                    font=dict(size=11)
                ),
                margin=dict(l=60, r=120, t=70, b=60)
            )

        # Display plot with selection enabled
        selection = st.plotly_chart(fig, use_container_width=True, key="scores_2d", on_select="rerun", selection_mode=["points", "lasso"])

        # Display metrics
        st.markdown("#### Variance Summary")
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        with metric_col1:
            st.metric(f"{pc_x} Variance", f"{var_x:.1f}%")
        with metric_col2:
            st.metric(f"{pc_y} Variance", f"{var_y:.1f}%")
        with metric_col3:
            st.metric("Combined Variance", f"{var_x + var_y:.1f}%")

        # === LASSO SELECTION ===
        st.markdown("---")
        st.markdown("### 📍 Lasso Selection")
        st.info("💡 Use the lasso tool in the plot above to select samples and compare their characteristics")

        # Extract selected points from selection
        selected_indices = []
        if selection and selection.selection and "point_indices" in selection.selection:
            point_indices = list(selection.selection["point_indices"])  # Convert to list
            if point_indices:
                # Store in session for persistence across reruns
                st.session_state.lasso_point_indices = point_indices
                # point_indices are POSITIONAL indices from plotly (0, 1, 2, ...)
                # Convert to actual DataFrame index values using list comprehension
                selected_indices = [scores.index[i] for i in point_indices]
        elif 'lasso_point_indices' in st.session_state:
            # Restore from session if plot rerun
            point_indices = st.session_state.lasso_point_indices
            selected_indices = [scores.index[i] for i in point_indices]

        if selected_indices and len(selected_indices) > 0:
            st.success(f"✅ Selected {len(selected_indices)} samples")

            # Display selected sample IDs
            with st.expander("📋 Selected Sample IDs"):
                st.write(selected_indices)

            # Reset button
            col1, col2 = st.columns(2)

            with col1:
                if st.button("🔄 Reset Lasso Selection", use_container_width=True):
                    if 'plotly_selection' in st.session_state:
                        del st.session_state.plotly_selection
                    if 'lasso_point_indices' in st.session_state:
                        del st.session_state.lasso_point_indices
                    st.rerun()

            with col2:
                st.info(f"📊 Selected: {len(selected_indices)} samples")

            st.divider()

            # === LASSO SELECTION ANALYSIS ===
            st.markdown("### 🎯 Lasso Selection Analysis")
            st.markdown(f"**Selected samples:** {len(selected_indices)}")

            # Ensure selected_indices are valid in the data DataFrame
            if data is not None:
                valid_indices = [idx for idx in selected_indices if idx in data.index]

                if len(valid_indices) != len(selected_indices):
                    st.warning(f"⚠️ {len(selected_indices) - len(valid_indices)} selected indices not found in data. Using {len(valid_indices)} valid indices.")
                    selected_indices = valid_indices

                if len(selected_indices) == 0:
                    st.error("❌ No valid indices found in the data DataFrame")
                else:
                    # Get selected and non-selected data
                    sel_data = data.loc[selected_indices]
                    not_sel_data = data.drop(selected_indices)

                    # === NORMAL FLOW: COMPARISON AND STATISTICS ===
                    # Calculate statistics for numeric columns only
                    numeric_cols = data.select_dtypes(include=[np.number]).columns

                    if len(numeric_cols) > 0:
                        comp_df = pd.DataFrame({
                            'Variable': numeric_cols,
                            'Selected Mean': sel_data[numeric_cols].mean(),
                            'Not Selected Mean': not_sel_data[numeric_cols].mean(),
                            'Difference': sel_data[numeric_cols].mean() - not_sel_data[numeric_cols].mean()
                        }).sort_values('Difference', key=abs, ascending=False)

                        st.markdown("#### 📊 Variable Comparison: Selected vs Not Selected")
                        st.dataframe(
                            comp_df.style.format({
                                'Selected Mean': '{:.4f}',
                                'Not Selected Mean': '{:.4f}',
                                'Difference': '{:.4f}'
                            }).background_gradient(subset=['Difference'], cmap='RdBu_r'),
                            use_container_width=True
                        )

                        # Show top 5 discriminating variables
                        st.markdown("#### 🎯 Top 5 Differentiating Variables")
                        top_vars = comp_df.head(5)
                        for _, row in top_vars.iterrows():
                            diff_pct = (abs(row['Difference']) / abs(row['Not Selected Mean']) * 100) if row['Not Selected Mean'] != 0 else 0
                            st.write(f"**{row['Variable']}**: Δ = {row['Difference']:.4f} ({diff_pct:.1f}% change)")

                        # Detailed selected samples table
                        st.markdown("#### 📋 Detailed Selected Samples Data")

                        # Create detailed table with sample ID and all variables
                        sel_samples_table = sel_data[numeric_cols].copy()
                        sel_samples_table.insert(0, 'Sample ID', selected_indices)

                        # Display with highlighting
                        st.dataframe(
                            sel_samples_table.style.highlight_max(color='lightgreen', axis=0, subset=numeric_cols.tolist())
                                                   .highlight_min(color='lightcoral', axis=0, subset=numeric_cols.tolist())
                                                   .format({col: '{:.4f}' for col in numeric_cols}),
                            use_container_width=True
                        )

                        # Summary statistics for selected samples
                        st.markdown("#### 📈 Summary Statistics (Selected Samples)")
                        summary_stats = pd.DataFrame({
                            'Statistic': ['Count', 'Mean', 'Std Dev', 'Min', 'Max'],
                            'Value': [
                                len(selected_indices),
                                sel_data[numeric_cols].mean().mean(),
                                sel_data[numeric_cols].std().mean(),
                                sel_data[numeric_cols].min().min(),
                                sel_data[numeric_cols].max().max()
                            ]
                        })
                        st.dataframe(
                            summary_stats.style.format({'Value': '{:.4f}'}, subset=pd.IndexSlice[1:, 'Value']),
                            use_container_width=True,
                            hide_index=True
                        )

                        st.divider()

                        # === DOWNLOAD SECTION ===
                        st.markdown("### 💾 Download Dataset")
                        st.markdown("Choose which dataset to download:")

                        download_choice = st.radio(
                            "Select download option:",
                            options=[
                                "⬇️ Download selected samples only",
                                "⬇️ Download excluded samples (keep others)",
                                "⬇️ Download comparison CSV"
                            ],
                            key="lasso_download_choice",
                            help="Choose which data to export as CSV"
                        )

                        # Download buttons based on selection
                        if download_choice == "⬇️ Download selected samples only":
                            import io
                            buffer_sel = io.BytesIO()
                            sel_data.to_excel(buffer_sel, sheet_name="Selected", index=True, engine='openpyxl')
                            buffer_sel.seek(0)
                            xlsx_sel = buffer_sel.getvalue()
                            st.download_button(
                                label="📥 Download Selected Samples (XLSX)",
                                data=xlsx_sel,
                                file_name=f"PCA_Selected_{len(selected_indices)}_samples.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                use_container_width=True
                            )
                            st.info(f"📊 File will contain {len(selected_indices)} samples")

                        elif download_choice == "⬇️ Download excluded samples (keep others)":
                            import io
                            buffer_excl = io.BytesIO()
                            not_sel_data.to_excel(buffer_excl, sheet_name="Excluded", index=True, engine='openpyxl')
                            buffer_excl.seek(0)
                            xlsx_excl = buffer_excl.getvalue()
                            st.download_button(
                                label="📥 Download Excluded Samples (XLSX)",
                                data=xlsx_excl,
                                file_name=f"PCA_Excluded_{len(selected_indices)}_removed.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                use_container_width=True
                            )
                            st.info(f"📊 File will contain {len(not_sel_data)} samples (original: {len(data)})")

                        elif download_choice == "⬇️ Download comparison CSV":
                            import io
                            buffer_comp = io.BytesIO()
                            comp_df.to_excel(buffer_comp, sheet_name="Comparison", index=False, engine='openpyxl')
                            buffer_comp.seek(0)
                            xlsx_comp = buffer_comp.getvalue()
                            st.download_button(
                                label="📥 Download Comparison Table (XLSX)",
                                data=xlsx_comp,
                                file_name=f"PCA_Comparison_{len(selected_indices)}_vs_{len(not_sel_data)}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                use_container_width=True
                            )
                            st.info(f"📊 File will contain comparison statistics for {len(numeric_cols)} variables")

                        # Tip for resetting selection after download
                        st.divider()
                        st.info("💡 Tip: Click '🔄 Reset Lasso Selection' above to clear and make a new selection")

                    else:
                        st.warning("⚠️ No numeric columns available for comparison")
        else:
            st.info("🔵 No points selected. Use the lasso tool to select samples.")

    else:
        # === 3D SCORE PLOT ===
        if len(scores.columns) < 3:
            st.warning("⚠️ Need at least 3 components for 3D plot")
            return

        col1, col2, col3 = st.columns(3)
        with col1:
            pc_x = st.selectbox("X-axis:", scores.columns, index=0, key='score_x_3d')
        with col2:
            pc_y = st.selectbox("Y-axis:", scores.columns, index=1, key='score_y_3d')
        with col3:
            pc_z = st.selectbox("Z-axis:", scores.columns, index=2, key='score_z_3d')

        # === POINT SIZE CONTROL ===
        point_size_3d = st.slider(
            "3D Point Size:",
            min_value=2,
            max_value=15,
            value=6,
            step=1,
            key="point_size_3d"
        )

        # === COLOR AND DISPLAY OPTIONS FOR 3D ===
        col4, col5 = st.columns(2)
        with col4:
            # Show ALL columns plus Index option
            color_options_3d = ["None"]
            if data is not None:
                color_options_3d.extend(list(data.columns))
            color_options_3d.append("Index")

            color_by_3d = st.selectbox("Color points by:", color_options_3d, key='color_by_3d')

        with col5:
            # Label options: None, Index, or any column
            label_options_3d = ["Index", "None"]
            if data is not None:
                label_options_3d.extend(list(data.columns))

            show_labels_from_3d = st.selectbox("Show labels:", label_options_3d, key='show_labels_3d')

        # Prepare color data and text labels for 3D
        color_data_3d = None
        if color_by_3d != "None":
            if color_by_3d == "Index":
                color_data_3d = pd.Series(range(len(scores)), index=scores.index, name="Row Index")
            elif data is not None:
                try:
                    color_data_3d = data.loc[scores.index, color_by_3d]
                except:
                    st.warning(f"⚠️ Could not align color variable '{color_by_3d}' with scores")
                    color_data_3d = None

        # Prepare text labels - show sample name + color variable value
        text_param_3d = None
        if show_labels_from_3d != "None":
            # Start with sample names/indices
            if show_labels_from_3d == "Index":
                base_labels_3d = [str(idx) for idx in scores.index]
            elif data is not None:
                try:
                    # Ensure column exists and has matching index
                    if show_labels_from_3d in data.columns:
                        # Get values, handle NaN
                        col_values_3d = data[show_labels_from_3d].reindex(scores.index)
                        base_labels_3d = [str(val) if pd.notna(val) else str(idx)
                                         for idx, val in zip(scores.index, col_values_3d)]
                    else:
                        # Column not found - fallback to index
                        st.warning(f"⚠️ Column '{show_labels_from_3d}' not found in data")
                        base_labels_3d = [str(idx) for idx in scores.index]
                except Exception as e:
                    # Debug: show what went wrong
                    st.warning(f"⚠️ Could not read labels from '{show_labels_from_3d}': {str(e)}")
                    base_labels_3d = [str(idx) for idx in scores.index]
            else:
                base_labels_3d = [str(idx) for idx in scores.index]

            # === FIX: Show ONLY the selected labels, never the color variable ===
            # The color variable affects point COLOR, not the LABEL text
            text_param_3d = base_labels_3d

        # Debug output for label verification
        if text_param_3d and len(text_param_3d) > 0:
            st.caption(f"📋 3D Sample labels: {text_param_3d[0]} (+ {len(text_param_3d)-1} more)")

        # Get variance for axis labels
        var_x = pca_results['explained_variance_ratio'][scores.columns.get_loc(pc_x)] * 100
        var_y = pca_results['explained_variance_ratio'][scores.columns.get_loc(pc_y)] * 100
        var_z = pca_results['explained_variance_ratio'][scores.columns.get_loc(pc_z)] * 100
        var_total_3d = var_x + var_y + var_z

        # Create plot using px.scatter_3d with color logic from pca_OLD.py
        color_discrete_map_3d = None  # Initialize

        if color_by_3d == "None":
            fig_3d = px.scatter_3d(
                x=scores[pc_x], y=scores[pc_y], z=scores[pc_z], text=text_param_3d,
                title=f"3D Scores: {pc_x}, {pc_y}, {pc_z}{title_suffix}<br>Total Explained Variance: {var_total_3d:.1f}%",
                labels={'x': f'{pc_x} ({var_x:.1f}%)', 'y': f'{pc_y} ({var_y:.1f}%)', 'z': f'{pc_z} ({var_z:.1f}%)'}
            )
        else:
            # Check if numeric and quantitative
            if (color_by_3d != "None" and color_by_3d != "Index" and
                hasattr(color_data_3d, 'dtype') and pd.api.types.is_numeric_dtype(color_data_3d)):

                if COLORS_AVAILABLE and is_quantitative_variable(color_data_3d):
                    # Quantitative: use blue-to-red color scale
                    # Format: [(0, 'blue'), (0.5, 'purple'), (1, 'red')]
                    color_palette = [(0.0, 'rgb(0, 0, 255)'), (0.5, 'rgb(128, 0, 128)'), (1.0, 'rgb(255, 0, 0)')]

                    fig_3d = px.scatter_3d(
                        x=scores[pc_x], y=scores[pc_y], z=scores[pc_z],
                        color=color_data_3d, text=text_param_3d,
                        title=f"3D Scores: {pc_x}, {pc_y}, {pc_z} (colored by {color_by_3d}){title_suffix}<br>Total Explained Variance: {var_total_3d:.1f}%",
                        labels={'x': f'{pc_x} ({var_x:.1f}%)', 'y': f'{pc_y} ({var_y:.1f}%)', 'z': f'{pc_z} ({var_z:.1f}%)', 'color': color_by_3d},
                        color_continuous_scale=color_palette
                    )
                else:
                    # Discrete numeric: use categorical color map
                    color_data_series_3d = pd.Series(color_data_3d)
                    unique_values_3d = color_data_series_3d.dropna().unique()
                    if COLORS_AVAILABLE:
                        color_discrete_map_3d = create_categorical_color_map(unique_values_3d)

                    fig_3d = px.scatter_3d(
                        x=scores[pc_x], y=scores[pc_y], z=scores[pc_z],
                        color=color_data_3d, text=text_param_3d,
                        title=f"3D Scores: {pc_x}, {pc_y}, {pc_z} (colored by {color_by_3d}){title_suffix}<br>Total Explained Variance: {var_total_3d:.1f}%",
                        labels={'x': f'{pc_x} ({var_x:.1f}%)', 'y': f'{pc_y} ({var_y:.1f}%)', 'z': f'{pc_z} ({var_z:.1f}%)', 'color': color_by_3d},
                        color_discrete_map=color_discrete_map_3d
                    )
            else:
                # Categorical (default for Index and string data)
                color_data_series_3d = pd.Series(color_data_3d)
                unique_values_3d = color_data_series_3d.dropna().unique()
                if COLORS_AVAILABLE:
                    color_discrete_map_3d = create_categorical_color_map(unique_values_3d)

                fig_3d = px.scatter_3d(
                    x=scores[pc_x], y=scores[pc_y], z=scores[pc_z],
                    color=color_data_3d, text=text_param_3d,
                    title=f"3D Scores: {pc_x}, {pc_y}, {pc_z} (colored by {color_by_3d}){title_suffix}<br>Total Explained Variance: {var_total_3d:.1f}%",
                    labels={'x': f'{pc_x} ({var_x:.1f}%)', 'y': f'{pc_y} ({var_y:.1f}%)', 'z': f'{pc_z} ({var_z:.1f}%)', 'color': color_by_3d},
                    color_discrete_map=color_discrete_map_3d
                )

        # Update text position
        if show_labels_from_3d != "None":
            fig_3d.update_traces(textposition="top center")

        # Update point size
        fig_3d.update_traces(marker=dict(size=point_size_3d))

        # Update layout with centered title
        fig_3d.update_layout(
            title=dict(
                x=0.5,
                xanchor='center',
                font=dict(size=16)
            ),
            height=700,
            template='plotly_white',
            scene=dict(
                xaxis_title=f'{pc_x} ({var_x:.1f}%)',
                yaxis_title=f'{pc_y} ({var_y:.1f}%)',
                zaxis_title=f'{pc_z} ({var_z:.1f}%)'
            ),
            margin=dict(t=100)
        )

        st.plotly_chart(fig_3d, use_container_width=True, key="scores_3d")

        # Display metrics
        st.markdown("#### Variance Summary")
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        with metric_col1:
            st.metric(f"{pc_x} Variance", f"{var_x:.1f}%")
        with metric_col2:
            st.metric(f"{pc_y} Variance", f"{var_y:.1f}%")
        with metric_col3:
            st.metric(f"{pc_z} Variance", f"{var_z:.1f}%")

    # =========================================================================
    # LINE PLOT: Score Evolution Over Sample Sequence
    # =========================================================================
    st.divider()
    st.markdown("### 📈 Score Plot Line (Time Series)")
    st.info("Show PC scores as a line plot over sample sequence")

    # Select PCs to show
    n_comp_available = scores.shape[1]
    pc_selection = st.multiselect(
        "Select components to display:",
        [f"PC{i+1}" for i in range(min(4, n_comp_available))],
        default=["PC1", "PC2"],
        key="score_line_pcs"
    )

    # Get numeric and categorical columns for color/encoding options
    if data is not None:
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = data.select_dtypes(exclude=[np.number]).columns.tolist()
    else:
        numeric_cols = []
        categorical_cols = []

    # Display options for line plot
    col1, col2 = st.columns(2)

    with col1:
        # Color segments by categorical variable
        color_segments_by = st.selectbox(
            "Color segments by:",
            ['None'] + categorical_cols,
            key="score_line_color",
            help="Select categorical variable to color line segments"
        )

    with col2:
        # Show labels on points
        show_line_labels = st.selectbox(
            "Show labels:",
            ['None'] + categorical_cols + numeric_cols,
            key="score_line_labels",
            help="Show code/label on each point"
        )

    if pc_selection:
        # Use dedicated function from pca_plots
        # Note: plot_line_scores is imported at module level (line 28)
        fig_line = plot_line_scores(
            scores=scores,
            pc_names=pc_selection,
            data=data,
            color_by=color_segments_by,
            encode_by=color_segments_by,  # Same as color_by for segments
            show_labels=show_line_labels,
            label_source=data
        )

        st.plotly_chart(fig_line, use_container_width=True)


# ============================================================================
# TAB 5: INTERPRETATION
# ============================================================================

#def _show_interpretation_tab():
    """Display component/factor interpretation."""
#    st.markdown("## 📝 Component Interpretation")
    
#    if 'pca_results' not in st.session_state:
#        st.warning("⚠️ Please compute PCA in **Model Computation** tab first")
#        return
    
#    pca_results = st.session_state['pca_results']
#    is_varimax = pca_results.get('varimax_applied', False)
    
#    st.info("🔮 AI-powered interpretation coming soon!")
    
#    st.markdown("""
    ### Manual Interpretation Guide
    
#    **For Standard PCA:**
#    1. Look at the scree plot to determine significant components
#    2. Examine loadings to understand which variables contribute to each PC
#    3. Interpret scores to identify sample patterns and clusters
#    4. Check diagnostics for outliers and model quality
    
#    **For Varimax-Rotated Factors:**
#    1. Rotated factors have simpler structure (easier interpretation)
#    2. Each variable typically loads highly on fewer factors
#    3. Look for factor themes based on high-loading variables
#    4. Factors remain orthogonal (uncorrelated)
    
#    👉 Check the **Loadings** and **Scores** tabs for detailed visualizations
#    """)


# ============================================================================
# TAB 6: ADVANCED DIAGNOSTICS
# ============================================================================

def _create_diagnostic_t2_q_plot(t2_values, q_values, t2_limit, q_limit,
                                   sample_names=None, show_labels=False,
                                   color_data=None, color_variable=None,
                                   show_trajectory=False,
                                   trajectory_style="Line",
                                   trajectory_colors=None):
    """
    Create T² vs Q diagnostic scatter plot with trajectory options.

    Args:
        t2_values: Array of T² statistics
        q_values: Array of Q residuals
        t2_limit: T² control limit
        q_limit: Q control limit
        sample_names: Optional sample identifiers
        show_labels: Whether to show sample labels on points
        color_data: Optional data for point coloring
        color_variable: Name of the color variable
        show_trajectory: Whether to connect points in sequence
        trajectory_style: "Line" for simple gray line, "Gradient (Blue→Red)" for color gradient
        trajectory_colors: List of RGB color strings for gradient trajectory

    Returns:
        plotly.graph_objects.Figure
    """
    import plotly.express as px
    import plotly.graph_objects as go

    # Get unified color scheme
    if COLORS_AVAILABLE:
        from color_utils import get_unified_color_schemes, create_categorical_color_map, is_quantitative_variable
        colors = get_unified_color_schemes()
    else:
        colors = {'background': 'white', 'paper': 'white', 'text': 'black',
                  'grid': '#e6e6e6', 'point_color': 'blue', 'control_colors': ['green', 'orange', 'red']}

    # Prepare hover text
    if sample_names is None:
        hover_text = [f"Sample_{i+1}" for i in range(len(t2_values))]
    else:
        hover_text = sample_names

    # Prepare text labels for display
    text_labels = hover_text if show_labels else None

    # Create figure based on color data type
    if color_data is not None and COLORS_AVAILABLE:
        color_series = pd.Series(color_data)

        if is_quantitative_variable(color_series):
            # QUANTITATIVE: Use continuous blue-to-red scale
            color_scale = [(0.0, 'rgb(0, 0, 255)'), (0.5, 'rgb(128, 0, 128)'), (1.0, 'rgb(255, 0, 0)')]

            fig = px.scatter(
                x=t2_values,
                y=q_values,
                color=color_data,
                text=text_labels,
                title=f"T² vs Q Diagnostic Plot" + (f" (colored by {color_variable})" if color_variable else ""),
                labels={'x': 'T² Statistic', 'y': 'Q Residual', 'color': color_variable},
                color_continuous_scale=color_scale
            )
        else:
            # CATEGORICAL: Use discrete color map
            unique_values = color_series.dropna().unique()
            color_discrete_map = create_categorical_color_map(unique_values)

            fig = px.scatter(
                x=t2_values,
                y=q_values,
                color=color_data,
                text=text_labels,
                title=f"T² vs Q Diagnostic Plot" + (f" (colored by {color_variable})" if color_variable else ""),
                labels={'x': 'T² Statistic', 'y': 'Q Residual', 'color': color_variable},
                color_discrete_map=color_discrete_map
            )

        # Update traces for better appearance
        fig.update_traces(
            marker=dict(size=8, opacity=0.7),
            textposition="top center"
        )
    else:
        # No color variable - use default
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=t2_values,
            y=q_values,
            mode='markers+text' if show_labels else 'markers',
            name='Samples',
            text=text_labels,
            hovertext=hover_text,
            hovertemplate='<b>%{hovertext}</b><br>T²: %{x:.2f}<br>Q: %{y:.2f}<extra></extra>',
            marker=dict(size=8, color=colors['point_color'], opacity=0.7),
            textposition="top center"
        ))

    # Add trajectory if requested
    if show_trajectory:
        if trajectory_style == "Gradient (Blue→Red)" and trajectory_colors is not None:
            # GRADIENT TRAJECTORY: colored segments
            for i in range(len(t2_values) - 1):
                fig.add_trace(go.Scatter(
                    x=[t2_values[i], t2_values[i+1]],
                    y=[q_values[i], q_values[i+1]],
                    mode='lines',
                    line=dict(color=trajectory_colors[i], width=2),
                    showlegend=False,
                    hoverinfo='skip'
                ))

            # Add legend entry for gradient
            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode='lines',
                name='Trajectory (Blue→Red)',
                line=dict(color='purple', width=2),
                showlegend=True
            ))
        else:
            # SIMPLE LINE: gray dotted
            fig.add_trace(go.Scatter(
                x=t2_values,
                y=q_values,
                mode='lines',
                name='Trajectory',
                line=dict(color='lightgray', width=1, dash='dot'),
                showlegend=True,
                hoverinfo='skip'
            ))

    # Add control limit lines
    fig.add_vline(x=t2_limit, line_dash="dash", line_color="red",
                  annotation_text=f"T² limit", annotation_position="top right")
    fig.add_hline(y=q_limit, line_dash="dash", line_color="red",
                  annotation_text=f"Q limit", annotation_position="right")

    # Add acceptance box (green region)
    fig.add_shape(
        type="rect",
        x0=0, x1=t2_limit,
        y0=0, y1=q_limit,
        fillcolor="green",
        opacity=0.1,
        layer="below",
        line_width=0
    )

    # Update layout with unified styling
    fig.update_layout(
        xaxis_title="T² Statistic",
        yaxis_title="Q Residual",
        height=600,
        hovermode='closest',
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['paper'],
        font=dict(color=colors['text']),
        xaxis=dict(gridcolor=colors['grid']),
        yaxis=dict(gridcolor=colors['grid'])
    )

    return fig




def _show_interpretation_tab():
    """
    Display Interpretation tab for joint loadings-scores analysis.

    Features ALL visualization options from Score Plots tab:
    - Color points by any variable (categorical or quantitative)
    - Show text labels
    - Convex hulls for categorical groups
    - Trajectory lines (Sequential or Categorical strategies)
    - Marker size control
    - Variable annotations panel
    - General interpretation notes
    """

    st.markdown("## 📝 Joint Loadings-Scores Interpretation")

    # Check PCA results
    if 'pca_results' not in st.session_state:
        st.warning("⚠️ Please compute PCA in **Model Computation** tab first")
        return

    pca_results = st.session_state['pca_results']
    loadings = pca_results['loadings']
    scores = pca_results['scores']
    explained_var = pca_results['explained_variance_ratio']

    # Get original data for color-by and label options
    data = st.session_state.get('current_data', None)

    # Initialize session state for variable annotations
    if 'variable_annotations' not in st.session_state:
        st.session_state.variable_annotations = {}

    # ========== PC SELECTION ==========
    st.markdown("#### ⚙️ Principal Component Selection")

    col1, col2 = st.columns(2)
    with col1:
        pc_x = st.selectbox("X-axis PC:", loadings.columns, index=0, key='interp_pc_x')
    with col2:
        pc_y_idx = 1 if len(loadings.columns) > 1 else 0
        pc_y = st.selectbox("Y-axis PC:", loadings.columns, index=pc_y_idx, key='interp_pc_y')

    # ========== VISUALIZATION OPTIONS (mirrored from Score Plots tab) ==========
    st.markdown("#### 🎨 Visualization Options")

    # === COLOR AND LABEL OPTIONS ===
    col3, col4 = st.columns(2)
    with col3:
        # Show ALL columns plus Index option
        color_options = ["None"]
        if data is not None:
            color_options.extend(list(data.columns))
        color_options.append("Index")

        color_by = st.selectbox("Color points by:", color_options, key='color_by_interp')

    with col4:
        # Label options: None, Index, or any column
        label_options = ["Index", "None"]
        if data is not None:
            label_options.extend(list(data.columns))

        show_labels_from = st.selectbox("Show labels:", label_options, key='show_labels_interp')

    # === CONVEX HULLS (only if color_by is set) ===
    if color_by != "None":
        col_hull1, col_hull2 = st.columns(2)
        with col_hull1:
            show_convex_hull = st.checkbox("Show convex hulls (categorical)", value=False, key='show_hull_interp')
        with col_hull2:
            hull_opacity = st.slider(
                "Hull opacity:",
                min_value=0.0,
                max_value=1.0,
                value=0.2,
                step=0.05,
                key='hull_opacity_interp'
            )
    else:
        show_convex_hull = False
        hull_opacity = 0.2

    # === TRAJECTORY LINES STRATEGY ===
    st.markdown("**🎯 Sample Trajectory Lines**")
    st.caption("Connect samples in order to visualize temporal or sequential progression")

    col_traj1, col_traj2, col_traj3 = st.columns(3)
    with col_traj1:
        trajectory_strategy = st.selectbox(
            "Trajectory strategy:",
            ["None", "Sequential", "Categorical"],
            help="None: no lines | Sequential: one line through all samples | Categorical: separate lines per group",
            key='trajectory_strategy_interp'
        )

    with col_traj2:
        trajectory_width = st.slider(
            "Line width:",
            min_value=1,
            max_value=5,
            value=2,
            step=1,
            key='trajectory_width_interp'
        )

    with col_traj3:
        trajectory_opacity = st.slider(
            "Line opacity:",
            min_value=0.0,
            max_value=1.0,
            value=0.6,
            step=0.05,
            key='trajectory_opacity_interp'
        )

    # === MARKER SIZE CONTROL ===
    st.markdown("**⚫ Marker Size**")
    marker_size = st.slider(
        "Point size:",
        min_value=1,
        max_value=20,
        value=8,
        step=1,
        key="marker_size_interp"
    )

    # === TRAJECTORY GROUPING (for Categorical strategy) ===
    trajectory_groupby_column = None
    trajectory_metadata_for_coloring = None

    if trajectory_strategy == "Categorical":
        if color_by != "None" and color_by != "Index":
            # Use the same variable as point coloring
            if data is not None and color_by in data.columns:
                try:
                    trajectory_groupby_column = data.loc[scores.index, color_by]
                    trajectory_metadata_for_coloring = trajectory_groupby_column
                    st.info(f"✅ Trajectory lines grouped by: **{color_by}** (same as point colors)")
                except Exception as e:
                    st.warning(f"⚠️ Could not use '{color_by}' for trajectory: {str(e)}")
                    trajectory_strategy = "None"
            else:
                st.warning("⚠️ Selected variable is not available in data")
                trajectory_strategy = "None"
        else:
            st.warning("⚠️ Please select a categorical variable in 'Color points by' to use Categorical trajectories")
            trajectory_strategy = "None"

    # === TRAJECTORY COLORING OPTIONS ===
    trajectory_color_variable = None
    trajectory_color_by_index = True  # Default for Sequential
    trajectory_color_vector = None

    if trajectory_strategy == "Sequential":
        # Sequential: Show gradient coloring
        st.markdown("---")
        st.markdown("**🎨 Trajectory Line Coloring (Sequential)**")
        st.caption("Colors flow from BLUE (first point) → RED (last point)")

        # Get numeric columns for coloring options
        numeric_cols = []
        if data is not None:
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()

        # Color by selector
        color_trajectory_by = st.selectbox(
            "Color gradient by:",
            ["Sample Sequence (1→N)"] + numeric_cols,
            key="color_trajectory_by_interp",
            help="Sample Sequence: BLUE(1st)→RED(last) | Numeric Variable: BLUE(min)→RED(max)"
        )

        # Determine coloring strategy
        if color_trajectory_by == "Sample Sequence (1→N)":
            trajectory_color_by_index = True
            trajectory_color_variable = None
            st.info("💡 Gradient: Sample sequence (1→N) | BLUE=first point, RED=last point")
        else:
            trajectory_color_variable = color_trajectory_by
            trajectory_color_by_index = False
            st.info(f"💡 Gradient: {color_trajectory_by} variable | BLUE=minimum, RED=maximum")

    elif trajectory_strategy == "Categorical":
        # Categorical: Show two MUTUALLY EXCLUSIVE options
        st.markdown("---")
        st.markdown("**🎨 Trajectory Coloring (Categorical)**")

        # Checkbox: Choose between Option 1 or Option 2
        use_custom_color = st.checkbox(
            "Apply coloring to specific batch only",
            value=False,
            key="use_custom_trajectory_color_interp",
            help="Highlight one batch (BRIGHT gradient BLUE→RED) while dimming others (BACKGROUND gray)"
        )

        if use_custom_color and trajectory_groupby_column is not None:
            # OPTION 2: Independent batch highlighting
            unique_categories = sorted(trajectory_groupby_column.dropna().unique())

            if len(unique_categories) > 0:
                selected_batch_for_color = st.selectbox(
                    "Apply coloring to batch:",
                    unique_categories,
                    key="batch_for_color_interp"
                )

                st.info(f"✨ **{selected_batch_for_color}** will be BRIGHT (BLUE→RED gradient)\nOther batches will be DIM (gray background)")

                # Use sequential_bright mode for gradient coloring
                trajectory_color_vector = {
                    'category': selected_batch_for_color,
                    'variable': None,
                    'mode': 'sequential_bright',
                    'values': None,
                    'min': None,
                    'max': None
                }

                # Disable Option 1 when Option 2 is active
                trajectory_metadata_for_coloring = None
            else:
                trajectory_color_vector = None
                trajectory_metadata_for_coloring = None

        else:
            # OPTION 1 (DEFAULT): Color by category
            st.markdown("**Option 1️⃣: Color by Category** (Default)")
            st.caption(f"Trajectory lines colored by category: **{color_by}**")
            st.info(f"✅ Each {color_by} category gets a distinct color")

            trajectory_color_vector = None
            # Keep trajectory_metadata_for_coloring as set earlier

        # Default coloring: by index (for backward compatibility)
        trajectory_color_by_index = True
        trajectory_color_variable = None

    # === PREPARE COLOR DATA AND TEXT LABELS ===
    color_data = None
    if color_by != "None":
        if color_by == "Index":
            color_data = pd.Series(range(len(scores)), index=scores.index, name="Row Index")
        elif data is not None:
            try:
                color_data = data.loc[scores.index, color_by]
            except:
                st.warning(f"⚠️ Could not align color variable '{color_by}' with scores")
                color_data = None

    # Prepare text labels
    text_param = None
    if show_labels_from != "None":
        # Start with sample names/indices
        if show_labels_from == "Index":
            base_labels = [str(idx) for idx in scores.index]
        elif data is not None:
            try:
                if show_labels_from in data.columns:
                    col_values = data[show_labels_from].reindex(scores.index)
                    base_labels = [str(val) if pd.notna(val) else str(idx)
                                  for idx, val in zip(scores.index, col_values)]
                else:
                    st.warning(f"⚠️ Column '{show_labels_from}' not found in data")
                    base_labels = [str(idx) for idx in scores.index]
            except Exception as e:
                st.warning(f"⚠️ Could not read labels from '{show_labels_from}': {str(e)}")
                base_labels = [str(idx) for idx in scores.index]
        else:
            base_labels = [str(idx) for idx in scores.index]

        text_param = pd.Series(base_labels, index=scores.index)

    # ========== MAIN PLOT AREA ==========
    st.markdown("#### 📊 Loadings-Scores Analysis")

    try:
        from pca_utils.pca_plots import plot_loadings_scores_side_by_side

        fig = plot_loadings_scores_side_by_side(
            loadings=loadings,
            scores=scores,
            pc_x=pc_x,
            pc_y=pc_y,
            explained_variance_ratio=explained_var,
            variable_annotations=st.session_state.get('variable_annotations', {}),
            arrow_scale=1.0,
            # === NEW PARAMETERS ===
            color_data=color_data,
            text_labels=text_param,
            show_convex_hull=show_convex_hull,
            hull_opacity=hull_opacity,
            trajectory_strategy=trajectory_strategy,
            trajectory_groupby_column=trajectory_groupby_column,
            trajectory_metadata_for_coloring=trajectory_metadata_for_coloring,
            trajectory_color_vector=trajectory_color_vector,
            trajectory_width=trajectory_width,
            trajectory_opacity=trajectory_opacity,
            trajectory_color_by_index=trajectory_color_by_index,
            trajectory_color_variable=trajectory_color_variable,
            marker_size=marker_size
        )

        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error creating plot: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return



def _show_advanced_diagnostics_tab():
    """Display advanced diagnostics (T², Q residuals, outliers)."""
    st.markdown("## 🔬 Advanced Diagnostics")
    st.markdown("*Statistical Quality Control - Hotelling T² and Q Statistics*")

    if 'pca_results' not in st.session_state:
        st.warning("⚠️ Please compute PCA in **Model Computation** tab first")
        return

    # Get PCA results
    pca_results = st.session_state['pca_results']
    scores = pca_results['scores']
    loadings = pca_results['loadings']
    n_components = pca_results['n_components']

    # === VALIDATION ===
    # Get the number of samples from PCA results (source of truth)
    n_samples_pca = len(scores)

    # Validate minimum samples
    if n_samples_pca < 10:
        st.error(f"❌ Insufficient samples for diagnostics: {n_samples_pca} samples found. Need at least 10.")
        return

    # Validate n_components vs n_samples
    if n_components >= n_samples_pca:
        st.error(f"❌ Too many components: {n_components} components for {n_samples_pca} samples. Reduce components in Model Computation.")
        return

    # Get original data from PCA results (not current_data which may have changed)
    data = pca_results.get('original_data', None)
    if data is None:
        # Fallback to current_data for backward compatibility
        data = st.session_state.get('current_data', None)
        if data is None:
            st.error("❌ Original data not available")
            return

    # CHECK for missing values (use data from PCA results, not current_data)
    has_missing_values = data.isna().sum().sum() > 0

    if has_missing_values:
        st.warning("⚠️ Dataset contains missing values - Q statistic cannot be calculated. Only T² plots are available.")

    st.divider()

    # === CRITICAL FIX: Get n_samples from PCA results ===
    # When PCA is computed on a subset of rows, we must use that subset
    # Otherwise we get shape mismatches (e.g., 125 scores vs 141 data rows)
    n_samples = len(pca_results['scores'])  # Number of samples in PCA model

    # Get numeric variables used in PCA
    selected_vars = pca_results.get('selected_vars', data.select_dtypes(include=[np.number]).columns.tolist())

    # Get ONLY the rows that were used in PCA (match scores index)
    pca_indices = pca_results['scores'].index
    X_data = data.loc[pca_indices, selected_vars]

    n_variables = len(selected_vars)

    # === SECTION 1 - CONFIGURATION ===
    st.markdown("### ⚙️ Diagnostic Configuration")

    col1, col2 = st.columns(2)

    with col1:
        # Handle case when n_components < 2
        if n_components < 2:
            st.error("❌ At least 2 components required for diagnostics")
            return
        elif n_components == 2:
            # If exactly 2 components, use number input instead of slider
            n_comp_diag = st.number_input(
                "Number of Components:",
                min_value=2,
                max_value=2,
                value=2,
                key="diag_components",
                help="Number of PCs to use for T² and Q calculations"
            )
        else:
            # Normal slider when n_components > 2
            n_comp_diag = st.slider(
                "Number of Components:",
                min_value=2,
                max_value=n_components,
                value=min(5, n_components),
                key="diag_components",
                help="Number of PCs to use for T² and Q calculations"
            )

    with col2:
        approach = st.radio(
            "Control Limit Approach:",
            ["Independent (95/99/99.9%)", "Joint (97.5/99.5/99.95%)"],
            key="diag_approach",
            help="Independent: separate thresholds | Joint: hierarchical classification"
        )

    # === SECTION 2 - CALCULATE STATISTICS (CORRECTED) ===
    # Get subset of scores and loadings
    scores_diag = scores.iloc[:, :n_comp_diag].values  # (n_samples, n_comp)
    loadings_diag = loadings.iloc[:, :n_comp_diag].values  # (n_vars, n_comp)
    eigenvalues_diag = pca_results['eigenvalues'][:n_comp_diag]  # (n_comp,)

    # Preprocess data (center/scale like during training)
    X_centered = X_data.values.copy()
    if pca_results.get('centering', True):
        means = pca_results['means']
        if hasattr(means, 'values'):
            means = means.values
        X_centered = X_centered - means
    if pca_results.get('scaling', False):
        stds = pca_results['stds']
        if hasattr(stds, 'values'):
            stds = stds.values
        X_centered = X_centered / stds

    # Import required functions
    from scipy.stats import f, chi2, t as t_dist
    from pca_utils.pca_statistics import calculate_hotelling_t2, calculate_q_residuals

    try:
        # === USE EXISTING TESTED FUNCTIONS (CORRECT!) ===
        # Calculate T² using existing function from pca_statistics
        # This function already implements the correct formula: T² = Σ(score_k² / λ_k)
        # and matches R-CAT values

        t2_values, t2_limit_single = calculate_hotelling_t2(
            scores_diag,
            eigenvalues_diag,
            alpha=0.95
        )

        # Calculate Q residuals using existing function from pca_statistics
        # This function implements: Q = ||X - X_reconstructed||²
        # SKIP Q calculation if missing values detected
        if not has_missing_values:
            q_values, q_limit_single = calculate_q_residuals(
                X_centered,
                scores_diag,
                loadings_diag,
                alpha=0.95
            )
        else:
            # Set Q values to None when missing data present
            q_values = None
            q_limit_single = None

        # === DEBUG: VERIFY T² CALCULATION ===
        with st.expander("🔍 DEBUG: T² Calculation Verification"):
            st.markdown("#### Array Shapes")
            col_d1, col_d2 = st.columns(2)
            with col_d1:
                st.write(f"**scores_diag shape**: {scores_diag.shape}")
                st.write(f"Expected: (n_samples={len(X_centered)}, n_comp={n_comp_diag})")
            with col_d2:
                st.write(f"**eigenvalues_diag shape**: {eigenvalues_diag.shape}")
                st.write(f"Expected: ({n_comp_diag},)")

            st.markdown("#### Eigenvalues (λ)")
            st.write(eigenvalues_diag)

            st.markdown("#### T² Statistics")
            col_t1, col_t2, col_t3 = st.columns(3)
            with col_t1:
                st.metric("T² min", f"{t2_values.min():.4f}")
            with col_t2:
                st.metric("T² max", f"{t2_values.max():.4f}")
            with col_t3:
                st.metric("T² mean", f"{t2_values.mean():.4f}")

            st.markdown("#### First 5 Samples T² Values")
            t2_verify_df = pd.DataFrame({
                'Sample': list(scores.index[:5]),
                'T²': t2_values[:5]
            })
            st.dataframe(t2_verify_df, use_container_width=True, hide_index=True)

            st.success(f"✅ **Sample 1 T²**: {t2_values[0]:.4f} (calculated using tested pca_statistics.calculate_hotelling_t2)")

            # Show component-wise contributions for first sample
            st.markdown("#### Sample 1 - Component-wise T² Contributions")
            sample1_scores = scores_diag[0, :]

            # Calculate sample variance eigenvalues for component-wise T² (CORRECTED)
            n_samples = len(scores_diag)
            eigenvalues_sample_var = eigenvalues_diag / (n_samples - 1)

            sample1_t2_components = sample1_scores**2 / eigenvalues_sample_var
            contrib_df = pd.DataFrame({
                'Component': [f'PC{i+1}' for i in range(n_comp_diag)],
                'Score': sample1_scores,
                'Score²': sample1_scores**2,
                'Eigenvalue (λ_sample)': eigenvalues_sample_var,
                'T² Contribution (score²/λ)': sample1_t2_components
            })
            st.dataframe(contrib_df.style.format({
                'Score': '{:.4f}',
                'Score²': '{:.4f}',
                'Eigenvalue (λ)': '{:.4f}',
                'T² Contribution (score²/λ)': '{:.4f}'
            }), use_container_width=True, hide_index=True)
            st.write(f"**Sum of contributions** (= T² for sample 1): {sample1_t2_components.sum():.4f}")

        # === CALCULATE CONTROL LIMITS FOR ALL CONFIDENCE LEVELS ===
        # Set confidence levels based on approach
        if "Independent" in approach:
            conf_levels = [0.95, 0.99, 0.999]
            conf_labels = ['95%', '99%', '99.9%']
        else:
            # Joint approach uses special alpha values (from R script lines 52-57)
            conf_levels = [0.974679, 0.994987, 0.9995]
            conf_labels = ['97.5%', '99.5%', '99.95%']

        # Calculate T² limits for all confidence levels using F-distribution
        # Using same formula as calculate_hotelling_t2 function
        # Note: n_samples is already defined above from pca_results['scores']
        t2_limits = {}
        for alpha, label in zip(conf_levels, conf_labels):
            f_val = f.ppf(alpha, n_comp_diag, n_samples - n_comp_diag)
            t2_lim = ((n_samples - 1) * n_comp_diag / (n_samples - n_comp_diag)) * f_val
            t2_limits[label] = t2_lim

        # Calculate Q limits for all confidence levels
        # Using log-normal approximation (same as calculate_q_residuals function from R script)
        # Q_limit = 10^(mean(log10(Q)) + t(alpha, n-1) * sd(log10(Q)))

        q_limits = {}
        if not has_missing_values and q_values is not None:
            # Calculate log-transformed Q values
            q_log = np.log10(q_values + 1e-10)  # Add small value to avoid log(0)
            q_mean_log = np.mean(q_log)
            q_std_log = np.std(q_log, ddof=1)  # Use sample std (ddof=1)

            for alpha, label in zip(conf_levels, conf_labels):
                # Get t-distribution critical value
                t_val = t_dist.ppf(alpha, n_samples - 1)
                # Calculate Q limit in log space, then convert back
                q_lim = 10 ** (q_mean_log + t_val * q_std_log)
                q_limits[label] = q_lim
        else:
            # Set dummy Q limits when missing data present
            for label in conf_labels:
                q_limits[label] = np.inf

        # Determine fault classification at first level
        alpha_main = conf_labels[0]
        if not has_missing_values and q_values is not None:
            faults = (t2_values > t2_limits[alpha_main]) | (q_values > q_limits[alpha_main])
            fault_types = []
            for i in range(len(t2_values)):
                if t2_values[i] > t2_limits[alpha_main] and q_values[i] > q_limits[alpha_main]:
                    fault_types.append("T²+Q")
                elif t2_values[i] > t2_limits[alpha_main]:
                    fault_types.append("T²")
                elif q_values[i] > q_limits[alpha_main]:
                    fault_types.append("Q")
                else:
                    fault_types.append("Normal")
        else:
            # Only T² classification when missing data present
            faults = t2_values > t2_limits[alpha_main]
            fault_types = []
            for i in range(len(t2_values)):
                if t2_values[i] > t2_limits[alpha_main]:
                    fault_types.append("T²")
                else:
                    fault_types.append("Normal")

        # === SECTION 3 - DIAGNOSTIC PLOTS ===
        st.divider()
        st.markdown("### 📊 Diagnostic Plots")
        st.markdown("*Visualize PCA scores with T² ellipses and influence plots*")

        # Get original dataset for coloring options (use data from PCA results)
        original_data = data

        # Get custom variables if available
        custom_vars = []
        if 'custom_variables' in st.session_state:
            custom_vars = list(st.session_state.custom_variables.keys())

        # Get available color variables (exclude PCA variables)
        available_color_vars = [col for col in original_data.columns if col not in selected_vars]

        # === DIAGNOSTIC PLOTS CONFIGURATION ===
        st.divider()
        st.markdown("### 📊 Diagnostic Plots Configuration")

        # Row 1: ONE Trajectory checkbox + Line Opacity slider
        row1_col1, row1_col2, row1_col3 = st.columns([0.8, 1.2, 0.6])

        with row1_col1:
            show_trajectory = st.checkbox(
                "Show Trajectory",
                value=True,
                key="show_trajectory_adv_diag"
            )
            # Use same flag for BOTH plots
            show_trajectory_score = show_trajectory
            show_trajectory_t2q = show_trajectory
            trajectory_style_score = 'gradient'
            trajectory_style_t2q = 'gradient'

        with row1_col2:
            # Empty space for alignment
            pass

        with row1_col3:
            trajectory_line_opacity = st.slider(
                "Line Opacity",
                min_value=0.2,
                max_value=1.0,
                value=1.0,
                step=0.1,
                key="trajectory_line_opacity_adv_diag",
                label_visibility="collapsed"
            )

        # Row 2: Color and Labels
        row2_col1, row2_col2 = st.columns([2, 1])

        with row2_col1:
            all_color_options = ["None", "Row Index"] + available_color_vars + custom_vars
            color_by_diag = st.selectbox(
                "Color Points By:",
                all_color_options,
                key="diag_color_points"
            )

        with row2_col2:
            show_sample_names_diag = st.checkbox(
                "Show Labels",
                value=False,
                key="diag_show_labels"
            )

        st.divider()

        # Prepare color data using unified color system
        from color_utils import is_quantitative_variable

        color_data_diag = None
        color_variable_diag = None

        if color_by_diag != "None":
            color_variable_diag = color_by_diag

            if color_by_diag == "Row Index":
                # Use row index as categorical
                color_data_diag = [str(i) for i in range(len(scores))]

            elif color_by_diag in custom_vars:
                # Custom variable from session state
                if 'custom_variables' in st.session_state:
                    custom_data = st.session_state.custom_variables.get(color_by_diag)
                    if custom_data is not None and len(custom_data) == len(scores):
                        color_data_diag = custom_data
                    else:
                        st.warning(f"⚠️ Custom variable '{color_by_diag}' length mismatch")
                        color_by_diag = "None"
                        color_variable_diag = None

            else:
                # Regular column from original data
                if color_by_diag in original_data.columns:
                    color_series = original_data[color_by_diag]

                    # Check if quantitative or categorical
                    if is_quantitative_variable(color_series):
                        # Quantitative: use continuous color scale (handled in plotting function)
                        color_data_diag = color_series.values
                        st.info(f"📊 Using continuous color scale for **{color_by_diag}** (quantitative)")
                    else:
                        # Categorical: convert to categorical color mapping
                        color_data_diag = color_series.astype(str).values
                        unique_count = len(pd.Series(color_data_diag).unique())
                        st.info(f"🎨 Using categorical colors for **{color_by_diag}** ({unique_count} categories)")
                else:
                    st.warning(f"⚠️ Column '{color_by_diag}' not found")
                    color_by_diag = "None"
                    color_variable_diag = None

        # Prepare trajectory color gradients for both plots
        trajectory_colors_score = None
        trajectory_colors_t2q = None

        if show_trajectory_score and trajectory_style_score == "gradient":
            # Create gradient from blue to red based on sample order
            n_samples_plot = len(scores_diag)
            # RGB gradient: blue (0,0,255) → purple (128,0,128) → red (255,0,0)
            trajectory_colors_score = []
            for i in range(n_samples_plot):
                t = i / max(n_samples_plot - 1, 1)  # Normalized position 0→1
                r = int(255 * t)
                g = 0
                b = int(255 * (1 - t))
                trajectory_colors_score.append(f'rgb({r},{g},{b})')

        if show_trajectory_t2q and trajectory_style_t2q == "gradient":
            # Create gradient for T²-Q plot
            n_samples_plot = len(scores_diag)
            trajectory_colors_t2q = []
            for i in range(n_samples_plot):
                t = i / max(n_samples_plot - 1, 1)  # Normalized position 0→1
                r = int(255 * t)
                g = 0
                b = int(255 * (1 - t))
                trajectory_colors_t2q.append(f'rgb({r},{g},{b})')

        # === CREATE PLOTS (SIDE-BY-SIDE) ===
        st.markdown("---")
        st.markdown("### 📈 Diagnostic Plots Display")

        # Prepare labels for plots
        labels_data_diag = None
        if show_sample_names_diag:
            labels_data_diag = data.index.tolist()

        col_left, col_right = st.columns(2)

        # Import score plot function and color helper (T² vs Q uses new function above)
        from pca_monitoring_page import create_score_plot, _color_to_rgba

        with col_left:
            st.subheader("Score Plot with T² Ellipses")

            # Prepare params for score plot
            pca_params_plot = {
                'n_samples_train': n_samples,
                'n_features': n_variables
            }

            # Call with trajectory, color_data and labels_data
            fig_score = create_score_plot(
                scores_diag,
                pca_results['explained_variance_ratio'][:n_comp_diag] * 100,
                timestamps=None,
                pca_params=pca_params_plot,
                start_sample_num=1,
                show_trajectory=show_trajectory_score,
                trajectory_style=trajectory_style_score,
                trajectory_colors=trajectory_colors_score,
                color_data=color_data_diag,
                labels_data=labels_data_diag,
                trajectory_line_opacity=trajectory_line_opacity
            )

            st.plotly_chart(fig_score, use_container_width=True, key="diag_score_plot")

        with col_right:
            if not has_missing_values:
                # === CREATE T² vs Q INFLUENCE PLOT WITH 3 NESTED BOXES ===
                try:
                    st.subheader("T² vs Q Influence Plot")

                    # Parse approach to get confidence levels
                    if "Independent" in approach:
                        alpha_levels = [0.95, 0.99, 0.999]
                        confidence_labels_plot = ['95%', '99%', '99.9%']
                    else:  # Joint
                        alpha_levels = [0.975, 0.995, 0.9995]
                        confidence_labels_plot = ['97.5%', '99.5%', '99.95%']

                    st.caption(f"Boxes define acceptancy regions at {', '.join(confidence_labels_plot)} limits")

                    # Create figure
                    fig_t2q = go.Figure()

                    # Determine which samples exceed each level
                    t2_exceed = [t2_values > t2_limits[conf_labels[i]] for i in range(3)]
                    q_exceed = [q_values > q_limits[conf_labels[i]] for i in range(3)]

                    # Color points based on which level they exceed (highest level takes priority)
                    point_colors = []
                    hover_texts = []

                    for i in range(len(t2_values)):
                        sample_id = data.index[i] if i < len(data.index) else f"Sample {i+1}"

                        # Check which levels are exceeded (highest level takes priority)
                        if t2_exceed[2][i] or q_exceed[2][i]:
                            color = 'red'
                            status = f'⚠️ EXCEEDS {confidence_labels_plot[2]} (SEVERE)'
                        elif t2_exceed[1][i] or q_exceed[1][i]:
                            color = 'orange'
                            status = f'⚠️ EXCEEDS {confidence_labels_plot[1]} (WARNING)'
                        elif t2_exceed[0][i] or q_exceed[0][i]:
                            color = 'gold'
                            status = f'⚠️ EXCEEDS {confidence_labels_plot[0]} (MILD)'
                        else:
                            color = 'blue'
                            status = '✅ WITHIN LIMITS'

                        point_colors.append(color)
                        hover_texts.append(f"{sample_id}<br>T²: {t2_values[i]:.2f}<br>Q: {q_values[i]:.2f}<br>{status}")

                    # Add scatter points
                    text_labels = data.index.tolist() if show_sample_names_diag else None
                    fig_t2q.add_trace(go.Scatter(
                        x=t2_values,
                        y=q_values,
                        mode='markers+text' if show_sample_names_diag else 'markers',
                        text=text_labels,
                        marker=dict(
                            size=8,
                            color=point_colors,
                            opacity=0.7,
                            line=dict(width=1, color='darkgray')
                        ),
                        hovertext=hover_texts,
                        hovertemplate='%{hovertext}<extra></extra>',
                        name='Samples',
                        textposition="top center"
                    ))

                    # Add trajectory if requested
                    if show_trajectory_t2q:
                        if trajectory_style_t2q == "gradient" and trajectory_colors_t2q is not None:
                            # GRADIENT TRAJECTORY: colored segments
                            for i in range(len(t2_values) - 1):
                                # Convert color to RGBA with opacity
                                line_color = _color_to_rgba(trajectory_colors_t2q[i], trajectory_line_opacity)
                                fig_t2q.add_trace(go.Scatter(
                                    x=[t2_values[i], t2_values[i+1]],
                                    y=[q_values[i], q_values[i+1]],
                                    mode='lines',
                                    line=dict(color=line_color, width=2),
                                    showlegend=False,
                                    hoverinfo='skip'
                                ))

                            # Add star at the end
                            fig_t2q.add_trace(go.Scatter(
                                x=[t2_values[-1]],
                                y=[q_values[-1]],
                                mode='markers',
                                marker=dict(symbol='star', size=15, color='red'),
                                name='Latest Sample',
                                hoverinfo='skip'
                            ))
                        else:
                            # SIMPLE LINE: gray dotted
                            # Convert lightgray to RGBA with opacity
                            line_color = _color_to_rgba('lightgray', trajectory_line_opacity)
                            fig_t2q.add_trace(go.Scatter(
                                x=t2_values,
                                y=q_values,
                                mode='lines',
                                name='Trajectory',
                                line=dict(color=line_color, width=1, dash='dot'),
                                showlegend=True,
                                hoverinfo='skip'
                            ))

                    # Add 3 nested boxes (green, orange, red) with different line styles
                    line_styles = ['solid', 'dash', 'dot']
                    box_colors = ['green', 'orange', 'red']

                    for level in range(3):
                        t2_lim = t2_limits[conf_labels[level]]
                        q_lim = q_limits[conf_labels[level]]

                        # Vertical line for T²
                        fig_t2q.add_vline(
                            x=t2_lim,
                            line_dash=line_styles[level],
                            line_color=box_colors[level],
                            line_width=2,
                            annotation_text=f"T² {confidence_labels_plot[level]}",
                            annotation_position="top right"
                        )

                        # Horizontal line for Q
                        fig_t2q.add_hline(
                            y=q_lim,
                            line_dash=line_styles[level],
                            line_color=box_colors[level],
                            line_width=2,
                            annotation_text=f"Q {confidence_labels_plot[level]}",
                            annotation_position="bottom right"
                        )

                        # Add acceptance boxes (lightest green for first level)
                        if level == 0:
                            fig_t2q.add_shape(
                                type="rect",
                                x0=0, x1=t2_lim,
                                y0=0, y1=q_lim,
                                fillcolor="green",
                                opacity=0.1,
                                layer="below",
                                line_width=0
                            )

                    fig_t2q.update_layout(
                        title=f"T² vs Q Influence Plot ({approach.split('(')[0].strip()} Approach)",
                        xaxis_title="Hotelling T²",
                        yaxis_title="Q-Residuals",
                        width=700,
                        height=500,
                        template='plotly_white',
                        hovermode='closest'
                    )

                    st.plotly_chart(fig_t2q, use_container_width=True, key="diag_t2q_plot")

                except Exception as e:
                    st.error(f"❌ Error creating influence plot: {str(e)}")
                    import traceback
                    with st.expander("🔍 Debug Info"):
                        st.code(traceback.format_exc())
            else:
                st.info("ℹ️ Q statistic requires complete data (no missing values)")

        # === SECTION 5 - FAULT SUMMARY TABLE ===
        st.markdown("---")
        st.markdown("### 📋 Fault Summary")

        # Build summary DataFrame with conditional Q column
        summary_data = {
            'Sample ID': scores.index,
            'T²': t2_values.round(4),
            f'T² Limit ({alpha_main})': [t2_limits[alpha_main]] * len(t2_values),
        }

        # Add Q columns only if no missing values
        if not has_missing_values and q_values is not None:
            summary_data['Q'] = q_values.round(4)
            summary_data[f'Q Limit ({alpha_main})'] = [q_limits[alpha_main]] * len(q_values)

        summary_data['Fault Type'] = fault_types

        summary_df = pd.DataFrame(summary_data)

        # Color code by fault type
        def color_fault(row):
            colors = []
            for col in row.index:
                if col == 'Fault Type':
                    val = row[col]
                    if val == "Normal":
                        colors.append('background-color: lightgreen')
                    elif val == "T²":
                        colors.append('background-color: lightyellow')
                    elif val == "Q":
                        colors.append('background-color: lightcyan')
                    else:  # T²+Q
                        colors.append('background-color: lightcoral')
                else:
                    colors.append('')
            return colors

        styled_summary = summary_df.style.apply(color_fault, axis=1)
        st.dataframe(styled_summary, use_container_width=True, hide_index=True)

        # Stats
        col_stats1, col_stats2, col_stats3 = st.columns(3)
        with col_stats1:
            st.metric("Total Samples", len(summary_df))
        with col_stats2:
            st.metric("Flagged Samples", faults.sum())
        with col_stats3:
            st.metric("Fault %", f"{faults.sum()/len(summary_df)*100:.1f}%")

        # === SECTION 6 - CONTRIBUTION ANALYSIS ===
        st.markdown("---")
        st.markdown("### 🔍 Contribution Analysis")
        st.markdown("*Analyze samples exceeding T²/Q control limits*")

        flagged_samples = summary_df[summary_df['Fault Type'] != 'Normal']['Sample ID'].tolist()

        if len(flagged_samples) == 0:
            st.success("✅ **No samples exceed control limits.** All samples are within normal operating conditions.")
        else:
            st.warning(f"⚠️ **{len(flagged_samples)} samples exceed control limits** (T²>limit OR Q>limit)")

            # Check if contribution functions are available
            if not CONTRIB_FUNCS_AVAILABLE:
                st.error("❌ Contribution analysis functions not available. Please ensure pca_monitoring_page.py is accessible.")
                st.info("✅ Showing simplified contribution analysis instead")

                # Fallback to simple contribution analysis
                selected_sample = st.selectbox(
                    "Select Outlier Sample:",
                    flagged_samples,
                    key="diag_outlier",
                    format_func=lambda x: f"{x} ({summary_df[summary_df['Sample ID']==x]['Fault Type'].values[0]})"
                )
                sample_idx = list(scores.index).index(selected_sample)

                # Get contributions (CORRECTED - use sample variance eigenvalues)
                sample_score = scores_diag[sample_idx]
                n_samples = len(scores_diag)
                eigenvalues_sample_var = eigenvalues_diag / (n_samples - 1)
                sample_t2_contrib = (sample_score**2 / eigenvalues_sample_var)

                # Calculate Q contributions only if no missing values
                if not has_missing_values:
                    residuals = X_centered[sample_idx:sample_idx+1] - (sample_score @ loadings_diag.T)
                    sample_q_contrib = (residuals.flatten()**2)
                    contrib_options = ["T² Contributions", "Q Contributions"]
                else:
                    sample_q_contrib = None
                    contrib_options = ["T² Contributions"]

                contrib_type = st.radio(
                    "Contribution Type:",
                    contrib_options,
                    horizontal=True,
                    key="diag_contrib_type"
                )

                if contrib_type == "T² Contributions":
                    contrib_data = sample_t2_contrib
                    top_n = min(10, len(contrib_data))
                    top_idx = np.argsort(contrib_data)[-top_n:][::-1]

                    fig_contrib = go.Figure(data=[
                        go.Bar(
                            x=[pca_results['loadings'].columns[i] for i in top_idx],
                            y=contrib_data[top_idx],
                            marker_color='steelblue'
                        )
                    ])
                    fig_contrib.update_layout(
                        title=f"Top Components Contributing to T² (Sample: {selected_sample})",
                        xaxis_title="Component",
                        yaxis_title="Contribution",
                        height=400,
                        template='plotly_white'
                    )
                else:  # Q Contributions
                    contrib_data = sample_q_contrib
                    top_n = min(10, len(contrib_data))
                    top_idx = np.argsort(contrib_data)[-top_n:][::-1]

                    fig_contrib = go.Figure(data=[
                        go.Bar(
                            x=[selected_vars[i] for i in top_idx],
                            y=contrib_data[top_idx],
                            marker_color='coral'
                        )
                    ])
                    fig_contrib.update_layout(
                        title=f"Top Variables Contributing to Q (Sample: {selected_sample})",
                        xaxis_title="Variable",
                        yaxis_title="Contribution",
                        xaxis=dict(tickangle=-45),
                        height=400,
                        template='plotly_white'
                    )

                st.plotly_chart(fig_contrib, use_container_width=True, key="diag_contrib")

            else:
                # COMPREHENSIVE CONTRIBUTION ANALYSIS (from pca_monitoring_page.py)

                if not has_missing_values:
                    # Calculate contributions (normalized by training set 95th percentile)
                    pca_params_contrib = {
                        's': scores_diag  # Use scores from diagnostics section
                    }

                    q_contrib, t2_contrib = calculate_all_contributions(
                        X_centered,
                        scores_diag,
                        loadings_diag,
                        pca_params_contrib
                    )
                else:
                    st.info("ℹ️ Q contribution analysis requires complete data (no missing values). Only T² contributions available.")

                if not has_missing_values:
                    # Normalize contributions by 95th percentile of training set
                    q_contrib_95th = np.percentile(np.abs(q_contrib), 95, axis=0)
                    t2_contrib_95th = np.percentile(np.abs(t2_contrib), 95, axis=0)

                    # Avoid division by zero
                    q_contrib_95th[q_contrib_95th == 0] = 1.0
                    t2_contrib_95th[t2_contrib_95th == 0] = 1.0

                    # Select sample from outliers only
                    sample_select_col, _ = st.columns([1, 1])
                    with sample_select_col:
                        # Get fault type for display
                        sample_fault_map = {row['Sample ID']: row['Fault Type']
                                           for _, row in summary_df.iterrows()}

                        selected_sample = st.selectbox(
                            "Select outlier sample for contribution analysis:",
                            options=flagged_samples,
                            format_func=lambda x: f"Sample {x} (T²={summary_df[summary_df['Sample ID']==x]['T²'].values[0]:.2f}, Q={summary_df[summary_df['Sample ID']==x]['Q'].values[0]:.2f}, Type: {sample_fault_map[x]})",
                            key="diag_contrib_sample"
                        )

                    # Get sample index
                    sample_idx = list(scores.index).index(selected_sample)

                    # Get contributions for selected sample
                    q_contrib_sample = q_contrib[sample_idx, :]
                    t2_contrib_sample = t2_contrib[sample_idx, :]

                    # Normalize
                    q_contrib_norm = q_contrib_sample / q_contrib_95th
                    t2_contrib_norm = t2_contrib_sample / t2_contrib_95th

                    # Determine which limits the sample exceeds
                    sample_t2 = t2_values[sample_idx]
                    sample_q = q_values[sample_idx] if q_values is not None else 0
                    exceeds_t2 = sample_t2 > t2_limits[alpha_main]
                    exceeds_q = (sample_q > q_limits[alpha_main]) if q_values is not None else False

                    # Dynamic contribution plot selection based on outlier type
                    if not exceeds_t2 and not exceeds_q:
                        # Sample is within limits
                        st.info(f"✅ Sample {selected_sample} is within control limits (T²={sample_t2:.2f}, Q={sample_q:.2f})")
                    elif exceeds_t2 and exceeds_q:
                        # Sample exceeds both limits - show both with selection option
                        st.markdown(f"⚠️ **Sample {selected_sample} exceeds BOTH T² and Q limits**")
                        contrib_display = st.radio(
                            "Select contribution plot to display:",
                            options=["Show Both", "T² Only", "Q Only"],
                            horizontal=True,
                            key="contrib_display_choice"
                        )

                        if contrib_display == "Show Both":
                            contrib_col1, contrib_col2 = st.columns(2)
                            with contrib_col1:
                                st.markdown(f"**T² Contributions - Sample {selected_sample}**")
                                fig_t2_contrib = create_contribution_plot_all_vars(
                                    t2_contrib_norm,
                                    selected_vars,
                                    statistic='T²'
                                )
                                st.plotly_chart(fig_t2_contrib, use_container_width=True)
                            with contrib_col2:
                                st.markdown(f"**Q Contributions - Sample {selected_sample}**")
                                fig_q_contrib = create_contribution_plot_all_vars(
                                    q_contrib_norm,
                                    selected_vars,
                                    statistic='Q'
                                )
                                st.plotly_chart(fig_q_contrib, use_container_width=True)
                        elif contrib_display == "T² Only":
                            st.markdown(f"**T² Contributions - Sample {selected_sample}**")
                            fig_t2_contrib = create_contribution_plot_all_vars(
                                t2_contrib_norm,
                                selected_vars,
                                statistic='T²'
                            )
                            st.plotly_chart(fig_t2_contrib, use_container_width=True)
                        else:  # Q Only
                            st.markdown(f"**Q Contributions - Sample {selected_sample}**")
                            fig_q_contrib = create_contribution_plot_all_vars(
                                q_contrib_norm,
                                selected_vars,
                                statistic='Q'
                            )
                            st.plotly_chart(fig_q_contrib, use_container_width=True)
                    elif exceeds_t2:
                        # Sample exceeds T² limit only - show T² contributions
                        st.markdown(f"**T² Contributions - Sample {selected_sample}** (exceeds T² limit)")
                        fig_t2_contrib = create_contribution_plot_all_vars(
                            t2_contrib_norm,
                            selected_vars,
                            statistic='T²'
                        )
                        st.plotly_chart(fig_t2_contrib, use_container_width=True)
                    else:  # exceeds_q
                        # Sample exceeds Q limit only - show Q contributions
                        st.markdown(f"**Q Contributions - Sample {selected_sample}** (exceeds Q limit)")
                        fig_q_contrib = create_contribution_plot_all_vars(
                            q_contrib_norm,
                            selected_vars,
                            statistic='Q'
                        )
                        st.plotly_chart(fig_q_contrib, use_container_width=True)

                    # Table: Variables where |contrib|>1 with real values vs training mean
                    # Only show if sample exceeds limits
                    if exceeds_t2 or exceeds_q:
                        st.markdown("### 🏆 Top Contributing Variables")
                        st.markdown("*Variables exceeding 95th percentile threshold (|contribution| > 1)*")

                        # Get training mean for comparison (from original data)
                        # Use .loc for label-based indexing (scores.index contains sample names like A1, A2, etc.)
                        try:
                            X_data_df = data[selected_vars].loc[scores.index]
                        except KeyError:
                            # If scores.index doesn't match data.index, try to align by position
                            if len(scores) == len(data):
                                X_data_df = data[selected_vars].reset_index(drop=True)
                            else:
                                st.error("❌ Cannot align sample indices between PCA results and original data")
                                X_data_df = data[selected_vars].iloc[:len(scores)]

                        training_mean = X_data_df.mean()

                        # Get real values for selected sample
                        sample_values = X_data_df.iloc[sample_idx]

                        # Filter variables based on which limits are exceeded and user choice
                        high_contrib_t2 = np.abs(t2_contrib_norm) > 1.0
                        high_contrib_q = np.abs(q_contrib_norm) > 1.0

                        # Determine which contributions to show based on outlier type
                        if exceeds_t2 and exceeds_q:
                            # Both exceeded - use user selection
                            if contrib_display == "Show Both":
                                high_contrib = high_contrib_t2 | high_contrib_q
                                show_both_columns = True
                            elif contrib_display == "T² Only":
                                high_contrib = high_contrib_t2
                                show_both_columns = False
                            else:  # Q Only
                                high_contrib = high_contrib_q
                                show_both_columns = False
                        elif exceeds_t2:
                            high_contrib = high_contrib_t2
                            show_both_columns = False
                        else:  # exceeds_q
                            high_contrib = high_contrib_q
                            show_both_columns = False

                        if high_contrib.sum() > 0:
                            contrib_table_data = []
                            for i, var in enumerate(selected_vars):
                                if high_contrib[i]:
                                    real_val = sample_values[var]
                                    mean_val = training_mean[var]
                                    diff = real_val - mean_val
                                    direction = "Higher ↑" if diff > 0 else "Lower ↓"

                                    row_data = {
                                        'Variable': var,
                                        'Real Value': f"{real_val:.3f}",
                                        'Training Mean': f"{mean_val:.3f}",
                                        'Difference': f"{diff:.3f}",
                                        'Direction': direction
                                    }

                                    # Add contribution columns based on what's being displayed
                                    if show_both_columns:
                                        row_data['|T² Contrib|'] = f"{abs(t2_contrib_norm[i]):.2f}"
                                        row_data['|Q Contrib|'] = f"{abs(q_contrib_norm[i]):.2f}"
                                    elif exceeds_t2 and not exceeds_q:
                                        row_data['|T² Contrib|'] = f"{abs(t2_contrib_norm[i]):.2f}"
                                    elif exceeds_q and not exceeds_t2:
                                        row_data['|Q Contrib|'] = f"{abs(q_contrib_norm[i]):.2f}"
                                    elif contrib_display == "T² Only":
                                        row_data['|T² Contrib|'] = f"{abs(t2_contrib_norm[i]):.2f}"
                                    else:  # Q Only
                                        row_data['|Q Contrib|'] = f"{abs(q_contrib_norm[i]):.2f}"

                                    contrib_table_data.append(row_data)

                            contrib_table = pd.DataFrame(contrib_table_data)
                            # Sort by contribution value
                            if show_both_columns:
                                contrib_table['Max_Contrib'] = contrib_table.apply(
                                    lambda row: max(float(row['|T² Contrib|']), float(row['|Q Contrib|'])),
                                    axis=1
                                )
                                contrib_table = contrib_table.sort_values('Max_Contrib', ascending=False).drop('Max_Contrib', axis=1)
                            elif '|T² Contrib|' in contrib_table.columns:
                                contrib_table['Sort_Val'] = contrib_table['|T² Contrib|'].astype(float)
                                contrib_table = contrib_table.sort_values('Sort_Val', ascending=False).drop('Sort_Val', axis=1)
                            else:
                                contrib_table['Sort_Val'] = contrib_table['|Q Contrib|'].astype(float)
                                contrib_table = contrib_table.sort_values('Sort_Val', ascending=False).drop('Sort_Val', axis=1)

                            st.dataframe(contrib_table, use_container_width=True, hide_index=True)
                        else:
                            st.info("No variables exceed the 95th percentile threshold.")

                    # Correlation scatter: training (grey), sample (red star)
                    # Only show if sample exceeds limits
                    if exceeds_t2 or exceeds_q:
                        # Determine which contribution type to analyze based on outlier type
                        if exceeds_t2 and exceeds_q:
                            # Both exceeded - use user selection
                            if contrib_display == "T² Only":
                                analyze_statistic = "T²"
                                contrib_norm_to_use = t2_contrib_norm
                            elif contrib_display == "Q Only":
                                analyze_statistic = "Q"
                                contrib_norm_to_use = q_contrib_norm
                            else:  # Show Both - default to Q
                                analyze_statistic = "Q"
                                contrib_norm_to_use = q_contrib_norm
                        elif exceeds_t2:
                            analyze_statistic = "T²"
                            contrib_norm_to_use = t2_contrib_norm
                        else:  # exceeds_q
                            analyze_statistic = "Q"
                            contrib_norm_to_use = q_contrib_norm

                        # === FLEXIBLE CORRELATION ANALYSIS ===
                        st.markdown("### 📈 Correlation Analysis - Flexible Variable Selection")
                        st.markdown("*Explore correlations by selecting variables from suggestions*")

                        # Get top contributors for X-axis selection
                        X_data_array = X_data_df.values
                        top_contrib = get_top_contributors(contrib_norm_to_use, selected_vars, n_top=10)
                        top_contrib_names = [name for name, _ in top_contrib]
                        top_contrib_values = {name: val for name, val in top_contrib}

                        # Step 1: Select X variable (from top contributors)
                        col1, col2, col3 = st.columns([2.5, 2.5, 2])

                        with col1:
                            default_x = top_contrib_names[0] if top_contrib_names else selected_vars[0]

                            selected_x_var = st.selectbox(
                                "🎯 Select Variable X (from top contributors):",
                                options=top_contrib_names,
                                index=0,
                                key="pca_diag_corr_x_var",
                                help="Variables ranked by T²/Q contribution to this outlier"
                            )

                            # Show contribution value
                            if selected_x_var in top_contrib_values:
                                contrib_val_x = top_contrib_values[selected_x_var]
                                st.caption(f"📊 Contribution: **{contrib_val_x:.4f}**")

                        # Step 2: Select Y variable (from top correlated to X)
                        with col2:
                            x_idx = selected_vars.index(selected_x_var)
                            top_corr = get_top_correlated(X_data_array, x_idx, selected_vars, n_top=10)
                            top_corr_names = [name for name, _ in top_corr]
                            top_corr_values = {name: corr for name, corr in top_corr}

                            if top_corr_names:
                                selected_y_var = st.selectbox(
                                    "🔗 Select Variable Y (top correlated):",
                                    options=top_corr_names,
                                    index=0,
                                    key="pca_diag_corr_y_var",
                                    help="Variables ranked by correlation to selected X variable"
                                )

                                # Show correlation value
                                if selected_y_var in top_corr_values:
                                    corr_val_xy = top_corr_values[selected_y_var]
                                    st.caption(f"📈 Correlation: **{corr_val_xy:.4f}**")
                            else:
                                st.error("❌ No correlated variables found")
                                selected_y_var = None

                        # Step 3: Show correlation coefficient
                        with col3:
                            if selected_x_var and selected_y_var:
                                x_idx = selected_vars.index(selected_x_var)
                                y_idx = selected_vars.index(selected_y_var)
                                actual_corr = np.corrcoef(X_data_array[:, x_idx], X_data_array[:, y_idx])[0, 1]

                                st.metric(
                                    "Correlation (training)",
                                    f"{actual_corr:.4f}",
                                    help="Pearson correlation coefficient"
                                )

                        # Create and display flexible correlation scatter plot
                        if selected_x_var and selected_y_var:
                            x_idx = selected_vars.index(selected_x_var)
                            y_idx = selected_vars.index(selected_y_var)
                            actual_corr = np.corrcoef(X_data_array[:, x_idx], X_data_array[:, y_idx])[0, 1]

                            fig_corr_scatter = create_correlation_scatter(
                                X_train=X_data_array,
                                X_sample=X_data_array[sample_idx, :],
                                var1_idx=x_idx,
                                var2_idx=y_idx,
                                var1_name=selected_x_var,
                                var2_name=selected_y_var,
                                correlation_val=actual_corr,
                                sample_idx=sample_idx
                            )

                            st.plotly_chart(fig_corr_scatter, use_container_width=True)

        # === SECTION 7 - EXPORT ===
        st.markdown("---")
        st.markdown("### 📥 Export Results")

        try:
            from io import BytesIO

            # Create Excel with multiple sheets
            excel_buffer = BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                # Sheet 1: Summary
                summary_df.to_excel(writer, sheet_name='Fault Summary', index=False)

                # Sheet 2: Flagged samples details
                if len(flagged_samples) > 0:
                    flagged_df = data.loc[flagged_samples]
                    flagged_df.to_excel(writer, sheet_name='Flagged Samples', index=True)

                    # Sheet 3: Outlier IDs
                    outlier_ids = pd.DataFrame({'Outlier Sample IDs': flagged_samples})
                    outlier_ids.to_excel(writer, sheet_name='Outlier List', index=False)

            excel_buffer.seek(0)

            st.download_button(
                "📊 Download Diagnostic Results (Excel)",
                excel_buffer.getvalue(),
                "pca_diagnostics.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="diag_download",
                use_container_width=True
            )

            st.success("✅ Advanced Diagnostics Complete")

        except ImportError as e:
            st.warning(f"⚠️ openpyxl not installed - Excel export not available ({str(e)})")
        except Exception as e:
            st.error(f"❌ Export failed: {str(e)}")

    except ImportError as e:
        st.error(f"❌ Required modules not available: {str(e)}")
        st.info("Please ensure pca_monitoring_page.py and scipy are installed")
    except Exception as e:
        st.error(f"❌ Error in diagnostics: {str(e)}")
        import traceback
        st.code(traceback.format_exc())


# ============================================================================
# TAB 7: EXTRACT & EXPORT
# ============================================================================

def _show_export_tab():
    """Display export options for PCA results."""
    st.markdown("## 💾 Extract & Export")
    #st.markdown("*Equivalent to R PCA_extract.r*")
    
    if 'pca_results' not in st.session_state:
        st.warning("⚠️ Please compute PCA in **Model Computation** tab first")
        return
    
    pca_results = st.session_state['pca_results']
    is_varimax = pca_results.get('varimax_applied', False)
    
    method_name = "Varimax" if is_varimax else "PCA"
    
    st.markdown("### 📁 Export Individual Files")
    
    # === EXPORT SCORES ===
    col1, col2, col3 = st.columns(3)
    
    with col1:
        scores_csv = pca_results['scores'].to_csv()
        st.download_button(
            "📊 Download Scores",
            scores_csv,
            f"{method_name}_scores.csv",
            "text/csv",
            help="Sample scores on principal components/factors"
        )
    
    with col2:
        loadings_csv = pca_results['loadings'].to_csv()
        st.download_button(
            "📈 Download Loadings",
            loadings_csv,
            f"{method_name}_loadings.csv",
            "text/csv",
            help="Variable loadings on principal components/factors"
        )
    
    with col3:
        # DEBUG: Check array lengths to prevent mismatches
        with st.expander("🔍 DEBUG: Array Lengths"):
            st.write(f"loadings.columns length: {len(pca_results['loadings'].columns)}")
            st.write(f"eigenvalues length: {len(pca_results['eigenvalues'])}")
            st.write(f"explained_variance_ratio length: {len(pca_results['explained_variance_ratio'])}")
            st.write(f"cumulative_variance length: {len(pca_results['cumulative_variance'])}")

        # FIX: Create component names explicitly based on eigenvalues array length
        # This ensures all arrays have matching dimensions
        n = len(pca_results['eigenvalues'])
        variance_df = pd.DataFrame({
            'Component': [f"PC{i+1}" for i in range(n)] if not is_varimax else [f"Factor{i+1}" for i in range(n)],
            'Eigenvalue': pca_results['eigenvalues'][:n],
            'Variance %': pca_results['explained_variance_ratio'][:n] * 100,
            'Cumulative %': pca_results['cumulative_variance'][:n] * 100
        })
        variance_csv = variance_df.to_csv(index=False)
        st.download_button(
            "📉 Download Variance",
            variance_csv,
            f"{method_name}_variance.csv",
            "text/csv",
            help="Variance explained by each component"
        )
    
    # === EXPORT COMPLETE ANALYSIS ===
    st.markdown("### 📦 Export Complete Analysis")
    
    try:
        from io import BytesIO
        
        excel_buffer = BytesIO()
        
        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
            # Write each component to separate sheet
            pca_results['scores'].to_excel(writer, sheet_name='Scores', index=True)
            pca_results['loadings'].to_excel(writer, sheet_name='Loadings', index=True)
            variance_df.to_excel(writer, sheet_name='Variance', index=False)
            
            # Add summary sheet
            summary_data = pd.DataFrame({
                'Parameter': [
                    'Analysis Method',
                    'Algorithm',
                    'Number of Components',
                    'Centering',
                    'Scaling',
                    'Total Variance Explained',
                    'Computation Time (s)'
                ],
                'Value': [
                    pca_results.get('method', 'Standard PCA'),
                    pca_results['algorithm'],
                    pca_results['n_components'],
                    'Yes' if pca_results['centering'] else 'No',
                    'Yes' if pca_results['scaling'] else 'No',
                    f"{pca_results['cumulative_variance'][-1]*100:.2f}%",
                    f"{pca_results.get('computation_time', 0):.3f}"
                ]
            })
            
            if is_varimax:
                summary_data = pd.concat([
                    summary_data,
                    pd.DataFrame({
                        'Parameter': ['Varimax Iterations'],
                        'Value': [pca_results.get('varimax_iterations', 'N/A')]
                    })
                ])
            
            summary_data.to_excel(writer, sheet_name='Summary', index=False)
        
        excel_buffer.seek(0)
        
        st.download_button(
            f"📄 Download Complete {method_name} Analysis (Excel)",
            excel_buffer.getvalue(),
            f"Complete_{method_name}_Analysis.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
        
        st.success("✅ Complete analysis ready for download!")
    
    except ImportError:
        st.warning("⚠️ openpyxl not installed - Excel export not available")
        st.info("Individual CSV exports are available above")
    except Exception as e:
        st.error(f"❌ Excel export failed: {str(e)}")
        st.info("Individual CSV exports are available above")
    
    # === DISPLAY DATA PREVIEW ===
    st.markdown("### 👁️ Data Preview")
    
    preview_choice = st.radio(
        "Select data to preview:",
        ["Scores", "Loadings", "Variance Summary"],
        horizontal=True
    )
    
    if preview_choice == "Scores":
        st.dataframe(pca_results['scores'], use_container_width=True)
    elif preview_choice == "Loadings":
        st.dataframe(pca_results['loadings'], use_container_width=True)
    else:
        st.dataframe(variance_df, use_container_width=True, hide_index=True)


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================
# TAB 8: MISSING DATA RECONSTRUCTION
# ============================================================================

def _show_missing_data_reconstruction_tab():
    """Display the Missing Data Reconstruction tab using NIPALS PCA."""
    st.markdown("## 🔄 Missing Data Reconstruction using NIPALS PCA")
    st.info("Reconstruct missing values using NIPALS algorithm - handles missing data natively during PCA decomposition")

    if not MISSING_DATA_AVAILABLE:
        st.warning("⚠️ Missing data reconstruction module not available")
        st.info("Please ensure missing_data_reconstruction.py is in the pca_utils directory")
        return

    # STEP 1: Check if data has missing values
    if 'current_data' not in st.session_state:
        st.warning("⚠️ No data loaded. Please load data first.")
        return

    data = st.session_state.current_data

    # Select only numeric columns
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) == 0:
        st.error("❌ No numeric columns found in dataset")
        return

    data_numeric = data[numeric_cols]

    # Count missing values
    n_missing, n_total, pct_missing = count_missing_values(data_numeric)

    # Display missing data statistics
    st.markdown("### 📊 Missing Data Statistics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🔍 Missing Values", f"{n_missing:,}")
    with col2:
        st.metric("📊 Total Cells", f"{n_total:,}")
    with col3:
        st.metric("📈 Missing Percentage", f"{pct_missing:.2f}%")

    if n_missing == 0:
        st.success("✅ No missing values detected - reconstruction not needed")
        st.info("💡 This tab is used when your dataset contains missing values (NaN)")
        return

    st.divider()

    # STEP 2: Configuration for NIPALS reconstruction
    st.markdown("### ⚙️ NIPALS Configuration")
    st.markdown("*Configure the NIPALS algorithm for missing data reconstruction*")

    col1, col2 = st.columns(2)

    with col1:
        n_comp_reconstruction = st.slider(
            "Number of Components:",
            min_value=1,
            max_value=min(20, len(numeric_cols), data_numeric.shape[0] - 1),
            value=min(5, len(numeric_cols), data_numeric.shape[0] - 1),
            key="n_comp_nipals",
            help="Number of principal components to use for reconstruction. More components = better reconstruction but possible overfitting."
        )

    with col2:
        center_nipals = st.checkbox(
            "Center data",
            value=True,
            key="center_nipals",
            help="Subtract column means before PCA (recommended)"
        )
        scale_nipals = st.checkbox(
            "Scale data (unit variance)",
            value=False,
            key="scale_nipals",
            help="Divide by column standard deviation (use if variables have different scales)"
        )

    st.divider()

    # STEP 3: Reconstruct missing data using NIPALS
    st.markdown("### 🚀 Run Reconstruction")

    if st.button("✨ Reconstruct Missing Values", key="run_nipals_reconstruction", use_container_width=True):
        with st.spinner("Running NIPALS algorithm to reconstruct missing values..."):
            try:
                # Call NIPALS reconstruction function
                X_reconstructed, info = reconstruct_missing_data(
                    X=data_numeric,
                    n_components=n_comp_reconstruction,
                    max_iter=1000,
                    tol=1e-6,
                    center=center_nipals,
                    scale=scale_nipals
                )

                # Store results in session state
                st.session_state.X_reconstructed = X_reconstructed
                st.session_state.reconstruction_info = info

                st.success("✅ Reconstruction completed successfully!")

            except Exception as e:
                st.error(f"❌ Reconstruction failed: {str(e)}")
                import traceback
                with st.expander("🔍 Error Details"):
                    st.code(traceback.format_exc())
                return

    # STEP 4: Display reconstruction results
    if 'X_reconstructed' in st.session_state and 'reconstruction_info' in st.session_state:
        st.divider()
        st.markdown("### 📈 Reconstruction Results")

        info = st.session_state.reconstruction_info
        X_recon = st.session_state.X_reconstructed

        # === FIX: PRESERVE METADATA COLUMNS ===
        # Check if original data has non-numeric columns
        original_data = st.session_state.current_data
        numeric_cols = original_data.select_dtypes(include=[np.number]).columns.tolist()
        non_numeric_cols = [col for col in original_data.columns if col not in numeric_cols]

        # If metadata exists, add it back to reconstructed data
        if non_numeric_cols:
            # X_recon only has numeric columns
            # Add back the metadata columns from original data
            X_recon_with_meta = X_recon.copy()

            for col in non_numeric_cols:
                if col in original_data.columns:
                    X_recon_with_meta[col] = original_data[col].values

            # Reorder columns: metadata first, then numeric
            cols_order = non_numeric_cols + numeric_cols
            X_recon_final = X_recon_with_meta[cols_order]

            # Store the corrected version
            st.session_state.X_reconstructed = X_recon_final
            X_recon = X_recon_final

        # Display key statistics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "✅ Values Filled",
                f"{info['n_missing_before']:,}",
                help="Number of missing values that were reconstructed"
            )

        with col2:
            variance_explained = info['total_variance_explained']
            st.metric(
                "📊 Variance Explained",
                f"{variance_explained:.1f}%",
                help=f"Total variance captured by {n_comp_reconstruction} components"
            )

        with col3:
            converged_status = "✅ Yes" if info['converged'] else "⚠️ No"
            st.metric(
                "🔄 Converged",
                converged_status,
                help="Whether NIPALS algorithm converged successfully"
            )

        with col4:
            st.metric(
                "🔢 Iterations",
                info['n_iterations'],
                help="Total iterations across all components"
            )

        # Display variance explained per component
        with st.expander("📊 Variance Explained by Component"):
            variance_df = pd.DataFrame({
                'Component': [f'PC{i+1}' for i in range(len(info['explained_variance']))],
                'Variance %': [f"{v:.2f}%" for v in info['explained_variance']]
            })
            st.dataframe(variance_df, use_container_width=True, hide_index=True)

        # Display convergence warning if needed
        if not info['converged']:
            st.warning("⚠️ NIPALS did not fully converge. Consider increasing max_iter or reducing n_components.")

        st.divider()

        # STEP 5: Export options
        st.markdown("### 💾 Export Reconstructed Data")

        base_name = st.session_state.get('dataset_name', 'dataset')
        export_name = st.text_input(
            "Export filename (without extension):",
            value=f"{base_name}_reconstructed",
            key="export_name_nipals",
            help="Name for exported file"
        )

        col1, col2 = st.columns(2)

        with col1:
            # Export to Excel
            try:
                from io import BytesIO
                excel_buffer = BytesIO()
                X_recon.to_excel(excel_buffer, index=True, sheet_name='Reconstructed Data')
                excel_buffer.seek(0)

                st.download_button(
                    "📥 Download Excel",
                    excel_buffer.getvalue(),
                    f"{export_name}.xlsx",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="download_excel_nipals",
                    use_container_width=True,
                    help="Download reconstructed data as Excel file (WITH metadata columns)"
                )
            except ImportError:
                st.warning("⚠️ openpyxl not installed - Excel export not available")

        with col2:
            # Export to CSV
            csv_buffer = X_recon.to_csv()
            st.download_button(
                "📥 Download CSV",
                csv_buffer,
                f"{export_name}.csv",
                "text/csv",
                key="download_csv_nipals",
                use_container_width=True,
                help="Download reconstructed data as CSV file (WITH metadata columns)"
            )

        st.divider()

        # STEP 6: Load to Workspace
        st.markdown("### 📂 Load to Workspace")
        st.markdown("*Make reconstructed data available for other analyses*")

        col_load1, col_load2 = st.columns([3, 1])

        with col_load2:
            if st.button("📥 Load to Workspace", key="load_workspace_nipals", use_container_width=True, type="primary"):
                try:
                    # Get the reconstructed data (with metadata preserved)
                    X_final = st.session_state.X_reconstructed

                    # Save to workspace split_datasets
                    if 'split_datasets' not in st.session_state:
                        st.session_state.split_datasets = {}

                    workspace_name = export_name
                    st.session_state.split_datasets[workspace_name] = {
                        'data': X_final,
                        'type': 'Reconstructed',
                        'parent': st.session_state.get('dataset_name', 'Original'),
                        'n_samples': len(X_final),
                        'creation_time': pd.Timestamp.now(),
                        'description': f'Missing data reconstructed using {info["n_components_used"]} PCA components'
                    }

                    st.success(f"✅ Data loaded to workspace as: **{workspace_name}**")
                    st.info("🔄 You can now use this dataset in other analyses")

                except Exception as e:
                    st.error(f"❌ Failed to load data to workspace: {str(e)}")
                    import traceback
                    with st.expander("🔍 Debug Info"):
                        st.code(traceback.format_exc())

        # STEP 7: Data preview
        st.divider()
        st.markdown("### 👁️ Data Preview")
        if non_numeric_cols:
            st.markdown(f"**Showing reconstructed data with {len(non_numeric_cols)} metadata column(s)**")
        else:
            st.markdown("**Showing reconstructed data (numeric columns only)**")

        preview_option = st.radio(
            "Select data to preview:",
            ["Reconstructed Data", "Original Data (with NaN)", "Comparison"],
            horizontal=True,
            key="preview_option_nipals"
        )

        if preview_option == "Reconstructed Data":
            # Show all columns (including metadata)
            st.dataframe(X_recon, use_container_width=True)

            # Show summary
            st.caption(f"Rows: {len(X_recon)} | Columns: {len(X_recon.columns)}")
            if non_numeric_cols:
                st.caption(f"Metadata columns: {', '.join(non_numeric_cols)}")

        elif preview_option == "Original Data (with NaN)":
            st.dataframe(original_data, use_container_width=True)
            st.caption(f"Rows: {len(original_data)} | Columns: {len(original_data.columns)}")

        else:  # Comparison
            st.markdown("**Side-by-side comparison (numeric columns only)**")
            col_a, col_b = st.columns(2)

            with col_a:
                st.markdown("*Original (with NaN) - Numeric only*")
                orig_numeric = original_data[numeric_cols].head(10)
                st.dataframe(orig_numeric, use_container_width=True)

            with col_b:
                st.markdown("*Reconstructed - Numeric only*")
                recon_numeric = X_recon[numeric_cols].head(10)
                st.dataframe(recon_numeric, use_container_width=True)


# ============================================================================

if __name__ == "__main__":
    # For testing this module standalone
    st.set_page_config(
        page_title="PCA Analysis - ChemometricSolutions",
        page_icon="🎯",
        layout="wide"
    )
    
    # Create dummy data for testing
    if 'current_data' not in st.session_state:
        np.random.seed(42)
        test_data = pd.DataFrame(
            np.random.randn(100, 20),
            columns=[f'Var{i+1}' for i in range(20)]
        )
        test_data.insert(0, 'SampleID', [f'S{i+1}' for i in range(100)])
        st.session_state['current_data'] = test_data
    
    show()

