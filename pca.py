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
from pca_utils.pca_calculations import compute_pca, varimax_rotation

# Try to import plotting and statistics modules (optional - from pca_utils)
try:
    from pca_utils.pca_plots import (
        plot_scores, plot_loadings, plot_scree,
        plot_cumulative_variance, plot_biplot, plot_loadings_line,
        add_convex_hulls
    )
    PLOTS_AVAILABLE = True
except ImportError:
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
        create_contribution_plot_all_vars,
        create_correlation_scatter
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


def show():
    """
    Main PCA Analysis page.
    
    Displays a multi-tab interface for complete PCA workflow:
    1. Model Computation
    2. Variance Plots
    3. Loadings Plots
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
    st.markdown("*Equivalent to R PCA_model_PCA.r*")

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
    
    n_components = st.slider(
        "Number of components to compute:",
        min_value=2,
        max_value=min(20, max_components),
        value=default_n,
        help="NIPALS computes only requested components (efficient for large datasets)"
    )
    
    st.info(f"⚡ NIPALS will compute **{n_components}** components (max possible: {max_components})")
    
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
                
                # Call NIPALS PCA
                pca_dict = compute_pca(
                    X=numeric_data,
                    n_components=n_components,
                    center=center_data,
                    scale=scale_data
                )
                
                elapsed = time.time() - start_time
                
                # Store results in session state
                st.session_state['pca_results'] = {
                    **pca_dict,
                    'method': 'Standard PCA',
                    'selected_vars': numeric_data.columns.tolist(),
                    'computation_time': elapsed,
                    'varimax_applied': False,
                    'original_data': numeric_data
                }
                
                st.success(f"✅ PCA computation completed in {elapsed:.2f} seconds!")

                # DEBUG: Display variance calculation details
                with st.expander("🔍 DEBUG: R-CAT Variance Calculation Details"):
                    st.markdown("#### R-CAT Formula: R² = λ / (total_var × (n-1))")

                    col_d1, col_d2, col_d3 = st.columns(3)
                    with col_d1:
                        st.metric("Total Variance (original)", f"{pca_dict['total_variance']:.4f}")
                    with col_d2:
                        st.metric("n_samples", pca_dict['n_samples'])
                    with col_d3:
                        total_ss = pca_dict['total_variance'] * (pca_dict['n_samples'] - 1)
                        st.metric("total_ss", f"{total_ss:.4f}")

                    st.markdown("#### Eigenvalues (λ = t't):")
                    st.write(pca_dict['eigenvalues'])

                    st.markdown("#### Variance Ratio Calculation:")
                    debug_df = pd.DataFrame({
                        'Component': [f'PC{i+1}' for i in range(n_components)],
                        'Eigenvalue (λ)': pca_dict['eigenvalues'],
                        'total_ss': [total_ss] * n_components,
                        'Ratio (λ/total_ss)': pca_dict['explained_variance_ratio'],
                        'Ratio %': pca_dict['explained_variance_ratio'] * 100,
                        'Cumulative %': pca_dict['cumulative_variance'] * 100
                    })
                    st.dataframe(debug_df.style.format({
                        'Eigenvalue (λ)': '{:.4f}',
                        'total_ss': '{:.4f}',
                        'Ratio (λ/total_ss)': '{:.6f}',
                        'Ratio %': '{:.2f}',
                        'Cumulative %': '{:.2f}'
                    }), use_container_width=True, hide_index=True)

                    final_cumul = pca_dict['cumulative_variance'][-1] * 100
                    if final_cumul < 95:
                        st.warning(f"⚠️ Cumulative variance is {final_cumul:.2f}% (computing only {n_components} out of {min(len(numeric_data), len(numeric_data.columns))-1} possible components)")
                    else:
                        st.success(f"✅ Cumulative variance: {final_cumul:.2f}%")

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
                    'Eigenvalue': pca_dict['eigenvalues'],
                    'Variance %': pca_dict['explained_variance_ratio'] * 100,
                    'Cumulative %': pca_dict['cumulative_variance'] * 100,
                    'Iterations': pca_dict['n_iterations']
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
                
                st.info("👉 Go to other tabs to visualize scores, loadings, and diagnostics")
        
        except Exception as e:
            st.error(f"❌ PCA computation failed: {str(e)}")
            st.exception(e)
    
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
                        # Extract loadings for rotation
                        loadings_subset = loadings.iloc[:, :n_factors]
                        
                        # Apply rotation
                        rotated_loadings, iterations = varimax_rotation(loadings_subset)
                        
                        # Calculate rotated scores
                        X_data = pca_results['original_data']
                        if pca_results['centering']:
                            X_data = X_data - pca_results['means']
                        if pca_results['scaling']:
                            X_data = X_data / pca_results['stds']
                        
                        rotated_scores = X_data @ rotated_loadings
                        
                        # Rename to Factor instead of PC
                        factor_names = [f'Factor{i+1}' for i in range(n_factors)]
                        rotated_loadings.columns = factor_names
                        rotated_scores.columns = factor_names
                        
                        # Update session state
                        st.session_state['pca_results'].update({
                            'method': 'Varimax Rotation',
                            'scores': rotated_scores,
                            'loadings': rotated_loadings,
                            'varimax_applied': True,
                            'varimax_iterations': iterations,
                            'n_components': n_factors
                        })
                        
                        st.success(f"✅ Varimax rotation completed in {iterations} iterations!")
                        st.info("♻️ Scores and loadings updated. Check other tabs to see rotated results.")
                        st.rerun()
                
                except Exception as e:
                    st.error(f"❌ Varimax rotation failed: {str(e)}")


# ============================================================================
# TAB 2: VARIANCE PLOTS (to be updated - from pca_OLD.py: copy contents)
# ============================================================================

def _show_variance_plots_tab():
    """Display advanced variance plots with multiple visualization options."""
    st.markdown("## 📊 Variance Plots")
    st.markdown("*Equivalent to R PCA_variance_plot.r and PCA_cumulative_var_plot.r*")

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
        st.markdown(f"### 🎯 Variable Contribution Analysis")
        st.markdown("*Based on significant components identified from Scree Plot*")

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

        # Step 2: Calculate weighted contributions
        st.markdown(f"#### Step 2: Variable Contributions (first {n_significant} {comp_label.lower()}s)")

        # Use calculate_contributions from pca_utils.pca_statistics
        contrib_df = calculate_contributions(
            pca_results['loadings'],
            pca_results['explained_variance_ratio'],
            n_components=n_significant,
            normalize=True
        )

        # Extract data for plotting
        contributions_pct = contrib_df['Contribution_%'].values
        var_names = contrib_df['Variable'].values

        # Sort for plot
        sorted_idx = np.argsort(contributions_pct)[::-1]
        sorted_vars = var_names[sorted_idx]
        sorted_contributions = contributions_pct[sorted_idx]

        # Create plot
        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=sorted_vars,
            y=sorted_contributions,
            name='Variable Contribution',
            marker_color='darkgreen',
            text=[f'{val:.1f}%' for val in sorted_contributions],
            textposition='outside'
        ))

        total_var_explained = pca_results['explained_variance_ratio'][:n_significant].sum() * 100

        fig.update_layout(
            title=f"Variable Contributions to Total Explained Variance<br>({n_significant} significant {comp_label.lower()}s: {total_var_explained:.1f}% total variance)",
            xaxis_title="Variables",
            yaxis_title="Contribution (%)",
            height=600,
            xaxis={'tickangle': 45}
        )

        st.plotly_chart(fig, use_container_width=True, key="contribution_plot")

        # Step 3: Detailed table
        st.markdown("#### Step 3: Detailed Contribution Table")

        # Sort contributions for display
        contrib_df_sorted = contrib_df.sort_values('Contribution_%', ascending=False)

        st.dataframe(contrib_df_sorted.round(2), use_container_width=True)

        # Interpretation
        st.markdown("#### 📋 Interpretation")
        top_vars = contrib_df_sorted.head(3)['Variable'].tolist()
        total_explained = pca_results['explained_variance_ratio'][:n_significant].sum() * 100

        st.success(f"""
        **Key Findings:**
        - **{n_significant} significant components** explain **{total_explained:.1f}%** of total variance
        - **Top 3 contributing variables**: {', '.join(top_vars)}
        - **Top variable ({top_vars[0]})** contributes **{contrib_df_sorted.iloc[0]['Contribution_%']:.1f}%** to the explained variance
        """)

        if n_significant >= 2:
            pc_names = pca_results['loadings'].columns[:n_significant].tolist()
            var_ratios = pca_results['explained_variance_ratio'][:n_significant]
            st.info(f"""
            **Contribution Breakdown:**
            - {pc_names[0]} explains {var_ratios[0]*100:.1f}% of total variance
            - {pc_names[1]} explains {var_ratios[1]*100:.1f}% of total variance
            """)


# ============================================================================
# TAB 3: LOADINGS PLOTS (to be updated - from pca_OLD.py: copy contents)
# ============================================================================

def _show_loadings_plots_tab():
    """Display loadings plots with multiple visualization options."""
    st.markdown("## 📈 Loadings Plots")
    st.markdown("*Equivalent to R PCA_plots_loadings.r*")

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

        # Use plot_loadings from pca_utils.pca_plots
        fig = plot_loadings(
            loadings,
            pc_x,
            pc_y,
            pca_results['explained_variance_ratio'],
            is_varimax=is_varimax,
            color_by_magnitude=is_varimax
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

    # === LOADING LINE PLOT ===
    elif loading_plot_type == "📈 Loading Line Plot":
        st.markdown(f"### 📈 Loading Line Plot{title_suffix}")

        selected_comps = st.multiselect(
            f"Select {'factors' if is_varimax else 'components'} to display:",
            loadings.columns.tolist(),
            default=loadings.columns[:min(3, len(loadings.columns))].tolist(),
            key="loading_line_components"
        )

        if selected_comps:
            # Use plot_loadings_line from pca_utils.pca_plots
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
    st.markdown("*Equivalent to R PCA_plots_scores.r*")

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

        # Prepare color data and text labels
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

        if color_by == "None":
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

        # UPDATE LAYOUT with equal scale and lasso selection
        fig.update_layout(
            height=600,
            width=600,  # FORCE SQUARE
            template='plotly_white',
            dragmode='lasso',  # Enable lasso selection by default
            xaxis=dict(range=axis_range, scaleanchor="y", scaleratio=1, constrain="domain"),
            yaxis=dict(range=axis_range, scaleanchor="x", scaleratio=1, constrain="domain")
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

            # Add color variable value if coloring is active
            if color_by_3d != "None" and color_data_3d is not None:
                try:
                    # Format labels with color variable value
                    if hasattr(color_data_3d, 'dtype') and pd.api.types.is_numeric_dtype(color_data_3d):
                        # Numeric color variable - show with 2 decimals
                        text_param_3d = [f"{base_labels_3d[i]}<br>{color_by_3d}: {color_data_3d.iloc[i]:.2f}"
                                        for i in range(len(base_labels_3d))]
                    else:
                        # Show only the label values
                        text_param_3d = base_labels_3d
                except:
                    # Fallback to base labels only
                    text_param_3d = base_labels_3d
            else:
                # No coloring - show base labels only
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

        # Update layout
        fig_3d.update_layout(
            height=700,
            template='plotly_white',
            scene=dict(
                xaxis_title=f'{pc_x} ({var_x:.1f}%)',
                yaxis_title=f'{pc_y} ({var_y:.1f}%)',
                zaxis_title=f'{pc_z} ({var_z:.1f}%)'
            )
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
        from pca_utils.pca_plots import plot_line_scores

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

def _show_interpretation_tab():
    """Display component/factor interpretation."""
    st.markdown("## 📝 Component Interpretation")
    
    if 'pca_results' not in st.session_state:
        st.warning("⚠️ Please compute PCA in **Model Computation** tab first")
        return
    
    pca_results = st.session_state['pca_results']
    is_varimax = pca_results.get('varimax_applied', False)
    
    st.info("🔮 AI-powered interpretation coming soon!")
    
    st.markdown("""
    ### Manual Interpretation Guide
    
    **For Standard PCA:**
    1. Look at the scree plot to determine significant components
    2. Examine loadings to understand which variables contribute to each PC
    3. Interpret scores to identify sample patterns and clusters
    4. Check diagnostics for outliers and model quality
    
    **For Varimax-Rotated Factors:**
    1. Rotated factors have simpler structure (easier interpretation)
    2. Each variable typically loads highly on fewer factors
    3. Look for factor themes based on high-loading variables
    4. Factors remain orthogonal (uncorrelated)
    
    👉 Check the **Loadings** and **Scores** tabs for detailed visualizations
    """)


# ============================================================================
# TAB 6: ADVANCED DIAGNOSTICS
# ============================================================================

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

    # Get original data
    data = st.session_state.get('current_data', None)
    if data is None:
        st.error("❌ Original data not available")
        return

    # CHECK for missing values
    has_missing_values = st.session_state.current_data.isna().sum().sum() > 0

    if has_missing_values:
        st.warning("⚠️ Dataset contains missing values - Q statistic cannot be calculated. Only T² plots are available.")

    st.divider()

    # Get numeric variables used in PCA
    selected_vars = pca_results.get('selected_vars', data.select_dtypes(include=[np.number]).columns.tolist())
    X_data = data[selected_vars]
    n_samples = len(X_data)
    n_variables = len(selected_vars)

    # === SECTION 1 - CONFIGURATION ===
    st.markdown("### ⚙️ Diagnostic Configuration")

    col1, col2, col3, col4 = st.columns(4)

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
            help="Independent: each statistic separately; Joint: combined box regions"
        )

    with col3:
        # Color by options
        available_cols = ["None"] + list(data.columns) + ["Index"]
        color_by_diag = st.selectbox(
            "Color Points By:",
            available_cols,
            key="diag_color"
        )

    with col4:
        show_sample_names_diag = st.checkbox(
            "Show Sample Labels",
            value=False,
            key="diag_labels"
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
            sample1_t2_components = sample1_scores**2 / eigenvalues_diag
            contrib_df = pd.DataFrame({
                'Component': [f'PC{i+1}' for i in range(n_comp_diag)],
                'Score': sample1_scores,
                'Score²': sample1_scores**2,
                'Eigenvalue (λ)': eigenvalues_diag,
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
        n_samples = len(X_centered)
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

        # === SECTION 3 - PREPARE COLOR DATA ===
        color_data_diag = None
        if color_by_diag != "None":
            if color_by_diag == "Index":
                color_data_diag = list(range(len(scores)))
            else:
                try:
                    color_data_diag = data[color_by_diag].values
                except:
                    color_data_diag = None

        # === SECTION 4 - CREATE PLOTS (SIDE-BY-SIDE) ===
        st.markdown("---")

        col1, col2 = st.columns([3, 1])

        with col1:
            st.markdown("### 📊 Diagnostic Plots")

        with col2:
            show_trajectory = st.checkbox("Show trajectory", value=True, key="show_traj_diag")

        st.markdown("Boxes define acceptancy regions at 97.5%, 99.5%, 99.95% limits")

        col_left, col_right = st.columns(2)

        # Import plotting functions
        from pca_monitoring_page import create_score_plot, create_t2_q_plot

        with col_left:
            st.subheader("Score Plot with T² Ellipses")

            # Prepare params for score plot
            pca_params_plot = {
                'n_samples_train': n_samples,
                'n_features': n_variables
            }

            # PASS show_trajectory parameter to create_score_plot
            fig_score = create_score_plot(
                scores_diag,
                pca_results['explained_variance_ratio'][:n_comp_diag] * 100,
                timestamps=None,
                pca_params=pca_params_plot,
                start_sample_num=1,
                show_trajectory=show_trajectory
            )

            st.plotly_chart(fig_score, use_container_width=True, key="diag_score_plot")

        with col_right:
            if not has_missing_values:
                st.subheader("T² vs Q Influence Plot")

                # Convert limits to list format for plot
                t2_limits_list = [t2_limits[label] for label in conf_labels]
                q_limits_list = [q_limits[label] for label in conf_labels]

                fig_t2q = create_t2_q_plot(
                    t2_values,
                    q_values,
                    t2_limits_list,
                    q_limits_list,
                    timestamps=None,
                    start_sample_num=1,
                    show_trajectory=show_trajectory
                )

                st.plotly_chart(fig_t2q, use_container_width=True, key="diag_t2q_plot")
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

                # Get contributions
                sample_score = scores_diag[sample_idx]
                sample_t2_contrib = (sample_score**2 / eigenvalues_diag)

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

                    # Bar plots side by side (ALL variables, red if |contrib|>1, blue otherwise)
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

                    # Table: Variables where |contrib|>1 with real values vs training mean
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

                    # Filter variables where |contrib|>1 for either T² or Q
                    high_contrib_t2 = np.abs(t2_contrib_norm) > 1.0
                    high_contrib_q = np.abs(q_contrib_norm) > 1.0
                    high_contrib = high_contrib_t2 | high_contrib_q

                    if high_contrib.sum() > 0:
                        contrib_table_data = []
                        for i, var in enumerate(selected_vars):
                            if high_contrib[i]:
                                real_val = sample_values[var]
                                mean_val = training_mean[var]
                                diff = real_val - mean_val
                                direction = "Higher ↑" if diff > 0 else "Lower ↓"

                                contrib_table_data.append({
                                    'Variable': var,
                                    'Real Value': f"{real_val:.3f}",
                                    'Training Mean': f"{mean_val:.3f}",
                                    'Difference': f"{diff:.3f}",
                                    'Direction': direction,
                                    '|T² Contrib|': f"{abs(t2_contrib_norm[i]):.2f}",
                                    '|Q Contrib|': f"{abs(q_contrib_norm[i]):.2f}"
                                })

                        contrib_table = pd.DataFrame(contrib_table_data)
                        # Sort by max absolute contribution
                        contrib_table['Max_Contrib'] = contrib_table.apply(
                            lambda row: max(float(row['|T² Contrib|']), float(row['|Q Contrib|'])),
                            axis=1
                        )
                        contrib_table = contrib_table.sort_values('Max_Contrib', ascending=False).drop('Max_Contrib', axis=1)

                        st.dataframe(contrib_table, use_container_width=True, hide_index=True)
                    else:
                        st.info("No variables exceed the 95th percentile threshold.")

                    # Correlation scatter: training (grey), sample (red star)
                    st.markdown("### 📈 Correlation Analysis - Top Q Contributor")
                    st.markdown("*Select from top Q contributors to see correlation with most correlated variable*")

                    # Get top Q contributors (variables with highest |Q contribution|)
                    q_contrib_abs = np.abs(q_contrib_norm)
                    top_q_indices = np.argsort(q_contrib_abs)[::-1][:5]
                    top_q_contributors = [selected_vars[i] for i in top_q_indices]

                    # Dropdown to select from top Q contributors
                    corr_col1, corr_col2 = st.columns([2, 1])

                    with corr_col1:
                        selected_q_var = st.selectbox(
                            "Select from top Q contributors:",
                            options=top_q_contributors,
                            key="diag_top_q_var"
                        )

                    # Calculate correlations for selected variable (from training data)
                    var1_idx = selected_vars.index(selected_q_var)
                    X_data_array = X_data_df.values
                    correlations = {}
                    for i, var in enumerate(selected_vars):
                        if var != selected_q_var:
                            corr = np.corrcoef(X_data_array[:, var1_idx], X_data_array[:, i])[0, 1]
                            correlations[var] = (corr, i)

                    # Find most correlated variable
                    most_corr_var = max(correlations, key=lambda k: abs(correlations[k][0]))
                    corr_coef, var2_idx = correlations[most_corr_var]

                    with corr_col2:
                        st.metric("Correlation (training)", f"{corr_coef:.4f}")

                    # Create scatter plot (training=grey, sample=red star)
                    fig_corr_scatter = create_correlation_scatter(
                        X_train=X_data_array,
                        X_test=X_data_array,  # Same as training in pca.py diagnostics
                        X_sample=X_data_array[sample_idx, :],
                        var1_idx=var1_idx,
                        var2_idx=var2_idx,
                        var1_name=selected_q_var,
                        var2_name=most_corr_var,
                        correlation_val=corr_coef,
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
    st.markdown("*Equivalent to R PCA_extract.r*")
    
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
    """Display the Missing Data Reconstruction tab using PCA."""
    st.markdown("## 🔄 Missing Data Reconstruction using PCA")
    st.info("Reconstruct missing values using PCA scores and loadings")

    if not MISSING_DATA_AVAILABLE:
        st.warning("⚠️ Missing data reconstruction module not available")
        st.info("Please ensure missing_data_reconstruction.py is in the project directory")
        return

    # STEP 1: Check if data has missing values
    if 'current_data' not in st.session_state:
        st.warning("No data loaded")
        return

    data = st.session_state.current_data
    n_missing, n_total, pct_missing = count_missing_values(data)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🔍 Missing Values", n_missing)
    with col2:
        st.metric("📊 Total Cells", n_total)
    with col3:
        st.metric("📈 Percentage", f"{pct_missing:.2f}%")

    if n_missing == 0:
        st.success("✅ No missing values - reconstruction not needed")
    else:
        st.divider()

        # STEP 2: PCA Configuration for reconstruction
        st.markdown("### 🎯 PCA Configuration")

        col1, col2 = st.columns(2)
        with col1:
            n_comp_reconstruction = st.slider(
                "Number of components for reconstruction:",
                min_value=2,
                max_value=min(10, data.shape[1]),
                value=min(5, data.shape[1]),
                key="n_comp_missing"
            )

        with col2:
            center_missing = st.checkbox("Center data", value=True, key="center_missing")
            scale_missing = st.checkbox("Scale data", value=False, key="scale_missing")

        st.divider()

        # STEP 3: Run PCA on data with missing values (NIPALS handles NaN)
        if st.button("🚀 Run PCA for Reconstruction", key="run_pca_missing"):
            with st.spinner("Computing PCA..."):
                from pca_utils.pca_calculations import RnipalsPca_exact

                # Compute PCA (NIPALS handles missing values)
                pca_results = RnipalsPca_exact(
                    data,
                    nPcs=n_comp_reconstruction,
                    center=center_missing,
                    scale=scale_missing
                )

                st.session_state.pca_missing = pca_results
                st.success(f"✅ PCA computed with {n_comp_reconstruction} components")

        st.divider()

        # STEP 4: Reconstruct missing data
        if 'pca_missing' in st.session_state:
            st.markdown("### 🔄 Reconstruct Missing Values")

            pca_res = st.session_state.pca_missing
            scores = pca_res['scores']
            loadings = pca_res['loadings']

            if st.button("✨ Reconstruct Missing Data"):
                with st.spinner("Reconstructing..."):
                    # Reconstruct
                    X_reconstructed = reconstruct_missing_data(
                        data,
                        scores,
                        loadings,
                        n_components=n_comp_reconstruction
                    )

                    # Get stats
                    info = get_reconstruction_info(data, X_reconstructed)

                    st.session_state.X_reconstructed = X_reconstructed

                    # Display reconstruction stats
                    st.markdown("### 📊 Reconstruction Statistics")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("✅ Values Filled", info['n_filled'])
                    with col2:
                        st.metric("📈 Mean Filled", f"{info['filled_mean']:.3f}")
                    with col3:
                        st.metric("📉 Std Filled", f"{info['filled_std']:.3f}")

                    st.success("✅ Reconstruction complete!")

            st.divider()

            # STEP 5: Export & Load to Workspace
            if 'X_reconstructed' in st.session_state:
                st.markdown("### 💾 Export Reconstructed Data")

                X_recon = st.session_state.X_reconstructed

                # Name for export
                base_name = st.session_state.get('dataset_name', 'dataset')
                export_name = st.text_input(
                    "Export name:",
                    value=f"{base_name}_reconstructed",
                    key="export_name_missing"
                )

                col1, col2 = st.columns(2)

                with col1:
                    if st.button("📥 Load to Workspace", key="load_workspace_missing"):
                        try:
                            from pca_utils.data_workspace import save_original_to_history
                            workspace_available = True
                        except ImportError:
                            workspace_available = False

                        # Save to workspace
                        workspace_path = Path('data') / f"{export_name}.xlsx"
                        workspace_path.parent.mkdir(parents=True, exist_ok=True)
                        X_recon.to_excel(workspace_path, index=True, sheet_name='reconstructed')

                        # UPDATE session state
                        st.session_state.current_data = X_recon
                        st.session_state.dataset_name = export_name

                        # SAVE to transformation history (data_workspace)
                        if workspace_available:
                            save_original_to_history(X_recon, export_name)

                        st.success(f"✅ Loaded to workspace: {export_name}")
                        st.rerun()

                with col2:
                    # Download Excel
                    excel_bytes = io.BytesIO()
                    X_recon.to_excel(excel_bytes, index=True, sheet_name='reconstructed')
                    excel_bytes.seek(0)

                    st.download_button(
                        "📥 Download Excel",
                        excel_bytes,
                        f"{export_name}.xlsx",
                        "application/vnd.ms-excel",
                        key="download_missing_excel"
                    )


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

