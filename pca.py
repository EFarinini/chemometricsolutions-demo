"""
CAT PCA Analysis Page
Equivalent to PCA_* R scripts for Principal Component Analysis
Enhanced with Varimax rotation and advanced diagnostics
"""

import streamlit as st
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import os
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go

# Import PCA utilities (modular package)
from pca_utils.pca_calculations import compute_pca, varimax_rotation
from pca_utils.pca_plots import (plot_scores, plot_loadings, plot_biplot, plot_scree,
                                  plot_cumulative_variance, plot_loadings_line, add_convex_hulls)
from pca_utils.pca_statistics import (calculate_hotelling_t2, calculate_q_residuals,
                                       calculate_contributions, calculate_leverage,
                                       cross_validate_pca)
from pca_utils.pca_workspace import (save_workspace_to_file, load_workspace_from_file,
                                      save_dataset_split, get_split_datasets_info,
                                      delete_split_dataset, clear_all_split_datasets)

# Try to import advanced diagnostics module
try:
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(parent_dir)
    from pca_diagnostics_complete import show_advanced_diagnostics_tab
    DIAGNOSTICS_AVAILABLE = True
except ImportError as e:
    DIAGNOSTICS_AVAILABLE = False
    print(f"Advanced diagnostics not available: {e}")

# Import color utilities
from color_utils import get_unified_color_schemes, create_categorical_color_map

def show():
    """Display the PCA Analysis page"""
    
    st.markdown("# 🎯 Principal Component Analysis (PCA)")
    
    if 'current_data' not in st.session_state:
        st.warning("⚠️ No data loaded. Please go to Data Handling to load your dataset first.")
        return
    
    data = st.session_state.current_data
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "🔧 Model Computation",
        "📊 Variance Plots", 
        "📈 Loadings Plots",
        "🎯 Score Plots",
        "🔍 Diagnostics",
        "👤 Extract & Export",
        "🔬 Advanced Diagnostics"
    ])

    # ===== MODEL COMPUTATION TAB =====
    with tab1:
        st.markdown("## 🔧 PCA Model Computation")
        st.markdown("*Equivalent to PCA_model_PCA.r and PCA_model_varimax.r*")
        
        # === DATA OVERVIEW SECTION ===
        st.markdown("### 📊 Dataset Overview")
        
        # Dataset info banner
        total_cols = len(data.columns)
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        non_numeric_columns = data.select_dtypes(exclude=[np.number]).columns.tolist()
        
        # Create info banner similar to transformation module
        col_info1, col_info2 = st.columns(2)
        
        with col_info1:
            st.info(f"📋 **Dataset**: {len(data)} samples, {total_cols} total columns")
        
        with col_info2:
            st.info(f"🔢 **Numeric**: {len(numeric_columns)} variables, **Non-numeric**: {len(non_numeric_columns)} variables")
        
        if len(numeric_columns) > 100:
            st.success(f"🔬 **Spectral data detected**: {len(numeric_columns)} variables")
        
        # === DATA PREVIEW SECTION ===
        st.markdown("### 👁️ Data Preview")
        
        # Preview options
        col_prev1, col_prev2, col_prev3 = st.columns(3)
        
        with col_prev1:
            preview_mode = st.selectbox(
                "Preview mode:",
                ["First 10 rows", "Random 10 rows", "Statistical summary"]
            )
        
        with col_prev2:
            show_all_columns = st.checkbox("Show all columns", value=False)
        
        with col_prev3:
            if not show_all_columns:
                max_cols_preview = st.number_input("Max columns to display:", 5, 50, 20)
        
        # Generate preview
        if preview_mode == "Statistical summary":
            if show_all_columns:
                preview_data = data.describe()
            else:
                preview_cols = data.columns[:max_cols_preview] if not show_all_columns else data.columns
                preview_data = data[preview_cols].describe()
        else:
            if preview_mode == "Random 10 rows":
                sample_data = data.sample(min(10, len(data))) if len(data) > 10 else data
            else:  # First 10 rows
                sample_data = data.head(10)
            
            if show_all_columns:
                preview_data = sample_data
            else:
                preview_cols = data.columns[:max_cols_preview] if not show_all_columns else data.columns
                preview_data = sample_data[preview_cols]
        
        # Display preview
        st.dataframe(preview_data, use_container_width=True, height=300)
        
        if not show_all_columns and len(data.columns) > max_cols_preview:
            st.caption(f"Showing {max_cols_preview} of {len(data.columns)} columns. Enable 'Show all columns' to see everything.")
        
        # === VARIABLE SELECTION SECTION ===
        st.markdown("### 🎯 Variable Selection for PCA")
        
        if not numeric_columns:
            st.error("❌ No numeric columns found in the dataset!")
            return
        
        # Column classification
        col_class1, col_class2 = st.columns(2)
        
        with col_class1:
            st.markdown("#### 📋 Column Classification")
            
            # Auto-detect metadata columns
            metadata_candidates = []
            for col in data.columns:
                col_str = str(col).lower()
                # Existing keywords
                basic_keywords = ['id', 'name', 'label', 'class', 'group', 'sample', 'date', 'time']
                # Additional patterns for analytical chemistry
                analytical_patterns = [
                    '%', 'percent', 'concentration', 'content', 'purity', 'composition',
                    'w/w', 'v/v', 'ppm', 'mg/kg', 'mg/l', 'g/kg', 'wt%'
                ]
                
                if (any(keyword in col_str for keyword in basic_keywords) or 
                    any(pattern in col_str for pattern in analytical_patterns)):
                    metadata_candidates.append(col)
            
            if metadata_candidates:
                st.info(f"🏷️ **Potential metadata columns detected**: {', '.join(metadata_candidates[:5])}")
                if len(metadata_candidates) > 5:
                    st.caption(f"... and {len(metadata_candidates) - 5} more")
            
            # Display column types
            if non_numeric_columns:
                st.markdown("**Non-numeric columns:**")
                for col in non_numeric_columns[:10]:  # Show first 10
                    st.write(f"• {col}")
                if len(non_numeric_columns) > 10:
                    st.caption(f"... and {len(non_numeric_columns) - 10} more")
        
        with col_class2:
            st.markdown("#### 🔢 Numeric Variables Available")
            st.write(f"**Total numeric variables**: {len(numeric_columns)}")
            
            # Show sample of numeric columns
            if len(numeric_columns) <= 20:
                st.write("**All numeric columns:**")
                for i, col in enumerate(numeric_columns):
                    st.write(f"{i+1}. {col}")
            else:
                st.write("**Sample of numeric columns:**")
                for i, col in enumerate(numeric_columns[:10]):
                    st.write(f"{i+1}. {col}")
                st.caption(f"... and {len(numeric_columns) - 10} more")
        
        # === VARIABLE SELECTION INTERFACE ===
        st.markdown("#### 🎛️ Select Variables for PCA Analysis")
        
        # Selection method for large datasets
        if len(numeric_columns) > 50:
            st.markdown("##### Variable Selection Method")
            selection_method = st.selectbox(
                "Choose selection method:",
                ["Select All Numeric", "Range Selection", "Manual Selection", "Exclude Metadata Only"]
            )
            
            if selection_method == "Select All Numeric":
                selected_vars = numeric_columns
                st.success(f"✅ Selected all {len(selected_vars)} numeric variables")
                
            elif selection_method == "Range Selection":
                st.markdown("**Range-based selection:**")
                col_range1, col_range2 = st.columns(2)
                
                with col_range1:
                    start_idx = st.number_input("Start variable index (1-based):", 1, len(numeric_columns), 1)
                with col_range2:
                    end_idx = st.number_input("End variable index (1-based):", start_idx, len(numeric_columns), min(50, len(numeric_columns)))
                
                selected_vars = numeric_columns[start_idx-1:end_idx]
                st.info(f"📊 Selected variables {start_idx} to {end_idx}: **{len(selected_vars)}** variables")
                
                # Show selected range
                if len(selected_vars) <= 10:
                    st.write("**Selected variables:**", ", ".join(selected_vars))
                else:
                    st.write("**Selected variables:**", ", ".join(selected_vars[:5]) + f" ... (+{len(selected_vars)-5} more)")
                
            elif selection_method == "Exclude Metadata Only":
                excluded_cols = st.multiselect(
                    "Select columns to EXCLUDE from PCA:",
                    data.columns.tolist(),
                    default=metadata_candidates + non_numeric_columns,
                    help="Metadata and non-numeric columns are pre-selected for exclusion"
                )
                
                selected_vars = [col for col in numeric_columns if col not in excluded_cols]
                st.success(f"✅ Selected {len(selected_vars)} variables (excluded {len(excluded_cols)} columns)")
                
            else:  # Manual Selection
                st.markdown("**Manual variable selection:**")
                selected_vars = st.multiselect(
                    "Choose specific variables for PCA:",
                    numeric_columns,
                    default=numeric_columns[:min(10, len(numeric_columns))],
                    help="Select the variables you want to include in the PCA analysis"
                )
        
        else:
            # Simple selection for smaller datasets
            st.markdown("##### Choose Variables")
            
            col_sel1, col_sel2 = st.columns(2)
            
            with col_sel1:
                select_all_numeric = st.checkbox("Select all numeric variables", value=True)
            
            with col_sel2:
                if not select_all_numeric:
                    exclude_metadata = st.checkbox("Auto-exclude metadata columns", value=True)
            
            if select_all_numeric:
                selected_vars = numeric_columns
            else:
                # Manual selection
                if exclude_metadata and metadata_candidates:
                    default_vars = [col for col in numeric_columns if col not in metadata_candidates]
                else:
                    default_vars = numeric_columns[:min(10, len(numeric_columns))]
                
                selected_vars = st.multiselect(
                    "Select variables for PCA:",
                    numeric_columns,
                    default=default_vars,
                    key="pca_variable_selection_manual"
                )
        
        # Validation
        if not selected_vars:
            st.warning("⚠️ Please select at least 2 variables for PCA analysis")
            return
        
        # Show final selection summary
        st.markdown("#### 📋 Final Selection Summary")
        col_summary1, col_summary2, col_summary3 = st.columns(3)
        
        with col_summary1:
            st.metric("Selected Variables", len(selected_vars))
        with col_summary2:
            st.metric("Total Samples", len(data))
        with col_summary3:
            if len(selected_vars) > 0:
                missing_pct = (data[selected_vars].isnull().sum().sum() / (len(data) * len(selected_vars))) * 100
                st.metric("Missing Data %", f"{missing_pct:.1f}%")
        
        # === OBJECT SELECTION ===
        st.markdown("### 🎯 Object (Sample) Selection")
        
        col_obj1, col_obj2 = st.columns(2)
        
        with col_obj1:
            use_all_objects = st.checkbox("Use all objects", value=True)
            
        with col_obj2:
            if not use_all_objects:
                n_objects = st.slider("Number of objects:", 5, len(data), len(data))
                selected_data = data[selected_vars].iloc[:n_objects]
                st.info(f"Using first {n_objects} samples")
            else:
                selected_data = data[selected_vars]
        
        # === PCA PARAMETERS ===
        st.markdown("### ⚙️ PCA Parameters")
        
        col_param1, col_param2 = st.columns(2)
        
        with col_param1:
            st.markdown("**Preprocessing Options**")
            center_data = st.checkbox("Center data", value=True, help="Remove column means")
            scale_data = st.checkbox("Scale data (unit variance)", value=True, help="Standardize to unit variance")
            
            st.markdown("**Analysis Method**")
            pca_method = st.selectbox("PCA Method:", ["Standard PCA", "Varimax Rotation"])
            
        with col_param2:
            st.markdown("**Model Parameters**")
            max_components = min(selected_data.shape) - 1
            n_components = st.slider("Number of components:", 2, max_components, 
                                   min(5, max_components))
            
            # Missing values handling
            missing_values = st.selectbox("Missing values:", ["Remove", "Impute mean"])
            
            # Advanced options
            with st.expander("🔧 Advanced Options"):
                if pca_method == "Varimax Rotation":
                    max_iter_varimax = st.number_input("Max Varimax iterations:", 50, 500, 100)
                    tolerance_varimax = st.number_input("Convergence tolerance:", 1e-8, 1e-3, 1e-6, format="%.0e")
                
                perform_validation = st.checkbox("Perform cross-validation", value=False)
                if perform_validation:
                    cv_folds = st.slider("CV folds:", 3, 10, 5)
        
        # === FINAL SUMMARY BEFORE COMPUTATION ===
        if selected_vars:
            st.markdown("### 📊 Pre-Analysis Summary")
            
            # Create summary info
            summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
            
            with summary_col1:
                st.metric("Variables", len(selected_vars))
            with summary_col2:
                st.metric("Samples", len(selected_data))
            with summary_col3:
                st.metric("Components", n_components)
            with summary_col4:
                preprocessing = []
                if center_data: preprocessing.append("Center")
                if scale_data: preprocessing.append("Scale")
                st.write(f"**Preprocessing**: {', '.join(preprocessing) if preprocessing else 'None'}")
            
            # Show selected variables preview
            with st.expander("🔍 Preview Selected Variables"):
                if len(selected_vars) <= 20:
                    st.write("**Selected variables:**")
                    cols_per_row = 4
                    for i in range(0, len(selected_vars), cols_per_row):
                        row_cols = st.columns(cols_per_row)
                        for j, col in enumerate(selected_vars[i:i+cols_per_row]):
                            with row_cols[j]:
                                st.write(f"• {col}")
                else:
                    st.write(f"**Selected variables** ({len(selected_vars)} total):")
                    st.write(", ".join(map(str, selected_vars[:10])) + f" ... (+{len(selected_vars)-10} more)")
                
                # Show data preview of selected variables
                st.markdown("**Data preview (selected variables):**")
                preview_selected = selected_data.head(5)
                st.dataframe(preview_selected, use_container_width=True)

        # === COMPUTE BUTTON ===
        st.markdown("---")
        
        button_text = "🚀 Compute Varimax Model" if pca_method == "Varimax Rotation" else "🚀 Compute PCA Model"
        button_type = "primary"
        
        if st.button(button_text, type=button_type, use_container_width=True):
            # [Rest of the existing computation logic stays the same...]
            try:
                # Data preprocessing
                X = selected_data.copy()
                
                # CORREZIONE: Converti tutti i nomi delle colonne in stringhe
                X.columns = X.columns.astype(str)
                
                # Handle missing values
                if X.isnull().any().any():
                    if missing_values == "Remove":
                        X = X.dropna()
                        st.info(f"ℹ️ Removed {len(selected_data) - len(X)} rows with missing values")
                    else:
                        X = X.fillna(X.mean())
                        st.info("ℹ️ Missing values imputed with column means")
                
                # Centering e scaling
                if center_data and scale_data:
                    scaler = StandardScaler()
                    X_processed = pd.DataFrame(
                        scaler.fit_transform(X), 
                        columns=X.columns, 
                        index=X.index
                    )
                    st.info("✅ Data centered and scaled (StandardScaler)")
                elif center_data:
                    X_processed = X - X.mean()
                    scaler = None
                    st.info("✅ Data centered (mean removed)")
                elif scale_data:
                    X_processed = X / X.std()
                    scaler = None
                    st.info("✅ Data scaled (unit variance)")
                else:
                    X_processed = X
                    scaler = None
                    st.info("ℹ️ No preprocessing applied")
                
                if pca_method == "Standard PCA":
                    # Use compute_pca function from pca_utils
                    pca_dict = compute_pca(
                        pd.DataFrame(X_processed, index=X.index, columns=X.columns),
                        n_components=n_components,
                        center=False,  # Already preprocessed
                        scale=False
                    )

                    # Store results with additional metadata
                    pca_results = {
                        **pca_dict,  # Include all results from compute_pca
                        'original_data': X,
                        'processed_data': pd.DataFrame(X_processed, index=X.index, columns=X.columns),
                        'scaler': scaler,
                        'method': 'Standard PCA',
                        'parameters': {
                            'n_components': n_components,
                            'center': center_data,
                            'scale': scale_data,
                            'variables': selected_vars,
                            'method': 'Standard PCA',
                            'n_selected_vars': len(selected_vars)
                        }
                    }

                    st.success("✅ Standard PCA model computed successfully!")

                else:  # Varimax Rotation
                    # First compute standard PCA using compute_pca
                    pca_dict = compute_pca(
                        pd.DataFrame(X_processed, index=X.index, columns=X.columns),
                        n_components=n_components,
                        center=False,  # Already preprocessed
                        scale=False
                    )

                    # Apply Varimax rotation
                    with st.spinner("🔄 Applying Varimax rotation..."):
                        rotated_loadings, iterations = varimax_rotation(
                            pca_dict['loadings'],
                            max_iter=max_iter_varimax if 'max_iter_varimax' in locals() else 100,
                            tol=tolerance_varimax if 'tolerance_varimax' in locals() else 1e-6
                        )

                    # Calculate rotated scores
                    rotated_scores = X_processed @ rotated_loadings.values

                    # Calculate variance explained by rotated factors
                    rotated_variance = np.var(rotated_scores, axis=0, ddof=1)
                    total_variance = np.sum(rotated_variance)
                    rotated_variance_ratio = rotated_variance / total_variance

                    # Sort by variance explained (descending)
                    sort_idx = np.argsort(rotated_variance_ratio)[::-1]
                    rotated_loadings = rotated_loadings.iloc[:, sort_idx]
                    rotated_scores = rotated_scores[:, sort_idx]
                    rotated_variance_ratio = rotated_variance_ratio[sort_idx]
                    rotated_variance = rotated_variance[sort_idx]

                    # Store Varimax results
                    pca_results = {
                        'model': pca_dict['model'],  # Keep original for reference
                        'scores': pd.DataFrame(
                            rotated_scores,
                            columns=[f'Factor{i+1}' for i in range(n_components)],
                            index=X.index
                        ),
                        'loadings': pd.DataFrame(
                            rotated_loadings.values,
                            columns=[f'Factor{i+1}' for i in range(n_components)],
                            index=X.columns
                        ),
                        'explained_variance': rotated_variance,
                        'explained_variance_ratio': rotated_variance_ratio,
                        'cumulative_variance': np.cumsum(rotated_variance_ratio),
                        'eigenvalues': rotated_variance,
                        'original_data': X,
                        'processed_data': pd.DataFrame(X_processed, index=X.index, columns=X.columns),
                        'scaler': scaler,
                        'method': 'Varimax Rotation',
                        'varimax_iterations': iterations,
                        'parameters': {
                            'n_components': n_components,
                            'center': center_data,
                            'scale': scale_data,
                            'variables': selected_vars,
                            'method': 'Varimax Rotation',
                            'iterations': iterations,
                            'n_selected_vars': len(selected_vars)
                        }
                    }

                    st.success(f"✅ Varimax rotation completed in {iterations} iterations!")
                
                # Store results in session state
                st.session_state.pca_model = pca_results
                
                # Display summary
                st.markdown("### 📋 Model Summary")
                
                summary_col1, summary_col2, summary_col3 = st.columns(3)
                
                with summary_col1:
                    st.metric("Objects", len(X))
                    st.metric("Variables", len(selected_vars))
                
                with summary_col2:
                    comp_label = "Components" if pca_method == "Standard PCA" else "Factors"
                    st.metric(comp_label, n_components)
                    st.metric("Total Variance Explained", 
                            f"{pca_results['cumulative_variance'][-1]:.1%}")
                
                with summary_col3:
                    first_label = "First PC" if pca_method == "Standard PCA" else "First Factor"
                    st.metric(f"{first_label} Variance", 
                            f"{pca_results['explained_variance_ratio'][0]:.1%}")
                    if len(pca_results['explained_variance_ratio']) > 1:
                        second_label = "Second PC" if pca_method == "Standard PCA" else "Second Factor"
                        st.metric(f"{second_label} Variance", 
                                f"{pca_results['explained_variance_ratio'][1]:.1%}")
                
                # Variance table
                table_title = "### 📊 Variance Explained" if pca_method == "Standard PCA" else "### 📊 Factor Variance Explained"
                st.markdown(table_title)
                
                component_labels = ([f'PC{i+1}' for i in range(n_components)] if pca_method == "Standard PCA" 
                                  else [f'Factor{i+1}' for i in range(n_components)])
                
                variance_df = pd.DataFrame({
                    'Component': component_labels,
                    'Eigenvalue': pca_results['eigenvalues'],
                    'Variance %': pca_results['explained_variance_ratio'] * 100,
                    'Cumulative %': pca_results['cumulative_variance'] * 100
                })
                
                st.dataframe(variance_df.round(3), use_container_width=True)
                
                if pca_method == "Varimax Rotation":
                    st.info(f"🔄 Varimax rotation converged in {iterations} iterations")
                    st.info("📊 Factors are now optimized for interpretability (simple structure)")
                
            except Exception as e:
                st.error(f"❌ Error computing {pca_method}: {str(e)}")
                st.error("Check your data for issues and try again.")
                # Debug info per sviluppo
                if st.checkbox("Show debug info"):
                    st.write("Selected variables:", len(selected_vars))
                    st.write("Data shape:", selected_data.shape)
                    st.write("Data types:", selected_data.dtypes.value_counts())
                    st.write("Column names sample:", list(selected_data.columns[:10]))

    # ===== VARIANCE PLOTS TAB =====
    with tab2:
        st.markdown("## 📊 Variance Plots")
        st.markdown("*Equivalent to PCA_variance_plot.r and PCA_cumulative_var_plot.r*")
        
        if 'pca_model' not in st.session_state:
            st.warning("⚠️ No PCA model computed. Please compute a model first.")
        else:
            pca_results = st.session_state.pca_model
            is_varimax = pca_results.get('method') == 'Varimax Rotation'
            
            plot_type = st.selectbox(
                "Select variance plot:",
                ["📈 Scree Plot", "📊 Cumulative Variance", "🎯 Individual Variable Contribution", "🎲 Random Comparison"]
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

                st.plotly_chart(fig, width='stretch')
            
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

                st.plotly_chart(fig, width='stretch')
            
            elif plot_type == "🎯 Individual Variable Contribution":
                comp_label = "Factor" if is_varimax else "PC"
                st.markdown(f"### 🎯 Variable Contribution Analysis")
                st.markdown("*Based on significant components identified from Scree Plot*")
                
                # Step 1: Selezione componenti significativi
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
                
                # Step 2: Calcolo contributi pesati
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

                st.plotly_chart(fig, use_container_width=True)

                # Step 3: Tabella dettagliata
                st.markdown("#### Step 3: Detailed Contribution Table")

                # Sort contributions for display
                contrib_df_sorted = contrib_df.sort_values('Contribution_%', ascending=False)
                
                st.dataframe(contrib_df_sorted.round(2), use_container_width=True)
                
                # Interpretazione
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
            
            elif plot_type == "🎲 Random Comparison":
                st.markdown("### 🎲 Random Data Comparison")
                st.markdown("*Equivalent to PCA_model_PCA_rnd.r*")
                
                if is_varimax:
                    st.warning("⚠️ Random comparison not typically performed for Varimax rotation")
                    st.info("Showing comparison for underlying PCA components")
                
                n_randomizations = st.number_input("Number of randomizations:", 
                                                 min_value=10, max_value=1000, value=100)
                
                if st.button("🎲 Compare with Random Data"):
                    try:
                        original_variance = pca_results['explained_variance_ratio'] * 100
                        n_components = len(original_variance)
                        n_samples, n_vars = pca_results['original_data'].shape
                        
                        random_variances = []
                        
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        for i in range(n_randomizations):
                            # Generate random data
                            random_data = np.random.randn(n_samples, n_vars)
                            
                            # Apply same preprocessing
                            if pca_results['parameters']['scale']:
                                random_data = (random_data - random_data.mean(axis=0)) / random_data.std(axis=0)
                            elif pca_results['parameters']['center']:
                                random_data = random_data - random_data.mean(axis=0)
                            
                            # Compute PCA
                            random_pca = PCA(n_components=n_components)
                            random_pca.fit(random_data)
                            random_variances.append(random_pca.explained_variance_ratio_ * 100)
                            
                            progress_bar.progress((i + 1) / n_randomizations)
                            status_text.text(f"Completed {i + 1}/{n_randomizations} randomizations")
                        
                        progress_bar.empty()
                        status_text.empty()
                        
                        # Calculate statistics
                        random_variances = np.array(random_variances)
                        random_mean = np.mean(random_variances, axis=0)
                        random_std = np.std(random_variances, axis=0)
                        
                        # 95% confidence intervals
                        t_value = t.ppf(0.975, n_randomizations - 1)
                        random_ci_upper = random_mean + t_value * random_std
                        random_ci_lower = random_mean - t_value * random_std
                        
                        # Create plot
                        fig_random = go.Figure()
                        
                        # Original data
                        fig_random.add_trace(go.Scatter(
                            x=list(range(1, n_components + 1)),
                            y=original_variance,
                            mode='lines+markers',
                            name='Original Data',
                            line=dict(color='red', width=3),
                            marker=dict(size=8)
                        ))
                        
                        # Random data mean
                        fig_random.add_trace(go.Scatter(
                            x=list(range(1, n_components + 1)),
                            y=random_mean,
                            mode='lines+markers',
                            name='Random Data (Mean)',
                            line=dict(color='green', width=2, dash='dash'),
                            marker=dict(size=6)
                        ))
                        
                        # Confidence intervals
                        fig_random.add_trace(go.Scatter(
                            x=list(range(1, n_components + 1)) + list(range(n_components, 0, -1)),
                            y=np.concatenate([random_ci_upper, random_ci_lower[::-1]]),
                            fill='toself',
                            fillcolor='rgba(0,255,0,0.2)',
                            line=dict(color='rgba(255,255,255,0)'),
                            hoverinfo="skip",
                            showlegend=True,
                            name='95% CI'
                        ))
                        
                        fig_random.update_layout(
                            title=f'PCA: Original vs Random Data ({n_randomizations} randomizations)',
                            xaxis_title='Component Number',
                            yaxis_title='% Explained Variance',
                            height=600
                        )
                        
                        st.plotly_chart(fig_random, width='stretch')
                        
                        # Summary
                        significant_components = sum(original_variance > random_ci_upper)
                        st.success(f"**Result:** {significant_components} components are statistically significant")
                        
                    except Exception as e:
                        st.error(f"Random comparison failed: {str(e)}")

    # ===== LOADINGS PLOTS TAB =====
    with tab3:
        st.markdown("## 📈 Loadings Plots")

        
        if 'pca_model' not in st.session_state:
            st.warning("⚠️ No PCA model computed. Please compute a model first.")
        else:
            pca_results = st.session_state.pca_model
            loadings = pca_results['loadings']
            is_varimax = pca_results.get('method') == 'Varimax Rotation'
            
            title_suffix = " (Varimax Factors)" if is_varimax else ""
            
            loading_plot_type = st.selectbox(
                "Select loading plot type:",
                ["📊 Loading Scatter Plot", "📈 Loading Line Plot", "🎯 Loading Bar Plot"]
            )
            
            # PC/Factor selection
            col1, col2 = st.columns(2)
            with col1:
                pc_x = st.selectbox("X-axis:", loadings.columns, index=0)
            with col2:
                pc_y = st.selectbox("Y-axis:", loadings.columns, index=1)
            
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

                st.plotly_chart(fig, use_container_width=True)

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
            
            elif loading_plot_type == "📈 Loading Line Plot":
                st.markdown(f"### 📈 Loading Line Plot{title_suffix}")

                selected_comps = st.multiselect(
                    f"Select {'factors' if is_varimax else 'components'} to display:",
                    loadings.columns.tolist(),
                    default=loadings.columns[:3].tolist(),
                    key="loading_line_components"
                )

                if selected_comps:
                    # Use plot_loadings_line from pca_utils.pca_plots
                    fig = plot_loadings_line(
                        loadings,
                        selected_comps,
                        is_varimax=is_varimax
                    )

                    st.plotly_chart(fig, use_container_width=True)

    # ===== SCORE PLOTS TAB =====
    with tab4:
        st.markdown("## 🎯 Score Plots")
        st.markdown("*Equivalent to PCA_score_plot.r and PCA_score3D.r*")
        
        if 'pca_model' not in st.session_state:
            st.warning("⚠️ No PCA model computed. Please compute a model first.")
        else:
            pca_results = st.session_state.pca_model
            scores = pca_results['scores']
            is_varimax = pca_results.get('method') == 'Varimax Rotation'
            
            title_suffix = " (Varimax)" if is_varimax else ""
            
            score_plot_type = st.selectbox(
                "Select score plot type:",
                ["📊 2D Score Plot", "🎲 3D Score Plot", "📈 Line Profiles Plot"]
            )

            if score_plot_type == "📊 2D Score Plot":
                st.markdown(f"### 📊 2D Score Plot{title_suffix}")
                
                # PC/Factor selection
                col1, col2 = st.columns(2)
                with col1:
                    pc_x = st.selectbox("X-axis:", scores.columns, index=0, key="score_x")
                with col2:
                    pc_y = st.selectbox("Y-axis:", scores.columns, index=1, key="score_y")
                
                # Ottieni la varianza spiegata per i componenti selezionati
                pc_x_idx = list(scores.columns).index(pc_x)
                pc_y_idx = list(scores.columns).index(pc_y)
                var_x = pca_results['explained_variance_ratio'][pc_x_idx] * 100
                var_y = pca_results['explained_variance_ratio'][pc_y_idx] * 100
                var_total = var_x + var_y
                
                # Get custom variables
                custom_vars = []
                if 'custom_variables' in st.session_state:
                    custom_vars = list(st.session_state.custom_variables.keys())
                
                # Display options
                col3, col4 = st.columns(2)
                with col3:
                    all_color_options = (["None", "Row Index"] + 
                                        [col for col in data.columns if col not in pca_results['parameters']['variables']] +
                                        custom_vars)
                    
                    color_by = st.selectbox("Color points by:", all_color_options)
                
                with col4:
                    label_options = ["None", "Row Index"] + [col for col in data.columns if col not in pca_results['parameters']['variables']]
                    show_labels = st.selectbox("Show labels:", label_options)

                # === OPZIONI VISUALIZZAZIONE ===
                st.markdown("### 🎨 Visualization Options")
                col_vis1, col_vis2, col_vis3 = st.columns(3)
                
                with col_vis1:
                    if color_by != "None":
                        show_convex_hull = st.checkbox("Show convex hulls", value=False, key="show_convex_hull")  # CHANGED: value=False
                    else:
                        show_convex_hull = False
                
                with col_vis2:
                    if color_by != "None":
                        hull_opacity = st.slider("Hull line opacity", 0.1, 1.0, 0.7, key="hull_opacity")
                    else:
                        hull_opacity = 0.7

                # Group Management
                st.markdown("### 🔧 Group Management")
                with st.expander("Create Custom Groups"):
                    st.markdown("#### Create Time Trend Variable")
                    col_time1, col_time2 = st.columns(2)
                    
                    with col_time1:
                        time_var_name = st.text_input("Time variable name:", value="Time_Trend")
                        
                    with col_time2:
                        if st.button("🕒 Create Time Trend"):
                            time_trend = pd.Series(range(1, len(data) + 1), index=data.index)
                            
                            if 'custom_variables' not in st.session_state:
                                st.session_state.custom_variables = {}
                            
                            st.session_state.custom_variables[time_var_name] = time_trend
                            st.success(f"✅ Created time trend variable: {time_var_name}")
                            st.rerun()
                    
                    # Show created variables
                    if 'custom_variables' in st.session_state and st.session_state.custom_variables:
                        st.markdown("#### 📋 Created Variables")
                        for var_name, var_data in st.session_state.custom_variables.items():
                            col_info, col_delete = st.columns([3, 1])
                            with col_info:
                                unique_vals = var_data.nunique()
                                st.write(f"**{var_name}**: {unique_vals} unique values")
                            with col_delete:
                                if st.button("🗑️", key=f"delete_{var_name}"):
                                    del st.session_state.custom_variables[var_name]
                                    st.rerun()
                
                # === CREA IL PLOT ===
                # DEFINISCI SEMPRE text_param E color_data PRIMA DEL PLOT
                text_param = None if show_labels == "None" else (data.index if show_labels == "Row Index" else data[show_labels])
                
                if color_by == "None":
                    color_data = None
                elif color_by == "Row Index":
                    color_data = data.index
                elif color_by in custom_vars:
                    color_data = st.session_state.custom_variables[color_by]
                else:
                    color_data = data[color_by]
                
                # Calcola il range per assi identici
                x_range = [scores[pc_x].min(), scores[pc_x].max()]
                y_range = [scores[pc_y].min(), scores[pc_y].max()]
                max_abs_range = max(abs(min(x_range + y_range)), abs(max(x_range + y_range)))
                axis_range = [-max_abs_range * 1.1, max_abs_range * 1.1]

                # Create plot e definisci sempre color_discrete_map
                color_discrete_map = None  # INIZIALIZZA SEMPRE
                
                if color_by == "None":
                    fig = px.scatter(
                        x=scores[pc_x], y=scores[pc_y], text=text_param,
                        title=f"Scores: {pc_x} vs {pc_y}{title_suffix}<br>Total Explained Variance: {var_total:.1f}%",
                        labels={'x': f'{pc_x} ({var_x:.1f}%)', 'y': f'{pc_y} ({var_y:.1f}%)'}
                    )
                else:
                    # NUOVO: Implementazione della scala blu-rossa per variabili quantitative
                    if color_by in custom_vars and ('Time' in color_by or 'time' in color_by):
                        # Time variables: usa scala blu-rossa
                        fig = px.scatter(
                            x=scores[pc_x], y=scores[pc_y], color=color_data, text=text_param,
                            title=f"Scores: {pc_x} vs {pc_y} (colored by {color_by}){title_suffix}<br>Total Explained Variance: {var_total:.1f}%",
                            labels={'x': f'{pc_x} ({var_x:.1f}%)', 'y': f'{pc_y} ({var_y:.1f}%)', 'color': color_by},
                            color_continuous_scale=[(0, 'blue'), (1, 'red')]  # CHANGED: scala blu-rossa pura
                        )
                    elif (color_by != "None" and color_by != "Row Index" and 
                          hasattr(color_data, 'dtype') and pd.api.types.is_numeric_dtype(color_data)):
                        # NUOVO: Controlla se è quantitativo usando le funzioni di utils
                        from color_utils import is_quantitative_variable
                        
                        if is_quantitative_variable(color_data):
                            # Variabile quantitativa: usa scala blu-rossa
                            fig = px.scatter(
                                x=scores[pc_x], y=scores[pc_y], color=color_data, text=text_param,
                                title=f"Scores: {pc_x} vs {pc_y} (colored by {color_by}){title_suffix}<br>Total Explained Variance: {var_total:.1f}%",
                                labels={'x': f'{pc_x} ({var_x:.1f}%)', 'y': f'{pc_y} ({var_y:.1f}%)', 'color': color_by},
                                color_continuous_scale=[(0, 'blue'), (1, 'red')]  # CHANGED: scala blu-rossa pura
                            )
                        else:
                            # Variabile categorica: usa sistema unificato
                            color_data_series = pd.Series(color_data)
                            unique_values = color_data_series.dropna().unique()
                            color_discrete_map = create_categorical_color_map(unique_values)
                            
                            fig = px.scatter(
                                x=scores[pc_x], y=scores[pc_y], color=color_data, text=text_param,
                                title=f"Scores: {pc_x} vs {pc_y} (colored by {color_by}){title_suffix}<br>Total Explained Variance: {var_total:.1f}%",
                                labels={'x': f'{pc_x} ({var_x:.1f}%)', 'y': f'{pc_y} ({var_y:.1f}%)', 'color': color_by},
                                color_discrete_map=color_discrete_map
                            )
                    else:
                        # Variabile categorica (default per Row Index e altri casi)
                        color_data_series = pd.Series(color_data)
                        unique_values = color_data_series.dropna().unique()
                        color_discrete_map = create_categorical_color_map(unique_values)
                        
                        fig = px.scatter(
                            x=scores[pc_x], y=scores[pc_y], color=color_data, text=text_param,
                            title=f"Scores: {pc_x} vs {pc_y} (colored by {color_by}){title_suffix}<br>Total Explained Variance: {var_total:.1f}%",
                            labels={'x': f'{pc_x} ({var_x:.1f}%)', 'y': f'{pc_y} ({var_y:.1f}%)', 'color': color_by},
                            color_discrete_map=color_discrete_map
                        )
                
                # Add zero lines
                fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.7)
                fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.7)

                # AGGIUNGI CONVEX HULL (solo per variabili categoriche)
                if (color_by != "None" and 
                    show_convex_hull and
                    not (color_by in custom_vars and ('Time' in color_by or 'time' in color_by)) and
                    not (hasattr(color_data, 'dtype') and pd.api.types.is_numeric_dtype(color_data) and 
                         is_quantitative_variable(color_data))):  # CHANGED: Non aggiungere hull per variabili quantitative
                    
                    try:
                        fig = add_convex_hulls(fig, scores, pc_x, pc_y, color_data, color_discrete_map, hull_opacity)
                    except Exception as e:
                        st.error(f"Error adding convex hulls: {e}")

                if show_labels != "None":
                    fig.update_traces(textposition="top center")
                
                # IMPOSTA SCALE IDENTICHE
                fig.update_layout(
                    height=600,
                    width=600,
                    xaxis=dict(
                        range=axis_range,
                        scaleanchor="y",
                        scaleratio=1,
                        constrain="domain"
                    ),
                    yaxis=dict(
                        range=axis_range,
                        constrain="domain"
                    )
                )
                
                st.plotly_chart(fig, width='stretch', key="pca_scores_plot")

                # === SELEZIONE PER COORDINATE ===
                st.markdown("### 🎯 Coordinate Selection")
                st.info("Define a rectangular area using PC coordinates to select multiple points at once.")

                col_coords, col_preview = st.columns([1, 1])

                # CORREZIONE: Gestione dinamica delle coordinate quando cambiano le PC
                # Crea una chiave unica per questa combinazione di assi
                current_axes_key = f"{pc_x}_{pc_y}"

                # Se gli assi sono cambiati, resetta le coordinate
                if 'last_axes_key' not in st.session_state or st.session_state.last_axes_key != current_axes_key:
                    # Gli assi sono cambiati - ricalcola i valori di default
                    st.session_state.coord_x_min = float(scores[pc_x].quantile(0.25))
                    st.session_state.coord_x_max = float(scores[pc_x].quantile(0.75))
                    st.session_state.coord_y_min = float(scores[pc_y].quantile(0.25))
                    st.session_state.coord_y_max = float(scores[pc_y].quantile(0.75))
                    st.session_state.last_axes_key = current_axes_key
                    
                    # Pulisci anche la selezione manuale se esistente
                    if 'manual_selected_points' in st.session_state:
                        del st.session_state.manual_selected_points
                    if 'manual_selection_input' in st.session_state:
                        st.session_state.manual_selection_input = ""

                # Initialize default values if not in session state (primo utilizzo)
                if 'coord_x_min' not in st.session_state:
                    st.session_state.coord_x_min = float(scores[pc_x].quantile(0.25))
                if 'coord_x_max' not in st.session_state:
                    st.session_state.coord_x_max = float(scores[pc_x].quantile(0.75))
                if 'coord_y_min' not in st.session_state:
                    st.session_state.coord_y_min = float(scores[pc_y].quantile(0.25))
                if 'coord_y_max' not in st.session_state:
                    st.session_state.coord_y_max = float(scores[pc_y].quantile(0.75))

                with col_coords:
                    st.markdown("#### Selection Box Coordinates")
                    
                    col_x, col_y = st.columns(2)
                    
                    with col_x:
                        x_min = st.number_input(f"{pc_x} Min:", 
                                            value=st.session_state.coord_x_min, 
                                            key="input_x_min",
                                            format="%.3f",
                                            on_change=lambda: setattr(st.session_state, 'coord_x_min', st.session_state.input_x_min))
                        x_max = st.number_input(f"{pc_x} Max:", 
                                            value=st.session_state.coord_x_max, 
                                            key="input_x_max", 
                                            format="%.3f",
                                            on_change=lambda: setattr(st.session_state, 'coord_x_max', st.session_state.input_x_max))
                    
                    with col_y:
                        y_min = st.number_input(f"{pc_y} Min:", 
                                            value=st.session_state.coord_y_min, 
                                            key="input_y_min",
                                            format="%.3f",
                                            on_change=lambda: setattr(st.session_state, 'coord_y_min', st.session_state.input_y_min))
                        y_max = st.number_input(f"{pc_y} Max:", 
                                            value=st.session_state.coord_y_max, 
                                            key="input_y_max",
                                            format="%.3f",
                                            on_change=lambda: setattr(st.session_state, 'coord_y_max', st.session_state.input_y_max))

                with col_preview:
                    st.markdown("#### Quick Presets")
                    
                    preset = st.selectbox(
                        "Selection presets:",
                        ["Custom", "Upper Right Quadrant", "Upper Left Quadrant", 
                        "Lower Right Quadrant", "Lower Left Quadrant", 
                        "Center Region", "Full Range"],
                        key="coord_preset"
                    )
                    
                    col_preset_btn, col_reset_btn = st.columns(2)
                    
                    with col_preset_btn:
                        apply_preset = st.button("Apply Preset", key="apply_coord_preset")
                    
                    with col_reset_btn:
                        reset_coords = st.button("Reset", key="reset_coordinates", help="Reset to default quartile values")
                    
                    if reset_coords:
                        # Reset to default quartile values
                        st.session_state.coord_x_min = float(scores[pc_x].quantile(0.25))
                        st.session_state.coord_x_max = float(scores[pc_x].quantile(0.75))
                        st.session_state.coord_y_min = float(scores[pc_y].quantile(0.25))
                        st.session_state.coord_y_max = float(scores[pc_y].quantile(0.75))
                        # Clear manual input field
                        if 'manual_selection_input' in st.session_state:
                            st.session_state.manual_selection_input = ""
                        # Clear any existing selection
                        if 'manual_selected_points' in st.session_state:
                            del st.session_state.manual_selected_points
                        st.rerun()
                    
                    if apply_preset:
                        x_center = scores[pc_x].median()
                        y_center = scores[pc_y].median()
                        
                        if preset == "Upper Right Quadrant":
                            st.session_state.coord_x_min = float(x_center)
                            st.session_state.coord_x_max = float(scores[pc_x].max())
                            st.session_state.coord_y_min = float(y_center)
                            st.session_state.coord_y_max = float(scores[pc_y].max())
                        elif preset == "Upper Left Quadrant":
                            st.session_state.coord_x_min = float(scores[pc_x].min())
                            st.session_state.coord_x_max = float(x_center)
                            st.session_state.coord_y_min = float(y_center)
                            st.session_state.coord_y_max = float(scores[pc_y].max())
                        elif preset == "Lower Right Quadrant":
                            st.session_state.coord_x_min = float(x_center)
                            st.session_state.coord_x_max = float(scores[pc_x].max())
                            st.session_state.coord_y_min = float(scores[pc_y].min())
                            st.session_state.coord_y_max = float(y_center)
                        elif preset == "Lower Left Quadrant":
                            st.session_state.coord_x_min = float(scores[pc_x].min())
                            st.session_state.coord_x_max = float(x_center)
                            st.session_state.coord_y_min = float(scores[pc_y].min())
                            st.session_state.coord_y_max = float(y_center)
                        elif preset == "Center Region":
                            x_std = scores[pc_x].std()
                            y_std = scores[pc_y].std()
                            st.session_state.coord_x_min = float(x_center - x_std)
                            st.session_state.coord_x_max = float(x_center + x_std)
                            st.session_state.coord_y_min = float(y_center - y_std)
                            st.session_state.coord_y_max = float(y_center + y_std)
                        elif preset == "Full Range":
                            st.session_state.coord_x_min = float(scores[pc_x].min())
                            st.session_state.coord_x_max = float(scores[pc_x].max())
                            st.session_state.coord_y_min = float(scores[pc_y].min())
                            st.session_state.coord_y_max = float(scores[pc_y].max())
                        
                        st.rerun()

                # Use the session state values for calculations
                x_min = st.session_state.get('input_x_min', st.session_state.coord_x_min)
                x_max = st.session_state.get('input_x_max', st.session_state.coord_x_max)
                y_min = st.session_state.get('input_y_min', st.session_state.coord_y_min)
                y_max = st.session_state.get('input_y_max', st.session_state.coord_y_max)

                # Update internal state with current values
                st.session_state.coord_x_min = x_min
                st.session_state.coord_x_max = x_max
                st.session_state.coord_y_min = y_min
                st.session_state.coord_y_max = y_max

                # Calculate automatically selected points
                mask = ((scores[pc_x] >= x_min) & (scores[pc_x] <= x_max) & 
                        (scores[pc_y] >= y_min) & (scores[pc_y] <= y_max))
                coordinate_selection = list(np.where(mask)[0])

                # Selection results and manual input
                col_result, col_apply = st.columns([2, 1])

                with col_result:
                    if len(coordinate_selection) > 0:
                        st.success(f"Coordinate box contains {len(coordinate_selection)} points")
                        st.info(f"Selection axes: {pc_x} vs {pc_y}")
                        st.info(f"Selection covers {len(coordinate_selection)/len(scores)*100:.1f}% of samples")
                        
                        selected_names = [scores.index[i] for i in coordinate_selection]
                        selected_indices_1based = [i+1 for i in coordinate_selection]
                        
                        if len(selected_names) <= 10:
                            sample_list = ', '.join(map(str, selected_names))
                            indices_list = ', '.join(map(str, selected_indices_1based))
                        else:
                            sample_list = ', '.join(map(str, selected_names[:8])) + f" ... (+{len(selected_names)-8} more)"
                            indices_list = ', '.join(map(str, selected_indices_1based[:8])) + f" ... (+{len(selected_indices_1based)-8} more)"
                        
                        st.markdown(f"**Samples:** {sample_list}")
                        st.markdown(f"**Indices:** {indices_list}")
                        
                        with st.expander("📋 Copy Sample Lists"):
                            current_selected_names = [scores.index[i] for i in coordinate_selection]
                            current_selected_indices_1based = [i+1 for i in coordinate_selection]
                            
                            selection_hash = hash(tuple(coordinate_selection)) if coordinate_selection else 0
                            
                            col_preview1, col_preview2 = st.columns(2)
                            
                            with col_preview1:
                                st.markdown("**Sample Names:**")
                                names_text = ', '.join(map(str, current_selected_names))
                                st.text_area(
                                    "Copy sample names:",
                                    names_text,
                                    height=100,
                                    key=f"copy_sample_names_{selection_hash}"
                                )
                            
                            with col_preview2:
                                st.markdown("**Row Indices (1-based):**")
                                indices_text = ','.join(map(str, current_selected_indices_1based))
                                st.text_area(
                                    "Copy row indices:",
                                    indices_text,
                                    height=100,
                                    key=f"copy_row_indices_{selection_hash}"
                                )
                    else:
                        st.warning("No points in current coordinate range")

                with col_apply:
                    if len(coordinate_selection) > 0:
                        if st.button("Apply Coordinate Selection", type="primary", key="apply_coords"):
                            st.session_state.manual_selected_points = coordinate_selection
                            
                            if 'manual_selection_input' in st.session_state:
                                st.session_state.manual_selection_input = ""

                            st.success("Selection applied!")
                            st.rerun()

                # Manual Input
                st.markdown("### 🔢 Alternative: Manual Input")

                manual_selection = st.text_input(
                    "Enter specific row indices (1-based):",
                    placeholder="1,5,10-15,20,25-30",
                    key="manual_selection_input",
                    help="Use 1-based indexing. Ranges supported: 10-15"
                )

                if manual_selection.strip():
                    try:
                        selected_indices = []
                        for part in manual_selection.split(','):
                            part = part.strip()
                            if '-' in part and part.count('-') == 1:
                                start, end = map(int, part.split('-'))
                                if start <= end:
                                    selected_indices.extend(range(start-1, end))
                                else:
                                    st.error(f"Invalid range: {start}-{end} (start must be <= end)")
                                    continue
                            else:
                                selected_indices.append(int(part)-1)
                        
                        selected_indices = sorted(list(set(selected_indices)))
                        valid_indices = [i for i in selected_indices if 0 <= i < len(scores)]
                        invalid_count = len(selected_indices) - len(valid_indices)
                        
                        if valid_indices:
                            st.success(f"Manual input: {len(valid_indices)} points selected")
                            if invalid_count > 0:
                                st.warning(f"{invalid_count} indices out of range (valid: 1-{len(scores)})")
                            st.session_state.manual_selected_points = valid_indices
                        else:
                            st.error("No valid indices found")
                            
                    except ValueError:
                        st.error("Invalid format. Use numbers and ranges: 1,5,10-15,20")
                else:
                    if 'manual_selected_points' in st.session_state and manual_selection.strip() == "":
                        del st.session_state.manual_selected_points

                # Selection Visualization and Actions
                if 'manual_selected_points' in st.session_state and st.session_state.manual_selected_points:
                    selected_indices = st.session_state.manual_selected_points
                    
                    st.markdown("### 🎯 Selected Points Visualization")
                    
                    # Create enhanced visualization
                    plot_data = pd.DataFrame({
                        'PC_X': scores[pc_x],
                        'PC_Y': scores[pc_y],
                        'Row_Index': range(1, len(scores)+1),
                        'Sample_Name': scores.index,
                        'Selection': ['Selected' if i in selected_indices else 'Not Selected' 
                                    for i in range(len(scores))],
                        'Point_Size': [12 if i in selected_indices else 6 for i in range(len(scores))]
                    })
                    
                    if color_by != "None":
                        plot_data['Color_Group'] = color_data
                        
                        fig_selected = px.scatter(
                            plot_data, x='PC_X', y='PC_Y', 
                            color='Color_Group',
                            symbol='Selection',
                            symbol_map={'Selected': 'diamond', 'Not Selected': 'circle'},
                            size='Point_Size',
                            title=f"Selected Points: {pc_x} vs {pc_y} (colored by {color_by})",
                            labels={'PC_X': f'{pc_x} ({var_x:.1f}%)', 'PC_Y': f'{pc_y} ({var_y:.1f}%)'},
                            hover_data=['Sample_Name', 'Row_Index']
                        )
                    else:
                        fig_selected = px.scatter(
                            plot_data, x='PC_X', y='PC_Y',
                            color='Selection',
                            color_discrete_map={'Selected': '#FF4B4B', 'Not Selected': '#1f77b4'},
                            size='Point_Size',
                            title=f"Selected Points: {pc_x} vs {pc_y}",
                            labels={'PC_X': f'{pc_x} ({var_x:.1f}%)', 'PC_Y': f'{pc_y} ({var_y:.1f}%)'},
                            hover_data=['Sample_Name', 'Row_Index']
                        )
                    
                    # Add selection box visualization
                    fig_selected.add_shape(
                        type="rect",
                        x0=x_min, y0=y_min, x1=x_max, y1=y_max,
                        line=dict(color="red", width=2, dash="dash"),
                        fillcolor="rgba(255,0,0,0.1)"
                    )
                    
                    fig_selected.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.7)
                    fig_selected.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.7)
                    
                    fig_selected.update_layout(
                        height=500,
                        xaxis=dict(range=axis_range, scaleanchor="y", scaleratio=1, constrain="domain"),
                        yaxis=dict(range=axis_range, constrain="domain")
                    )
                    
                    st.plotly_chart(fig_selected, width='stretch', key="selection_visualization")
                    
                    # Export and Actions
                    st.markdown("### 💾 Export & Actions")

                    selected_sample_indices = [scores.index[i] for i in selected_indices]
                    original_data = st.session_state.current_data

                    try:
                        selected_data = original_data.loc[selected_sample_indices]
                        remaining_data = original_data.drop(selected_sample_indices)
                        
                        # Prima riga: Download e Split
                        col_exp1, col_exp2, col_exp3, col_exp4 = st.columns(4)
                        
                        with col_exp1:
                            selected_csv = selected_data.to_csv(index=True)
                            st.download_button(
                                f"Download Selected ({len(selected_data)})",
                                selected_csv,
                                "selected_samples.csv",
                                "text/csv",
                                key="download_selected"
                            )
                        
                        with col_exp2:
                            remaining_csv = remaining_data.to_csv(index=True)
                            st.download_button(
                                f"Download Remaining ({len(remaining_data)})",
                                remaining_csv,
                                "remaining_samples.csv",
                                "text/csv",
                                key="download_remaining"
                            )
                        
                        with col_exp3:
                            if st.button("💾 Save Split to Workspace", key="save_split", type="primary"):
                                # Determine selection method
                                selection_method = 'Coordinate' if 'coord_x_min' in st.session_state else 'Manual'

                                # Use workspace function to save split
                                selected_name, remaining_name = save_dataset_split(
                                    selected_data=selected_data,
                                    remaining_data=remaining_data,
                                    pc_x=pc_x,
                                    pc_y=pc_y,
                                    parent_name=st.session_state.get('current_dataset', 'Dataset'),
                                    selection_method=selection_method
                                )

                                st.success(f"✅ Split saved to workspace!")
                                st.info(f"**Selected**: {selected_name} ({len(selected_data)} samples)")
                                st.info(f"**Remaining**: {remaining_name} ({len(remaining_data)} samples)")
                                st.info("📂 Go to **Data Handling → Workspace** to load these datasets")
                        
                        with col_exp4:
                            with st.expander("🔧 More Actions"):
                                if st.button("🔄 Invert Selection", key="invert_selection", use_container_width=True):
                                    all_indices = set(range(len(scores)))
                                    current_selected = set(selected_indices)
                                    inverted_indices = list(all_indices - current_selected)
                                    st.session_state.manual_selected_points = inverted_indices
                                    st.rerun()
                                
                                if st.button("🗑️ Clear Selection", key="clear_selection", use_container_width=True):
                                    del st.session_state.manual_selected_points
                                    st.rerun()
                                    
                    except Exception as e:
                        st.error(f"Error processing selection: {e}")

                # Quick Selection Actions
                st.markdown("### ⚡ Quick Selection Actions")

                col_quick1, col_quick2, col_quick3 = st.columns(3)

                with col_quick1:
                    if st.button("Select All Samples", key="select_all_samples"):
                        st.session_state.manual_selected_points = list(range(len(scores)))
                        st.rerun()

                with col_quick2:
                    if st.button("Random 20 Samples", key="random_20"):
                        import random
                        n_samples = min(20, len(scores))
                        random_indices = random.sample(range(len(scores)), n_samples)
                        st.session_state.manual_selected_points = random_indices
                        st.rerun()

                with col_quick3:
                    if st.button("First 10 Samples", key="first_10"):
                        n_samples = min(10, len(scores))
                        st.session_state.manual_selected_points = list(range(n_samples))
                        st.rerun()

                # Metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(f"{pc_x} Variance", f"{var_x:.1f}%")
                with col2:
                    st.metric(f"{pc_y} Variance", f"{var_y:.1f}%")
                with col3:
                    st.metric("Combined Variance", f"{var_total:.1f}%")
            
            elif score_plot_type == "🎲 3D Score Plot":
                st.markdown(f"### 🎲 3D Score Plot{title_suffix}")
                
                if len(scores.columns) >= 3:
                    # Selezione assi
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        pc_x = st.selectbox("X-axis:", scores.columns, index=0, key="score3d_x")
                    with col2:
                        pc_y = st.selectbox("Y-axis:", scores.columns, index=1, key="score3d_y")
                    with col3:
                        pc_z = st.selectbox("Z-axis:", scores.columns, index=2, key="score3d_z")
                    
                    # Opzioni di visualizzazione per 3D
                    col4, col5 = st.columns(2)
                    
                    with col4:
                        custom_vars = []
                        if 'custom_variables' in st.session_state:
                            custom_vars = list(st.session_state.custom_variables.keys())
                        
                        all_color_options_3d = (["None", "Row Index"] + 
                                            [col for col in data.columns if col not in pca_results['parameters']['variables']] +
                                            custom_vars)
                        
                        color_by_3d = st.selectbox("Color points by:", all_color_options_3d, key="color_3d")
                    
                    with col5:
                        label_options_3d = ["None", "Row Index"] + [col for col in data.columns if col not in pca_results['parameters']['variables']]
                        show_labels_3d = st.selectbox("Show labels:", label_options_3d, key="labels_3d")
                    
                    # Opzioni visualizzazione 3D
                    st.markdown("### 🎨 3D Visualization Options")
                    col_vis1, col_vis2, col_vis3 = st.columns(3)
                    
                    with col_vis1:
                        point_size_3d = st.slider("Point size", 2, 15, 6, key="point_size_3d")
                    
                    with col_vis3:
                        show_axes_3d = st.checkbox("Show axis planes", value=True, key="show_axes_3d")
                    
                    # Prepara dati per il plot
                    text_param_3d = None if show_labels_3d == "None" else (data.index if show_labels_3d == "Row Index" else data[show_labels_3d])
                    
                    if color_by_3d == "None":
                        color_data_3d = None
                    elif color_by_3d == "Row Index":
                        color_data_3d = data.index
                    elif color_by_3d in custom_vars:
                        color_data_3d = st.session_state.custom_variables[color_by_3d]
                    else:
                        color_data_3d = data[color_by_3d]
                    
                    # Calcola varianza spiegata
                    pc_x_idx = list(scores.columns).index(pc_x)
                    pc_y_idx = list(scores.columns).index(pc_y)
                    pc_z_idx = list(scores.columns).index(pc_z)
                    var_x = pca_results['explained_variance_ratio'][pc_x_idx] * 100
                    var_y = pca_results['explained_variance_ratio'][pc_y_idx] * 100
                    var_z = pca_results['explained_variance_ratio'][pc_z_idx] * 100
                    var_total_3d = var_x + var_y + var_z
                    
                    # Crea il plot 3D
                    if color_by_3d == "None":
                        fig_3d = px.scatter_3d(
                            x=scores[pc_x], y=scores[pc_y], z=scores[pc_z],
                            text=text_param_3d,
                            title=f"3D Scores: {pc_x}, {pc_y}, {pc_z}{title_suffix}<br>Total Explained Variance: {var_total_3d:.1f}%",
                            labels={'x': f'{pc_x} ({var_x:.1f}%)', 'y': f'{pc_y} ({var_y:.1f}%)', 'z': f'{pc_z} ({var_z:.1f}%)'}
                        )
                    else:
                        # NUOVO: Implementazione della scala blu-rossa per variabili quantitative in 3D
                        if color_by_3d in custom_vars and ('Time' in color_by_3d or 'time' in color_by_3d):
                            fig_3d = px.scatter_3d(
                                x=scores[pc_x], y=scores[pc_y], z=scores[pc_z], 
                                color=color_data_3d, text=text_param_3d,
                                title=f"3D Scores: {pc_x}, {pc_y}, {pc_z} (colored by {color_by_3d}){title_suffix}<br>Total Explained Variance: {var_total_3d:.1f}%",
                                labels={'x': f'{pc_x} ({var_x:.1f}%)', 'y': f'{pc_y} ({var_y:.1f}%)', 'z': f'{pc_z} ({var_z:.1f}%)', 'color': color_by_3d},
                                color_continuous_scale=[(0, 'blue'), (1, 'red')]  # CHANGED: scala blu-rossa pura
                            )
                        elif (color_by_3d != "None" and color_by_3d != "Row Index" and 
                              hasattr(color_data_3d, 'dtype') and pd.api.types.is_numeric_dtype(color_data_3d)):
                            # NUOVO: Controlla se è quantitativo usando le funzioni di utils
                            from color_utils import is_quantitative_variable
                            
                            if is_quantitative_variable(color_data_3d):
                                # Variabile quantitativa: usa scala blu-rossa
                                fig_3d = px.scatter_3d(
                                    x=scores[pc_x], y=scores[pc_y], z=scores[pc_z], 
                                    color=color_data_3d, text=text_param_3d,
                                    title=f"3D Scores: {pc_x}, {pc_y}, {pc_z} (colored by {color_by_3d}){title_suffix}<br>Total Explained Variance: {var_total_3d:.1f}%",
                                    labels={'x': f'{pc_x} ({var_x:.1f}%)', 'y': f'{pc_y} ({var_y:.1f}%)', 'z': f'{pc_z} ({var_z:.1f}%)', 'color': color_by_3d},
                                    color_continuous_scale=[(0, 'blue'), (1, 'red')]  # CHANGED: scala blu-rossa pura
                                )
                            else:
                                # Variabile categorica: usa sistema unificato
                                color_data_series_3d = pd.Series(color_data_3d)
                                unique_values_3d = color_data_series_3d.dropna().unique()
                                color_discrete_map_3d = create_categorical_color_map(unique_values_3d)
                                    
                                fig_3d = px.scatter_3d(
                                    x=scores[pc_x], y=scores[pc_y], z=scores[pc_z], 
                                    color=color_data_3d, text=text_param_3d,
                                    title=f"3D Scores: {pc_x}, {pc_y}, {pc_z} (colored by {color_by_3d}){title_suffix}<br>Total Explained Variance: {var_total_3d:.1f}%",
                                    labels={'x': f'{pc_x} ({var_x:.1f}%)', 'y': f'{pc_y} ({var_y:.1f}%)', 'z': f'{pc_z} ({var_z:.1f}%)', 'color': color_by_3d},
                                    color_discrete_map=color_discrete_map_3d
                                )
                        else:
                            # Variabile categorica (default per Row Index e altri casi)
                            color_data_series_3d = pd.Series(color_data_3d)
                            unique_values_3d = color_data_series_3d.dropna().unique()
                            color_discrete_map_3d = create_categorical_color_map(unique_values_3d)
                                
                            fig_3d = px.scatter_3d(
                                x=scores[pc_x], y=scores[pc_y], z=scores[pc_z], 
                                color=color_data_3d, text=text_param_3d,
                                title=f"3D Scores: {pc_x}, {pc_y}, {pc_z} (colored by {color_by_3d}){title_suffix}<br>Total Explained Variance: {var_total_3d:.1f}%",
                                labels={'x': f'{pc_x} ({var_x:.1f}%)', 'y': f'{pc_y} ({var_y:.1f}%)', 'z': f'{pc_z} ({var_z:.1f}%)', 'color': color_by_3d},
                                color_discrete_map=color_discrete_map_3d
                            )
                    
                    # Aggiorna layout del plot 3D
                    fig_3d.update_traces(marker_size=point_size_3d)
                    
                    if show_labels_3d != "None":
                        fig_3d.update_traces(textposition="top center")
                    
                    # Configurazione avanzata del layout 3D
                    scene_dict = dict(
                        xaxis_title=f'{pc_x} ({var_x:.1f}%)',
                        yaxis_title=f'{pc_y} ({var_y:.1f}%)',
                        zaxis_title=f'{pc_z} ({var_z:.1f}%)',
                        camera=dict(eye=dict(x=1.2, y=1.2, z=1.2))  # Vista default
                    )
                    
                    # Aggiungi piani degli assi se richiesto
                    if show_axes_3d:
                        scene_dict.update({
                            'xaxis': dict(
                                title=f'{pc_x} ({var_x:.1f}%)',
                                showgrid=True,
                                zeroline=True,
                                zerolinecolor='gray',
                                zerolinewidth=2
                            ),
                            'yaxis': dict(
                                title=f'{pc_y} ({var_y:.1f}%)',
                                showgrid=True,
                                zeroline=True,
                                zerolinecolor='gray',
                                zerolinewidth=2
                            ),
                            'zaxis': dict(
                                title=f'{pc_z} ({var_z:.1f}%)',
                                showgrid=True,
                                zeroline=True,
                                zerolinecolor='gray',
                                zerolinewidth=2
                            )
                        })
                    
                    fig_3d.update_layout(
                        height=700,
                        scene=scene_dict
                    )
                    
                    st.plotly_chart(fig_3d, width='stretch')
                    
                    # Metriche varianza per 3D
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric(f"{pc_x} Variance", f"{var_x:.1f}%")
                    with col2:
                        st.metric(f"{pc_y} Variance", f"{var_y:.1f}%")
                    with col3:
                        st.metric(f"{pc_z} Variance", f"{var_z:.1f}%")
                    with col4:
                        st.metric("Combined Variance", f"{var_total_3d:.1f}%")
                    
                else:
                    st.warning("⚠️ Need at least 3 components for 3D plot")


            elif score_plot_type == "📈 Line Profiles Plot":
                st.markdown(f"### 📈 Line Profiles Plot{title_suffix}")
                
                col1, col2 = st.columns(2)
                with col1:
                    selected_comp = st.selectbox("Select component for profile:", scores.columns, index=0)
                with col2:
                    custom_vars = []
                    if 'custom_variables' in st.session_state:
                        custom_vars = list(st.session_state.custom_variables.keys())
                    
                    all_color_options = (["None", "Row Index"] + 
                                        [col for col in data.columns if col not in pca_results['parameters']['variables']] +
                                        custom_vars)
                    
                    profile_color_by = st.selectbox("Color profiles by:", all_color_options, key="profile_color")
                
                # Create line profile plot
                if profile_color_by == "None":
                    color_data = None
                elif profile_color_by == "Row Index":
                    color_data = data.index
                elif profile_color_by in custom_vars:
                    color_data = st.session_state.custom_variables[profile_color_by]
                else:
                    color_data = data[profile_color_by]
                
                fig = go.Figure()
                
                if profile_color_by == "None":
                    fig.add_trace(go.Scatter(
                        x=list(range(len(scores))), y=scores[selected_comp],
                        mode='lines+markers', name=f'{selected_comp} Profile',
                        line=dict(width=2), marker=dict(size=4),
                        text=scores.index
                    ))
                else:
                    if profile_color_by in custom_vars and ('Time' in profile_color_by or 'time' in profile_color_by):
                        fig.add_trace(go.Scatter(
                            x=list(range(len(scores))), y=scores[selected_comp],
                            mode='lines+markers', name=f'{selected_comp} Profile',
                            marker=dict(color=color_data, colorscale='RdBu_r', showscale=True),
                            text=scores.index
                        ))
                    else:
                        unique_groups = pd.Series(color_data).dropna().unique()
                        for group in unique_groups:
                            group_mask = pd.Series(color_data) == group
                            group_indices = [i for i, mask in enumerate(group_mask) if mask]
                            group_scores = scores[selected_comp][group_mask]
                            
                            fig.add_trace(go.Scatter(
                                x=group_indices, y=group_scores,
                                mode='lines+markers', name=f'{group}',
                                line=dict(width=2), marker=dict(size=4)
                            ))
                
                fig.update_layout(
                    title=f"Line Profile: {selected_comp} Scores{title_suffix}",
                    xaxis_title="Sample Index",
                    yaxis_title=f"{selected_comp} Score",
                    height=500
                )
                
                fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.7)
                st.plotly_chart(fig, width='stretch')
                
                # Statistics
                st.markdown("#### 📊 Profile Statistics")
                col_stats1, col_stats2, col_stats3, col_stats4 = st.columns(4)
                
                with col_stats1:
                    st.metric("Mean", f"{scores[selected_comp].mean():.3f}")
                with col_stats2:
                    st.metric("Std Dev", f"{scores[selected_comp].std():.3f}")
                with col_stats3:
                    st.metric("Min", f"{scores[selected_comp].min():.3f}")
                with col_stats4:
                    st.metric("Max", f"{scores[selected_comp].max():.3f}")

    # ===== DIAGNOSTICS TAB =====
    with tab5:
        st.markdown("## 🔍 PCA Diagnostics")

        if 'pca_model' not in st.session_state:
            st.warning("⚠️ No PCA model computed. Please compute a model first.")
        else:
            pca_results = st.session_state.pca_model
            is_varimax = pca_results.get('method') == 'Varimax Rotation'
            
            # Extended diagnostic options
            diagnostic_options = [
                "📊 Model Quality Metrics", 
                "🧠 Integrated PCA Interpretation",  # Rinominato per chiarezza
                "🎯 Factor Interpretation" if is_varimax else "📈 Component Analysis"
            ]
            
            diagnostic_type = st.selectbox(
                "Select diagnostic type:",
                diagnostic_options
            )
            
            # =================================================================
            # INTEGRATED PCA INTERPRETATION (NO AI)
            # =================================================================
            if diagnostic_type == "🧠 Integrated PCA Interpretation":
                st.markdown("### 🧠 Integrated PCA Interpretation")
                st.markdown("*Geometric interpretation based on PCA theory*")
                
                # PCA Theory
                with st.expander("📚 PCA Interpretation Theory", expanded=False):
                    st.markdown("""
                    **Geometric PCA Interpretation:**

                    PCA finds directions (PCs) that explain the maximum variance in the data.

                    **Loadings (coefficients):**
                    - **Distance from origin** = variable importance for that PC
                    - **Variables close together** = positively correlated
                    - **Variables in opposite directions** = negatively correlated (anticorrelated)
                    - **Orthogonal variables** = uncorrelated

                    **Scores (sample coordinates):**
                    - **Samples close together** = similar characteristics
                    - **Samples far apart** = different characteristics
                    - **Outliers** = samples with atypical characteristics
                    - **Clusters** = natural groups with common properties

                    The integrated interpretation connects loadings and scores to understand
                    WHY samples distribute as observed.
                    """)
                
                try:
                    # Import the new interpretation module
                    from pca_ai_utils import (
                        interpret_pca_geometry,
                        analyze_pca_complete,
                        quick_pca_interpretation
                    )
                    
                    # Get PCA data
                    loadings = pca_results.get('loadings', pd.DataFrame())
                    scores = pca_results.get('scores', pd.DataFrame())
                    
                    # Validation
                    if loadings.empty or scores.empty:
                        st.error("❌ Missing loadings or scores data")
                    else:
                        st.success("✅ PCA data ready for interpretation")

                        # =================================================================
                        # CONFIGURATION SECTION
                        # =================================================================
                        with st.expander("⚙️ Analysis Configuration", expanded=True):
                            st.markdown("### Configure Analysis Parameters")

                            col_config1, col_config2, col_config3 = st.columns(3)

                            available_pcs = [int(col.replace('PC', '').replace('Factor', ''))
                                            for col in loadings.columns
                                            if col.startswith(('PC', 'Factor'))]

                            with col_config1:
                                pc_x = st.selectbox(
                                    "X-axis Component:",
                                    options=available_pcs,
                                    index=0,
                                    help="First component for biplot analysis"
                                )

                            with col_config2:
                                pc_y = st.selectbox(
                                    "Y-axis Component:",
                                    options=available_pcs,
                                    index=1 if len(available_pcs) > 1 else 0,
                                    help="Second component for biplot analysis"
                                )

                            with col_config3:
                                threshold = st.slider(
                                    "Significance threshold:",
                                    min_value=0.1, max_value=0.8,
                                    value=0.3, step=0.05,
                                    help="Minimum loading magnitude for significance"
                                )

                            # Data type
                            data_type = st.selectbox(
                                "Data type (for contextualization):",
                                ["Generic", "Spectroscopy/NIR", "Chemical Parameters",
                                "Process Data", "Quality Analysis", "Materials",
                                "Pharmaceutical", "Food", "Environmental"],
                                help="Helps provide domain-specific interpretation"
                            )

                        # =================================================================
                        # METAVARIABLE SELECTION
                        # =================================================================
                        with st.expander("🏷️ Sample Grouping (Optional)", expanded=False):
                            st.markdown("### Select Metavariable for Enhanced Interpretation")
                            st.info("Choose a metavariable to analyze how sample groups distribute in the PCA space. This helps understand if samples with the same code cluster together.")

                            # Collect available metavariables
                            available_metavars = ["None"]

                            # Get non-numeric columns from current_data (metadata)
                            if 'current_data' in st.session_state:
                                data = st.session_state.current_data
                                non_numeric_cols = data.select_dtypes(exclude=['number']).columns.tolist()
                                available_metavars.extend(non_numeric_cols)

                            # Get custom variables
                            if 'custom_variables' in st.session_state and st.session_state.custom_variables:
                                custom_vars = list(st.session_state.custom_variables.keys())
                                available_metavars.extend(custom_vars)

                            # Metavariable selector
                            selected_metavar = st.selectbox(
                                "Select metavariable for sample grouping:",
                                options=available_metavars,
                                help="Choose a metavariable to help interpret sample groupings (e.g., batch, treatment, etc.)"
                            )

                            # Get metavariable data if selected
                            metavar_data = None
                            if selected_metavar != "None":
                                if 'current_data' in st.session_state and selected_metavar in st.session_state.current_data.columns:
                                    metavar_data = st.session_state.current_data[selected_metavar]
                                elif 'custom_variables' in st.session_state and selected_metavar in st.session_state.custom_variables:
                                    metavar_data = st.session_state.custom_variables[selected_metavar]

                                if metavar_data is not None:
                                    # Align metavar_data with scores index
                                    metavar_data = metavar_data.reindex(scores.index)

                                    # Show preview of groups
                                    unique_groups = metavar_data.dropna().unique()
                                    st.success(f"✅ Using '{selected_metavar}' with {len(unique_groups)} unique groups")

                                    # Show group counts
                                    group_counts = metavar_data.value_counts()
                                    st.write("**Group distribution:**")
                                    for group, count in group_counts.items():
                                        st.write(f"  - {group}: {count} samples")

                        # =================================================================
                        # RUN INTERPRETATION
                        # =================================================================
                        st.markdown("---")
                        st.markdown("### 🚀 Generate Interpretation")
                        
                        if st.button("📊 **Generate Full Interpretation**", type="primary"):

                            with st.spinner(f"Analyzing PC{pc_x} vs PC{pc_y}..."):

                                # Run geometric interpretation
                                interpretation = analyze_pca_complete(
                                    loadings=loadings,
                                    scores=scores,
                                    pc_x=pc_x,
                                    pc_y=pc_y,
                                    threshold=threshold,
                                    data_type=data_type,
                                    metavar_data=metavar_data,
                                    metavar_name=selected_metavar if selected_metavar != "None" else None
                                )
                                
                                st.success("✅ Interpretation completed!")
                                
                                # Display full interpretation
                                st.markdown("---")
                                st.markdown(interpretation)
                                
                                # Export option
                                st.download_button(
                                    "📥 Download Interpretation Report",
                                    interpretation,
                                    f"PCA_Interpretation_PC{pc_x}_vs_PC{pc_y}.md",
                                    "text/markdown",
                                    key="download_interpretation"
                                )
                        
                        # =================================================================
                        # QUICK ANALYSIS TOOLS
                        # =================================================================
                        st.markdown("---")

                        # Quick Statistics Expander
                        with st.expander("📈 Quick Statistics", expanded=False):
                            st.markdown("### Quick Statistical Summary")

                            if st.button("🔄 Calculate Statistics", key="calc_stats"):
                                # Get quick stats using the geometric analysis
                                result = interpret_pca_geometry(
                                    loadings, scores, pc_x, pc_y, threshold
                                )

                                if result['success']:
                                    loadings_interp = result['loadings_interpretation']
                                    scores_interp = result['scores_interpretation']

                                    col1, col2 = st.columns(2)

                                    with col1:
                                        st.markdown("**📊 Loadings Analysis:**")
                                        st.metric("Significant Variables", loadings_interp['n_significant'])

                                        # Top variables
                                        top_vars = list(loadings_interp['importance_ranking'].keys())[:3]
                                        st.write(f"**Top 3 Variables:**")
                                        for i, var in enumerate(top_vars, 1):
                                            st.write(f"{i}. {var}")

                                        # Variable correlations
                                        n_corr = len(loadings_interp['correlations']['strongly_correlated'])
                                        n_anti = len(loadings_interp['correlations']['anticorrelated'])
                                        st.metric("Strong Correlations", f"{n_corr} pairs")
                                        st.metric("Anticorrelations", f"{n_anti} pairs")

                                    with col2:
                                        st.markdown("**📈 Scores Analysis:**")
                                        st.metric("Total Samples", scores_interp['n_samples'])
                                        st.metric("Outliers", scores_interp['n_outliers'])
                                        st.metric("Natural Clusters", len(scores_interp['clusters']))

                                        # Distribution
                                        dist = scores_interp['distribution']
                                        st.metric("X-axis Spread (std)", f"{dist['statistics']['x_spread']:.3f}")
                                        st.metric("Y-axis Spread (std)", f"{dist['statistics']['y_spread']:.3f}")

                        # Variable Importance Expander
                        with st.expander("🎯 Variable Importance Ranking", expanded=False):
                            st.markdown("### Variable Importance by Distance from Origin")

                            if st.button("🔄 Calculate Importance", key="calc_importance"):
                                result = interpret_pca_geometry(
                                    loadings, scores, pc_x, pc_y, threshold
                                )

                                if result['success']:
                                    importance = result['loadings_interpretation']['importance_ranking']

                                    # Create bar chart of importance
                                    top_n = min(15, len(importance))
                                    vars_to_plot = list(importance.keys())[:top_n]
                                    values_to_plot = [importance[v] for v in vars_to_plot]

                                    fig = go.Figure(data=[
                                        go.Bar(
                                            x=values_to_plot,
                                            y=vars_to_plot,
                                            orientation='h',
                                            marker_color='lightblue'
                                        )
                                    ])

                                    fig.update_layout(
                                        title=f"Top {top_n} Variables by Importance",
                                        xaxis_title="Distance from Origin",
                                        yaxis_title="Variable",
                                        height=400
                                    )

                                    st.plotly_chart(fig, use_container_width=True)

                                    # Show table with values
                                    st.markdown("**Importance Values:**")
                                    importance_df = pd.DataFrame({
                                        'Variable': vars_to_plot,
                                        'Importance': values_to_plot
                                    })
                                    st.dataframe(importance_df, use_container_width=True)
                        
                        # =================================================================
                        # INTERPRETATION GUIDELINES
                        # =================================================================
                        with st.expander("💡 Interpretation Guidelines", expanded=False):
                            st.markdown("### How to Interpret PCA Plots")

                            col_guide1, col_guide2 = st.columns(2)

                            with col_guide1:
                                st.info("""
                                **Loading Plot Interpretation:**
                                - Variables far from origin are most important
                                - Variables in same direction are correlated
                                - Variables in opposite directions are anticorrelated
                                - Angle between variables indicates correlation strength
                                """)

                            with col_guide2:
                                st.info("""
                                **Score Plot Interpretation:**
                                - Sample position indicates its characteristics
                                - Samples in direction of a variable have high values for it
                                - Clusters indicate groups with similar properties
                                - Outliers may indicate measurement errors or special conditions
                                """)

                            st.markdown("---")
                            st.markdown("**Additional Tips:**")
                            st.write("""
                            - **Biplot**: Overlay loadings and scores to see relationships
                            - **Variance Explained**: Check cumulative variance to determine if enough PCs are used
                            - **Q² and R²**: Validate model quality with cross-validation metrics
                            - **Metavariable Analysis**: Use sample grouping to understand if known factors explain PCA structure
                            """)
                    
                except ImportError as e:
                    st.error("❌ Interpretation module not properly configured")
                    st.info("""
                    **To enable integrated interpretation:**
                    
                    1. Ensure `pca_ai_utils.py` is in the same directory as `pca.py`
                    2. The module provides pure geometric interpretation (no AI needed)
                    3. Based on established PCA theory for reliable results
                    """)
                    
                    # Fallback: show basic statistics
                    st.markdown("#### 📊 Basic Statistics (Fallback)")
                    
                    loadings = pca_results.get('loadings', pd.DataFrame())
                    scores = pca_results.get('scores', pd.DataFrame())
                    
                    if not loadings.empty and not scores.empty:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**Loadings Summary:**")
                            st.write(f"- Shape: {loadings.shape}")
                            st.write(f"- Variables: {len(loadings)}")
                            st.write(f"- Components: {len(loadings.columns)}")
                        
                        with col2:
                            st.markdown("**Scores Summary:**")
                            st.write(f"- Shape: {scores.shape}")
                            st.write(f"- Samples: {len(scores)}")
                            st.write(f"- Components: {len(scores.columns)}")
            
            # =================================================================
            # MODEL QUALITY METRICS (UNCHANGED)
            # =================================================================
            elif diagnostic_type == "📊 Model Quality Metrics":
                st.markdown("### 📊 Model Quality Metrics")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### Variance Criteria")
                    kaiser_components = sum(pca_results['eigenvalues'] > 1)
                    st.metric("Kaiser Criterion Components", kaiser_components)
                    
                    components_80 = sum(pca_results['cumulative_variance'] < 0.8) + 1
                    st.metric("Components for 80% Variance", components_80)
                    
                    components_95 = sum(pca_results['cumulative_variance'] < 0.95) + 1
                    st.metric("Components for 95% Variance", components_95)
                    
                    if is_varimax:
                        st.metric("Varimax Iterations", pca_results.get('varimax_iterations', 'N/A'))
                
                with col2:
                    st.markdown("#### Loading Statistics")
                    loadings = pca_results['loadings']
                    first_comp = loadings.columns[0]
                    
                    st.metric(f"Max Loading ({first_comp})", f"{loadings.iloc[:, 0].abs().max():.3f}")
                    st.metric(f"Min Loading ({first_comp})", f"{loadings.iloc[:, 0].abs().min():.3f}")
                    st.metric(f"Loading Range ({first_comp})", 
                            f"{loadings.iloc[:, 0].max() - loadings.iloc[:, 0].min():.3f}")
                    
                    if is_varimax:
                        loadings_squared = loadings.values ** 2
                        simple_structure = np.mean(np.var(loadings_squared, axis=1))
                        st.metric("Simple Structure Index", f"{simple_structure:.3f}")
            
            # Other diagnostic options...
            else:
                # Component/Factor analysis
                st.markdown(f"### {'🎯 Factor' if is_varimax else '📈 Component'} Analysis")
                st.info("Detailed component/factor interpretation coming soon")

    # ===== EXTRACT & EXPORT TAB =====
    with tab6:
        st.markdown("## 💤 Extract & Export")
        st.markdown("*Equivalent to PCA_extract.r*")
        
        if 'pca_model' not in st.session_state:
            st.warning("⚠️ No PCA model computed. Please compute a model first.")
        else:
            pca_results = st.session_state.pca_model
            is_varimax = pca_results.get('method') == 'Varimax Rotation'
            
            st.markdown("### 📊 Available Data for Export")
            
            # Export options
            score_label = "📊 Factor Scores" if is_varimax else "📊 Scores"
            loading_label = "📈 Factor Loadings" if is_varimax else "📈 Loadings"
            component_label = "Factor" if is_varimax else "Component"
            
            export_options = {
                score_label: pca_results['scores'],
                loading_label: pca_results['loadings'],
                "📋 Variance Summary": pd.DataFrame({
                    component_label: pca_results['scores'].columns.tolist(),
                    'Eigenvalue': pca_results['eigenvalues'],
                    'Variance_Ratio': pca_results['explained_variance_ratio'],
                    'Cumulative_Variance': pca_results['cumulative_variance']
                })
            }
            
            for name, df in export_options.items():
                with st.expander(f"{name} ({df.shape[0]}×{df.shape[1]})"):
                    st.dataframe(df, use_container_width=True)
                    
                    # Download button
                    csv_data = df.to_csv(index=True)
                    method_suffix = "_Varimax" if is_varimax else "_PCA"
                    clean_name = name.replace("📊 ", "").replace("📈 ", "").replace("📋 ", "").replace(" ", "_")
                    filename = f"PCA_{clean_name}{method_suffix}.csv"
                    
                    st.download_button(
                        f"💾 Download {name} as CSV",
                        csv_data, filename, "text/csv",
                        key=f"download_{clean_name}"
                    )
            
            # Model parameters export
            st.markdown("### ⚙️ Model Parameters")
            
            params = [
                ['Analysis Method', pca_results.get('method', 'Standard PCA')],
                ['Number of Components/Factors', pca_results['parameters']['n_components']],
                ['Data Centering', pca_results['parameters']['center']],
                ['Data Scaling', pca_results['parameters']['scale']],
                ['Number of Variables', len(pca_results['parameters']['variables'])],
                ['Number of Objects', len(pca_results['scores'])]
            ]
            
            if is_varimax:
                params.append(['Varimax Iterations', pca_results.get('varimax_iterations', 'N/A')])
            
            params_df = pd.DataFrame(params, columns=['Parameter', 'Value'])
            st.dataframe(params_df, use_container_width=True)
            
            # Export all results as single file
            if st.button("📦 Export Complete Analysis"):
                try:
                    from io import BytesIO
                    
                    excel_buffer = BytesIO()
                    method_name = "Varimax" if is_varimax else "PCA"
                    
                    with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                        # Export all data to different sheets
                        pca_results['scores'].to_excel(writer, sheet_name='Scores', index=True)
                        pca_results['loadings'].to_excel(writer, sheet_name='Loadings', index=True)
                        export_options["📋 Variance Summary"].to_excel(writer, sheet_name='Variance', index=False)
                        params_df.to_excel(writer, sheet_name='Parameters', index=False)
                        
                        # Add original data reference
                        summary_data = pd.DataFrame({
                            'Analysis_Type': [method_name],
                            'Total_Variance_Explained': [f"{pca_results['cumulative_variance'][-1]:.1%}"],
                            'Export_Date': [pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')]
                        })
                        summary_data.to_excel(writer, sheet_name='Summary', index=False)
                    
                    excel_buffer.seek(0)
                    
                    st.download_button(
                        f"📄 Download Complete {method_name} Analysis (XLSX)",
                        excel_buffer.getvalue(),
                        f"Complete_{method_name}_Analysis.xlsx",
                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                    
                    st.success(f"✅ Complete {method_name} analysis ready for download!")
                    
                except Exception as e:
                    st.error(f"Excel export failed: {str(e)}")
                    st.info("Individual CSV exports are still available above")

    # ===== ADVANCED DIAGNOSTICS TAB =====
    with tab7:
        st.markdown("## 🔬 Advanced PCA Diagnostics")

        
        if not DIAGNOSTICS_AVAILABLE:
            st.warning("⚠️ Advanced diagnostics module not available in this demo")
            st.info("""
            🔬 **Want full T² vs Q diagnostics, multivariate control charts, and process monitoring?**
            
            Professional versions include complete diagnostic suites:
            
            ✅ Real-time process monitoring  
            ✅ Advanced outlier detection  
            ✅ Multivariate control charts  
            ✅ MSPC (Multivariate SPC)  
            ✅ Custom alert systems  
            
            📞 **Contact:** [chemometricsolutions.com](https://chemometricsolutions.com)
            """)
        elif 'pca_model' not in st.session_state:
            st.warning("⚠️ No PCA model computed. Please compute a model first.")
        else:
            pca_results = st.session_state.pca_model
            
            # Check if we have the necessary data for diagnostics
            if 'processed_data' not in pca_results or 'scores' not in pca_results:
                st.error("❌ PCA model missing required data for diagnostics")
            else:
                processed_data = pca_results['processed_data']
                scores = pca_results['scores']
                
                # Call the advanced diagnostics function
                show_advanced_diagnostics_tab(
                    processed_data=processed_data,
                    scores=scores,
                    pca_params=pca_results,
                    timestamps=None,
                    start_sample=1
                )

                