"""
CAT PCA Analysis Page
Equivalent to PCA_* R scripts for Principal Component Analysis
Enhanced with Varimax rotation and advanced diagnostics
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.figure_factory as ff
from scipy.stats import f, t, chi2
import json
import os
from pathlib import Path



# Con questo:
import sys
import os
try:
    # Aggiungi il path della directory parent
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(parent_dir)
    from pca_diagnostics_complete import show_advanced_diagnostics_tab
    DIAGNOSTICS_AVAILABLE = True
except ImportError as e:
    DIAGNOSTICS_AVAILABLE = False
    print(f"Import error: {e}")

# Aggiungi questo import all'inizio del file
from color_utils import get_unified_color_schemes, create_categorical_color_map

# Sostituisci la funzione get_custom_color_map esistente con:
def get_custom_color_map(theme='auto'):
    """
    Mappa colori personalizzata per variabili categoriche - VERSIONE UNIFICATA
    theme: 'light', 'dark', o 'auto'
    """
    if theme == 'light':
        return get_unified_color_schemes(dark_mode=False)['color_map']
    elif theme == 'dark':
        return get_unified_color_schemes(dark_mode=True)['color_map']
    else:
        # Auto: usa tema scuro come default
        return get_unified_color_schemes(dark_mode=True)['color_map']

def save_workspace_to_file():
    """Salva il workspace su file JSON"""
    if 'split_datasets' in st.session_state:
        workspace_data = {}
        for name, info in st.session_state.split_datasets.items():
            # Salva i metadati e il CSV data
            workspace_data[name] = {
                'type': info['type'],
                'parent': info['parent'],
                'n_samples': info['n_samples'],
                'creation_time': info['creation_time'].isoformat(),
                'csv_data': info['data'].to_csv(index=True)
            }
        
        # Salva su file
        workspace_file = Path("pca_workspace.json")
        with open(workspace_file, 'w') as f:
            json.dump(workspace_data, f, indent=2)
        
        return True
    return False

def load_workspace_from_file():
    """Carica il workspace da file JSON"""
    workspace_file = Path("pca_workspace.json")
    if workspace_file.exists():
        try:
            with open(workspace_file, 'r') as f:
                workspace_data = json.load(f)
            
            # Ricostruisci i dataset
            st.session_state.split_datasets = {}
            for name, info in workspace_data.items():
                # Ricostruisci il DataFrame dal CSV
                from io import StringIO
                csv_data = StringIO(info['csv_data'])
                df = pd.read_csv(csv_data, index_col=0)
                
                st.session_state.split_datasets[name] = {
                    'data': df,
                    'type': info['type'],
                    'parent': info['parent'],
                    'n_samples': info['n_samples'],
                    'creation_time': pd.Timestamp.fromisoformat(info['creation_time'])
                }
            
            return True
        except Exception as e:
            st.error(f"Error loading workspace: {str(e)}")
            return False
    return False

def get_custom_color_map(theme='auto'):
    """
    Mappa colori personalizzata per variabili categoriche
    theme: 'light', 'dark', o 'auto'
    """
    
    # Colori per tema chiaro (sfondo bianco)
    light_theme_colors = {
        'A': 'black',
        'B': 'red',
        'C': 'green',
        'D': 'blue',
        'E': 'orange',
        'F': 'purple',
        'G': 'brown',
        'H': 'hotpink',
        'I': 'gray',
        'J': 'olive',
        'K': 'cyan',
        'L': 'magenta',
        'M': 'gold',
        'N': 'navy',
        'O': 'darkgreen',
        'P': 'darkred',
        'Q': 'indigo',
        'R': 'coral',
        'S': 'teal',
        'T': 'chocolate',
        'U': 'crimson',
        'V': 'darkviolet',
        'W': 'darkorange',
        'X': 'darkslategray',
        'Y': 'royalblue',
        'Z': 'saddlebrown'
    }
    
    # Colori per tema scuro (sfondo nero/grigio scuro)
    dark_theme_colors = {
        'A': 'white',
        'B': 'lightcoral',
        'C': 'lightgreen',
        'D': 'lightblue',
        'E': 'orange',
        'F': 'violet',
        'G': 'tan',
        'H': 'hotpink',
        'I': 'lightgray',
        'J': 'yellowgreen',
        'K': 'cyan',
        'L': 'magenta',
        'M': 'gold',
        'N': 'cornflowerblue',
        'O': 'lime',
        'P': 'tomato',
        'Q': 'mediumpurple',
        'R': 'coral',
        'S': 'turquoise',
        'T': 'sandybrown',
        'U': 'crimson',
        'V': 'plum',
        'W': 'darkorange',
        'X': 'lightsteelblue',
        'Y': 'royalblue',
        'Z': 'wheat'
    }
    
    if theme == 'light':
        return light_theme_colors
    elif theme == 'dark':
        return dark_theme_colors
    else:
        # Auto: usa tema scuro come default (più comune)
        return dark_theme_colors

def varimax_rotation(loadings, gamma=1.0, max_iter=100, tol=1e-6):
    """
    Perform Varimax rotation on loading matrix
    Equivalent to PCA_model_varimax.r algorithm
    """
    p, k = loadings.shape
    rotated_loadings = loadings.copy()
    
    # Iterative rotation algorithm from R script
    converged = False
    iteration = 0
    
    while not converged and iteration < max_iter:
        prev_loadings = rotated_loadings.copy()
        
        # Pairwise rotations (equivalent to nested loops in R script)
        for i in range(k-1):
            for j in range(i+1, k):
                # Extract two columns for rotation
                lo = rotated_loadings[:, [i, j]]
                
                # Find optimal rotation angle
                best_angle = 0
                best_criterion = np.sum(lo**4)
                
                # Search for best rotation angle (equivalent to R loop)
                for angle_deg in np.arange(-90, 90.1, 0.1):
                    angle_rad = angle_deg * np.pi / 180
                    
                    # Rotation matrix
                    cos_a = np.cos(angle_rad)
                    sin_a = np.sin(angle_rad)
                    rotation_matrix = np.array([[cos_a, -sin_a], 
                                              [sin_a, cos_a]])
                    
                    # Apply rotation
                    rotated_pair = lo @ rotation_matrix
                    criterion = np.sum(rotated_pair**4)
                    
                    if criterion > best_criterion:
                        best_criterion = criterion
                        best_angle = angle_deg
                
                # Apply best rotation if improvement found
                if best_angle != 0:
                    angle_rad = best_angle * np.pi / 180
                    cos_a = np.cos(angle_rad)
                    sin_a = np.sin(angle_rad)
                    rotation_matrix = np.array([[cos_a, -sin_a], 
                                              [sin_a, cos_a]])
                    
                    rotated_loadings[:, [i, j]] = lo @ rotation_matrix
        
        # Check convergence
        if np.allclose(rotated_loadings, prev_loadings, atol=tol):
            converged = True
        
        iteration += 1
    
    return rotated_loadings, iteration

def add_convex_hulls(fig, scores, pc_x, pc_y, color_data, color_discrete_map=None, hull_opacity=0.7, dark_mode=True):
    """
    Aggiunge convex hull per ogni gruppo categorico nel plot PCA
    Bordi sottili e continui - VERSIONE UNIFICATA
    
    Args:
        fig: plotly figure
        scores: DataFrame con i punteggi PCA
        pc_x, pc_y: nomi delle colonne per gli assi
        color_data: dati categorici per i colori
        color_discrete_map: mappa colori personalizzata (opzionale)
        hull_opacity: opacità delle linee hull
        dark_mode: se True usa colori per tema scuro
    """
    try:
        from scipy.spatial import ConvexHull
        import numpy as np
        
        if color_data is None:
            return fig
        
        # Converte color_data in Series
        if hasattr(color_data, 'index'):
            color_series = pd.Series(color_data, index=color_data.index)
        else:
            color_series = pd.Series(color_data, index=scores.index)
        
        color_series = color_series.reindex(scores.index)
        unique_groups = color_series.dropna().unique()
        
        if len(unique_groups) == 0:
            return fig
        
        # USA SISTEMA DI COLORI UNIFICATO se non fornita mappa personalizzata
        if color_discrete_map is None:
            color_discrete_map = create_categorical_color_map(unique_groups, dark_mode)
        
        hulls_added = 0
        
        # Calcola convex hull per ogni gruppo
        for group in unique_groups:
            group_mask = color_series == group
            n_points = group_mask.sum()
            
            if n_points < 3:
                continue
            
            # Estrai coordinate
            group_scores_x = scores.loc[group_mask, pc_x].values
            group_scores_y = scores.loc[group_mask, pc_y].values
            group_points = np.column_stack([group_scores_x, group_scores_y])
            
            try:
                # Calcola convex hull
                hull = ConvexHull(group_points)
                hull_vertices = hull.vertices
                hull_points = group_points[hull_vertices]
                
                # Chiudi il poligono
                hull_x = np.append(hull_points[:, 0], hull_points[0, 0])
                hull_y = np.append(hull_points[:, 1], hull_points[0, 1])
                
                # Ottieni colore del gruppo dal sistema unificato
                group_color = color_discrete_map.get(group, 'gray')
                
                # Bordi sottili e continui
                fig.add_trace(go.Scatter(
                    x=hull_x, 
                    y=hull_y,
                    mode='lines',
                    line=dict(
                        color=group_color, 
                        width=1  # Sottile e continuo
                    ),
                    opacity=hull_opacity,
                    fill=None,
                    name=f'{group}_hull',
                    showlegend=False,
                    hoverinfo='skip'
                ))
                
                hulls_added += 1
                
            except Exception as hull_error:
                print(f"Hull error for {group}: {hull_error}")
                continue
        
        return fig
        
    except ImportError:
        print("scipy not available")
        return fig
    except Exception as e:
        print(f"Error in add_convex_hulls: {e}")
        return fig
    
def show():
    """Display the PCA Analysis page"""
    
    st.markdown("# 🎯 Principal Component Analysis (PCA)")
    st.markdown("*Complete PCA analysis suite equivalent to PCA_* R scripts*")
    
    if 'current_data' not in st.session_state:
        st.warning("⚠️ No data loaded. Please go to Data Handling to load your dataset first.")
        return
    
    data = st.session_state.current_data
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "🔧 Model Computation",
        "📊 Variance Plots", 
        "📈 Loadings Plots",
        "🎯 Scores Plots",
        "🔍 Diagnostics",
        "👤 Extract & Export",
        "🔬 Advanced Diagnostics"
    ])
    # ===== MODEL COMPUTATION TAB =====
    with tab1:
        st.markdown("## 🔧 PCA Model Computation")
        st.markdown("*Equivalent to PCA_model_PCA.r and PCA_model_varimax.r*")
        
        # Data selection
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 Data Selection")
            
            # Variable selection
            numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
            if not numeric_columns:
                st.error("❌ No numeric columns found in the dataset!")
                return
            
            # GESTIONE SPECIALE PER DATI SPETTRALI (molte variabili)
            if len(numeric_columns) > 100:
                st.info(f"🔬 **Spectral data detected**: {len(numeric_columns)} variables")
                st.markdown("#### Variable Range Selection")
                
                # Opzioni per dati spettrali
                selection_method = st.selectbox(
                    "Variable selection method:",
                    ["All variables", "Range selection", "Every N variables", "Manual list"]
                )
                
                if selection_method == "All variables":
                    selected_vars = numeric_columns
                    
                elif selection_method == "Range selection":
                    col_range1, col_range2 = st.columns(2)
                    with col_range1:
                        start_idx = st.number_input("Start variable index:", 0, len(numeric_columns)-1, 0)
                    with col_range2:
                        end_idx = st.number_input("End variable index:", start_idx+1, len(numeric_columns), len(numeric_columns))
                    
                    selected_vars = numeric_columns[start_idx:end_idx]
                    st.info(f"Selected {len(selected_vars)} variables (index {start_idx} to {end_idx-1})")
                    
                elif selection_method == "Every N variables":
                    step_size = st.number_input("Select every N variables:", 1, 50, 5)
                    selected_vars = numeric_columns[::step_size]
                    st.info(f"Selected {len(selected_vars)} variables (every {step_size} variables)")
                    
                else:  # Manual list
                    var_indices = st.text_input(
                        "Variable indices (comma-separated, e.g., 1,5,10-20,25):",
                        value="1,10,50,100"
                    )
                    
                    # Parse manual selection
                    try:
                        selected_indices = []
                        for part in var_indices.split(','):
                            part = part.strip()
                            if '-' in part:
                                start, end = map(int, part.split('-'))
                                selected_indices.extend(range(start-1, end))  # Convert to 0-based
                            else:
                                selected_indices.append(int(part)-1)  # Convert to 0-based
                        
                        # Remove duplicates and sort
                        selected_indices = sorted(list(set(selected_indices)))
                        # Filter valid indices
                        selected_indices = [i for i in selected_indices if 0 <= i < len(numeric_columns)]
                        selected_vars = [numeric_columns[i] for i in selected_indices]
                        
                        st.info(f"Selected {len(selected_vars)} variables from manual indices")
                        
                    except Exception as e:
                        st.error(f"Invalid format: {e}")
                        selected_vars = numeric_columns[:10]
                
            else:
                # GESTIONE NORMALE per pochi variabili
                select_all_vars = st.checkbox("Select all variables", value=True)

                if select_all_vars:
                    default_vars = numeric_columns
                else:
                    default_vars = numeric_columns[:min(10, len(numeric_columns))]

                selected_vars = st.multiselect(
                    "Select variables for PCA:",
                    numeric_columns,
                    default=default_vars,
                    key="pca_variable_selection"
                )
            
            if not selected_vars:
                st.warning("⚠️ Please select at least 2 variables for PCA")
                return
                
            # Object selection
            use_all_objects = st.checkbox("Use all objects", value=True)
            if not use_all_objects:
                n_objects = st.slider("Number of objects:", 5, len(data), len(data))
                selected_data = data[selected_vars].iloc[:n_objects]
            else:
                selected_data = data[selected_vars]  
                
        with col2:
            st.markdown("### ⚙️ PCA Parameters")
            
            # Preprocessing options
            center_data = st.checkbox("Center data", value=True)
            scale_data = st.checkbox("Scale data (unit variance)", value=True)
            
            # PCA Method selection (KEY ADDITION)
            pca_method = st.selectbox("PCA Method:", ["Standard PCA", "Varimax Rotation"])
            
            # Number of components
            max_components = min(selected_data.shape) - 1
            n_components = st.slider("Number of components:", 2, max_components, 
                                   min(5, max_components))
            
            # Varimax specific options
            if pca_method == "Varimax Rotation":
                st.info("🔄 Varimax rotation will maximize factor interpretability")
                st.markdown("*Equivalent to PCA_model_varimax.r*")
            
            # Advanced options
            with st.expander("🔧 Advanced Options"):
                missing_values = st.selectbox("Missing values:",["Remove", "Impute mean"])
                
                if pca_method == "Varimax Rotation":
                    max_iter_varimax = st.number_input("Max Varimax iterations:", 50, 500, 100)
                    tolerance_varimax = st.number_input("Convergence tolerance:", 1e-8, 1e-3, 1e-6, format="%.0e")
                
                # Validation options
                perform_validation = st.checkbox("Perform cross-validation", value=False)
                if perform_validation:
                    cv_folds = st.slider("CV folds:", 3, 10, 5)
        
        # Pre-processing per dati spettrali
        if len(selected_vars) > 50:  # Assumiamo dati spettrali/profili
            st.markdown("### 🔬 Spectral Data Pre-processing")
            
            with st.expander("⚙️ Data Transformation (WIP - Future Feature)"):
                st.info("🚧 **Coming Soon**: Advanced spectral pre-processing")
                st.markdown("""
                **Planned transformations:**
                - SNV (Standard Normal Variate)
                - MSC (Multiplicative Scatter Correction)
                - Detrending
                - Derivatives (1st, 2nd)
                - Baseline correction
                - Smoothing filters
                """)
                
                # Placeholder per futuro collegamento
                if st.button("🔄 Go to Pre-processing Page (WIP)", disabled=True):
                    st.info("This will redirect to a dedicated pre-processing page")
                
                st.warning("⚠️ For now, use raw data or pre-process externally")

        # Compute PCA/Varimax
        button_text = "🚀 Compute Varimax Model" if pca_method == "Varimax Rotation" else "🚀 Compute PCA Model"
        
        if st.button(button_text, type="primary"):
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
                    # Standard PCA
                    pca = PCA(n_components=n_components)
                    scores = pca.fit_transform(X_processed)
                    
                    # Store results
                    pca_results = {
                        'model': pca,
                        'scores': pd.DataFrame(
                            scores, 
                            columns=[f'PC{i+1}' for i in range(n_components)], 
                            index=X.index
                        ),
                        'loadings': pd.DataFrame(
                            pca.components_.T, 
                            columns=[f'PC{i+1}' for i in range(n_components)],
                            index=X.columns
                        ),
                        'explained_variance': pca.explained_variance_,
                        'explained_variance_ratio': pca.explained_variance_ratio_,
                        'cumulative_variance': np.cumsum(pca.explained_variance_ratio_),
                        'eigenvalues': pca.explained_variance_,
                        'original_data': X,
                        'processed_data': X_processed,
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
                    # First compute standard PCA
                    pca_full = PCA(n_components=n_components)
                    scores_initial = pca_full.fit_transform(X_processed)
                    loadings_initial = pca_full.components_.T
                    
                    # Apply Varimax rotation
                    with st.spinner("🔄 Applying Varimax rotation..."):
                        rotated_loadings, iterations = varimax_rotation(
                            loadings_initial, 
                            max_iter=max_iter_varimax if 'max_iter_varimax' in locals() else 100,
                            tol=tolerance_varimax if 'tolerance_varimax' in locals() else 1e-6
                        )
                    
                    # Calculate rotated scores
                    rotated_scores = X_processed.values @ rotated_loadings
                    
                    # Calculate variance explained by rotated factors
                    rotated_variance = np.var(rotated_scores, axis=0, ddof=1)
                    total_variance = np.sum(rotated_variance)
                    rotated_variance_ratio = rotated_variance / total_variance
                    
                    # Sort by variance explained (descending)
                    sort_idx = np.argsort(rotated_variance_ratio)[::-1]
                    rotated_loadings = rotated_loadings[:, sort_idx]
                    rotated_scores = rotated_scores[:, sort_idx]
                    rotated_variance_ratio = rotated_variance_ratio[sort_idx]
                    rotated_variance = rotated_variance[sort_idx]
                    
                    # Store Varimax results
                    pca_results = {
                        'model': pca_full,  # Keep original for reference
                        'scores': pd.DataFrame(
                            rotated_scores, 
                            columns=[f'Factor{i+1}' for i in range(n_components)],
                            index=X.index
                        ),
                        'loadings': pd.DataFrame(
                            rotated_loadings, 
                            columns=[f'Factor{i+1}' for i in range(n_components)],
                            index=X.columns
                        ),
                        'explained_variance': rotated_variance,
                        'explained_variance_ratio': rotated_variance_ratio,
                        'cumulative_variance': np.cumsum(rotated_variance_ratio),
                        'eigenvalues': rotated_variance,
                        'original_data': X,
                        'processed_data': X_processed,
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
                
                fig = go.Figure()
                
                x_labels = (pca_results['loadings'].columns.tolist() if is_varimax 
                           else [f'PC{i+1}' for i in range(len(pca_results['explained_variance_ratio']))])
                
                # Add bars
                fig.add_trace(go.Bar(
                    x=x_labels,
                    y=pca_results['explained_variance_ratio'] * 100,
                    name='Variance Explained',
                    marker_color='lightblue' if not is_varimax else 'lightgreen'
                ))
                
                # Add line
                fig.add_trace(go.Scatter(
                    x=x_labels,
                    y=pca_results['explained_variance_ratio'] * 100,
                    mode='lines+markers',
                    name='Variance Line',
                    line=dict(color='red', width=2),
                    marker=dict(size=8)
                ))
                
                x_title = "Factor Number" if is_varimax else "Principal Component"
                main_title = f"Scree Plot - Variance Explained{title_suffix}"
                
                fig.update_layout(
                    title=main_title,
                    xaxis_title=x_title,
                    yaxis_title="Variance Explained (%)",
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            elif plot_type == "📊 Cumulative Variance":
                title_suffix = " (Varimax)" if is_varimax else ""
                st.markdown(f"### 📊 Cumulative Variance Plot{title_suffix}")
                
                fig = go.Figure()
                
                x_labels = (pca_results['loadings'].columns.tolist() if is_varimax 
                           else [f'PC{i+1}' for i in range(len(pca_results['cumulative_variance']))])
                
                fig.add_trace(go.Scatter(
                    x=x_labels,
                    y=pca_results['cumulative_variance'] * 100,
                    mode='lines+markers',
                    name='Cumulative Variance',
                    line=dict(color='blue' if not is_varimax else 'green', width=3),
                    marker=dict(size=10),
                    fill='tonexty'
                ))
                
                # Add reference lines
                fig.add_hline(y=80, line_dash="dash", line_color="red", annotation_text="80%")
                fig.add_hline(y=95, line_dash="dash", line_color="orange", annotation_text="95%")
                
                x_title = "Factor Number" if is_varimax else "Principal Component"
                
                fig.update_layout(
                    title=f"Cumulative Variance Explained{title_suffix}",
                    xaxis_title=x_title,
                    yaxis_title="Cumulative Variance (%)",
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
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
                
                # Calcola i contributi pesati per i componenti significativi
                loadings = pca_results['loadings'].iloc[:, :n_significant]
                explained_variance = pca_results['explained_variance_ratio'][:n_significant]
                
                # Formula: Contributo = Σ(loading²_i × varianza_spiegata_i) per i componenti significativi
                contributions = np.zeros(len(loadings.index))
                
                for i in range(n_significant):
                    contributions += (loadings.iloc[:, i] ** 2) * explained_variance[i]
                
                # Converti in percentuale
                contributions_pct = (contributions / np.sum(contributions)) * 100
                
                # Crea il plot
                fig = go.Figure()
                
                # Ordina le variabili per contributo decrescente
                sorted_idx = np.argsort(contributions_pct)[::-1]
                sorted_vars = loadings.index[sorted_idx]
                sorted_contributions = contributions_pct[sorted_idx]
                
                fig.add_trace(go.Bar(
                    x=sorted_vars,
                    y=sorted_contributions,
                    name='Variable Contribution',
                    marker_color='darkgreen',
                    text=[f'{val:.1f}%' for val in sorted_contributions],
                    textposition='outside'
                ))
                
                fig.update_layout(
                    title=f"Variable Contributions to Total Explained Variance<br>({n_significant} significant {comp_label.lower()}s: {explained_variance.sum()*100:.1f}% total variance)",
                    xaxis_title="Variables",
                    yaxis_title="Contribution (%)",
                    height=600,
                    xaxis={'tickangle': 45}
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Step 3: Tabella dettagliata
                st.markdown("#### Step 3: Detailed Contribution Table")
                
                # Crea tabella con dettagli
                contrib_df = pd.DataFrame({
                    'Variable': loadings.index,
                    'Total_Contribution_%': contributions_pct,
                    'Cumulative_%': np.cumsum(sorted_contributions)[np.argsort(sorted_idx)]
                })
                
                # Aggiungi contributi per singolo componente
                for i in range(n_significant):
                    pc_name = loadings.columns[i]
                    contrib_df[f'{pc_name}_Loading²×Var_%'] = (loadings.iloc[:, i] ** 2) * explained_variance[i] * 100
                
                # Ordina per contributo totale
                contrib_df_sorted = contrib_df.sort_values('Total_Contribution_%', ascending=False)
                
                st.dataframe(contrib_df_sorted.round(2), use_container_width=True)
                
                # Interpretazione
                st.markdown("#### 📋 Interpretation")
                top_vars = contrib_df_sorted.head(3)['Variable'].tolist()
                total_explained = explained_variance.sum() * 100
                
                st.success(f"""
                **Key Findings:**
                - **{n_significant} significant components** explain **{total_explained:.1f}%** of total variance
                - **Top 3 contributing variables**: {', '.join(top_vars)}
                - **Top variable ({top_vars[0]})** contributes **{contrib_df_sorted.iloc[0]['Total_Contribution_%']:.1f}%** to the explained variance
                """)
                
                if n_significant >= 2:
                    st.info(f"""
                    **Contribution Breakdown:**
                    - {loadings.columns[0]} explains {explained_variance[0]*100:.1f}% of total variance
                    - {loadings.columns[1]} explains {explained_variance[1]*100:.1f}% of total variance
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
                        
                        st.plotly_chart(fig_random, use_container_width=True)
                        
                        # Summary
                        significant_components = sum(original_variance > random_ci_upper)
                        st.success(f"**Result:** {significant_components} components are statistically significant")
                        
                    except Exception as e:
                        st.error(f"Random comparison failed: {str(e)}")

    # ===== LOADINGS PLOTS TAB =====
    with tab3:
        st.markdown("## 📈 Loadings Plots")
        st.markdown("*Equivalent to PCA_loading_plot_* R scripts*")
        
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
                
                # Ottieni gli indici dei componenti selezionati
                pc_x_idx = list(loadings.columns).index(pc_x)
                pc_y_idx = list(loadings.columns).index(pc_y)
                
                # Ottieni la varianza spiegata per i componenti selezionati
                var_x = pca_results['explained_variance_ratio'][pc_x_idx] * 100
                var_y = pca_results['explained_variance_ratio'][pc_y_idx] * 100
                var_total = var_x + var_y
                
                fig = px.scatter(
                    x=loadings[pc_x],
                    y=loadings[pc_y],
                    text=loadings.index,
                    title=f"Loadings Plot: {pc_x} vs {pc_y}{title_suffix}<br>Total Explained Variance: {var_total:.1f}%",
                    labels={
                        'x': f'{pc_x} Loadings ({var_x:.1f}%)', 
                        'y': f'{pc_y} Loadings ({var_y:.1f}%)'
                    }
                )
                
                # Calcola il range massimo per rendere le scale identiche
                x_range = [loadings[pc_x].min(), loadings[pc_x].max()]
                y_range = [loadings[pc_y].min(), loadings[pc_y].max()]
                
                # Trova il range massimo assoluto
                max_abs_range = max(abs(min(x_range + y_range)), abs(max(x_range + y_range)))
                
                # Imposta range simmetrico e identico per entrambi gli assi
                axis_range = [-max_abs_range * 1.1, max_abs_range * 1.1]
                
                # Add zero lines
                fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.7)
                fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.7)
                
                fig.update_traces(textposition="top center")
                
                # CHIAVE: Imposta scale identiche e aspect ratio 1:1
                fig.update_layout(
                    height=600,
                    width=600,  # Forza aspect ratio quadrato
                    xaxis=dict(
                        range=axis_range,
                        scaleanchor="y",  # Ancora la scala X alla Y
                        scaleratio=1,     # Ratio 1:1
                        constrain="domain"
                    ),
                    yaxis=dict(
                        range=axis_range,
                        constrain="domain"
                    )
                )
                
                # Color by loading magnitude if Varimax
                if is_varimax:
                    fig.update_traces(marker=dict(
                        color=np.sqrt(loadings[pc_x]**2 + loadings[pc_y]**2),
                        colorscale='viridis',
                        showscale=True,
                        colorbar=dict(title="Loading Magnitude")
                    ))
                
                st.plotly_chart(fig, use_container_width=True)
                
                if is_varimax:
                    st.info("💡 In Varimax rotation, variables should load highly on few factors (simple structure)")
                
                # Aggiungi informazioni sulla varianza
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(f"{pc_x} Variance", f"{var_x:.1f}%")
                with col2:
                    st.metric(f"{pc_y} Variance", f"{var_y:.1f}%")
                with col3:
                    st.metric("Combined Variance", f"{var_total:.1f}%")
                
                if is_varimax:
                    st.info("💡 In Varimax rotation, variables should load highly on few factors (simple structure)")
            
            elif loading_plot_type == "📈 Loading Line Plot":
                st.markdown(f"### 📈 Loading Line Plot{title_suffix}")
                
                selected_comps = st.multiselect(
                    f"Select {'factors' if is_varimax else 'components'} to display:",
                    loadings.columns.tolist(),
                    default=loadings.columns[:3].tolist(),
                    key="loading_line_components"  # Fixed: Added unique key
                )
                
                if selected_comps:
                    fig = go.Figure()
                    
                    for comp in selected_comps:
                        fig.add_trace(go.Scatter(
                            x=list(range(len(loadings.index))),
                            y=loadings[comp],
                            mode='lines+markers',
                            name=comp,
                            text=loadings.index,
                            hovertemplate='Variable: %{text}<br>Loading: %{y:.3f}<extra></extra>'
                        ))
                    
                    fig.update_layout(
                        title=f"Loading Line Plot{title_suffix}",
                        xaxis_title="Variable Index",
                        yaxis_title="Loading Value",
                        height=500
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)

    # ===== SCORES PLOTS TAB =====
    with tab4:
        st.markdown("## 🎯 Scores Plots")
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
                ["📊 2D Scores Plot", "🎲 3D Scores Plot", "📈 Line Profiles Plot"]
            )

            if score_plot_type == "📊 2D Scores Plot":
                st.markdown(f"### 📊 2D Scores Plot{title_suffix}")
                
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
                        show_convex_hull = st.checkbox("Show convex hulls", value=True, key="show_convex_hull")
                    else:
                        show_convex_hull = False
                
                with col_vis2:
                    if color_by != "None":
                        hull_opacity = st.slider("Hull line opacity", 0.1, 1.0, 0.7, key="hull_opacity")
                    else:
                        hull_opacity = 0.7
                
                with col_vis3:
                    if color_by != "None" and not (color_by in custom_vars and ('Time' in color_by or 'time' in color_by)):
                        use_dark_theme = st.checkbox("Dark mode colors", value=True, key="dark_mode_colors")
                    else:
                        use_dark_theme = True

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
                    if color_by in custom_vars and ('Time' in color_by or 'time' in color_by):
                        fig = px.scatter(
                            x=scores[pc_x], y=scores[pc_y], color=color_data, text=text_param,
                            title=f"Scores: {pc_x} vs {pc_y} (colored by {color_by}){title_suffix}<br>Total Explained Variance: {var_total:.1f}%",
                            labels={'x': f'{pc_x} ({var_x:.1f}%)', 'y': f'{pc_y} ({var_y:.1f}%)', 'color': color_by},
                            color_continuous_scale='RdBu_r'
                        )
                    else:
                        # USA SISTEMA DI COLORI UNIFICATO
                        color_data_series = pd.Series(color_data)
                        unique_values = color_data_series.dropna().unique()
                        
                        # Crea mappa colori unificata
                        color_discrete_map = create_categorical_color_map(unique_values, use_dark_theme)
                        
                        fig = px.scatter(
                            x=scores[pc_x], y=scores[pc_y], color=color_data, text=text_param,
                            title=f"Scores: {pc_x} vs {pc_y} (colored by {color_by}){title_suffix}<br>Total Explained Variance: {var_total:.1f}%",
                            labels={'x': f'{pc_x} ({var_x:.1f}%)', 'y': f'{pc_y} ({var_y:.1f}%)', 'color': color_by},
                            color_discrete_map=color_discrete_map
                        )
                
                # Add zero lines
                fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.7)
                fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.7)

                # AGGIUNGI CONVEX HULL
                if (color_by != "None" and 
                    show_convex_hull and
                    not (color_by in custom_vars and ('Time' in color_by or 'time' in color_by))):
                    
                    try:
                        fig = add_convex_hulls(fig, scores, pc_x, pc_y, color_data, color_discrete_map, hull_opacity, use_dark_theme)
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
                
                st.plotly_chart(fig, use_container_width=True, key="pca_scores_plot")

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
                        st.info(f"Selection axes: {pc_x} vs {pc_y}")  # AGGIUNTO: mostra gli assi correnti
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
                    
                    st.plotly_chart(fig_selected, use_container_width=True, key="selection_visualization")
                    
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
                            # NUOVO: Bottone per salvare lo split nel workspace
                            if st.button("💾 Save Split to Workspace", key="save_split", type="primary"):
                                # Inizializza il dizionario se non esiste
                                if 'split_datasets' not in st.session_state:
                                    st.session_state.split_datasets = {}
                                
                                # Crea nomi unici per i dataset
                                parent_name = st.session_state.get('current_dataset', 'Dataset')
                                timestamp = pd.Timestamp.now().strftime('%H%M%S')
                                
                                selected_name = f"{parent_name}_Selected_{timestamp}"
                                remaining_name = f"{parent_name}_Remaining_{timestamp}"
                                
                                # Salva i dataset splittati
                                st.session_state.split_datasets[selected_name] = {
                                    'data': selected_data,
                                    'type': 'PCA_Selection',
                                    'parent': parent_name,
                                    'n_samples': len(selected_data),
                                    'creation_time': pd.Timestamp.now(),
                                    'selection_method': 'Coordinate' if 'coord_x_min' in st.session_state else 'Manual',
                                    'pc_axes': f"{pc_x} vs {pc_y}"
                                }
                                
                                st.session_state.split_datasets[remaining_name] = {
                                    'data': remaining_data,
                                    'type': 'PCA_Remaining',
                                    'parent': parent_name,
                                    'n_samples': len(remaining_data),
                                    'creation_time': pd.Timestamp.now(),
                                    'selection_method': 'Coordinate' if 'coord_x_min' in st.session_state else 'Manual',
                                    'pc_axes': f"{pc_x} vs {pc_y}"
                                }
                                
                                st.success(f"✅ Split saved to workspace!")
                                st.info(f"**Selected**: {selected_name} ({len(selected_data)} samples)")
                                st.info(f"**Remaining**: {remaining_name} ({len(remaining_data)} samples)")
                                st.info("📂 Go to **Data Handling → Workspace** to load these datasets")
                        
                        with col_exp4:
                            # Mantieni i bottoni esistenti ma in un expander per risparmiare spazio
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
            
            elif score_plot_type == "🎲 3D Scores Plot":
                st.markdown(f"### 🎲 3D Scores Plot{title_suffix}")
                
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
                        # Get custom variables
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
                    
                    with col_vis2:
                        if color_by_3d != "None" and not (color_by_3d in custom_vars and ('Time' in color_by_3d or 'time' in color_by_3d)):
                            use_dark_theme_3d = st.checkbox("Dark mode colors", value=True, key="dark_mode_3d")
                        else:
                            use_dark_theme_3d = True
                    
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
                        if color_by_3d in custom_vars and ('Time' in color_by_3d or 'time' in color_by_3d):
                            fig_3d = px.scatter_3d(
                                x=scores[pc_x], y=scores[pc_y], z=scores[pc_z], 
                                color=color_data_3d, text=text_param_3d,
                                title=f"3D Scores: {pc_x}, {pc_y}, {pc_z} (colored by {color_by_3d}){title_suffix}<br>Total Explained Variance: {var_total_3d:.1f}%",
                                labels={'x': f'{pc_x} ({var_x:.1f}%)', 'y': f'{pc_y} ({var_y:.1f}%)', 'z': f'{pc_z} ({var_z:.1f}%)', 'color': color_by_3d},
                                color_continuous_scale='RdBu_r'
                            )
                        else:
                            # USA SISTEMA DI COLORI UNIFICATO PER 3D
                            color_data_series_3d = pd.Series(color_data_3d)
                            unique_values_3d = color_data_series_3d.dropna().unique()
                            
                            color_discrete_map_3d = create_categorical_color_map(unique_values_3d, use_dark_theme_3d)
                                
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
                    
                    st.plotly_chart(fig_3d, use_container_width=True)
                    
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
                st.plotly_chart(fig, use_container_width=True)
                
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
        st.markdown("*Equivalent to PCA_diagnostic_* R scripts*")
        
        if 'pca_model' not in st.session_state:
            st.warning("⚠️ No PCA model computed. Please compute a model first.")
        else:
            pca_results = st.session_state.pca_model
            is_varimax = pca_results.get('method') == 'Varimax Rotation'
            
            # Check for missing data
            if pca_results['original_data'].isnull().any().any():
                st.warning("⚠️ T² and Q diagnostics require complete data (no missing values)")
                st.info("Missing values detected. Advanced diagnostics may not be accurate.")
            
            diagnostic_type = st.selectbox(
                "Select diagnostic type:",
                ["📊 Model Quality Metrics", "🎯 Factor Interpretation"] if is_varimax 
                else ["📊 Model Quality Metrics", "⚠️ T² vs Q Plot (Advanced)"]
            )
            
            if diagnostic_type == "📊 Model Quality Metrics":
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
                        # Calculate simple structure measure
                        loadings_squared = loadings.values ** 2
                        simple_structure = np.mean(np.var(loadings_squared, axis=1))
                        st.metric("Simple Structure Index", f"{simple_structure:.3f}")
            
            elif diagnostic_type == "🎯 Factor Interpretation" and is_varimax:
                st.markdown("### 🎯 Varimax Factor Interpretation")
                st.markdown("*Analysis of rotated factor structure*")
                
                loadings = pca_results['loadings']
                
                # Factor interpretation table
                st.markdown("#### Factor Loading Matrix")
                
                # Highlight high loadings
                def highlight_high_loadings(val):
                    if abs(val) > 0.5:
                        return 'background-color: yellow'
                    elif abs(val) > 0.3:
                        return 'background-color: lightblue'
                    return ''
                
                styled_loadings = loadings.style.applymap(highlight_high_loadings)
                st.dataframe(styled_loadings, use_container_width=True)
                
                st.info("🟡 High loadings (>0.5) | 🔵 Moderate loadings (0.3-0.5)")
                
                # Simple structure analysis
                st.markdown("#### Simple Structure Analysis")
                
                loadings_abs = np.abs(loadings.values)
                
                for i, factor in enumerate(loadings.columns):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # High loading variables
                        high_loading_vars = loadings.index[loadings_abs[:, i] > 0.5].tolist()
                        st.markdown(f"**{factor} - High Loading Variables:**")
                        if high_loading_vars:
                            for var in high_loading_vars:
                                loading_val = loadings.loc[var, factor]
                                st.write(f"• {var}: {loading_val:.3f}")
                        else:
                            st.write("No variables with loadings > 0.5")
                    
                    with col2:
                        # Factor interpretation
                        st.markdown(f"**{factor} - Variance Explained:**")
                        st.write(f"{pca_results['explained_variance_ratio'][i]*100:.2f}%")
                        
                        # Communality for this factor
                        communality = np.sum(loadings.iloc[:, i]**2)
                        st.write(f"Factor communality: {communality:.3f}")
            
            else:
                st.info("🚧 Advanced T²/Q diagnostics will be implemented in future versions")
                st.markdown("### Basic Model Information")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Samples", len(pca_results['scores']))
                
                with col2:
                    st.metric("Total Variables", len(pca_results['loadings']))
                
                with col3:
                    st.metric("Model Type", pca_results.get('method', 'Standard PCA'))

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
        st.markdown("*T² and Q diagnostics equivalent to R scripts*")
        
        if not DIAGNOSTICS_AVAILABLE:
            st.warning("⚠️ Advanced diagnostics module not available")
            st.info("Please ensure pca_diagnostics_complete.py is in the same directory")
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

                