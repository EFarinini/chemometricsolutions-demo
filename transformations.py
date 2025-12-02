"""
CAT Transformations Page - FIXED VERSION
Complete suite for spectral/analytical data transformations
Equivalent to TR_* R scripts
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.signal import savgol_filter
# Import sistema di colori unificato
from color_utils import (get_unified_color_schemes, create_categorical_color_map,
                        create_quantitative_color_map, is_quantitative_variable,
                        get_continuous_color_for_value)
# Import column DoE coding from transforms module
from transforms.column_transforms import column_doe_coding, detect_column_type
# Import preprocessing theory module (optional)
try:
    from preprocessing_theory_module import SimulatedSpectralDataGenerator, PreprocessingEffectsAnalyzer, get_all_simulated_datasets
    PREPROCESSING_THEORY_AVAILABLE = True
except ImportError:
    PREPROCESSING_THEORY_AVAILABLE = False

# ===========================================
# ROW TRANSFORMATIONS (Spectral/Analytical)
# ===========================================

def snv_transform(data, col_range):
    """Standard Normal Variate (row autoscaling)"""
    M = data.iloc[:, col_range[0]:col_range[1]].copy()
    M_t = M.T
    M_scaled = (M_t - M_t.mean(axis=0)) / M_t.std(axis=0, ddof=1)
    return M_scaled.T

def first_derivative_row(data, col_range):
    """First derivative by row"""
    M = data.iloc[:, col_range[0]:col_range[1]].copy()
    M_diff = M.diff(axis=1).iloc[:, 1:]
    return M_diff

def second_derivative_row(data, col_range):
    """Second derivative by row"""
    M = data.iloc[:, col_range[0]:col_range[1]].copy()
    M_diff = M.diff(axis=1).diff(axis=1).iloc[:, 2:]
    return M_diff

def savitzky_golay_transform(data, col_range, window_length, polyorder, deriv):
    """Savitzky-Golay filter"""
    M = data.iloc[:, col_range[0]:col_range[1]].copy()
    M_sg = M.apply(lambda row: savgol_filter(row, window_length, polyorder, deriv=deriv), axis=1)
    return pd.DataFrame(M_sg.tolist(), index=M.index)

def moving_average_row(data, col_range, window):
    """Moving average by row"""
    M = data.iloc[:, col_range[0]:col_range[1]].copy()
    M_ma = M.rolling(window=window, axis=1, center=True).mean()
    return M_ma.dropna(axis=1)

def row_sum100(data, col_range):
    """Normalize row sum to 100"""
    M = data.iloc[:, col_range[0]:col_range[1]].copy()
    row_sums = M.sum(axis=1)
    M_norm = M.div(row_sums, axis=0) * 100
    return M_norm

def binning_transform(data, col_range, bin_width):
    """Binning (averaging adjacent variables)"""
    M = data.iloc[:, col_range[0]:col_range[1]].copy()
    n_cols = M.shape[1]
    
    if n_cols % bin_width != 0:
        raise ValueError(f"Number of columns ({n_cols}) must be multiple of bin width ({bin_width})")
    
    n_bins = n_cols // bin_width
    binned_data = []
    
    for i in range(n_bins):
        start_idx = i * bin_width
        end_idx = start_idx + bin_width
        bin_mean = M.iloc[:, start_idx:end_idx].mean(axis=1)
        binned_data.append(bin_mean)
    
    return pd.DataFrame(binned_data).T

# ===========================================
# COLUMN TRANSFORMATIONS
# ===========================================

def column_centering(data, col_range):
    """Column centering (mean removal)"""
    M = data.iloc[:, col_range[0]:col_range[1]].copy()
    M_centered = M - M.mean(axis=0)
    return M_centered

def column_scaling(data, col_range):
    """Column scaling (unit variance)"""
    M = data.iloc[:, col_range[0]:col_range[1]].copy()
    M_scaled = M / M.std(axis=0, ddof=1)
    return M_scaled

def column_autoscale(data, col_range):
    """Column autoscaling (centering + scaling)"""
    M = data.iloc[:, col_range[0]:col_range[1]].copy()
    M_auto = (M - M.mean(axis=0)) / M.std(axis=0, ddof=1)
    return M_auto

def column_range_01(data, col_range):
    """Scale columns to [0,1] range"""
    M = data.iloc[:, col_range[0]:col_range[1]].copy()
    M_01 = (M - M.min(axis=0)) / (M.max(axis=0) - M.min(axis=0))
    return M_01

def column_range_11(data, col_range):
    """Scale columns to [-1,1] range (DoE coding)"""
    M = data.iloc[:, col_range[0]:col_range[1]].copy()
    M_11 = 2 * (M - M.min(axis=0)) / (M.max(axis=0) - M.min(axis=0)) - 1
    return M_11

def column_max100(data, col_range):
    """Scale column maximum to 100"""
    M = data.iloc[:, col_range[0]:col_range[1]].copy()
    M_max100 = (M / M.max(axis=0)) * 100
    return M_max100

def column_sum100(data, col_range):
    """Scale column sum to 100"""
    M = data.iloc[:, col_range[0]:col_range[1]].copy()
    M_sum100 = (M / M.sum(axis=0)) * 100
    return M_sum100

def column_length1(data, col_range):
    """Scale column length to 1"""
    M = data.iloc[:, col_range[0]:col_range[1]].copy()
    col_lengths = np.sqrt((M**2).sum(axis=0))
    M_l1 = M / col_lengths
    return M_l1

def column_log(data, col_range):
    """Log10 transformation with delta handling"""
    M = data.iloc[:, col_range[0]:col_range[1]].copy()
    
    if (M <= 0).any().any():
        min_val = M.min().min()
        delta = abs(min_val) + 1
        st.warning(f"Negative/zero values found. Adding delta: {delta}")
        M = M + delta
    
    M_log = np.log10(M)
    return M_log

def column_first_derivative(data, col_range):
    """First derivative by column"""
    M = data.iloc[:, col_range[0]:col_range[1]].copy()
    M_diff = M.diff(axis=0).iloc[1:, :]
    return M_diff

def column_second_derivative(data, col_range):
    """Second derivative by column"""
    M = data.iloc[:, col_range[0]:col_range[1]].copy()
    M_diff = M.diff(axis=0).diff(axis=0).iloc[2:, :]
    return M_diff

def moving_average_column(data, col_range, window):
    """Moving average by column"""
    M = data.iloc[:, col_range[0]:col_range[1]].copy()
    M_ma = M.rolling(window=window, axis=0, center=True).mean()
    return M_ma.dropna(axis=0)

def block_scaling(data, blocks_config):
    """Block scaling (autoscale + divide by sqrt(n_vars_in_block))"""
    transformed = data.copy()
    
    for block_name, col_range in blocks_config.items():
        M = data.iloc[:, col_range[0]:col_range[1]].copy()
        n_vars = M.shape[1]
        
        M_auto = (M - M.mean(axis=0)) / M.std(axis=0, ddof=1)
        M_block = M_auto / np.sqrt(n_vars)
        
        transformed.iloc[:, col_range[0]:col_range[1]] = M_block
    
    return transformed

# ===========================================
# VISUALIZATION FUNCTIONS
# ===========================================

def plot_comparison(original_data, transformed_data, title_original, title_transformed, 
                   color_data=None, color_variable=None):
    """Create two line plots for comparison"""
    
    from plotly.subplots import make_subplots
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(title_original, title_transformed),
        horizontal_spacing=0.12
    )
    
    if color_data is not None:
        # Converti color_data in lista per accesso facile con indici numerici
        if hasattr(color_data, 'values'):
            color_values = color_data.values
        else:
            color_values = list(color_data)
        
        # Determina se la variabile è quantitativa o categorica
        is_quantitative = is_quantitative_variable(color_data)
        
        if is_quantitative:
            # Variabile quantitativa: usa scala blu-rosso
            color_data_series = pd.Series(color_values).dropna()
            min_val = color_data_series.min()
            max_val = color_data_series.max()
            
            # Plot dei dati originali
            for i, idx in enumerate(original_data.index):
                if i < len(color_values) and pd.notna(color_values[i]):
                    color = get_continuous_color_for_value(color_values[i], min_val, max_val, 'blue_to_red')
                    hover_text = f'Sample: {idx}<br>{color_variable}: {color_values[i]:.3f}<br>Value: %{{y:.3f}}<extra></extra>'
                else:
                    color = 'rgb(128, 128, 128)'
                    hover_text = f'Sample: {idx}<br>Value: %{{y:.3f}}<extra></extra>'
                
                fig.add_trace(
                    go.Scatter(
                        x=list(range(len(original_data.columns))),
                        y=original_data.iloc[i].values,
                        mode='lines',
                        line=dict(color=color, width=1.5),
                        showlegend=False,
                        hovertemplate=hover_text
                    ),
                    row=1, col=1
                )
            
            # Plot dei dati trasformati
            for i, idx in enumerate(transformed_data.index):
                if i < len(color_values) and pd.notna(color_values[i]):
                    color = get_continuous_color_for_value(color_values[i], min_val, max_val, 'blue_to_red')
                    hover_text = f'Sample: {idx}<br>{color_variable}: {color_values[i]:.3f}<br>Value: %{{y:.3f}}<extra></extra>'
                else:
                    color = 'rgb(128, 128, 128)'
                    hover_text = f'Sample: {idx}<br>Value: %{{y:.3f}}<extra></extra>'
                
                fig.add_trace(
                    go.Scatter(
                        x=list(range(len(transformed_data.columns))),
                        y=transformed_data.iloc[i].values,
                        mode='lines',
                        line=dict(color=color, width=1.5),
                        showlegend=False,
                        hovertemplate=hover_text
                    ),
                    row=1, col=2
                )
            
            # Aggiungi colorbar migliorata con valori e dettagli
            n_ticks = 6
            tick_vals = [min_val + i * (max_val - min_val) / (n_ticks - 1) for i in range(n_ticks)]
            
            fig.add_trace(
                go.Scatter(
                    x=[None], y=[None],
                    mode='markers',
                    marker=dict(
                        colorscale=[[0, 'rgb(0,0,255)'], [1, 'rgb(255,0,0)']],
                        cmin=min_val,
                        cmax=max_val,
                        colorbar=dict(
                            title=dict(
                                text=f"<b>{color_variable}</b>",
                                side="right",
                                font=dict(size=12)
                            ),
                            titleside="right",
                            x=1.02,
                            len=0.8,
                            y=0.5,
                            thickness=15,
                            tickmode="array",
                            tickvals=tick_vals,
                            ticktext=[f"{val:.2f}" for val in tick_vals],
                            tickfont=dict(size=10),
                            showticklabels=True,
                            ticks="outside",
                            ticklen=5
                        ),
                        showscale=True
                    ),
                    showlegend=False
                ),
                row=1, col=1
            )
        
        else:
            # Variabile categorica: usa colori discreti
            unique_values = pd.Series(color_values).dropna().unique()
            color_discrete_map = create_categorical_color_map(unique_values)
            
            for group in unique_values:
                group_indices = [i for i, val in enumerate(color_values) if val == group]
                first_idx = group_indices[0] if group_indices else None
                
                for i in group_indices:
                    is_first = bool(i == first_idx)
                    if i < len(original_data):
                        fig.add_trace(
                            go.Scatter(
                                x=list(range(len(original_data.columns))),
                                y=original_data.iloc[i].values,
                                mode='lines',
                                name=str(group),
                                line=dict(color=color_discrete_map[group], width=1),
                                showlegend=is_first,
                                legendgroup=str(group),
                                hovertemplate=f'Sample: {original_data.index[i]}<br>{color_variable}: {group}<br>Value: %{{y:.3f}}<extra></extra>'
                            ),
                            row=1, col=1
                        )
                    
                    if i < len(transformed_data):
                        fig.add_trace(
                            go.Scatter(
                                x=list(range(len(transformed_data.columns))),
                                y=transformed_data.iloc[i].values,
                                mode='lines',
                                name=str(group),
                                line=dict(color=color_discrete_map[group], width=1),
                                showlegend=False,
                                legendgroup=str(group),
                                hovertemplate=f'Sample: {transformed_data.index[i]}<br>{color_variable}: {group}<br>Value: %{{y:.3f}}<extra></extra>'
                            ),
                            row=1, col=2
                        )
    else:
        # Nessuna colorazione
        for i, idx in enumerate(original_data.index):
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(original_data.columns))),
                    y=original_data.iloc[i].values,
                    mode='lines',
                    line=dict(color='blue', width=1),
                    showlegend=False,
                    hovertemplate=f'Sample: {idx}<br>Value: %{{y:.3f}}<extra></extra>'
                ),
                row=1, col=1
            )
            
            if i < len(transformed_data):
                fig.add_trace(
                    go.Scatter(
                        x=list(range(len(transformed_data.columns))),
                        y=transformed_data.iloc[i].values,
                        mode='lines',
                        line=dict(color='blue', width=1),
                        showlegend=False,
                        hovertemplate=f'Sample: {idx}<br>Value: %{{y:.3f}}<extra></extra>'
                    ),
                    row=1, col=2
                )
    
    fig.update_xaxes(title_text="Variable Index", row=1, col=1, gridcolor='lightgray')
    fig.update_xaxes(title_text="Variable Index", row=1, col=2, gridcolor='lightgray')
    fig.update_yaxes(title_text="Value", row=1, col=1, gridcolor='lightgray')
    fig.update_yaxes(title_text="Value", row=1, col=2, gridcolor='lightgray')

    fig.update_layout(
        height=500,
        width=1400,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='black', size=11),
        hovermode='closest',
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99,
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="lightgray",
            borderwidth=1
        )
    )
    
    return fig

# ===========================================
# MAIN SHOW FUNCTION - COMPLETELY FIXED
# ===========================================

def show():
    """Display the Transformations page"""
    
    st.markdown("# Data Transformations")
    st.markdown("*Complete transformation suite for spectral and analytical data*")
    
    # Professional Services Note
    st.info("""
    💡 **Demo includes core transformations.** Professional versions include specialized transformations for different analytical techniques:
    
    🔧 **Contact:** [chemometricsolutions.com](https://chemometricsolutions.com)
    """)
    
    if 'current_data' not in st.session_state:
        st.warning("No data loaded. Please go to Data Handling to load your dataset first.")
        return
    
    data = st.session_state.current_data
    
    # Get original untransformed data for comparison
    original_dataset_name = st.session_state.get('current_dataset', 'Dataset')
    if original_dataset_name.endswith('_ORIGINAL'):
        original_data = data
    elif 'transformation_history' in st.session_state:
        original_key = f"{original_dataset_name.split('.')[0]}_ORIGINAL"
        if original_key in st.session_state.transformation_history:
            original_data = st.session_state.transformation_history[original_key]['data']
        else:
            original_data = data
    else:
        original_data = data
    
    tab1, tab2, tab3 = st.tabs([
        "Row Transformations",
        "Column Transformations",
        "🎓 Preprocessing Theory"
    ])
    
    # ===== ROW TRANSFORMATIONS TAB =====
    with tab1:
        st.markdown("## Row Transformations")
        st.markdown("*For spectral/analytical profiles - transformations applied across variables*")
        
        row_transforms = {
            "SNV (Standard Normal Variate)": "snv",
            "First Derivative": "der1r",
            "Second Derivative": "der2r",
            "Savitzky-Golay": "sg",
            "Moving Average": "mar",
            "Row Sum = 100": "sum100r",
            "Binning": "bin"
        }
        
        selected_transform = st.selectbox(
            "Select row transformation:",
            list(row_transforms.keys())
        )
        
        transform_code = row_transforms[selected_transform]
        
        st.markdown("### Variable Selection")
        
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_columns:
            st.error("No numeric columns found!")
            return
        
        all_columns = data.columns.tolist()
        first_numeric_pos = all_columns.index(numeric_columns[0]) + 1
        last_numeric_pos = all_columns.index(numeric_columns[-1]) + 1
        
        st.info(f"Dataset: {len(all_columns)} total columns, {len(numeric_columns)} numeric (positions {first_numeric_pos}-{last_numeric_pos})")
        
        col1, col2 = st.columns(2)
        
        with col1:
            first_col = st.number_input(
                "First column (1-based):", 
                min_value=1, 
                max_value=len(all_columns),
                value=first_numeric_pos
            )
        
        with col2:
            last_col = st.number_input(
                "Last column (1-based):", 
                min_value=first_col,
                max_value=len(all_columns),
                value=last_numeric_pos
            )
        
        n_selected = last_col - first_col + 1
        st.info(f"Will transform {n_selected} columns (from column {first_col} to {last_col})")

        col_range = (first_col-1, last_col)

        # === DATA PREVIEW (Original) ===
        st.markdown("### 👁️ Data Preview (Original)")

        col_prev1, col_prev2 = st.columns(2)

        with col_prev1:
            preview_type_orig = st.radio(
                "Preview type:",
                ["First 10 rows", "Random samples", "Statistics"],
                horizontal=True,
                key="row_preview_before"
            )

        with col_prev2:
            n_preview_orig = st.slider("Rows to show:", 5, 20, 10, key="row_preview_rows_before")

        # Show preview of ORIGINAL data
        original_preview = data.iloc[:, first_col-1:last_col]

        # Create format dict: only format numeric columns
        format_dict = {col: "{:.3f}" for col in original_preview.select_dtypes(include=[np.number]).columns}

        if preview_type_orig == "First 10 rows":
            st.dataframe(
                original_preview.head(n_preview_orig).style.format(format_dict, na_rep="-"),
                use_container_width=True,
                height=300
            )
        elif preview_type_orig == "Random samples":
            st.dataframe(
                original_preview.sample(n=min(n_preview_orig, len(original_preview)), random_state=42).style.format(format_dict, na_rep="-"),
                use_container_width=True,
                height=300
            )
        else:  # Statistics
            st.dataframe(
                original_preview.describe().style.format("{:.3f}"),
                use_container_width=True
            )

        st.info(f"📊 Original data: {original_preview.shape[0]} samples × {original_preview.shape[1]} variables")

        st.markdown("### Transformation Parameters")
        
        params = {}
        
        if transform_code == "sg":
            col_p1, col_p2, col_p3 = st.columns(3)
            with col_p1:
                params['window_length'] = st.number_input("Window length (odd):", 3, 51, 11, step=2)
            with col_p2:
                params['polyorder'] = st.number_input("Polynomial order:", 1, 5, 2)
            with col_p3:
                params['deriv'] = st.number_input("Derivative:", 0, 2, 0)
        
        elif transform_code == "mar":
            params['window'] = st.number_input("Window size (odd):", 3, 51, 5, step=2)
        
        elif transform_code == "bin":
            n_vars = last_col - first_col + 1
            params['bin_width'] = st.number_input("Bin width:", 2, n_vars, 5)
            
            if n_vars % params['bin_width'] != 0:
                st.warning(f"Number of variables ({n_vars}) must be multiple of bin width")
        
        st.markdown("### Visualization Options")

        
        col_vis1, col_vis2 = st.columns(2)
        
        with col_vis1:
            custom_vars = []
            if 'custom_variables' in st.session_state:
                custom_vars = list(st.session_state.custom_variables.keys())
            
            spectral_vars = numeric_columns[first_col-1:last_col]
            available_color_vars = [col for col in data.columns if col not in spectral_vars]
            
            all_color_options = (["None", "Row Index"] + available_color_vars + custom_vars)
            
            color_by = st.selectbox("Color profiles by:", all_color_options, key="row_transform_color")
        
        color_data = None
        color_variable = None
        
        if color_by != "None":
            color_variable = color_by
            if color_by == "Row Index":
                color_data = [f"Sample_{i+1}" for i in range(len(data))]
            elif color_by in custom_vars:
                color_data = st.session_state.custom_variables[color_by].reindex(data.index).fillna("Unknown")
            else:
                color_data = data[color_by].reindex(data.index).fillna("Unknown")
        
        # Store transformation results in session state to persist across save button clicks
        if st.button("Apply Transformation", type="primary", key="apply_row_transform"):
            try:
                with st.spinner(f"Applying {selected_transform}..."):
                    if transform_code == "snv":
                        transformed = snv_transform(data, col_range)
                    elif transform_code == "der1r":
                        transformed = first_derivative_row(data, col_range)
                    elif transform_code == "der2r":
                        transformed = second_derivative_row(data, col_range)
                    elif transform_code == "sg":
                        transformed = savitzky_golay_transform(data, col_range, 
                                                              params['window_length'], 
                                                              params['polyorder'], 
                                                              params['deriv'])
                    elif transform_code == "mar":
                        transformed = moving_average_row(data, col_range, params['window'])
                    elif transform_code == "sum100r":
                        transformed = row_sum100(data, col_range)
                    elif transform_code == "bin":
                        transformed = binning_transform(data, col_range, params['bin_width'])
                    
                    # Store in session state
                    st.session_state.current_transform_result = {
                        'transformed': transformed,
                        'original_slice': original_data.iloc[:, col_range[0]:col_range[1]],
                        'transform_code': transform_code,
                        'selected_transform': selected_transform,
                        'params': params,
                        'col_range': col_range,
                        'color_data': color_data,
                        'color_variable': color_variable
                    }
                    
                    st.success("Transformation applied successfully!")
                    
            except Exception as e:
                st.error(f"Error applying transformation: {str(e)}")
                import traceback
                if st.checkbox("Show debug info", key="row_debug"):
                    st.code(traceback.format_exc())
        
        # Display results if transformation has been applied
        if 'current_transform_result' in st.session_state:
            result = st.session_state.current_transform_result
            
            fig = plot_comparison(
                result['original_slice'], 
                result['transformed'],
                f"Original Data ({result['original_slice'].shape[0]} × {result['original_slice'].shape[1]})",
                f"Transformed Data ({result['transformed'].shape[0]} × {result['transformed'].shape[1]}) - {result['selected_transform']}",
                color_data=result['color_data'],
                color_variable=result['color_variable']
            )
            
            st.plotly_chart(fig, width='stretch')

            # Statistics
            st.markdown("### Transformation Statistics")
            
            col_stat1, col_stat2, col_stat3 = st.columns(3)
            
            with col_stat1:
                st.metric("Original Shape", f"{result['original_slice'].shape[0]} × {result['original_slice'].shape[1]}")
            with col_stat2:
                st.metric("Transformed Shape", f"{result['transformed'].shape[0]} × {result['transformed'].shape[1]}")
            with col_stat3:
                variance_ratio = result['transformed'].var().mean() / result['original_slice'].var().mean()
                st.metric("Variance Ratio", f"{variance_ratio:.3f}")

            # Data Preview (Transformed)
            st.markdown("### 👁️ Data Preview (Transformed)")

            col_prev_t1, col_prev_t2 = st.columns(2)

            with col_prev_t1:
                preview_type_trans = st.radio(
                    "Preview type:",
                    ["First 10 rows", "Random samples", "Statistics"],
                    horizontal=True,
                    key="row_preview_after"
                )

            with col_prev_t2:
                n_preview_trans = st.slider("Rows to show:", 5, 20, 10, key="row_preview_rows_after")

            # Show preview of TRANSFORMED data
            # Create format dict: only format numeric columns
            format_dict_trans = {col: "{:.3f}" for col in result['transformed'].select_dtypes(include=[np.number]).columns}

            if preview_type_trans == "First 10 rows":
                st.dataframe(
                    result['transformed'].head(n_preview_trans).style.format(format_dict_trans, na_rep="-"),
                    use_container_width=True,
                    height=300
                )
            elif preview_type_trans == "Random samples":
                st.dataframe(
                    result['transformed'].sample(n=min(n_preview_trans, len(result['transformed'])), random_state=42).style.format(format_dict_trans, na_rep="-"),
                    use_container_width=True,
                    height=300
                )
            else:  # Statistics
                st.dataframe(
                    result['transformed'].describe().style.format("{:.3f}"),
                    use_container_width=True
                )

            st.info(f"📊 Transformed data: {result['transformed'].shape[0]} samples × {result['transformed'].shape[1]} variables")

            # Save section
            st.markdown("---")
            st.markdown("### Save Transformation")
            st.info("Review the transformation above, then save it to workspace if satisfied")
            
            # FIXED SAVE LOGIC FOR ROW TRANSFORMATIONS
            col_save, col_download = st.columns(2)

            with col_save:
                if st.button("💾 Save to Workspace", type="primary", key="save_row_transform", use_container_width=True):
                    try:
                        # Ensure transformation_history exists
                        if 'transformation_history' not in st.session_state:
                            st.session_state.transformation_history = {}

                        # Get current transformation result
                        result = st.session_state.current_transform_result

                        # CORREZIONE: Preserva SEMPRE la struttura originale del dataset
                        full_transformed = data.copy()  # Copia completa del dataset originale
                        transformed = result['transformed']
                        col_range = result['col_range']

                        # Handle shape changes properly - MA PRESERVA METADATA
                        if transformed.shape[1] != (col_range[1] - col_range[0]):
                            # Variables were removed (derivatives, etc.)

                            # Handle row changes FIRST se necessario
                            if transformed.shape[0] != data.shape[0]:
                                # Row reduction (derivatives) - taglia tutto il dataset
                                full_transformed = full_transformed.iloc[:transformed.shape[0], :].copy()

                            # Calcola quante colonne sono state rimosse
                            original_cols = col_range[1] - col_range[0]
                            transformed_cols = transformed.shape[1]
                            cols_removed = original_cols - transformed_cols

                            if cols_removed > 0:
                                # Alcune colonne sono state rimosse (es. derivate)
                                # SOLUZIONE SEMPLIFICATA: Sostituisci le colonne trasformate e shifta le successive

                                # Colonne prima della trasformazione: mantieni invariate
                                before_data = full_transformed.iloc[:, :col_range[0]] if col_range[0] > 0 else pd.DataFrame(index=full_transformed.index)

                                # Colonne dopo la trasformazione: shifta indietro
                                after_start = col_range[1]
                                if after_start < len(data.columns):
                                    after_data = full_transformed.iloc[:, after_start:]
                                else:
                                    after_data = pd.DataFrame(index=full_transformed.index)

                                # Concatena: before + transformed + after
                                data_parts = []
                                column_names = []

                                if col_range[0] > 0:
                                    data_parts.append(before_data)
                                    column_names.extend(before_data.columns.tolist())

                                data_parts.append(transformed)
                                # Mantieni nomi originali per le colonne trasformate (se possibile)
                                original_transform_cols = data.columns[col_range[0]:col_range[0]+transformed.shape[1]]
                                column_names.extend(original_transform_cols.tolist())

                                if after_start < len(data.columns):
                                    data_parts.append(after_data)
                                    column_names.extend(after_data.columns.tolist())

                                # Combina tutto
                                full_transformed = pd.concat(data_parts, axis=1)
                                full_transformed.columns = column_names
                            else:
                                # Nessuna colonna rimossa - semplice sostituzione
                                full_transformed.iloc[:, col_range[0]:col_range[1]] = transformed

                        else:
                            # NO shape changes - semplice sostituzione
                            if transformed.shape[0] != data.shape[0]:
                                # Handle row reduction (derivatives)
                                full_transformed = full_transformed.iloc[:transformed.shape[0], :].copy()

                            # SOSTITUISCI SOLO LE COLONNE TRASFORMATE
                            full_transformed.iloc[:, col_range[0]:col_range[1]] = transformed

                        # Create transformation name
                        dataset_name = st.session_state.get('current_dataset', 'Dataset')
                        base_name = dataset_name.split('.')[0].replace('_ORIGINAL', '')
                        transformed_name = f"{base_name}.{result['transform_code']}"

                        # Save to workspace with all required metadata
                        st.session_state.transformation_history[transformed_name] = {
                            'data': full_transformed,
                            'transform': result['selected_transform'],
                            'params': result['params'],
                            'col_range': col_range,
                            'timestamp': pd.Timestamp.now(),
                            'original_dataset': dataset_name,
                            'transform_type': 'row_transformation'
                        }

                        # Update current data
                        st.session_state.current_data = full_transformed
                        st.session_state.current_dataset = transformed_name

                        # Clear the transformation result
                        del st.session_state.current_transform_result

                        # Show success messages
                        st.success(f"✅ Transformation saved as: **{transformed_name}**")
                        st.info("📊 Dataset is now active in Data Handling and ready for PCA/DOE")
                        st.info(f"🔒 **Structure preserved**: All metadata columns maintained")

                        # Force refresh of the interface
                        st.rerun()

                    except Exception as e:
                        st.error(f"❌ Error saving transformation: {str(e)}")
                        st.error("Please try applying the transformation again")

                        # Debug traceback
                        import traceback
                        st.code(traceback.format_exc())

            with col_download:
                # Download XLSX button
                if 'current_transform_result' in st.session_state:
                    result = st.session_state.current_transform_result

                    # Build full transformed dataset (same logic as save)
                    full_transformed = data.copy()
                    transformed = result['transformed']
                    col_range = result['col_range']

                    # Handle shape changes
                    if transformed.shape[1] != (col_range[1] - col_range[0]):
                        if transformed.shape[0] != data.shape[0]:
                            full_transformed = full_transformed.iloc[:transformed.shape[0], :].copy()

                        original_cols = col_range[1] - col_range[0]
                        transformed_cols = transformed.shape[1]
                        cols_removed = original_cols - transformed_cols

                        if cols_removed > 0:
                            before_data = full_transformed.iloc[:, :col_range[0]] if col_range[0] > 0 else pd.DataFrame(index=full_transformed.index)
                            after_start = col_range[1]
                            if after_start < len(data.columns):
                                after_data = full_transformed.iloc[:, after_start:]
                            else:
                                after_data = pd.DataFrame(index=full_transformed.index)

                            data_parts = []
                            column_names = []

                            if col_range[0] > 0:
                                data_parts.append(before_data)
                                column_names.extend(before_data.columns.tolist())

                            data_parts.append(transformed)
                            original_transform_cols = data.columns[col_range[0]:col_range[0]+transformed.shape[1]]
                            column_names.extend(original_transform_cols.tolist())

                            if after_start < len(data.columns):
                                data_parts.append(after_data)
                                column_names.extend(after_data.columns.tolist())

                            full_transformed = pd.concat(data_parts, axis=1)
                            full_transformed.columns = column_names
                        else:
                            full_transformed.iloc[:, col_range[0]:col_range[1]] = transformed
                    else:
                        if transformed.shape[0] != data.shape[0]:
                            full_transformed = full_transformed.iloc[:transformed.shape[0], :].copy()
                        full_transformed.iloc[:, col_range[0]:col_range[1]] = transformed

                    # Create XLSX file
                    import io
                    buffer = io.BytesIO()
                    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                        full_transformed.to_excel(writer, index=True, sheet_name='Transformed Data')
                    buffer.seek(0)

                    # Generate filename
                    dataset_name = st.session_state.get('current_dataset', 'Dataset')
                    base_name = dataset_name.split('.')[0].replace('_ORIGINAL', '')
                    filename = f"{base_name}.{result['transform_code']}.xlsx"

                    st.download_button(
                        label="📥 Download XLSX",
                        data=buffer,
                        file_name=filename,
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True,
                        key="download_row_transform_xlsx"
                    )
    
    # ===== COLUMN TRANSFORMATIONS TAB =====
    with tab2:
        st.markdown("## Column Transformations")
        st.markdown("*Transformations applied within each variable*")
        
        col_transforms = {
            "🚀 Automatic DoE Coding": "auto_doe",
            "Centering": "centc",
            "Scaling (Unit Variance)": "scalc",
            "Autoscaling": "autosc",
            "Range [0,1]": "01c",
            "Range [-1,1]": "cod",
            "Maximum = 100": "max100c",
            "Sum = 100": "sum100c",
            "Length = 1": "l1c",
            "Log10": "log",
            "First Derivative": "der1c",
            "Second Derivative": "der2c",
            "Moving Average": "mac",
            "Block Scaling": "blsc"
        }
        
        selected_transform_col = st.selectbox(
            "Select column transformation:",
            list(col_transforms.keys()),
            key="col_transform_select"
        )
        
        transform_code_col = col_transforms[selected_transform_col]
        
        st.markdown("### Variable Selection")
        
        # Reuse variables from row transformations
        st.info(f"Dataset: {len(all_columns)} total columns, {len(numeric_columns)} numeric (positions {first_numeric_pos}-{last_numeric_pos})")
        
        col1_sel, col2_sel = st.columns(2)
        
        with col1_sel:
            first_col_c = st.number_input(
                "First column (1-based):", 
                min_value=1, 
                max_value=len(all_columns),
                value=first_numeric_pos,
                key="first_col_c"
            )
        
        with col2_sel:
            last_col_c = st.number_input(
                "Last column (1-based):", 
                min_value=first_col_c,
                max_value=len(all_columns),
                value=min(first_col_c + 9, last_numeric_pos),
                key="last_col_c"
            )
        
        n_selected_c = last_col_c - first_col_c + 1
        st.info(f"Will transform {n_selected_c} columns (from column {first_col_c} to {last_col_c})")
        
        col_range_c = (first_col_c-1, last_col_c)
        
        params_col = {}
        
        if transform_code_col == "mac":
            params_col['window'] = st.number_input("Window size (odd):", 3, 51, 5, step=2, key="mac_window")
        
        elif transform_code_col == "blsc":
            st.markdown("### Block Configuration")
            n_blocks = st.number_input("Number of blocks:", 1, 10, 2, key="n_blocks")
            
            blocks_config = {}
            for i in range(n_blocks):
                col_b1, col_b2 = st.columns(2)
                with col_b1:
                    block_first = st.number_input(f"Block {i+1} first col:", 1, len(all_columns), 1, key=f"block_{i}_first")
                with col_b2:
                    block_last = st.number_input(f"Block {i+1} last col:", block_first, len(all_columns), 
                                                block_first, key=f"block_{i}_last")
                
                blocks_config[f"Block_{i+1}"] = (block_first-1, block_last)
            
            params_col['blocks'] = blocks_config


        # Data Preview (Original)
        st.markdown("### 👁️ Data Preview (Original)")

        col_prev_orig1, col_prev_orig2 = st.columns(2)

        with col_prev_orig1:
            preview_type_col_orig = st.radio(
                "Preview type:",
                ["First 10 rows", "Random samples", "Statistics"],
                horizontal=True,
                key="col_preview_before_type"
            )

        with col_prev_orig2:
            n_preview_col_orig = st.slider("Rows to show:", 5, 20, 10, key="col_preview_before_rows")

        # Show preview of ORIGINAL data (before transformation)
        original_preview_col = data.iloc[:, col_range_c[0]:col_range_c[1]]

        # Create format dict: only format numeric columns
        format_dict_col_orig = {col: "{:.3f}" for col in original_preview_col.select_dtypes(include=[np.number]).columns}

        if preview_type_col_orig == "First 10 rows":
            st.dataframe(
                original_preview_col.head(n_preview_col_orig).style.format(format_dict_col_orig, na_rep="-"),
                use_container_width=True,
                height=300
            )
        elif preview_type_col_orig == "Random samples":
            st.dataframe(
                original_preview_col.sample(n=min(n_preview_col_orig, len(original_preview_col)), random_state=42).style.format(format_dict_col_orig, na_rep="-"),
                use_container_width=True,
                height=300
            )
        else:  # Statistics
            st.dataframe(
                original_preview_col.describe().style.format("{:.3f}"),
                use_container_width=True
            )

        st.info(f"📊 Original data: {original_preview_col.shape[0]} samples × {original_preview_col.shape[1]} variables")

        # ========================================================================
        # AUTOMATIC DoE CODING - REFERENCE LEVEL SELECTOR
        # ========================================================================

        reference_levels = {}

        if transform_code_col == "auto_doe":
            st.markdown("---")
            st.markdown("### 🎯 Automatic DoE Coding Configuration")
            st.markdown("*Intelligently encodes: 2-level→[-1,+1], numeric→range, categorical→dummy*")

            # Preview what will happen
            st.info("""
**How it works:**
- **2-level (numeric or categorical)** → Maps to [-1, +1]
- **3+ levels numeric only** → Scales to [-1, ..., +1] range
- **3+ levels categorical** → Dummy coding (k-1) with reference level
            """)

            # Check for multiclass categorical in selected range
            multiclass_cols = {}
            for col in data.columns[col_range_c[0]:col_range_c[1]]:
                detection = detect_column_type(data[col])
                if detection['dtype_detected'] == 'multiclass_cat':
                    multiclass_cols[col] = detection

            if multiclass_cols:
                st.markdown("---")
                st.markdown("#### 📋 Categorical Variables with 3+ Levels")
                st.markdown("*Select the IMPLICIT (reference) level for each variable*")

                # Initialize session state
                if 'doe_reference_levels' not in st.session_state:
                    st.session_state.doe_reference_levels = {}

                for col_name, col_detection in multiclass_cols.items():
                    st.markdown(f"**Variable: `{col_name}`**")

                    unique_vals = col_detection['unique_values']
                    value_counts = col_detection['value_counts']

                    # Find auto-suggested reference (highest frequency)
                    auto_suggested = max(value_counts, key=value_counts.get)
                    auto_freq = value_counts[auto_suggested]

                    # Create frequency display table
                    freq_df = pd.DataFrame({
                        'Level': list(value_counts.keys()),
                        'Frequency': list(value_counts.values()),
                        'Percentage': [f"{v/sum(value_counts.values())*100:.1f}%" for v in value_counts.values()]
                    }).sort_values('Frequency', ascending=False)

                    col_freq, col_select = st.columns([1, 1])

                    with col_freq:
                        st.dataframe(freq_df, use_container_width=True, hide_index=True, height=150)

                    with col_select:
                        # Selectbox with auto-suggested as default
                        default_idx = unique_vals.index(auto_suggested) if auto_suggested in unique_vals else 0

                        selected_ref = st.selectbox(
                            label="Reference level:",
                            options=unique_vals,
                            index=default_idx,
                            key=f"doe_ref_{col_name}",
                            help=f"✨ Suggested: '{auto_suggested}' (freq={auto_freq}). This level will be coded as all zeros."
                        )

                        st.session_state.doe_reference_levels[col_name] = selected_ref

                        st.success(f"✓ Reference: **{selected_ref}** → [0, 0, ...]")

                        # Show dummy columns that will be created
                        dummy_levels = [v for v in sorted(unique_vals) if v != selected_ref]
                        st.caption(f"Will create {len(dummy_levels)} dummy columns:")
                        for level in dummy_levels:
                            st.caption(f"  • `{col_name}_{level}`")

                    # Show preview
                    with st.expander(f"📊 Preview encoding for {col_name}", expanded=False):
                        st.write(f"**Reference (implicit):** `{selected_ref}` → [0, 0, ..., 0]")
                        dummy_levels = [v for v in sorted(unique_vals) if v != selected_ref]
                        for i, level in enumerate(dummy_levels):
                            encoding = [0] * len(dummy_levels)
                            encoding[i] = 1
                            st.write(f"**`{level}`** → {encoding}")

                    st.divider()

                # Store reference levels for use in transformation
                reference_levels = st.session_state.doe_reference_levels

                st.success(f"✅ Configuration complete! {len(multiclass_cols)} categorical variable(s) configured.")
            else:
                st.success("✅ No multiclass categorical variables detected. All variables will be automatically encoded!")

        if st.button("Apply Transformation", type="primary", key="apply_col_transform"):
            try:
                with st.spinner(f"Applying {selected_transform_col}..."):
                    # Special handling for Automatic DoE Coding
                    if transform_code_col == "auto_doe":
                        # Use reference levels from UI (already selected above)
                        transformed_col, encoding_metadata, multiclass_info = column_doe_coding(
                            data, col_range_c, reference_levels=reference_levels
                        )

                        # Store metadata for later use
                        st.session_state.doe_encoding_metadata = encoding_metadata
                        st.session_state.doe_multiclass_info = multiclass_info

                    elif transform_code_col == "centc":
                        transformed_col = column_centering(data, col_range_c)
                    elif transform_code_col == "scalc":
                        transformed_col = column_scaling(data, col_range_c)
                    elif transform_code_col == "autosc":
                        transformed_col = column_autoscale(data, col_range_c)
                    elif transform_code_col == "01c":
                        transformed_col = column_range_01(data, col_range_c)
                    elif transform_code_col == "cod":
                        transformed_col = column_range_11(data, col_range_c)
                    elif transform_code_col == "max100c":
                        transformed_col = column_max100(data, col_range_c)
                    elif transform_code_col == "sum100c":
                        transformed_col = column_sum100(data, col_range_c)
                    elif transform_code_col == "l1c":
                        transformed_col = column_length1(data, col_range_c)
                    elif transform_code_col == "log":
                        transformed_col = column_log(data, col_range_c)
                    elif transform_code_col == "der1c":
                        transformed_col = column_first_derivative(data, col_range_c)
                    elif transform_code_col == "der2c":
                        transformed_col = column_second_derivative(data, col_range_c)
                    elif transform_code_col == "mac":
                        transformed_col = moving_average_column(data, col_range_c, params_col['window'])
                    elif transform_code_col == "blsc":
                        transformed_col = block_scaling(data, params_col['blocks'])
                    
                    # Store in session state
                    if transform_code_col == "blsc":
                        original_slice_col = original_data
                    elif transform_code_col == "auto_doe":
                        # For DoE coding, original slice is just the selected columns
                        original_slice_col = original_data.iloc[:, col_range_c[0]:col_range_c[1]]
                    else:
                        original_slice_col = original_data.iloc[:, col_range_c[0]:col_range_c[1]]

                    st.session_state.current_col_transform_result = {
                        'transformed_col': transformed_col,
                        'original_slice_col': original_slice_col,
                        'transform_code_col': transform_code_col,
                        'selected_transform_col': selected_transform_col,
                        'params_col': params_col,
                        'col_range_c': col_range_c
                    }
                    
                    st.success("Column transformation applied successfully!")
                    
            except Exception as e:
                st.error(f"Error applying transformation: {str(e)}")
                import traceback
                if st.checkbox("Show debug info", key="col_debug"):
                    st.code(traceback.format_exc())
        
        # Display column transformation results
        if 'current_col_transform_result' in st.session_state:
            result_col = st.session_state.current_col_transform_result

            # Skip plots for Automatic DoE Coding (categorical data doesn't need line plots)
            if result_col['transform_code_col'] != 'auto_doe':
                fig_col = plot_comparison(
                    result_col['original_slice_col'],
                    result_col['transformed_col'],
                    f"Original Data ({result_col['original_slice_col'].shape[0]} × {result_col['original_slice_col'].shape[1]})",
                    f"Transformed Data ({result_col['transformed_col'].shape[0]} × {result_col['transformed_col'].shape[1]}) - {result_col['selected_transform_col']}"
                )

                st.plotly_chart(fig_col, width='stretch')

            # Statistics
            st.markdown("### Transformation Statistics")
            
            col_stat1, col_stat2, col_stat3 = st.columns(3)
            
            with col_stat1:
                st.metric("Original Shape", f"{result_col['original_slice_col'].shape[0]} × {result_col['original_slice_col'].shape[1]}")
            with col_stat2:
                st.metric("Transformed Shape", f"{result_col['transformed_col'].shape[0]} × {result_col['transformed_col'].shape[1]}")
            with col_stat3:
                mean_val = result_col['transformed_col'].mean().mean()
                st.metric("Mean Value", f"{mean_val:.3f}")

            # Special section for DoE Encoding metadata
            if result_col.get('transform_code_col') == 'auto_doe' and 'doe_encoding_metadata' in st.session_state:
                st.markdown("---")
                st.markdown("### 🔢 DoE Encoding Report")

                metadata = st.session_state.doe_encoding_metadata

                # Create summary table
                summary_data = []
                for col_name, meta in metadata.items():
                    row = {
                        'Column': col_name,
                        'Type': meta['type'],
                        'N Levels': meta['n_levels'],
                        'Encoding': meta['encoding_rule']
                    }

                    # Add dummy columns info if multiclass
                    if meta['type'] == 'categorical_multiclass':
                        row['Dummy Columns'] = ', '.join(meta['dummy_columns'])
                        row['Reference'] = meta['reference_level']
                    else:
                        row['Dummy Columns'] = '-'
                        row['Reference'] = '-'

                    summary_data.append(row)

                summary_df = pd.DataFrame(summary_data)
                st.dataframe(summary_df, use_container_width=True, hide_index=True)

                # Expandable detailed encoding maps
                with st.expander("📋 Detailed Encoding Maps"):
                    for col_name, meta in metadata.items():
                        st.markdown(f"**{col_name}** ({meta['type']})")

                        if meta['type'] in ['numeric_2level', 'categorical_2level']:
                            # Show simple encoding map
                            encoding_df = pd.DataFrame({
                                'Original Value': list(meta['encoding_map'].keys()),
                                'Encoded Value': list(meta['encoding_map'].values())
                            })
                            st.dataframe(encoding_df, use_container_width=True, hide_index=True)

                        elif meta['type'] == 'numeric_multiclass':
                            # Show formula
                            st.code(meta['formula'])
                            st.caption(f"Maps: [{meta['min']:.2f}, {meta['max']:.2f}] → [-1, +1]")

                        elif meta['type'] == 'categorical_multiclass':
                            # Show dummy encoding pattern
                            st.caption(f"**Reference level:** {meta['reference_level']} (implicit, all zeros)")
                            st.caption(f"**Dummy columns:** {', '.join(meta['dummy_columns'])}")

                            # Show encoding pattern table
                            encoding_rows = []
                            for orig_val, pattern in meta['encoding_map'].items():
                                encoding_rows.append({
                                    'Original Value': orig_val,
                                    'Encoding Pattern': str(pattern),
                                    'Is Reference': '✓' if orig_val == meta['reference_level'] else ''
                                })

                            encoding_pattern_df = pd.DataFrame(encoding_rows)
                            st.dataframe(encoding_pattern_df, use_container_width=True, hide_index=True)

                        st.markdown("---")

                # ════════════════════════════════════════════════════════════════════════════
                # SPECIAL SECTION FOR DoE CODING: DATA COMPARISON (Original ↔ Transformed)
                # ════════════════════════════════════════════════════════════════════════════
                st.markdown("---")
                st.markdown("### 📊 Data Comparison: Original ↔ Transformed")

                col_toggle_orig, col_toggle_trans = st.columns(2)

                with col_toggle_orig:
                    show_original_doe = st.checkbox("📥 Show Original Data", value=True, key="doe_show_original")

                with col_toggle_trans:
                    show_transformed_doe = st.checkbox("📤 Show Transformed Data", value=True, key="doe_show_transformed")

                if show_original_doe and show_transformed_doe:
                    # Show side by side
                    col_data_orig, col_data_trans = st.columns(2)

                    with col_data_orig:
                        st.markdown("**🔵 Original Data**")
                        st.dataframe(
                            result_col['original_slice_col'].head(20),
                            use_container_width=True,
                            height=400
                        )

                    with col_data_trans:
                        st.markdown("**🟢 Transformed Data (Coded)**")
                        format_dict_doe = {col: "{:.3f}" for col in result_col['transformed_col'].select_dtypes(include=[np.number]).columns}
                        st.dataframe(
                            result_col['transformed_col'].head(20).style.format(format_dict_doe, na_rep="-"),
                            use_container_width=True,
                            height=400
                        )

                elif show_original_doe:
                    st.markdown("**🔵 Original Data**")
                    st.dataframe(result_col['original_slice_col'], use_container_width=True)

                elif show_transformed_doe:
                    st.markdown("**🟢 Transformed Data (Coded)**")
                    format_dict_doe = {col: "{:.3f}" for col in result_col['transformed_col'].select_dtypes(include=[np.number]).columns}
                    st.dataframe(
                        result_col['transformed_col'].style.format(format_dict_doe, na_rep="-"),
                        use_container_width=True
                    )

                # Export options for DoE Coding
                st.markdown("---")
                st.markdown("### 💾 Export Options")

                col_export_a, col_export_b = st.columns(2)

                with col_export_a:
                    # Download transformed data as CSV
                    csv_data = result_col['transformed_col'].to_csv(index=False)
                    st.download_button(
                        label="📥 Download Transformed Data (CSV)",
                        data=csv_data,
                        file_name="doe_coded_data.csv",
                        mime="text/csv",
                        use_container_width=True,
                        key="doe_download_csv"
                    )

                with col_export_b:
                    # Download encoding metadata as JSON
                    import json
                    encoding_json = json.dumps(metadata, indent=2, default=str)
                    st.download_button(
                        label="📥 Download Encoding Metadata (JSON)",
                        data=encoding_json,
                        file_name="doe_encoding_metadata.json",
                        mime="application/json",
                        use_container_width=True,
                        key="doe_download_json"
                    )

            # Data Preview section (ONLY for NON-DoE transformations)
            else:
                st.markdown("### 👁️ Data Preview")

                col_preview1, col_preview2 = st.columns(2)

                with col_preview1:
                    preview_type_col = st.radio(
                        "Preview type:",
                        ["First 10 rows", "Random samples", "Statistics"],
                        horizontal=True,
                        key="col_preview_type"
                    )

                with col_preview2:
                    n_preview_col = st.slider("Rows to show:", 5, 20, 10, key="col_preview_rows")

                # Show preview of transformed data
                transformed_data_col = result_col['transformed_col']

                # Create format dict: only format numeric columns
                format_dict_col_trans = {col: "{:.3f}" for col in transformed_data_col.select_dtypes(include=[np.number]).columns}

                if preview_type_col == "First 10 rows":
                    st.dataframe(
                        transformed_data_col.head(n_preview_col).style.format(format_dict_col_trans, na_rep="-"),
                        use_container_width=True,
                        height=300
                    )
                elif preview_type_col == "Random samples":
                    st.dataframe(
                        transformed_data_col.sample(n=min(n_preview_col, len(transformed_data_col)), random_state=42).style.format(format_dict_col_trans, na_rep="-"),
                        use_container_width=True,
                        height=300
                    )
                else:  # Statistics
                    st.dataframe(
                        transformed_data_col.describe().style.format("{:.3f}"),
                        use_container_width=True
                    )

                st.info(f"📊 Showing transformed data: {transformed_data_col.shape[0]} samples × {transformed_data_col.shape[1]} variables")

            # Save section
            st.markdown("---")
            st.markdown("### Save Transformation")
            st.info("Review the transformation above, then save it to workspace if satisfied")
            
            # FIXED SAVE LOGIC FOR COLUMN TRANSFORMATIONS
            col_save_col, col_download_col = st.columns(2)

            with col_save_col:
                if st.button("💾 Save to Workspace", type="primary", key="save_col_transform", use_container_width=True):
                    try:
                        # Ensure transformation_history exists
                        if 'transformation_history' not in st.session_state:
                            st.session_state.transformation_history = {}

                        # Get current transformation result
                        result_col = st.session_state.current_col_transform_result

                        # CORREZIONE: Preserva SEMPRE la struttura originale del dataset
                        full_transformed_col = data.copy()  # Copia completa del dataset originale
                        transformed_col = result_col['transformed_col']
                        transform_code_col = result_col['transform_code_col']
                        col_range_c = result_col['col_range_c']

                        if transform_code_col == "blsc":
                            # Block scaling affects entire dataset - CASO SPECIALE
                            full_transformed_col = transformed_col
                        else:
                            # Handle shape changes - MA PRESERVA METADATA
                            if transformed_col.shape[0] != data.shape[0]:
                                # Row reduction (column derivatives)
                                full_transformed_col = full_transformed_col.iloc[:transformed_col.shape[0], :].copy()

                            # SOSTITUISCI SOLO LE COLONNE TRASFORMATE - MANTIENI TUTTO IL RESTO
                            full_transformed_col.iloc[:, col_range_c[0]:col_range_c[1]] = transformed_col

                        # Create transformation name
                        dataset_name = st.session_state.get('current_dataset', 'Dataset')
                        base_name = dataset_name.split('.')[0].replace('_ORIGINAL', '')
                        transformed_name_col = f"{base_name}.{transform_code_col}"

                        # Save to workspace with all required metadata
                        st.session_state.transformation_history[transformed_name_col] = {
                            'data': full_transformed_col,
                            'transform': result_col['selected_transform_col'],
                            'params': result_col['params_col'],
                            'col_range': col_range_c,
                            'timestamp': pd.Timestamp.now(),
                            'original_dataset': dataset_name,
                            'transform_type': 'column_transformation'
                        }

                        # Update current data
                        st.session_state.current_data = full_transformed_col
                        st.session_state.current_dataset = transformed_name_col

                        # Clear the transformation result
                        del st.session_state.current_col_transform_result

                        # Show success messages
                        st.success(f"✅ Transformation saved as: **{transformed_name_col}**")
                        st.info("📊 Dataset is now active and ready for PCA/DOE")
                        st.info(f"🔒 **Structure preserved**: All metadata columns maintained")

                        # Force refresh of the interface
                        st.rerun()

                    except Exception as e:
                        st.error(f"❌ Error saving transformation: {str(e)}")
                        st.error("Please try applying the transformation again")

                        # Debug traceback
                        import traceback
                        st.code(traceback.format_exc())

            with col_download_col:
                # Download XLSX button
                if 'current_col_transform_result' in st.session_state:
                    result_col = st.session_state.current_col_transform_result

                    # Build full transformed dataset (same logic as save)
                    full_transformed_col = data.copy()
                    transformed_col = result_col['transformed_col']
                    transform_code_col = result_col['transform_code_col']
                    col_range_c = result_col['col_range_c']

                    if transform_code_col == "blsc":
                        # Block scaling affects entire dataset
                        full_transformed_col = transformed_col
                    else:
                        # Handle shape changes
                        if transformed_col.shape[0] != data.shape[0]:
                            full_transformed_col = full_transformed_col.iloc[:transformed_col.shape[0], :].copy()
                        full_transformed_col.iloc[:, col_range_c[0]:col_range_c[1]] = transformed_col

                    # Create XLSX file
                    import io
                    buffer = io.BytesIO()
                    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                        full_transformed_col.to_excel(writer, index=True, sheet_name='Transformed Data')
                    buffer.seek(0)

                    # Generate filename
                    dataset_name = st.session_state.get('current_dataset', 'Dataset')
                    base_name = dataset_name.split('.')[0].replace('_ORIGINAL', '')
                    filename = f"{base_name}.{transform_code_col}.xlsx"

                    st.download_button(
                        label="📥 Download XLSX",
                        data=buffer,
                        file_name=filename,
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True,
                        key="download_col_transform_xlsx"
                    )

    # ===== PREPROCESSING THEORY TAB =====
    with tab3:
        st.markdown("## Interactive Preprocessing Effects Tutorial")
        st.markdown("*Visual demonstration of how preprocessing methods handle spectral artifacts*")

        # Get unified color scheme for consistent styling across tabs
        colors = get_unified_color_schemes()

        if not PREPROCESSING_THEORY_AVAILABLE:
            st.warning("⚠️ **Preprocessing Theory Module Not Available**")
            st.info("""
            The preprocessing theory module is not yet installed. This tab provides:
            - Interactive simulated spectral datasets
            - Visual demonstrations of preprocessing effects
            - Educational content on preprocessing best practices

            Contact [chemometricsolutions.com](https://chemometricsolutions.com) for the full version.
            """)
        else:
            # ========================================================================
            # NEW COMPACT VISUALIZATION - Poster Style
            # ========================================================================

            st.markdown("---")

            # Method Selector (Horizontal Radio)
            st.markdown("### Select Preprocessing Method")
            selected_method = st.radio(
                "Method:",
                options=[
                    "SNV Transform",
                    "1st Derivative",
                    "2nd Derivative",
                    "1st Derivative + SG",
                    "2nd Derivative + SG",
                    "SNV + 1st Derivative (SG)",
                    "SNV + 2nd Derivative (SG)"
                ],
                horizontal=True,
                key="preproc_method_selector"
            )

            # Show SG parameters when applicable
            if "SG" in selected_method:
                st.markdown("---")

                # Determine which parameters are being used
                if "1st" in selected_method:
                    deriv_order = 1
                    window_size = 11
                else:  # 2nd derivative
                    deriv_order = 2
                    window_size = 15

                poly_order = 3  # Always use 3rd order (more conservative)

                col_param1, col_param2, col_param3 = st.columns(3)

                with col_param1:
                    st.metric("Derivative Order", f"{deriv_order}")
                with col_param2:
                    st.metric("Polynomial Order", f"{poly_order}")
                with col_param3:
                    st.metric("Window Size", f"{window_size}")

                st.caption("""
                **Savitzky-Golay Parameters:**
                - **Derivative order**: 0 (smoothing only), 1 (slope), or 2 (curvature)
                - **Polynomial order**: 3rd degree recommended (more conservative, preserves features). 2nd degree is more aggressive
                - **Window size**: Odd number of points. Larger = more smoothing but may lose detail
                """)

            st.markdown("---")

            # Generate simulated data for 3 scenarios (only need 3 samples per scenario)
            try:
                # Get baseline shift, baseline drift, and global intensity datasets
                generator = SimulatedSpectralDataGenerator(
                    n_samples=3,
                    n_variables=500,
                    wavenumber_min=400.0,
                    wavenumber_max=1800.0,
                    noise_level_db=70.0,  # Higher dB = better SNR = cleaner signal for education
                    random_state=42
                )

                # Generate 4 datasets
                baseline_shift_data, wavenumbers = generator.generate_baseline_shift_spectra()
                baseline_drift_data, _ = generator.generate_baseline_drift_spectra()
                global_intensity_data, _ = generator.generate_global_intensity_spectra()

                # Generate Combined Effects data (all 3 problems together)
                combined_effects_data, _, _ = generator.generate_combined_effects()

                # Select 2 samples from each dataset for visualization
                sample_indices = [0, 2]  # Use first and third sample

                # Create 2x4 subplot grid
                from plotly.subplots import make_subplots

                # Create dynamic title for transformed row with SG parameters
                if "SG" in selected_method:
                    if "1st" in selected_method:
                        transform_title = f"{selected_method.replace(' (SG)', '')} (w=11, p=3)"
                    else:
                        transform_title = f"{selected_method.replace(' (SG)', '')} (w=15, p=3)"
                else:
                    transform_title = f"After {selected_method}"

                fig = make_subplots(
                    rows=2, cols=4,
                    subplot_titles=(
                        "Baseline Shift", "Baseline Drift", "Global Intensity", "Combined Effects",
                        transform_title, transform_title, transform_title, transform_title
                    ),
                    vertical_spacing=0.15,
                    horizontal_spacing=0.06,  # Reduced spacing for 4 columns
                    row_heights=[0.5, 0.5]
                )

                # Colors for 2 samples
                colors_samples = ['rgb(0, 0, 255)', 'rgb(255, 0, 0)']  # Blue and Red

                # ========================================================================
                # ROW 1: ORIGINAL DATA (3 scenarios)
                # ========================================================================

                # Column 1: Baseline Shift
                for i, idx in enumerate(sample_indices):
                    fig.add_trace(
                        go.Scatter(
                            x=wavenumbers,
                            y=baseline_shift_data.iloc[idx].values,
                            mode='lines',
                            name='',
                            line=dict(color=colors_samples[i], width=2),
                            showlegend=False
                        ),
                        row=1, col=1
                    )

                # Column 2: Baseline Drift
                for i, idx in enumerate(sample_indices):
                    fig.add_trace(
                        go.Scatter(
                            x=wavenumbers,
                            y=baseline_drift_data.iloc[idx].values,
                            mode='lines',
                            name=f'Sample {i+1}' if i == 0 else '',
                            line=dict(color=colors_samples[i], width=2),
                            showlegend=False,
                            legendgroup=f'sample{i+1}'
                        ),
                        row=1, col=2
                    )

                # Column 3: Global Intensity
                for i, idx in enumerate(sample_indices):
                    fig.add_trace(
                        go.Scatter(
                            x=wavenumbers,
                            y=global_intensity_data.iloc[idx].values,
                            mode='lines',
                            name=f'Sample {i+1}' if i == 0 else '',
                            line=dict(color=colors_samples[i], width=2),
                            showlegend=False,
                            legendgroup=f'sample{i+1}'
                        ),
                        row=1, col=3
                    )

                # Column 4: Combined Effects (all 3 problems)
                for i, idx in enumerate(sample_indices):
                    fig.add_trace(
                        go.Scatter(
                            x=wavenumbers,
                            y=combined_effects_data.iloc[idx].values,
                            mode='lines',
                            name='',
                            line=dict(color=colors_samples[i], width=2),
                            showlegend=False,
                            legendgroup=f'sample{i+1}'
                        ),
                        row=1, col=4
                    )

                # ========================================================================
                # ROW 2: TRANSFORMED DATA (apply selected method)
                # ========================================================================

                # Create analyzers for all 4 scenarios
                analyzer_bs = PreprocessingEffectsAnalyzer(baseline_shift_data.iloc[sample_indices])
                analyzer_bd = PreprocessingEffectsAnalyzer(baseline_drift_data.iloc[sample_indices])
                analyzer_gi = PreprocessingEffectsAnalyzer(global_intensity_data.iloc[sample_indices])
                analyzer_ce = PreprocessingEffectsAnalyzer(combined_effects_data.iloc[sample_indices])

                # Store original wavenumbers before any modification
                wavenumbers_plot = wavenumbers.copy()

                # Apply transformation based on selected method
                if selected_method == "SNV Transform":
                    transformed_bs = analyzer_bs.snv_transform()
                    transformed_bd = analyzer_bd.snv_transform()
                    transformed_gi = analyzer_gi.snv_transform()
                    transformed_ce = analyzer_ce.snv_transform()
                    ylabel_transform = "SNV"

                elif selected_method == "1st Derivative":
                    transformed_bs = analyzer_bs.first_derivative()
                    transformed_bd = analyzer_bd.first_derivative()
                    transformed_gi = analyzer_gi.first_derivative()
                    transformed_ce = analyzer_ce.first_derivative()
                    ylabel_transform = "dI/dλ"
                    wavenumbers_plot = wavenumbers[:-1]  # One fewer point

                elif selected_method == "2nd Derivative":
                    transformed_bs = analyzer_bs.second_derivative()
                    transformed_bd = analyzer_bd.second_derivative()
                    transformed_gi = analyzer_gi.second_derivative()
                    transformed_ce = analyzer_ce.second_derivative()
                    ylabel_transform = "d²I/dλ²"
                    wavenumbers_plot = wavenumbers[:-2]  # Two fewer points

                elif selected_method == "1st Derivative + SG":
                    transformed_bs = analyzer_bs.first_derivative_savitzky_golay(window=11, polyorder=3)
                    transformed_bd = analyzer_bd.first_derivative_savitzky_golay(window=11, polyorder=3)
                    transformed_gi = analyzer_gi.first_derivative_savitzky_golay(window=11, polyorder=3)
                    transformed_ce = analyzer_ce.first_derivative_savitzky_golay(window=11, polyorder=3)
                    ylabel_transform = "dI/dλ (SG w=11, p=3)"

                elif selected_method == "2nd Derivative + SG":
                    transformed_bs = analyzer_bs.second_derivative_savitzky_golay(window=15, polyorder=3)
                    transformed_bd = analyzer_bd.second_derivative_savitzky_golay(window=15, polyorder=3)
                    transformed_gi = analyzer_gi.second_derivative_savitzky_golay(window=15, polyorder=3)
                    transformed_ce = analyzer_ce.second_derivative_savitzky_golay(window=15, polyorder=3)
                    ylabel_transform = "d²I/dλ² (SG w=15, p=3)"

                elif selected_method == "SNV + 1st Derivative (SG)":
                    # Step 1: SNV
                    snv_bs = analyzer_bs.snv_transform()
                    snv_bd = analyzer_bd.snv_transform()
                    snv_gi = analyzer_gi.snv_transform()
                    snv_ce = analyzer_ce.snv_transform()
                    # Step 2: 1st Derivative + SG on SNV result
                    transformed_bs = PreprocessingEffectsAnalyzer(snv_bs).first_derivative_savitzky_golay(window=11, polyorder=3)
                    transformed_bd = PreprocessingEffectsAnalyzer(snv_bd).first_derivative_savitzky_golay(window=11, polyorder=3)
                    transformed_gi = PreprocessingEffectsAnalyzer(snv_gi).first_derivative_savitzky_golay(window=11, polyorder=3)
                    transformed_ce = PreprocessingEffectsAnalyzer(snv_ce).first_derivative_savitzky_golay(window=11, polyorder=3)
                    ylabel_transform = "SNV → dI/dλ (SG w=11, p=3)"

                elif selected_method == "SNV + 2nd Derivative (SG)":
                    # Step 1: SNV
                    snv_bs = analyzer_bs.snv_transform()
                    snv_bd = analyzer_bd.snv_transform()
                    snv_gi = analyzer_gi.snv_transform()
                    snv_ce = analyzer_ce.snv_transform()
                    # Step 2: 2nd Derivative + SG on SNV result (use SG to reduce noise)
                    transformed_bs = PreprocessingEffectsAnalyzer(snv_bs).second_derivative_savitzky_golay(window=15, polyorder=3)
                    transformed_bd = PreprocessingEffectsAnalyzer(snv_bd).second_derivative_savitzky_golay(window=15, polyorder=3)
                    transformed_gi = PreprocessingEffectsAnalyzer(snv_gi).second_derivative_savitzky_golay(window=15, polyorder=3)
                    transformed_ce = PreprocessingEffectsAnalyzer(snv_ce).second_derivative_savitzky_golay(window=15, polyorder=3)
                    ylabel_transform = "SNV → d²I/dλ² (SG w=15, p=3)"

                # Column 1: Transformed Baseline Shift
                for i in range(len(sample_indices)):
                    fig.add_trace(
                        go.Scatter(
                            x=wavenumbers_plot,
                            y=transformed_bs.iloc[i].values,
                            mode='lines',
                            name='',
                            line=dict(color=colors_samples[i], width=2),
                            showlegend=False
                        ),
                        row=2, col=1
                    )

                # Column 2: Transformed Baseline Drift
                for i in range(len(sample_indices)):
                    fig.add_trace(
                        go.Scatter(
                            x=wavenumbers_plot,
                            y=transformed_bd.iloc[i].values,
                            mode='lines',
                            name='',
                            line=dict(color=colors_samples[i], width=2),
                            showlegend=False
                        ),
                        row=2, col=2
                    )

                # Column 3: Transformed Global Intensity
                for i in range(len(sample_indices)):
                    fig.add_trace(
                        go.Scatter(
                            x=wavenumbers_plot,
                            y=transformed_gi.iloc[i].values,
                            mode='lines',
                            name='',
                            line=dict(color=colors_samples[i], width=2),
                            showlegend=False
                        ),
                        row=2, col=3
                    )

                # Column 4: Transformed Combined Effects
                for i in range(len(sample_indices)):
                    fig.add_trace(
                        go.Scatter(
                            x=wavenumbers_plot,
                            y=transformed_ce.iloc[i].values,
                            mode='lines',
                            name='',
                            line=dict(color=colors_samples[i], width=2),
                            showlegend=False
                        ),
                        row=2, col=4
                    )

                # Update axis labels
                fig.update_xaxes(title_text="Wavenumber (cm⁻¹)", row=1, col=1)
                fig.update_xaxes(title_text="Wavenumber (cm⁻¹)", row=1, col=2)
                fig.update_xaxes(title_text="Wavenumber (cm⁻¹)", row=1, col=3)
                fig.update_xaxes(title_text="Wavenumber (cm⁻¹)", row=1, col=4)
                fig.update_xaxes(title_text="Wavenumber (cm⁻¹)", row=2, col=1)
                fig.update_xaxes(title_text="Wavenumber (cm⁻¹)", row=2, col=2)
                fig.update_xaxes(title_text="Wavenumber (cm⁻¹)", row=2, col=3)
                fig.update_xaxes(title_text="Wavenumber (cm⁻¹)", row=2, col=4)

                fig.update_yaxes(title_text="Intensity", row=1, col=1)
                fig.update_yaxes(title_text="Intensity", row=1, col=2)
                fig.update_yaxes(title_text="Intensity", row=1, col=3)
                fig.update_yaxes(title_text="Intensity", row=1, col=4)
                fig.update_yaxes(title_text=ylabel_transform, row=2, col=1)
                fig.update_yaxes(title_text=ylabel_transform, row=2, col=2)
                fig.update_yaxes(title_text=ylabel_transform, row=2, col=3)
                fig.update_yaxes(title_text=ylabel_transform, row=2, col=4)

                # Update layout
                fig.update_layout(
                    height=750,
                    showlegend=False,
                    hovermode='closest'
                )

                st.plotly_chart(fig, use_container_width=True)

                # ========================================================================
                # INFO PANEL - Formula and Effectiveness
                # ========================================================================

                st.markdown("---")
                st.markdown("### Method Information")

                if selected_method == "SNV Transform":
                    st.latex(r"x_{SNV} = \frac{x - \bar{x}}{\sigma_x}")
                    st.info("""
                    **Standard Normal Variate (SNV)**

                    Row-wise autoscaling: centers and scales each spectrum to zero mean and unit variance.

                    **Effectiveness:**
                    - ✅ **Baseline Shift**: Highly effective (removes constant offsets)
                    - ⚠️ **Baseline Drift**: Partially effective (reduces but doesn't eliminate linear drift)
                    - ✅ **Global Intensity**: Highly effective (normalizes intensity differences)
                    """)

                    st.warning("""
                    ⚠️ **Caution - Loading Interpretation (Oliveri et al., 2019):**

                    While SNV effectively removes scatter effects, it may **shift spectral information
                    along the signal profile**. This can lead to misinterpretation of PCA loadings -
                    important variables may appear at unexpected spectral positions!

                    **The problem:** SNV normalizes using mean and std calculated across ALL variables.
                    This creates artificial correlations between variables that were originally independent.
                    """)

                    st.success("""
                    ✅ **Alternatives to SNV:**

                    1. **MSC (Multiplicative Scatter Correction)** - Similar correction but uses a reference spectrum
                    2. **Derivatives** - If baseline correction is the main goal, use 1st or 2nd derivative instead
                    3. **SNV + Derivative** - Combine SNV with derivative to mitigate the loading shift problem
                    4. **Selective SNV** - Weight variables before calculating mean/std (Roger et al., 2018)

                    **Recommendation:** For PCA interpretation, prefer **derivatives** over SNV alone,
                    or use **SNV → 2nd Derivative** combination.
                    """)

                elif selected_method == "1st Derivative":
                    st.latex(r"\frac{dI}{d\lambda} \approx I_{i+1} - I_i")
                    st.info("""
                    **First Derivative (Finite Differences)**

                    Computes rate of change between adjacent points.

                    **Effectiveness:**
                    - ✅ **Baseline Shift**: Highly effective (removes constant offsets)
                    - ❌ **Baseline Drift**: NOT effective (linear drift becomes constant offset in derivative)
                    - ❌ **Global Intensity**: Not effective (preserves relative differences)
                    """)

                    st.warning("""
                    ⚠️ **Caution - Loading Interpretation (Oliveri et al., 2019):**

                    After derivative transformation, **PCA loading interpretation changes completely**:

                    - Original **peak maxima** → correspond to **zero-crossings** in derivative loadings
                    - **High loading values** → indicate slopes of original peaks, NOT peak positions
                    - Positive/negative loading correspondence is **inverted**

                    This is the "**loading paradox**" described by Oliveri et al.
                    """)

                    st.success("""
                    ✅ **Solution - Anti-derivative Transform:**

                    To correctly interpret loadings after derivative preprocessing:

                    1. **Apply anti-derivative** (integral) to the loading profile
                    2. This reconstructs the original peak shapes
                    3. Now loadings can be interpreted with classical rules

                    **Formula:** F(x) = ∫f(t)dt

                    The anti-derivative transform recovers the contribution of original variables.
                    """)

                elif selected_method == "2nd Derivative":
                    st.latex(r"\frac{d^2I}{d\lambda^2} \approx I_{i+1} - 2I_i + I_{i-1}")
                    st.info("""
                    **Second Derivative (Finite Differences)**

                    Emphasizes peak curvature and resolves overlapping peaks.

                    **Effectiveness:**
                    - ✅ **Baseline Shift**: Highly effective (removes constant and linear baselines)
                    - ✅ **Baseline Drift**: Highly effective (eliminates linear and quadratic baselines)
                    - ❌ **Global Intensity**: Not effective (preserves relative differences)

                    ⚠️ **Warning**: Amplifies noise significantly!
                    """)

                    st.warning("""
                    ⚠️ **Caution - Loading Interpretation (Oliveri et al., 2019):**

                    Second derivative causes **double inversion** of loading interpretation:

                    - Original **peaks** → become **negative peaks** (inverted) in 2nd derivative
                    - Original **peak maxima** → now appear as **minima** in loadings
                    - Sign correspondence is **doubly inverted**

                    **Example from Raman data (Oliveri et al.):**
                    Variables characterizing minerals appeared in opposite positions of the loading plot!
                    """)

                    st.success("""
                    ✅ **Solution - Double Anti-derivative Transform:**

                    For 2nd derivative, apply anti-derivative **TWICE** to recover original variable importance:

                    1. First anti-derivative: converts 2nd derivative back to 1st derivative shape
                    2. Second anti-derivative: recovers original peak shape

                    This allows direct interpretation matching original spectral features.
                    """)

                elif selected_method == "1st Derivative + SG":
                    st.latex(r"\frac{dI}{d\lambda} \text{ (Savitzky-Golay smoothing)}")
                    st.info("""
                    **First Derivative with Savitzky-Golay Smoothing**

                    Polynomial smoothing + derivative calculation (window=11, polyorder=3).

                    **Effectiveness:**
                    - ✅ **Baseline Shift**: Highly effective
                    - ❌ **Baseline Drift**: NOT effective (same as finite differences - SG doesn't change this)
                    - ❌ **Global Intensity**: Not effective
                    - ✅ **Noise Reduction**: Good balance between smoothing and peak preservation
                    """)

                    st.warning("""
                    ⚠️ **Loading Interpretation:** Same caution as simple 1st derivative applies.
                    The Savitzky-Golay smoothing helps with noise but doesn't change the loading paradox.
                    Use anti-derivative transform on loadings for correct interpretation.
                    """)

                elif selected_method == "2nd Derivative + SG":
                    st.latex(r"\frac{d^2I}{d\lambda^2} \text{ (Savitzky-Golay smoothing)}")
                    st.info("""
                    **Second Derivative with Savitzky-Golay Smoothing**

                    Polynomial smoothing + second derivative (window=15, polyorder=3).

                    **Effectiveness:**
                    - ✅ **Baseline Shift**: Highly effective
                    - ✅ **Baseline Drift**: Highly effective
                    - ❌ **Global Intensity**: Not effective
                    - ✅ **Noise Reduction**: Essential for second derivatives!

                    **Recommended** for resolving overlapping peaks with noisy data.
                    """)

                    st.warning("""
                    ⚠️ **Loading Interpretation:** Same caution as simple 2nd derivative applies.
                    Apply double anti-derivative to loadings for correct interpretation.
                    """)

                elif selected_method == "SNV + 1st Derivative (SG)":
                    st.latex(r"\text{SNV} \rightarrow \frac{dI}{d\lambda} \text{ (SG)}")
                    st.info("""
                    **SNV followed by First Derivative with Savitzky-Golay Smoothing**

                    Two-step preprocessing: normalization then smoothed baseline removal.

                    **Effectiveness:**
                    - ✅ **Baseline Shift**: Highly effective
                    - ⚠️ **Baseline Drift**: Partially effective (SNV helps, but 1st Der doesn't fully remove drift)
                    - ✅ **Global Intensity**: Highly effective (SNV normalizes first)
                    - ✅ **Combined Effects**: Good for multiple artifacts

                    **Why this combination?**
                    - **SNV first** normalizes intensity differences between samples
                    - **1st Derivative + SG** then removes baseline shift with noise reduction
                    - Result: Clean spectra ready for analysis without noise amplification
                    """)

                    st.success("""
                    ✅ **Advantage of this combination:**

                    The derivative step **partially mitigates** the SNV loading shift problem!

                    By applying derivative after SNV, the artificial correlations created by SNV
                    are reduced, leading to more interpretable loadings.

                    **Still recommended:** Apply anti-derivative to loadings for best interpretation.
                    """)

                elif selected_method == "SNV + 2nd Derivative (SG)":
                    st.latex(r"\text{SNV} \rightarrow \frac{d^2I}{d\lambda^2} \text{ (SG)}")
                    st.success("""
                    **⭐ SNV followed by Second Derivative with Savitzky-Golay Smoothing**

                    **RECOMMENDED for complex datasets with multiple artifacts!**

                    **Effectiveness:**
                    - ✅ **Baseline Shift**: Highly effective
                    - ✅ **Baseline Drift**: Highly effective
                    - ✅ **Global Intensity**: Highly effective
                    - ⭐ **Combined Effects**: Best overall performance

                    **Why this combination works best:**
                    1. **SNV first** normalizes intensity differences and stabilizes baseline
                    2. **2nd Derivative** removes remaining baselines AND mitigates SNV loading shift
                    3. **SG smoothing** prevents noise amplification from derivative operation

                    **Look at Column 4** (Combined Effects) to see how this method handles all three problems simultaneously!
                    """)

                    st.info("""
                    💡 **Loading Interpretation (Oliveri et al., 2019):**

                    This combination is the **best compromise** for PCA interpretation:
                    - The 2nd derivative helps reduce the SNV loading shift problem
                    - Still apply double anti-derivative transform to loadings for best results

                    **This is why SNV → 2nd Der is preferred over SNV alone for PCA!**
                    """)

                # Add detailed SG explanation for methods using SG
                if "SG" in selected_method:
                    st.markdown("---")
                    st.markdown("### 📚 Savitzky-Golay Filter Details")

                    # Determine current parameters
                    if "1st" in selected_method:
                        current_deriv = 1
                        current_window = 11
                    else:
                        current_deriv = 2
                        current_window = 15

                    st.info(f"""
**Savitzky-Golay (1964)** - Smoothing and differentiation by simplified least squares

📖 *Anal. Chem. 1964, 36, 8, 1627–1639* — One of the most cited papers in analytical chemistry (>10,000 citations)

**Current Parameters:**

| Parameter | Description | Value |
|-----------|-------------|-------|
| **Derivative order** | 0=smoothing, 1=1st deriv, 2=2nd deriv | **{current_deriv}** |
| **Polynomial order** | Degree of fitting polynomial | **3** (recommended) |
| **Window size** | Number of points (must be odd) | **{current_window}** |

**Why polynomial order 3?**
- **3rd degree (cubic)**: More conservative, better preserves spectral features ✅
- **2nd degree (quadratic)**: More aggressive smoothing, may over-smooth and lose information ⚠️
- **1st degree (linear)**: Linear interpolation only - too aggressive, loses all curvature! ❌

**Window size trade-off:**
- **Larger window**: More noise reduction, but may smooth out real peaks
- **Smaller window**: Preserves detail, but less noise reduction
- **Rule of thumb**: Window should be ~2-3× the FWHM (full width at half maximum) of your narrowest peak

**Why these specific windows?**
- **1st derivative (w=11)**: Smaller window preserves peak shape while removing baseline
- **2nd derivative (w=15)**: Larger window needed because 2nd derivative is more noise-sensitive
                    """)

                # Add Combined Effects explanation
                st.markdown("---")
                st.markdown("### 🔬 Handling Combined Effects")
                st.warning("""
**When all THREE problems occur together** (Baseline Shift + Baseline Drift + Global Intensity):

**Recommended Preprocessing Sequence:**

1. **First: SNV (Standard Normal Variate)**
   - Normalizes intensity differences
   - Partially corrects baseline issues

2. **Then: 2nd Derivative + Savitzky-Golay**
   - Removes remaining baseline drift
   - Enhances peak resolution
   - Smoothing prevents noise amplification

**Why this order?**
- SNV stabilizes the intensity first, preventing the derivative from amplifying intensity variations
- 2nd Derivative then focuses on peak shape without being confused by intensity differences

**The 4th column** (Combined Effects) demonstrates this combined preprocessing applied to data with all 3 artifacts simultaneously. Notice how the combination handles all three problems effectively!
                """)

                # Scientific reference
                st.markdown("---")
                st.markdown("### 📚 Scientific Reference")

                st.info("""
**This tutorial is based on:**

Oliveri P., Malegori C., Simonetti R., Casale M. (2019).
*The impact of signal pre-processing on the final interpretation of analytical outcomes – A tutorial.*
**Analytica Chimica Acta**, 1058, 9-17.
[DOI: 10.1016/j.aca.2018.10.055](https://doi.org/10.1016/j.aca.2018.10.055)

**Key findings from the tutorial:**
- **Row pre-processing** (SNV, derivatives) can cause **misinterpretation of PCA loadings**
- **Anti-derivative transform** recovers correct variable importance after derivation
- **SNV** may shift information along the spectrum - use with caution for interpretation
- **First derivative**: removes constant baseline offsets only (NOT drift!)
- **Second derivative**: removes both offsets AND linear drifts
- **Polynomial order 3** recommended for Savitzky-Golay (more conservative than order 2)
- **SNV + 2nd Derivative**: best combination for complex data with multiple artifacts
                """)

                # Summary comparison table
                st.markdown("---")
                st.markdown("### Quick Reference: Method Effectiveness")

                st.caption("**Note:** All SG methods use polynomial order=3 (conservative). Window: 11 for 1st derivative, 15 for 2nd derivative.")

                comparison_data = {
                    "Method": ["SNV", "1st Der", "2nd Der", "1st+SG", "2nd+SG", "SNV+1st", "SNV+2nd"],
                    "Baseline Shift": ["✅", "✅", "✅", "✅", "✅", "✅", "✅"],
                    "Baseline Drift": ["⚠️", "❌", "✅", "❌", "✅", "⚠️", "✅"],
                    "Global Intensity": ["✅", "❌", "❌", "❌", "❌", "✅", "✅"],
                    "Loading Interpr.": ["⚠️ Shifted", "⚠️ Inverted", "⚠️ 2x Inv.", "⚠️ Inverted", "⚠️ 2x Inv.", "⚠️", "⚠️"],
                    "Combined Effects": ["⚠️", "❌", "⚠️", "❌", "⚠️", "✅", "⭐"],
                    "Noise": ["Low", "Med", "High", "Low", "Med", "Med", "Med"]
                }

                comparison_df = pd.DataFrame(comparison_data)
                st.dataframe(comparison_df, use_container_width=True, hide_index=True)

                st.caption("""
**Legend:**
- ✅ = Effective | ⚠️ = Partial/Caution | ❌ = Not effective | ⭐ = Best choice for complex data

**Loading Interpretation (Oliveri et al., 2019):**
- *Shifted* = Information may appear at wrong spectral positions
- *Inverted* = Apply anti-derivative to loadings for correct interpretation
- *2x Inv.* = Apply anti-derivative TWICE for correct interpretation
- For all derivative methods: Use anti-derivative transform on PCA loadings!
                """)

                st.success("""
                💡 **Best Practices (Oliveri et al., 2019):**
                - **For PCA interpretation**: Prefer derivatives over SNV alone to avoid loading shift
                - **1st Derivative**: Removes baseline shift only (NOT drift!) - apply anti-derivative to loadings
                - **2nd Derivative**: Removes both shift AND drift - apply anti-derivative TWICE to loadings
                - **SNV + 2nd Der + SG**: ⭐ Best for complex data - handles all artifacts + mitigates loading problems
                - **Always use polynomial order 3** for Savitzky-Golay (more conservative than order 2)
                """)

            except Exception as e:
                st.error(f"Error generating visualization: {str(e)}")
                import traceback
                st.code(traceback.format_exc())

        # OLD SECTIONS REMOVED - Keep only the new compact visualization above

    # IMPORTANTE: Posiziona la sidebar FUORI dai tabs
    display_transformation_sidebar()

def display_transformation_sidebar():
    """Display transformation history in sidebar - VERSIONE CORRETTA"""
    if 'transformation_history' in st.session_state and st.session_state.transformation_history:
        with st.sidebar:
            st.markdown("### 🔬 Transformation History")
            
            # Debug info
            st.write(f"Total transformations: {len(st.session_state.transformation_history)}")
            
            # Show recent transformations (last 5)
            recent_transforms = sorted(
                st.session_state.transformation_history.items(),
                key=lambda x: x[1]['timestamp'],
                reverse=True
            )[:5]
            
            for name, info in recent_transforms:
                # Create a cleaner display name
                display_name = name.split('.')[-1] if '.' in name else name
                
                with st.expander(f"**{display_name}**", expanded=False):
                    st.write(f"**Transform:** {info.get('transform', 'Unknown')}")
                    st.write(f"**Shape:** {info['data'].shape[0]} × {info['data'].shape[1]}")
                    st.write(f"**Time:** {info['timestamp'].strftime('%H:%M:%S')}")
                    
                    # Load button with unique key
                    button_key = f"sidebar_load_{name.replace('.', '_').replace(' ', '_')}"
                    if st.button(f"Load {display_name}", key=button_key):
                        st.session_state.current_data = info['data']
                        st.session_state.current_dataset = name
                        st.success(f"✅ Loaded: {display_name}")
                        st.rerun()
            
            if len(st.session_state.transformation_history) > 5:
                st.info(f"+ {len(st.session_state.transformation_history) - 5} more in workspace")
    else:
        # Debug per capire perché non appare
        with st.sidebar:
            st.markdown("### 🔬 Transformation History")
            st.write("No transformations saved yet")
            if 'transformation_history' in st.session_state:
                st.write(f"History exists but empty: {len(st.session_state.transformation_history)} items")
            else:
                st.write("transformation_history not in session_state")