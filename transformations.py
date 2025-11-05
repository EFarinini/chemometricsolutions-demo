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
        
        st.markdown("### Visualization Options")
        use_dark_theme_col = st.checkbox("Dark mode colors", value=True, key="col_transform_dark")

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
        st.markdown("*Learn how different preprocessing methods affect spectral data*")

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

        # Section 1: Dataset Selection
        with st.expander("📊 **1. Dataset Selection**", expanded=True):
            if PREPROCESSING_THEORY_AVAILABLE:
                st.subheader("Select Simulated Dataset")

                # Create two columns for scenario selector and sample count
                col1, col2 = st.columns(2)

                with col1:
                    scenario = st.selectbox(
                        "Choose scenario",
                        ["clean", "baseline_shift", "baseline_drift", "global_intensity", "combined_effects"],
                        key="preproc_scenario_v2",
                        help="Select the type of systematic variation in simulated spectra"
                    )

                with col2:
                    n_samples = st.slider(
                        "Number of samples",
                        min_value=10,
                        max_value=50,
                        value=30,
                        key="preproc_samples_v2",
                        help="Number of spectra to generate"
                    )

                # Generate datasets
                try:
                    all_datasets = get_all_simulated_datasets(n_samples=n_samples)

                    if scenario in all_datasets:
                        selected_dataset = all_datasets[scenario]

                        # Store in session state
                        st.session_state.preproc_data = selected_dataset['data']
                        st.session_state.preproc_wavenumbers = selected_dataset['wavenumbers']
                        st.session_state.preproc_scenario = scenario
                        st.session_state.preproc_dataset_info = selected_dataset

                        # Display dataset information
                        st.info(f"""
                        **{selected_dataset['effect_type']}**

                        {selected_dataset['description']}

                        - **Shape**: {selected_dataset['data'].shape[0]} samples × {selected_dataset['data'].shape[1]} variables
                        - **Wavenumber range**: {selected_dataset['wavenumber_min']:.1f} - {selected_dataset['wavenumber_max']:.1f} cm⁻¹
                        """)

                        # Show dataset preview
                        st.markdown("**Dataset Preview** (first 5 samples)")

                        preview_data = selected_dataset['data'].iloc[:5]
                        wavenumbers = selected_dataset['wavenumbers']

                        fig_preview = go.Figure()
                        colors_preview = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

                        for i in range(min(5, len(preview_data))):
                            fig_preview.add_trace(go.Scatter(
                                x=wavenumbers,
                                y=preview_data.iloc[i].values,
                                mode='lines',
                                name='',  # No legend labels
                                line=dict(color=colors_preview[i], width=1.5),
                                showlegend=False
                            ))

                        fig_preview.update_layout(
                            xaxis_title="Wavenumber (cm⁻¹)",
                            yaxis_title="Intensity",
                            height=350,
                            hovermode='closest',
                            showlegend=False
                        )

                        st.plotly_chart(fig_preview, use_container_width=True)

                        st.success(f"✅ Dataset loaded: {n_samples} samples ready for preprocessing")

                        # ===================================================================
                        # RECOMMENDED PREPROCESSING FOR COMBINED_EFFECTS
                        # ===================================================================

                        if scenario == 'combined_effects':
                            st.info("""
                            💡 **Suggested Preprocessing for Combined Effects**

                            **When you have ALL THREE problems together:**
                            - Baseline variations
                            - Intensity differences
                            - Noise + artifacts

                            **Recommended preprocessing sequence:**

                            1. **SNV (Standard Normal Variate)** - Row preprocessing
                               - Normalizes intensity differences
                               - Handles baseline variations

                            2. **2nd Derivative** - Row preprocessing
                               - Emphasizes peak shapes
                               - Removes remaining baseline artifacts

                            **Why this combination?**
                            - SNV first: Corrects intensity scaling and baseline level
                            - 2nd Derivative after: Resolves peak shape differences without amplifying noise (since SNV stabilizes it)

                            Try these settings in the preprocessing section below:
                            - ☑ Standard Normal Variate (SNV)
                            - ☑ 2nd Derivative (with Savitzky-Golay smoothing)
                            """)

                    else:
                        st.error(f"Scenario '{scenario}' not found in generated datasets")

                except Exception as e:
                    st.error(f"Error generating datasets: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())

            else:
                st.info("💡 This section will allow you to choose from various simulated spectral datasets with different characteristics (baseline drift, noise, scatter effects, etc.)")

        # Section 1.5: Test Preprocessing Strategies (only for combined_effects)
        if (PREPROCESSING_THEORY_AVAILABLE and
            'preproc_scenario' in st.session_state and
            st.session_state.preproc_scenario == 'combined_effects' and
            'category_metadata' in st.session_state):

            with st.expander("🧪 **1.5. Test Preprocessing Strategies**", expanded=False):
                st.subheader("🧪 Test Category-Specific Preprocessing")
                st.info("Compare preprocessing methods tailored to each category. See which methods work best for each type of spectral challenge.")

                # ===================================================================
                # 1. CATEGORY SELECTOR
                # ===================================================================
                st.markdown("### 1️⃣ Select Category to Test")

                test_category = st.radio(
                    "Choose category:",
                    options=[
                        "Category 1: Peak Height Variation",
                        "Category 2: Peak Shape Distortion",
                        "Category 3: Noise & Spikes"
                    ],
                    key="test_preproc_category",
                    horizontal=True
                )

                # Determine category index and sample range
                if "Category 1" in test_category:
                    cat_idx = 0
                    sample_start = 0
                    sample_end = 5
                    cat_color = '#1f77b4'
                    cat_name = "Category_1_Peak_Height"
                elif "Category 2" in test_category:
                    cat_idx = 1
                    sample_start = 10
                    sample_end = 15
                    cat_color = '#ff7f0e'
                    cat_name = "Category_2_Peak_Shape"
                else:  # Category 3
                    cat_idx = 2
                    sample_start = 20
                    sample_end = 25
                    cat_color = '#2ca02c'
                    cat_name = "Category_3_Noise_Spikes"

                st.markdown("---")

                # ===================================================================
                # 2. PREPROCESSING OPTIONS (Category-Specific)
                # ===================================================================
                st.markdown("### 2️⃣ Select Preprocessing Methods")

                # Get category data
                cat_data = st.session_state.preproc_data.iloc[sample_start:sample_end].copy()
                wavenumbers = st.session_state.preproc_wavenumbers

                selected_methods = []

                if cat_idx == 0:  # Category 1: Peak Height
                    st.markdown("**Recommended for Peak Height Variation:**")

                    col_opt1, col_opt2, col_opt3 = st.columns(3)

                    with col_opt1:
                        use_snv = st.checkbox("✓ Standard Normal Variate (SNV)", value=True, key="test_use_snv")
                        if use_snv:
                            selected_methods.append("SNV")

                    with col_opt2:
                        use_col_auto = st.checkbox("Column Autoscaling", value=False, key="test_use_col_auto")
                        if use_col_auto:
                            selected_methods.append("Column Autoscaling")

                    with col_opt3:
                        use_col_center = st.checkbox("Column Centering", value=False, key="test_use_col_center")
                        if use_col_center:
                            selected_methods.append("Column Centering")

                elif cat_idx == 1:  # Category 2: Peak Shape
                    st.markdown("**Recommended for Peak Shape Distortion:**")

                    col_opt1, col_opt2 = st.columns(2)

                    with col_opt1:
                        use_1st_sg = st.checkbox("✓ 1st Derivative (Savitzky-Golay)", value=True, key="test_use_1st_sg")
                        if use_1st_sg:
                            sg_window_1st = st.slider("Window", 5, 21, 11, 2, key="test_sg_win_1st")
                            sg_poly_1st = st.slider("Polyorder", 2, 4, 3, key="test_sg_poly_1st")
                            selected_methods.append(f"1st Derivative SG (w={sg_window_1st}, p={sg_poly_1st})")

                    with col_opt2:
                        use_2nd_sg = st.checkbox("2nd Derivative (Savitzky-Golay)", value=False, key="test_use_2nd_sg")
                        if use_2nd_sg:
                            sg_window_2nd = st.slider("Window", 5, 25, 15, 2, key="test_sg_win_2nd")
                            sg_poly_2nd = st.slider("Polyorder", 2, 4, 3, key="test_sg_poly_2nd")
                            selected_methods.append(f"2nd Derivative SG (w={sg_window_2nd}, p={sg_poly_2nd})")

                    use_snv_before = st.checkbox("Apply SNV before derivatives (optional)", value=False, key="test_use_snv_before")
                    if use_snv_before:
                        selected_methods.insert(0, "SNV (before derivatives)")

                else:  # Category 3: Noise & Spikes
                    st.markdown("**Recommended for Noise & Spike Artifacts:**")

                    col_opt1, col_opt2 = st.columns(2)

                    with col_opt1:
                        use_sg_smooth = st.checkbox("✓ Savitzky-Golay Smoothing", value=True, key="test_use_sg_smooth")
                        if use_sg_smooth:
                            sg_window_smooth = st.slider("Window", 5, 25, 15, 2, key="test_sg_win_smooth")
                            sg_poly_smooth = st.slider("Polyorder", 2, 4, 3, key="test_sg_poly_smooth")
                            selected_methods.append(f"SG Smoothing (w={sg_window_smooth}, p={sg_poly_smooth})")

                    with col_opt2:
                        use_moving_avg = st.checkbox("Moving Average", value=False, key="test_use_moving_avg")
                        if use_moving_avg:
                            ma_window = st.slider("Window", 3, 9, 5, 2, key="test_ma_window")
                            selected_methods.append(f"Moving Average (w={ma_window})")

                    use_outlier = st.checkbox("Outlier Removal (spike detection)", value=False, key="test_use_outlier")
                    if use_outlier:
                        sigma_thresh = st.slider("σ Threshold", 2.0, 4.0, 3.0, 0.5, key="test_sigma_thresh")
                        selected_methods.insert(0, f"Outlier Removal (σ={sigma_thresh})")

                    use_snv_after = st.checkbox("Apply SNV after smoothing (optional)", value=False, key="test_use_snv_after")
                    if use_snv_after:
                        selected_methods.append("SNV (after smoothing)")

                st.markdown("---")

                # ===================================================================
                # 3. APPLY PREPROCESSING AND VISUALIZE
                # ===================================================================
                st.markdown("### 3️⃣ Results Comparison")

                if len(selected_methods) > 0:
                    try:
                        # Apply selected preprocessing
                        processed_data = cat_data.copy()
                        analyzer = PreprocessingEffectsAnalyzer(processed_data)

                        # Category 1: Peak Height
                        if cat_idx == 0:
                            if use_snv:
                                processed_data = analyzer.snv_transform()
                            if use_col_auto:
                                # Column autoscaling
                                means = processed_data.mean(axis=0)
                                stds = processed_data.std(axis=0)
                                stds[stds == 0] = 1.0
                                processed_data = (processed_data - means) / stds
                            if use_col_center:
                                # Column centering
                                means = processed_data.mean(axis=0)
                                processed_data = processed_data - means

                        # Category 2: Peak Shape
                        elif cat_idx == 1:
                            if use_snv_before:
                                analyzer_temp = PreprocessingEffectsAnalyzer(processed_data)
                                processed_data = analyzer_temp.snv_transform()
                                analyzer = PreprocessingEffectsAnalyzer(processed_data)

                            if use_1st_sg:
                                processed_data = analyzer.first_derivative_savitzky_golay(
                                    window=sg_window_1st, polyorder=sg_poly_1st
                                )
                            elif use_2nd_sg:
                                processed_data = analyzer.second_derivative_savitzky_golay(
                                    window=sg_window_2nd, polyorder=sg_poly_2nd
                                )

                        # Category 3: Noise & Spikes
                        else:
                            if use_outlier:
                                # Simple outlier removal (clip values > threshold*std from mean)
                                for idx in processed_data.index:
                                    row = processed_data.loc[idx].values
                                    row_mean = row.mean()
                                    row_std = row.std()
                                    threshold = sigma_thresh * row_std
                                    processed_data.loc[idx] = np.clip(row, row_mean - threshold, row_mean + threshold)

                            if use_sg_smooth:
                                from scipy.signal import savgol_filter
                                processed_data.iloc[:, :] = savgol_filter(
                                    processed_data.values,
                                    window_length=sg_window_smooth,
                                    polyorder=sg_poly_smooth,
                                    deriv=0,
                                    axis=1,
                                    mode='nearest'
                                )

                            if use_moving_avg:
                                # Simple moving average
                                from scipy.ndimage import uniform_filter1d
                                processed_data.iloc[:, :] = uniform_filter1d(
                                    processed_data.values, size=ma_window, axis=1, mode='nearest'
                                )

                            if use_snv_after:
                                analyzer_temp = PreprocessingEffectsAnalyzer(processed_data)
                                processed_data = analyzer_temp.snv_transform()

                        # Visualization: Side-by-side comparison
                        from plotly.subplots import make_subplots

                        fig_compare = make_subplots(
                            rows=1, cols=2,
                            subplot_titles=(
                                f"Original ({test_category})",
                                f"After: {', '.join(selected_methods)}"
                            ),
                            horizontal_spacing=0.12
                        )

                        # Color gradient based on category
                        if cat_idx == 0:
                            colors = ['#c6dbef', '#9ecae1', '#6baed6', '#3182bd', '#08519c']
                        elif cat_idx == 1:
                            colors = ['#fee6ce', '#fdbe85', '#fd8d3c', '#e6550d', '#a63603']
                        else:
                            colors = ['#c7e9c0', '#a1d99b', '#74c476', '#31a354', '#006d2c']

                        # Original data
                        for i in range(5):
                            fig_compare.add_trace(
                                go.Scatter(
                                    x=wavenumbers,
                                    y=cat_data.iloc[i].values,
                                    mode='lines',
                                    name=f'Sample {sample_start+i}',
                                    line=dict(color=colors[i], width=1.5),
                                    showlegend=True,
                                    legendgroup='samples'
                                ),
                                row=1, col=1
                            )

                        # Processed data (handle potential length differences for derivatives)
                        wn_processed = wavenumbers
                        if len(processed_data.columns) < len(wavenumbers):
                            # Derivative reduced length
                            wn_processed = wavenumbers[:len(processed_data.columns)]

                        for i in range(5):
                            fig_compare.add_trace(
                                go.Scatter(
                                    x=wn_processed,
                                    y=processed_data.iloc[i].values,
                                    mode='lines',
                                    name=f'Sample {sample_start+i}',
                                    line=dict(color=colors[i], width=1.5),
                                    showlegend=False,
                                    legendgroup='samples'
                                ),
                                row=1, col=2
                            )

                        fig_compare.update_xaxes(title_text="Wavenumber (cm⁻¹)", row=1, col=1)
                        fig_compare.update_xaxes(title_text="Wavenumber (cm⁻¹)", row=1, col=2)
                        fig_compare.update_yaxes(title_text="Intensity", row=1, col=1)
                        fig_compare.update_yaxes(title_text="Processed Intensity", row=1, col=2)

                        fig_compare.update_layout(
                            height=450,
                            hovermode='x unified',
                            legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5)
                        )

                        st.plotly_chart(fig_compare, use_container_width=True)

                        # ===================================================================
                        # 4. STATISTICS COMPARISON
                        # ===================================================================
                        st.markdown("### 📊 Statistical Comparison")

                        # Calculate metrics
                        orig_peak_std = cat_data.max(axis=1).std()
                        proc_peak_std = processed_data.max(axis=1).std()

                        # SNR calculation (signal = mean of max values, noise = std of last 50 points)
                        orig_signal = cat_data.max(axis=1).mean()
                        orig_noise = cat_data.iloc[:, -50:].std(axis=1).mean()
                        orig_snr = orig_signal / orig_noise if orig_noise > 0 else np.inf

                        proc_signal = processed_data.max(axis=1).mean()
                        proc_noise = processed_data.iloc[:, -50:].std(axis=1).mean()
                        proc_snr = proc_signal / proc_noise if proc_noise > 0 else np.inf

                        # Peak height range
                        orig_range = cat_data.max(axis=1).max() - cat_data.max(axis=1).min()
                        proc_range = processed_data.max(axis=1).max() - processed_data.max(axis=1).min()

                        # Create comparison table
                        stats_data = {
                            'Metric': [
                                'Peak Height Std Dev',
                                'Signal-to-Noise Ratio',
                                'Peak Height Range',
                                'Mean Intensity'
                            ],
                            'Before': [
                                f"{orig_peak_std:.4f}",
                                f"{orig_snr:.2f}",
                                f"{orig_range:.4f}",
                                f"{cat_data.mean().mean():.4f}"
                            ],
                            'After': [
                                f"{proc_peak_std:.4f}",
                                f"{proc_snr:.2f}",
                                f"{proc_range:.4f}",
                                f"{processed_data.mean().mean():.4f}"
                            ],
                            'Improvement': [
                                f"{((orig_peak_std - proc_peak_std) / orig_peak_std * 100):.1f}%",
                                f"{((proc_snr - orig_snr) / orig_snr * 100):.1f}%",
                                f"{((orig_range - proc_range) / orig_range * 100):.1f}%",
                                "N/A"
                            ]
                        }

                        stats_df = pd.DataFrame(stats_data)
                        st.dataframe(stats_df, use_container_width=True, hide_index=True)

                        # ===================================================================
                        # 5. SUCCESS INDICATORS
                        # ===================================================================
                        st.markdown("### ✅ Success Criteria")

                        if cat_idx == 0:  # Peak Height
                            peak_alignment = (proc_peak_std / orig_peak_std) < 0.15  # < 15% of original std
                            shape_preserved = abs(processed_data.mean().mean()) < 0.5  # Near zero after SNV

                            if peak_alignment:
                                st.success("✅ All peaks align to similar height (std dev reduced by >85%)")
                            else:
                                st.warning("⚠️ Peak alignment incomplete - consider stronger normalization")

                            if shape_preserved:
                                st.success("✅ Spectral shape preserved (mean near zero)")
                            else:
                                st.info("ℹ️ Shape slightly altered - check if additional preprocessing needed")

                        elif cat_idx == 1:  # Peak Shape
                            if use_1st_sg or use_2nd_sg:
                                zero_crossings_visible = True  # Placeholder - would need peak detection
                                st.success("✅ Derivative transformation applied - check for consistent zero-crossings")
                                st.info("💡 **Interpretation:** Look for zero-crossings (1st derivative) or extrema (2nd derivative) at consistent wavenumber positions across samples")
                            else:
                                st.warning("⚠️ No derivative applied - shape differences may not be resolved")

                        else:  # Noise & Spikes
                            snr_improved = proc_snr > orig_snr * 1.5  # At least 50% improvement
                            background_low = proc_noise < 0.1

                            if snr_improved:
                                st.success(f"✅ SNR improved from {orig_snr:.1f} to {proc_snr:.1f} (+{((proc_snr/orig_snr - 1)*100):.0f}%)")
                            else:
                                st.warning("⚠️ SNR improvement < 50% - consider more aggressive smoothing")

                            if background_low:
                                st.success("✅ Background noise < 0.1 (clean signal)")
                            else:
                                st.info(f"ℹ️ Background noise = {proc_noise:.3f} - acceptable but could be improved")

                    except Exception as e:
                        st.error(f"Error applying preprocessing: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())

                else:
                    st.info("👆 Select at least one preprocessing method above to see the comparison")

        # Section 2: Preprocessing Method
        with st.expander("🔧 **2. Preprocessing Method**", expanded=False):
            if PREPROCESSING_THEORY_AVAILABLE:
                st.subheader("Select Preprocessing Method")

                # Check if dataset is available
                if 'preproc_data' not in st.session_state:
                    st.warning("⚠️ Please select a dataset first in Section 1")
                else:
                    # Radio button for preprocessing method selection
                    preproc_method = st.radio(
                        "Choose preprocessing method:",
                        [
                            "Original",
                            "SNV Transform",
                            "1st Derivative Simple",
                            "1st Derivative + SG Smoothing",
                            "2nd Derivative Simple",
                            "2nd Derivative + SG Smoothing"
                        ],
                        horizontal=False,
                        key="preproc_method_v3",
                        help="Select the preprocessing transformation to apply"
                    )

                    # Show SG parameter sliders if SG smoothing is selected
                    sg_window = 11
                    sg_polyorder = 3

                    if "SG Smoothing" in preproc_method:
                        st.markdown("#### Savitzky-Golay Parameters")

                        col_sg1, col_sg2 = st.columns(2)

                        with col_sg1:
                            sg_window = st.slider(
                                "Window Length (must be odd)",
                                min_value=5,
                                max_value=21,
                                value=11,
                                step=2,
                                key="sg_window_v3",
                                help="Size of the smoothing window. Larger = more smoothing."
                            )

                        with col_sg2:
                            sg_polyorder = st.slider(
                                "Polynomial Order",
                                min_value=2,
                                max_value=4,
                                value=3,
                                step=1,
                                key="sg_polyorder_v3",
                                help="Order of polynomial fit. Usually 2 or 3."
                            )

                        # Educational section for understanding SG parameters
                        st.markdown("---")
                        st.markdown("### 📚 Understanding Savitzky-Golay Parameters")

                        # Use a checkbox to show/hide the educational content
                        show_sg_guide = st.checkbox(
                            "Show detailed parameter guide",
                            value=False,
                            key="show_sg_parameter_guide",
                            help="Click to see detailed explanations of window length and polynomial order"
                        )

                        if show_sg_guide:
                            st.markdown("#### Window Length Effects")
                            st.markdown("""
**Window length** controls the **smoothing strength**:

- **Larger window** (15-21):
  - ✅ More aggressive smoothing
  - ✅ Better noise reduction
  - ❌ May lose fine spectral details
  - ❌ Broader peaks
  - **Best for**: Noisy data where broad features are important

- **Smaller window** (5-9):
  - ✅ Preserves sharp peaks and fine details
  - ✅ Better spectral resolution
  - ❌ Keeps more noise
  - ❌ Less effective smoothing
  - **Best for**: Clean data with important narrow peaks
                            """)

                            st.markdown("---")
                            st.markdown("#### Polynomial Order Effects")
                            st.markdown("""
**Polynomial order** is the degree of the fitting polynomial:

- **Order 2** (Quadratic):
  - More smoothing, simpler curve fitting
  - Good for broad, smooth features
  - May oversmooth sharp peaks

- **Order 3** (Cubic):
  - **Recommended default** - balanced approach
  - Preserves spectral features better
  - Good compromise between smoothing and detail

- **Order 4** (Quartic):
  - Minimal smoothing, maximum feature preservation
  - Best for complex peak shapes
  - May preserve more noise
                            """)

                            st.markdown("---")
                            st.markdown("#### Your Current Settings")

                            # Determine smoothing interpretation based on parameters
                            if sg_window >= 17:
                                if sg_polyorder <= 2:
                                    smoothing_type = "🟦 **Aggressive smoothing** - Good for very noisy data"
                                else:
                                    smoothing_type = "🟦 **Strong smoothing** - Good for noisy data with broad peaks"
                            elif sg_window >= 13:
                                if sg_polyorder <= 2:
                                    smoothing_type = "🟨 **Moderate-strong smoothing** - Good for moderately noisy data"
                                else:
                                    smoothing_type = "🟨 **Balanced smoothing** - Good general-purpose setting"
                            elif sg_window >= 9:
                                if sg_polyorder <= 2:
                                    smoothing_type = "🟩 **Moderate smoothing** - Preserves most features"
                                else:
                                    smoothing_type = "🟩 **Conservative smoothing** - Good for detailed peaks"
                            else:  # window < 9
                                if sg_polyorder <= 2:
                                    smoothing_type = "🟧 **Minimal smoothing** - Preserves fine details, keeps some noise"
                                else:
                                    smoothing_type = "🟧 **Very light smoothing** - Maximum detail preservation"

                            st.markdown(f"""
**Current settings:** `window_length = {sg_window}`, `polyorder = {sg_polyorder}`

{smoothing_type}
                            """)

                            st.markdown("---")
                            st.markdown("#### Mathematical Formula")
                            st.markdown(r"""
The Savitzky-Golay filter computes a smoothed value using:

$$
y_{\text{smoothed}}[i] = \sum_{j=-m}^{m} c_j \cdot y[i+j]
$$

Where:
- $y_{\text{smoothed}}[i]$ is the smoothed value at position $i$
- $c_j$ are **Savitzky-Golay coefficients** computed from the window length and polynomial order
- $m = \text{(window_length - 1)} / 2$ is the half-window size
- The sum spans $2m + 1$ points (the full window)

**Key insight:** The coefficients $c_j$ are derived by fitting a polynomial to the data points in the window, then evaluating the derivative at the center point. This is more sophisticated than simple finite differences.
                            """)

                            st.markdown("---")
                            st.markdown("#### Best Practices for Derivatives")
                            st.success("""
**⭐ Recommended settings for derivatives:**

- **1st Derivative**: Use `window = 11-15`, `order = 3-4`
  - Balances smoothing with feature preservation
  - Reduces derivative noise amplification

- **2nd Derivative**: Use `window = 15-21`, `order = 3-4`
  - Larger window essential for 2nd derivatives
  - 2nd derivatives amplify noise significantly
  - More aggressive smoothing needed

**Why?** Derivatives amplify high-frequency noise. Larger windows and higher polynomial orders help suppress this noise while preserving true spectral features.
                            """)

                        # Display Savitzky-Golay formula explanation
                        st.markdown("---")
                        st.markdown("**Savitzky-Golay Smoothing Formula:**")
                        st.markdown(r"""
The Savitzky-Golay filter applies a convolution to smooth data:

$$
y_i = \sum_{j=-m}^{m} c_j \cdot y_{i+j}
$$

Where:
- $y_i$ is the smoothed value at position $i$
- $c_j$ are convolution coefficients derived from polynomial fitting
- $m = \text{(window_length - 1)} / 2$ is the half-window size

**Key advantage:** The window smooths derivative noise while preserving spectral features better than simple moving averages.
                        """)
                        st.markdown("---")

                    # Apply selected preprocessing
                    try:
                        original_data = st.session_state.preproc_data
                        analyzer = PreprocessingEffectsAnalyzer(original_data)

                        if preproc_method == "Original":
                            processed_data = original_data.copy()
                            description = "No preprocessing applied - original data"
                            st.info("ℹ️ Using original data (no preprocessing applied)")

                        elif preproc_method == "SNV Transform":
                            processed_data, description = analyzer.apply_preprocessing('snv')
                            st.success(f"✅ Applied: **{description}**")

                        elif preproc_method == "1st Derivative Simple":
                            processed_data, description = analyzer.apply_preprocessing('first_derivative')
                            st.success(f"✅ Applied: **{description}**")

                        elif preproc_method == "1st Derivative + SG Smoothing":
                            # Use Savitzky-Golay with custom parameters
                            processed_data = analyzer.first_derivative_savitzky_golay(
                                window=sg_window,
                                polyorder=sg_polyorder
                            )
                            description = f"First Derivative (Savitzky-Golay, window={sg_window}, polyorder={sg_polyorder})"
                            st.success(f"✅ Applied: **{description}**")

                        elif preproc_method == "2nd Derivative Simple":
                            processed_data, description = analyzer.apply_preprocessing('second_derivative')
                            st.success(f"✅ Applied: **{description}**")

                        elif preproc_method == "2nd Derivative + SG Smoothing":
                            # Use Savitzky-Golay with custom parameters
                            processed_data = analyzer.second_derivative_savitzky_golay(
                                window=sg_window,
                                polyorder=sg_polyorder
                            )
                            description = f"Second Derivative (Savitzky-Golay, window={sg_window}, polyorder={sg_polyorder})"
                            st.success(f"✅ Applied: **{description}**")

                        # Store processed data in session state
                        st.session_state.preproc_processed = processed_data
                        st.session_state.preproc_method_name = preproc_method
                        st.session_state.preproc_description = description

                        # Immediately display comparison plot after preprocessing is applied
                        if preproc_method != "Original":
                            try:
                                from plotly.subplots import make_subplots

                                # Get wavenumbers from session state
                                wavenumbers = st.session_state.preproc_wavenumbers

                                # Use first 5 samples for comparison
                                original_subset = original_data.iloc[:5]
                                processed_subset = processed_data.iloc[:5]

                                # Create side-by-side comparison plot
                                fig_comparison = make_subplots(
                                    rows=1, cols=2,
                                    subplot_titles=("Original Spectra", "Preprocessed Spectra"),
                                    horizontal_spacing=0.12
                                )

                                # Get unified colors
                                colors_samples = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

                                # Plot original data (left panel)
                                for i in range(min(5, len(original_subset))):
                                    fig_comparison.add_trace(
                                        go.Scatter(
                                            x=wavenumbers,
                                            y=original_subset.iloc[i].values,
                                            mode='lines',
                                            name=f'Sample {i+1}',
                                            line=dict(color=colors_samples[i], width=1.5),
                                            showlegend=True,
                                            legendgroup=f'sample{i+1}'
                                        ),
                                        row=1, col=1
                                    )

                                # Plot preprocessed data (right panel)
                                for i in range(min(5, len(processed_subset))):
                                    fig_comparison.add_trace(
                                        go.Scatter(
                                            x=wavenumbers,
                                            y=processed_subset.iloc[i].values,
                                            mode='lines',
                                            name=f'Sample {i+1}',
                                            line=dict(color=colors_samples[i], width=1.5),
                                            showlegend=False,
                                            legendgroup=f'sample{i+1}'
                                        ),
                                        row=1, col=2
                                    )

                                # Update layout
                                fig_comparison.update_xaxes(title_text="Wavenumber (cm⁻¹)", row=1, col=1)
                                fig_comparison.update_xaxes(title_text="Wavenumber (cm⁻¹)", row=1, col=2)
                                fig_comparison.update_yaxes(title_text="Intensity", row=1, col=1)
                                fig_comparison.update_yaxes(title_text="Intensity", row=1, col=2)

                                fig_comparison.update_layout(
                                    height=450,
                                    title=dict(text="Original vs Preprocessed Comparison", x=0.5, xanchor='center'),
                                    hovermode='closest',
                                    showlegend=True,
                                    legend=dict(
                                        orientation="h",
                                        yanchor="bottom",
                                        y=1.08,
                                        xanchor="center",
                                        x=0.5
                                    )
                                )

                                # Display the plot
                                st.plotly_chart(fig_comparison, use_container_width=True)

                            except Exception as e:
                                st.warning(f"Could not display comparison plot: {str(e)}")

                    except Exception as e:
                        st.error(f"Error applying preprocessing: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())

            else:
                st.info("💡 This section will provide controls to select and configure preprocessing methods (SNV, derivatives, baseline correction, etc.)")

        # Section 3: Interactive PCA Explorer
        with st.expander("📈 **3. Interactive PCA Explorer**", expanded=False):
            if PREPROCESSING_THEORY_AVAILABLE:
                st.subheader("📈 PCA Analysis: Explore Preprocessing Effects Interactively")

                # Check if dataset is available
                if 'preproc_data' not in st.session_state:
                    st.warning("⚠️ Please select a dataset first in Section 1")
                else:
                    st.info("""
                    **Interactively explore how different preprocessing methods affect PCA results**

                    Select a row preprocessing method and column scaling method to see the PCA score plot update in real-time.
                    """)

                    # Create two columns for radio buttons
                    col_radio1, col_radio2 = st.columns(2)

                    with col_radio1:
                        row_preprocessing = st.radio(
                            "🔧 Choose row preprocessing:",
                            options=['Original', 'SNV', '1st Der Simple', '1st Der SG', '2nd Der Simple', '2nd Der SG'],
                            index=0,
                            key="pca_row_preprocessing",
                            help="Row preprocessing transforms each spectrum individually"
                        )

                    with col_radio2:
                        col_preprocessing = st.radio(
                            "📊 Choose column preprocessing:",
                            options=['Centering', 'Autoscaling'],
                            index=0,
                            key="pca_col_preprocessing",
                            help="Column preprocessing standardizes variables across samples"
                        )

                    # Color-by selector and sample size control
                    col_control1, col_control2 = st.columns(2)

                    with col_control1:
                        # Check if combined_effects scenario with categories
                        color_options = ['Sample Index']
                        if (st.session_state.get('preproc_scenario') == 'combined_effects' and
                            'category_metadata' in st.session_state):
                            color_options.append('Category')

                        color_by_option = st.selectbox(
                            "🎨 Color points by:",
                            options=color_options,
                            index=0,
                            key="pca_color_by",
                            help="Choose how to color the score plot points"
                        )

                    with col_control2:
                        n_samples_pca = st.slider(
                            "Number of samples to analyze:",
                            min_value=5,
                            max_value=min(50, len(st.session_state.preproc_data)),
                            value=min(30, len(st.session_state.preproc_data)),
                            key="n_samples_pca_interactive",
                            help="More samples = more comprehensive analysis"
                        )

                    # Reactive computation - triggers on any selection change
                    try:
                        from pca_utils.pca_calculations import compute_pca
                        from pca_utils.pca_plots import plot_scores

                        # Get original data
                        original_data = st.session_state.preproc_data.iloc[:n_samples_pca].copy()

                        # Create analyzer for preprocessing
                        analyzer = PreprocessingEffectsAnalyzer(original_data)

                        # Apply row preprocessing based on selection
                        if row_preprocessing == 'Original':
                            preprocessed = original_data.copy()
                        elif row_preprocessing == 'SNV':
                            preprocessed, _ = analyzer.apply_preprocessing('snv')
                        elif row_preprocessing == '1st Der Simple':
                            preprocessed = analyzer.first_derivative()
                        elif row_preprocessing == '1st Der SG':
                            preprocessed = analyzer.first_derivative_savitzky_golay(window=11, polyorder=3)
                        elif row_preprocessing == '2nd Der Simple':
                            preprocessed = analyzer.second_derivative()
                        elif row_preprocessing == '2nd Der SG':
                            preprocessed = analyzer.second_derivative_savitzky_golay(window=15, polyorder=3)

                        # Calculate PCA with column preprocessing
                        if col_preprocessing == 'Centering':
                            pca_model = compute_pca(preprocessed, n_components=2, center=True, scale=False)
                        else:  # Autoscaling
                            pca_model = compute_pca(preprocessed, n_components=2, center=True, scale=True)

                        # Extract results
                        scores = pca_model['scores']
                        variance_explained_ratio = pca_model['explained_variance_ratio']
                        variance_explained = variance_explained_ratio[:2] * 100
                        eigenvalues = pca_model['eigenvalues'][:2]

                        # Calculate condition number (ratio of largest to smallest eigenvalue)
                        if len(eigenvalues) >= 2 and eigenvalues[1] > 0:
                            condition_number = eigenvalues[0] / eigenvalues[1]
                        else:
                            condition_number = np.inf

                        # Determine color mapping based on selection
                        if color_by_option == 'Category' and 'category_metadata' in st.session_state:
                            # Create category labels for the samples
                            category_labels_list = []
                            for i in range(n_samples_pca):
                                if i < 10:
                                    category_labels_list.append('Category_1_Peak_Height')
                                elif i < 20:
                                    category_labels_list.append('Category_2_Peak_Shape')
                                else:
                                    category_labels_list.append('Category_3_Noise_Spikes')

                            color_by_data = pd.Series(category_labels_list, index=scores.index, name='Category')
                            show_hulls = True
                        else:
                            # Use sample index for coloring
                            color_by_data = pd.Series(range(1, len(scores) + 1), index=scores.index, name='Sample Index')
                            show_hulls = False

                        # Use plot_scores from pca_plots with appropriate coloring
                        fig = plot_scores(
                            scores=scores,
                            pc_x='PC1',
                            pc_y='PC2',
                            explained_variance_ratio=variance_explained_ratio,
                            color_by=color_by_data,
                            text_labels=None,  # Will use default index labels
                            is_varimax=False,
                            show_labels=False,
                            show_convex_hull=show_hulls
                        )

                        # Update title to show current preprocessing combination
                        fig.update_layout(
                            title=f"PCA Score Plot: Row={row_preprocessing} + Column={col_preprocessing}<br><sub>Total Explained Variance: {sum(variance_explained):.1f}%</sub>",
                            height=600
                        )

                        # Display the plot
                        st.plotly_chart(fig, use_container_width=True)

                        # Display metrics table
                        st.markdown("---")
                        st.markdown("### 📊 PCA Metrics")

                        metrics_data = {
                            'Metric': [
                                'Variance PC1 (%)',
                                'Variance PC2 (%)',
                                'Total Variance (%)',
                                'Eigenvalue PC1',
                                'Eigenvalue PC2',
                                'Condition Number'
                            ],
                            'Value': [
                                f"{variance_explained[0]:.3f}",
                                f"{variance_explained[1]:.3f}",
                                f"{sum(variance_explained):.3f}",
                                f"{eigenvalues[0]:.3f}",
                                f"{eigenvalues[1]:.3f}",
                                f"{condition_number:.3f}" if condition_number != np.inf else "∞"
                            ]
                        }

                        metrics_df = pd.DataFrame(metrics_data)
                        st.dataframe(
                            metrics_df.style.set_properties(**{
                                'text-align': 'center'
                            }).set_table_styles([
                                {'selector': 'th', 'props': [('text-align', 'center'), ('font-weight', 'bold')]}
                            ]),
                            use_container_width=True,
                            hide_index=True
                        )

                        # Add interpretation guidance
                        st.markdown("---")
                        st.markdown("### 💡 Quick Guide")

                        col_guide1, col_guide2 = st.columns(2)

                        with col_guide1:
                            st.info("""
**Metrics Interpretation:**

- **Variance %**: Higher values indicate better data representation
- **Eigenvalues**: Magnitude of variance along each PC
- **Condition Number**: Lower is better (< 10 excellent, > 100 poor)
- **Total Variance**: Aim for > 70% with first 2 PCs
                            """)

                        with col_guide2:
                            st.success("""
**Preprocessing Tips:**

- **Original**: Baseline comparison, no transformation
- **SNV**: Removes scatter effects, normalizes intensity
- **1st Derivative**: Enhances slopes, removes baseline
- **2nd Derivative**: Enhances peaks, removes slope + baseline
- **Centering**: Variables have similar scales
- **Autoscaling**: Variables have different scales/units
                            """)

                    except Exception as e:
                        st.error(f"Error computing PCA: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())

            else:
                st.info("💡 This section will display an interactive PCA explorer showing how different preprocessing methods affect data structure")

        # Section 4: Statistics
        with st.expander("📉 **4. Statistics**", expanded=False):
            if PREPROCESSING_THEORY_AVAILABLE:
                st.subheader("📉 Statistical Analysis")

                # Check if preprocessing has been applied
                if 'preproc_processed' not in st.session_state:
                    st.warning("⚠️ Please apply a preprocessing method first in Section 2")
                else:
                    st.info("""
                    💡 **Basic Statistics Available**

                    Simple statistical comparison between original and preprocessed data:
                    """)

                    # Get data from session state
                    original_data = st.session_state.preproc_data
                    processed_data = st.session_state.preproc_processed

                    # Simple statistics comparison
                    col_stat1, col_stat2 = st.columns(2)

                    with col_stat1:
                        st.markdown("**Original Data:**")
                        st.write(f"Mean: {original_data.mean().mean():.4f}")
                        st.write(f"Std Dev: {original_data.std().mean():.4f}")
                        st.write(f"Variance: {original_data.var().mean():.4f}")

                    with col_stat2:
                        st.markdown("**Preprocessed Data:**")
                        st.write(f"Mean: {processed_data.mean().mean():.4f}")
                        st.write(f"Std Dev: {processed_data.std().mean():.4f}")
                        st.write(f"Variance: {processed_data.var().mean():.4f}")

                    st.markdown("---")
                    st.info("""
                    🚧 **Advanced statistics coming soon!**

                    Future features will include:
                    - SNR improvement calculation
                    - Baseline correction effectiveness
                    - Feature enhancement metrics
                    - Quality score assessment
                    - Downloadable statistical reports
                    """)

                    # Placeholder for future statistics
                    if 'preproc_stats' in st.session_state:
                        stats = st.session_state.preproc_stats

                        # Display key metrics in columns
                        st.markdown("### Key Metrics")

                        col_met1, col_met2, col_met3, col_met4 = st.columns(4)

                        with col_met1:
                            variance_reduction = ((stats.get('original_variance', 1) - stats.get('preprocessed_variance', 1))
                                                 / stats.get('original_variance', 1) * 100)
                            st.metric(
                                "Variance Reduction",
                                f"{variance_reduction:.1f}%",
                                help="Percentage reduction in total variance"
                            )

                        with col_met2:
                            snr_improvement = stats.get('snr_improvement', 0)
                            st.metric(
                                "SNR Improvement",
                                f"{snr_improvement:.2f}x",
                                help="Signal-to-Noise ratio improvement factor"
                            )

                        with col_met3:
                            baseline_correction = stats.get('baseline_correction_score', 0) * 100
                            st.metric(
                                "Baseline Correction",
                                f"{baseline_correction:.1f}%",
                                help="Effectiveness of baseline correction (0-100%)"
                            )

                        with col_met4:
                            feature_enhancement = stats.get('feature_enhancement', 0) * 100
                            st.metric(
                                "Feature Enhancement",
                                f"{feature_enhancement:.1f}%",
                                help="Enhancement of spectral features (0-100%)"
                            )

                        # Detailed statistics table
                        st.markdown("---")
                        st.markdown("### Detailed Statistics")

                        col_table1, col_table2 = st.columns(2)

                        with col_table1:
                            st.markdown("**Original Data:**")
                            orig_stats_df = pd.DataFrame({
                                'Metric': [
                                    'Mean Signal',
                                    'Total Variance',
                                    'Baseline Offset',
                                    'Noise Level'
                                ],
                                'Value': [
                                    f"{stats.get('original_mean', 0):.4f}",
                                    f"{stats.get('original_variance', 0):.4f}",
                                    f"{stats.get('original_baseline', 0):.4f}",
                                    f"{stats.get('original_noise', 0):.4f}"
                                ]
                            })
                            st.dataframe(orig_stats_df, use_container_width=True, hide_index=True)

                        with col_table2:
                            st.markdown("**Preprocessed Data:**")
                            prep_stats_df = pd.DataFrame({
                                'Metric': [
                                    'Mean Signal',
                                    'Total Variance',
                                    'Baseline Offset',
                                    'Noise Level'
                                ],
                                'Value': [
                                    f"{stats.get('preprocessed_mean', 0):.4f}",
                                    f"{stats.get('preprocessed_variance', 0):.4f}",
                                    f"{stats.get('preprocessed_baseline', 0):.4f}",
                                    f"{stats.get('preprocessed_noise', 0):.4f}"
                                ]
                            })
                            st.dataframe(prep_stats_df, use_container_width=True, hide_index=True)

                        # Quality assessment
                        st.markdown("---")
                        st.markdown("### Quality Assessment")

                        overall_quality = stats.get('overall_quality_score', 0)

                        if overall_quality >= 0.8:
                            quality_color = "🟢"
                            quality_text = "Excellent - Preprocessing highly effective"
                        elif overall_quality >= 0.6:
                            quality_color = "🟡"
                            quality_text = "Good - Preprocessing moderately effective"
                        elif overall_quality >= 0.4:
                            quality_color = "🟠"
                            quality_text = "Fair - Preprocessing partially effective"
                        else:
                            quality_color = "🔴"
                            quality_text = "Poor - Consider different preprocessing methods"

                        st.info(f"{quality_color} **Overall Quality Score: {overall_quality*100:.1f}%**\n\n{quality_text}")

                        # Recommendations
                        if 'recommendations' in stats and stats['recommendations']:
                            st.markdown("---")
                            st.markdown("### Recommendations")
                            for rec in stats['recommendations']:
                                st.markdown(f"- {rec}")

                        # Export statistics
                        st.markdown("---")
                        if st.button("Download Statistics as CSV", key="download_stats"):
                            # Prepare statistics for download
                            stats_df = pd.DataFrame({
                                'Category': ['Original', 'Original', 'Original', 'Original',
                                           'Preprocessed', 'Preprocessed', 'Preprocessed', 'Preprocessed',
                                           'Quality', 'Quality', 'Quality', 'Quality'],
                                'Metric': ['Mean Signal', 'Total Variance', 'Baseline Offset', 'Noise Level',
                                          'Mean Signal', 'Total Variance', 'Baseline Offset', 'Noise Level',
                                          'Variance Reduction %', 'SNR Improvement', 'Baseline Correction %', 'Overall Quality %'],
                                'Value': [
                                    stats.get('original_mean', 0),
                                    stats.get('original_variance', 0),
                                    stats.get('original_baseline', 0),
                                    stats.get('original_noise', 0),
                                    stats.get('preprocessed_mean', 0),
                                    stats.get('preprocessed_variance', 0),
                                    stats.get('preprocessed_baseline', 0),
                                    stats.get('preprocessed_noise', 0),
                                    variance_reduction,
                                    stats.get('snr_improvement', 0),
                                    baseline_correction,
                                    overall_quality * 100
                                ]
                            })

                            csv_data = stats_df.to_csv(index=False)
                            st.download_button(
                                label="Download CSV",
                                data=csv_data,
                                file_name=f"preprocessing_statistics_{st.session_state.preproc_scenario}.csv",
                                mime="text/csv"
                            )
            else:
                st.info("💡 This section will show quantitative metrics (variance, SNR, baseline correction effectiveness, etc.)")

        # Section 5: Educational Content
        with st.expander("🎓 **5. Educational Content**", expanded=False):
            if PREPROCESSING_THEORY_AVAILABLE:
                st.subheader("🎓 Step 5: Learn Preprocessing Theory")

                # Topic selector
                edu_topic = st.selectbox(
                    "Select topic:",
                    [
                        "Overview",
                        "Baseline Correction",
                        "Scatter Correction",
                        "Derivatives",
                        "Method Selection Guidelines",
                        "Common Pitfalls"
                    ],
                    key="edu_topic",
                    help="Choose a topic to learn more about"
                )

                st.markdown("---")

                if edu_topic == "Overview":
                    st.markdown("### Why Preprocessing?")
                    st.markdown("""
                    Preprocessing spectral data is essential for removing **systematic variations** that are not related to
                    the chemical information of interest. These variations can arise from:

                    - **Instrument effects**: Baseline drift, detector noise, light source fluctuations
                    - **Physical effects**: Scattering, path length variations, sample presentation
                    - **Environmental effects**: Temperature, humidity, pressure changes

                    **Goal**: Enhance **chemical signal** while removing **systematic noise**
                    """)

                    st.markdown("### Common Preprocessing Categories")

                    col_cat1, col_cat2, col_cat3 = st.columns(3)

                    with col_cat1:
                        st.info("""
                        **Baseline Correction**

                        Removes additive offsets
                        - Linear detrend
                        - Polynomial baseline
                        - Asymmetric least squares
                        """)

                    with col_cat2:
                        st.info("""
                        **Scatter Correction**

                        Removes multiplicative effects
                        - SNV (Standard Normal Variate)
                        - MSC (Multiplicative Scatter Correction)
                        - EMSC (Extended MSC)
                        """)

                    with col_cat3:
                        st.info("""
                        **Derivatives**

                        Enhances spectral features
                        - 1st derivative (slope)
                        - 2nd derivative (curvature)
                        - Savitzky-Golay smoothing
                        """)

                elif edu_topic == "Baseline Correction":
                    st.markdown("### Baseline Correction Methods")

                    st.markdown("""
                    **Baseline variations** appear as additive offsets in spectra, caused by:
                    - Instrument drift
                    - Background fluorescence
                    - Incomplete blank subtraction
                    """)

                    st.markdown("#### Linear Detrend")
                    st.markdown("""
                    - Fits a linear function to the spectrum
                    - Subtracts the linear trend
                    - **Best for**: Gradual, monotonic baseline drift
                    - **Formula**: `y_corrected = y - (a*x + b)`
                    """)

                    st.markdown("#### Polynomial Baseline")
                    st.markdown("""
                    - Fits a polynomial (order 2-5) to the spectrum
                    - More flexible than linear detrend
                    - **Best for**: Non-linear baseline curvature
                    - **Caution**: High orders can remove real spectral features
                    """)

                    st.success("**Recommendation**: Start with linear detrend. Use polynomial only if clear non-linear baseline is visible.")

                elif edu_topic == "Scatter Correction":
                    st.markdown("### Scatter Correction Methods")

                    st.markdown("""
                    **Scatter effects** appear as multiplicative variations, caused by:
                    - Particle size differences
                    - Packing density variations
                    - Path length differences
                    """)

                    st.markdown("#### SNV (Standard Normal Variate)")
                    st.markdown("""
                    - Centers and scales each spectrum to mean=0, std=1
                    - Simple and effective
                    - **Best for**: Removing intensity variations between samples
                    - **Formula**: `SNV = (spectrum - mean) / std`
                    - **Advantage**: No reference spectrum needed
                    """)

                    st.markdown("#### MSC (Multiplicative Scatter Correction)")
                    st.markdown("""
                    - Uses a reference spectrum (usually mean spectrum)
                    - Corrects each spectrum relative to reference
                    - **Best for**: Removing baseline offset + scatter simultaneously
                    - **Advantage**: Physical interpretation (slope = scatter, intercept = baseline)
                    - **Limitation**: Requires representative reference spectrum
                    """)

                    st.success("**Recommendation**: SNV for quick analysis. MSC when you have good reference spectrum and want physical interpretation.")

                elif edu_topic == "Derivatives":
                    st.markdown("### Derivative Methods")

                    st.markdown("""
                    **Derivatives** emphasize changes in spectral features:
                    - Remove additive baselines automatically
                    - Enhance overlapping peaks
                    - Amplify noise (requires smoothing)
                    """)

                    st.markdown("#### First Derivative")
                    st.markdown("""
                    - Measures slope/rate of change
                    - Zero-crossing at peak maximum
                    - **Best for**: Separating overlapping peaks
                    - **Effect**: Sharpens bands, removes constant baseline
                    """)

                    st.markdown("#### Second Derivative")
                    st.markdown("""
                    - Measures curvature
                    - Negative peak at original peak maximum
                    - **Best for**: Resolving closely overlapping peaks
                    - **Caution**: Amplifies noise significantly
                    """)

                    st.markdown("#### Savitzky-Golay")
                    st.markdown("""
                    - Combines smoothing + derivatives
                    - Polynomial fitting in moving window
                    - **Parameters**:
                      - Window length: Larger = more smoothing (must be odd)
                      - Polynomial order: Usually 2-3
                      - Derivative order: 0 (smoothing), 1, or 2
                    - **Advantage**: Less noise amplification than simple derivatives
                    """)

                    st.warning("**Important**: Always apply smoothing (e.g., Savitzky-Golay) when using derivatives to avoid excessive noise amplification.")

                elif edu_topic == "Method Selection Guidelines":
                    st.markdown("### How to Choose Preprocessing Methods")

                    st.markdown("#### Decision Tree")
                    st.markdown("""
                    1. **Identify the problem**:
                       - Baseline variations? → Baseline correction
                       - Intensity variations? → Scatter correction
                       - Overlapping peaks? → Derivatives

                    2. **Check data characteristics**:
                       - Noisy data? → Avoid 2nd derivatives without smoothing
                       - Linear baseline? → Linear detrend sufficient
                       - Non-linear baseline? → Polynomial or MSC

                    3. **Consider analysis method**:
                       - PCA/clustering: SNV often sufficient
                       - Quantitative modeling: May need MSC or derivatives
                       - Peak identification: Derivatives helpful
                    """)

                    st.markdown("#### Common Preprocessing Pipelines")

                    st.info("""
                    **NIR Spectroscopy (diffuse reflectance)**:
                    1. SNV or MSC (scatter correction)
                    2. Savitzky-Golay smoothing (optional)
                    3. 1st or 2nd derivative (if needed)
                    """)

                    st.info("""
                    **Raman Spectroscopy**:
                    1. Baseline correction (polynomial or asymmetric least squares)
                    2. Smoothing (Savitzky-Golay)
                    3. Normalization (peak area or internal standard)
                    """)

                    st.info("""
                    **FTIR Spectroscopy (transmission)**:
                    1. Baseline correction (if needed)
                    2. Normalization (peak height or area)
                    3. Derivatives (optional, for peak resolution)
                    """)

                    # ========================================================================
                    # NEW SECTION: Preprocessing by Dataset Category
                    # ========================================================================
                    st.markdown("---")
                    st.markdown("### 🎯 Preprocessing by Dataset Category")
                    st.markdown("When using the **Combined Effects** dataset, follow these category-specific guidelines:")

                    # Category 1: Peak Height Variation
                    st.markdown("#### 🔵 Category 1: Peak Height Variation")
                    st.info("""
**Problem:** Different samples have different peak intensities but same underlying chemistry

**Why it happens:**
- Different detector responses
- Different sample concentrations
- Different path lengths
- Variable sample thickness

**Recommended Workflow:**
1. **Intensity Normalization:**
   - **Option A:** SNV (Standard Normal Variate) - row-wise scaling
   - **Option B:** MSC (Multiplicative Scatter Correction)
   - **Option C:** Column autoscaling (normalize each wavelength)

2. **Expected Result:** All peaks align to same height, revealing chemical similarity

3. **Validation:** Overlay before/after plots - peaks should stack vertically

**Why Other Methods Fail:**
- ❌ **Derivatives:** Won't help - just shifts peaks down uniformly
- ❌ **Smoothing:** Unnecessary - no noise problem to solve
- ✅ **SNV/MSC:** Removes intensity bias without distorting spectral shape

**PCA Expectation:** After SNV, all Category 1 samples should cluster together in score plot
                    """)

                    # Category 2: Peak Shape Distortion
                    st.markdown("#### 🟠 Category 2: Peak Shape Distortion")
                    st.warning("""
**Problem:** Same analyte but different peak widths/shapes across samples

**Why it happens:**
- Instrumental misalignment or drift
- Sample preparation differences (particle size, moisture)
- Temperature/pressure effects during measurement
- Overlapping with background features
- Matrix effects in complex samples

**Recommended Workflow:**
1. **First Step:** Baseline correction (if baseline varies)

2. **Derivatives:**
   - **1st Derivative:** Shows slope changes, zero-crossing at peak maximum
   - **2nd Derivative:** More selective for resolving overlapping peaks

3. **Combine with Smoothing:** Use Savitzky-Golay during derivation
   - Window: 11-21 (odd number)
   - Polynomial order: 2-3

4. **Expected Result:** Peak maxima become sharp and distinctive, width differences emphasized

5. **Validation:** Look for consistent zero-crossings or extrema positions

**Why Other Methods Fail:**
- ❌ **Simple Normalization:** Doesn't resolve shape differences - all peaks still overlap
- ❌ **2nd Derivative Alone:** Too noise-sensitive without smoothing
- ✅ **1st Derivative + Savitzky-Golay:** Preserves chemical information while highlighting shape

**PCA Expectation:** After derivatives, Category 2 samples should separate along shape gradient
                    """)

                    # Category 3: Noise & Spike Artifacts
                    st.markdown("#### 🟢 Category 3: Noise & Spike Artifacts")
                    st.success("""
**Problem:** Underlying signal obscured by random noise and instrumental spikes

**Why it happens:**
- Detector thermal noise
- Cosmic ray events (especially in Raman spectroscopy)
- Environmental electromagnetic interference
- Electrical transients in detector electronics
- Shot noise from low photon counts

**Recommended Workflow:**
1. **Outlier Detection:** Remove spikes > 3σ from local mean (optional)

2. **Smoothing Methods:**
   - **Savitzky-Golay (BEST):** Preserves peak shape while smoothing
     - Window: 15-25
     - Polynomial: 2-3
   - **Moving Average:** Simple but distorts peak shapes
   - **Median Filter:** Excellent for spike removal, preserves edges

3. **Optional SNV:** After smoothing, for intensity normalization

4. **Expected Result:** Clean baseline with sharp, artifact-free peaks

5. **Validation:** Calculate SNR (Signal-to-Noise Ratio) improvement

**Why Other Methods Fail:**
- ❌ **Direct Derivatives:** Amplifies noise 5-10x - unusable results
- ❌ **Normalization Without Smoothing:** Biased by spike artifacts
- ✅ **Smoothing First, Then Analysis:** Maintains chemical information

**PCA Expectation:** After smoothing, Category 3 samples cluster tightly (noise removed)
                    """)

                    # Interactive Decision Support
                    st.markdown("---")
                    st.markdown("### 🔍 Interactive Diagnosis Tool")
                    st.markdown("Answer these questions to get preprocessing recommendations for your data:")

                    # Decision tree implementation
                    with st.expander("📋 Click to diagnose your data", expanded=False):
                        st.markdown("#### Step 1: Identify the Primary Issue")

                        primary_issue = st.radio(
                            "What is the main problem with your spectra?",
                            options=[
                                "Select an option...",
                                "Different peak intensities, but peaks have same shape",
                                "Same peak intensity, but peaks have different widths/shapes",
                                "Spectra are noisy with random spikes/artifacts",
                                "Combination of multiple issues"
                            ],
                            key="diagnosis_primary_issue"
                        )

                        if primary_issue != "Select an option...":
                            st.markdown("#### Step 2: Additional Characteristics")

                            has_baseline = st.checkbox(
                                "My spectra have baseline drift or offset",
                                key="diagnosis_baseline"
                            )

                            noise_level = st.select_slider(
                                "Noise level in your data:",
                                options=["Very Low", "Low", "Moderate", "High", "Very High"],
                                value="Moderate",
                                key="diagnosis_noise"
                            )

                            # Generate recommendations
                            st.markdown("---")
                            st.markdown("#### 🎯 Recommended Preprocessing Sequence:")

                            if primary_issue == "Different peak intensities, but peaks have same shape":
                                st.success("""
**Your data matches Category 1: Peak Height Variation**

**Recommended Sequence:**
1. **Baseline Correction** (if you checked baseline issue)
   - Method: Asymmetric Least Squares or polynomial baseline

2. **SNV (Standard Normal Variate)** ← PRIMARY METHOD
   - This normalizes each spectrum to zero mean and unit variance
   - Removes multiplicative and additive scatter effects

3. **Alternative:** MSC (Multiplicative Scatter Correction)
   - If you have a reference spectrum

4. **Column Centering** before PCA
   - Removes variable means

**Expected PCA Result:** All samples should cluster together since chemistry is identical
                                """)

                            elif primary_issue == "Same peak intensity, but peaks have different widths/shapes":
                                st.warning("""
**Your data matches Category 2: Peak Shape Distortion**

**Recommended Sequence:**
1. **Baseline Correction** (if you checked baseline issue)

2. **Choose Derivative Method:**
   - **1st Derivative (Savitzky-Golay):** For moderate shape differences
     - `window=11, polyorder=3`
   - **2nd Derivative (Savitzky-Golay):** For subtle overlapping peaks
     - `window=15, polyorder=3`

3. **Do NOT** normalize before derivatives (loses relative information)

4. **Optional:** Column centering after derivatives

**Expected PCA Result:** Samples separate along a gradient reflecting shape differences
                                """)

                            elif primary_issue == "Spectra are noisy with random spikes/artifacts":
                                noise_advice = ""
                                if noise_level in ["High", "Very High"]:
                                    noise_advice = "⚠️ **High noise detected** - use aggressive smoothing (window=21-25)"
                                else:
                                    noise_advice = "✓ Moderate noise - use standard smoothing (window=15-17)"

                                st.success(f"""
**Your data matches Category 3: Noise & Spike Artifacts**

{noise_advice}

**Recommended Sequence:**
1. **Spike Removal** (if visible spikes present)
   - Method: Median filter or iterative sigma-clipping

2. **Savitzky-Golay Smoothing** ← PRIMARY METHOD
   - Window: {'21-25 (high noise)' if noise_level in ['High', 'Very High'] else '15-17 (moderate)'}
   - Polyorder: 2-3
   - Derivative: 0 (smoothing only)

3. **Baseline Correction** (if you checked baseline issue)

4. **Optional SNV** for intensity normalization

5. **Column centering** before PCA

**Expected PCA Result:** Clean clustering without noise-driven scatter
                                """)

                            elif primary_issue == "Combination of multiple issues":
                                st.info("""
**Your data has multiple issues - use a combined approach:**

**Multi-Stage Preprocessing Pipeline:**

**Stage 1: Clean the data**
- Spike removal (if needed)
- Savitzky-Golay smoothing (window=15, polyorder=3)

**Stage 2: Correct systematic effects**
- Baseline correction (polynomial or asymmetric least squares)
- SNV or MSC for scatter correction

**Stage 3: Enhance features (if needed)**
- 1st or 2nd derivative (Savitzky-Golay)
- Use larger window (17-21) since data is already smoothed

**Stage 4: Final scaling**
- Column centering before PCA
- Optional: column autoscaling if variables have different units

**⚠️ Important:** Test each stage separately in PCA Explorer to see which steps help!

**Validation:** Compare PCA results after each stage - stop when clustering improves
                                """)

                            # Add PCA Explorer link
                            st.markdown("---")
                            st.info("💡 **Next Step:** Try these recommendations in the **PCA Explorer (Section 3)** to visualize the effects!")

                elif edu_topic == "Common Pitfalls":
                    st.markdown("### Common Pitfalls and How to Avoid Them")

                    st.error("""
                    **❌ Pitfall 1: Over-preprocessing**

                    - Applying too many preprocessing steps
                    - Can remove real chemical information
                    - **Solution**: Use minimal preprocessing. Always compare results with/without preprocessing.
                    """)

                    st.error("""
                    **❌ Pitfall 2: Wrong Preprocessing Order**

                    - Example: Derivatives before scatter correction
                    - Can amplify artifacts
                    - **Solution**: Generally use order: Baseline → Scatter → Derivatives → Normalization
                    """)

                    st.error("""
                    **❌ Pitfall 3: High-order Derivatives Without Smoothing**

                    - 2nd derivatives amplify noise dramatically
                    - **Solution**: Always use Savitzky-Golay or moving average before/during derivation
                    """)

                    st.error("""
                    **❌ Pitfall 4: Using Same Preprocessing for All Datasets**

                    - Different analytical techniques need different preprocessing
                    - Same technique, different sample types may need adjustment
                    - **Solution**: Inspect your data. Use interactive tools to test different methods.
                    """)

                    st.error("""
                    **❌ Pitfall 5: Not Validating Preprocessing Effect**

                    - Applying preprocessing blindly
                    - **Solution**: Always visualize before/after. Check statistical metrics (variance, SNR).
                    """)

                    st.success("""
                    **✅ Best Practice**:

                    1. Visualize original data
                    2. Test one preprocessing method at a time
                    3. Check statistics and plots
                    4. Validate on test set
                    5. Document preprocessing pipeline
                    """)

                # References section
                st.markdown("---")
                st.markdown("### Further Reading")
                st.markdown("""
                **Books**:
                - Bro, R., & Smilde, A. K. (2014). *Principal component analysis*. Analytical Methods, 6(9), 2812-2831.
                - Rinnan, Å., van den Berg, F., & Engelsen, S. B. (2009). *Review of the most common pre-processing techniques for near-infrared spectra*. TrAC Trends in Analytical Chemistry, 28(10), 1201-1222.

                **Online Resources**:
                - NIR spectroscopy tutorials: eigenvector.com
                - Chemometric theory: chemometry.com
                """)

            else:
                st.info("💡 This section will provide theoretical background, guidelines for method selection, and best practices for preprocessing spectral data")

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