"""
CAT Transformations Page
Complete suite for spectral/analytical data transformations
Equivalent to TR_* R scripts
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.signal import savgol_filter
from color_utils import get_unified_color_schemes, create_categorical_color_map

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
                   color_data=None, color_variable=None, dark_mode=True):
    """Create two line plots for comparison"""
    
    colors = get_unified_color_schemes(dark_mode)
    from plotly.subplots import make_subplots
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(title_original, title_transformed),
        vertical_spacing=0.15
    )
    
    if color_data is not None:
        unique_values = pd.Series(color_data).dropna().unique()
        color_discrete_map = create_categorical_color_map(unique_values, dark_mode)
        
        for group in unique_values:
            group_mask = pd.Series(color_data) == group
            group_color = color_discrete_map[group]
            group_indices = original_data.index[group_mask].tolist()
            first_idx = group_indices[0] if group_indices else None
            
            for idx in group_indices:
                is_first = bool(idx == first_idx)
                
                fig.add_trace(
                    go.Scatter(
                        x=list(range(len(original_data.columns))),
                        y=original_data.loc[idx].values,
                        mode='lines',
                        name=str(group),
                        line=dict(color=group_color, width=1),
                        showlegend=is_first,
                        legendgroup=str(group),
                        hovertemplate=f'Sample: {idx}<br>Value: %{{y:.3f}}<extra></extra>'
                    ),
                    row=1, col=1
                )
            
            transformed_indices = [idx for idx in group_indices if idx in transformed_data.index]
            
            for idx in transformed_indices:
                fig.add_trace(
                    go.Scatter(
                        x=list(range(len(transformed_data.columns))),
                        y=transformed_data.loc[idx].values,
                        mode='lines',
                        name=str(group),
                        line=dict(color=group_color, width=1),
                        showlegend=False,
                        legendgroup=str(group),
                        hovertemplate=f'Sample: {idx}<br>Value: %{{y:.3f}}<extra></extra>'
                    ),
                    row=2, col=1
                )
    else:
        for idx in original_data.index:
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(original_data.columns))),
                    y=original_data.loc[idx].values,
                    mode='lines',
                    name=str(idx),
                    line=dict(color=colors['point_color'], width=1),
                    showlegend=False,
                    hovertemplate=f'Sample: {idx}<br>Value: %{{y:.3f}}<extra></extra>'
                ),
                row=1, col=1
            )
            
            if idx in transformed_data.index:
                fig.add_trace(
                    go.Scatter(
                        x=list(range(len(transformed_data.columns))),
                        y=transformed_data.loc[idx].values,
                        mode='lines',
                        name=str(idx),
                        line=dict(color=colors['point_color'], width=1),
                        showlegend=False,
                        hovertemplate=f'Sample: {idx}<br>Value: %{{y:.3f}}<extra></extra>'
                    ),
                    row=2, col=1
                )
    
    fig.update_xaxes(title_text="Variable Index", row=1, col=1, gridcolor=colors['grid'])
    fig.update_xaxes(title_text="Variable Index", row=2, col=1, gridcolor=colors['grid'])
    fig.update_yaxes(title_text="Original Value", row=1, col=1, gridcolor=colors['grid'])
    fig.update_yaxes(title_text="Transformed Value", row=2, col=1, gridcolor=colors['grid'])
    
    fig.update_layout(
        height=800,
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['paper'],
        font=dict(color=colors['text']),
        hovermode='closest'
    )
    
    return fig

# ===========================================
# MAIN SHOW FUNCTION
# ===========================================

def show():
    """Display the Transformations page"""
    
    st.markdown("# Data Transformations")
    st.markdown("*Complete transformation suite for spectral and analytical data*")
    st.markdown("*Equivalent to TR_* R scripts*")
    
    if 'current_data' not in st.session_state:
        st.warning("No data loaded. Please go to Data Handling to load your dataset first.")
        return
    
    data = st.session_state.current_data
    
    # CRITICAL: Get original untransformed data for comparison
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
    
    tab1, tab2 = st.tabs([
        "Row Transformations",
        "Column Transformations"
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
        
        with col_vis2:
            use_dark_theme = st.checkbox("Dark mode colors", value=True, key="row_transform_dark")
        
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
                    
                    # CRITICAL: Always use original data for comparison
                    original_slice = original_data.iloc[:, col_range[0]:col_range[1]]
                    
                    fig = plot_comparison(
                        original_slice, 
                        transformed,
                        f"Original Data ({original_slice.shape[0]} × {original_slice.shape[1]})",
                        f"Transformed Data ({transformed.shape[0]} × {transformed.shape[1]}) - {selected_transform}",
                        color_data=color_data,
                        color_variable=color_variable,
                        dark_mode=use_dark_theme
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Statistics BEFORE saving
                    st.markdown("### Transformation Statistics")
                    
                    col_stat1, col_stat2, col_stat3 = st.columns(3)
                    
                    with col_stat1:
                        st.metric("Original Shape", f"{original_slice.shape[0]} × {original_slice.shape[1]}")
                    with col_stat2:
                        st.metric("Transformed Shape", f"{transformed.shape[0]} × {transformed.shape[1]}")
                    with col_stat3:
                        variance_ratio = transformed.var().mean() / original_slice.var().mean()
                        st.metric("Variance Ratio", f"{variance_ratio:.3f}")
                    
                    # Save section - SEPARATED WITH BUTTON
                    st.markdown("---")
                    st.markdown("### Save Transformation")
                    st.info("Review the transformation above, then save it to workspace if satisfied")
                    
                    if st.button("Save to Workspace", type="primary", key="save_row_transform"):
                        full_transformed = data.copy()
                        
                        if transformed.shape[1] != (col_range[1] - col_range[0]):
                            cols_before = list(range(col_range[0]))
                            cols_after = list(range(col_range[1], len(data.columns)))
                            
                            if cols_before:
                                before_data = data.iloc[:, cols_before]
                            else:
                                before_data = pd.DataFrame(index=data.index)
                            
                            if cols_after:
                                after_data = data.iloc[:, cols_after]
                            else:
                                after_data = pd.DataFrame(index=data.index)
                            
                            if transformed.shape[0] != data.shape[0]:
                                before_data = before_data.iloc[:transformed.shape[0], :]
                                after_data = after_data.iloc[:transformed.shape[0], :]
                            
                            full_transformed = pd.concat([before_data, transformed, after_data], axis=1)
                        else:
                            full_transformed.iloc[:, col_range[0]:col_range[1]] = transformed
                        
                        dataset_name = st.session_state.get('current_dataset', 'Dataset')
                        base_name = dataset_name.split('.')[0].replace('_ORIGINAL', '')
                        transformed_name = f"{base_name}.{transform_code}"
                        
                        st.session_state.current_data = full_transformed
                        st.session_state.current_dataset = transformed_name
                        
                        if 'transformation_history' not in st.session_state:
                            st.session_state.transformation_history = {}
                        
                        st.session_state.transformation_history[transformed_name] = {
                            'data': full_transformed,
                            'transform': selected_transform,
                            'params': params,
                            'col_range': col_range,
                            'timestamp': pd.Timestamp.now()
                        }
                        
                        st.success(f"Transformation saved as: **{transformed_name}**")
                        st.info("Dataset is now active in Data Handling and ready for PCA/DOE")
                        st.rerun()
                    
            except Exception as e:
                st.error(f"Error applying transformation: {str(e)}")
                import traceback
                if st.checkbox("Show debug info", key="row_debug"):
                    st.code(traceback.format_exc())
    
    # ===== COLUMN TRANSFORMATIONS TAB =====
    with tab2:
        st.markdown("## Column Transformations")
        st.markdown("*Transformations applied within each variable*")
        
        col_transforms = {
            "Centering": "centc",
            "Scaling (Unit Variance)": "scalc",
            "Autoscaling": "autosc",
            "Range [0,1]": "01c",
            "Range [-1,1] (DoE Coding)": "cod",
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
        
        if st.button("Apply Transformation", type="primary", key="apply_col_transform"):
            try:
                with st.spinner(f"Applying {selected_transform_col}..."):
                    if transform_code_col == "centc":
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
                    
                    # CRITICAL: Always use original data for comparison
                    if transform_code_col != "blsc":
                        original_slice_col = original_data.iloc[:, col_range_c[0]:col_range_c[1]]
                    else:
                        original_slice_col = original_data
                    
                    fig_col = plot_comparison(
                        original_slice_col,
                        transformed_col,
                        f"Original Data ({original_slice_col.shape[0]} × {original_slice_col.shape[1]})",
                        f"Transformed Data ({transformed_col.shape[0]} × {transformed_col.shape[1]}) - {selected_transform_col}",
                        dark_mode=use_dark_theme_col
                    )
                    
                    st.plotly_chart(fig_col, use_container_width=True)
                    
                    # Statistics BEFORE saving
                    st.markdown("### Transformation Statistics")
                    
                    col_stat1, col_stat2, col_stat3 = st.columns(3)
                    
                    with col_stat1:
                        st.metric("Original Shape", f"{original_slice_col.shape[0]} × {original_slice_col.shape[1]}")
                    with col_stat2:
                        st.metric("Transformed Shape", f"{transformed_col.shape[0]} × {transformed_col.shape[1]}")
                    with col_stat3:
                        mean_val = transformed_col.mean().mean()
                        st.metric("Mean Value", f"{mean_val:.3f}")
                    
                    # Save section - SEPARATED WITH BUTTON
                    st.markdown("---")
                    st.markdown("### Save Transformation")
                    st.info("Review the transformation above, then save it to workspace if satisfied")
                    
                    if st.button("Save to Workspace", type="primary", key="save_col_transform"):
                        full_transformed_col = data.copy()
                        
                        if transform_code_col == "blsc":
                            full_transformed_col = transformed_col
                        else:
                            if transformed_col.shape != original_slice_col.shape:
                                if transformed_col.shape[0] != original_slice_col.shape[0]:
                                    full_transformed_col = full_transformed_col.iloc[:transformed_col.shape[0], :]
                            
                            full_transformed_col.iloc[:, col_range_c[0]:col_range_c[1]] = transformed_col
                        
                        dataset_name = st.session_state.get('current_dataset', 'Dataset')
                        base_name = dataset_name.split('.')[0].replace('_ORIGINAL', '')
                        transformed_name_col = f"{base_name}.{transform_code_col}"
                        
                        st.session_state.current_data = full_transformed_col
                        st.session_state.current_dataset = transformed_name_col
                        
                        if 'transformation_history' not in st.session_state:
                            st.session_state.transformation_history = {}
                        
                        st.session_state.transformation_history[transformed_name_col] = {
                            'data': full_transformed_col,
                            'transform': selected_transform_col,
                            'params': params_col,
                            'col_range': col_range_c,
                            'timestamp': pd.Timestamp.now()
                        }
                        
                        st.success(f"Transformation saved as: **{transformed_name_col}**")
                        st.info("Dataset is now active and ready for PCA/DOE")
                        st.rerun()
                    
            except Exception as e:
                st.error(f"Error applying transformation: {str(e)}")
                import traceback
                if st.checkbox("Show debug info", key="col_debug"):
                    st.code(traceback.format_exc())
    
    if 'transformation_history' in st.session_state and st.session_state.transformation_history:
        with st.sidebar:
            st.markdown("### Transformation History")
            
            for name, info in st.session_state.transformation_history.items():
                with st.expander(name):
                    st.write(f"**Transform:** {info['transform']}")
                    st.write(f"**Shape:** {info['data'].shape[0]} × {info['data'].shape[1]}")
                    st.write(f"**Time:** {info['timestamp'].strftime('%H:%M:%S')}")
                    
                    if st.button(f"Load {name}", key=f"load_{name}"):
                        st.session_state.current_data = info['data']
                        st.session_state.current_dataset = name
                        st.success(f"Loaded: {name}")
                        st.rerun()