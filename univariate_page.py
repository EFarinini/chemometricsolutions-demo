"""
Univariate Analysis Page - Streamlit Interface (ENHANCED v2.0)

Interactive page for univariate statistical analysis:
- Column-wise with CATEGORICAL vs NUMERIC detection (R-CAT style)
- Batch/group profiles analysis (separate profiles per batch)
- Row profile visualization with enhanced color modes
- Complete statistics export (CSV + Excel)

Author: ChemometricSolutions
Version: 2.0 Enhanced
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Tuple
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import univariate utilities
try:
    from univariate_utils import (
        calculate_descriptive_stats, calculate_dispersion_stats, calculate_robust_stats,
        get_column_statistics_summary, get_row_profile_stats,
        plot_histogram, plot_density, plot_boxplot, plot_stripchart, plot_eda_plot,
        plot_row_profiles, plot_row_profiles_colored, plot_row_profiles_enhanced,
        save_univariate_results, export_statistics_to_csv, export_statistics_to_excel
    )
    UNIVARIATE_AVAILABLE = True
except ImportError as e:
    UNIVARIATE_AVAILABLE = False
    st.error(f"❌ Univariate utilities: {e}")

# Import workspace utilities
try:
    from workspace_utils import get_workspace_datasets
    WORKSPACE_AVAILABLE = True
except ImportError:
    WORKSPACE_AVAILABLE = False

try:
    from color_utils import get_unified_color_schemes
    COLOR_UTILS_AVAILABLE = True
except ImportError:
    COLOR_UTILS_AVAILABLE = False


# ========== FEATURE 1: CATEGORICAL DETECTION (R-CAT STYLE) ==========

def detect_categorical_variables(dataframe: pd.DataFrame, threshold: float = 0.05) -> Dict[str, str]:
    """
    Detect categorical vs numeric variables (similar to R-CAT).

    Rule:
    - Try numeric conversion
    - If < 5% unique values per row count → categorical
    - If > 50% missing → categorical

    Parameters
    ----------
    dataframe : pd.DataFrame
    threshold : float (default 0.05)
        If (unique_values / n_rows) < threshold → categorical

    Returns
    -------
    dict : {'column_name': 'numeric' | 'categorical'}
    """
    variable_types = {}

    for col in dataframe.columns:
        try:
            numeric_data = pd.to_numeric(dataframe[col], errors='coerce')
            n_valid = numeric_data.notna().sum()
            n_total = len(dataframe)
            n_unique = numeric_data.nunique()

            # Conditions for categorical
            if n_valid == 0:  # All non-numeric
                variable_types[col] = 'categorical'
            elif (n_valid / n_total) < 0.5:  # >50% missing
                variable_types[col] = 'categorical'
            elif (n_unique / n_total) < threshold:  # Few unique values
                variable_types[col] = 'categorical'
            else:
                variable_types[col] = 'numeric'
        except:
            variable_types[col] = 'categorical'

    return variable_types


# ========== FEATURE 2: BATCH PROFILES ANALYSIS ==========

def plot_single_batch_profiles(
    batch_data: pd.DataFrame,
    numeric_columns: List[str],
    batch_id: str,
    n_cols: int = 2
) -> go.Figure:
    """
    Create grid of subplots (one per variable) for a single batch.
    Each subplot shows X=sample index, Y=variable value.

    Parameters
    ----------
    batch_data : pd.DataFrame
        Data for this batch only (already filtered)
    numeric_columns : list
        List of numeric column names to plot
    batch_id : str
        Batch identifier for title
    n_cols : int
        Number of columns in grid (default 2)

    Returns
    -------
    go.Figure
        Subplot figure with one plot per variable
    """
    numeric_data = batch_data[numeric_columns].select_dtypes(include=[np.number])
    n_vars = len(numeric_data.columns)
    n_rows = (n_vars + n_cols - 1) // n_cols  # Ceiling division

    # Calculate optimal vertical spacing based on number of rows
    max_spacing = 1.0 / max(2, n_rows - 1) if n_rows > 1 else 0.5
    # Use 70% of max spacing to have safe margin, minimum 0.01
    optimal_spacing = max(0.01, min(0.05, max_spacing * 0.7))

    # Create subplots with dynamic spacing
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=[f"<b>{col}</b>" for col in numeric_data.columns],
        vertical_spacing=optimal_spacing,
        horizontal_spacing=0.10
    )

    # Plot each variable
    for idx, col in enumerate(numeric_data.columns):
        row = idx // n_cols + 1
        col_pos = idx % n_cols + 1

        # Get values for this variable
        values = numeric_data[col].values
        x_indices = np.arange(1, len(values) + 1)

        # Add trace
        fig.add_trace(
            go.Scatter(
                x=x_indices,
                y=values,
                mode='lines+markers',
                name=col,
                line=dict(color='steelblue', width=2),
                marker=dict(size=3, color='steelblue', opacity=0.8),
                hovertemplate=f"<b>{col}</b><br>Sample: %{{x}}<br>Value: %{{y:.3f}}<extra></extra>"
            ),
            row=row,
            col=col_pos
        )

        # Update axes
        fig.update_xaxes(title_text="Sample Index", row=row, col=col_pos)
        fig.update_yaxes(title_text="Value", row=row, col=col_pos)

    # Scale height based on number of rows
    # More rows = less height per row to keep plot manageable
    if n_rows > 20:
        height_per_row = 200  # Reduce height for many subplots
    elif n_rows > 10:
        height_per_row = 250
    else:
        height_per_row = 300

    height = height_per_row * n_rows + 100

    # Update layout
    fig.update_layout(
        title=f"<b>Batch {batch_id}</b> - Independent Variable Profiles ({len(batch_data)} samples)",
        height=height,
        showlegend=False,
        template='plotly_white',
        hovermode='closest'
    )

    return fig


def get_batch_statistics(
    dataframe: pd.DataFrame,
    batch_column: str,
    numeric_columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """Calculate descriptive statistics per batch."""
    if numeric_columns is None:
        numeric_columns = dataframe.select_dtypes(include=[np.number]).columns.tolist()

    batch_stats = []

    for batch_id, batch_data in dataframe.groupby(batch_column):
        row_data = {'Batch': batch_id, 'N_Samples': len(batch_data)}

        for col in numeric_columns:
            try:
                col_numeric = pd.to_numeric(batch_data[col], errors='coerce')
                row_data[f'{col}_mean'] = col_numeric.mean()
                row_data[f'{col}_std'] = col_numeric.std()
                row_data[f'{col}_n'] = col_numeric.notna().sum()
            except:
                pass

        batch_stats.append(row_data)

    return pd.DataFrame(batch_stats)


# ========== TAB 1: EDA BY CATEGORY ==========

def color_to_rgba(color_name: str, alpha: float = 0.5) -> str:
    """
    Convert Plotly color name to RGBA format with transparency.

    Parameters
    ----------
    color_name : str
        Plotly color name (e.g., 'red', 'blue', 'black')
    alpha : float
        Transparency 0-1 (0=transparent, 1=opaque)

    Returns
    -------
    str
        RGBA color string (e.g., 'rgba(255, 0, 0, 0.5)')
    """
    # Color name to RGB mapping
    color_rgb_map = {
        'black': (0, 0, 0),
        'red': (255, 0, 0),
        'green': (0, 128, 0),
        'blue': (0, 0, 255),
        'orange': (255, 165, 0),
        'purple': (128, 0, 128),
        'brown': (165, 42, 42),
        'hotpink': (255, 105, 180),
        'gray': (128, 128, 128),
        'olive': (128, 128, 0),
        'cyan': (0, 255, 255),
        'magenta': (255, 0, 255),
        'gold': (255, 215, 0),
        'navy': (0, 0, 128),
        'darkgreen': (0, 100, 0),
        'darkred': (139, 0, 0),
        'indigo': (75, 0, 130),
        'coral': (255, 127, 80),
        'teal': (0, 128, 128),
        'chocolate': (210, 105, 30),
        'crimson': (220, 20, 60),
        'darkviolet': (148, 0, 211),
        'darkorange': (255, 140, 0),
        'darkslategray': (47, 79, 79),
        'royalblue': (65, 105, 225),
        'saddlebrown': (139, 69, 19)
    }

    if color_name.lower() in color_rgb_map:
        r, g, b = color_rgb_map[color_name.lower()]
        return f'rgba({r}, {g}, {b}, {alpha})'

    # If not found, return with default transparency
    return f'rgba(128, 128, 128, {alpha})'


def plot_eda_by_category(
    dataframe: pd.DataFrame,
    column_name: str,
    category_column: str
) -> go.Figure:
    """
    Create 4-subplot EDA plot with all plots grouped by category.

    Parameters
    ----------
    dataframe : pd.DataFrame
    column_name : str
        Numeric column to analyze
    category_column : str
        Categorical column for grouping

    Returns
    -------
    go.Figure
        2x2 subplot figure with Histogram, Q-Q, Boxplot, Density
    """
    from scipy import stats
    from scipy.stats import gaussian_kde
    from color_utils import create_categorical_color_map

    # Get categories and create color map
    categories = sorted(dataframe[category_column].dropna().unique())
    color_map = create_categorical_color_map(categories)

    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            f"Histogram by {category_column}",
            f"Q-Q Plot by {category_column}",
            f"Boxplot by {category_column}",
            f"Density by {category_column}"
        )
    )

    for idx, category in enumerate(categories):
        color = color_map[category]

        # Filter data for this category
        cat_data = dataframe[dataframe[category_column] == category][column_name]
        cat_data_clean = pd.to_numeric(cat_data, errors='coerce').dropna().values

        if len(cat_data_clean) == 0:
            continue

        # 1. HISTOGRAM (row=1, col=1) - Overlayed
        fig.add_trace(
            go.Histogram(
                x=cat_data_clean,
                name=str(category),
                marker_color=color,
                opacity=0.6,
                nbinsx=20,
                legendgroup=str(category),
                showlegend=True
            ),
            row=1, col=1
        )

        # 2. Q-Q PLOT (row=1, col=2) - Multiple traces
        sorted_data = np.sort(cat_data_clean)
        theoretical_quantiles = stats.norm.ppf(
            np.linspace(0.01, 0.99, len(sorted_data))
        )
        fig.add_trace(
            go.Scatter(
                x=theoretical_quantiles,
                y=sorted_data,
                mode='markers',
                name=str(category),
                marker=dict(color=color, size=4),
                legendgroup=str(category),
                showlegend=False
            ),
            row=1, col=2
        )

        # 3. BOXPLOT (row=2, col=1) - Side-by-side
        fig.add_trace(
            go.Box(
                y=cat_data_clean,
                name=str(category),
                marker_color=color,
                boxmean='sd',
                legendgroup=str(category),
                showlegend=False
            ),
            row=2, col=1
        )

        # 4. DENSITY (row=2, col=2) - Overlayed curves with RGBA transparency
        try:
            kde = gaussian_kde(cat_data_clean)
            x_range = np.linspace(
                cat_data_clean.min() - cat_data_clean.std(),
                cat_data_clean.max() + cat_data_clean.std(),
                200
            )
            density = kde(x_range)

            # Convert color to RGBA with alpha=0.5 for proper transparency
            rgba_color = color_to_rgba(color, alpha=0.5)

            fig.add_trace(
                go.Scatter(
                    x=x_range,
                    y=density,
                    mode='lines',
                    name=str(category),
                    line=dict(color=color, width=2.5),  # Solid line
                    fill='tozeroy',
                    fillcolor=rgba_color,  # RGBA with transparency
                    opacity=1.0,  # No additional opacity (already in RGBA)
                    legendgroup=str(category),
                    showlegend=False,
                    hovertemplate=f"<b>{category}</b><br>Value: %{{x:.3f}}<br>Density: %{{y:.4f}}<extra></extra>"
                ),
                row=2, col=2
            )
        except:
            pass

    # Update axes
    fig.update_xaxes(title_text=column_name, row=1, col=1)
    fig.update_xaxes(title_text="Theoretical Quantiles", row=1, col=2)
    fig.update_yaxes(title_text="Frequency", row=1, col=1)
    fig.update_yaxes(title_text="Sample Quantiles", row=1, col=2)

    fig.update_xaxes(title_text=category_column, row=2, col=1)
    fig.update_xaxes(title_text=column_name, row=2, col=2)
    fig.update_yaxes(title_text=column_name, row=2, col=1)
    fig.update_yaxes(title_text="Density", row=2, col=2)

    fig.update_layout(
        title=f"<b>EDA Analysis: {column_name} by {category_column}</b>",
        height=700,
        template='plotly_white',
        hovermode='closest',
        barmode='overlay'
    )

    return fig


def get_statistics_by_category(
    dataframe: pd.DataFrame,
    column_name: str,
    category_column: str
) -> pd.DataFrame:
    """
    Calculate statistics grouped by category.

    Parameters
    ----------
    dataframe : pd.DataFrame
    column_name : str
        Numeric column to analyze
    category_column : str
        Categorical column for grouping

    Returns
    -------
    pd.DataFrame
        Statistics table with one row per category
    """
    stats_list = []

    for category in sorted(dataframe[category_column].dropna().unique()):
        cat_data = dataframe[dataframe[category_column] == category][column_name]
        cat_data_clean = pd.to_numeric(cat_data, errors='coerce').dropna()

        if len(cat_data_clean) == 0:
            continue

        stats_dict = {
            'Category': category,
            'Count': len(cat_data_clean),
            'Mean': cat_data_clean.mean(),
            'Median': cat_data_clean.median(),
            'Std Dev': cat_data_clean.std(),
            'Min': cat_data_clean.min(),
            'Max': cat_data_clean.max(),
            'Q1': cat_data_clean.quantile(0.25),
            'Q3': cat_data_clean.quantile(0.75),
            'IQR': cat_data_clean.quantile(0.75) - cat_data_clean.quantile(0.25)
        }
        stats_list.append(stats_dict)

    return pd.DataFrame(stats_list)


# ========== FEATURE 3: ENHANCED ROW COLORING ==========

def create_colored_profiles_from_column(
    dataframe: pd.DataFrame,
    row_indices: List[int],
    color_column: str
) -> Tuple[np.ndarray, str]:
    """
    Create color vector from dataset column (not external).
    Improves control and clarity.
    """
    try:
        color_data = pd.to_numeric(dataframe[color_column], errors='coerce')
        color_vector = color_data.values
        color_label = f"Colored by: {color_column}"
        return color_vector, color_label
    except:
        return None, "Could not process color column"


def parse_row_input(row_input: str, max_rows: int) -> List[int]:
    """Parse '1,3,5-7' into indices (0-based)"""
    indices = []
    try:
        for part in row_input.split(','):
            part = part.strip()
            if '-' in part:
                s, e = map(int, part.split('-'))
                indices.extend(range(s-1, min(e, max_rows)))
            else:
                idx = int(part) - 1
                if 0 <= idx < max_rows:
                    indices.append(idx)
    except:
        st.warning("⚠️ Invalid format")
        return []
    return sorted(list(set(indices)))


def show():
    """Main function - Univariate Analysis Page v2.0"""

    if not UNIVARIATE_AVAILABLE:
        st.error("❌ Univariate utilities not available")
        return

    st.markdown("""
    # 📊 Univariate Analysis v2.0

    **Enhanced analysis with categorical detection, batch profiles, and advanced coloring**

    Features:
    - 🏷️ Automatic categorical vs numeric detection (R-CAT style)
    - 📊 Batch/group profile analysis
    - 📈 Enhanced row profile coloring options
    - 💾 Complete statistics export
    """)

    # ===== DATASET SELECTION =====
    st.markdown("---")
    st.subheader("1️⃣ Dataset Selection")

    if WORKSPACE_AVAILABLE:
        # Get available datasets
        available_datasets = get_workspace_datasets()

        if not available_datasets:
            st.warning("⚠️ No datasets available. Please upload data or run other analyses first.")
            return

        dataset_name = st.selectbox(
            "Select dataset:",
            options=list(available_datasets.keys()),
            help="Choose from datasets in workspace"
        )
        selected_data = available_datasets[dataset_name]
    else:
        if 'current_data' not in st.session_state or st.session_state.current_data is None:
            st.error("❌ No data available")
            return
        dataset_name = "Current Data"
        selected_data = st.session_state.current_data

    st.success(f"✅ {dataset_name} ({len(selected_data)} samples × {len(selected_data.columns)} vars)")

    # Detect variable types - show summary
    var_types = detect_categorical_variables(selected_data)
    numeric_vars = [col for col, vtype in var_types.items() if vtype == 'numeric']
    categorical_vars = [col for col, vtype in var_types.items() if vtype == 'categorical']

    with st.expander(f"🔍 Variable Detection: {len(numeric_vars)} numeric, {len(categorical_vars)} categorical"):
        c1, c2 = st.columns(2)
        with c1:
            st.write("**Numeric:**")
            st.write(numeric_vars)
        with c2:
            st.write("**Categorical:**")
            st.write(categorical_vars)

    # ===== MAIN TABS (4 TABS NOW) =====
    st.markdown("---")
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Column Analysis (+ Detection)",
        "📊 Batch Profiles",
        "📈 Row Profiles (Enhanced)",
        "💾 Export"
    ])

    # ========== TAB 1: COLUMN ANALYSIS WITH EDA BY CATEGORY ==========
    with tab1:
        st.markdown("## 📈 Column Analysis - EDA with Category Grouping")

        # ===== SECTION 1: COLUMN SELECTION =====
        col1, col2 = st.columns([3, 2])

        with col1:
            if not numeric_vars:
                st.error("❌ No numeric columns available")
                st.stop()

            selected_column = st.selectbox(
                "📈 Select Numeric Column:",
                numeric_vars,
                key="tab1_col_select",
                help="Choose a numeric variable to analyze"
            )

        with col2:
            # Group by category checkbox
            group_by_category = st.checkbox(
                "📂 Group by Category",
                value=False,
                key="tab1_group_by_cat",
                help="Enable to see distributions grouped by a categorical variable"
            )

        # ===== SECTION 2: CATEGORY SELECTION (IF ENABLED) =====
        category_column = None
        if group_by_category:
            if not categorical_vars:
                st.warning("⚠️ No categorical columns available for grouping")
                group_by_category = False
            else:
                category_column = st.selectbox(
                    "Select Category Column:",
                    categorical_vars,
                    key="tab1_cat_col",
                    help="Choose a categorical variable for grouping"
                )

        st.markdown("---")

        # ===== SECTION 3: EDA PLOT =====
        st.markdown("### 📊 Exploratory Data Analysis")

        try:
            if group_by_category and category_column:
                # BY CATEGORY: 4-subplot EDA with all plots grouped
                st.markdown(f"**Analysis: {selected_column} grouped by {category_column}**")

                fig = plot_eda_by_category(
                    selected_data,
                    selected_column,
                    category_column
                )
                st.plotly_chart(fig, use_container_width=True)

            else:
                # SINGLE: Standard EDA plot
                st.markdown(f"**Analysis: {selected_column}**")

                column_data = pd.to_numeric(selected_data[selected_column], errors='coerce')
                col_clean = column_data.dropna().values

                if len(col_clean) == 0:
                    st.error("❌ No valid numeric data")
                else:
                    fig = plot_eda_plot(col_clean, column_name=selected_column)
                    st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"❌ Error creating plot: {e}")
            import traceback
            st.code(traceback.format_exc())

        # ===== SECTION 4: STATISTICS TABLE =====
        st.markdown("---")
        st.markdown("### 📊 Statistics")

        try:
            if group_by_category and category_column:
                # Statistics BY CATEGORY
                stats_df = get_statistics_by_category(
                    selected_data,
                    selected_column,
                    category_column
                )

                # Format numeric columns
                numeric_cols = ['Count', 'Mean', 'Median', 'Std Dev', 'Min', 'Max', 'Q1', 'Q3', 'IQR']
                format_dict = {col: "{:.3f}" for col in numeric_cols if col in stats_df.columns}
                format_dict['Count'] = "{:.0f}"  # Integer format for count

                st.dataframe(
                    stats_df.style.format(format_dict),
                    use_container_width=True
                )

                # Export button
                csv_data = stats_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Statistics (CSV)",
                    data=csv_data,
                    file_name=f"{selected_column}_by_{category_column}_statistics.csv",
                    mime="text/csv",
                    key="tab1_stats_csv"
                )

            else:
                # Statistics OVERALL
                column_data = pd.to_numeric(selected_data[selected_column], errors='coerce')

                if column_data.notna().sum() > 0:
                    cs1, cs2, cs3 = st.columns(3)

                    with cs1:
                        st.write("**Descriptive**")
                        desc = calculate_descriptive_stats(column_data)
                        desc_df = pd.DataFrame({
                            'Statistic': list(desc.keys()),
                            'Value': [f"{v:.4f}" if isinstance(v, (int, float)) and not pd.isna(v) else str(v) for v in desc.values()]
                        })
                        st.dataframe(desc_df, use_container_width=True, hide_index=True)

                    with cs2:
                        st.write("**Dispersion**")
                        disp = calculate_dispersion_stats(column_data)
                        disp_df = pd.DataFrame({
                            'Measure': list(disp.keys()),
                            'Value': [f"{v:.4f}" for v in disp.values()]
                        })
                        st.dataframe(disp_df, use_container_width=True, hide_index=True)

                    with cs3:
                        st.write("**Robust**")
                        robust = calculate_robust_stats(column_data)
                        robust_df = pd.DataFrame({
                            'Statistic': list(robust.keys()),
                            'Value': [f"{v:.4f}" for v in robust.values()]
                        })
                        st.dataframe(robust_df, use_container_width=True, hide_index=True)
                else:
                    st.warning("⚠️ No valid data for statistics")

        except Exception as e:
            st.error(f"❌ Error calculating statistics: {e}")

    # ========== TAB 2: BATCH PROFILES - INDEPENDENT PLOTS PER VARIABLE ==========
    with tab2:
        st.markdown("## 📊 Batch Profiles - Independent Variable Analysis")
        st.markdown("*Select a batch to view profiles of all variables independently*")

        # ===== SECTION 1: BATCH SELECTION =====
        st.markdown("### 🔍 Batch Selection")
        col1, col2, col3 = st.columns([2, 2, 1])

        with col1:
            # Find categorical columns
            if categorical_vars:
                batch_col = st.selectbox(
                    "📂 Select Batch Column:",
                    categorical_vars,
                    key="batch_col",
                    help="Column containing batch identifiers"
                )
            else:
                st.warning("⚠️ No categorical columns found for batching")
                batch_col = None

        if batch_col:
            with col2:
                # Get unique batches
                unique_batches = sorted(selected_data[batch_col].unique())
                selected_batch = st.selectbox(
                    "🎯 Select Batch to Analyze:",
                    unique_batches,
                    key="selected_batch",
                    help="Choose one batch to view all variable profiles"
                )

            with col3:
                batch_size = len(selected_data[selected_data[batch_col] == selected_batch])
                st.metric("📊 Samples", batch_size)

            # ===== SECTION 2: FILTER DATA & CREATE PLOTS =====
            batch_data = selected_data[selected_data[batch_col] == selected_batch].copy()
            batch_numeric = batch_data[numeric_vars] if numeric_vars else pd.DataFrame()

            if len(batch_numeric) > 0 and len(batch_numeric.columns) > 0:

                # Create the subplot figure
                st.markdown(f"### 📈 Profiles for **{batch_col} = {selected_batch}**")

                fig = plot_single_batch_profiles(
                    batch_data,
                    batch_numeric.columns.tolist(),
                    selected_batch,
                    n_cols=2
                )

                # Display
                st.plotly_chart(fig, use_container_width=True)

                # ===== SECTION 3: STATISTICS TABLE =====
                st.markdown("### 📊 Batch Statistics")

                stats_data = {
                    'Variable': batch_numeric.columns,
                    'Mean': batch_numeric.mean().values,
                    'Std Dev': batch_numeric.std().values,
                    'Min': batch_numeric.min().values,
                    'Max': batch_numeric.max().values,
                    'Q1': batch_numeric.quantile(0.25).values,
                    'Median': batch_numeric.median().values,
                    'Q3': batch_numeric.quantile(0.75).values,
                }

                stats_df = pd.DataFrame(stats_data)
                stats_df_transposed = stats_df.set_index('Variable').T
                st.dataframe(
                    stats_df_transposed.style.format("{:.3f}"),
                    use_container_width=True
                )

                # ===== SECTION 4: EXPORT =====
                st.markdown("### 💾 Export Options")

                col1, col2 = st.columns(2)

                with col1:
                    # Export CSV
                    csv_data = stats_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Statistics (CSV)",
                        data=csv_data,
                        file_name=f"{batch_col}_{selected_batch}_statistics.csv",
                        mime="text/csv",
                        key="batch_stats_csv"
                    )

                with col2:
                    # Export raw batch data
                    raw_csv = batch_numeric.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Raw Data (CSV)",
                        data=raw_csv,
                        file_name=f"{batch_col}_{selected_batch}_raw_data.csv",
                        mime="text/csv",
                        key="batch_raw_csv"
                    )

            else:
                st.warning("⚠️ No numeric data available for this batch")

        else:
            st.info("👆 Select a batch column above to get started")

    # ========== TAB 3: ROW PROFILES ENHANCED COLORING ==========
    with tab3:
        st.markdown("## 📊 Row Profiles (Enhanced Coloring)")
        st.markdown("*Visualize sample profiles with selective variable range and metavariable coloring*")

        st.divider()

        # ===== STEP 1: SELECT X MATRIX (COLUMN RANGE) =====
        st.markdown("### 📊 Step 1: Select X Matrix (Variables)")

        all_columns = selected_data.columns.tolist()
        numeric_cols = selected_data.select_dtypes(include=[np.number]).columns.tolist()

        # Info box
        col_info1, col_info2 = st.columns(2)
        with col_info1:
            st.metric("Total Columns", len(all_columns))
        with col_info2:
            st.metric("Numeric Columns", len(numeric_cols))

        # Determine default numeric range
        if len(numeric_cols) > 0:
            first_numeric_idx = all_columns.index(numeric_cols[0]) + 1
            last_numeric_idx = all_columns.index(numeric_cols[-1]) + 1
        else:
            first_numeric_idx = 1
            last_numeric_idx = len(all_columns)

        # Column range selection
        st.markdown("#### 🎯 Define X column range (1-based):")
        col_range_1, col_range_2 = st.columns(2)

        with col_range_1:
            first_col = st.number_input(
                "First column:",
                min_value=1,
                max_value=len(all_columns),
                value=first_numeric_idx,
                key="tab3_first_col",
                help="Start column index (1-based)"
            )

        with col_range_2:
            last_col = st.number_input(
                "Last column:",
                min_value=first_col,
                max_value=len(all_columns),
                value=last_numeric_idx,
                key="tab3_last_col",
                help="End column index (1-based, inclusive)"
            )

        # Extract X matrix columns
        x_cols = all_columns[first_col - 1:last_col]
        st.info(f"**Selected X columns:** {first_col} to {last_col} = `{x_cols[0]}` ... `{x_cols[-1]}`  \n({len(x_cols)} variables)")

        st.divider()

        # ===== STEP 2: SELECT ROW RANGE (SAMPLES) =====
        st.markdown("### 🎯 Step 2: Select Row Range (Samples)")

        n_samples = len(selected_data)
        st.info(f"Dataset: {n_samples} total samples")

        # Row range selection
        st.markdown("#### 🎯 Define row range (1-based):")
        row_range_1, row_range_2 = st.columns(2)

        with row_range_1:
            first_row = st.number_input(
                "First row:",
                min_value=1,
                max_value=n_samples,
                value=1,
                key="tab3_first_row",
                help="Start row index (1-based)"
            )

        with row_range_2:
            last_row = st.number_input(
                "Last row:",
                min_value=first_row,
                max_value=n_samples,
                value=n_samples,
                key="tab3_last_row",
                help="End row index (1-based, inclusive)"
            )

        # Extract subset
        X_matrix = selected_data.iloc[first_row-1:last_row, first_col-1:last_col]
        st.info(f"**Selected rows:** {first_row} to {last_row} ({len(X_matrix)} samples)  \n**Selected columns:** {len(X_matrix.columns)} variables  \n**Shape:** {X_matrix.shape}")

        st.divider()

        # ===== STEP 3: SELECT COLORING (BY METAVARIABLE) =====
        st.markdown("### 🎨 Step 3: Select Coloring Mode")

        coloring_mode = st.radio(
            "Choose coloring:",
            ["Uniform", "By Row Index", "By Metavariable"],
            horizontal=True,
            key="tab3_coloring_mode"
        )

        color_variable = None
        color_values = None

        if coloring_mode == "Uniform":
            st.write("All profiles will be colored uniformly (steel blue)")

        elif coloring_mode == "By Row Index":
            st.write("Profiles colored by sample index (blue → red gradient)")

        elif coloring_mode == "By Metavariable":
            st.markdown("#### Select metavariable for coloring:")

            # Show all columns (for potential grouping/coloring by metadata)
            available_for_coloring = [col for col in all_columns if col not in x_cols]

            if available_for_coloring:
                color_variable = st.selectbox(
                    "Metavariable for coloring:",
                    available_for_coloring,
                    key="tab3_color_variable",
                    help="Select a column outside X range to color the profiles"
                )

                # Get color values for selected rows
                color_values = selected_data.loc[selected_data.index[first_row-1:last_row], color_variable].values

                # Detect if numeric or categorical
                try:
                    color_numeric = pd.Series(color_values)
                    color_numeric = pd.to_numeric(color_numeric, errors='coerce')
                    if color_numeric.notna().sum() > 0:
                        st.write(f"**Type:** Numeric (will use blue-red gradient)")
                    else:
                        st.write(f"**Type:** Categorical (will use discrete colors)")
                except:
                    st.write(f"**Type:** Categorical (will use discrete colors)")
            else:
                st.warning("⚠️ No metavariables available (all columns are in X range)")

        st.divider()

        # ===== PLOT SECTION =====
        st.markdown("### 📈 Row Profiles Plot")

        # Marker size slider - COMPACT
        col_m1, col_m2 = st.columns([1, 3])
        with col_m1:
            st.write("Marker:")
        with col_m2:
            marker_size = st.slider(
                "Size",
                min_value=1,
                max_value=10,
                value=3,
                step=1,
                key="tab3_marker_size",
                label_visibility="collapsed"
            )

        if len(X_matrix) == 0 or len(X_matrix.columns) == 0:
            st.error("❌ No data selected. Please adjust column and row ranges.")
        else:
            try:
                # Map mode to function parameter
                is_numeric_color = False
                if color_variable and color_values is not None:
                    try:
                        color_series = pd.Series(color_values)
                        color_numeric = pd.to_numeric(color_series, errors='coerce')
                        is_numeric_color = color_numeric.notna().sum() > 0
                    except:
                        is_numeric_color = False

                mode_map = {
                    "Uniform": "uniform",
                    "By Row Index": "row_index",
                    "By Metavariable": "column_value" if is_numeric_color else "category"
                }

                # Create plot data with color variable if needed
                if coloring_mode == "By Metavariable" and color_variable:
                    # Add color variable to X_matrix for plotting
                    plot_data = X_matrix.copy()
                    plot_data[color_variable] = selected_data.iloc[first_row-1:last_row][color_variable].values
                else:
                    plot_data = X_matrix

                fig = plot_row_profiles_enhanced(
                    plot_data,
                    color_mode=mode_map[coloring_mode],
                    color_variable=color_variable,
                    row_indices=list(range(len(plot_data))),
                    marker_size=marker_size
                )

                st.plotly_chart(fig, use_container_width=True)
                st.success(f"✅ Plotted {len(X_matrix)} samples × {len(X_matrix.columns)} variables")

            except Exception as e:
                st.error(f"❌ Error creating plot: {e}")
                import traceback
                st.code(traceback.format_exc())

        st.divider()

        # ===== STATISTICS SECTION =====
        st.markdown("### 📊 Profile Statistics")

        if len(X_matrix) > 0 and len(X_matrix.columns) > 0:
            stats_dict = {
                'Statistic': ['Count', 'Mean', 'Median', 'Std Dev', 'Min', 'Max', 'Range'],
                'Value': [
                    len(X_matrix),
                    X_matrix.values.mean(),
                    np.median(X_matrix.values),
                    X_matrix.values.std(),
                    X_matrix.values.min(),
                    X_matrix.values.max(),
                    X_matrix.values.max() - X_matrix.values.min()
                ]
            }

            stats_df = pd.DataFrame(stats_dict)
            st.dataframe(
                stats_df.style.format({'Value': '{:.4f}'}),
                use_container_width=True
            )

            # Export
            st.markdown("#### 💾 Export")
            exp_col1, exp_col2 = st.columns(2)

            with exp_col1:
                csv_stats = stats_df.to_csv(index=False)
                st.download_button(
                    "📥 Statistics (CSV)",
                    data=csv_stats,
                    file_name=f"row_profiles_stats_r{first_row}_{last_row}_c{first_col}_{last_col}.csv",
                    mime="text/csv",
                    key="tab3_stats_csv"
                )

            with exp_col2:
                csv_data = X_matrix.to_csv(index=False)
                st.download_button(
                    "📥 Subset Data (CSV)",
                    data=csv_data,
                    file_name=f"row_profiles_data_r{first_row}_{last_row}_c{first_col}_{last_col}.csv",
                    mime="text/csv",
                    key="tab3_subset_csv"
                )

    # ========== TAB 4: EXPORT ==========
    with tab4:
        st.markdown("### 4. Statistics Export")
        st.info("💾 Export multi-column statistics (CSV + Excel)")

        ce1, ce2 = st.columns([1, 1])

        with ce1:
            st.markdown("**Columns:**")
            exp_all = st.checkbox("All", True, key="exp_all")
            exp_cols = selected_data.columns.tolist() if exp_all else st.multiselect(
                "Select:", selected_data.columns.tolist(), list(selected_data.columns[:3]), key="exp_cols"
            )

        with ce2:
            st.markdown("**Stats:**")
            exp_desc = st.checkbox("Descriptive", True, key="exp_desc")
            exp_disp = st.checkbox("Dispersion", True, key="exp_disp")
            exp_rob = st.checkbox("Robust", True, key="exp_rob")

        st.markdown("---")

        all_stats = {}

        for col in exp_cols:
            try:
                col_data = pd.to_numeric(selected_data[col], errors='coerce')
                col_stats = {}

                if exp_desc:
                    col_stats.update(calculate_descriptive_stats(col_data))
                if exp_disp:
                    col_stats.update(calculate_dispersion_stats(col_data))
                if exp_rob:
                    col_stats.update(calculate_robust_stats(col_data))

                all_stats[col] = col_stats
            except:
                pass

        if all_stats:
            exp_df = pd.DataFrame(all_stats).T
            exp_df.index.name = 'Column'

            st.markdown("#### Table")
            st.dataframe(exp_df, use_container_width=True)

            st.markdown("#### Download")

            ce_csv, ce_xlsx = st.columns(2)

            with ce_csv:
                csv_data = exp_df.to_csv()
                st.download_button(
                    "📥 CSV",
                    data=csv_data,
                    file_name="stats.csv",
                    mime="text/csv",
                    key="exp_csv"
                )

            with ce_xlsx:
                try:
                    # Create proper format for export_statistics_to_excel
                    export_dict = {'Statistics': exp_df.reset_index()}
                    xlsx_buf = export_statistics_to_excel(export_dict, include_metadata=True)
                    st.download_button(
                        "📥 Excel",
                        data=xlsx_buf.getvalue(),
                        file_name="stats.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key="exp_xlsx"
                    )
                except Exception as e:
                    st.warning(f"⚠️ Excel export error: {e}")
