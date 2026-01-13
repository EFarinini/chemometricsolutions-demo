"""
Bivariate Analysis Page
Interactive bivariate analysis with correlation ranking, scatter plots, and statistical measures
"""

import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats

# Import workspace utilities
from workspace_utils import display_workspace_dataset_selector

# Import bivariate utilities
from bivariate_utils.statistics import (
    compute_correlation_matrix,
    compute_covariance_matrix,
    get_correlation_summary
)
from bivariate_utils.plotting import (
    create_scatter_plot,
    create_pairs_plot,
    create_correlation_heatmap
)


@st.cache_data
def compute_all_pair_correlations(numeric_data: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Pearson correlations for ALL variable pairs.
    Returns sorted DataFrame with highest |r| values first.

    Parameters
    ----------
    numeric_data : pd.DataFrame
        DataFrame with only numeric columns (no NaN)

    Returns
    -------
    pd.DataFrame
        Columns: ['Variable 1', 'Variable 2', 'Pearson r', 'P-value', '|r|']
        Sorted by |r| descending (strongest correlations first)
    """
    results = []
    n_vars = len(numeric_data.columns)

    # Compute all unique pairs
    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            var1 = numeric_data.columns[i]
            var2 = numeric_data.columns[j]

            # Clean data (remove NaN for this pair)
            pair_data = numeric_data[[var1, var2]].dropna()

            if len(pair_data) >= 2:
                # Compute Pearson correlation
                corr, pval = stats.pearsonr(pair_data[var1], pair_data[var2])

                results.append({
                    'Variable 1': var1,
                    'Variable 2': var2,
                    'Pearson r': corr,
                    'P-value': pval,
                    '|r|': abs(corr)
                })

    # Convert to DataFrame and sort by |r| descending
    corr_df = pd.DataFrame(results)
    if len(corr_df) > 0:
        corr_df = corr_df.sort_values('|r|', ascending=False).reset_index(drop=True)

    return corr_df


def show():
    """
    Main function to display the Bivariate Analysis page
    """
    # Initialize session state variables
    if 'bivariate_dataset_selector' not in st.session_state:
        st.session_state.bivariate_dataset_selector = None
    if 'bivariate_var1' not in st.session_state:
        st.session_state.bivariate_var1 = None
    if 'bivariate_var2' not in st.session_state:
        st.session_state.bivariate_var2 = None

    st.title("📊 Bivariate Analysis")
    st.markdown("""
    Explore relationships between pairs of variables through correlation analysis,
    scatter plots, and statistical measures.
    """)

    # === DATASET SELECTION (using workspace_utils - SAME AS PCA) ===
    st.markdown("---")
    st.markdown("## 📁 Dataset Selection")

    result = display_workspace_dataset_selector(
        label="Select dataset:",
        key="bivariate_dataset_selector",
        help_text="Choose a dataset from your workspace",
        show_info=True
    )

    if result is None:
        return

    dataset_name, data = result

    # === VARIABLE & SAMPLE SELECTION (LIKE PCA) ===
    st.markdown("---")
    st.markdown("## 🎯 Variable & Sample Selection")

    col1, col2 = st.columns(2)
    with col1:
        first_col = st.number_input(
            "First column (1-based):",
            min_value=1,
            max_value=len(data.columns),
            value=2 if len(data.select_dtypes(exclude=['number']).columns) > 0 else 1,
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

    if len(numeric_data.columns) < 2:
        st.error(f"❌ Need at least 2 numeric variables for bivariate analysis. Found: {len(numeric_data.columns)}")
        return

    st.success(f"✅ Will analyze {len(numeric_data.columns)} numeric variables")

    # === MAIN TABS ===
    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Scatter Plot Analysis",
        "📊 Correlation Matrix",
        "📑 Pairs Plot",
        "📋 Covariance & Summary"
    ])

    # =========================================================================
    # TAB 1: SCATTER PLOT ANALYSIS (MAIN) - Ranking + Plot Together
    # =========================================================================
    with tab1:
        st.markdown("## 📊 Scatter Plot Analysis")

        # ======================================================================
        # SECTION A: CORRELATION RANKING (for selecting variables)
        # ======================================================================
        st.markdown("### 📈 Correlation Ranking - Select Variable Pair")
        st.markdown("*Click 'Select' to view scatter plot for that pair below*")

        # Compute correlations
        corr_ranking_df = compute_all_pair_correlations(numeric_data)

        if len(corr_ranking_df) == 0:
            st.warning("⚠️ Could not compute correlations")
        else:
            # Display full table in expander (optional detail)
            with st.expander("Show all pairs (complete table)"):
                display_df = corr_ranking_df.copy()
                display_df['Pearson r'] = display_df['Pearson r'].apply(lambda x: f"{x:.4f}")
                display_df['P-value'] = display_df['P-value'].apply(lambda x: f"{x:.2e}")
                display_df['|r|'] = display_df['|r|'].apply(lambda x: f"{x:.4f}")

                st.dataframe(
                    display_df[['Variable 1', 'Variable 2', 'Pearson r', 'P-value', '|r|']],
                    use_container_width=True,
                    hide_index=True
                )

            # Top 10 with select buttons
            st.markdown("#### 💡 Top 10 Strongest Correlations")
            top_correlations = corr_ranking_df.head(10).copy()

            if len(top_correlations) > 0:
                col_left, col_right = st.columns([2, 1])

                with col_left:
                    for idx, row in top_correlations.iterrows():
                        var1 = row['Variable 1']
                        var2 = row['Variable 2']
                        r_value = row['Pearson r']
                        abs_r = row['|r|']

                        col_rank, col_vars, col_r, col_button = st.columns([0.5, 2, 1, 0.8])

                        with col_rank:
                            st.markdown(f"**{idx + 1}.**")
                        with col_vars:
                            st.markdown(f"`{var1}` → `{var2}`")
                        with col_r:
                            color = "🔴" if abs_r > 0.8 else "🟠" if abs_r > 0.6 else "🟡" if abs_r > 0.4 else "🟢"
                            st.markdown(f"{color} r = {r_value:.4f}")
                        with col_button:
                            if st.button("Select", key=f"select_pair_{idx}"):
                                st.session_state.bivariate_var1 = var1
                                st.session_state.bivariate_var2 = var2
                                st.rerun()

                with col_right:
                    st.markdown("**Strength Legend:**")
                    st.markdown("🔴 |r| > 0.8: Very Strong")
                    st.markdown("🟠 0.6 < |r| ≤ 0.8: Strong")
                    st.markdown("🟡 0.4 < |r| ≤ 0.6: Moderate")
                    st.markdown("🟢 |r| ≤ 0.4: Weak")

        # ======================================================================
        # SECTION B: SCATTER PLOT VISUALIZATION
        # ======================================================================
        st.markdown("---")
        st.markdown("### 🎨 Scatter Plot Visualization")
        st.markdown("*(Select pair from ranking above, or choose manually)*")

        numeric_vars = numeric_data.columns.tolist()
        metadata_cols = selected_data.select_dtypes(exclude=['number']).columns.tolist()

        # Add custom variables from session state if available
        if 'custom_variables' in st.session_state:
            custom_vars = list(st.session_state.custom_variables.keys())
            metadata_cols = metadata_cols + custom_vars

        # Variable selectors (may be pre-filled by ranking selection)
        col1, col2 = st.columns(2)

        with col1:
            # Try to get from session state (set by ranking buttons)
            default_var1_idx = 0
            if hasattr(st.session_state, 'bivariate_var1') and st.session_state.bivariate_var1 in numeric_vars:
                default_var1_idx = numeric_vars.index(st.session_state.bivariate_var1)

            current_var1 = st.selectbox(
                "Variable 1 (X-axis):",
                options=numeric_vars,
                index=default_var1_idx,
                key="bivariate_scatter_var1"
            )

        with col2:
            # Try to get from session state (set by ranking buttons)
            default_var2_idx = min(1, len(numeric_vars)-1)
            if hasattr(st.session_state, 'bivariate_var2') and st.session_state.bivariate_var2 in numeric_vars:
                default_var2_idx = numeric_vars.index(st.session_state.bivariate_var2)

            current_var2 = st.selectbox(
                "Variable 2 (Y-axis):",
                options=numeric_vars,
                index=default_var2_idx,
                key="bivariate_scatter_var2"
            )

        # Visualization Settings
        st.markdown("---")
        st.markdown("#### Visualization Settings")

        col1, col2 = st.columns(2)

        with col1:
            color_by = st.selectbox(
                "Color By (Metadata):",
                options=["None"] + metadata_cols,
                index=0,
                help="Select a categorical column to color-code points",
                key="bivariate_color_by"
            )
            color_by_val = None if color_by == "None" else color_by

            label_by = st.selectbox(
                "Label By (Optional):",
                options=["None", "Index"] + metadata_cols,
                index=0,
                help="Show row indices or column values as labels on points",
                key="bivariate_label_by"
            )
            label_by_val = None if label_by == "None" else label_by

        with col2:
            point_size = st.slider(
                "Point Size:",
                min_value=10,
                max_value=500,
                value=100,
                step=10,
                key="bivariate_point_size"
            )

            opacity = st.slider(
                "Opacity:",
                min_value=0.1,
                max_value=1.0,
                value=0.7,
                step=0.1,
                key="bivariate_opacity"
            )

        # Advanced Options
        st.markdown("#### Advanced Options")

        col1, col2 = st.columns([2, 1])

        with col1:
            convex_hull_option = st.checkbox(
                "🔹 Show Convex Hull",
                value=False,
                key="bivariate_convex_hull"
            )

        # Convex hull settings
        if convex_hull_option and color_by_val is not None:
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                hull_fill = st.checkbox("Fill Area", value=True, key="bivariate_hull_fill")
            with col2:
                hull_opacity = st.slider("Hull Opacity", 0.0, 1.0, 0.2, key="bivariate_hull_opacity")
            with col3:
                hull_line_style = st.selectbox("Line Style", ["solid", "dash", "dot", "dashdot"], key="bivariate_hull_style")
            with col4:
                hull_line_width = st.slider("Line Width", 1, 5, 2, key="bivariate_hull_width")
        else:
            hull_fill = True
            hull_opacity = 0.2
            hull_line_style = 'dash'
            hull_line_width = 2

        # Create scatter plot
        if current_var1 and current_var2 and current_var1 != current_var2:
            try:
                fig = create_scatter_plot(
                    data=selected_data,
                    x_var=current_var1,
                    y_var=current_var2,
                    color_by=color_by_val,
                    label_by=label_by_val,
                    custom_variables=st.session_state.get('custom_variables', None),
                    point_size=point_size,
                    opacity=opacity,
                    show_convex_hull=convex_hull_option,
                    hull_fill=hull_fill,
                    hull_opacity=hull_opacity,
                    hull_line_style=hull_line_style,
                    hull_line_width=hull_line_width
                )

                # Center plot
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.plotly_chart(fig, use_container_width=True)

                # Statistics
                st.markdown("---")

                pair_data = selected_data[[current_var1, current_var2]].dropna()
                if len(pair_data) >= 2:
                    corr, pval = stats.pearsonr(pair_data[current_var1], pair_data[current_var2])

                    metric_col1, metric_col2, metric_col3 = st.columns(3)

                    with metric_col1:
                        st.metric("Pearson r", f"{corr:.4f}")
                    with metric_col2:
                        st.metric("P-value", f"{pval:.2e}")
                    with metric_col3:
                        st.metric("Valid Points", len(pair_data))

            except Exception as e:
                st.error(f"❌ Error creating scatter plot: {str(e)}")

        else:
            if current_var1 == current_var2:
                st.info("ℹ️ Please select **two different variables**")
            else:
                st.info("💡 Select **both variables** to generate scatter plot")

    # =========================================================================
    # TAB 2: CORRELATION MATRIX
    # =========================================================================
    with tab2:
        st.markdown("### Correlation Matrix")

        corr_method = st.radio(
            "Correlation method:",
            options=['pearson', 'spearman', 'kendall'],
            index=0,
            horizontal=True,
            key="bivariate_corr_method"
        )

        try:
            corr_matrix, pval_matrix = compute_correlation_matrix(numeric_data, method=corr_method)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Correlation Coefficients**")
                st.dataframe(corr_matrix.round(3), use_container_width=True)
            with col2:
                st.markdown("**P-values**")
                st.dataframe(pval_matrix.round(4), use_container_width=True)

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

    # =========================================================================
    # TAB 3: PAIRS PLOT
    # =========================================================================
    with tab3:
        st.markdown("### Pairs Plot")
        st.markdown("*Pairwise scatter plots of top correlated variables*")

        # Get numeric variable list
        numeric_vars = numeric_data.columns.tolist()

        # Select top correlated variable pairs (by absolute correlation)
        # Use the already computed correlation ranking
        selected_vars_set = set()
        pairs_plot_vars = []

        for _, row in corr_ranking_df.iterrows():
            var1 = row['Variable 1']
            var2 = row['Variable 2']

            # Add variables until we have 7 (maximum)
            if len(pairs_plot_vars) >= 7:
                break

            if var1 not in selected_vars_set:
                selected_vars_set.add(var1)
                pairs_plot_vars.append(var1)

            if var2 not in selected_vars_set and len(pairs_plot_vars) < 7:
                selected_vars_set.add(var2)
                pairs_plot_vars.append(var2)

        # Fallback if needed
        if len(pairs_plot_vars) < 2:
            pairs_plot_vars = numeric_vars[:min(7, len(numeric_vars))]

        # Display selected variables
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"**Variables:** {', '.join(pairs_plot_vars)}")
        with col2:
            st.metric("Subplots", f"{len(pairs_plot_vars)}² = {len(pairs_plot_vars)**2}")

        # Plot settings
        col1, col2 = st.columns(2)

        with col1:
            # Metadata columns for coloring
            pairs_metadata_cols = data.select_dtypes(exclude=['number']).columns.tolist()

            # Add custom variables from session state if available
            if 'custom_variables' in st.session_state:
                custom_vars = list(st.session_state.custom_variables.keys())
                pairs_metadata_cols = pairs_metadata_cols + custom_vars

            pairs_color_by = st.selectbox(
                "Color By (Metadata):",
                options=["None"] + pairs_metadata_cols,
                index=0,
                key="bivariate_pairs_color_by"
            )
            pairs_color_by_val = None if pairs_color_by == "None" else pairs_color_by

        with col2:
            pairs_opacity = st.slider(
                "Point Opacity:",
                min_value=0.1,
                max_value=1.0,
                value=0.7,
                step=0.05,
                key="bivariate_pairs_opacity"
            )

        # Create pairs plot
        if len(pairs_plot_vars) >= 2:
            try:
                fig_pairs = create_pairs_plot(
                    data=data,
                    variables=pairs_plot_vars,
                    color_by=pairs_color_by_val,
                    opacity=pairs_opacity
                )
                st.plotly_chart(fig_pairs, use_container_width=True)
            except Exception as e:
                st.error(f"❌ Error creating pairs plot: {str(e)}")
        else:
            st.info("💡 Select at least 2 variables for pairs plot")

    # =========================================================================
    # TAB 4: COVARIANCE & SUMMARY
    # =========================================================================
    with tab4:
        st.markdown("## 📋 Covariance Matrix & Correlation Summary")

        # Covariance Matrix Section
        st.markdown("### Covariance Matrix")

        try:
            cov_matrix = compute_covariance_matrix(numeric_data)
            st.dataframe(cov_matrix.round(4), use_container_width=True)

            st.markdown("**Variances (Diagonal)**")
            variances = pd.DataFrame({
                'Variable': cov_matrix.index,
                'Variance': np.diag(cov_matrix)
            })
            st.dataframe(variances, use_container_width=True)

        except Exception as e:
            st.error(f"❌ Error computing covariance: {str(e)}")

        # Correlation Summary Section
        st.markdown("---")
        st.markdown("### Correlation Summary")

        significance_level = st.slider(
            "Significance level (α):",
            min_value=0.01,
            max_value=0.10,
            value=0.05,
            step=0.01,
            key="bivariate_sig_level"
        )

        try:
            corr_matrix, pval_matrix = compute_correlation_matrix(numeric_data, method='pearson')
            summary = get_correlation_summary(corr_matrix, pval_matrix, significance_level)

            st.dataframe(
                summary.style.format({
                    'Correlation': '{:.3f}',
                    'P-value': '{:.4f}'
                }),
                use_container_width=True
            )

            n_significant = (summary['Significant'] == 'Yes').sum()
            st.info(f"📊 Found {n_significant} significant correlation(s) at α = {significance_level}")

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

    # Export section
    st.markdown("---")
    st.markdown("## 💾 Export Results")

    if st.button("📊 Export Correlation Matrix", use_container_width=True):
        try:
            corr_matrix, pval_matrix = compute_correlation_matrix(numeric_data, method='pearson')

            # Save to workspace
            if 'bivariate_results' not in st.session_state:
                st.session_state.bivariate_results = {}

            result_name = f"Correlation_{dataset_name}"
            st.session_state.bivariate_results[result_name] = {
                'correlation': corr_matrix,
                'p_values': pval_matrix,
                'variables': numeric_data.columns.tolist()
            }

            st.success(f"✅ Results exported as '{result_name}'")

        except Exception as e:
            st.error(f"❌ Export failed: {str(e)}")


if __name__ == "__main__":
    show()
