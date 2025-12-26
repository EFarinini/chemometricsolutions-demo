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

    # === PEARSON CORRELATION RANKING TABLE ===
    st.markdown("---")
    st.markdown("## 📈 Pearson Correlation Ranking")
    st.markdown("*All variable pairs ranked by correlation strength (|r|)*")

    # Compute correlations for ALL pairs
    corr_ranking_df = compute_all_pair_correlations(numeric_data)

    if len(corr_ranking_df) == 0:
        st.warning("⚠️ Could not compute correlations - check for NaN values")
    else:
        # Display formatted table
        st.markdown("### Top Correlations")

        # Format for display
        display_df = corr_ranking_df.copy()
        display_df['Pearson r'] = display_df['Pearson r'].apply(lambda x: f"{x:.4f}")
        display_df['P-value'] = display_df['P-value'].apply(lambda x: f"{x:.2e}")
        display_df['|r|'] = display_df['|r|'].apply(lambda x: f"{x:.4f}")

        # Display with columns selection
        st.dataframe(
            display_df[['Variable 1', 'Variable 2', 'Pearson r', 'P-value', '|r|']],
            use_container_width=True,
            hide_index=True
        )

        st.info(f"📊 Total pairs analyzed: {len(corr_ranking_df)}")

    # === CORRELATION ANALYSIS ===
    st.markdown("---")
    st.markdown("## 📊 Correlation Analysis")
    st.markdown("*Correlation rankings and pairwise relationships*")

    # === SCATTER PLOT VARIABLE PAIR SELECTOR (OPTIONAL) ===
    st.markdown("---")
    st.markdown("## 🎨 Scatter Plot Visualization (Optional)")
    st.markdown("*Select a specific variable pair for detailed scatter plot with metadata coloring*")

    # === TOP CORRELATIONS SUGGESTIONS ===
    st.markdown("### 💡 Top 10 Strongest Correlations (Suggestions)")

    # Get top 10 pairs
    top_correlations = corr_ranking_df.head(10).copy()

    if len(top_correlations) > 0:
        # Create two columns for display
        col_left, col_right = st.columns([2, 1])

        with col_left:
            # Display suggestions as clickable cards/rows
            st.markdown("**Click any pair below to select it:**")

            for idx, row in top_correlations.iterrows():
                var1 = row['Variable 1']
                var2 = row['Variable 2']
                r_value = row['Pearson r']
                abs_r = row['|r|']

                # Create a formatted row with button
                col_rank, col_vars, col_r, col_button = st.columns([0.5, 2, 1, 0.8])

                with col_rank:
                    # Rank number (1-10)
                    st.markdown(f"**{idx + 1}.**")

                with col_vars:
                    # Variable pair with arrow
                    st.markdown(f"`{var1}` → `{var2}`")

                with col_r:
                    # Correlation value with color indicator
                    color = "🔴" if abs_r > 0.8 else "🟠" if abs_r > 0.6 else "🟡" if abs_r > 0.4 else "🟢"
                    st.markdown(f"{color} r = {r_value:.4f}")

                with col_button:
                    # Button to select this pair
                    if st.button("Select", key=f"select_pair_{idx}"):
                        st.session_state.bivariate_var1 = var1
                        st.session_state.bivariate_var2 = var2
                        st.rerun()

        with col_right:
            # Legend for color intensity
            st.markdown("**Correlation Strength:**")
            st.markdown("🔴 |r| > 0.8 (Very Strong)")
            st.markdown("🟠 |r| > 0.6 (Strong)")
            st.markdown("🟡 |r| > 0.4 (Moderate)")
            st.markdown("🟢 |r| ≤ 0.4 (Weak)")

    st.markdown("---")

    # === MANUAL VARIABLE SELECTION ===
    st.markdown("### 📌 Or Select Variables Manually")

    col1, col2 = st.columns(2)

    with col1:
        var1_options = numeric_data.columns.tolist()
        default_var1_idx = 0
        if st.session_state.bivariate_var1 is not None and st.session_state.bivariate_var1 in var1_options:
            default_var1_idx = var1_options.index(st.session_state.bivariate_var1)

        selected_var1 = st.selectbox(
            "Variable 1 (X-axis):",
            options=var1_options,
            index=default_var1_idx,
            key="bivariate_var1_selector"
        )

    with col2:
        var2_options = [v for v in numeric_data.columns.tolist() if v != st.session_state.bivariate_var1]
        if len(var2_options) == 0:
            st.error("❌ Need at least 2 different variables for scatter plot")
            var2_options = var1_options

        default_var2_idx = 0
        if st.session_state.bivariate_var2 is not None and st.session_state.bivariate_var2 in var2_options:
            default_var2_idx = var2_options.index(st.session_state.bivariate_var2)

        selected_var2 = st.selectbox(
            "Variable 2 (Y-axis):",
            options=var2_options,
            index=default_var2_idx,
            key="bivariate_var2_selector"
        )

    # Update session state when selections change
    if selected_var1 != st.session_state.bivariate_var1:
        st.session_state.bivariate_var1 = selected_var1
        st.rerun()

    if selected_var2 != st.session_state.bivariate_var2:
        st.session_state.bivariate_var2 = selected_var2
        st.rerun()

    # Visualization controls
    st.markdown("### Visualization Settings")

    col1, col2, col3 = st.columns(3)

    with col1:
        # Identify metadata columns (non-numeric from original data + custom variables)
        metadata_cols = data.select_dtypes(exclude=['number']).columns.tolist()

        # Add custom variables from session state if available
        if 'custom_variables' in st.session_state:
            custom_vars = list(st.session_state.custom_variables.keys())
            metadata_cols = metadata_cols + custom_vars

        color_options = ["None"] + metadata_cols
        color_by = st.selectbox(
            "Color By (Metadata):",
            options=color_options,
            index=0,
            help="Select a categorical column to color-code points",
            key="bivariate_color_by"
        )

        label_by = st.selectbox(
            "Label By (Optional):",
            options=["None", "Index"] + metadata_cols,
            index=0,
            help="Select a column for point labels on hover",
            key="bivariate_label_by"
        )

    with col2:
        point_size = st.slider(
            "Point Size:",
            min_value=10,
            max_value=500,
            value=100,
            step=10,
            help="Size of scatter plot points",
            key="bivariate_point_size"
        )

        opacity = st.slider(
            "Opacity:",
            min_value=0.1,
            max_value=1.0,
            value=0.7,
            step=0.1,
            help="Point transparency",
            key="bivariate_opacity"
        )

    # === CONVEX HULL OPTIONS ===
    st.markdown("### Advanced Options")

    col_hull, col_spacer = st.columns([2, 1])

    with col_hull:
        convex_hull_option = st.checkbox(
            "🔹 Show Convex Hull",
            value=False,
            help="Display convex hull boundaries around categorical groups (requires Color By selection)",
            key="bivariate_convex_hull"
        )

    # Show hull customization options only if hull is enabled and color_by is selected
    if convex_hull_option and color_by != "None":
        st.markdown("**Convex Hull Settings**")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            hull_fill = st.checkbox(
                "Fill Area",
                value=True,
                help="Fill the convex hull area with color",
                key="bivariate_hull_fill"
            )

        with col2:
            hull_opacity = st.slider(
                "Fill Opacity",
                min_value=0.0,
                max_value=0.5,
                value=0.05,
                step=0.01,
                help="Transparency of hull fill (0 = transparent, 0.5 = opaque)",
                key="bivariate_hull_opacity",
                disabled=not hull_fill  # Disable if no fill
            )

        with col3:
            hull_line_style = st.selectbox(
                "Line Style",
                options=['solid', 'dash', 'dot', 'dashdot'],
                index=1,  # Default to 'dash'
                help="Line style for hull boundary",
                key="bivariate_hull_line_style"
            )

        with col4:
            hull_line_width = st.slider(
                "Line Width",
                min_value=1,
                max_value=5,
                value=2,
                step=1,
                help="Thickness of hull boundary line",
                key="bivariate_hull_line_width"
            )

    else:
        # Default values when hull is disabled
        hull_fill = True
        hull_opacity = 0.05
        hull_line_style = 'dash'
        hull_line_width = 2

    # === AUTO-GENERATE SCATTER PLOT ===
    st.markdown("---")

    # Get current selections from session state
    current_var1 = st.session_state.bivariate_var1
    current_var2 = st.session_state.bivariate_var2
    color_by_val = None if color_by == "None" else color_by
    label_by_val = None if label_by == "None" else label_by

    # Check if both variables are selected and different
    if current_var1 and current_var2 and current_var1 != current_var2:
        try:
            # Prepare custom_variables dict
            custom_vars = None
            if 'custom_variables' in st.session_state:
                custom_vars = st.session_state.custom_variables

            # AUTO-GENERATE SCATTER PLOT
            fig = create_scatter_plot(
                data=data,
                x_var=current_var1,
                y_var=current_var2,
                color_by=color_by_val,
                label_by=label_by_val,
                custom_variables=custom_vars,
                point_size=point_size,
                opacity=opacity,
                show_convex_hull=convex_hull_option,
                hull_fill=hull_fill,
                hull_opacity=hull_opacity,
                hull_line_style=hull_line_style,
                hull_line_width=hull_line_width
            )

            # CENTER THE PLOT HORIZONTALLY
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.plotly_chart(fig, use_container_width=True)

            # SHOW STATISTICS BELOW
            st.markdown("---")

            pair_data = data[[current_var1, current_var2]].dropna()
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
        # Show helpful messages when conditions not met
        if current_var1 and current_var2 and current_var1 == current_var2:
            st.info("ℹ️ Please select **two different variables** for scatter plot")
        elif current_var1 or current_var2:
            st.info("ℹ️ Please select **both variables** for scatter plot")
        else:
            st.info("💡 Select **Variable 1** and **Variable 2** to generate scatter plot")

    # === ADDITIONAL ANALYSIS TABS ===
    st.markdown("---")
    st.markdown("## 📊 Advanced Statistical Analysis")

    tab1, tab2, tab3, tab4 = st.tabs([
        "Correlation Matrix",
        "Covariance Matrix",
        "Pairs Plot",
        "Correlation Summary"
    ])

    with tab1:
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

    with tab2:
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
            st.error(f"❌ Error: {str(e)}")

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

    with tab4:
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
