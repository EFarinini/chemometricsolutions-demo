"""
PCA Process Monitoring Page
Statistical Process Control using PCA with T² and Q statistics
"""

import streamlit as st
import pandas as pd
import numpy as np
import io
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import PCA monitoring module
from pca_utils import PCAMonitor, plot_combined_monitoring_chart


def show():
    """Display the PCA Process Monitoring page"""

    st.markdown("# 📊 PCA Process Monitoring")
    st.markdown("*Statistical Process Control using Multivariate PCA*")

    # Introduction
    with st.expander("ℹ️ About PCA Process Monitoring", expanded=False):
        st.markdown("""
        **PCA-based Statistical Process Control (MSPC)** enables real-time monitoring of multivariate processes.

        **Key Features:**
        - **T² Statistic (Hotelling)**: Detects unusual patterns within the model space
        - **Q Statistic (SPE)**: Detects deviations from the model structure
        - **Multiple Control Limits**: 97.5%, 99.5%, 99.95% confidence levels
        - **Fault Diagnosis**: Variable contribution analysis for root cause identification
        - **Model Persistence**: Save and load trained models for production deployment

        **Typical Workflow:**
        1. **Train**: Build model from normal operating condition (NOC) data
        2. **Test**: Monitor new process data for faults
        3. **Diagnose**: Identify which variables caused detected faults
        4. **Deploy**: Save model for production monitoring
        """)

    # Main tabs
    tab1, tab2, tab3 = st.tabs([
        "🔧 Model Training",
        "🔍 Testing & Monitoring",
        "💾 Model Management"
    ])

    # ===== TAB 1: MODEL TRAINING =====
    with tab1:
        st.markdown("## 🔧 Train Monitoring Model")
        st.markdown("*Build PCA model from normal operating condition (NOC) data*")

        # Data source selection
        st.markdown("### 📊 Training Data Source")

        col1, col2 = st.columns([1, 1])

        with col1:
            data_source = st.radio(
                "Select training data source:",
                ["Use Current Dataset", "Upload Training File"],
                help="Train from currently loaded data or upload a new file"
            )

        train_data = None

        if data_source == "Use Current Dataset":
            if 'current_data' in st.session_state:
                train_data = st.session_state.current_data.copy()
                st.success(f"✅ Using current dataset: **{st.session_state.get('current_dataset', 'Unknown')}**")
                st.info(f"📊 Shape: {train_data.shape[0]} samples × {train_data.shape[1]} variables")
            else:
                st.warning("⚠️ No dataset loaded. Please go to **Data Handling** to load data first, or upload a file below.")

        else:  # Upload Training File
            with col2:
                uploaded_file = st.file_uploader(
                    "Upload training data (CSV or Excel)",
                    type=['csv', 'xlsx', 'xls'],
                    help="Upload file containing normal operating condition data"
                )

                if uploaded_file is not None:
                    try:
                        if uploaded_file.name.endswith('.csv'):
                            train_data = pd.read_csv(uploaded_file)
                        else:
                            train_data = pd.read_excel(uploaded_file)

                        st.success(f"✅ File loaded: **{uploaded_file.name}**")
                        st.info(f"📊 Shape: {train_data.shape[0]} samples × {train_data.shape[1]} variables")
                    except Exception as e:
                        st.error(f"❌ Error loading file: {str(e)}")

        # Show data preview and variable selection
        if train_data is not None:
            st.markdown("### 🎯 Variable Selection")

            # Identify numeric columns
            numeric_cols = train_data.select_dtypes(include=[np.number]).columns.tolist()
            non_numeric_cols = train_data.select_dtypes(exclude=[np.number]).columns.tolist()

            col_sel1, col_sel2 = st.columns(2)

            with col_sel1:
                st.info(f"**Numeric columns**: {len(numeric_cols)}")
                if len(non_numeric_cols) > 0:
                    st.info(f"**Non-numeric columns**: {len(non_numeric_cols)} (will be excluded)")

            with col_sel2:
                use_all_numeric = st.checkbox(
                    "Use all numeric columns",
                    value=True,
                    help="Use all numeric variables for PCA model"
                )

                if not use_all_numeric:
                    selected_vars = st.multiselect(
                        "Select variables for monitoring:",
                        numeric_cols,
                        default=numeric_cols[:min(10, len(numeric_cols))],
                        help="Choose which variables to include in the model"
                    )
                else:
                    selected_vars = numeric_cols

            if len(selected_vars) == 0:
                st.warning("⚠️ Please select at least one variable")
            else:
                # Prepare training matrix
                X_train = train_data[selected_vars].values

                # Data preview
                with st.expander("👁️ Preview Training Data"):
                    st.dataframe(train_data[selected_vars].head(10), use_container_width=True)

                    # Statistics
                    st.markdown("**Basic Statistics:**")
                    stats_df = train_data[selected_vars].describe()
                    st.dataframe(stats_df, use_container_width=True)

                st.markdown("### ⚙️ Model Configuration")

                config_col1, config_col2, config_col3 = st.columns(3)

                with config_col1:
                    n_components = st.number_input(
                        "Number of components:",
                        min_value=1,
                        max_value=min(X_train.shape[0]-1, X_train.shape[1]),
                        value=min(5, X_train.shape[1]),
                        help="Number of principal components to retain"
                    )

                with config_col2:
                    scaling = st.selectbox(
                        "Data scaling:",
                        ["auto", "pareto", "none"],
                        help="auto=standardization, pareto=Pareto scaling, none=no scaling"
                    )

                with config_col3:
                    st.markdown("**Control Limits:**")
                    use_custom_limits = st.checkbox("Custom limits", value=False)

                if use_custom_limits:
                    alpha_97_5 = st.checkbox("97.5% (2.5% false alarm)", value=True)
                    alpha_99_5 = st.checkbox("99.5% (0.5% false alarm)", value=True)
                    alpha_99_95 = st.checkbox("99.95% (0.05% false alarm)", value=True)

                    alpha_levels = []
                    if alpha_97_5:
                        alpha_levels.append(0.975)
                    if alpha_99_5:
                        alpha_levels.append(0.995)
                    if alpha_99_95:
                        alpha_levels.append(0.9995)

                    if len(alpha_levels) == 0:
                        st.warning("Please select at least one confidence level")
                        alpha_levels = [0.975, 0.995, 0.9995]
                else:
                    alpha_levels = [0.975, 0.995, 0.9995]

                # Train button
                st.markdown("---")

                if st.button("🚀 Train Monitoring Model", type="primary", use_container_width=True):
                    with st.spinner("Training PCA monitoring model..."):
                        try:
                            # Create and train monitor
                            monitor = PCAMonitor(
                                n_components=n_components,
                                scaling=scaling,
                                alpha_levels=alpha_levels
                            )

                            monitor.fit(X_train, feature_names=selected_vars)

                            # Store in session state
                            st.session_state.pca_monitor = monitor
                            st.session_state.pca_monitor_vars = selected_vars
                            st.session_state.pca_monitor_trained = True

                            # Get model summary
                            summary = monitor.get_model_summary()

                            # Success message
                            st.success("✅ **Model trained successfully!**")

                            # Display results
                            st.markdown("### 📊 Model Summary")

                            sum_col1, sum_col2, sum_col3, sum_col4 = st.columns(4)

                            with sum_col1:
                                st.metric("Components", summary['n_components'])
                            with sum_col2:
                                st.metric("Variables", summary['n_features'])
                            with sum_col3:
                                st.metric("Training Samples", summary['n_samples_train'])
                            with sum_col4:
                                st.metric("Variance Explained", f"{summary['variance_explained']*100:.1f}%")

                            # Variance per component
                            st.markdown("**Variance Explained per Component:**")
                            var_df = pd.DataFrame({
                                'Component': [f'PC{i+1}' for i in range(len(summary['variance_per_pc']))],
                                'Variance (%)': [v*100 for v in summary['variance_per_pc']],
                                'Cumulative (%)': np.cumsum([v*100 for v in summary['variance_per_pc']])
                            })
                            st.dataframe(var_df, use_container_width=True)

                            # Control limits
                            st.markdown("**Control Limits:**")
                            limits_data = []
                            for alpha in sorted(summary['t2_limits'].keys()):
                                limits_data.append({
                                    'Confidence Level': f"{alpha*100:.2f}%",
                                    'T² Limit': f"{summary['t2_limits'][alpha]:.2f}",
                                    'Q Limit': f"{summary['q_limits'][alpha]:.2f}"
                                })
                            limits_df = pd.DataFrame(limits_data)
                            st.dataframe(limits_df, use_container_width=True)

                            st.info("🎯 Model is ready! Go to **Testing & Monitoring** tab to test new data.")

                        except Exception as e:
                            st.error(f"❌ Error training model: {str(e)}")
                            import traceback
                            st.code(traceback.format_exc())

    # ===== TAB 2: TESTING & MONITORING =====
    with tab2:
        st.markdown("## 🔍 Testing & Monitoring")
        st.markdown("*Test new data and detect faults*")

        # Check if model is trained
        if 'pca_monitor_trained' not in st.session_state or not st.session_state.pca_monitor_trained:
            st.warning("⚠️ **No model trained yet.** Please train a model in the **Model Training** tab first.")
        else:
            monitor = st.session_state.pca_monitor
            model_vars = st.session_state.pca_monitor_vars

            st.success(f"✅ **Model loaded** ({len(model_vars)} variables, {monitor.pca_model_.n_components_} components)")

            # Test data source
            st.markdown("### 📊 Test Data Source")

            test_col1, test_col2 = st.columns([1, 1])

            with test_col1:
                test_source = st.radio(
                    "Select test data source:",
                    ["Use Current Dataset", "Upload Test File"],
                    help="Test data from current dataset or upload new file"
                )

            test_data = None

            if test_source == "Use Current Dataset":
                if 'current_data' in st.session_state:
                    test_data = st.session_state.current_data.copy()
                    st.success(f"✅ Using current dataset: **{st.session_state.get('current_dataset', 'Unknown')}**")
                else:
                    st.warning("⚠️ No dataset loaded.")
            else:
                with test_col2:
                    test_uploaded = st.file_uploader(
                        "Upload test data (CSV or Excel)",
                        type=['csv', 'xlsx', 'xls'],
                        key="test_file_uploader"
                    )

                    if test_uploaded is not None:
                        try:
                            if test_uploaded.name.endswith('.csv'):
                                test_data = pd.read_csv(test_uploaded)
                            else:
                                test_data = pd.read_excel(test_uploaded)

                            st.success(f"✅ File loaded: **{test_uploaded.name}**")
                        except Exception as e:
                            st.error(f"❌ Error loading file: {str(e)}")

            # Test the data
            if test_data is not None:
                # Check if required variables are present
                missing_vars = [v for v in model_vars if v not in test_data.columns]

                if len(missing_vars) > 0:
                    st.error(f"❌ **Missing variables in test data**: {missing_vars}")
                    st.info("Test data must contain the same variables used for training.")
                else:
                    X_test = test_data[model_vars].values

                    st.info(f"📊 Test data: {X_test.shape[0]} samples × {X_test.shape[1]} variables")

                    # Test options
                    test_opt_col1, test_opt_col2 = st.columns(2)

                    with test_opt_col1:
                        calc_contributions = st.checkbox(
                            "Calculate contributions",
                            value=True,
                            help="Calculate variable contributions for fault diagnosis"
                        )

                    with test_opt_col2:
                        if 'timestamp' in test_data.columns or 'Timestamp' in test_data.columns:
                            use_timestamps = st.checkbox("Use timestamps for labels", value=True)
                            timestamp_col = 'timestamp' if 'timestamp' in test_data.columns else 'Timestamp'
                        else:
                            use_timestamps = False

                    # Test button
                    if st.button("🔍 Test Data", type="primary", use_container_width=True):
                        with st.spinner("Testing data..."):
                            try:
                                # Predict
                                results = monitor.predict(X_test, return_contributions=calc_contributions)

                                # Store results
                                st.session_state.pca_monitor_results = results
                                st.session_state.pca_monitor_test_data = test_data

                                # Summary metrics
                                n_faults = results['faults'].sum()
                                fault_rate = (n_faults / len(results['faults'])) * 100

                                st.success("✅ **Testing complete!**")

                                st.markdown("### 📊 Fault Detection Summary")

                                met_col1, met_col2, met_col3, met_col4 = st.columns(4)

                                with met_col1:
                                    st.metric("Total Samples", len(results['faults']))
                                with met_col2:
                                    st.metric("Faults Detected", n_faults)
                                with met_col3:
                                    st.metric("Fault Rate", f"{fault_rate:.1f}%")
                                with met_col4:
                                    # Count by type
                                    fault_types = pd.Series(results['fault_type']).value_counts()
                                    most_common = fault_types.index[0] if len(fault_types) > 0 else 'none'
                                    st.metric("Most Common", most_common)

                                # Fault type distribution
                                st.markdown("**Fault Type Distribution:**")
                                fault_dist = pd.Series(results['fault_type']).value_counts().reset_index()
                                fault_dist.columns = ['Fault Type', 'Count']
                                fault_dist['Percentage'] = (fault_dist['Count'] / len(results['fault_type']) * 100).round(1)
                                st.dataframe(fault_dist, use_container_width=True)

                                # Monitoring chart
                                st.markdown("### 📈 Monitoring Charts")

                                # Create sample labels
                                if use_timestamps:
                                    sample_labels = test_data[timestamp_col].astype(str).tolist()
                                else:
                                    sample_labels = [f"Sample {i+1}" for i in range(len(results['t2']))]

                                fig = monitor.plot_monitoring_chart(
                                    results,
                                    sample_labels=sample_labels,
                                    title="PCA Process Monitoring Chart"
                                )

                                st.plotly_chart(fig, use_container_width=True)

                                # T² vs Q chart
                                st.markdown("### 🎯 T² vs Q Chart")

                                fig_combined = plot_combined_monitoring_chart(
                                    results,
                                    results['t2_limits'],
                                    results['q_limits'],
                                    sample_labels=sample_labels,
                                    title="T² vs Q - Fault Detection Regions"
                                )

                                st.plotly_chart(fig_combined, use_container_width=True)

                                # Fault details table
                                st.markdown("### 📋 Fault Details")

                                fault_summary = monitor.get_fault_summary(results)

                                # Filter options
                                show_all = st.checkbox("Show all samples", value=False)

                                if show_all:
                                    display_df = fault_summary
                                else:
                                    display_df = fault_summary[fault_summary['Fault_Detected']]

                                if len(display_df) == 0:
                                    st.info("No faults detected!")
                                else:
                                    st.dataframe(display_df, use_container_width=True)

                                    # Download option
                                    csv = display_df.to_csv(index=False)
                                    st.download_button(
                                        label="📥 Download Fault Summary (CSV)",
                                        data=csv,
                                        file_name="fault_summary.csv",
                                        mime="text/csv"
                                    )

                                # Contribution analysis
                                if calc_contributions and n_faults > 0:
                                    st.markdown("### 🔬 Fault Diagnosis - Contribution Analysis")

                                    faulty_indices = np.where(results['faults'])[0]

                                    contrib_col1, contrib_col2, contrib_col3 = st.columns(3)

                                    with contrib_col1:
                                        selected_fault_idx = st.selectbox(
                                            "Select sample to analyze:",
                                            faulty_indices,
                                            format_func=lambda x: f"Sample {x+1} ({results['fault_type'][x]})"
                                        )

                                    with contrib_col2:
                                        contrib_stat = st.selectbox(
                                            "Statistic:",
                                            ["q", "t2"],
                                            format_func=lambda x: "Q (SPE)" if x == "q" else "T² (Hotelling)"
                                        )

                                    with contrib_col3:
                                        top_n = st.number_input(
                                            "Top N contributors:",
                                            min_value=5,
                                            max_value=min(30, len(model_vars)),
                                            value=min(15, len(model_vars))
                                        )

                                    # Show statistics for selected sample
                                    st.markdown(f"**Sample {selected_fault_idx+1} Statistics:**")
                                    stat_col1, stat_col2, stat_col3 = st.columns(3)

                                    primary_alpha = min(results['t2_limits'].keys())

                                    with stat_col1:
                                        st.metric(
                                            "T² Statistic",
                                            f"{results['t2'][selected_fault_idx]:.2f}",
                                            delta=f"{results['t2'][selected_fault_idx] - results['t2_limits'][primary_alpha]:.2f} above limit" if results['t2'][selected_fault_idx] > results['t2_limits'][primary_alpha] else None
                                        )

                                    with stat_col2:
                                        st.metric(
                                            "Q Statistic",
                                            f"{results['q'][selected_fault_idx]:.2f}",
                                            delta=f"{results['q'][selected_fault_idx] - results['q_limits'][primary_alpha]:.2f} above limit" if results['q'][selected_fault_idx] > results['q_limits'][primary_alpha] else None
                                        )

                                    with stat_col3:
                                        st.metric("Fault Type", results['fault_type'][selected_fault_idx])

                                    # Contribution plot
                                    fig_contrib = monitor.plot_contribution_chart(
                                        results,
                                        sample_idx=selected_fault_idx,
                                        statistic=contrib_stat,
                                        top_n=top_n
                                    )

                                    st.plotly_chart(fig_contrib, use_container_width=True)

                                    # Top contributors table
                                    if contrib_stat == 'q':
                                        contributions = results['contributions_q'][selected_fault_idx]
                                        stat_name = 'Q'
                                    else:
                                        contributions = results['contributions_t2'][selected_fault_idx]
                                        stat_name = 'T²'

                                    top_indices = np.argsort(np.abs(contributions))[-top_n:][::-1]

                                    contrib_table = pd.DataFrame({
                                        'Variable': [model_vars[i] for i in top_indices],
                                        f'{stat_name} Contribution': contributions[top_indices],
                                        'Absolute': np.abs(contributions[top_indices])
                                    })

                                    st.markdown(f"**Top {top_n} Contributors:**")
                                    st.dataframe(contrib_table, use_container_width=True)

                            except Exception as e:
                                st.error(f"❌ Error testing data: {str(e)}")
                                import traceback
                                st.code(traceback.format_exc())

    # ===== TAB 3: MODEL MANAGEMENT =====
    with tab3:
        st.markdown("## 💾 Model Management")
        st.markdown("*Save and load monitoring models*")

        mgmt_col1, mgmt_col2 = st.columns(2)

        with mgmt_col1:
            st.markdown("### 💾 Save Model")

            if 'pca_monitor_trained' not in st.session_state or not st.session_state.pca_monitor_trained:
                st.info("ℹ️ No model trained yet. Train a model first.")
            else:
                monitor = st.session_state.pca_monitor

                # Model info
                summary = monitor.get_model_summary()
                st.info(f"""
                **Current Model:**
                - Components: {summary['n_components']}
                - Variables: {summary['n_features']}
                - Variance: {summary['variance_explained']*100:.1f}%
                """)

                model_name = st.text_input(
                    "Model filename:",
                    value="pca_monitor_model.pkl",
                    help="Enter filename for saved model"
                )

                if st.button("💾 Save Model", type="primary", use_container_width=True):
                    try:
                        # Save to bytes for download
                        import pickle

                        model_data = {
                            'monitor': monitor,
                            'variables': st.session_state.pca_monitor_vars
                        }

                        buffer = io.BytesIO()
                        pickle.dump(model_data, buffer)
                        buffer.seek(0)

                        st.download_button(
                            label="📥 Download Model File",
                            data=buffer,
                            file_name=model_name,
                            mime="application/octet-stream",
                            use_container_width=True
                        )

                        st.success("✅ Model ready for download!")

                    except Exception as e:
                        st.error(f"❌ Error saving model: {str(e)}")

        with mgmt_col2:
            st.markdown("### 📥 Load Model")

            uploaded_model = st.file_uploader(
                "Upload model file (.pkl)",
                type=['pkl'],
                help="Load a previously saved monitoring model"
            )

            if uploaded_model is not None:
                if st.button("📥 Load Model", type="primary", use_container_width=True):
                    try:
                        import pickle

                        model_data = pickle.load(uploaded_model)

                        # Extract monitor and variables
                        if isinstance(model_data, dict):
                            monitor = model_data['monitor']
                            variables = model_data['variables']
                        else:
                            # Legacy format - just the monitor
                            monitor = model_data
                            variables = monitor.feature_names_

                        # Store in session state
                        st.session_state.pca_monitor = monitor
                        st.session_state.pca_monitor_vars = variables
                        st.session_state.pca_monitor_trained = True

                        # Show model info
                        summary = monitor.get_model_summary()

                        st.success("✅ **Model loaded successfully!**")
                        st.info(f"""
                        **Model Information:**
                        - Components: {summary['n_components']}
                        - Variables: {summary['n_features']}
                        - Variance: {summary['variance_explained']*100:.1f}%
                        - Training samples: {summary['n_samples_train']}
                        """)

                        st.balloons()

                    except Exception as e:
                        st.error(f"❌ Error loading model: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())

        # Model comparison section
        st.markdown("---")
        st.markdown("### 📊 Model Information")

        if 'pca_monitor_trained' in st.session_state and st.session_state.pca_monitor_trained:
            monitor = st.session_state.pca_monitor
            summary = monitor.get_model_summary()

            info_col1, info_col2 = st.columns(2)

            with info_col1:
                st.markdown("**Model Configuration:**")
                config_df = pd.DataFrame({
                    'Parameter': ['Components', 'Variables', 'Training Samples', 'Scaling Method', 'Variance Explained'],
                    'Value': [
                        summary['n_components'],
                        summary['n_features'],
                        summary['n_samples_train'],
                        monitor.scaling,
                        f"{summary['variance_explained']*100:.2f}%"
                    ]
                })
                st.dataframe(config_df, use_container_width=True, hide_index=True)

            with info_col2:
                st.markdown("**Control Limits:**")
                limits_data = []
                for alpha in sorted(summary['t2_limits'].keys()):
                    limits_data.append({
                        'Confidence': f"{alpha*100:.2f}%",
                        'T² Limit': f"{summary['t2_limits'][alpha]:.3f}",
                        'Q Limit': f"{summary['q_limits'][alpha]:.3f}"
                    })
                limits_df = pd.DataFrame(limits_data)
                st.dataframe(limits_df, use_container_width=True, hide_index=True)

            # Variable list
            with st.expander("📋 Model Variables"):
                var_df = pd.DataFrame({
                    'Index': range(1, len(st.session_state.pca_monitor_vars) + 1),
                    'Variable': st.session_state.pca_monitor_vars
                })
                st.dataframe(var_df, use_container_width=True, hide_index=True)
        else:
            st.info("ℹ️ No model loaded. Train or load a model to see details.")


if __name__ == "__main__":
    show()
