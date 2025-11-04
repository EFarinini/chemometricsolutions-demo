"""
Classification Module - Streamlit Interface
============================================

Interactive page for supervised classification methods:
- LDA (Linear Discriminant Analysis)
- QDA (Quadratic Discriminant Analysis)
- kNN (k-Nearest Neighbors)
- SIMCA (Soft Independent Modeling of Class Analogy)
- UNEQ (Unequal Class Dispersions)

Author: Dr. Emanuele Farinini, PhD - ChemometricSolutions
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
import plotly.graph_objects as go

# Import classification utilities
try:
    from classification_utils import (
        # Preprocessing
        validate_classification_data,
        prepare_training_test,

        # Classifiers
        fit_lda, predict_lda, cross_validate_lda, predict_lda_detailed,
        fit_qda, predict_qda, cross_validate_qda, predict_qda_detailed,
        fit_knn, predict_knn, cross_validate_knn, predict_knn_detailed,
        fit_simca, predict_simca, predict_simca_detailed, cross_validate_simca,
        fit_uneq, predict_uneq, predict_uneq_detailed, cross_validate_uneq,

        # Diagnostics
        calculate_classification_metrics,
        compute_classification_metrics,
        calculate_simca_uneq_metrics,
        find_best_k,
        compare_models,
        get_misclassified_samples,
        cross_validate_classifier,

        # Plotting
        plot_confusion_matrix,
        plot_classification_metrics,
        plot_coomans,
        plot_distance_distributions,
        plot_knn_performance,
        plot_model_comparison,
        plot_decision_boundary_2d,
        plot_class_separation,
        plot_classification_report,
        calculate_distance_matrix,

        # Constants
        AVAILABLE_DISTANCE_METRICS,
        get_available_classifiers
    )
    CLASSIFICATION_AVAILABLE = True
except ImportError as e:
    CLASSIFICATION_AVAILABLE = False

# Import workspace utilities
try:
    from workspace_utils import get_workspace_datasets
    WORKSPACE_AVAILABLE = True
except ImportError:
    WORKSPACE_AVAILABLE = False


def show():
    """Main function to display the classification page"""

    if not CLASSIFICATION_AVAILABLE:
        st.error("❌ Classification utilities not available. Please check installation.")
        st.stop()

    st.markdown("# 🎲 Classification Analysis")
    st.markdown("*Supervised classification with multiple methods*")

    # Initialize session state
    if 'classification_results' not in st.session_state:
        st.session_state.classification_results = {}
    if 'selected_classifier' not in st.session_state:
        st.session_state.selected_classifier = 'LDA'
    if 'trained_model' not in st.session_state:
        st.session_state.trained_model = None

    # === SIDEBAR: Dataset selection (KEEP UNCHANGED) ===
    with st.sidebar:
        st.header("📁 Data Selection")

        if not WORKSPACE_AVAILABLE:
            st.error("❌ Workspace utilities not available")
            st.stop()

        try:
            datasets = get_workspace_datasets()
            if not datasets:
                st.warning("⚠️ **No datasets available in workspace.**")
                st.info("💡 Load data in the **Data Handling** page first")
                st.stop()

            dataset_names = list(datasets.keys())
            selected_dataset_name = st.selectbox(
                "Select Dataset",
                dataset_names,
                key="classification_dataset",
                help="Choose a dataset from your workspace"
            )
            data = datasets[selected_dataset_name]

            all_columns = data.columns.tolist()
            class_column = st.selectbox(
                "📍 Classification Target (Class Variable)",
                all_columns,
                help="Select ONLY ONE column with class labels"
            )

        except Exception as e:
            st.error(f"❌ Error loading datasets: {str(e)}")
            import traceback
            st.error(traceback.format_exc())
            st.stop()

    # Check for empty dataset
    if len(data) == 0:
        st.error("❌ Dataset is empty - no samples available!")
        return

    if len(data.columns) == 0:
        st.error("❌ Dataset has no columns!")
        return

    # === CREATE TABS ===
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Setup & Configuration",
        "🎲 Classification Analysis",
        "🏆 Model Comparison",
        "📋 Test & Validation"
    ])

    # ========== TAB 1: SETUP & CONFIGURATION ==========
    with tab1:
        st.markdown("## 📊 Dataset Overview & Configuration")

        # --- Section 1: Dataset Overview ---
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Samples", len(data))
        with col2:
            st.metric("Total Variables", len(data.columns))
        with col3:
            unique_classes = data[class_column].unique()
            st.metric("Classes", len(unique_classes))
        with col4:
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            st.metric("Numeric Variables", len(numeric_cols))

        # Preview
        with st.expander("👀 Preview First 5 Rows", expanded=False):
            st.dataframe(data.head(5), use_container_width=True)

        with st.expander("📋 Dataset Detailed Overview", expanded=False):
            st.markdown("### Full Dataset")
            st.dataframe(data, use_container_width=True, height=300)

            st.markdown("### Summary Statistics")
            numeric_data = data.select_dtypes(include=[np.number])
            if len(numeric_data.columns) > 0:
                summary_stats = numeric_data.describe().T
                summary_stats['missing'] = numeric_data.isnull().sum()
                summary_stats['missing_%'] = (numeric_data.isnull().sum() / len(data) * 100).round(2)
                st.dataframe(summary_stats, use_container_width=True)
            else:
                st.info("No numeric columns to summarize")

            st.markdown("### Missing Values Check")
            missing_summary = pd.DataFrame({
                'Column': data.columns,
                'Missing Count': data.isnull().sum().values,
                'Missing %': (data.isnull().sum() / len(data) * 100).round(2).values,
                'Data Type': data.dtypes.values
            })
            missing_summary = missing_summary[missing_summary['Missing Count'] > 0]

            if len(missing_summary) > 0:
                st.warning(f"⚠️ **{len(missing_summary)} columns have missing values:**")
                st.dataframe(missing_summary, use_container_width=True, hide_index=True)
            else:
                st.success("✅ No missing values detected in dataset!")

        st.divider()

        # --- Section 2: Variable & Sample Selection ---
        st.markdown("## 🎯 Variable & Sample Selection")

        if len(numeric_cols) > 0:
            first_numeric_pos = all_columns.index(numeric_cols[0]) + 1
            last_numeric_pos = all_columns.index(numeric_cols[-1]) + 1
        else:
            first_numeric_pos = 1
            last_numeric_pos = 1

        st.info(f"Dataset: {len(all_columns)} total columns, {len(numeric_cols)} numeric (positions {first_numeric_pos}-{last_numeric_pos})")

        st.markdown("#### 📊 Column Selection (Variables)")

        col1, col2 = st.columns(2)
        with col1:
            first_var = st.number_input(
                "First column (1-based):",
                min_value=1,
                max_value=len(all_columns),
                value=first_numeric_pos,
                key="tab1_first_var"
            )
        with col2:
            last_var = st.number_input(
                "Last column (1-based):",
                min_value=first_var,
                max_value=len(all_columns),
                value=last_numeric_pos,
                key="tab1_last_var"
            )

        st.markdown("#### 🎯 Row Selection (Objects/Samples)")

        col1, col2 = st.columns(2)
        with col1:
            first_sample = st.number_input(
                "First sample (1-based):",
                min_value=1,
                max_value=len(data),
                value=1,
                key="tab1_first_sample"
            )
        with col2:
            last_sample = st.number_input(
                "Last sample (1-based):",
                min_value=first_sample,
                max_value=len(data),
                value=len(data),
                key="tab1_last_sample"
            )

        # Extract selected variables and samples
        selected_cols = all_columns[first_var-1:last_var]
        selected_sample_indices = list(range(first_sample-1, last_sample))

        # Identify categorical columns
        categorical_cols = [col for col in selected_cols
                          if not pd.api.types.is_numeric_dtype(data[col])]

        if len(categorical_cols) == 0:
            st.error("❌ No categorical columns found in selected range. Classification requires a non-numeric target column with class labels.")
            st.info("💡 **Tip:** Select a column range that includes categorical columns (e.g., 'Category', 'Type', 'Class')")
            return

        st.markdown("#### 📍 Target Variable (Class) Selection")

        selected_class_var = st.selectbox(
            "Select Target Column (Classification Class):",
            categorical_cols,
            index=categorical_cols.index(class_column) if class_column in categorical_cols else 0,
            help="Choose the column containing class/category labels (must be categorical, not numeric)",
            key="tab1_class_variable"
        )

        # Feature columns: ONLY numeric columns (excluding class variable)
        selected_vars = [col for col in selected_cols
                        if col != selected_class_var
                        and pd.api.types.is_numeric_dtype(data[col])]

        if len(selected_vars) == 0:
            st.error("❌ No numeric feature columns available! Classification requires numeric features.")
            st.info("💡 **Tip:** Select a column range that includes numeric columns for features")
            return

        # Extract data
        X_data = data.iloc[selected_sample_indices][selected_vars]
        y_labels = data.iloc[selected_sample_indices][selected_class_var].values
        classes = np.unique(y_labels)

        st.success(f"📊 **Classes found:** {', '.join(map(str, classes))}")
        st.info(f"✅ **Configuration:** {len(classes)} classes, {len(selected_vars)} features, {len(selected_sample_indices)} samples")

        st.divider()

        # --- Section 3: Preprocessing Configuration ---
        st.markdown("## ⚙️ Preprocessing Configuration")

        col1, col2 = st.columns(2)
        with col1:
            scaling_method = st.selectbox(
                "Scaling Method",
                options=['autoscale', 'center', 'scale', 'none'],
                index=0,
                key="tab1_scaling"
            )
        with col2:
            st.write("")  # Spacer

        col1, col2, col3 = st.columns(3)
        with col1:
            confidence_level = st.slider(
                "Confidence Level (SIMCA/UNEQ)",
                min_value=0.80,
                max_value=0.99,
                value=0.95,
                step=0.01,
                key="tab1_confidence"
            )
        with col2:
            k_value = st.slider(
                "k (kNN)",
                min_value=1,
                max_value=15,
                value=5,
                key="tab1_k"
            )
        with col3:
            n_pcs = st.slider(
                "PCs (SIMCA/UNEQ)",
                min_value=1,
                max_value=min(10, len(selected_vars)-1) if len(selected_vars) > 1 else 1,
                value=min(3, len(selected_vars)-1) if len(selected_vars) > 1 else 1,
                key="tab1_pcs"
            )

        # Store in session state for use in other tabs
        st.session_state['tab1_data'] = {
            'X_data': X_data,
            'y_labels': y_labels,
            'classes': classes,
            'X_cols': selected_vars,
            'scaling_method': scaling_method,
            'confidence_level': confidence_level,
            'k_value': k_value,
            'n_pcs': n_pcs
        }

        st.divider()

        # --- Section 4: Model Training ---
        st.markdown("## 🎯 Model Training")

        # Prepare data with scaling
        prep_data = prepare_training_test(
            X_data.values,
            y_labels,
            scaling_method=scaling_method
        )

        X_scaled = prep_data['X_train']

        # Store scaled data for other tabs
        st.session_state['X_scaled'] = X_scaled

        # Classifier selection
        col1, col2 = st.columns([1, 2])
        with col1:
            selected_classifier = st.selectbox(
                "Classification Method:",
                options=['LDA', 'QDA', 'kNN', 'SIMCA', 'UNEQ'],
                key="tab1_classifier"
            )

        # Store selected classifier
        st.session_state.selected_classifier = selected_classifier

        # Classifier-specific parameters
        if selected_classifier == 'kNN':
            with col2:
                metric = st.selectbox(
                    "Distance Metric:",
                    AVAILABLE_DISTANCE_METRICS,
                    key="tab1_knn_metric"
                )

        # === PCA PREPROCESSING OPTION (LDA, QDA, kNN only) ===
        if selected_classifier in ['LDA', 'QDA', 'kNN']:
            st.divider()
            st.markdown("#### 🔍 Preprocessing Options")

            use_pca_preprocessing = st.checkbox(
                "Use PCA preprocessing",
                value=False,
                key="tab1_use_pca",
                help="Apply PCA dimensionality reduction before classification. Useful for high-dimensional data to reduce noise and improve performance."
            )

            if use_pca_preprocessing:
                st.info("📊 Data will be reduced to 3 components via PCA before classification")
        else:
            # SIMCA and UNEQ have PCA built-in, no need for preprocessing
            use_pca_preprocessing = False
            if selected_classifier in ['SIMCA', 'UNEQ']:
                st.caption("ℹ️ PCA preprocessing not needed - this classifier includes PCA internally")

        # --- Train Model Button ---
        if st.button("🚀 Train Model", type="primary", use_container_width=True, key="train_btn_tab1"):
            import time

            train_start = time.time()

            with st.spinner(f"Training {selected_classifier} model..."):
                try:
                    if selected_classifier == 'LDA':
                        model = fit_lda(X_scaled, y_labels)
                        params = {'use_pca_preprocessing': use_pca_preprocessing}
                    elif selected_classifier == 'QDA':
                        model = fit_qda(X_scaled, y_labels)
                        params = {'use_pca_preprocessing': use_pca_preprocessing}
                    elif selected_classifier == 'kNN':
                        model = fit_knn(X_scaled, y_labels, metric=metric)
                        params = {'k': k_value, 'metric': metric, 'use_pca_preprocessing': use_pca_preprocessing}
                    elif selected_classifier == 'SIMCA':
                        model = fit_simca(X_scaled, y_labels, n_pcs, confidence_level)
                        params = {'n_pcs_per_class': n_pcs, 'confidence_level': confidence_level}
                    elif selected_classifier == 'UNEQ':
                        model = fit_uneq(X_scaled, y_labels, n_pcs, confidence_level, use_pca=False)
                        params = {'n_components': n_pcs, 'confidence_level': confidence_level, 'use_pca': False}

                    train_time = time.time() - train_start

                    # Store trained model with comprehensive parameters
                    st.session_state.trained_model = {
                        'name': selected_classifier,
                        'model': model,
                        'training_time': train_time,
                        'n_features': X_scaled.shape[1],
                        'n_samples': X_scaled.shape[0],
                        'classes': classes,
                        'parameters': {
                            'scaling': scaling_method,
                            'k': k_value if selected_classifier == 'kNN' else None,
                            'metric': metric if selected_classifier == 'kNN' else None,
                            'n_pcs': n_pcs if selected_classifier in ['SIMCA', 'UNEQ'] else None,
                            'confidence_level': confidence_level if selected_classifier in ['SIMCA', 'UNEQ'] else None,
                            'use_pca': use_pca_preprocessing if selected_classifier in ['LDA', 'QDA', 'kNN'] else None
                        }
                    }

                    st.success(f"✅ {selected_classifier} model trained in {train_time:.3f}s")
                    st.rerun()

                except Exception as e:
                    st.error(f"❌ Training failed: {str(e)}")
                    import traceback
                    st.error(traceback.format_exc())

        # Display trained model info
        if st.session_state.trained_model is not None:
            st.divider()
            st.success(f"✅ **Model Trained:** {st.session_state.trained_model['name']}")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Training Time", f"{st.session_state.trained_model['training_time']:.3f}s")
            with col2:
                st.metric("Features", st.session_state.trained_model['n_features'])
            with col3:
                st.metric("Training Samples", st.session_state.trained_model['n_samples'])

            with st.expander("📋 Model Summary", expanded=False):
                model_summary = {
                    'Classifier': st.session_state.trained_model['name'],
                    'Training Time (s)': round(st.session_state.trained_model['training_time'], 3),
                    'Classes': st.session_state.trained_model['classes'].tolist(),
                    'n_features': st.session_state.trained_model['n_features'],
                    'n_samples': st.session_state.trained_model['n_samples'],
                    'parameters': st.session_state.trained_model['parameters']
                }
                st.json(model_summary)

            st.info("ℹ️ **Next Step:** Go to **Tab 2 (Classification Analysis)** to see cross-validation results and model evaluation.")

    # ========== TAB 2: CLASSIFICATION ANALYSIS ==========
    with tab2:
        st.markdown("## 🎲 Classification Analysis - Model Evaluation")

        if st.session_state.trained_model is None:
            st.warning("⚠️ Train a model first in **Tab 1: Setup & Configuration**")
            st.info("💡 Go to Tab 1, configure your dataset, select a classifier, and click 'Train Model'")
            return

        # Get stored data
        tab1_data = st.session_state.get('tab1_data', {})
        if not tab1_data:
            st.error("❌ No configuration data found. Please return to Tab 1 and configure the analysis.")
            return

        X_data = tab1_data['X_data']
        y_labels = tab1_data['y_labels']
        classes = tab1_data['classes']
        X_scaled = st.session_state.get('X_scaled')

        trained = st.session_state.trained_model

        st.info(f"🎯 **Current Model:** {trained['name']} | **Training Samples:** {trained['n_samples']} | **Features:** {trained['n_features']}")

        st.divider()

        # --- Cross-Validation Section ---
        st.markdown("### ✅ Cross-Validation Evaluation")

        col1, col2 = st.columns([2, 1])
        with col1:
            n_folds = st.number_input(
                "Number of Folds",
                min_value=2,
                max_value=10,
                value=5,
                key="cv_n_folds_tab2"
            )
        with col2:
            random_seed = st.number_input(
                "Random Seed",
                min_value=0,
                max_value=9999,
                value=42,
                key="cv_random_seed_tab2"
            )

        if st.button("🔄 Run Cross-Validation", type="primary", use_container_width=True, key="run_cv_btn_tab2"):
            import time

            # Validate data consistency before running CV
            if len(y_labels) != len(X_scaled):
                st.error(f"❌ Data inconsistency detected: {len(y_labels)} labels vs {len(X_scaled)} samples")
                st.warning("⚠️ Please return to Tab 1 and retrain the model to ensure data consistency.")
                st.stop()

            cv_start = time.time()

            with st.spinner(f"Running {n_folds}-fold cross-validation..."):
                try:
                    # Prepare CV parameters based on classifier type
                    if trained['name'] == 'LDA':
                        cv_results = cross_validate_classifier(
                            X_scaled, y_labels,
                            classifier_type='lda',
                            n_folds=n_folds,
                            classifier_params={
                                'use_pca': trained['parameters'].get('use_pca', False)
                            },
                            random_state=random_seed
                        )
                    elif trained['name'] == 'QDA':
                        cv_results = cross_validate_classifier(
                            X_scaled, y_labels,
                            classifier_type='qda',
                            n_folds=n_folds,
                            classifier_params={
                                'use_pca': trained['parameters'].get('use_pca', False)
                            },
                            random_state=random_seed
                        )
                    elif trained['name'] == 'kNN':
                        cv_results = cross_validate_classifier(
                            X_scaled, y_labels,
                            classifier_type='knn',
                            n_folds=n_folds,
                            classifier_params={
                                'k': trained['parameters']['k'],
                                'metric': trained['parameters']['metric'],
                                'use_pca': trained['parameters'].get('use_pca', False)
                            },
                            random_state=random_seed
                        )
                    elif trained['name'] == 'SIMCA':
                        cv_results = cross_validate_classifier(
                            X_scaled, y_labels,
                            classifier_type='simca',
                            n_folds=n_folds,
                            classifier_params={
                                'n_components': tab1_data.get('n_pcs', 3),
                                'confidence_level': tab1_data.get('confidence_level', 0.95)
                            },
                            random_state=random_seed
                        )
                    elif trained['name'] == 'UNEQ':
                        cv_results = cross_validate_classifier(
                            X_scaled, y_labels,
                            classifier_type='uneq',
                            n_folds=n_folds,
                            classifier_params={
                                'n_components': tab1_data.get('n_pcs', 3),
                                'confidence_level': tab1_data.get('confidence_level', 0.95),
                                'use_pca': False
                            },
                            random_state=random_seed
                        )

                    cv_time = time.time() - cv_start
                    st.session_state['cv_results'] = cv_results
                    st.session_state['cv_time'] = cv_time

                    st.success(f"✅ Cross-validation completed in {cv_time:.2f}s!")
                    st.rerun()

                except Exception as e:
                    st.error(f"❌ Cross-validation failed: {str(e)}")
                    import traceback
                    st.error(traceback.format_exc())

        # Display CV results
        if 'cv_results' in st.session_state:
            cv_res = st.session_state['cv_results']
            y_pred_cv = cv_res['cv_predictions']

            # Validate that y_labels and y_pred_cv have the same length
            if len(y_labels) != len(y_pred_cv):
                st.error(f"❌ Data mismatch detected: Training labels ({len(y_labels)} samples) don't match CV predictions ({len(y_pred_cv)} samples)")
                st.warning("This usually happens when the model was trained with different data. Please retrain the model in Tab 1.")
                st.stop()

            cv_metrics = compute_classification_metrics(y_labels, y_pred_cv, classes)

            st.markdown("### 📊 Cross-Validation Results")

            # CV metrics - Row 1: Performance Metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    "CV Accuracy",
                    f"{cv_metrics['accuracy']:.1f}%",
                    delta=f"{cv_metrics['accuracy'] - 50:.1f}%" if cv_metrics['accuracy'] != 50 else None
                )
            with col2:
                st.metric("Avg Sensitivity", f"{cv_metrics['average_sensitivity']:.1f}%")
            with col3:
                st.metric("Avg Specificity", f"{cv_metrics['average_specificity']:.1f}%")

            # CV metrics - Row 2: Efficiency Metrics
            col4, col5, col6 = st.columns(3)
            with col4:
                cv_time = st.session_state.get('cv_time', 0)
                st.metric("CV Time", f"{cv_time:.2f}s")
            with col5:
                pred_time_avg = cv_time / len(y_pred_cv) if len(y_pred_cv) > 0 else 0
                st.metric("Prediction Time (Avg)", f"{pred_time_avg:.4f}s/sample")
            with col6:
                n_features = trained['n_features']
                n_classes = len(classes)
                st.metric("Model Complexity", f"{n_features} features × {n_classes} classes")

            st.divider()

            # CV Confusion Matrix
            st.markdown("#### Confusion Matrix (CV Predictions)")
            fig_cm_cv = plot_confusion_matrix(
                cv_metrics['confusion_matrix'],
                classes.tolist(),
                title=f"Cross-Validation Confusion Matrix - {trained['name']}"
            )
            st.plotly_chart(fig_cm_cv, use_container_width=True)

            # Coomans plot for SIMCA/UNEQ (with debug diagnostics and class selection)
            st.divider()
            st.markdown("#### 📍 Coomans Plot (CV Diagnostics - 2-Class Comparison)")

            # Debug info: Show model and class information
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Model Type", trained['name'])
            with col2:
                st.metric("Number of Classes", len(classes))
            with col3:
                classes_str = ', '.join([str(c) for c in classes])
                st.metric("Classes", classes_str)

            # Check 1: Classifier type
            if trained['name'] not in ['SIMCA', 'UNEQ']:
                st.info(f"ℹ️ Coomans plot is only available for SIMCA and UNEQ classifiers. Current model: {trained['name']}")
                st.caption("Coomans plots visualize distances to two class models, which is specific to SIMCA and UNEQ methods.")
            # Check 2: Need at least 2 classes
            elif len(classes) < 2:
                st.warning(f"⚠️ Coomans plot requires at least 2 classes. Current classes: {classes_str} ({len(classes)} class)")
                st.caption("Coomans plot is a two-dimensional visualization showing distances to two class models.")
            # Both checks passed: allow class selection and render plot
            else:
                # Class selection interface
                if len(classes) == 2:
                    st.info("Coomans plot showing distance patterns for the 2 classes in your dataset")
                    selected_class_1 = classes[0]
                    selected_class_2 = classes[1]
                else:
                    st.info(f"Coomans plot requires 2 classes. You have {len(classes)} classes. Select which two to compare:")
                    col1, col2 = st.columns(2)
                    with col1:
                        selected_class_1 = st.selectbox(
                            "Class 1",
                            options=classes.tolist(),
                            index=0,
                            key="coomans_class1_tab2"
                        )
                    with col2:
                        available_classes_2 = [c for c in classes if c != selected_class_1]
                        selected_class_2 = st.selectbox(
                            "Class 2",
                            options=available_classes_2,
                            index=0,
                            key="coomans_class2_tab2"
                        )
                st.success(f"✅ Comparing: {selected_class_1} vs {selected_class_2}")

                try:
                    # Get predictions and distances for all samples
                    if trained['name'] == 'SIMCA':
                        pred_detailed = predict_simca_detailed(X_scaled, trained['model'])
                        distances_array_all = pred_detailed['distances_per_class']
                    elif trained['name'] == 'UNEQ':
                        pred_detailed = predict_uneq_detailed(X_scaled, trained['model'])
                        distances_array_all = pred_detailed['distances_per_class']

                    # Find indices of selected classes in the full class list
                    class_list = classes.tolist()
                    idx_class1 = class_list.index(selected_class_1)
                    idx_class2 = class_list.index(selected_class_2)

                    # Extract distances for the two selected classes
                    dist_class1 = distances_array_all[:, idx_class1]
                    dist_class2 = distances_array_all[:, idx_class2]

                    # Get critical distances for selected classes
                    if trained['name'] == 'SIMCA':
                        crit_dist1 = trained['model']['class_models'][selected_class_1]['f_critical']
                        crit_dist2 = trained['model']['class_models'][selected_class_2]['f_critical']
                    elif trained['name'] == 'UNEQ':
                        crit_dist1 = trained['model']['class_models'][selected_class_1]['t2_critical']
                        crit_dist2 = trained['model']['class_models'][selected_class_2]['t2_critical']

                    # Filter samples to only those belonging to the two selected classes
                    mask_selected = np.isin(y_labels, [selected_class_1, selected_class_2])
                    dist_class1_filtered = dist_class1[mask_selected]
                    dist_class2_filtered = dist_class2[mask_selected]
                    y_labels_filtered = y_labels[mask_selected]

                    # Preserve original 1-based sample indices
                    original_indices = np.where(mask_selected)[0]

                    if hasattr(X_scaled, 'index'):
                        sample_names = X_scaled.index[original_indices].tolist()
                    else:
                        sample_names = [str(i+1) for i in original_indices]

                    # Ensure y_labels is in the correct format
                    y_true_list = y_labels_filtered.tolist() if hasattr(y_labels_filtered, 'tolist') else list(y_labels_filtered)

                    # Debug info on data shapes
                    with st.expander("🔍 Debug Information", expanded=False):
                        st.write(f"**Total Samples**: {len(y_labels)}")
                        st.write(f"**Filtered Samples (selected 2 classes)**: {len(y_labels_filtered)}")
                        st.write(f"**Selected Classes**: {selected_class_1} (index {idx_class1}), {selected_class_2} (index {idx_class2})")
                        st.write(f"**Distance Array Shape (all classes)**: {distances_array_all.shape}")
                        st.write(f"**Distance to {selected_class_1}**: min={dist_class1_filtered.min():.3f}, max={dist_class1_filtered.max():.3f}, mean={dist_class1_filtered.mean():.3f}")
                        st.write(f"**Distance to {selected_class_2}**: min={dist_class2_filtered.min():.3f}, max={dist_class2_filtered.max():.3f}, mean={dist_class2_filtered.mean():.3f}")
                        st.write(f"**Critical Distance {selected_class_1}**: {crit_dist1:.3f}")
                        st.write(f"**Critical Distance {selected_class_2}**: {crit_dist2:.3f}")
                        st.write(f"**y_labels_filtered type**: {type(y_labels_filtered)}, length={len(y_labels_filtered)}")
                        st.write(f"**y_labels_filtered preview (first 10)**: {y_true_list[:10]}")
                        st.write(f"**Unique classes in filtered labels**: {np.unique(y_true_list)}")
                        st.write(f"**Sample Names (first 10)**: {sample_names[:10]}")
                        st.write(f"**Original Indices (first 10)**: {original_indices[:10]}")
                        st.write(f"**DEBUG - About to call plot_coomans:**")
                        st.write(f"dist_class1_filtered shape: {dist_class1_filtered.shape}, values: {dist_class1_filtered[:5]}")
                        st.write(f"dist_class2_filtered shape: {dist_class2_filtered.shape}, values: {dist_class2_filtered[:5]}")
                        st.write(f"y_true_list: {y_true_list[:10]}")
                        st.write(f"sample_names: {sample_names[:5]}")

                    fig_coomans_cv = plot_coomans(
                        dist_class1=dist_class1_filtered,
                        dist_class2=dist_class2_filtered,
                        y_true=y_true_list,
                        crit_dist1=crit_dist1,
                        crit_dist2=crit_dist2,
                        class_names=[str(selected_class_1), str(selected_class_2)],
                        title=f"Coomans Plot - {trained['name']}: {selected_class_1} vs {selected_class_2} (CV)",
                        normalize=False,
                        sample_names=sample_names
                    )
                    st.plotly_chart(fig_coomans_cv, use_container_width=True, key="coomans_cv_tab2")

                except Exception as e:
                    st.error(f"❌ Could not generate Coomans plot: {str(e)}")

                    # Enhanced debug information on error
                    with st.expander("🐛 Error Debug Information", expanded=True):
                        st.write("**Error Details:**")
                        st.code(str(e))

                        st.write("**Traceback:**")
                        import traceback
                        st.code(traceback.format_exc())

                        st.write("**Data Diagnostics:**")
                        try:
                            st.write(f"- X_scaled shape: {X_scaled.shape if hasattr(X_scaled, 'shape') else 'N/A'}")
                            st.write(f"- y_labels shape/length: {y_labels.shape if hasattr(y_labels, 'shape') else len(y_labels)}")
                            st.write(f"- classes: {classes}")
                            st.write(f"- trained['name']: {trained['name']}")
                            if 'distances_array' in locals():
                                st.write(f"- distances_array shape: {distances_array.shape}")
                                st.write(f"- distances_array sample values: {distances_array[:3]}")
                        except Exception as debug_err:
                            st.write(f"Could not retrieve debug info: {debug_err}")

            # Coomans Comparison: SIMCA vs UNEQ (if user wants comparison)
            if len(classes) == 2:
                st.divider()
                st.markdown("#### 📊 Coomans Comparison: SIMCA vs UNEQ")

                if st.checkbox("Show SIMCA vs UNEQ Comparison", value=False, key="show_coomans_comparison_tab2"):
                    st.info("Side-by-side comparison of SIMCA and UNEQ class modeling approaches")

                    try:
                        # Train both SIMCA and UNEQ for comparison
                        with st.spinner("Training SIMCA and UNEQ for comparison..."):
                            # Get parameters
                            n_pcs = tab1_data.get('n_pcs', 3)
                            confidence_level = tab1_data.get('confidence_level', 0.95)

                            # Train SIMCA
                            simca_model = fit_simca(X_scaled, y_labels, n_pcs, confidence_level)
                            simca_pred_detailed = predict_simca_detailed(X_scaled, simca_model)
                            simca_distances = simca_pred_detailed['distances_per_class']

                            # Train UNEQ
                            uneq_model = fit_uneq(X_scaled, y_labels, n_pcs, confidence_level, use_pca=False)
                            uneq_pred_detailed = predict_uneq_detailed(X_scaled, uneq_model)
                            uneq_distances = uneq_pred_detailed['distances_per_class']

                            # Get class names and thresholds
                            cls1, cls2 = classes[0], classes[1]
                            simca_crit1 = simca_model['class_models'][cls1]['f_critical']
                            simca_crit2 = simca_model['class_models'][cls2]['f_critical']
                            uneq_crit1 = uneq_model['class_models'][cls1]['t2_critical']
                            uneq_crit2 = uneq_model['class_models'][cls2]['t2_critical']

                            # Create subplots
                            from plotly.subplots import make_subplots

                            fig_comparison = make_subplots(
                                rows=1, cols=2,
                                subplot_titles=(
                                    f"SIMCA (F-statistic)",
                                    f"UNEQ (T²-statistic)"
                                ),
                                horizontal_spacing=0.12
                            )

                            # Prepare data
                            y_true_list = y_labels.tolist() if hasattr(y_labels, 'tolist') else list(y_labels)

                            # Define colors for classes
                            color_map = {cls1: '#1f77b4', cls2: '#ff7f0e'}  # Blue and Orange
                            colors = [color_map[cls] for cls in y_true_list]

                            # Left plot: SIMCA
                            fig_comparison.add_trace(
                                go.Scatter(
                                    x=simca_distances[:, 0],
                                    y=simca_distances[:, 1],
                                    mode='markers',
                                    marker=dict(color=colors, size=8, line=dict(width=1, color='white')),
                                    name='Samples',
                                    showlegend=False,
                                    text=[f"Sample {i}<br>Class: {y_true_list[i]}" for i in range(len(y_true_list))],
                                    hovertemplate='<b>%{text}</b><br>Dist to %s: %%{x:.3f}<br>Dist to %s: %%{y:.3f}<extra></extra>' % (cls1, cls2)
                                ),
                                row=1, col=1
                            )

                            # SIMCA critical lines
                            fig_comparison.add_hline(y=simca_crit2, line_dash="dash", line_color="red",
                                                    annotation_text=f"{cls2} threshold", row=1, col=1)
                            fig_comparison.add_vline(x=simca_crit1, line_dash="dash", line_color="blue",
                                                    annotation_text=f"{cls1} threshold", row=1, col=1)

                            # Right plot: UNEQ
                            fig_comparison.add_trace(
                                go.Scatter(
                                    x=uneq_distances[:, 0],
                                    y=uneq_distances[:, 1],
                                    mode='markers',
                                    marker=dict(color=colors, size=8, line=dict(width=1, color='white')),
                                    name='Samples',
                                    showlegend=False,
                                    text=[f"Sample {i}<br>Class: {y_true_list[i]}" for i in range(len(y_true_list))],
                                    hovertemplate='<b>%{text}</b><br>Dist to %s: %%{x:.3f}<br>Dist to %s: %%{y:.3f}<extra></extra>' % (cls1, cls2)
                                ),
                                row=1, col=2
                            )

                            # UNEQ critical lines
                            fig_comparison.add_hline(y=uneq_crit2, line_dash="dash", line_color="red",
                                                    annotation_text=f"{cls2} threshold", row=1, col=2)
                            fig_comparison.add_vline(x=uneq_crit1, line_dash="dash", line_color="blue",
                                                    annotation_text=f"{cls1} threshold", row=1, col=2)

                            # Update layout
                            fig_comparison.update_xaxes(title_text=f"Distance to Class {cls1}", row=1, col=1)
                            fig_comparison.update_yaxes(title_text=f"Distance to Class {cls2}", row=1, col=1)
                            fig_comparison.update_xaxes(title_text=f"Distance to Class {cls1}", row=1, col=2)
                            fig_comparison.update_yaxes(title_text=f"Distance to Class {cls2}", row=1, col=2)

                            fig_comparison.update_layout(
                                title_text="Coomans Plot Comparison: SIMCA vs UNEQ (CV Diagnostics)",
                                height=600,
                                width=1400,
                                showlegend=False
                            )

                            st.plotly_chart(fig_comparison, use_container_width=True)

                            # Add interpretation
                            st.caption(
                                "**Left (SIMCA)**: Uses F-statistic distances based on PCA models per class. "
                                "**Right (UNEQ)**: Uses Mahalanobis T²-statistic distances with different dispersions per class. "
                                "Points closer to origin in each plot indicate better fit to the respective class model."
                            )

                    except Exception as e:
                        st.error(f"Could not generate SIMCA vs UNEQ comparison: {str(e)}")
                        import traceback
                        st.error(traceback.format_exc())

            st.divider()

            # Per-class metrics
            st.markdown("#### Per-Class Metrics (CV)")
            per_class_cv_data = []
            for cls in classes:
                per_class_cv_data.append({
                    'Class': cls,
                    'Sensitivity %': f"{cv_metrics['sensitivity_per_class'][cls]:.2f}",
                    'Specificity %': f"{cv_metrics['specificity_per_class'][cls]:.2f}",
                    'Precision %': f"{cv_metrics['precision_per_class'][cls]:.2f}",
                    'F1 %': f"{cv_metrics['f1_per_class'][cls]:.2f}"
                })
            per_class_cv_df = pd.DataFrame(per_class_cv_data)
            st.dataframe(per_class_cv_df, use_container_width=True, hide_index=True)

            # Sensitivity & Specificity Detailed Table
            st.markdown("#### Sensitivity & Specificity by Class")
            st.info("Detailed breakdown of True Positive Rate (Sensitivity) and True Negative Rate (Specificity)")

            # Calculate class support (number of samples per class)
            class_support = {}
            for cls in classes:
                class_support[cls] = int(np.sum(y_labels == cls))

            # Create detailed sensitivity/specificity table
            sens_spec_data = []
            for cls in classes:
                sens = cv_metrics['sensitivity_per_class'][cls]
                spec = cv_metrics['specificity_per_class'][cls]

                # Determine status based on both metrics
                if sens > 80 and spec > 80:
                    status = "🟢 Good"
                elif sens > 70 and spec > 70:
                    status = "🟡 OK"
                else:
                    status = "🔴 Low"

                sens_spec_data.append({
                    'Class': cls,
                    'Sensitivity %': sens,
                    'Specificity %': spec,
                    'Support': class_support[cls],
                    'Status': status
                })

            sens_spec_df = pd.DataFrame(sens_spec_data)

            # Apply color coding with styling
            def color_metrics(row):
                colors = []
                for col in row.index:
                    if col == 'Status':
                        if '🟢' in str(row[col]):
                            colors.append('background-color: #d4edda')  # Light green
                        elif '🟡' in str(row[col]):
                            colors.append('background-color: #fff3cd')  # Light yellow
                        elif '🔴' in str(row[col]):
                            colors.append('background-color: #f8d7da')  # Light red
                        else:
                            colors.append('')
                    elif col in ['Sensitivity %', 'Specificity %']:
                        val = row[col]
                        if val > 80:
                            colors.append('background-color: #d4edda')  # Light green
                        elif val > 70:
                            colors.append('background-color: #fff3cd')  # Light yellow
                        else:
                            colors.append('background-color: #f8d7da')  # Light red
                    else:
                        colors.append('')
                return colors

            styled_sens_spec_df = sens_spec_df.style.apply(color_metrics, axis=1).format({
                'Sensitivity %': '{:.2f}',
                'Specificity %': '{:.2f}'
            })

            st.dataframe(styled_sens_spec_df, use_container_width=True, hide_index=True)

            # Add explanatory footnote
            st.caption(
                "**Sensitivity (True Positive Rate)**: Percentage of actual positives correctly identified for each class. "
                "**Specificity (True Negative Rate)**: Percentage of actual negatives correctly identified. "
                "**Thresholds**: 🟢 Good (>80%), 🟡 OK (70-80%), 🔴 Low (<70%)"
            )

            # Classification report heatmap
            st.markdown("#### Classification Report Heatmap")
            fig_report = plot_classification_report(
                cv_metrics,
                classes.tolist(),
                title=f"{trained['name']} CV Performance"
            )
            st.plotly_chart(fig_report, use_container_width=True)

            # Misclassified samples
            st.divider()
            st.markdown("#### 🔍 Misclassified Samples Analysis")

            misclassified_indices = np.where(y_labels != y_pred_cv)[0]
            n_misclassified = len(misclassified_indices)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Misclassified", n_misclassified)
            with col2:
                st.metric("Correct", len(y_labels) - n_misclassified)
            with col3:
                accuracy = (len(y_labels) - n_misclassified) / len(y_labels) * 100
                st.metric("Accuracy", f"{accuracy:.1f}%")

            if n_misclassified > 0:
                misclass_data = []
                for idx in misclassified_indices[:20]:
                    misclass_data.append({
                        'Sample Index': idx + 1,
                        'True Class': y_labels[idx],
                        'Predicted Class': y_pred_cv[idx],
                        'Error': 'Misclassification'
                    })

                misclass_df = pd.DataFrame(misclass_data)
                st.dataframe(misclass_df, use_container_width=True, hide_index=True)

                if n_misclassified > 20:
                    st.caption(f"Showing first 20 of {n_misclassified} misclassified samples")
            else:
                st.success("✅ All samples correctly classified!")

            # Distance distributions
            st.divider()
            st.markdown("#### 📈 Distance Distributions")

            if trained['name'] in ['SIMCA', 'UNEQ']:
                st.info("Distance to class models - shows sample distribution relative to each class model")

                if trained['name'] == 'SIMCA':
                    pred_detailed = predict_simca_detailed(X_scaled, trained['model'])
                    distances_array = pred_detailed['distances_per_class']
                elif trained['name'] == 'UNEQ':
                    pred_detailed = predict_uneq_detailed(X_scaled, trained['model'])
                    distances_array = pred_detailed['distances_per_class']

                for i, cls in enumerate(classes):
                    distances_dict = {cls: distances_array[:, i]}

                    if trained['name'] == 'SIMCA':
                        threshold = trained['model']['class_models'][cls]['f_critical']
                    else:
                        threshold = trained['model']['class_models'][cls]['t2_critical']

                    fig_dist = plot_distance_distributions(
                        distances_dict,
                        y_labels,
                        selected_class=cls,
                        threshold=threshold,
                        title=f"{trained['name']} Distance to Class {cls}"
                    )
                    st.plotly_chart(fig_dist, use_container_width=True)

            elif trained['name'] in ['LDA', 'QDA']:
                st.info("Mahalanobis distance distributions to each class centroid")

                if trained['name'] == 'LDA':
                    y_pred, distances_array = predict_lda(X_scaled, trained['model'])
                elif trained['name'] == 'QDA':
                    y_pred, distances_array = predict_qda(X_scaled, trained['model'])

                for i, cls in enumerate(classes):
                    distances_dict = {cls: distances_array[:, i]}

                    fig_dist = plot_distance_distributions(
                        distances_dict,
                        y_labels,
                        selected_class=cls,
                        threshold=None,
                        title=f"{trained['name']} Distance to Class {cls}"
                    )
                    st.plotly_chart(fig_dist, use_container_width=True)

            elif trained['name'] == 'kNN':
                st.info("Within-class distance statistics for kNN classifier")

                st.markdown("**Average distances within each class:**")

                distance_summary = []
                for cls in classes:
                    cls_mask = y_labels == cls
                    X_cls = X_scaled[cls_mask]

                    if X_cls.shape[0] > 1:
                        within_dist = calculate_distance_matrix(
                            X_cls, X_cls,
                            metric=trained['parameters']['metric']
                        )
                        within_mean = np.mean(within_dist[np.triu_indices_from(within_dist, k=1)])
                    else:
                        within_mean = 0.0

                    distance_summary.append({
                        'Class': cls,
                        'Within-Class Avg Distance': f"{within_mean:.3f}",
                        'Samples': int(np.sum(cls_mask))
                    })

                summary_df = pd.DataFrame(distance_summary)
                st.dataframe(summary_df, use_container_width=True, hide_index=True)
                st.caption(f"Distance metric: {trained['parameters']['metric']}")

            # Category-Specific Analysis Section
            st.divider()
            st.markdown("### 🎯 Category-Specific Analysis")

            # Class selector
            selected_class_tab2 = st.selectbox(
                "Select class to analyze",
                options=classes.tolist(),
                key="class_selector_tab2"
            )

            st.markdown(f"#### Performance Metrics for {selected_class_tab2}")

            # Show metrics for selected class
            col1, col2, col3 = st.columns(3)
            with col1:
                sensitivity = cv_metrics['sensitivity_per_class'][selected_class_tab2]
                st.metric(f"Sensitivity ({selected_class_tab2})", f"{sensitivity:.1f}%")
            with col2:
                specificity = cv_metrics['specificity_per_class'][selected_class_tab2]
                st.metric(f"Specificity ({selected_class_tab2})", f"{specificity:.1f}%")
            with col3:
                f1 = cv_metrics['f1_per_class'][selected_class_tab2]
                st.metric(f"F1 Score ({selected_class_tab2})", f"{f1:.1f}%")

            # Distance distribution for this class
            st.markdown(f"#### Distance Distribution to Class {selected_class_tab2}")

            try:
                if trained['name'] in ['SIMCA', 'UNEQ']:
                    # Get distances array and threshold for SIMCA/UNEQ
                    if trained['name'] == 'SIMCA':
                        pred_detailed = predict_simca_detailed(X_scaled, trained['model'])
                        distances_array = pred_detailed['distances_per_class']
                        threshold = trained['model']['class_models'][selected_class_tab2]['f_critical']
                    else:  # UNEQ
                        pred_detailed = predict_uneq_detailed(X_scaled, trained['model'])
                        distances_array = pred_detailed['distances_per_class']
                        threshold = trained['model']['class_models'][selected_class_tab2]['t2_critical']

                    # Find the index of the selected class
                    class_idx = list(classes).index(selected_class_tab2)
                    distances_dict = {selected_class_tab2: distances_array[:, class_idx]}

                    fig_dist = plot_distance_distributions(
                        distances_dict,
                        y_labels,
                        selected_class=selected_class_tab2,
                        threshold=threshold,
                        title=f"{trained['name']} Distance to Class {selected_class_tab2}"
                    )
                    st.plotly_chart(fig_dist, use_container_width=True)

                elif trained['name'] in ['LDA', 'QDA']:
                    # Get distances array for LDA/QDA
                    if trained['name'] == 'LDA':
                        y_pred, distances_array = predict_lda(X_scaled, trained['model'])
                    else:  # QDA
                        y_pred, distances_array = predict_qda(X_scaled, trained['model'])

                    # Find the index of the selected class
                    class_idx = list(classes).index(selected_class_tab2)
                    distances_dict = {selected_class_tab2: distances_array[:, class_idx]}

                    fig_dist = plot_distance_distributions(
                        distances_dict,
                        y_labels,
                        selected_class=selected_class_tab2,
                        threshold=None,
                        title=f"{trained['name']} Mahalanobis Distance to Class {selected_class_tab2}"
                    )
                    st.plotly_chart(fig_dist, use_container_width=True)

                elif trained['name'] == 'kNN':
                    # For kNN, show within-class distance statistics
                    cls_mask = y_labels == selected_class_tab2
                    X_cls = X_scaled[cls_mask]

                    if X_cls.shape[0] > 1:
                        within_dist = calculate_distance_matrix(
                            X_cls, X_cls,
                            metric=trained['parameters']['metric']
                        )
                        within_mean = np.mean(within_dist[np.triu_indices_from(within_dist, k=1)])
                        within_std = np.std(within_dist[np.triu_indices_from(within_dist, k=1)])

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Avg Within-Class Distance", f"{within_mean:.3f}")
                        with col2:
                            st.metric("Std Deviation", f"{within_std:.3f}")
                        with col3:
                            st.metric("Class Samples", int(np.sum(cls_mask)))

                        st.info(f"Within-class distance statistics for {selected_class_tab2} using {trained['parameters']['metric']} metric")
                    else:
                        st.warning(f"Not enough samples in class {selected_class_tab2} for distance analysis")

            except Exception as e:
                st.warning(f"Could not generate distance plot for {selected_class_tab2}: {str(e)}")

            # Single Sample Analysis Section
            st.divider()
            st.markdown("### 📌 Single Sample Analysis (CV)")

            # Reorder samples: misclassified first, then correct
            misclassified_idx = np.where(y_labels != y_pred_cv)[0]
            correct_idx = np.where(y_labels == y_pred_cv)[0]
            ordered_indices = np.concatenate([misclassified_idx, correct_idx])

            # Create sample names dictionary
            if hasattr(X_data, 'index'):
                sample_names_dict = {i: X_data.index[i] for i in range(len(X_data))}
            else:
                sample_names_dict = {i: str(i+1) for i in range(len(X_data))}

            # Sample selector with formatted display
            selected_sample_idx_tab2 = st.selectbox(
                "Select sample to analyze",
                options=ordered_indices,
                format_func=lambda x: f"Sample {sample_names_dict[x]}: True={y_labels[x]}, Pred={y_pred_cv[x]}",
                key="sample_selector_tab2"
            )

            # Sample details
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("True Class", y_labels[selected_sample_idx_tab2])
            with col2:
                st.metric("Predicted", y_pred_cv[selected_sample_idx_tab2])
            with col3:
                match = "✅ Correct" if y_labels[selected_sample_idx_tab2] == y_pred_cv[selected_sample_idx_tab2] else "❌ Error"
                st.metric("Result", match)

            # Feature values for selected sample
            st.markdown("#### Feature Values")
            feature_vals = X_data.iloc[selected_sample_idx_tab2]
            feature_df = pd.DataFrame({
                'Feature': tab1_data.get('X_cols', [f'Feature {i}' for i in range(len(feature_vals))]),
                'Value': feature_vals
            })
            st.dataframe(feature_df, use_container_width=True, hide_index=True)

            # Distance to each class (classifier-specific)
            st.markdown("#### Distance to Each Class")

            try:
                distances_to_classes = []

                if trained['name'] == 'LDA':
                    # Get distances for all samples
                    _, distances_array = predict_lda(X_scaled, trained['model'])
                    distances_to_classes = distances_array[selected_sample_idx_tab2, :].tolist()

                elif trained['name'] == 'QDA':
                    # Get distances for all samples
                    _, distances_array = predict_qda(X_scaled, trained['model'])
                    distances_to_classes = distances_array[selected_sample_idx_tab2, :].tolist()

                elif trained['name'] == 'kNN':
                    # Calculate distance from selected sample to each class centroid
                    for cls in classes:
                        cls_mask = y_labels == cls
                        X_cls = X_scaled[cls_mask]
                        # Calculate distance to class centroid
                        centroid = np.mean(X_cls, axis=0)
                        dist = np.linalg.norm(X_scaled[selected_sample_idx_tab2] - centroid)
                        distances_to_classes.append(dist)

                elif trained['name'] == 'SIMCA':
                    # Get distances for all samples
                    pred_detailed = predict_simca_detailed(X_scaled, trained['model'])
                    distances_array = pred_detailed['distances_per_class']
                    distances_to_classes = distances_array[selected_sample_idx_tab2, :].tolist()

                elif trained['name'] == 'UNEQ':
                    # Get distances for all samples
                    pred_detailed = predict_uneq_detailed(X_scaled, trained['model'])
                    distances_array = pred_detailed['distances_per_class']
                    distances_to_classes = distances_array[selected_sample_idx_tab2, :].tolist()

                # Create distance dataframe
                dist_df = pd.DataFrame({
                    'Class': [str(c) for c in classes.tolist()],
                    'Distance': [f"{d:.4f}" for d in distances_to_classes],
                    'Match': ['✅ TRUE CLASS' if c == y_labels[selected_sample_idx_tab2] else
                              '🔵 PREDICTED' if c == y_pred_cv[selected_sample_idx_tab2] else '⚪ Other'
                              for c in classes]
                })
                st.dataframe(dist_df, use_container_width=True, hide_index=True)

                # Visualization
                fig_dist_sample = go.Figure()

                # Determine bar colors
                bar_colors = []
                for c in classes:
                    if c == y_labels[selected_sample_idx_tab2]:
                        bar_colors.append('#28a745')  # Green for true class
                    elif c == y_pred_cv[selected_sample_idx_tab2]:
                        bar_colors.append('#ffc107')  # Orange for predicted class
                    else:
                        bar_colors.append('#dc3545')  # Red for other classes

                fig_dist_sample.add_trace(go.Bar(
                    x=[str(c) for c in classes.tolist()],
                    y=distances_to_classes,
                    marker_color=bar_colors,
                    text=[f"{d:.4f}" for d in distances_to_classes],
                    textposition='auto'
                ))

                fig_dist_sample.update_layout(
                    title=f"Sample {selected_sample_idx_tab2} - Distance to Each Class",
                    xaxis_title="Class",
                    yaxis_title=f"Distance ({'Mahalanobis' if trained['name'] in ['LDA', 'QDA'] else 'F-statistic' if trained['name'] == 'SIMCA' else 'T²-statistic' if trained['name'] == 'UNEQ' else 'Euclidean'})",
                    showlegend=False,
                    height=400
                )

                st.plotly_chart(fig_dist_sample, use_container_width=True)

                # Add interpretation
                st.caption(
                    "🟢 **Green bar**: True class | "
                    "🟡 **Orange bar**: Predicted class | "
                    "🔴 **Red bars**: Other classes. "
                    f"Lower distance = Higher similarity ({trained['name']} classifier)"
                )

            except Exception as e:
                st.error(f"Could not calculate distances for sample: {str(e)}")
                import traceback
                st.error(traceback.format_exc())

    # ========== TAB 3: MODEL COMPARISON ==========
    with tab3:
        st.markdown("## 🏆 Model Comparison (Optional)")

        tab1_data = st.session_state.get('tab1_data', {})
        if not tab1_data:
            st.warning("⚠️ Configure and prepare data in **Tab 1: Setup & Configuration** first")
            return

        X_data = tab1_data['X_data']
        y_labels = tab1_data['y_labels']
        classes = tab1_data['classes']

        st.info("📊 Compare performance of multiple classifiers on the same dataset")

        col1, col2 = st.columns([2, 1])
        with col1:
            classifiers_to_test = st.multiselect(
                "Select classifiers to compare",
                options=['LDA', 'QDA', 'kNN', 'SIMCA', 'UNEQ'],
                default=['LDA', 'QDA', 'kNN'],
                key="comparison_classifiers_tab3"
            )

        with col2:
            cv_folds = st.number_input(
                "CV Folds",
                min_value=2,
                max_value=10,
                value=5,
                key="comparison_cv_folds_tab3"
            )

        if st.button("🚀 Run Comparison", type="primary", use_container_width=True, key="run_comparison_btn_tab3"):
            if len(classifiers_to_test) < 2:
                st.warning("⚠️ Please select at least 2 classifiers to compare")
            else:
                with st.spinner(f"Comparing {len(classifiers_to_test)} classifiers..."):
                    import time

                    comparison_start = time.time()

                    try:
                        prep_data = prepare_training_test(
                            X_data.values,
                            y_labels,
                            scaling_method=tab1_data.get('scaling_method', 'autoscale')
                        )
                        X_scaled = prep_data['X_train']

                        # Run CV for each classifier
                        cv_results_dict = {}
                        models_data = []

                        for clf_name in classifiers_to_test:
                            clf_start = time.time()

                            if clf_name == 'LDA':
                                cv_res = cross_validate_classifier(
                                    X_scaled, y_labels,
                                    classifier_type='lda',
                                    n_folds=cv_folds,
                                    classifier_params=None,
                                    random_state=42
                                )
                            elif clf_name == 'QDA':
                                cv_res = cross_validate_classifier(
                                    X_scaled, y_labels,
                                    classifier_type='qda',
                                    n_folds=cv_folds,
                                    classifier_params=None,
                                    random_state=42
                                )
                            elif clf_name == 'kNN':
                                cv_res = cross_validate_classifier(
                                    X_scaled, y_labels,
                                    classifier_type='knn',
                                    n_folds=cv_folds,
                                    classifier_params={'k': 5, 'metric': 'euclidean'},
                                    random_state=42
                                )
                            elif clf_name == 'SIMCA':
                                cv_res = cross_validate_classifier(
                                    X_scaled, y_labels,
                                    classifier_type='simca',
                                    n_folds=cv_folds,
                                    classifier_params={'n_components': 3, 'confidence_level': 0.95},
                                    random_state=42
                                )
                            elif clf_name == 'UNEQ':
                                cv_res = cross_validate_classifier(
                                    X_scaled, y_labels,
                                    classifier_type='uneq',
                                    n_folds=cv_folds,
                                    classifier_params={'n_components': None, 'use_pca': False, 'confidence_level': 0.95},
                                    random_state=42
                                )

                            clf_time = time.time() - clf_start

                            y_pred = cv_res['cv_predictions']
                            cv_metrics = compute_classification_metrics(y_labels, y_pred, classes)

                            per_class_metrics = {}
                            for cls in classes:
                                per_class_metrics[cls] = {
                                    'sensitivity': cv_metrics['sensitivity_per_class'][cls],
                                    'specificity': cv_metrics['specificity_per_class'][cls],
                                    'f1': cv_metrics['f1_per_class'][cls]
                                }

                            models_data.append({
                                'classifier': clf_name,
                                'cv_accuracy': cv_metrics['accuracy'],
                                'avg_sensitivity': cv_metrics['average_sensitivity'],
                                'avg_specificity': cv_metrics['average_specificity'],
                                'avg_f1': cv_metrics['macro_f1'],
                                'training_time': clf_time,
                                'per_class_metrics': per_class_metrics
                            })

                            cv_results_dict[clf_name] = {
                                'metrics': {
                                    'accuracy': cv_metrics['accuracy'],
                                    'avg_sensitivity': cv_metrics['average_sensitivity'],
                                    'avg_specificity': cv_metrics['average_specificity'],
                                    'avg_efficiency': cv_metrics['macro_f1']
                                }
                            }

                        comparison_time = time.time() - comparison_start

                        st.session_state['comparison_results'] = {
                            'models': models_data,
                            'comparison_df': compare_models(cv_results_dict)
                        }
                        st.session_state['comparison_time'] = comparison_time

                        st.success(f"✅ Model comparison completed in {comparison_time:.2f}s!")
                        st.rerun()

                    except Exception as e:
                        st.error(f"❌ Comparison failed: {str(e)}")
                        import traceback
                        st.error(traceback.format_exc())

        # Display comparison results
        if 'comparison_results' in st.session_state:
            results = st.session_state['comparison_results']

            st.divider()
            st.markdown("### 📊 Comparison Results")

            # Summary table
            st.markdown("#### Performance Summary Table")

            summary_data = []
            for model_result in results['models']:
                summary_data.append({
                    'Classifier': model_result['classifier'],
                    'CV Accuracy': f"{model_result['cv_accuracy']:.2f}%",
                    'Avg Sensitivity': f"{model_result['avg_sensitivity']:.2f}%",
                    'Avg Specificity': f"{model_result['avg_specificity']:.2f}%",
                    'Avg F1-Score': f"{model_result['avg_f1']:.2f}%",
                    'Training Time': f"{model_result['training_time']:.3f}s"
                })

            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True, hide_index=True)

            # Highlight best model
            best_model_idx = np.argmax([m['cv_accuracy'] for m in results['models']])
            best_model = results['models'][best_model_idx]
            st.success(f"🏆 **Best Model:** {best_model['classifier']} with {best_model['cv_accuracy']:.2f}% CV Accuracy")

            st.divider()

            # Comparison plot
            st.markdown("#### Visual Comparison")
            fig_comp = plot_model_comparison(
                results['comparison_df'],
                title="Model Performance Comparison"
            )
            st.plotly_chart(fig_comp, use_container_width=True)

            st.divider()

            # Detailed per-model results
            with st.expander("📋 Detailed Results per Classifier", expanded=False):
                for model_result in results['models']:
                    st.markdown(f"### {model_result['classifier']}")

                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("CV Accuracy", f"{model_result['cv_accuracy']:.2f}%")
                    with col2:
                        st.metric("Avg Sensitivity", f"{model_result['avg_sensitivity']:.2f}%")
                    with col3:
                        st.metric("Avg Specificity", f"{model_result['avg_specificity']:.2f}%")
                    with col4:
                        st.metric("Training Time", f"{model_result['training_time']:.3f}s")

                    if 'per_class_metrics' in model_result:
                        st.markdown("**Per-Class Performance:**")
                        per_class_comp_data = []
                        for cls in classes:
                            if cls in model_result['per_class_metrics']:
                                per_class_comp_data.append({
                                    'Class': cls,
                                    'Sensitivity %': f"{model_result['per_class_metrics'][cls]['sensitivity']:.2f}",
                                    'Specificity %': f"{model_result['per_class_metrics'][cls]['specificity']:.2f}",
                                    'F1 %': f"{model_result['per_class_metrics'][cls]['f1']:.2f}"
                                })

                        per_class_comp_df = pd.DataFrame(per_class_comp_data)
                        st.dataframe(per_class_comp_df, use_container_width=True, hide_index=True)

                    st.divider()

    # ========== TAB 4: TEST & VALIDATION (Quality Control Style) ==========
    with tab4:
        st.markdown("## 📋 Test Set & Model Validation")
        st.markdown("*Apply the trained model to test data (Quality Control workflow)*")

        if st.session_state.trained_model is None:
            st.warning("⚠️ Train a model first in Tab 1 before testing")
            st.info("💡 Go to **Tab 1: Setup & Configuration** and click 'Train Model'")
        else:
            # Get stored data from TAB 1
            tab1_data = st.session_state.get('tab1_data', {})
            X_data = tab1_data.get('X_data')
            y_labels = tab1_data.get('y_labels')
            classes = tab1_data.get('classes')
            X_cols = tab1_data.get('X_cols')

            trained = st.session_state.trained_model

            # --- Model Info Banner ---
            st.info(f"🤖 **Active Model**: {trained['name']} | Training Time: {trained['training_time']:.3f}s | Features: {trained['n_features']}")

            st.divider()

            # --- SECTION 1: Select Test Data (FROM WORKSPACE) ---
            st.markdown("## 📥 Section 1: Select Test Data")

            # Get available datasets from workspace
            available_datasets = get_workspace_datasets()

            if len(available_datasets) == 0:
                st.warning("⚠️ No datasets available in workspace for testing")
                st.info("💡 Load data in the **Data Handling** page first")
            else:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown("### Select Dataset from Workspace")
                    selected_test_dataset = st.selectbox(
                        "Choose test dataset:",
                        options=list(available_datasets.keys()),
                        key="tab4_test_dataset",
                        help="Select dataset to use for testing"
                    )
                with col2:
                    st.markdown("### Available Data")
                    st.metric("Training Samples", len(X_data))

                # Load selected dataset
                if selected_test_dataset:
                    test_df = available_datasets[selected_test_dataset].copy()

                    st.divider()

                    # --- SECTION 1B: Select Sample Subset ---
                    st.markdown("### Select Sample Subset from Test Data")

                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Samples", len(test_df))
                    with col2:
                        st.metric("Variables", len(test_df.columns))
                    with col3:
                        numeric_cols_test = test_df.select_dtypes(include=[np.number]).columns
                        st.metric("Numeric", len(numeric_cols_test))
                    with col4:
                        st.markdown("**Status**")
                        st.markdown("✅ Ready")

                    st.divider()

                    # Add user guidance
                    st.info(
                        "💡 **Tip:** By default, all samples are selected. "
                        "Adjust the range below to test on a specific subset if needed."
                    )

                    # Range selection for subset
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        test_start = st.number_input(
                            "Start sample (1-based):",
                            min_value=1,
                            max_value=len(test_df),
                            value=1,
                            key="tab4_test_start"
                        )
                    with col2:
                        test_end = st.number_input(
                            "End sample (1-based):",
                            min_value=test_start,
                            max_value=len(test_df),
                            value=len(test_df),
                            key="tab4_test_end"
                        )
                    with col3:
                        subset_size = test_end - test_start + 1
                        percentage = (subset_size / len(test_df)) * 100
                        st.metric("Subset Size", f"{subset_size} ({percentage:.1f}%)")

                    # Extract test subset (inclusive range: from test_start to test_end)
                    # Note: iloc uses 0-based indexing and is exclusive at end, so:
                    # iloc[test_start-1:test_end] correctly includes samples test_start through test_end (1-based)
                    test_subset_df = test_df.iloc[test_start-1:test_end].copy()

                    # VALIDATION: Confirm subset was created with correct number of rows
                    expected_rows = test_end - test_start + 1
                    actual_rows = len(test_subset_df)

                    if actual_rows != expected_rows:
                        st.error(f"❌ Subset extraction error: Expected {expected_rows} rows, got {actual_rows}")
                        st.warning(f"DEBUG: iloc[{test_start-1}:{test_end}] on df with {len(test_df)} total rows")
                        st.stop()

                    # Verify features match
                    if all(col in test_subset_df.columns for col in X_cols):
                        X_test = test_subset_df[X_cols].values

                        # Try to get true labels if available - with fallback for corrupted files
                        class_col_candidates = [col for col in test_subset_df.columns
                                               if isinstance(col, str) and col.lower() in ['class', 'category', 'label', 'target',
                                                             'classe', 'classe_label', 'y']]

                        y_test = None
                        has_true_labels = False

                        if class_col_candidates:
                            # Found named class column
                            class_col = class_col_candidates[0]
                            y_test = test_subset_df[class_col].values
                            has_true_labels = True
                            st.success(f"✅ Labels found in column: '{class_col}'")
                        else:
                            # Fallback: Try column position (for corrupted files)
                            # Training typically: col0=Name, col1=Category, col2+=Features
                            try:
                                # If TRAIN model info available, use same position
                                if 'tab1_data' in st.session_state and 'label_col_index' in st.session_state:
                                    label_idx = st.session_state['label_col_index']
                                    y_test = test_subset_df.iloc[:, label_idx].values
                                    has_true_labels = True
                                    st.success(f"✅ Labels found at column position {label_idx}")
                                else:
                                    # Last resort: Assume column 1 (position 1) is class
                                    # This works for: [Name, Category, Features...]
                                    if len(test_subset_df.columns) > 1:
                                        potential_class = test_subset_df.iloc[:, 1].values
                                        # Check if looks like class labels (few unique values, mostly strings/objects)
                                        unique_vals = pd.Series(potential_class).nunique()
                                        if unique_vals <= 10:  # Reasonable number of classes
                                            y_test = potential_class
                                            has_true_labels = True
                                            st.warning(f"⚠️ Labels auto-detected at column position 1 (found {unique_vals} classes)")
                                        else:
                                            st.warning("⚠️ Could not detect label column. Showing predictions only.")
                                    else:
                                        st.warning("⚠️ Could not detect label column. Showing predictions only.")
                            except Exception as e:
                                st.warning(f"⚠️ Error detecting labels: {str(e)}")
                                has_true_labels = False

                        test_data_info = f"Dataset: {selected_test_dataset} | Subset: {test_start}-{test_end} ({subset_size} samples)"

                        st.success(f"📊 {test_data_info}")

                        # Show label information before predictions
                        if has_true_labels and y_test is not None:
                            st.markdown("### ✅ Labels Information")

                            label_col1, label_col2, label_col3 = st.columns(3)
                            with label_col1:
                                st.metric("Labels Found", "Yes ✅")
                            with label_col2:
                                unique_labels = len(np.unique(y_test))
                                st.metric("Unique Classes", unique_labels)
                            with label_col3:
                                st.metric("Total Samples", len(y_test))

                            # Show class distribution
                            unique_classes, class_counts = np.unique(y_test, return_counts=True)
                            distribution_text = ", ".join([f"{cls}: {cnt}" for cls, cnt in zip(unique_classes, class_counts)])
                            st.caption(f"Class distribution: {distribution_text}")
                        else:
                            st.markdown("### ⚠️ No Labels Available")
                            st.warning(
                                "True labels not found in test dataset. "
                                "Predictions will be shown, but accuracy metrics cannot be calculated."
                            )

                        st.divider()

                        # Add validation summary before Section 2
                        st.markdown("### ✅ Test Data Validation Summary")
                        validation_cols = st.columns(4)
                        with validation_cols[0]:
                            st.metric("Selected Samples", len(X_test))
                        with validation_cols[1]:
                            st.metric("Features Used", len(X_cols))
                        with validation_cols[2]:
                            st.metric("True Labels", "Yes" if has_true_labels else "No")
                        with validation_cols[3]:
                            st.metric("Status", "✅ Ready")

                        if len(X_test) == 0:
                            st.error("❌ No samples in test subset!")
                            st.stop()

                        if len(X_test) != expected_rows:
                            st.warning(f"⚠️ Sample count mismatch: expected {expected_rows}, got {len(X_test)}")

                        st.divider()

                        # --- SECTION 2: Make Predictions ---
                        st.markdown("## 🎯 Section 2: Generate Predictions")

                        # Prepare/scale test data using training scaling parameters
                        prep_data = prepare_training_test(
                            tab1_data.get('X_data').values,
                            y_labels,
                            X_test,
                            scaling_method=tab1_data.get('scaling_method', 'autoscale')
                        )
                        X_test_scaled = prep_data['X_test']

                        # ===== TEST DATA PREVIEW =====
                        st.markdown("### 📋 Test Data Preview")

                        # Show first few rows of test data
                        test_preview_rows = min(5, len(X_test))
                        preview_df = pd.DataFrame(
                            X_test[:test_preview_rows],
                            columns=X_cols,
                            index=range(test_start, test_start + test_preview_rows)
                        )

                        st.info(
                            f"📊 Showing first {test_preview_rows} rows of test data "
                            f"({len(X_test)} total samples selected, range {test_start}-{test_end})"
                        )
                        st.dataframe(preview_df, use_container_width=True)

                        st.divider()

                        # Generate predictions with timing
                        import time
                        test_pred_start = time.time()

                        if trained['name'] == 'LDA':
                            y_pred_test, _ = predict_lda(X_test_scaled, trained['model'])
                        elif trained['name'] == 'QDA':
                            y_pred_test, _ = predict_qda(X_test_scaled, trained['model'])
                        elif trained['name'] == 'kNN':
                            y_pred_test, _ = predict_knn(X_test_scaled, trained['model'], k=tab1_data.get('k_value', 3))
                        elif trained['name'] == 'SIMCA':
                            pred_detailed = predict_simca_detailed(X_test_scaled, trained['model'])
                            y_pred_test = pred_detailed['predicted_classes']
                        elif trained['name'] == 'UNEQ':
                            pred_detailed = predict_uneq_detailed(X_test_scaled, trained['model'])
                            y_pred_test = pred_detailed['predicted_classes']

                        test_pred_time = time.time() - test_pred_start

                        # === PCA PREPROCESSING NOTE ===
                        # Show note if PCA was used in training
                        if trained['parameters'].get('use_pca', False) and trained['name'] in ['LDA', 'QDA', 'kNN']:
                            st.caption(
                                f"📊 **Note**: Predictions generated using PCA-preprocessed features "
                                f"({tab1_data.get('n_pcs', 3)} components)"
                            )

                        st.success("✅ Predictions generated successfully")

                        st.divider()

                        # --- SECTION 3: Predictions Table ---
                        st.markdown("## 📊 Section 3: Predictions Summary")

                        # Create predictions table with explicit formatting
                        pred_df = pd.DataFrame({
                            'Sample #': [f"{test_start + i}" for i in range(len(y_pred_test))],  # Explicit string formatting
                            'Predicted Class': [str(cls) for cls in y_pred_test],  # Convert to string for clean display
                            'True Class': [str(cls) for cls in y_test] if has_true_labels else ['?' for _ in y_pred_test],  # Convert to string
                            'Match': ['✅' if (y_pred_test[i] == y_test[i]) else '❌'
                                     for i in range(len(y_pred_test))] if has_true_labels else ['?' for _ in y_pred_test]
                        })

                        # Highlight rows
                        def highlight_match(row):
                            if row['Match'] == '❌':
                                return ['background-color: #ffcccc'] * len(row)
                            elif row['Match'] == '✅':
                                return ['background-color: #ccffcc'] * len(row)
                            else:
                                return [''] * len(row)

                        styled_df = pred_df.style.apply(highlight_match, axis=1)

                        st.dataframe(styled_df, use_container_width=True, hide_index=True)

                        st.divider()

                        # --- SECTION 4: Validation Metrics (if true labels available) ---
                        if has_true_labels and y_test is not None and len(y_test) > 0:
                            st.markdown("## ✅ Section 4: Validation Metrics")

                            test_metrics = calculate_classification_metrics(y_test, y_pred_test, classes)

                            # Summary metrics - Row 1: Accuracy Metrics
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric(
                                    "Test Accuracy",
                                    f"{test_metrics['accuracy']:.1f}%",
                                    delta=f"{test_metrics['accuracy'] - 50:.1f}%"
                                )
                            with col2:
                                st.metric(
                                    "Avg Sensitivity",
                                    f"{test_metrics['avg_sensitivity']:.1f}%"
                                )
                            with col3:
                                st.metric(
                                    "Avg Specificity",
                                    f"{test_metrics['avg_specificity']:.1f}%"
                                )
                            with col4:
                                # Calculate average F1 score
                                avg_f1 = np.mean([test_metrics['class_metrics'][cls]['f1_score'] for cls in classes])
                                st.metric(
                                    "F1 Score",
                                    f"{avg_f1:.1f}%"
                                )

                            # Summary metrics - Row 2: Efficiency Metrics
                            col5, col6, col7, col8 = st.columns(4)
                            with col5:
                                n_correct = (y_test == y_pred_test).sum()
                                st.metric(
                                    "Correct Predictions",
                                    f"{n_correct}/{len(y_test)}"
                                )
                            with col6:
                                st.metric(
                                    "Prediction Time",
                                    f"{test_pred_time:.3f}s"
                                )
                            with col7:
                                per_sample_time = test_pred_time / len(y_test) if len(y_test) > 0 else 0
                                st.metric(
                                    "Per-Sample",
                                    f"{per_sample_time:.4f}s"
                                )
                            with col8:
                                throughput = len(y_test) / test_pred_time if test_pred_time > 0 else 0
                                st.metric(
                                    "Throughput",
                                    f"{throughput:.0f} samples/sec"
                                )

                            st.divider()

                            # Confusion matrix
                            st.markdown("### Confusion Matrix (Test Data)")
                            fig_cm_test = plot_confusion_matrix(
                                test_metrics['confusion_matrix'],
                                classes.tolist(),
                                title=f"Test Set Confusion Matrix - {trained['name']}"
                            )
                            st.plotly_chart(fig_cm_test, use_container_width=True)

                            # Coomans plot for SIMCA/UNEQ (Test Data - with debug diagnostics and class selection)
                            st.divider()
                            st.markdown("#### 📍 Coomans Plot (Test Data - 2-Class Comparison)")

                            # Debug info: Show model and class information
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Model Type", trained['name'])
                            with col2:
                                st.metric("Number of Classes", len(classes))
                            with col3:
                                classes_str = ', '.join([str(c) for c in classes])
                                st.metric("Classes", classes_str)

                            # Check 1: Classifier type
                            if trained['name'] not in ['SIMCA', 'UNEQ']:
                                st.info(f"ℹ️ Coomans plot is only available for SIMCA and UNEQ classifiers. Current classifier: {trained['name']}")
                            # Check 2: Need at least 2 classes
                            elif len(classes) < 2:
                                st.warning(f"⚠️ Coomans plot requires at least 2 classes. Your test dataset has {len(classes)} class(es).")
                            # Both checks passed: allow class selection and render plot
                            else:
                                # Class selection interface
                                if len(classes) == 2:
                                    st.info("Coomans plot showing distance patterns for the 2 classes in your test dataset")
                                    selected_class_1 = classes[0]
                                    selected_class_2 = classes[1]
                                else:
                                    st.info(f"Coomans plot requires 2 classes. Your test dataset has {len(classes)} classes. Select which two to compare:")
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        selected_class_1 = st.selectbox(
                                            "Class 1", options=classes.tolist(), index=0,
                                            key="coomans_class1_tab4"
                                        )
                                    with col2:
                                        available_classes_2 = [c for c in classes if c != selected_class_1]
                                        selected_class_2 = st.selectbox(
                                            "Class 2", options=available_classes_2, index=0,
                                            key="coomans_class2_tab4"
                                        )

                                st.success(f"✅ Comparing: {selected_class_1} vs {selected_class_2}")

                                try:
                                    # Get predictions and distances for all test samples
                                    if trained['name'] == 'SIMCA':
                                        pred_detailed_test = predict_simca_detailed(X_test_scaled, trained['model'])
                                        distances_array_test_all = pred_detailed_test['distances_per_class']
                                    elif trained['name'] == 'UNEQ':
                                        pred_detailed_test = predict_uneq_detailed(X_test_scaled, trained['model'])
                                        distances_array_test_all = pred_detailed_test['distances_per_class']

                                    # Find indices of selected classes in the full class list
                                    class_list = classes.tolist()
                                    idx_class1 = class_list.index(selected_class_1)
                                    idx_class2 = class_list.index(selected_class_2)

                                    # Extract distances for the two selected classes
                                    dist_class1_test = distances_array_test_all[:, idx_class1]
                                    dist_class2_test = distances_array_test_all[:, idx_class2]

                                    # Get critical distances for selected classes
                                    if trained['name'] == 'SIMCA':
                                        crit_dist1 = trained['model']['class_models'][selected_class_1]['f_critical']
                                        crit_dist2 = trained['model']['class_models'][selected_class_2]['f_critical']
                                    elif trained['name'] == 'UNEQ':
                                        crit_dist1 = trained['model']['class_models'][selected_class_1]['t2_critical']
                                        crit_dist2 = trained['model']['class_models'][selected_class_2]['t2_critical']

                                    # Filter test samples to only those belonging to the two selected classes
                                    mask_selected = np.isin(y_test, [selected_class_1, selected_class_2])
                                    dist_class1_test_filtered = dist_class1_test[mask_selected]
                                    dist_class2_test_filtered = dist_class2_test[mask_selected]
                                    y_test_filtered = y_test[mask_selected]

                                    # Preserve original 1-based sample indices for test data
                                    original_indices_test = np.where(mask_selected)[0]

                                    # For test data, we use the test_start offset to get actual row numbers from dataset
                                    if hasattr(X_test, 'index'):
                                        sample_names_test = X_test.index[original_indices_test].tolist()
                                    else:
                                        # Use test_start offset to show actual dataset row numbers
                                        sample_names_test = [str(test_start + i + 1) for i in original_indices_test]

                                    # Ensure y_test is in the correct format
                                    y_test_list = y_test_filtered.tolist() if hasattr(y_test_filtered, 'tolist') else list(y_test_filtered)

                                    # Debug info on data shapes
                                    with st.expander("🔍 Debug Information", expanded=False):
                                        st.write(f"**Total Test Samples**: {len(y_test)}")
                                        st.write(f"**Filtered Test Samples (selected 2 classes)**: {len(y_test_filtered)}")
                                        st.write(f"**Selected Classes**: {selected_class_1} (index {idx_class1}), {selected_class_2} (index {idx_class2})")
                                        st.write(f"**Distance Array Shape (all classes)**: {distances_array_test_all.shape}")
                                        st.write(f"**Distance to {selected_class_1}**: min={dist_class1_test_filtered.min():.3f}, max={dist_class1_test_filtered.max():.3f}, mean={dist_class1_test_filtered.mean():.3f}")
                                        st.write(f"**Distance to {selected_class_2}**: min={dist_class2_test_filtered.min():.3f}, max={dist_class2_test_filtered.max():.3f}, mean={dist_class2_test_filtered.mean():.3f}")
                                        st.write(f"**Critical Distance {selected_class_1}**: {crit_dist1:.3f}")
                                        st.write(f"**Critical Distance {selected_class_2}**: {crit_dist2:.3f}")
                                        st.write(f"**y_test_filtered type**: {type(y_test_filtered)}, length={len(y_test_filtered)}")
                                        st.write(f"**y_test_filtered preview (first 10)**: {y_test_list[:10]}")
                                        st.write(f"**Unique classes in filtered labels**: {np.unique(y_test_list)}")
                                        st.write(f"**Sample Names (first 10)**: {sample_names_test[:10]}")
                                        st.write(f"**Original Indices (first 10)**: {original_indices_test[:10]}")

                                    fig_coomans_test = plot_coomans(
                                        dist_class1=dist_class1_test_filtered,
                                        dist_class2=dist_class2_test_filtered,
                                        y_true=y_test_list,
                                        crit_dist1=crit_dist1,
                                        crit_dist2=crit_dist2,
                                        class_names=[str(selected_class_1), str(selected_class_2)],
                                        title=f"Coomans Plot - {trained['name']}: {selected_class_1} vs {selected_class_2} (Test)",
                                        normalize=False,
                                        sample_names=sample_names_test
                                    )
                                    st.plotly_chart(fig_coomans_test, use_container_width=True, key="coomans_test_tab4")

                                except Exception as e:
                                    st.error(f"❌ Could not generate Coomans plot: {str(e)}")

                                    # Enhanced debug information on error
                                    with st.expander("🐛 Error Debug Information", expanded=True):
                                        st.write("**Error Details:**")
                                        st.code(str(e))
                                        st.write("**Traceback:**")
                                        import traceback
                                        st.code(traceback.format_exc())
                                        st.write("**Data Diagnostics:**")
                                        try:
                                            st.write(f"- X_test_scaled shape: {X_test_scaled.shape if hasattr(X_test_scaled, 'shape') else 'N/A'}")
                                            st.write(f"- y_test shape/length: {y_test.shape if hasattr(y_test, 'shape') else len(y_test)}")
                                            st.write(f"- classes: {classes}")
                                            st.write(f"- trained['name']: {trained['name']}")
                                            if 'distances_array_test_all' in locals():
                                                st.write(f"- distances_array_test_all shape: {distances_array_test_all.shape}")
                                                st.write(f"- distances_array_test_all sample values: {distances_array_test_all[:3]}")
                                        except Exception as debug_err:
                                            st.write(f"Could not retrieve debug info: {debug_err}")

                            # Coomans Comparison: SIMCA vs UNEQ (Test Data)
                            if len(classes) == 2:
                                st.divider()
                                st.markdown("#### 📊 Coomans Comparison: SIMCA vs UNEQ (Test Data)")

                                if st.checkbox("Show SIMCA vs UNEQ Comparison (Test)", value=False, key="show_coomans_comparison_tab4"):
                                    st.info("Side-by-side comparison of SIMCA and UNEQ on test data")

                                    try:
                                        # Train both SIMCA and UNEQ for comparison on test data
                                        with st.spinner("Training SIMCA and UNEQ for test data comparison..."):
                                            # Get parameters
                                            n_pcs = tab1_data.get('n_pcs', 3)
                                            confidence_level = tab1_data.get('confidence_level', 0.95)

                                            # Train SIMCA on training data, predict on test
                                            simca_model_test = fit_simca(st.session_state.get('X_scaled'), tab1_data.get('y_labels'), n_pcs, confidence_level)
                                            simca_pred_detailed_test = predict_simca_detailed(X_test_scaled, simca_model_test)
                                            simca_distances_test = simca_pred_detailed_test['distances_per_class']

                                            # Train UNEQ on training data, predict on test
                                            uneq_model_test = fit_uneq(st.session_state.get('X_scaled'), tab1_data.get('y_labels'), n_pcs, confidence_level, use_pca=False)
                                            uneq_pred_detailed_test = predict_uneq_detailed(X_test_scaled, uneq_model_test)
                                            uneq_distances_test = uneq_pred_detailed_test['distances_per_class']

                                            # Get class names and thresholds
                                            cls1, cls2 = classes[0], classes[1]
                                            simca_crit1 = simca_model_test['class_models'][cls1]['f_critical']
                                            simca_crit2 = simca_model_test['class_models'][cls2]['f_critical']
                                            uneq_crit1 = uneq_model_test['class_models'][cls1]['t2_critical']
                                            uneq_crit2 = uneq_model_test['class_models'][cls2]['t2_critical']

                                            # Create subplots
                                            from plotly.subplots import make_subplots

                                            fig_comparison_test = make_subplots(
                                                rows=1, cols=2,
                                                subplot_titles=(
                                                    f"SIMCA (F-statistic)",
                                                    f"UNEQ (T²-statistic)"
                                                ),
                                                horizontal_spacing=0.12
                                            )

                                            # Prepare data
                                            y_test_list = y_test.tolist() if hasattr(y_test, 'tolist') else list(y_test)

                                            # Define colors for classes
                                            color_map = {cls1: '#1f77b4', cls2: '#ff7f0e'}  # Blue and Orange
                                            colors_test = [color_map[cls] for cls in y_test_list]

                                            # Left plot: SIMCA
                                            fig_comparison_test.add_trace(
                                                go.Scatter(
                                                    x=simca_distances_test[:, 0],
                                                    y=simca_distances_test[:, 1],
                                                    mode='markers',
                                                    marker=dict(color=colors_test, size=8, line=dict(width=1, color='white')),
                                                    name='Samples',
                                                    showlegend=False,
                                                    text=[f"Sample {test_start + i}<br>Class: {y_test_list[i]}" for i in range(len(y_test_list))],
                                                    hovertemplate='<b>%{text}</b><br>Dist to %s: %%{x:.3f}<br>Dist to %s: %%{y:.3f}<extra></extra>' % (cls1, cls2)
                                                ),
                                                row=1, col=1
                                            )

                                            # SIMCA critical lines
                                            fig_comparison_test.add_hline(y=simca_crit2, line_dash="dash", line_color="red",
                                                                        annotation_text=f"{cls2} threshold", row=1, col=1)
                                            fig_comparison_test.add_vline(x=simca_crit1, line_dash="dash", line_color="blue",
                                                                        annotation_text=f"{cls1} threshold", row=1, col=1)

                                            # Right plot: UNEQ
                                            fig_comparison_test.add_trace(
                                                go.Scatter(
                                                    x=uneq_distances_test[:, 0],
                                                    y=uneq_distances_test[:, 1],
                                                    mode='markers',
                                                    marker=dict(color=colors_test, size=8, line=dict(width=1, color='white')),
                                                    name='Samples',
                                                    showlegend=False,
                                                    text=[f"Sample {test_start + i}<br>Class: {y_test_list[i]}" for i in range(len(y_test_list))],
                                                    hovertemplate='<b>%{text}</b><br>Dist to %s: %%{x:.3f}<br>Dist to %s: %%{y:.3f}<extra></extra>' % (cls1, cls2)
                                                ),
                                                row=1, col=2
                                            )

                                            # UNEQ critical lines
                                            fig_comparison_test.add_hline(y=uneq_crit2, line_dash="dash", line_color="red",
                                                                        annotation_text=f"{cls2} threshold", row=1, col=2)
                                            fig_comparison_test.add_vline(x=uneq_crit1, line_dash="dash", line_color="blue",
                                                                        annotation_text=f"{cls1} threshold", row=1, col=2)

                                            # Update layout
                                            fig_comparison_test.update_xaxes(title_text=f"Distance to Class {cls1}", row=1, col=1)
                                            fig_comparison_test.update_yaxes(title_text=f"Distance to Class {cls2}", row=1, col=1)
                                            fig_comparison_test.update_xaxes(title_text=f"Distance to Class {cls1}", row=1, col=2)
                                            fig_comparison_test.update_yaxes(title_text=f"Distance to Class {cls2}", row=1, col=2)

                                            fig_comparison_test.update_layout(
                                                title_text="Coomans Plot Comparison: SIMCA vs UNEQ (Test Data)",
                                                height=600,
                                                width=1400,
                                                showlegend=False
                                            )

                                            st.plotly_chart(fig_comparison_test, use_container_width=True)

                                            # Add interpretation
                                            st.caption(
                                                "**Left (SIMCA)**: Uses F-statistic distances based on PCA models per class. "
                                                "**Right (UNEQ)**: Uses Mahalanobis T²-statistic distances with different dispersions per class. "
                                                "Points closer to origin in each plot indicate better fit to the respective class model."
                                            )

                                    except Exception as e:
                                        st.error(f"Could not generate SIMCA vs UNEQ comparison: {str(e)}")
                                        import traceback
                                        st.error(traceback.format_exc())

                            st.divider()

                            # Per-class metrics
                            st.markdown("### Per-Class Metrics (Test Data)")
                            per_class_test_data = []
                            for cls in classes:
                                per_class_test_data.append({
                                    'Class': cls,
                                    'Sensitivity %': f"{test_metrics['class_metrics'][cls]['sensitivity']:.2f}",
                                    'Specificity %': f"{test_metrics['class_metrics'][cls]['specificity']:.2f}",
                                    'Precision %': f"{test_metrics['class_metrics'][cls]['precision']:.2f}",
                                    'F1 %': f"{test_metrics['class_metrics'][cls]['f1_score']:.2f}",
                                    'Support': int(test_metrics['class_metrics'][cls]['support'])
                                })
                            per_class_test_df = pd.DataFrame(per_class_test_data)
                            st.dataframe(per_class_test_df, use_container_width=True, hide_index=True)

                            # Sensitivity & Specificity Detailed Table
                            st.markdown("#### Sensitivity & Specificity by Class")
                            st.info("Detailed breakdown of True Positive Rate (Sensitivity) and True Negative Rate (Specificity)")

                            # Create detailed sensitivity/specificity table
                            sens_spec_test_data = []
                            for cls in classes:
                                sens = test_metrics['class_metrics'][cls]['sensitivity']
                                spec = test_metrics['class_metrics'][cls]['specificity']
                                prec = test_metrics['class_metrics'][cls]['precision']
                                support = int(test_metrics['class_metrics'][cls]['support'])

                                # Determine status based on both metrics
                                if sens > 80 and spec > 80:
                                    status = "🟢 Good"
                                elif sens > 70 and spec > 70:
                                    status = "🟡 OK"
                                else:
                                    status = "🔴 Low"

                                sens_spec_test_data.append({
                                    'Class': cls,
                                    'Sensitivity %': sens,
                                    'Specificity %': spec,
                                    'Precision %': prec,
                                    'Support': support,
                                    'Status': status
                                })

                            sens_spec_test_df = pd.DataFrame(sens_spec_test_data)

                            # Apply color coding with styling
                            def color_test_metrics(row):
                                colors = []
                                for col in row.index:
                                    if col == 'Status':
                                        if '🟢' in str(row[col]):
                                            colors.append('background-color: #d4edda')  # Light green
                                        elif '🟡' in str(row[col]):
                                            colors.append('background-color: #fff3cd')  # Light yellow
                                        elif '🔴' in str(row[col]):
                                            colors.append('background-color: #f8d7da')  # Light red
                                        else:
                                            colors.append('')
                                    elif col in ['Sensitivity %', 'Specificity %', 'Precision %']:
                                        val = row[col]
                                        if val > 80:
                                            colors.append('background-color: #d4edda')  # Light green
                                        elif val > 70:
                                            colors.append('background-color: #fff3cd')  # Light yellow
                                        else:
                                            colors.append('background-color: #f8d7da')  # Light red
                                    else:
                                        colors.append('')
                                return colors

                            styled_sens_spec_test_df = sens_spec_test_df.style.apply(color_test_metrics, axis=1).format({
                                'Sensitivity %': '{:.2f}',
                                'Specificity %': '{:.2f}',
                                'Precision %': '{:.2f}'
                            })

                            st.dataframe(styled_sens_spec_test_df, use_container_width=True, hide_index=True)

                            # Add explanatory footnote
                            st.caption(
                                "**Sensitivity (True Positive Rate)**: Percentage of actual positives correctly identified for each class. "
                                "**Specificity (True Negative Rate)**: Percentage of actual negatives correctly identified. "
                                "**Precision**: Percentage of predicted positives that are correct. "
                                "**Thresholds**: 🟢 Good (>80%), 🟡 OK (70-80%), 🔴 Low (<70%)"
                            )

                            st.divider()

                            # === Category-Specific Analysis (Test) ===
                            st.markdown("### 🎯 Category-Specific Analysis (Test)")

                            # Class selector for Tab4
                            selected_class_tab4 = st.selectbox(
                                "Select class to analyze",
                                options=classes.tolist(),
                                key="class_selector_tab4"
                            )

                            # Metrics for selected class
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                sensitivity = test_metrics['class_metrics'][selected_class_tab4]['sensitivity']
                                st.metric(f"Sensitivity ({selected_class_tab4})", f"{sensitivity:.1f}%")
                            with col2:
                                specificity = test_metrics['class_metrics'][selected_class_tab4]['specificity']
                                st.metric(f"Specificity ({selected_class_tab4})", f"{specificity:.1f}%")
                            with col3:
                                precision = test_metrics['class_metrics'][selected_class_tab4]['precision']
                                st.metric(f"Precision ({selected_class_tab4})", f"{precision:.1f}%")

                            # Distance distribution for test set
                            st.markdown(f"#### Distance Distribution to Class {selected_class_tab4}")
                            try:
                                if trained['name'] in ['SIMCA', 'UNEQ']:
                                    # Get distances array and threshold for SIMCA/UNEQ
                                    if trained['name'] == 'SIMCA':
                                        pred_detailed = predict_simca_detailed(X_test_scaled, trained['model'])
                                        distances_array = pred_detailed['distances_per_class']
                                        threshold = trained['model']['class_models'][selected_class_tab4]['f_critical']
                                    else:  # UNEQ
                                        pred_detailed = predict_uneq_detailed(X_test_scaled, trained['model'])
                                        distances_array = pred_detailed['distances_per_class']
                                        threshold = trained['model']['class_models'][selected_class_tab4]['t2_critical']

                                    # Find the index of the selected class
                                    class_idx = list(classes).index(selected_class_tab4)
                                    distances_dict = {selected_class_tab4: distances_array[:, class_idx]}

                                    fig_dist_test = plot_distance_distributions(
                                        distances_dict,
                                        y_test,
                                        selected_class=selected_class_tab4,
                                        threshold=threshold,
                                        title=f"{trained['name']} Distance to Class {selected_class_tab4} (Test)"
                                    )
                                    st.plotly_chart(fig_dist_test, use_container_width=True)

                                elif trained['name'] in ['LDA', 'QDA']:
                                    # Get distances array for LDA/QDA
                                    if trained['name'] == 'LDA':
                                        _, distances_array = predict_lda(X_test_scaled, trained['model'])
                                    else:  # QDA
                                        _, distances_array = predict_qda(X_test_scaled, trained['model'])

                                    # Find the index of the selected class
                                    class_idx = list(classes).index(selected_class_tab4)
                                    distances_dict = {selected_class_tab4: distances_array[:, class_idx]}

                                    fig_dist_test = plot_distance_distributions(
                                        distances_dict,
                                        y_test,
                                        selected_class=selected_class_tab4,
                                        threshold=None,
                                        title=f"{trained['name']} Mahalanobis Distance to Class {selected_class_tab4} (Test)"
                                    )
                                    st.plotly_chart(fig_dist_test, use_container_width=True)

                                elif trained['name'] == 'kNN':
                                    # For kNN, show within-class distance statistics for test set
                                    cls_mask = y_test == selected_class_tab4
                                    X_cls_test = X_test_scaled[cls_mask]

                                    if X_cls_test.shape[0] > 1:
                                        within_dist = calculate_distance_matrix(
                                            X_cls_test, X_cls_test,
                                            metric=trained['parameters']['metric']
                                        )
                                        within_mean = np.mean(within_dist[np.triu_indices_from(within_dist, k=1)])
                                        within_std = np.std(within_dist[np.triu_indices_from(within_dist, k=1)])

                                        col1, col2, col3 = st.columns(3)
                                        with col1:
                                            st.metric("Avg Within-Class Distance", f"{within_mean:.3f}")
                                        with col2:
                                            st.metric("Std Deviation", f"{within_std:.3f}")
                                        with col3:
                                            st.metric("Class Samples", int(np.sum(cls_mask)))

                                        st.info(f"Within-class distance statistics for {selected_class_tab4} in test set using {trained['parameters']['metric']} metric")
                                    else:
                                        st.warning(f"Not enough samples in class {selected_class_tab4} in test set for distance analysis")

                            except Exception as e:
                                st.warning(f"Could not generate plot: {str(e)}")

                            # Misclassified samples (Test Set)
                            st.divider()
                            st.markdown("#### 🔍 Misclassified Samples Analysis (Test Set)")

                            misclassified_indices_test = np.where(y_test != y_pred_test)[0]
                            n_misclassified_test = len(misclassified_indices_test)

                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Misclassified", n_misclassified_test)
                            with col2:
                                st.metric("Correct", len(y_test) - n_misclassified_test)
                            with col3:
                                accuracy_test = (len(y_test) - n_misclassified_test) / len(y_test) * 100
                                st.metric("Accuracy", f"{accuracy_test:.1f}%")

                            if n_misclassified_test > 0:
                                misclass_data_test = []
                                for idx in misclassified_indices_test[:20]:
                                    misclass_data_test.append({
                                        'Sample Index': test_start + idx + 1,
                                        'True Class': y_test[idx],
                                        'Predicted Class': y_pred_test[idx],
                                        'Error': 'Misclassification'
                                    })

                                misclass_df_test = pd.DataFrame(misclass_data_test)
                                st.dataframe(misclass_df_test, use_container_width=True, hide_index=True)

                                if n_misclassified_test > 20:
                                    st.caption(f"Showing first 20 of {n_misclassified_test} misclassified samples")
                            else:
                                st.success("✅ All test samples correctly classified!")

                            st.divider()

                            # === Single Sample Analysis (Test) ===
                            st.markdown("### 📌 Single Sample Analysis (Test)")

                            # Reorder samples: misclassified first, then correct
                            misclassified_idx_test = np.where(y_test != y_pred_test)[0]
                            correct_idx_test = np.where(y_test == y_pred_test)[0]
                            ordered_indices_test = np.concatenate([misclassified_idx_test, correct_idx_test])

                            # Create sample names dictionary for test set
                            if hasattr(X_test, 'index'):
                                sample_names_dict_test = {i: X_test.index[i] for i in range(len(X_test))}
                            else:
                                sample_names_dict_test = {i: str(test_start + i + 1) for i in range(len(X_test))}

                            # Sample selector for Tab4
                            selected_sample_idx_tab4 = st.selectbox(
                                "Select sample to analyze",
                                options=ordered_indices_test,
                                format_func=lambda x: f"Sample {sample_names_dict_test[x]}: True={y_test[x]}, Pred={y_pred_test[x]}",
                                key="sample_selector_tab4"
                            )

                            # Sample details for test set
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("True Class", y_test[selected_sample_idx_tab4])
                            with col2:
                                st.metric("Predicted", y_pred_test[selected_sample_idx_tab4])
                            with col3:
                                match = "✅ Correct" if y_test[selected_sample_idx_tab4] == y_pred_test[selected_sample_idx_tab4] else "❌ Error"
                                st.metric("Result", match)

                            # Feature values
                            st.markdown("#### Feature Values (Test Sample)")
                            feature_vals_test = X_test[selected_sample_idx_tab4]
                            feature_df_test = pd.DataFrame({
                                'Feature': X_cols,
                                'Value': feature_vals_test
                            })
                            st.dataframe(feature_df_test, use_container_width=True, hide_index=True)

                            # Distance to each class
                            st.markdown("#### Distance to Each Class")

                            try:
                                distances_to_classes_test = []

                                if trained['name'] == 'LDA':
                                    # Get distances for all test samples
                                    _, distances_array = predict_lda(X_test_scaled, trained['model'])
                                    distances_to_classes_test = distances_array[selected_sample_idx_tab4, :].tolist()

                                elif trained['name'] == 'QDA':
                                    # Get distances for all test samples
                                    _, distances_array = predict_qda(X_test_scaled, trained['model'])
                                    distances_to_classes_test = distances_array[selected_sample_idx_tab4, :].tolist()

                                elif trained['name'] == 'kNN':
                                    # Calculate distance from selected test sample to each class centroid
                                    for cls in classes:
                                        cls_mask = y_test == cls
                                        X_cls_test = X_test_scaled[cls_mask]
                                        # Calculate distance to class centroid
                                        centroid = np.mean(X_cls_test, axis=0)
                                        dist = np.linalg.norm(X_test_scaled[selected_sample_idx_tab4] - centroid)
                                        distances_to_classes_test.append(dist)

                                elif trained['name'] == 'SIMCA':
                                    # Get distances for all test samples
                                    pred_detailed = predict_simca_detailed(X_test_scaled, trained['model'])
                                    distances_array = pred_detailed['distances_per_class']
                                    distances_to_classes_test = distances_array[selected_sample_idx_tab4, :].tolist()

                                elif trained['name'] == 'UNEQ':
                                    # Get distances for all test samples
                                    pred_detailed = predict_uneq_detailed(X_test_scaled, trained['model'])
                                    distances_array = pred_detailed['distances_per_class']
                                    distances_to_classes_test = distances_array[selected_sample_idx_tab4, :].tolist()

                                # Create distance dataframe
                                dist_df_test = pd.DataFrame({
                                    'Class': [str(c) for c in classes.tolist()],
                                    'Distance': [f"{d:.4f}" for d in distances_to_classes_test],
                                    'Match': ['✅ TRUE CLASS' if c == y_test[selected_sample_idx_tab4] else
                                              '🔵 PREDICTED' if c == y_pred_test[selected_sample_idx_tab4] else '⚪ Other'
                                              for c in classes]
                                })
                                st.dataframe(dist_df_test, use_container_width=True, hide_index=True)

                                # Visualization
                                fig_dist_sample_test = go.Figure()

                                # Determine bar colors
                                bar_colors_test = []
                                for c in classes:
                                    if c == y_test[selected_sample_idx_tab4]:
                                        bar_colors_test.append('#28a745')  # Green for true class
                                    elif c == y_pred_test[selected_sample_idx_tab4]:
                                        bar_colors_test.append('#ffc107')  # Orange for predicted class
                                    else:
                                        bar_colors_test.append('#dc3545')  # Red for other classes

                                fig_dist_sample_test.add_trace(go.Bar(
                                    x=[str(c) for c in classes.tolist()],
                                    y=distances_to_classes_test,
                                    marker_color=bar_colors_test,
                                    text=[f"{d:.4f}" for d in distances_to_classes_test],
                                    textposition='auto'
                                ))

                                fig_dist_sample_test.update_layout(
                                    title=f"Test Sample {test_start + selected_sample_idx_tab4} - Distance to Each Class",
                                    xaxis_title="Class",
                                    yaxis_title=f"Distance ({'Mahalanobis' if trained['name'] in ['LDA', 'QDA'] else 'F-statistic' if trained['name'] == 'SIMCA' else 'T²-statistic' if trained['name'] == 'UNEQ' else 'Euclidean'})",
                                    showlegend=False,
                                    height=400
                                )

                                st.plotly_chart(fig_dist_sample_test, use_container_width=True)

                                # Add interpretation
                                st.caption(
                                    "🟢 **Green bar**: True class | "
                                    "🟡 **Orange bar**: Predicted class | "
                                    "🔴 **Red bars**: Other classes. "
                                    f"Lower distance = Higher similarity ({trained['name']} classifier)"
                                )

                            except Exception as e:
                                st.error(f"Could not calculate distances for test sample: {str(e)}")
                                import traceback
                                st.error(traceback.format_exc())

                            st.divider()

                            # Quality assessment
                            st.markdown("### 🎖️ Model Quality Assessment")

                            if test_metrics['accuracy'] >= 90:
                                quality_status = "🟢 EXCELLENT"
                                quality_color = "green"
                            elif test_metrics['accuracy'] >= 80:
                                quality_status = "🟡 GOOD"
                                quality_color = "orange"
                            elif test_metrics['accuracy'] >= 70:
                                quality_status = "🟠 ACCEPTABLE"
                                quality_color = "orange"
                            else:
                                quality_status = "🔴 POOR"
                                quality_color = "red"

                            col1, col2 = st.columns([1, 1])
                            with col1:
                                st.markdown(f"**Quality Status**: {quality_status}")
                            with col2:
                                st.markdown(f"**Accuracy**: {test_metrics['accuracy']:.1f}%")

                            # Recommendation
                            if test_metrics['accuracy'] >= 80:
                                st.success("✅ Model is suitable for deployment")
                            elif test_metrics['accuracy'] >= 70:
                                st.warning("⚠️ Model acceptable, consider retraining or feature engineering")
                            else:
                                st.error("❌ Model performance insufficient - retrain with different parameters")

                        else:
                            st.info("ℹ️ True labels not available for test set - no validation metrics calculated")
                            st.markdown("**Predictions generated successfully** but cannot validate without true class labels")


if __name__ == "__main__":
    show()
