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

    st.markdown("# 🧬 Classification & Class-Modelling techniques")
    st.markdown("*Train classification models with clear X (features) and Y (target) separation*")

    # Initialize session state
    if 'classification_results' not in st.session_state:
        st.session_state.classification_results = {}
    if 'selected_classifier' not in st.session_state:
        st.session_state.selected_classifier = 'LDA'
    if 'trained_model' not in st.session_state:
        st.session_state.trained_model = None

    # Initialize local variables to avoid UnboundLocalError
    X_data = None
    y_labels = None
    classes = None
    X_train = None
    X_test = None
    y_train = None
    y_test = None
    X_train_scaled = None
    X_test_scaled = None

    # === CREATE TABS ===
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Data Setup (X, Y, Split)",
        "🎲 Classification Analysis",
        "🏆 Model Comparison",
        "📋 Test & Validation"
    ])

    # ========== TAB 1: DATA SETUP (X, Y, SPLIT) ==========
    with tab1:
        st.markdown("## 📊 Data Setup: X Matrix, Y Target, and Train/Test Split")

        # ========================================
        # STEP 1: SELECT PREDICTOR MATRIX X
        # ========================================
        st.markdown("### 📊 Step 1: Select Predictor Matrix (X)")

        st.info("""
        **X = Predictor Matrix** (the features/variables used to make predictions)
        - Contains ONLY numeric columns
        - Example: wavelengths, concentrations, spectral values
        - Dimensions: n_samples × n_variables
        """)

        # Load dataset from workspace
        if not WORKSPACE_AVAILABLE:
            st.error("❌ Workspace utilities not available")
            st.stop()

        try:
            datasets = get_workspace_datasets()
            available_datasets = list(datasets.keys())
        except Exception as e:
            st.error(f"❌ Error accessing workspace: {str(e)}")
            available_datasets = []

        if not available_datasets:
            st.error("❌ No datasets in workspace! Go to Data Handling and load data first.")
            st.stop()

        dataset_name = st.selectbox(
            "📂 Select dataset:",
            available_datasets,
            key="dataset_select",
            help="Choose which dataset contains both X and Y"
        )

        full_dataset = datasets[dataset_name]

        st.write(f"**Dataset info:** {full_dataset.shape[0]} rows × {full_dataset.shape[1]} columns")

        # === CHOOSE X COLUMNS ===
        st.markdown("#### 📋 Select X Column Range (numeric predictors)")

        all_columns = full_dataset.columns.tolist()
        numeric_cols = full_dataset.select_dtypes(include=[np.number]).columns.tolist()
        non_numeric_cols = [c for c in all_columns if c not in numeric_cols]

        col_info1, col_info2, col_info3 = st.columns(3)
        with col_info1:
            st.metric("Total columns", len(all_columns))
        with col_info2:
            st.metric("Numeric columns", len(numeric_cols))
        with col_info3:
            st.metric("Non-numeric columns", len(non_numeric_cols))

        if len(non_numeric_cols) > 0:
            st.write(f"**Non-numeric columns (potential Y):** {non_numeric_cols}")

        # Column range selection
        col_range1, col_range2 = st.columns(2)

        # Determine default range for X (numeric columns)
        if len(numeric_cols) > 0:
            first_numeric_idx = all_columns.index(numeric_cols[0]) + 1
            last_numeric_idx = all_columns.index(numeric_cols[-1]) + 1
        else:
            first_numeric_idx = 1
            last_numeric_idx = len(all_columns)

        with col_range1:
            first_col = st.number_input(
                "First column (1-based):",
                min_value=1,
                max_value=len(all_columns),
                value=first_numeric_idx,
                key="first_col_input",
                help="Start of predictor columns"
            )

        with col_range2:
            last_col = st.number_input(
                "Last column (1-based):",
                min_value=first_col,
                max_value=len(all_columns),
                value=last_numeric_idx,
                key="last_col_input",
                help="End of predictor columns"
            )

        # Extract X columns (convert 1-based to 0-based indexing)
        x_col_indices = list(range(first_col - 1, last_col))
        x_columns = [all_columns[i] for i in x_col_indices]

        X_full = full_dataset.iloc[:, x_col_indices].copy()

        # Validate X
        if X_full.shape[1] == 0:
            st.error("❌ No columns selected for X! Adjust column range.")
            st.stop()

        # Check if X is numeric
        non_numeric_in_x = X_full.select_dtypes(exclude=[np.number]).columns.tolist()
        if len(non_numeric_in_x) > 0:
            st.warning(f"⚠️ Non-numeric columns in X: {non_numeric_in_x}")
            st.info("These will be excluded from training")
            X_full = X_full.select_dtypes(include=[np.number])

        st.success(f"""
        ✅ **X Matrix Selected:**
        - Dimensions: {X_full.shape[0]} samples × {X_full.shape[1]} variables
        - Columns {first_col} to {last_col}: {x_columns[:5]}{'...' if len(x_columns) > 5 else ''}
        """)

        # Preview X
        with st.expander("👀 Preview X Matrix"):
            st.dataframe(X_full.head(10), use_container_width=True)
            st.write("**X Statistics:**")
            st.dataframe(X_full.describe(), use_container_width=True)

        st.divider()

        # ========================================
        # STEP 2: SELECT TARGET VARIABLE Y
        # ========================================
        st.markdown("### 🎯 Step 2: Select Target Variable (Y)")

        st.info("""
        **Y = Target Variable** (the category/class to predict)
        - Can be numeric (0,1,2,...) or categorical (A, B, C,...)
        - Must have SAME number of rows as X
        - Example: milk type, quality grade, classification
        """)

        # Y column selection
        st.markdown("#### 📋 Choose Y Column")

        st.write(f"**Available columns from dataset:**")

        # Show available Y columns (typically the ones we didn't use for X)
        remaining_cols = [c for c in all_columns if c not in x_columns]

        if len(remaining_cols) == 0:
            st.warning("⚠️ No columns left for Y! Adjust X column range.")
            st.stop()

        st.write(f"Potential Y columns: {remaining_cols}")

        y_col = st.selectbox(
            "🔍 Select Y column (target variable):",
            remaining_cols,
            key="y_col_select",
            help="This column will be the classification target"
        )

        y_full = full_dataset[y_col].copy()

        # Extract classes
        classes = np.unique(y_full)

        st.success(f"""
        ✅ **Y Vector Selected:**
        - Column: {y_col}
        - Length: {len(y_full)} samples
        - Data type: {y_full.dtype}
        """)

        # === VALIDATE X and Y ===
        st.markdown("#### ✔️ Validation")

        if len(X_full) != len(y_full):
            st.error(f"❌ Dimension mismatch! X has {len(X_full)} rows, Y has {len(y_full)} rows")
            st.stop()
        else:
            st.success(f"✅ X and Y have same length: {len(X_full)} samples")

        if y_full.isnull().any():
            st.warning(f"⚠️ Y has {y_full.isnull().sum()} null values")

        if X_full.isnull().any().any():
            st.warning(f"⚠️ X has {X_full.isnull().sum().sum()} null values total")

        # === CLASS INFORMATION ===
        st.markdown("#### 🏷️ Class Information")

        col_class1, col_class2, col_class3 = st.columns(3)

        with col_class1:
            st.metric("Number of classes", len(classes))

        with col_class2:
            st.metric("Total samples", len(y_full))

        with col_class3:
            min_class_size = pd.Series(y_full).value_counts().min()
            st.metric("Min class size", min_class_size)

        # Class distribution
        st.markdown("**Class Distribution:**")

        class_counts = pd.Series(y_full).value_counts().sort_index()

        col_dist1, col_dist2 = st.columns([2, 1])

        with col_dist1:
            # Bar chart
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=class_counts.index.astype(str),
                y=class_counts.values,
                marker_color=[f'hsl({i*360/len(classes)}, 70%, 50%)'
                             for i in range(len(classes))],
                text=class_counts.values,
                textposition='auto',
                hovertemplate='<b>Class %{x}</b><br>Count: %{y}<extra></extra>'
            ))
            fig.update_layout(
                title=f"Class Distribution - {y_col}",
                xaxis_title="Class",
                yaxis_title="Number of Samples",
                height=300,
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_dist2:
            st.write("**Counts:**")
            for cls, count in class_counts.items():
                pct = 100 * count / len(y_full)
                st.write(f"{cls}: {count} ({pct:.1f}%)")

        st.divider()

        # ========================================
        # STEP 3: TRAIN/TEST SPLIT (OPTIONAL!)
        # ========================================
        st.markdown("### 🔀 Step 3: Train/Test Split (Optional for Tab 4)")

        st.info("""
        **Split is OPTIONAL:**
        - ✅ **Enable split**: Reserve 30% for final holdout test in Tab 4
        - ❌ **No split**: Use all data for cross-validation (Tab 2)
        - Cross-validation (Tab 2) is ALWAYS stratified per class regardless of this choice
        """)

        # Checkbox to enable/disable split
        use_split = st.checkbox(
            "📊 Create 70-30 train/test split for Tab 4?",
            value=False,
            key="use_split_checkbox",
            help="If TRUE: reserve 30% for final holdout test in Tab 4. If FALSE: use full dataset for CV in Tab 2."
        )

        # Always save X_full and y_full regardless of split choice
        st.session_state['X_full'] = X_full
        st.session_state['y_full'] = y_full
        st.session_state['classes'] = classes
        st.session_state['dataset_name'] = dataset_name
        st.session_state['y_column'] = y_col
        st.session_state['x_columns'] = x_columns

        if use_split:
            st.markdown("**Split enabled** - 30% will be reserved for Tab 4 testing")

            # Random state selection
            random_state = st.number_input(
                "Random seed:",
                min_value=0,
                max_value=9999,
                value=42,
                key="random_state_split",
                help="For reproducible splits"
            )

            # Button to perform split
            if st.button("🔄 Create Stratified Split (70-30 per class)", type="primary", use_container_width=True):
                # Perform stratified split per class
                def stratified_split_per_class(X, y, test_size=0.3, random_state=None):
                    """
                    Split data 70-30 (or custom ratio) for EACH CLASS separately
                    """
                    from sklearn.model_selection import train_test_split

                    # Convert to DataFrame/Series if needed for easier handling
                    if not isinstance(X, pd.DataFrame):
                        X = pd.DataFrame(X)
                    if not isinstance(y, pd.Series):
                        y = pd.Series(y)

                    classes_list = np.unique(y)

                    X_train_list = []
                    X_test_list = []
                    y_train_list = []
                    y_test_list = []

                    st.write("**Split per class:**")

                    # For each class, split 70-30
                    for cls in classes_list:
                        # Get indices for this class
                        cls_mask = (y == cls)
                        X_cls = X[cls_mask]
                        y_cls = y[cls_mask]

                        # Split this class
                        X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(
                            X_cls, y_cls,
                            test_size=test_size,
                            random_state=random_state
                        )

                        # Append to lists
                        X_train_list.append(X_train_cls)
                        X_test_list.append(X_test_cls)
                        y_train_list.append(y_train_cls)
                        y_test_list.append(y_test_cls)

                        st.write(f"  - Class **{cls}**: {len(y_train_cls)} train ({100*len(y_train_cls)/len(y_cls):.0f}%), {len(y_test_cls)} test ({100*len(y_test_cls)/len(y_cls):.0f}%)")

                    # Concatenate all classes
                    X_train = pd.concat(X_train_list, ignore_index=True)
                    X_test = pd.concat(X_test_list, ignore_index=True)
                    y_train = pd.concat(y_train_list, ignore_index=True)
                    y_test = pd.concat(y_test_list, ignore_index=True)

                    # Shuffle (to mix classes)
                    if random_state is not None:
                        np.random.seed(random_state)

                    shuffle_idx_train = np.random.permutation(len(X_train))
                    shuffle_idx_test = np.random.permutation(len(X_test))

                    X_train = X_train.iloc[shuffle_idx_train].reset_index(drop=True)
                    X_test = X_test.iloc[shuffle_idx_test].reset_index(drop=True)
                    y_train = y_train.iloc[shuffle_idx_train].reset_index(drop=True)
                    y_test = y_test.iloc[shuffle_idx_test].reset_index(drop=True)

                    return X_train, X_test, y_train, y_test

                # Perform the split
                X_train, X_test, y_train, y_test = stratified_split_per_class(
                    X_full, y_full,
                    test_size=0.30,  # 30% test, 70% train
                    random_state=random_state
                )

                st.success("✅ Split completed!")

                # Show split info
                st.markdown("#### 📊 Split Summary")

                col_info1, col_info2, col_info3 = st.columns(3)
                with col_info1:
                    st.metric("Train samples (total)", len(X_train))
                with col_info2:
                    st.metric("Test samples (total)", len(X_test))
                with col_info3:
                    train_pct = 100 * len(X_train) / (len(X_train) + len(X_test))
                    st.metric("Train %", f"{train_pct:.1f}%")

                # Show per-class breakdown
                col_breakdown1, col_breakdown2 = st.columns(2)

                with col_breakdown1:
                    st.markdown("**Training set composition:**")
                    for cls in np.unique(y_train):
                        count = (y_train == cls).sum()
                        pct = 100 * count / len(y_train)
                        st.write(f"  - {cls}: {count} samples ({pct:.1f}%)")

                with col_breakdown2:
                    st.markdown("**Test set composition:**")
                    for cls in np.unique(y_test):
                        count = (y_test == cls).sum()
                        pct = 100 * count / len(y_test)
                        st.write(f"  - {cls}: {count} samples ({pct:.1f}%)")

                # Save to session state
                st.session_state['X_train'] = X_train
                st.session_state['X_test'] = X_test
                st.session_state['y_train'] = y_train
                st.session_state['y_test'] = y_test
                st.session_state['split_done'] = True

        else:
            # No split - use full dataset
            st.markdown("**No split** - Full dataset will be used for cross-validation in Tab 2")
            st.session_state['split_done'] = False
            st.info("💡 Click the checkbox above if you want to enable 70-30 split for Tab 4 testing")

        st.divider()

        # Show status and prepare data
        if st.session_state.get('split_done', False):
            st.success("✅ **Data split complete! 70% training, 30% test**")
        else:
            st.success("✅ **Data loaded! Full dataset ready for cross-validation**")

        # Always create tab1_data for backward compatibility with Tabs 2, 3, 4
        X_full = st.session_state.get('X_full')
        y_full = st.session_state.get('y_full')
        classes = st.session_state.get('classes')
        x_columns = st.session_state.get('x_columns')

        st.session_state['tab1_data'] = {
            'X_data': X_full,
            'y_labels': y_full,
            'classes': classes,
            'X_cols': x_columns,
            'scaling_method': 'autoscale',  # default, will be updated in Step 4
            'confidence_level': 0.95,
            'k_value': 5,
            'n_pcs': 3
        }

        st.divider()

        # ========================================
        # STEP 4: PREPROCESSING CONFIGURATION
        # ========================================
        st.markdown("### ⚙️ Step 4: Preprocessing Configuration")

        col1, col2 = st.columns(2)
        with col1:
            scaling_method = st.selectbox(
                "Scaling Method",
                options=['autoscale', 'center', 'scale', 'none'],
                index=0,
                key="tab1_scaling",
                help="Autoscale: center + scale; Center: subtract mean; Scale: divide by std; None: no scaling"
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
            # Get number of features from X_full (always available)
            X_full = st.session_state.get('X_full')
            n_features = X_full.shape[1]
            n_pcs = st.slider(
                "PCs (SIMCA/UNEQ)",
                min_value=1,
                max_value=min(10, n_features-1) if n_features > 1 else 1,
                value=min(3, n_features-1) if n_features > 1 else 1,
                key="tab1_pcs"
            )

        st.divider()

        # ========================================
        # STEP 5: MODEL SELECTION & TRAINING
        # ========================================
        st.markdown("### 🎯 Step 5: Model Selection & Training")

        # Determine which data to use based on split status
        split_done = st.session_state.get('split_done', False)

        if split_done:
            # Use training set (70%)
            X_for_training = st.session_state['X_train']
            y_for_training = st.session_state['y_train']
            X_test = st.session_state['X_test']
            st.info("Training on 70% training set (30% reserved for Tab 4)")
        else:
            # Use full dataset (100%)
            X_for_training = st.session_state.get('X_full')
            y_for_training = st.session_state.get('y_full')
            X_test = None
            st.info("Training on full dataset (100%) - No test set reserved")

        classes = st.session_state['classes']
        x_columns = st.session_state['x_columns']

        # Prepare data with scaling
        prep_data = prepare_training_test(
            X_for_training.values if hasattr(X_for_training, 'values') else X_for_training,
            y_for_training.values if hasattr(y_for_training, 'values') else y_for_training,
            X_test=X_test.values if X_test is not None and hasattr(X_test, 'values') else None,
            scaling_method=scaling_method
        )

        X_train_scaled = prep_data['X_train']
        X_test_scaled = prep_data.get('X_test')  # May be None if no split

        # Store scaled data and config for other tabs
        st.session_state['X_train_scaled'] = X_train_scaled
        if X_test_scaled is not None:
            st.session_state['X_test_scaled'] = X_test_scaled

        # Store for backward compatibility (Tab 2 CV might use these names)
        st.session_state['X_scaled'] = X_train_scaled
        st.session_state['y_labels'] = y_for_training

        st.session_state['scaling_method'] = scaling_method
        st.session_state['confidence_level'] = confidence_level
        st.session_state['k_value'] = k_value
        st.session_state['n_pcs'] = n_pcs

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
                    # Get y values as array
                    y_values = y_for_training.values if hasattr(y_for_training, 'values') else y_for_training

                    if selected_classifier == 'LDA':
                        model = fit_lda(X_train_scaled, y_values)
                    elif selected_classifier == 'QDA':
                        model = fit_qda(X_train_scaled, y_values)
                    elif selected_classifier == 'kNN':
                        model = fit_knn(X_train_scaled, y_values, metric=metric)
                    elif selected_classifier == 'SIMCA':
                        model = fit_simca(X_train_scaled, y_values, n_pcs, confidence_level)
                    elif selected_classifier == 'UNEQ':
                        model = fit_uneq(X_train_scaled, y_values, n_pcs, confidence_level, use_pca=False)

                    train_time = time.time() - train_start

                    # Store trained model with comprehensive parameters
                    st.session_state.trained_model = {
                        'name': selected_classifier,
                        'model': model,
                        'training_time': train_time,
                        'n_features': X_train_scaled.shape[1],
                        'n_samples': X_train_scaled.shape[0],
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
        st.markdown("## 🎲 Classification Analysis - Cross-Validation")

        # Check if data is available
        X_full = st.session_state.get('X_full')
        y_full = st.session_state.get('y_full')
        classes = st.session_state.get('classes')

        if X_full is None or y_full is None:
            st.warning("⚠️ No data available")
            st.info("💡 Go to Tab 1 and select X (features) and Y (target)")
            return

        # Determine which data to use for CV
        split_done = st.session_state.get('split_done', False)

        if split_done:
            # Use training set (70%)
            X_for_cv = st.session_state.get('X_train')
            y_for_cv = st.session_state.get('y_train')
            X_for_cv_scaled = st.session_state.get('X_train_scaled')
            cv_data_source = "training set (70%)"
            st.info(f"📊 **CV Data**: Using {cv_data_source} - Test set (30%) is reserved for Tab 4")
        else:
            # Use full dataset (100%)
            X_for_cv = X_full
            y_for_cv = y_full
            X_for_cv_scaled = st.session_state.get('X_scaled')  # Use X_scaled (saved in Step 5)
            cv_data_source = "full dataset (100%)"
            st.info(f"📊 **CV Data**: Using {cv_data_source} - No test set reserved (split disabled in Tab 1)")

        if X_for_cv is None or y_for_cv is None:
            st.error("❌ Data not found")
            st.info("💡 Go back to Tab 1 and complete data selection")
            return

        trained = st.session_state.get('trained_model')
        if trained is None:
            st.warning("⚠️ No model trained yet")
            st.info("💡 Train a model in Tab 1 first (Steps 4-5)")
            return

        st.success(f"🎯 **Model:** {trained['name']} | **Samples for CV:** {len(y_for_cv)} | **Features:** {trained['n_features']}")

        st.divider()

        # --- Cross-Validation Section ---
        st.markdown("### ✅ Cross-Validation Evaluation")

        st.info("""
        **Stratified K-Fold CV:**
        - Each fold maintains class proportions from the full dataset
        - Every class is represented in each fold
        - Provides robust performance estimates
        """)

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

        if st.button("🔄 Run Stratified Cross-Validation", type="primary", use_container_width=True, key="run_cv_btn_tab2"):
            import time

            # Check if we have scaled data
            if X_for_cv_scaled is None:
                st.error("❌ Scaled data not found")
                st.info("💡 Go to Tab 1 and complete Steps 4-5 (configure preprocessing and train model)")
                st.stop()

            # Validate data consistency before running CV
            if len(y_for_cv) != len(X_for_cv_scaled):
                st.error(f"❌ Data inconsistency detected: {len(y_for_cv)} labels vs {len(X_for_cv_scaled)} samples")
                st.warning("⚠️ Please return to Tab 1 and retrain the model to ensure data consistency.")
                st.stop()

            cv_start = time.time()

            with st.spinner(f"Running {n_folds}-fold cross-validation..."):
                try:
                    # Get preprocessing params from session state
                    n_pcs = st.session_state.get('n_pcs', 3)
                    confidence_level = st.session_state.get('confidence_level', 0.95)

                    # Prepare CV parameters based on classifier type
                    if trained['name'] == 'LDA':
                        cv_results = cross_validate_classifier(
                            X_for_cv_scaled, y_for_cv.values,
                            classifier_type='lda',
                            n_folds=n_folds,
                            classifier_params={
                                'use_pca': trained['parameters'].get('use_pca', False)
                            },
                            random_state=random_seed
                        )
                    elif trained['name'] == 'QDA':
                        cv_results = cross_validate_classifier(
                            X_for_cv_scaled, y_for_cv.values,
                            classifier_type='qda',
                            n_folds=n_folds,
                            classifier_params={
                                'use_pca': trained['parameters'].get('use_pca', False)
                            },
                            random_state=random_seed
                        )
                    elif trained['name'] == 'kNN':
                        cv_results = cross_validate_classifier(
                            X_for_cv_scaled, y_for_cv.values,
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
                            X_for_cv_scaled, y_for_cv.values,
                            classifier_type='simca',
                            n_folds=n_folds,
                            classifier_params={
                                'n_components': n_pcs,
                                'confidence_level': confidence_level
                            },
                            random_state=random_seed
                        )
                    elif trained['name'] == 'UNEQ':
                        cv_results = cross_validate_classifier(
                            X_for_cv_scaled, y_for_cv.values,
                            classifier_type='uneq',
                            n_folds=n_folds,
                            classifier_params={
                                'n_components': n_pcs,
                                'confidence_level': confidence_level,
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

            # Validate that y_for_cv and y_pred_cv have the same length
            if len(y_for_cv) != len(y_pred_cv):
                st.error(f"❌ Data mismatch detected: Training labels ({len(y_for_cv)} samples) don't match CV predictions ({len(y_pred_cv)} samples)")
                st.warning("This usually happens when the model was trained with different data. Please retrain the model in Tab 1.")
                st.stop()

            cv_metrics = compute_classification_metrics(y_for_cv.values, y_pred_cv, classes)

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
                cv_metrics['classes'].tolist(),
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
                        pred_detailed = predict_simca_detailed(X_for_cv_scaled, trained['model'])
                        distances_array_all = pred_detailed['distances_per_class']
                    elif trained['name'] == 'UNEQ':
                        pred_detailed = predict_uneq_detailed(X_for_cv_scaled, trained['model'])
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
                    # Convert y_for_cv to array for consistent indexing
                    y_for_cv_arr = y_for_cv.values if hasattr(y_for_cv, 'values') else np.array(y_for_cv)

                    mask_selected = np.isin(y_for_cv_arr, [selected_class_1, selected_class_2])
                    dist_class1_filtered = dist_class1[mask_selected]
                    dist_class2_filtered = dist_class2[mask_selected]
                    y_filtered = y_for_cv_arr[mask_selected]

                    # Preserve original 1-based sample indices
                    original_indices = np.where(mask_selected)[0]

                    if hasattr(X_for_cv_scaled, 'index'):
                        sample_names = X_for_cv_scaled.index[original_indices].tolist()
                    else:
                        sample_names = [str(i+1) for i in original_indices]

                    # Ensure labels are in the correct format
                    y_true_list = y_filtered.tolist() if hasattr(y_filtered, 'tolist') else list(y_filtered)

                    # Debug info on data shapes
                    with st.expander("🔍 Debug Information", expanded=False):
                        st.write(f"**Total Samples**: {len(y_for_cv)}")
                        st.write(f"**Filtered Samples (selected 2 classes)**: {len(y_filtered)}")
                        st.write(f"**Selected Classes**: {selected_class_1} (index {idx_class1}), {selected_class_2} (index {idx_class2})")
                        st.write(f"**Distance Array Shape (all classes)**: {distances_array_all.shape}")
                        st.write(f"**Distance to {selected_class_1}**: min={dist_class1_filtered.min():.3f}, max={dist_class1_filtered.max():.3f}, mean={dist_class1_filtered.mean():.3f}")
                        st.write(f"**Distance to {selected_class_2}**: min={dist_class2_filtered.min():.3f}, max={dist_class2_filtered.max():.3f}, mean={dist_class2_filtered.mean():.3f}")
                        st.write(f"**Critical Distance {selected_class_1}**: {crit_dist1:.3f}")
                        st.write(f"**Critical Distance {selected_class_2}**: {crit_dist2:.3f}")
                        st.write(f"**y_filtered type**: {type(y_filtered)}, length={len(y_filtered)}")
                        st.write(f"**y_filtered preview (first 10)**: {y_true_list[:10]}")
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
                            st.write(f"- X_for_cv_scaled shape: {X_for_cv_scaled.shape if hasattr(X_for_cv_scaled, 'shape') else 'N/A'}")
                            st.write(f"- y_for_cv shape/length: {y_for_cv.shape if hasattr(y_for_cv, 'shape') else len(y_for_cv)}")
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
                            simca_model = fit_simca(X_for_cv_scaled, y_for_cv, n_pcs, confidence_level)
                            simca_pred_detailed = predict_simca_detailed(X_for_cv_scaled, simca_model)
                            simca_distances = simca_pred_detailed['distances_per_class']

                            # Train UNEQ
                            uneq_model = fit_uneq(X_for_cv_scaled, y_for_cv, n_pcs, confidence_level, use_pca=False)
                            uneq_pred_detailed = predict_uneq_detailed(X_for_cv_scaled, uneq_model)
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
                            y_true_list = y_for_cv.tolist() if hasattr(y_for_cv, 'tolist') else list(y_for_cv)

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
                class_support[cls] = int(np.sum(y_for_cv == cls))

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

            misclassified_indices = np.where(y_for_cv != y_pred_cv)[0]
            n_misclassified = len(misclassified_indices)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Misclassified", n_misclassified)
            with col2:
                st.metric("Correct", len(y_for_cv) - n_misclassified)
            with col3:
                accuracy = (len(y_for_cv) - n_misclassified) / len(y_for_cv) * 100
                st.metric("Accuracy", f"{accuracy:.1f}%")

            if n_misclassified > 0:
                misclass_data = []
                for idx in misclassified_indices[:20]:
                    misclass_data.append({
                        'Sample Index': idx + 1,
                        'True Class': y_for_cv[idx],
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
                    pred_detailed = predict_simca_detailed(X_for_cv_scaled, trained['model'])
                    distances_array = pred_detailed['distances_per_class']
                elif trained['name'] == 'UNEQ':
                    pred_detailed = predict_uneq_detailed(X_for_cv_scaled, trained['model'])
                    distances_array = pred_detailed['distances_per_class']

                for i, cls in enumerate(classes):
                    distances_dict = {cls: distances_array[:, i]}

                    if trained['name'] == 'SIMCA':
                        threshold = trained['model']['class_models'][cls]['f_critical']
                    else:
                        threshold = trained['model']['class_models'][cls]['t2_critical']

                    fig_dist = plot_distance_distributions(
                        distances_dict,
                        y_for_cv,
                        selected_class=cls,
                        threshold=threshold,
                        title=f"{trained['name']} Distance to Class {cls}"
                    )
                    st.plotly_chart(fig_dist, use_container_width=True)

            elif trained['name'] in ['LDA', 'QDA']:
                st.info("📊 Mahalanobis distance distributions to each class centroid")

                if trained['name'] == 'LDA':
                    y_pred, distances_array = predict_lda(X_for_cv_scaled, trained['model'])
                elif trained['name'] == 'QDA':
                    y_pred, distances_array = predict_qda(X_for_cv_scaled, trained['model'])

                for i, cls in enumerate(classes):
                    distances_dict = {cls: distances_array[:, i]}

                    fig_dist = plot_distance_distributions(
                        distances_dict,
                        y_for_cv,
                        selected_class=cls,
                        threshold=None,
                        title=f"{trained['name']} Distance to Class {cls}"
                    )
                    st.plotly_chart(fig_dist, use_container_width=True)

            elif trained['name'] == 'kNN':
                st.info(f"📊 Within-class {trained['parameters']['metric']} distance statistics for kNN classifier")

                st.markdown("**Average distances within each class:**")

                distance_summary = []
                for cls in classes:
                    cls_mask = y_for_cv == cls
                    X_cls = X_for_cv_scaled[cls_mask]

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
            if trained['name'] in ['LDA', 'QDA', 'SIMCA', 'UNEQ']:
                st.markdown(f"#### Distance Distribution to Class {selected_class_tab2}")
            elif trained['name'] == 'kNN':
                st.markdown(f"#### Within-Class Distance Statistics for {selected_class_tab2}")

            try:
                if trained['name'] in ['SIMCA', 'UNEQ']:
                    # Get distances array and threshold for SIMCA/UNEQ
                    if trained['name'] == 'SIMCA':
                        pred_detailed = predict_simca_detailed(X_for_cv_scaled, trained['model'])
                        distances_array = pred_detailed['distances_per_class']
                        threshold = trained['model']['class_models'][selected_class_tab2]['f_critical']
                    else:  # UNEQ
                        pred_detailed = predict_uneq_detailed(X_for_cv_scaled, trained['model'])
                        distances_array = pred_detailed['distances_per_class']
                        threshold = trained['model']['class_models'][selected_class_tab2]['t2_critical']

                    # Find the index of the selected class
                    class_idx = list(classes).index(selected_class_tab2)
                    distances_dict = {selected_class_tab2: distances_array[:, class_idx]}

                    fig_dist = plot_distance_distributions(
                        distances_dict,
                        y_for_cv,
                        selected_class=selected_class_tab2,
                        threshold=threshold,
                        title=f"{trained['name']} Distance to Class {selected_class_tab2}"
                    )
                    st.plotly_chart(fig_dist, use_container_width=True)

                elif trained['name'] in ['LDA', 'QDA']:
                    # Get distances array for LDA/QDA
                    if trained['name'] == 'LDA':
                        y_pred, distances_array = predict_lda(X_for_cv_scaled, trained['model'])
                    else:  # QDA
                        y_pred, distances_array = predict_qda(X_for_cv_scaled, trained['model'])

                    # Find the index of the selected class
                    class_idx = list(classes).index(selected_class_tab2)
                    distances_dict = {selected_class_tab2: distances_array[:, class_idx]}

                    fig_dist = plot_distance_distributions(
                        distances_dict,
                        y_for_cv,
                        selected_class=selected_class_tab2,
                        threshold=None,
                        title=f"{trained['name']} Mahalanobis Distance to Class {selected_class_tab2}"
                    )
                    st.plotly_chart(fig_dist, use_container_width=True)

                elif trained['name'] == 'kNN':
                    # For kNN, show within-class distance statistics
                    cls_mask = y_for_cv == selected_class_tab2
                    X_cls = X_for_cv_scaled[cls_mask]

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

            # Convert to arrays for consistent indexing
            y_for_cv_arr = y_for_cv.values if hasattr(y_for_cv, 'values') else np.array(y_for_cv)

            # Reorder samples: misclassified first, then correct
            misclassified_idx = np.where(y_for_cv_arr != y_pred_cv)[0]
            correct_idx = np.where(y_for_cv_arr == y_pred_cv)[0]
            ordered_indices = np.concatenate([misclassified_idx, correct_idx])

            # Create sample names dictionary
            if hasattr(X_for_cv, 'index'):
                sample_names_dict = {i: X_for_cv.index[i] for i in range(len(X_for_cv))}
            else:
                sample_names_dict = {i: str(i+1) for i in range(len(X_for_cv))}

            # Sample selector with formatted display
            selected_sample_idx_tab2 = st.selectbox(
                "Select sample to analyze",
                options=ordered_indices,
                format_func=lambda x: f"Sample {sample_names_dict[x]}: True={y_for_cv_arr[x]}, Pred={y_pred_cv[x]}",
                key="sample_selector_tab2"
            )

            # Sample details
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("True Class", y_for_cv_arr[selected_sample_idx_tab2])
            with col2:
                st.metric("Predicted", y_pred_cv[selected_sample_idx_tab2])
            with col3:
                match = "✅ Correct" if y_for_cv_arr[selected_sample_idx_tab2] == y_pred_cv[selected_sample_idx_tab2] else "❌ Error"
                st.metric("Result", match)

            # Feature values for selected sample
            st.markdown("#### Feature Values")
            # Get feature values from X_for_cv
            if hasattr(X_for_cv, 'iloc'):
                feature_vals = X_for_cv.iloc[selected_sample_idx_tab2]
            else:
                feature_vals = X_for_cv[selected_sample_idx_tab2]

            # Get feature names
            x_columns = st.session_state.get('x_columns', [f'Feature {i}' for i in range(len(feature_vals))])

            feature_df = pd.DataFrame({
                'Feature': x_columns,
                'Value': feature_vals
            })
            st.dataframe(feature_df, use_container_width=True, hide_index=True)

            # Distance to each class (classifier-specific)
            st.markdown("#### Distance to Each Class")

            try:
                distances_to_classes = []

                if trained['name'] == 'LDA':
                    # Get distances for all samples
                    _, distances_array = predict_lda(X_for_cv_scaled, trained['model'])
                    distances_to_classes = distances_array[selected_sample_idx_tab2, :].tolist()

                elif trained['name'] == 'QDA':
                    # Get distances for all samples
                    _, distances_array = predict_qda(X_for_cv_scaled, trained['model'])
                    distances_to_classes = distances_array[selected_sample_idx_tab2, :].tolist()

                elif trained['name'] == 'kNN':
                    # Calculate distance from selected sample to each class centroid using specified metric
                    from scipy.spatial.distance import cdist
                    sample = X_for_cv_scaled[selected_sample_idx_tab2].reshape(1, -1)
                    for cls in classes:
                        cls_mask = y_for_cv == cls
                        X_cls = X_for_cv_scaled[cls_mask]
                        # Calculate distance to class centroid
                        centroid = np.mean(X_cls, axis=0).reshape(1, -1)
                        dist = cdist(sample, centroid, metric=trained['parameters']['metric'])[0, 0]
                        distances_to_classes.append(dist)

                elif trained['name'] == 'SIMCA':
                    # Get distances for all samples
                    pred_detailed = predict_simca_detailed(X_for_cv_scaled, trained['model'])
                    distances_array = pred_detailed['distances_per_class']
                    distances_to_classes = distances_array[selected_sample_idx_tab2, :].tolist()

                elif trained['name'] == 'UNEQ':
                    # Get distances for all samples
                    pred_detailed = predict_uneq_detailed(X_for_cv_scaled, trained['model'])
                    distances_array = pred_detailed['distances_per_class']
                    distances_to_classes = distances_array[selected_sample_idx_tab2, :].tolist()

                # Create distance dataframe
                dist_df = pd.DataFrame({
                    'Class': [str(c) for c in classes.tolist()],
                    'Distance': [f"{d:.4f}" for d in distances_to_classes],
                    'Match': ['✅ TRUE CLASS' if c == y_for_cv[selected_sample_idx_tab2] else
                              '🔵 PREDICTED' if c == y_pred_cv[selected_sample_idx_tab2] else '⚪ Other'
                              for c in classes]
                })
                st.dataframe(dist_df, use_container_width=True, hide_index=True)

                # Visualization
                fig_dist_sample = go.Figure()

                # Determine bar colors
                bar_colors = []
                for c in classes:
                    if c == y_for_cv[selected_sample_idx_tab2]:
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

                # Determine distance metric label
                if trained['name'] in ['LDA', 'QDA']:
                    distance_label = 'Mahalanobis'
                elif trained['name'] == 'SIMCA':
                    distance_label = 'F-statistic'
                elif trained['name'] == 'UNEQ':
                    distance_label = 'T²-statistic'
                elif trained['name'] == 'kNN':
                    distance_label = f"{trained['parameters']['metric'].capitalize()} to Centroid"
                else:
                    distance_label = 'Distance'

                fig_dist_sample.update_layout(
                    title=f"Sample {selected_sample_idx_tab2} - Distance to Each Class",
                    xaxis_title="Class",
                    yaxis_title=f"Distance ({distance_label})",
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
                        X_train_scaled = prep_data['X_train']

                        # Run CV for each classifier
                        cv_results_dict = {}
                        models_data = []

                        for clf_name in classifiers_to_test:
                            clf_start = time.time()

                            if clf_name == 'LDA':
                                cv_res = cross_validate_classifier(
                                    X_train_scaled, y_labels,
                                    classifier_type='lda',
                                    n_folds=cv_folds,
                                    classifier_params=None,
                                    random_state=42
                                )
                            elif clf_name == 'QDA':
                                cv_res = cross_validate_classifier(
                                    X_train_scaled, y_labels,
                                    classifier_type='qda',
                                    n_folds=cv_folds,
                                    classifier_params=None,
                                    random_state=42
                                )
                            elif clf_name == 'kNN':
                                cv_res = cross_validate_classifier(
                                    X_train_scaled, y_labels,
                                    classifier_type='knn',
                                    n_folds=cv_folds,
                                    classifier_params={'k': 5, 'metric': 'euclidean'},
                                    random_state=42
                                )
                            elif clf_name == 'SIMCA':
                                cv_res = cross_validate_classifier(
                                    X_train_scaled, y_labels,
                                    classifier_type='simca',
                                    n_folds=cv_folds,
                                    classifier_params={'n_components': 3, 'confidence_level': 0.95},
                                    random_state=42
                                )
                            elif clf_name == 'UNEQ':
                                cv_res = cross_validate_classifier(
                                    X_train_scaled, y_labels,
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

    # ========== TAB 4: TEST & VALIDATION ==========
    with tab4:
        st.markdown("## 📋 Test Set Validation")
        st.markdown("*Final holdout test evaluation*")

        if st.session_state.trained_model is None:
            st.warning("⚠️ Train a model first in Tab 1 before testing")
            st.info("💡 Go to **Tab 1: Setup & Configuration** and click 'Train Model'")
        else:
            trained = st.session_state.trained_model

            # --- CHECK FOR OPTIONAL TRAIN/TEST SPLIT FROM TAB 1 ---
            st.markdown("## 📥 Section 1: Select Test Data")

            split_done = st.session_state.get('split_done', False)

            if split_done:
                # OPTION 1: Use split from Tab 1
                st.success("✅ **Train/test split found from Tab 1!**")
                st.info(f"🤖 **Model**: {trained['name']} | Features: {trained['n_features']}")

                use_tab1_split = st.checkbox(
                    "Use 30% holdout test set from Tab 1?",
                    value=True,
                    key="use_tab1_split_checkbox",
                    help="Test on the reserved 30% from Tab 1 split"
                )

                if use_tab1_split:
                    # Get test set from session state
                    X_test = st.session_state.get('X_test')
                    y_test = st.session_state.get('y_test')
                    X_test_scaled = st.session_state.get('X_test_scaled')
                    classes = st.session_state.get('classes')
                    x_columns = st.session_state.get('x_columns')

                    if X_test is None or y_test is None or X_test_scaled is None:
                        st.error("❌ Holdout test set not found in session state")
                        st.info("💡 Return to Tab 1 and recreate the split")
                        return

                    st.divider()

                    # --- Display Test Set Info ---
                    st.markdown("## 📥 Holdout Test Set from Tab 1")

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Test Samples", len(X_test))
                    with col2:
                        st.metric("Features", len(x_columns))
                    with col3:
                        st.metric("Classes", len(np.unique(y_test)))

                    st.info("""
                    **Holdout Test Evaluation:**
                    - Uses the 30% reserved in Tab 1 (never seen during training or CV)
                    - Provides unbiased performance estimate
                    - Final validation before deployment
                    """)

                    st.divider()

                    # Set flags for unified prediction section
                    has_true_labels = True
                    test_data_ready = True
                    test_data_source = "Tab 1 Holdout (30%)"

                else:
                    # User unchecked - load from workspace instead
                    st.info("💡 Loading test data from workspace...")
                    test_data_ready = False  # Will be set after workspace loading

            else:
                # OPTION 2: Load from workspace (old behavior)
                st.warning("⚠️ **No train/test split from Tab 1**")
                st.info("""
                **Note**: Train/test split was NOT enabled in Tab 1

                **Options**:
                1. **Load test data from workspace** (continue below)
                2. Go to Tab 1 → Enable "Create 70-30 split" → Recreate split for better validation
                3. Or use Tab 2 Cross-Validation results
                """)

                st.divider()
                test_data_ready = False  # Will be set after workspace loading

            # --- WORKSPACE DATASET LOADER (if needed) ---
            if not locals().get('test_data_ready', False):
                st.markdown("## 📥 Load Test Data from Workspace")

                # Get available datasets (imported at top of file)
                if not WORKSPACE_AVAILABLE:
                    st.error("❌ Workspace utilities not available")
                    st.info("💡 Check workspace_utils.py is present")
                    return

                available_datasets = get_workspace_datasets()

                if len(available_datasets) == 0:
                    st.error("❌ No datasets available in workspace")
                    st.info("💡 Load data in **Data Handling** page first")
                    return

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
                    st.metric("Datasets", len(available_datasets))

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
                    if all(col in test_subset_df.columns for col in x_columns):
                        X_test = test_subset_df[x_columns].values

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
                            st.metric("Features Used", len(x_columns))
                        with validation_cols[2]:
                            st.metric("True Labels", "Yes" if has_true_labels else "No")
                        with validation_cols[3]:
                            st.metric("Status", "✅ Ready")

                        if len(X_test) == 0:
                            st.error("❌ No samples in test subset!")
                            st.stop()

                        if len(X_test) != expected_rows:
                            st.warning(f"⚠️ Sample count mismatch: expected {expected_rows}, got {len(X_test)}")

                        # Get tab1_data for scaling parameters
                        tab1_data = st.session_state.get('tab1_data', {})
                        if not tab1_data:
                            st.error("❌ Training data configuration not found")
                            st.info("💡 Return to Tab 1 and train a model first")
                            return

                        # Prepare/scale test data using training scaling parameters
                        y_train = tab1_data['y_labels']  # Only needed for prepare_training_test
                        prep_data = prepare_training_test(
                            tab1_data.get('X_data').values,
                            y_train,
                            X_test,
                            scaling_method=tab1_data.get('scaling_method', 'autoscale')
                        )
                        X_test_scaled = prep_data['X_test']

                        # Mark workspace loading as complete
                        test_data_ready = True
                        test_data_source = f"Workspace: {selected_test_dataset}"

                        st.divider()

            # --- UNIFIED PREDICTION SECTION (for both Tab 1 split and workspace data) ---
            if locals().get('test_data_ready', False) and 'X_test_scaled' in locals():
                # Get tab1_data if not already loaded (needed for kNN k_value)
                if 'tab1_data' not in locals():
                    tab1_data = st.session_state.get('tab1_data', {})

                # --- SECTION 2: Make Predictions ---
                st.markdown("## 🎯 Section 2: Generate Predictions")

                # Show test data source
                st.info(f"📊 **Test Data Source**: {test_data_source} | Samples: {len(X_test)} | Features: {len(x_columns)}")

                # ===== TEST DATA PREVIEW =====
                st.markdown("### 📋 Test Data Preview")

                # Show first few rows of test data
                test_preview_rows = min(5, len(X_test))
                preview_df = pd.DataFrame(
                    X_test[:test_preview_rows],
                    columns=x_columns
                )

                st.info(f"📊 Showing first {test_preview_rows} rows of {len(X_test)} test samples")
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

                st.success(f"✅ Predictions generated successfully in {test_pred_time:.3f}s")

                st.divider()

                # --- SECTION 3: Predictions Table ---
                st.markdown("## 📊 Section 3: Predictions Summary")

                # Create predictions table with explicit formatting
                pred_df = pd.DataFrame({
                    'Sample #': [f"{i+1}" for i in range(len(y_pred_test))],
                    'Predicted Class': [str(cls) for cls in y_pred_test],
                    'True Class': [str(cls) for cls in y_test] if has_true_labels else ['?' for _ in y_pred_test],
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

                    # Validate class compatibility between training and test sets
                    classes_in_test = np.unique(y_test)
                    classes_in_training = classes

                    # Check for common classes
                    common_classes = np.intersect1d(classes_in_training, classes_in_test)

                    if len(common_classes) == 0:
                        # NO overlap - completely different datasets
                        st.error("❌ **Class Mismatch: Cannot Calculate Metrics**")
                        st.error(
                            f"**Training classes**: {', '.join(map(str, classes_in_training.tolist()))}\n\n"
                            f"**Test classes**: {', '.join(map(str, classes_in_test.tolist()))}\n\n"
                            f"**Common classes**: None"
                        )
                        st.warning(
                            "⚠️ **The test dataset contains completely different classes than the training data.**\n\n"
                            "**Possible causes:**\n"
                            "1. Wrong dataset selected from workspace\n"
                            "2. Different encoding/naming for classes\n"
                            "3. Test set from a different experiment\n\n"
                            "**Solution:** Select a test dataset with the same class labels as your training data."
                        )

                        # Show predictions table but skip metrics
                        st.info("💡 Predictions were generated, but validation metrics cannot be calculated without common classes.")

                    else:
                        # Some overlap exists - proceed with metrics
                        missing_classes = np.setdiff1d(classes_in_training, classes_in_test)
                        extra_classes = np.setdiff1d(classes_in_test, classes_in_training)

                        if len(missing_classes) > 0:
                            st.warning(
                                f"⚠️ **Test set is missing {len(missing_classes)} training class(es)**: "
                                f"{', '.join(map(str, missing_classes.tolist()))}"
                            )

                        if len(extra_classes) > 0:
                            st.warning(
                                f"⚠️ **Test set contains {len(extra_classes)} unknown class(es)**: "
                                f"{', '.join(map(str, extra_classes.tolist()))} (not seen during training)"
                            )

                        if len(missing_classes) > 0 or len(extra_classes) > 0:
                            st.info(
                                f"ℹ️ Metrics will be calculated for {len(common_classes)} common class(es): "
                                f"{', '.join(map(str, common_classes.tolist()))}"
                            )

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
                            test_metrics['classes'].tolist(),
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
                                    test_start_offset = locals().get('test_start', 1)
                                    sample_names_test = [str(test_start_offset + i) for i in original_indices_test]

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
                                        simca_model_test = fit_simca(st.session_state.get('X_train_scaled'), tab1_data.get('y_train'), n_pcs, confidence_level)
                                        simca_pred_detailed_test = predict_simca_detailed(X_test_scaled, simca_model_test)
                                        simca_distances_test = simca_pred_detailed_test['distances_per_class']

                                        # Train UNEQ on training data, predict on test
                                        uneq_model_test = fit_uneq(st.session_state.get('X_train_scaled'), tab1_data.get('y_train'), n_pcs, confidence_level, use_pca=False)
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
                                                text=[f"Sample {locals().get('test_start', 1) + i}<br>Class: {y_test_list[i]}" for i in range(len(y_test_list))],
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
                                                text=[f"Sample {locals().get('test_start', 1) + i}<br>Class: {y_test_list[i]}" for i in range(len(y_test_list))],
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
                        if trained['name'] in ['LDA', 'QDA', 'SIMCA', 'UNEQ']:
                            st.markdown(f"#### Distance Distribution to Class {selected_class_tab4}")
                        elif trained['name'] == 'kNN':
                            st.markdown(f"#### Within-Class Distance Statistics for {selected_class_tab4}")

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
                                # Calculate sample index - use test_start if available, else just idx+1
                                sample_idx = (locals().get('test_start', 1) - 1 + idx + 1) if 'test_start' in locals() else idx + 1
                                misclass_data_test.append({
                                    'Sample Index': sample_idx,
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
                            # Use test_start if available (workspace data), else just sequential numbering
                            test_start_offset = locals().get('test_start', 1)
                            sample_names_dict_test = {i: str(test_start_offset + i) for i in range(len(X_test))}

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
                        # Handle both DataFrame and numpy array
                        if isinstance(X_test, pd.DataFrame):
                            feature_vals_test = X_test.iloc[selected_sample_idx_tab4].values
                        else:
                            feature_vals_test = X_test[selected_sample_idx_tab4]

                        feature_df_test = pd.DataFrame({
                            'Feature': x_columns,
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
                                # Calculate distance from selected test sample to each class centroid using specified metric
                                from scipy.spatial.distance import cdist
                                sample_test = X_test_scaled[selected_sample_idx_tab4].reshape(1, -1)
                                for cls in classes:
                                    cls_mask = y_test == cls
                                    X_cls_test = X_test_scaled[cls_mask]
                                    # Calculate distance to class centroid
                                    centroid = np.mean(X_cls_test, axis=0).reshape(1, -1)
                                    dist = cdist(sample_test, centroid, metric=trained['parameters']['metric'])[0, 0]
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

                            # Determine distance metric label
                            if trained['name'] in ['LDA', 'QDA']:
                                distance_label_test = 'Mahalanobis'
                            elif trained['name'] == 'SIMCA':
                                distance_label_test = 'F-statistic'
                            elif trained['name'] == 'UNEQ':
                                distance_label_test = 'T²-statistic'
                            elif trained['name'] == 'kNN':
                                distance_label_test = f"{trained['parameters']['metric'].capitalize()} to Centroid"
                            else:
                                distance_label_test = 'Distance'

                            # Calculate sample display index
                            sample_display_idx = locals().get('test_start', 1) + selected_sample_idx_tab4

                            fig_dist_sample_test.update_layout(
                                title=f"Test Sample {sample_display_idx} - Distance to Each Class",
                                xaxis_title="Class",
                                yaxis_title=f"Distance ({distance_label_test})",
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
