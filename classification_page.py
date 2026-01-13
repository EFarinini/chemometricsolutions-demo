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
        suggest_n_components_pca,

        # Classifiers
        fit_lda, predict_lda, cross_validate_lda, predict_lda_detailed,
        fit_qda, predict_qda, cross_validate_qda, predict_qda_detailed,
        fit_knn, predict_knn, cross_validate_knn, predict_knn_detailed,
        fit_simca, predict_simca, predict_simca_detailed, cross_validate_simca,
        fit_uneq, predict_uneq, predict_uneq_detailed, cross_validate_uneq,

        # PCA Preprocessing for classifiers
        fit_pca_preprocessor,
        project_onto_pca,
        fit_lda_with_pca, predict_lda_with_pca,
        fit_qda_with_pca, predict_qda_with_pca,
        fit_knn_with_pca, predict_knn_with_pca,

        # CV with PCA preprocessing
        cross_validate_lda_with_pca,
        cross_validate_qda_with_pca,
        cross_validate_knn_with_pca,

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
        plot_mahalanobis_distances,
        plot_classification_report,
        calculate_distance_matrix,
        plot_pca_variance_explained,
        plot_pca_scores_2d,

        # Constants
        AVAILABLE_DISTANCE_METRICS,
        get_available_classifiers,
        DEFAULT_N_COMPONENTS_PCA
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
    # Ensure tab1_data exists (for backward compatibility and error prevention)
    if 'tab1_data' not in st.session_state:
        st.session_state['tab1_data'] = {}

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

    # Initialize tab1_data from session state (accessible across all tabs)
    tab1_data = st.session_state.get('tab1_data', {})

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

        # ═══════════════════════════════════════════════════════════════════════
        # PRESERVE SAMPLE NAMES (DataFrame index)
        # ═══════════════════════════════════════════════════════════════════════

        # CRITICAL: Save sample names for later use in plots
        # X_full is already a DataFrame, so we can get the index directly
        st.session_state['sample_names'] = X_full.index.tolist()

        # Always save X_full and y_full regardless of split choice
        st.session_state['X_full'] = X_full  # Keep as DataFrame to preserve index
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
                    ✅ PRESERVES ORIGINAL INDICES/SAMPLE NAMES
                    """
                    from sklearn.model_selection import train_test_split

                    # ✅ FIX: Preserve original index when converting to DataFrame/Series
                    original_X_index = None
                    original_y_index = None

                    # Save original indices if they exist
                    if hasattr(X, 'index'):
                        original_X_index = X.index
                    if hasattr(y, 'index'):
                        original_y_index = y.index

                    # Convert to DataFrame/Series if needed, preserving indices
                    if not isinstance(X, pd.DataFrame):
                        X = pd.DataFrame(X, index=original_X_index)
                    if not isinstance(y, pd.Series):
                        # If y has an index, use it; otherwise align with X's index
                        if original_y_index is not None:
                            y = pd.Series(y, index=original_y_index)
                        elif original_X_index is not None:
                            y = pd.Series(y, index=original_X_index)
                        else:
                            y = pd.Series(y)

                    classes_list = np.unique(y)

                    X_train_list = []
                    X_test_list = []
                    y_train_list = []
                    y_test_list = []

                    # Debug: Show index info
                    with st.expander("🔍 Debug: Index Preservation Check", expanded=False):
                        st.write(f"**X DataFrame info:**")
                        st.write(f"- Type: {type(X)}")
                        st.write(f"- Index type: {type(X.index)}")
                        st.write(f"- Is RangeIndex: {isinstance(X.index, pd.RangeIndex)}")
                        st.write(f"- First 5 indices: {X.index[:5].tolist()}")
                        st.write(f"\n**y Series info:**")
                        st.write(f"- Type: {type(y)}")
                        st.write(f"- Index type: {type(y.index)}")
                        st.write(f"- First 5 indices: {y.index[:5].tolist()}")

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
                    # ✅ CRITICAL: Keep ignore_index=False to preserve sample names
                    X_train = pd.concat(X_train_list, ignore_index=False)
                    X_test = pd.concat(X_test_list, ignore_index=False)
                    y_train = pd.concat(y_train_list, ignore_index=False)
                    y_test = pd.concat(y_test_list, ignore_index=False)

                    # Shuffle (to mix classes)
                    if random_state is not None:
                        np.random.seed(random_state)

                    shuffle_idx_train = np.random.permutation(len(X_train))
                    shuffle_idx_test = np.random.permutation(len(X_test))

                    # ✅ CRITICAL: Don't reset indices, just shuffle
                    # Preserve the original sample names from X_full
                    X_train = X_train.iloc[shuffle_idx_train]
                    X_test = X_test.iloc[shuffle_idx_test]
                    y_train = y_train.iloc[shuffle_idx_train]
                    y_test = y_test.iloc[shuffle_idx_test]

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

                # ✅ FIX 1: Robustly extract sample names with multiple fallbacks
                test_sample_names = None

                # Try to get names from DataFrame index (preferred)
                if hasattr(X_test, 'index'):
                    try:
                        test_sample_names = X_test.index.tolist()
                        st.write(f"✓ Extracted {len(test_sample_names)} test sample names from X_test.index")
                    except Exception as e:
                        st.warning(f"Could not get index from X_test: {e}")

                # Fallback: if X_test is numpy array, try y_test
                if test_sample_names is None and hasattr(y_test, 'index'):
                    try:
                        test_sample_names = y_test.index.tolist()
                        st.write(f"✓ Extracted {len(test_sample_names)} test sample names from y_test.index (fallback)")
                    except Exception as e:
                        st.warning(f"Could not get index from y_test: {e}")

                st.session_state['test_sample_names'] = test_sample_names

                # Same for training set
                train_sample_names = None
                if hasattr(X_train, 'index'):
                    try:
                        train_sample_names = X_train.index.tolist()
                        st.write(f"✓ Extracted {len(train_sample_names)} train sample names from X_train.index")
                    except:
                        pass
                if train_sample_names is None and hasattr(y_train, 'index'):
                    try:
                        train_sample_names = y_train.index.tolist()
                        st.write(f"✓ Extracted {len(train_sample_names)} train sample names from y_train.index (fallback)")
                    except:
                        pass
                st.session_state['train_sample_names'] = train_sample_names

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

        # kNN metric selection (needed for preprocessing info)
        metric = None
        if selected_classifier == 'kNN':
            with col2:
                metric = st.selectbox(
                    "Distance Metric:",
                    AVAILABLE_DISTANCE_METRICS,
                    key="tab1_knn_metric"
                )

            # Save kNN parameters to session state
            st.session_state.k_value = k_value
            st.session_state.metric = metric

        # ═══════════════════════════════════════════════════════════════════════
        # PREPROCESSING REQUIREMENTS BY METHOD
        # ═══════════════════════════════════════════════════════════════════════

        st.divider()

        # Display method-specific preprocessing requirements
        preprocessing_info = {
            'LDA': {
                'scaling': 'OPTIONAL',
                'centering': 'Optional',
                'why': 'Uses Mahalanobis distance (scale-invariant). Preprocessing optional but may help numerical stability',
                'recommendation': 'none',
                'icon': 'ℹ️',
                'color': 'info'
            },
            'QDA': {
                'scaling': 'OPTIONAL',
                'centering': 'Optional',
                'why': 'Uses Mahalanobis distance (scale-invariant). Scaling recommended for numerical stability with many parameters',
                'recommendation': 'autoscale',
                'icon': '⚠️',
                'color': 'warning'
            },
            'kNN': {
                'scaling': 'CRITICAL',
                'centering': 'Optional',
                'why': 'Euclidean/Manhattan distances are scale-sensitive; large-variance features dominate (Mahalanobis is scale-invariant)',
                'recommendation': 'autoscale',
                'icon': '🚨',
                'color': 'error'
            },
            'SIMCA': {
                'scaling': 'CRITICAL',
                'centering': 'Internal (NIPALS)',
                'why': 'PCA is NOT scale-invariant; high-variance features dominate PC directions without scaling',
                'recommendation': 'autoscale',
                'icon': '🚨',
                'color': 'error'
            },
            'UNEQ': {
                'scaling': 'OPTIONAL',
                'centering': 'Optional',
                'why': 'Uses Mahalanobis distance (scale-invariant). Scaling recommended for numerical stability like QDA',
                'recommendation': 'autoscale',
                'icon': '⚠️',
                'color': 'warning'
            }
        }

        info = preprocessing_info[selected_classifier]

        # Display requirements in an expander
        with st.expander(f"{info['icon']} {selected_classifier} Preprocessing Requirements", expanded=True):
            col_req1, col_req2, col_req3 = st.columns([1, 1, 2])

            with col_req1:
                st.metric("Scaling", info['scaling'])

            with col_req2:
                st.metric("Centering", info['centering'])

            with col_req3:
                st.info(f"**Why?** {info['why']}")

            # Show current preprocessing status
            if scaling_method == 'autoscale':
                st.success(f"""
                ✅ **Current Preprocessing:** Autoscale (Center + Scale)
                - Formula: `(X - mean) / std`
                - All features contribute equally to {selected_classifier}
                - ✓ {'Optimal' if info['scaling'] == 'CRITICAL' else 'Recommended'} for {selected_classifier}
                """)
            elif scaling_method == 'center':
                if info['scaling'] == 'CRITICAL':
                    st.error(f"""
                    ❌ **Current Preprocessing:** Center only
                    - ⚠️ {selected_classifier} requires SCALING for optimal performance
                    - Change to 'Autoscale' in Step 1 settings
                    """)
                elif info['scaling'] == 'OPTIONAL':
                    st.info(f"""
                    ℹ️ **Current Preprocessing:** Center only
                    - {selected_classifier} is scale-invariant (Mahalanobis distance)
                    - Autoscale may improve numerical stability
                    """)
                else:
                    st.warning(f"""
                    ⚠️ **Current Preprocessing:** Center only
                    - Scaling recommended for {selected_classifier}
                    - Consider changing to 'Autoscale' in Step 1
                    """)
            elif scaling_method == 'none':
                if info['scaling'] == 'CRITICAL':
                    st.error(f"""
                    ❌ **No Preprocessing Applied**
                    - {selected_classifier} requires preprocessing!
                    - Change to 'Autoscale' in Step 1 settings
                    """)
                elif info['scaling'] == 'OPTIONAL':
                    st.info(f"""
                    ℹ️ **No Preprocessing Applied**
                    - {selected_classifier} is scale-invariant (works without scaling)
                    - Autoscale recommended for numerical stability
                    """)
                else:
                    st.warning(f"""
                    ⚠️ **No Preprocessing Applied**
                    - Preprocessing recommended for {selected_classifier}
                    - Consider using 'Autoscale' in Step 1
                    """)

            # Method-specific notes
            if selected_classifier == 'kNN' and metric:
                if metric == 'mahalanobis':
                    st.caption("📌 **Note:** Mahalanobis distance is scale-invariant (uses covariance). Autoscale recommended for numerical stability.")
                else:
                    st.caption(f"📌 **Note:** {metric.capitalize()} distance is scale-sensitive. Autoscale is CRITICAL.")
            elif selected_classifier == 'SIMCA':
                st.caption("📌 **Note:** SIMCA uses PCA (NOT scale-invariant). High-variance features dominate PCs without scaling. Autoscale is CRITICAL.")
            elif selected_classifier == 'LDA':
                st.caption("📌 **Note:** LDA uses Mahalanobis distance (scale-invariant). Scaling optional but may help numerical stability.")
            elif selected_classifier in ['QDA', 'UNEQ']:
                st.caption("📌 **Note:** Uses Mahalanobis distance (scale-invariant). Scaling recommended for numerical stability with many parameters.")

            # Quick reference table
            st.markdown("---")
            st.markdown("**📋 Quick Reference: Preprocessing Impact**")

            impact_table = {
                'LDA': {'Without Scaling': 'ℹ️ Mathematically equivalent (scale-invariant)', 'With Autoscale': '✅ Better numerical stability'},
                'QDA': {'Without Scaling': '⚠️ May have numerical issues', 'With Autoscale': '✅ More stable covariance estimation'},
                'kNN': {'Without Scaling': '🚨 Distances biased by scale (Euclidean/Manhattan)', 'With Autoscale': '✅ True nearest neighbors'},
                'SIMCA': {'Without Scaling': '🚨 PCA dominated by high-variance features', 'With Autoscale': '✅ Balanced PC components'},
                'UNEQ': {'Without Scaling': '⚠️ May have covariance issues', 'With Autoscale': '✅ More stable distance calculation'}
            }

            current_impact = impact_table[selected_classifier]
            col_impact1, col_impact2 = st.columns(2)
            with col_impact1:
                msg = current_impact['Without Scaling']
                if msg.startswith('🚨'):
                    st.error(msg)
                elif msg.startswith('⚠️'):
                    st.warning(msg)
                else:
                    st.info(msg)
            with col_impact2:
                st.success(current_impact['With Autoscale'])

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
                # Get number of features for component limits
                n_features_available = X_train_scaled.shape[1] if hasattr(X_train_scaled, 'shape') else len(X_train_scaled[0])
                n_samples_available = X_train_scaled.shape[0] if hasattr(X_train_scaled, 'shape') else len(X_train_scaled)
                max_pca_components = min(n_features_available - 1, n_samples_available - 1, 15)

                # Get recommended number of components
                try:
                    pca_suggestion = suggest_n_components_pca(
                        X_train_scaled,
                        cumsum_threshold=0.95,
                        max_components=max_pca_components
                    )
                    recommended_n = pca_suggestion['recommended_n_components']
                    recommended_variance = pca_suggestion['variance_explained']
                except Exception:
                    recommended_n = min(5, max_pca_components)
                    recommended_variance = None

                col_pca1, col_pca2 = st.columns([2, 1])

                with col_pca1:
                    n_components_pca = st.slider(
                        "PCA Components",
                        min_value=1,
                        max_value=max(max_pca_components, 2),
                        value=min(recommended_n, max_pca_components),
                        key="tab1_n_pca_components",
                        help=f"Recommended: {recommended_n} components for 95% variance"
                    )

                with col_pca2:
                    if recommended_variance is not None:
                        st.metric("Recommended", f"{recommended_n} PCs", f"{recommended_variance*100:.1f}% var")
                    else:
                        st.metric("Selected", f"{n_components_pca} PCs")

                # Store in session state
                st.session_state.use_pca_preprocessing = True
                st.session_state.n_components_pca = n_components_pca
            else:
                st.session_state.use_pca_preprocessing = False
                st.session_state.n_components_pca = None
                n_components_pca = None
        else:
            # SIMCA and UNEQ have PCA built-in, no need for preprocessing
            use_pca_preprocessing = False
            st.session_state.use_pca_preprocessing = False
            st.session_state.n_components_pca = None
            n_components_pca = None
            if selected_classifier in ['SIMCA', 'UNEQ']:
                st.caption("ℹ️ PCA preprocessing not needed - this classifier includes PCA internally")

        # ═══════════════════════════════════════════════════════════════════════════
        # UNIFIED CROSS-VALIDATION SECTION (All Classifiers)
        # ═══════════════════════════════════════════════════════════════════════════
        """
        CROSS-VALIDATION EXECUTION
        ══════════════════════════════════════════════════════════════════

        CV can be executed in TWO modes:

        1️⃣  CV-ONLY MODE (Recommended for model evaluation):
            - Executes k-fold cross-validation on selected data
            - Saves results in st.session_state['cv_results']
            - Does NOT train a final model (trained_model remains None)
            - Tab 2 displays CV results immediately
            - Use this for: model selection, hyperparameter tuning

        2️⃣  TRAIN + CV MODE (For production deployment):
            - Trains final model on full training set (Step 4)
            - Then runs CV for evaluation (Step 5)
            - Saves both trained_model AND cv_results
            - Tab 3/4 use trained_model for predictions

        DEFAULT: CV-ONLY mode is active (Step 4 training is skipped)
        """

        st.divider()
        st.markdown("### 🔄 Cross-Validation Evaluation")

        # Show PCA info if enabled
        if use_pca_preprocessing and selected_classifier in ['LDA', 'QDA', 'kNN']:
            st.info(
                "**PCA Preprocessing Enabled:**\n\n"
                "For each fold:\n"
                "1. Fit PCA **ONLY** on training data\n"
                "2. Project training and evaluation data onto PCA space\n"
                "3. Train classifier and predict\n\n"
                "This prevents data leakage and ensures proper validation."
            )
        else:
            st.info(
                "**Stratified K-Fold CV:**\n"
                "- Each fold maintains class proportions\n"
                "- Every class is represented in each fold\n"
                "- Provides robust performance estimates"
            )

        # CV parameters
        col_cv1, col_cv2 = st.columns([2, 1])
        with col_cv1:
            n_folds_cv = st.number_input(
                "Number of Folds",
                min_value=2,
                max_value=10,
                value=5,
                key="cv_n_folds_unified"
            )
        with col_cv2:
            random_seed_cv = st.number_input(
                "Random Seed",
                min_value=0,
                max_value=9999,
                value=42,
                key="cv_random_seed_unified"
            )

        # Unified CV button
        if st.button(
            "🔄 Run Cross-Validation",
            type="primary",
            use_container_width=True,
            key="run_cv_unified_btn"
        ):
            import time
            cv_start = time.time()

            spinner_text = f"Running {n_folds_cv}-fold cross-validation"
            if use_pca_preprocessing and selected_classifier in ['LDA', 'QDA', 'kNN']:
                spinner_text += f" with PCA ({n_components_pca} components)"
            spinner_text += "..."

            with st.spinner(spinner_text):
                try:
                    y_values = y_for_training.values if hasattr(y_for_training, 'values') else y_for_training

                    # Branch based on PCA preprocessing
                    if use_pca_preprocessing and selected_classifier in ['LDA', 'QDA', 'kNN']:
                        # PCA-based CV
                        if selected_classifier == 'LDA':
                            cv_results_raw = cross_validate_lda_with_pca(
                                X=X_train_scaled,
                                y=y_values,
                                n_components_pca=n_components_pca,
                                n_folds=n_folds_cv,
                                scaling_method=scaling_method,
                                random_state=random_seed_cv
                            )
                        elif selected_classifier == 'QDA':
                            cv_results_raw = cross_validate_qda_with_pca(
                                X=X_train_scaled,
                                y=y_values,
                                n_components_pca=n_components_pca,
                                n_folds=n_folds_cv,
                                scaling_method=scaling_method,
                                random_state=random_seed_cv
                            )
                        elif selected_classifier == 'kNN':
                            cv_results_raw = cross_validate_knn_with_pca(
                                X=X_train_scaled,
                                y=y_values,
                                n_components_pca=n_components_pca,
                                k_values=[k_value],
                                metric=metric,
                                n_folds=n_folds_cv,
                                scaling_method=scaling_method,
                                random_state=random_seed_cv
                            )

                        # Standardize format for PCA results
                        standardized_results = {
                            'cv_predictions': cv_results_raw['predictions'],
                            'y_true': cv_results_raw['y_true'],
                            'metrics': cv_results_raw['metrics'],
                            'cv_details': cv_results_raw.get('cv_details', []),
                            'n_components_pca': n_components_pca,
                            'use_pca_preprocessing': True,
                            'misclassified_indices': cv_results_raw.get('misclassified_indices', []),
                            'mahalanobis_distances': cv_results_raw.get('mahalanobis_distances', {}),
                            # ✅ ADD PCA PREPROCESSOR IF AVAILABLE
                            'pca_preprocessor': cv_results_raw.get('pca_preprocessor'),
                            'pca_loadings': cv_results_raw.get('pca_loadings'),
                            'pca_variance_explained': cv_results_raw.get('pca_variance_explained'),
                        }
                        cv_method = 'with_pca'

                    else:
                        # Standard CV (no PCA)
                        n_pcs = st.session_state.get('n_pcs', 3)
                        confidence_level = st.session_state.get('confidence_level', 0.95)

                        if selected_classifier == 'LDA':
                            cv_results = cross_validate_lda(
                                X_train_scaled, y_values,
                                n_folds=n_folds_cv,
                                random_state=random_seed_cv
                            )
                        elif selected_classifier == 'QDA':
                            cv_results = cross_validate_qda(
                                X_train_scaled, y_values,
                                n_folds=n_folds_cv,
                                random_state=random_seed_cv
                            )
                        elif selected_classifier == 'kNN':
                            cv_results = cross_validate_classifier(
                                X_train_scaled, y_values,
                                classifier_type='knn',
                                n_folds=n_folds_cv,
                                classifier_params={
                                    'k': k_value,
                                    'metric': metric,
                                    'use_pca': False
                                },
                                random_state=random_seed_cv
                            )
                        elif selected_classifier == 'SIMCA':
                            cv_results = cross_validate_classifier(
                                X_train_scaled, y_values,
                                classifier_type='simca',
                                n_folds=n_folds_cv,
                                classifier_params={
                                    'n_components': n_pcs,
                                    'confidence_level': confidence_level
                                },
                                random_state=random_seed_cv
                            )
                        elif selected_classifier == 'UNEQ':
                            cv_results = cross_validate_classifier(
                                X_train_scaled, y_values,
                                classifier_type='uneq',
                                n_folds=n_folds_cv,
                                classifier_params={
                                    'n_components': n_pcs,
                                    'confidence_level': confidence_level,
                                    'use_pca': False
                                },
                                random_state=random_seed_cv
                            )

                        # Standardize format for standard results
                        if selected_classifier in ['LDA', 'QDA']:
                            # Direct format from cross_validate_lda/qda
                            standardized_results = {
                                'cv_predictions': cv_results.get('predictions', []),
                                'y_true': y_values,
                                'metrics': cv_results.get('metrics', {}),
                                'cv_details': cv_results.get('cv_details', []),
                                'use_pca_preprocessing': False,
                                'misclassified_indices': cv_results.get('misclassified_indices', []),
                                'mahalanobis_distances': cv_results.get('mahalanobis_distances', {})
                            }
                        else:
                            # Format from cross_validate_classifier
                            standardized_results = {
                                'cv_predictions': cv_results.get('cv_predictions', []),
                                'y_true': y_values,
                                'metrics': {
                                    'accuracy': cv_results.get('cv_accuracy', 0),
                                    'average_sensitivity': np.mean(list(cv_results.get('cv_sensitivity_per_class', {}).values())) if cv_results.get('cv_sensitivity_per_class') else 0,
                                    'average_specificity': np.mean(list(cv_results.get('cv_specificity_per_class', {}).values())) if cv_results.get('cv_specificity_per_class') else 0,
                                    'sensitivity_per_class': cv_results.get('cv_sensitivity_per_class', {}),
                                    'specificity_per_class': cv_results.get('cv_specificity_per_class', {}),
                                    'precision_per_class': {},
                                    'f1_per_class': cv_results.get('cv_f1_per_class', {}),
                                    'confusion_matrix': cv_results.get('cv_confusion_matrix', np.array([])),
                                    'classes': classes
                                },
                                'cv_details': [
                                    {
                                        'fold': fold_res['fold'],
                                        'accuracy': fold_res['accuracy'],
                                        'n_train': 'N/A',
                                        'n_eval': 'N/A'
                                    }
                                    for fold_res in cv_results.get('fold_results', [])
                                ],
                                'use_pca_preprocessing': False,
                                'misclassified_indices': np.where(y_values != cv_results.get('cv_predictions', []))[0].tolist() if len(cv_results.get('cv_predictions', [])) > 0 else []
                            }
                        cv_method = 'standard'

                    # ═══════════════════════════════════════════════════════════════════════
                    # ✅ ADD kNN-SPECIFIC DATA TO standardized_results BEFORE SAVING
                    # ═══════════════════════════════════════════════════════════════════════
                    if selected_classifier == 'kNN':
                        # Get kNN parameters from session state with fallback
                        k_val = st.session_state.get('k_value', k_value if 'k_value' in locals() else 3)
                        met = st.session_state.get('metric', metric if 'metric' in locals() else 'euclidean')

                        # Add kNN-specific data to standardized_results
                        standardized_results['X_train'] = X_train_scaled
                        standardized_results['y_train'] = y_values
                        standardized_results['k_value'] = k_val
                        standardized_results['metric'] = met

                    cv_time = time.time() - cv_start

                    # ✅ SAVE CV RESULTS COMPLETELY
                    st.session_state['cv_results'] = standardized_results
                    st.session_state['cv_results_time'] = cv_time
                    st.session_state['cv_time'] = cv_time
                    st.session_state['cv_method'] = cv_method
                    st.session_state['cv_n_folds'] = n_folds_cv
                    st.session_state['cv_classifier'] = selected_classifier

                    # ✅ ALSO SAVE FOR EXPORT (Training data)
                    st.session_state['X_train_scaled_cv'] = X_train_scaled
                    st.session_state['y_train_cv'] = y_values
                    st.session_state['X_test_scaled_cv'] = X_test_scaled
                    st.session_state['y_test_cv'] = y_test_values if 'y_test_values' in locals() else None

                    # ✅ Save PCA if used
                    if 'pca_preprocessor' in locals() and pca_preprocessor is not None:
                        st.session_state['pca_preprocessor'] = pca_preprocessor
                        st.session_state['use_pca_cv'] = True
                        # ✅ SAVE PCA DATA FROM CV RESULTS
                        st.session_state['pca_loadings'] = standardized_results.get('pca_loadings')
                        st.session_state['pca_variance_explained'] = standardized_results.get('pca_variance_explained')
                        st.session_state['n_components_pca'] = standardized_results.get('n_components_pca')
                    else:
                        st.session_state['use_pca_cv'] = False

                    # ✅ VERIFICATION
                    with st.expander("✅ CV Storage Verification", expanded=False):
                        st.success("✅ Cross-validation results stored!")
                        st.write(f"**Classifier:** {selected_classifier}")
                        st.write(f"**CV Method:** {cv_method}")
                        st.write(f"**CV Results Keys:** {list(standardized_results.keys())[:10]}...")
                        if selected_classifier == 'kNN':
                            st.write(f"**kNN k_value in results:** {'k_value' in standardized_results}")
                            st.write(f"**kNN metric in results:** {'metric' in standardized_results}")
                            st.write(f"**X_train in results:** {'X_train' in standardized_results}")

                    # ═══════════════════════════════════════════════════════════════════════
                    # TRAIN FINAL MODEL ON FULL TRAINING DATA (for Tab 2 detailed analysis)
                    # ═══════════════════════════════════════════════════════════════════════

                    try:
                        with st.spinner("Training final model on full training data..."):
                            # ✅ DEBUG: Print training parameters
                            print("\n" + "=" * 80)
                            print("🔧 MODEL TRAINING STARTED")
                            print(f"Classifier: {selected_classifier}")
                            print(f"Use PCA: {use_pca_preprocessing}")
                            print(f"N Components: {n_components_pca if use_pca_preprocessing else 'N/A'}")
                            print(f"X_train_scaled shape: {X_train_scaled.shape}")
                            print(f"y_values shape: {y_values.shape}")
                            print(f"Classes: {classes}")
                            print("=" * 80 + "\n")

                            if use_pca_preprocessing and selected_classifier in ['LDA', 'QDA', 'kNN']:
                                # Train with PCA preprocessing
                                print(f"📊 Training {selected_classifier} with PCA ({n_components_pca} components)...")

                                if selected_classifier == 'LDA':
                                    print(f"🎯 Training LDA with PCA...")
                                    final_model = fit_lda_with_pca(
                                        X_train_scaled, y_values,
                                        n_components_pca=n_components_pca
                                    )
                                    # Extract PCA info from combined model
                                    pca_info = final_model['pca_model']
                                    X_train_pca = final_model['pca_model']['scores']

                                elif selected_classifier == 'QDA':
                                    print(f"🎯 Training QDA with PCA...")
                                    final_model = fit_qda_with_pca(
                                        X_train_scaled, y_values,
                                        n_components_pca=n_components_pca
                                    )
                                    # Extract PCA info from combined model
                                    pca_info = final_model['pca_model']
                                    X_train_pca = final_model['pca_model']['scores']

                                elif selected_classifier == 'kNN':
                                    print(f"🎯 Training kNN with PCA...")
                                    final_model = fit_knn_with_pca(
                                        X_train_scaled, y_values,
                                        n_components_pca=n_components_pca,
                                        metric=metric
                                    )
                                    # Extract PCA info from combined model
                                    pca_info = final_model['pca_model']
                                    X_train_pca = final_model['pca_model']['scores']

                                print(f"✅ Model trained successfully!")
                                print(f"✅ PCA shape: {X_train_pca.shape}")
                            else:
                                # ✅ INITIALIZE pca_info FOR NON-PCA CASE
                                pca_info = None
                                X_train_pca = None

                                # Standard model training (no PCA)
                                if selected_classifier == 'LDA':
                                    final_model = fit_lda(X_train_scaled, y_values)
                                elif selected_classifier == 'QDA':
                                    final_model = fit_qda(X_train_scaled, y_values)
                                elif selected_classifier == 'kNN':
                                    final_model = fit_knn(
                                        X_train_scaled, y_values,
                                        k=k_value,
                                        metric=metric
                                    )
                                elif selected_classifier == 'SIMCA':
                                    n_pcs = st.session_state.get('n_pcs', 3)
                                    confidence_level = st.session_state.get('confidence_level', 0.95)
                                    final_model = fit_simca(
                                        X_train_scaled, y_values,
                                        n_components=n_pcs,
                                        confidence_level=confidence_level
                                    )
                                elif selected_classifier == 'UNEQ':
                                    n_pcs = st.session_state.get('n_pcs', 3)
                                    confidence_level = st.session_state.get('confidence_level', 0.95)
                                    final_model = fit_uneq(
                                        X_train_scaled, y_values,
                                        n_components=n_pcs,
                                        confidence_level=confidence_level
                                    )

                        # Save trained model to session state
                        # Get kNN parameters from session state as fallback
                        if selected_classifier == 'kNN':
                            k_val = st.session_state.get('k_value', k_value if 'k_value' in locals() else 3)
                            met = st.session_state.get('metric', metric if 'metric' in locals() else 'euclidean')
                        else:
                            k_val = None
                            met = None

                        st.session_state.trained_model = {
                            'model': final_model,
                            'name': selected_classifier,
                            'n_features': X_train_scaled.shape[1],
                            'classes': classes.tolist() if hasattr(classes, 'tolist') else list(classes),
                            'scaling_method': scaling_method,
                            'X_train': X_train_scaled,  # ← For all classifiers
                            'y_train': y_values,         # ← For all classifiers
                            'parameters': {
                                'k': k_val,
                                'metric': met,
                                'n_components': n_components_pca if use_pca_preprocessing else None,
                                'use_pca': use_pca_preprocessing
                            }
                        }

                        # ✅ STEP 3: SAVE PCA PREPROCESSOR IF USED
                        if use_pca_preprocessing and selected_classifier in ['LDA', 'QDA', 'kNN']:
                            st.session_state.trained_model['pca_preprocessor'] = pca_info
                            st.session_state.trained_model['X_train_pca'] = X_train_pca  # Transformed data
                            st.session_state.trained_model['model_type'] = final_model.get('model_type', f'{selected_classifier.lower()}_with_pca')
                            # Also save the inner classifier model for direct access
                            if selected_classifier == 'LDA':
                                st.session_state.trained_model['classifier_model'] = final_model['lda_model']
                            elif selected_classifier == 'QDA':
                                st.session_state.trained_model['classifier_model'] = final_model['qda_model']
                            elif selected_classifier == 'kNN':
                                st.session_state.trained_model['classifier_model'] = final_model['knn_model']
                        else:
                            st.session_state.trained_model['pca_preprocessor'] = None
                            st.session_state.trained_model['model_type'] = 'standard'
                            st.session_state.trained_model['classifier_model'] = final_model

                        # ✅ ALSO SAVE TO cv_results FOR TAB 2 ACCESS
                        standardized_results['trained_model'] = st.session_state.trained_model
                        standardized_results['final_model'] = final_model

                        # For kNN, also save X_train and y_train for distance analysis
                        if selected_classifier == 'kNN':
                            standardized_results['X_train'] = X_train_scaled
                            standardized_results['y_train'] = y_values
                            standardized_results['k_value'] = k_val
                            standardized_results['metric'] = met

                        # ✅ FORCE REPLICATION: Ensure trained_model is accessible
                        st.session_state['cv_results'] = standardized_results

                        # ✅ Verify model was saved
                        if st.session_state.trained_model is None:
                            st.error("❌ CRITICAL: Model not saved to session state!")
                            st.stop()

                        # ✅ MODEL SAVED - Silent success
                        st.success(f"✅ Model ready for analysis in Tab 2")

                    except Exception as e:
                        st.error(f"🔴 CRITICAL ERROR: Could not train final model!")
                        st.error(f"**Error Type:** {type(e).__name__}")
                        st.error(f"**Error Message:** {str(e)}")

                        # Show full traceback for debugging
                        import traceback
                        error_msg = traceback.format_exc()
                        with st.expander("📋 Full Error Traceback", expanded=True):
                            st.code(error_msg, language='python')

                        # ✅ ALSO PRINT TO CONSOLE FOR VISIBILITY
                        print("=" * 80)
                        print("🔴 MODEL TRAINING ERROR DETAILS:")
                        print(error_msg)
                        print("=" * 80)
                        print(f"Selected Classifier: {selected_classifier}")
                        print(f"Use PCA: {use_pca_preprocessing}")
                        print(f"N Components: {n_components_pca if use_pca_preprocessing else 'N/A'}")
                        print(f"X_train shape: {X_train_scaled.shape if 'X_train_scaled' in locals() else 'N/A'}")
                        print(f"y_values shape: {y_values.shape if 'y_values' in locals() else 'N/A'}")
                        print("=" * 80)

                        st.info("⚠️ Tab 2 analysis will use CV results only")
                        # Clear trained_model if training failed
                        st.session_state.trained_model = None

                        # ✅ DON'T STOP - Let user see the error and continue
                        st.warning("⚠️ Continuing without trained model. CV results are still available in Tab 2.")

                    st.success(f"✅ Cross-validation completed in {cv_time:.2f}s!")

                    # Model saved - proceed to Tab 2 for analysis
                    st.rerun()

                except Exception as e:
                    st.error(f"❌ Cross-validation failed: {str(e)}")
                    import traceback
                    st.error(traceback.format_exc())

        # Display CV results if available
        if 'cv_results' in st.session_state:
            cv_res = st.session_state['cv_results']
            cv_time = st.session_state.get('cv_results_time', 0)
            cv_method = st.session_state.get('cv_method', 'standard')

            st.divider()
            st.success(f"✅ **CV Results Available** ({st.session_state.get('cv_classifier', selected_classifier)})")

            # Metrics display
            if cv_method == 'with_pca':
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Accuracy", f"{cv_res['metrics'].get('accuracy', 0):.1f}%")
                with col2:
                    st.metric("PCA Components", cv_res.get('n_components_pca', 'N/A'))
                with col3:
                    st.metric("Folds", st.session_state.get('cv_n_folds', n_folds_cv))
                with col4:
                    st.metric("Time", f"{cv_time:.2f}s")
            else:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Accuracy", f"{cv_res.get('metrics', {}).get('accuracy', 0):.1f}%")
                with col2:
                    st.metric("Folds", st.session_state.get('cv_n_folds', n_folds_cv))
                with col3:
                    st.metric("Time", f"{cv_time:.2f}s")
                with col4:
                    st.metric("Samples", len(cv_res.get('cv_predictions', [])))

            st.info("ℹ️ **Next Step:** Go to **Tab 2 (Classification Analysis)** to see detailed results, confusion matrix, and diagnostics.")


    # ========== TAB 2: CLASSIFICATION ANALYSIS ==========
    with tab2:
        st.markdown("## 🎲 Classification Analysis - Cross-Validation")

        # Initialize tab1_data from session state with validation
        tab1_data = st.session_state.get('tab1_data', {})
        if not tab1_data:
            st.warning("⚠️ Tab 1 data not initialized properly")
            st.info("💡 Please go to Tab 1 and complete the setup first (X, Y, Split)")
            st.stop()

        # ═══════════════════════════════════════════════════════════════════════
        # LOAD TRAINED MODEL AND CROSS-VALIDATION RESULTS
        # ═══════════════════════════════════════════════════════════════════════

        trained_model_info = st.session_state.get('trained_model')
        cv_results_info = st.session_state.get('cv_results')
        cv_method = st.session_state.get('cv_method', None)

        # ✅ VALIDATION: Accept both standard model AND PCA-CV results
        if trained_model_info is None and cv_results_info is None:
            st.error("❌ **No trained model available**")
            st.info("💡 **Please complete Tab 1 first:**")
            st.info("   1. Select your dataset")
            st.info("   2. Choose X (features) and Y (target)")
            st.info("   3. Configure Train/Test split")
            st.info("   4. Run Cross-Validation")
            st.stop()

        # ✅ If we have CV results but no trained_model (PCA case), that's OK
        if cv_results_info is None:
            st.error("❌ **Cross-validation results not found**")
            st.stop()

        # ✅ LOAD TRAINED MODEL (final model from Tab 1)
        has_trained_model = trained_model_info is not None and trained_model_info.get('model') is not None

        if has_trained_model:
            st.success(f"✅ Final Model Loaded: {trained_model_info['name']}")
            final_model = trained_model_info['model']
            trained = trained_model_info
            # Check if PCA was used in training
            model_params = trained_model_info.get('parameters', {})
            use_pca_in_model = model_params.get('use_pca', False)
            n_components_pca_model = model_params.get('n_components')
            pca_preprocessor_model = trained_model_info.get('pca_preprocessor')
        else:
            st.warning("⚠️ No trained model available - using CV results only for metrics")
            final_model = None
            trained = None
            use_pca_in_model = cv_results_info.get('use_pca_preprocessing', False)
            n_components_pca_model = cv_results_info.get('n_components_pca')
            # ✅ LOAD PCA FROM CV_RESULTS, NOT FROM SESSION_STATE!
            pca_preprocessor_model = cv_results_info.get('pca_preprocessor')

        # Check if data is available
        X_full = st.session_state.get('X_full')
        y_full = st.session_state.get('y_full')
        classes = st.session_state.get('classes')

        if X_full is None or y_full is None:
            st.warning("⚠️ No data available")
            st.info("💡 Go to Tab 1 and select X (features) and Y (target)")
            return

        # ═══════════════════════════════════════════════════════════════════════════
        # CHECK IF CV WAS ALREADY RUN IN TAB 1 WITH PCA PREPROCESSING
        # ═══════════════════════════════════════════════════════════════════════════
        cv_method = st.session_state.get('cv_method', None)
        cv_results_from_tab1 = st.session_state.get('cv_results', None)
        use_pca_preprocessing = st.session_state.get('use_pca_preprocessing', False)

        # ✅ FALLBACK: Load PCA data from cv_results if not in session_state
        if cv_method == 'with_pca' and cv_results_from_tab1 is not None:
            if 'pca_preprocessor' not in st.session_state and cv_results_from_tab1.get('pca_preprocessor'):
                st.session_state['pca_preprocessor'] = cv_results_from_tab1.get('pca_preprocessor')
                st.session_state['pca_loadings'] = cv_results_from_tab1.get('pca_loadings')
                st.session_state['pca_variance_explained'] = cv_results_from_tab1.get('pca_variance_explained')
                st.session_state['n_components_pca'] = cv_results_from_tab1.get('n_components_pca')
                st.session_state['use_pca_cv'] = True

        if cv_method == 'with_pca' and cv_results_from_tab1 is not None:
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # PCA PREPROCESSING WAS USED: CV was already run in Tab 1
            # Tab 2 is now DISPLAY-ONLY mode
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

            st.info(
                "**PCA Preprocessing Mode Active**\n\n"
                "Cross-validation was executed in Tab 1 with PCA preprocessing. "
                "This tab displays the results - no additional CV can be run.\n\n"
                "*To run CV with different parameters, go back to Tab 1.*"
            )

            cv_classifier = st.session_state.get('cv_classifier', 'Unknown')
            cv_n_folds = st.session_state.get('cv_n_folds', 5)
            cv_time = st.session_state.get('cv_results_time', 0)
            n_components_pca = cv_results_from_tab1.get('n_components_pca', 'N/A')

            st.success(
                f"🎯 **Classifier:** {cv_classifier} | "
                f"**PCA Components:** {n_components_pca} | "
                f"**CV Folds:** {cv_n_folds} | "
                f"**Time:** {cv_time:.2f}s"
            )

            st.divider()

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # DISPLAY CV RESULTS FROM TAB 1
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            st.markdown("### 📊 Cross-Validation Results (from Tab 1)")

            cv_res = cv_results_from_tab1
            cv_metrics = cv_res.get('metrics', {})

            # Get predictions and true labels
            y_pred_cv = cv_res.get('cv_predictions', cv_res.get('predictions', []))
            y_true_cv = cv_res.get('y_true', [])

            # Display main metrics
            col_acc, col_sens, col_spec = st.columns(3)
            with col_acc:
                st.metric(
                    "Overall Accuracy",
                    f"{cv_metrics.get('accuracy', 0):.1f}%",
                    help="Percentage of correctly classified samples"
                )
            with col_sens:
                avg_sens = np.mean(list(cv_metrics.get('sensitivity_per_class', {}).values())) if cv_metrics.get('sensitivity_per_class') else 0
                st.metric(
                    "Mean Sensitivity",
                    f"{avg_sens:.1f}%",
                    help="Average true positive rate across classes"
                )
            with col_spec:
                avg_spec = np.mean(list(cv_metrics.get('specificity_per_class', {}).values())) if cv_metrics.get('specificity_per_class') else 0
                st.metric(
                    "Mean Specificity",
                    f"{avg_spec:.1f}%",
                    help="Average true negative rate across classes"
                )

            # Confusion Matrix
            st.markdown("#### Confusion Matrix")
            if 'confusion_matrix' in cv_metrics:
                classes_list = list(cv_metrics.get('sensitivity_per_class', {}).keys()) if cv_metrics.get('sensitivity_per_class') else []
                if classes_list:
                    fig_cm = plot_confusion_matrix(
                        cv_metrics['confusion_matrix'],
                        classes=classes_list,
                        title=f"{cv_classifier} with PCA - CV Confusion Matrix"
                    )
                    st.plotly_chart(fig_cm, use_container_width=True)

            # Per-class metrics
            st.markdown("#### Per-Class Metrics")
            if cv_metrics.get('sensitivity_per_class') and cv_metrics.get('specificity_per_class'):
                metrics_df = pd.DataFrame({
                    'Class': list(cv_metrics['sensitivity_per_class'].keys()),
                    'Sensitivity (%)': [f"{v:.1f}" for v in cv_metrics['sensitivity_per_class'].values()],
                    'Specificity (%)': [f"{v:.1f}" for v in cv_metrics['specificity_per_class'].values()],
                    'F1 Score (%)': [f"{v:.1f}" for v in cv_metrics.get('f1_per_class', {}).values()] if cv_metrics.get('f1_per_class') else ['N/A'] * len(cv_metrics['sensitivity_per_class'])
                })
                st.dataframe(metrics_df, use_container_width=True, hide_index=True)

            # ✅ LOAD ALL PCA DATA FROM CV_RESULTS, NOT FROM SESSION_STATE!
            pca_preprocessor = cv_results_from_tab1.get('pca_preprocessor')
            pca_loadings = cv_results_from_tab1.get('pca_loadings')
            pca_variance_explained = cv_results_from_tab1.get('pca_variance_explained')
            X_train_scaled_cv = st.session_state.get('X_train_scaled_cv')
            y_train_cv = st.session_state.get('y_train_cv')
            n_components_pca = cv_results_from_tab1.get('n_components_pca', 'N/A')

            # PCA-specific information
            st.markdown("#### PCA Preprocessing Summary")

            # ✅ DISPLAY SUMMARY WITH ACTUAL DATA (not N/A)
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("PCA Components", n_components_pca)
            with col2:
                if pca_variance_explained is not None:
                    st.metric("Total Variance %", f"{pca_variance_explained.sum()*100:.1f}%")
                else:
                    st.metric("Total Variance %", "Computing...")
            with col3:
                cv_n_folds = st.session_state.get('cv_n_folds', 5)
                st.metric("CV Folds", cv_n_folds)

            st.info("✅ PCA preprocessing applied correctly. See confusion matrix above for per-fold accuracy.")

            # ✅ DISPLAY PCA PLOTS IF DATA IS AVAILABLE
            if pca_preprocessor is not None and pca_loadings is not None:
                st.markdown("#### PCA Visualization")

                # Variance Explained Plot
                if pca_variance_explained is not None:
                    st.markdown("**Variance Explained by Components**")
                    fig_var = go.Figure()
                    fig_var.add_trace(go.Bar(
                        x=[f"PC{i+1}" for i in range(len(pca_variance_explained))],
                        y=pca_variance_explained * 100,
                        name='Variance Explained (%)'
                    ))
                    fig_var.update_layout(
                        title="PCA Variance Explained",
                        xaxis_title="Principal Component",
                        yaxis_title="Variance Explained (%)",
                        height=400
                    )
                    st.plotly_chart(fig_var, use_container_width=True)

                # Loadings Plot
                st.markdown("**PCA Loadings (PC1 vs PC2)**")
                if X_train_scaled_cv is not None:
                    feature_names = tab1_data.get('X_columns', [f"Var {i+1}" for i in range(pca_loadings.shape[0])])
                    fig_loadings = go.Figure()
                    fig_loadings.add_trace(go.Scatter(
                        x=pca_loadings[:, 0],
                        y=pca_loadings[:, 1],
                        mode='markers+text',
                        text=feature_names,
                        textposition='top center',
                        marker=dict(size=10, color='blue')
                    ))
                    fig_loadings.update_layout(
                        title="PCA Loadings Plot",
                        xaxis_title=f"PC1 ({pca_variance_explained[0]*100:.1f}%)" if pca_variance_explained is not None else "PC1",
                        yaxis_title=f"PC2 ({pca_variance_explained[1]*100:.1f}%)" if pca_variance_explained is not None and len(pca_variance_explained) > 1 else "PC2",
                        height=500
                    )
                    st.plotly_chart(fig_loadings, use_container_width=True)

            # ============================================================================
            # 💾 EXPORT TRAINING DATA (after CV is complete)
            # ============================================================================
            st.divider()
            st.markdown("### 💾 Export Training Data")
            st.markdown("Download the preprocessed training/test data used in cross-validation")

            export_col1, export_col2, export_col3 = st.columns(3)

            with export_col1:
                if st.button("📥 Export Training Set", key="export_train_tab2", use_container_width=True):
                    X_train = st.session_state.get('X_train_scaled_cv')
                    y_train = st.session_state.get('y_train_cv')

                    # ✅ LOAD PCA FROM trained_model (where it's actually stored!)
                    trained_model = st.session_state.get('trained_model')
                    pca_preprocessor = trained_model.get('pca_preprocessor') if trained_model else None

                    X_to_export = X_train
                    X_columns = tab1_data.get('X_columns', [f"Feature_{i+1}" for i in range(X_train.shape[1])])
                    feature_info = "original scaled features"

                    # ✅ IF PCA IS AVAILABLE, USE IT!
                    if pca_preprocessor is not None and X_train is not None:
                        try:
                            from classification_utils import project_onto_pca
                            X_train_pca = project_onto_pca(X_train, pca_preprocessor)
                            X_to_export = X_train_pca
                            n_pc = X_train_pca.shape[1]
                            X_columns = [f"PC{i+1}" for i in range(n_pc)]
                            feature_info = f"PCA-transformed: {n_pc} components"
                        except Exception as e:
                            st.warning(f"⚠️ PCA transformation failed: {str(e)}")

                    if X_to_export is not None and y_train is not None:
                        df = pd.DataFrame(X_to_export, columns=X_columns)
                        df.insert(0, 'Class', y_train)
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="📂 Download X_train.csv",
                            data=csv,
                            file_name="classification_X_train.csv",
                            mime="text/csv",
                            key="dl_train"
                        )
                        st.success(f"✅ Training set: {df.shape[0]} samples × {X_to_export.shape[1]} features ({feature_info})")
                    else:
                        st.error("⚠️ Training data not available")

            with export_col2:
                if st.button("📥 Export Test Set", key="export_test_tab2", use_container_width=True):
                    X_test = st.session_state.get('X_test_scaled_cv')
                    y_test = st.session_state.get('y_test_cv')

                    # ✅ LOAD PCA FROM trained_model
                    trained_model = st.session_state.get('trained_model')
                    pca_preprocessor = trained_model.get('pca_preprocessor') if trained_model else None

                    X_to_export = X_test
                    X_columns = tab1_data.get('X_columns', [f"Feature_{i+1}" for i in range(X_test.shape[1])])
                    feature_info = "original scaled features"

                    # ✅ IF PCA IS AVAILABLE, USE IT!
                    if pca_preprocessor is not None and X_test is not None:
                        try:
                            from classification_utils import project_onto_pca
                            X_test_pca = project_onto_pca(X_test, pca_preprocessor)
                            X_to_export = X_test_pca
                            n_pc = X_test_pca.shape[1]
                            X_columns = [f"PC{i+1}" for i in range(n_pc)]
                            feature_info = f"PCA-transformed: {n_pc} components"
                        except Exception as e:
                            st.warning(f"⚠️ PCA transformation failed: {str(e)}")

                    if X_to_export is not None and y_test is not None:
                        df = pd.DataFrame(X_to_export, columns=X_columns)
                        df.insert(0, 'Class', y_test)
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="📂 Download X_test.csv",
                            data=csv,
                            file_name="classification_X_test.csv",
                            mime="text/csv",
                            key="dl_test"
                        )
                        st.success(f"✅ Test set: {df.shape[0]} samples × {X_to_export.shape[1]} features ({feature_info})")
                    else:
                        st.info("ℹ️ Test data not available (no train/test split)")

            with export_col3:
                if st.button("📥 Export Labels", key="export_labels_tab2", use_container_width=True):
                    y_train = st.session_state.get('y_train_cv')
                    y_test = st.session_state.get('y_test_cv')
                    classes = st.session_state.get('classes', [])

                    if y_train is not None:
                        # Create labels mapping DataFrame
                        labels_dict = {
                            'Class_Index': list(range(len(classes))),
                            'Class_Name': classes
                        }
                        df = pd.DataFrame(labels_dict)

                        # Convert to CSV
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="📂 Download labels.csv",
                            data=csv,
                            file_name="classification_labels.csv",
                            mime="text/csv",
                            key="dl_labels"
                        )
                        st.success(f"✅ Labels: {len(classes)} classes")
                    else:
                        st.error("⚠️ Labels not available")

            st.info("💡 **Tip:** Use these files for external validation, documentation, or retraining models in other software (Python, R, MATLAB).")

            # Skip the rest of Tab 2 (standard CV execution) when in PCA mode
            # Just show a button to go back to Tab 1
            st.divider()
            st.markdown("---")
            st.info("💡 **To modify CV parameters or re-run:** Go back to **Tab 1** and adjust PCA or CV settings.")

            # Set flag to skip standard CV section
            skip_standard_cv = True

        else:
            skip_standard_cv = False
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # STANDARD MODE: No PCA preprocessing - standard CV in Tab 2
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

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

            st.divider()

            # --- Cross-Validation Section ---
            st.markdown("### ✅ Cross-Validation Evaluation")

            st.info("""
            **Stratified K-Fold CV:**
            - Each fold maintains class proportions from the full dataset
            - Every class is represented in each fold
            - Provides robust performance estimates
            """)

        # ═══════════════════════════════════════════════════════════════════════════
        # CV RESULTS DISPLAY (Tab 2 is display-only, CV execution is in Tab 1)
        # ═══════════════════════════════════════════════════════════════════════════

        # Check if CV results are available
        if 'cv_results' not in st.session_state:
            st.warning("⚠️ No cross-validation results available")
            st.info("💡 Go to **Tab 1** and click **'Run Cross-Validation'** to evaluate your model.")
            return

        # Display CV results
        cv_res = st.session_state['cv_results']
        y_pred_cv = cv_res['cv_predictions']

        # Get trained model info (may be None if using CV-only mode)
        # NOTE: 'trained' is now set earlier (line 1402) with fallback from cv_results
        # Do NOT overwrite it here!
        # trained = st.session_state.get('trained_model')  # ← COMMENTED OUT - now set at line 1402
        cv_classifier = st.session_state.get('cv_classifier', 'Cross-Validated Classifier')

        # ═══════════════════════════════════════════════════════════════════════
        # DEBUG LOGGING
        # ═══════════════════════════════════════════════════════════════════════

        with st.expander("🔍 DEBUG INFO - Check model loading (click to expand)", expanded=False):
            st.write("**Session State Contents:**")
            st.write(f"- trained_model exists: {'trained_model' in st.session_state}")
            st.write(f"- cv_classifier: {cv_classifier}")
            st.write(f"- cv_results exists: {'cv_results' in st.session_state}")

            trained_info = st.session_state.get('trained_model')

            if trained_info is not None:
                st.write(f"\n**Trained Model Info:**")
                st.write(f"- Name: {trained_info.get('name')}")
                st.write(f"- Has 'X_train': {'X_train' in trained_info}")
                st.write(f"- Has 'y_train': {'y_train' in trained_info}")
                st.write(f"- Has 'model': {'model' in trained_info}")
                st.write(f"- Parameters: {trained_info.get('parameters')}")
            else:
                st.error("❌ trained_model is None or NOT in session_state!")
                st.write("\n**Possible causes:**")
                st.write("1. Model training failed in Tab 1")
                st.write("2. Session was cleared")
                st.write("3. Exception occurred during model saving")
                st.write("4. trained_model was explicitly set to None")

            if 'cv_results' in st.session_state:
                cv_info = st.session_state.get('cv_results')
                st.write(f"\n**CV Results Available:**")
                st.write(f"- Has 'k_value': {'k_value' in cv_info}")
                st.write(f"- Has 'metric': {'metric' in cv_info}")
                st.write(f"- Has 'trained_model': {'trained_model' in cv_info}")
                st.write(f"- Keys: {list(cv_info.keys())[:10]}...")  # Show first 10 keys

        # Get data for CV
        split_done = st.session_state.get('split_done', False)
        if split_done:
            X_for_cv = st.session_state.get('X_train')
            y_for_cv = st.session_state.get('y_train')
            X_for_cv_scaled = st.session_state.get('X_train_scaled')
        else:
            X_for_cv = st.session_state.get('X_full')
            y_for_cv = st.session_state.get('y_full')
            X_for_cv_scaled = st.session_state.get('X_scaled')

        if y_for_cv is None:
            st.error("❌ Data not found. Please go back to Tab 1.")
            return

        # Validate that y_for_cv and y_pred_cv have the same length
        if len(y_for_cv) != len(y_pred_cv):
            st.error(f"❌ Data mismatch detected: Training labels ({len(y_for_cv)} samples) don't match CV predictions ({len(y_pred_cv)} samples)")
            st.warning("This usually happens when the model was trained with different data. Please retrain the model in Tab 1.")
            st.stop()

        cv_metrics = compute_classification_metrics(y_for_cv.values if hasattr(y_for_cv, 'values') else y_for_cv, y_pred_cv, classes)

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
            cv_time = st.session_state.get('cv_time', st.session_state.get('cv_results_time', 0))
            st.metric("CV Time", f"{cv_time:.2f}s")
        with col5:
            pred_time_avg = cv_time / len(y_pred_cv) if len(y_pred_cv) > 0 else 0
            st.metric("Prediction Time (Avg)", f"{pred_time_avg:.4f}s/sample")
        with col6:
            n_features = X_for_cv_scaled.shape[1] if X_for_cv_scaled is not None else X_for_cv.shape[1] if X_for_cv is not None else 0
            n_classes = len(classes)
            st.metric("Model Complexity", f"{n_features} features × {n_classes} classes")

        st.divider()

        # CV Confusion Matrix
        st.markdown("#### Confusion Matrix (CV Predictions)")
        fig_cm_cv = plot_confusion_matrix(
            cv_metrics['confusion_matrix'],
            cv_metrics['classes'].tolist(),
            title=f"Cross-Validation Confusion Matrix - {cv_classifier}"
        )
        st.plotly_chart(fig_cm_cv, use_container_width=True)

        # ═══════════════════════════════════════════════════════════════════════════
        # CLASSIFIER-SPECIFIC DIAGNOSTICS
        # ═══════════════════════════════════════════════════════════════════════════

        st.divider()
        st.markdown("### 🔬 Classifier-Specific Diagnostics")

        # --- LDA/QDA: Mahalanobis Distance Distributions ---
        if cv_classifier in ['LDA', 'QDA']:
            st.markdown("#### 📏 Mahalanobis Distance Distributions")
            st.info(
                "Mahalanobis distances from each sample to class centroids. "
                "Proper class separation shows distinct distribution peaks."
            )

            try:
                # Get Mahalanobis distances from cv_results if available
                mahal_distances = cv_res.get('mahalanobis_distances', None)

                if mahal_distances is not None:
                    # Validate matrix format - should be ndarray of shape (n_samples, n_classes)
                    if isinstance(mahal_distances, dict):
                        st.error("❌ Mahalanobis distances are in old dictionary format. Please re-run CV to update.")
                        st.info("""
        **Action Required:**
        Re-run Cross-Validation in Tab 1 (Step 5) to recalculate distances in the correct format.
        """)
                    elif isinstance(mahal_distances, np.ndarray):
                        # Verify expected shape (n_samples, n_classes)
                        expected_shape = (len(y_for_cv), len(classes))
                        if mahal_distances.shape != expected_shape:
                            st.warning(f"⚠️ Unexpected matrix shape: {mahal_distances.shape}, expected {expected_shape}")

                        # Create 3-tab interface for Mahalanobis distance analysis
                        tab_closest, tab_category, tab_sample = st.tabs([
                            "📊 Closest Category",
                            "🎯 Distance to Specific Class",
                            "🔍 Sample Analysis"
                        ])

                        # ═══════════════════════════════════════════════════════════════════════
                        # NORMALIZE CLASSES AND SAMPLE NAMES
                        # ═══════════════════════════════════════════════════════════════════════

                        y_cv_array = y_for_cv.values if hasattr(y_for_cv, 'values') else y_for_cv

                        # CRITICAL: Normalize all classes to strings for consistent matching
                        classes_normalized = np.array([str(c) for c in sorted(np.unique(y_cv_array))])
                        class_names_list = classes_normalized.tolist()

                        # Convert y_cv_array to strings to match normalized classes
                        y_cv_array_normalized = np.array([str(c) for c in y_cv_array])

                        # ═══════════════════════════════════════════════════════════════════════
                        # GET SAMPLE NAMES FROM SESSION STATE
                        # ═══════════════════════════════════════════════════════════════════════

                        # CRITICAL: Get original sample names from session_state
                        if 'sample_names' in st.session_state:
                            sample_names = st.session_state['sample_names']
                            st.write(f"✅ Using original sample names from dataset ({len(sample_names)} names)")
                        elif hasattr(X_for_cv_scaled, 'index'):
                            # Try to get from DataFrame index
                            sample_names = X_for_cv_scaled.index.tolist()
                            st.write(f"✅ Using sample names from X_for_cv_scaled index")
                        else:
                            # Fallback: use row names if available, otherwise generate names
                            if 'current_data' in st.session_state and hasattr(st.session_state['current_data'], 'index'):
                                sample_names = st.session_state['current_data'].index.tolist()
                                st.write(f"✅ Using sample names from current_data index")
                            else:
                                # Last resort: generate generic names
                                sample_names = [f"Sample_{i}" for i in range(len(y_cv_array))]
                                st.warning("⚠️ Using generated sample names (original names not available)")

                        st.write(f"**Debug Info:**")
                        st.write(f"- Classes (normalized): {class_names_list}")
                        st.write(f"- Sample count: {len(y_cv_array_normalized)}")
                        st.write(f"- Sample names: {sample_names[:5]}... (showing first 5)")

                        # TAB 1: Distance to closest category
                        with tab_closest:
                            st.markdown("**Distance from each sample to its closest class**")
                            st.caption("Shows the minimum distance to any class model. Bars colored by true class.")

                            from classification_utils.plots import plot_mahalanobis_distance_closest_category
                            fig_closest = plot_mahalanobis_distance_closest_category(
                                mahal_distances,
                                y_cv_array_normalized,  # Use normalized array
                                class_names=class_names_list,
                                title=f"{cv_classifier} - Distance to Closest Category (CV)",
                                sample_names=sample_names  # Pass sample names
                            )
                            st.plotly_chart(fig_closest, use_container_width=True)

                        # TAB 2: Distance to specific class
                        with tab_category:
                            st.markdown("**Distance from all samples to one specific class model**")
                            st.caption("Select a class to see how all samples relate to that class's model.")

                            col1, col2 = st.columns([1, 3])
                            with col1:
                                # Use normalized class names for selection
                                target_class_str = st.selectbox(
                                    "Select Target Class",
                                    options=class_names_list,
                                    format_func=lambda x: f"Class {x}",
                                    key="mahal_cv_target_class"
                                )

                            try:
                                from classification_utils.plots import plot_mahalanobis_distance_category

                                fig_category = plot_mahalanobis_distance_category(
                                    mahal_distances,
                                    y_cv_array_normalized,  # Use normalized array
                                    target_class=target_class_str,  # Already a string
                                    class_names=class_names_list,
                                    title=f"{cv_classifier} - Distance to Class {target_class_str} (CV)",
                                    sample_names=sample_names  # Pass sample names
                                )
                                st.plotly_chart(fig_category, use_container_width=True)

                            except Exception as e:
                                st.error(f"❌ Error plotting: {str(e)}")
                                with st.expander("📋 Debug Details"):
                                    st.write(f"**Error Type**: {type(e).__name__}")
                                    st.write(f"**Message**: {str(e)}")
                                    st.write(f"**Target Class**: {target_class_str} (type: {type(target_class_str).__name__})")
                                    st.write(f"**Available Classes**: {class_names_list}")
                                    st.write(f"**Y Unique Values**: {np.unique(y_cv_array_normalized)}")

                        # TAB 3: Sample-specific analysis
                        with tab_sample:
                            st.markdown("**Distance from one sample to all class models**")
                            st.caption("Examine a specific sample to understand its classification.")

                            col1, col2, col3 = st.columns([2, 1, 1])
                            with col1:
                                # Create selectbox with sample names for easy selection
                                sample_options = [f"{sample_names[i]} (Index {i})"
                                                for i in range(len(sample_names))]
                                selected_option = st.selectbox(
                                    "Select Sample",
                                    options=sample_options,
                                    key="mahal_cv_sample_select"
                                )
                                # Extract index from selected option
                                sample_idx = int(selected_option.split("(Index ")[-1].rstrip(")"))

                            # Show sample info
                            with col2:
                                st.metric("Sample Name", sample_names[sample_idx])

                            with col3:
                                true_class = y_cv_array_normalized[sample_idx]
                                st.metric("True Class", true_class)

                            try:
                                from classification_utils.plots import plot_mahalanobis_distance_object

                                # Get sample name for title
                                sample_name = sample_names[sample_idx]

                                fig_sample = plot_mahalanobis_distance_object(
                                    mahal_distances,
                                    sample_idx=sample_idx,
                                    y_true=y_cv_array_normalized,  # Use normalized array
                                    class_names=class_names_list,
                                    title=f"{cv_classifier} - {sample_name} (True: {true_class}) (CV)",
                                    sample_names=sample_names  # Pass sample names
                                )
                                st.plotly_chart(fig_sample, use_container_width=True)

                            except Exception as e:
                                st.error(f"❌ Error plotting: {str(e)}")
                                with st.expander("📋 Debug Details"):
                                    st.write(f"**Error Type**: {type(e).__name__}")
                                    st.write(f"**Message**: {str(e)}")
                                    st.write(f"**Sample Index**: {sample_idx}")
                                    st.write(f"**Sample Name**: {sample_names[sample_idx] if sample_idx < len(sample_names) else 'N/A'}")
                                    st.write(f"**True Class**: {y_cv_array_normalized[sample_idx]}")
                                    st.write(f"**Mahal Distances Shape**: {mahal_distances.shape}")
                    else:
                        st.error(f"❌ Invalid mahalanobis_distances type: {type(mahal_distances).__name__}")
                else:
                    st.warning("⚠️ Mahalanobis distances not available in CV results")
                    st.info("""
        **Possible causes:**
        1. CV was executed without Mahalanobis distance calculation
        2. Data type mismatch during CV processing
        3. Model requires training first (Tab 1, Step 4)

        **Solution:**
        - Re-run CV in Tab 1 (Step 5) to recalculate distances
        - Or train a model in Tab 1 (Step 4) before CV
        """)

            except Exception as e:
                st.warning(f"Could not display Mahalanobis distance plot: {str(e)}")

        # --- kNN: Distance & Neighbor Analysis ---
        elif cv_classifier == 'kNN':
            st.markdown("#### 📍 Distance & Neighbor Analysis (kNN)")

            # Debug expander (can be removed later)
            with st.expander("🔍 kNN Model Loading Debug", expanded=False):
                st.write(f"trained variable is None: {trained is None}")
                st.write(f"cv_res has 'trained_model': {'trained_model' in cv_res if cv_res else 'N/A'}")
                if cv_res:
                    st.write(f"cv_res has 'X_train': {'X_train' in cv_res}")
                    st.write(f"cv_res has 'k_value': {'k_value' in cv_res}")
                    st.write(f"cv_res keys: {list(cv_res.keys())}")

            try:
                # Try multiple sources to get the model
                knn_model = None

                # Priority 1: From trained variable
                if trained and trained.get('name') == 'kNN':
                    knn_model = trained
                    st.success("✓ Using trained model from session state")

                # Priority 2: From cv_results
                elif cv_res and cv_res.get('trained_model'):
                    knn_model = cv_res.get('trained_model')
                    st.success("✓ Using trained model from cv_results")

                # Priority 3: Check if cv_results has the data directly
                elif cv_res and 'X_train' in cv_res:
                    # Model might not be in trained_model key, but data is there
                    knn_model = {
                        'model': cv_res.get('model'),
                        'X_train': cv_res.get('X_train'),
                        'y_train': cv_res.get('y_train'),
                        'parameters': {
                            'k': cv_res.get('k_value', 3),
                            'metric': cv_res.get('metric', 'euclidean')
                        }
                    }
                    st.warning("⚠️ Reconstructed model from cv_results data")

                if knn_model is None:
                    st.error("❌ Could not find kNN model in any location")
                    st.info("**Debugging info:**")
                    st.write(f"- trained: {trained}")
                    st.write(f"- cv_res keys: {list(cv_res.keys()) if cv_res else 'No cv_res'}")
                else:
                    # Extract parameters safely with multiple fallbacks
                    k_value = knn_model.get('parameters', {}).get('k', cv_res.get('k_value', 3))
                    metric = knn_model.get('parameters', {}).get('metric', cv_res.get('metric', 'euclidean'))

                    # Get data - with fallback to cv_results (explicit None check to avoid array ambiguity)
                    X_train = cv_res.get('X_train') if cv_res.get('X_train') is not None else knn_model.get('X_train')
                    y_train = cv_res.get('y_train') if cv_res.get('y_train') is not None else knn_model.get('y_train')

                    # Get the actual model object
                    knn_model_obj = knn_model.get('model') if isinstance(knn_model, dict) else knn_model

                    if X_train is None or y_train is None:
                        st.warning("⚠️ Training data not available in model")
                        st.info("ℹ️ The model was trained but training data is not stored. Showing CV metrics only.")

                        # Show basic metrics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("k (Neighbors)", k_value)
                        with col2:
                            st.metric("Distance Metric", metric.capitalize())
                        with col3:
                            accuracy = cv_metrics.get('accuracy', 0)
                            st.metric("CV Accuracy", f"{accuracy:.1f}%")
                    else:
                        # === METRICS DISPLAY ===
                        st.markdown("##### 📈 Model Configuration")

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("k (Neighbors)", k_value)
                        with col2:
                            st.metric("Distance Metric", metric.capitalize())
                        with col3:
                            accuracy = cv_metrics.get('accuracy', 0)
                            st.metric("CV Accuracy", f"{accuracy:.1f}%")

                        # === DISTANCE STATISTICS ===
                        st.markdown("##### 📊 Distance Statistics")

                        try:
                            # Calculate pairwise distance matrix
                            from classification_utils.calculations import calculate_distance_matrix
                            dist_matrix = calculate_distance_matrix(X_train, X_train, metric=metric)

                            # Remove diagonal (self-distances = 0)
                            n_samples = dist_matrix.shape[0]
                            mask = ~np.eye(n_samples, dtype=bool)
                            distances_flat = dist_matrix[mask]

                            # Statistics
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Mean Distance", f"{np.mean(distances_flat):.3f}")
                            with col2:
                                st.metric("Median Distance", f"{np.median(distances_flat):.3f}")
                            with col3:
                                st.metric("Min Distance", f"{np.min(distances_flat):.3f}")
                            with col4:
                                st.metric("Max Distance", f"{np.max(distances_flat):.3f}")

                            # === DISTANCE HISTOGRAM ===
                            st.markdown("##### 📊 Distance Distribution")

                            fig_hist = go.Figure()
                            fig_hist.add_trace(go.Histogram(
                                x=distances_flat,
                                nbinsx=50,
                                name="Distances",
                                marker=dict(color='steelblue', line=dict(color='black', width=1))
                            ))

                            fig_hist.update_layout(
                                title=f"Distribution of Pairwise {metric.capitalize()} Distances",
                                xaxis_title="Distance",
                                yaxis_title="Frequency",
                                height=400,
                                showlegend=False,
                                plot_bgcolor='white',
                                xaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray'),
                                yaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray')
                            )

                            st.plotly_chart(fig_hist, use_container_width=True)

                        except Exception as e:
                            st.warning(f"Could not calculate distance statistics: {str(e)}")

                        # ═══════════════════════════════════════════════════════════════
                        # NEW: DISTANCE DISTRIBUTION BY CLASS
                        # ═══════════════════════════════════════════════════════════════

                        st.markdown("##### 📊 Distance Distribution by True Class")

                        try:
                            classes_list = sorted(np.unique(y_train))
                            fig_by_class = go.Figure()

                            colors = ['#0000FF', '#FF00FF', '#00FFFF', '#FF0000', '#00FF00']  # Blue, Magenta, Cyan, Red, Green

                            for idx, class_label in enumerate(classes_list):
                                class_mask = y_train == class_label
                                class_indices = np.where(class_mask)[0]

                                if len(class_indices) > 1:
                                    class_dists = []
                                    for i in class_indices:
                                        for j in class_indices:
                                            if i < j:
                                                class_dists.append(dist_matrix[i, j])

                                    if class_dists:
                                        fig_by_class.add_trace(go.Histogram(
                                            x=class_dists,
                                            nbinsx=30,
                                            name=f"Class {class_label}",
                                            marker_color=colors[idx % len(colors)],
                                            opacity=0.7
                                        ))

                            fig_by_class.update_layout(
                                title="Intra-Class Distance Distribution (How tight each class is)",
                                xaxis_title="Distance",
                                yaxis_title="Frequency",
                                barmode='overlay',
                                height=400,
                                hovermode='x unified',
                                plot_bgcolor='white',
                                xaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray'),
                                yaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray')
                            )

                            st.plotly_chart(fig_by_class, use_container_width=True)
                            st.caption("Lower distances within a class indicate tighter clusters (easier to classify)")

                        except Exception as e:
                            st.warning(f"Could not plot distance by class: {str(e)}")

                        # ═══════════════════════════════════════════════════════════════
                        # NEW: NEIGHBOR VOTING ANALYSIS
                        # ═══════════════════════════════════════════════════════════════

                        st.markdown("##### 🗳️ Neighbor Voting Analysis")

                        try:
                            y_true_arr = y_for_cv.values if hasattr(y_for_cv, 'values') else np.asarray(y_for_cv)
                            y_pred_arr = y_pred_cv.values if hasattr(y_pred_cv, 'values') else np.asarray(y_pred_cv)

                            # Calculate distance from test samples to training samples
                            from classification_utils.calculations import calculate_distance_matrix
                            dist_test_train = calculate_distance_matrix(X_for_cv_scaled, X_train, metric=metric)

                            voting_data = []

                            for sample_idx in range(min(len(X_for_cv_scaled), 100)):  # Limit to first 100 for performance
                                distances = dist_test_train[sample_idx]
                                k_nearest_idx = np.argsort(distances)[:k_value]
                                k_nearest_labels = y_train[k_nearest_idx]

                                true_label = y_true_arr[sample_idx]
                                pred_label = y_pred_arr[sample_idx]

                                # Count votes
                                classes_unique = sorted(np.unique(y_train))
                                votes = {}
                                for cls in classes_unique:
                                    votes[cls] = int(np.sum(k_nearest_labels == cls))

                                # Confidence
                                votes_for_pred = votes.get(pred_label, 0)
                                confidence = votes_for_pred / k_value * 100

                                # Status
                                is_correct = true_label == pred_label
                                status = "✓" if is_correct else "✗"

                                voting_data.append({
                                    'Sample': sample_idx,
                                    'True': true_label,
                                    'Pred': pred_label,
                                    'Votes': f"{votes_for_pred}/{k_value}",
                                    'Conf%': f"{confidence:.0f}",
                                    'Status': status
                                })

                            voting_df = pd.DataFrame(voting_data)

                            # Filter to show interesting samples
                            interesting = voting_df[
                                (voting_df['Conf%'] != '100') |
                                (voting_df['Status'] == '✗')
                            ].head(20)

                            if len(interesting) > 0:
                                st.write(f"**Samples with low confidence or misclassified** (showing {len(interesting)}):")
                                st.dataframe(interesting, use_container_width=True, hide_index=True)
                            else:
                                st.success("✅ All samples have 100% confidence and are correct!")

                        except Exception as e:
                            st.warning(f"Could not analyze voting: {str(e)}")

                        # ═══════════════════════════════════════════════════════════════
                        # NEW: CORRECT VS MISCLASSIFIED DISTANCE COMPARISON
                        # ═══════════════════════════════════════════════════════════════

                        st.markdown("##### 📏 Distance Comparison: Correct vs Misclassified")

                        try:
                            y_true_arr = y_for_cv.values if hasattr(y_for_cv, 'values') else np.asarray(y_for_cv)
                            y_pred_arr = y_pred_cv.values if hasattr(y_pred_cv, 'values') else np.asarray(y_pred_cv)

                            correct_mask = y_true_arr == y_pred_arr
                            misclass_mask = ~correct_mask

                            # Mean distance to nearest neighbor
                            mean_dist_correct = []
                            mean_dist_misclass = []

                            for sample_idx in range(len(X_for_cv_scaled)):
                                distances = dist_test_train[sample_idx]
                                nearest_dist = np.min(distances[distances > 0]) if np.any(distances > 0) else 0

                                if correct_mask[sample_idx]:
                                    mean_dist_correct.append(nearest_dist)
                                else:
                                    mean_dist_misclass.append(nearest_dist)

                            fig_comparison = go.Figure()

                            if len(mean_dist_correct) > 0:
                                fig_comparison.add_trace(go.Box(
                                    y=mean_dist_correct,
                                    name='Correctly Classified',
                                    marker_color='lightgreen',
                                    boxmean='sd'
                                ))

                            if len(mean_dist_misclass) > 0:
                                fig_comparison.add_trace(go.Box(
                                    y=mean_dist_misclass,
                                    name='Misclassified',
                                    marker_color='lightcoral',
                                    boxmean='sd'
                                ))

                            fig_comparison.update_layout(
                                title="Distance to Nearest Neighbor: Correct vs Misclassified Samples",
                                yaxis_title="Distance",
                                height=400,
                                showlegend=True,
                                plot_bgcolor='white'
                            )

                            st.plotly_chart(fig_comparison, use_container_width=True)

                            # Statistics
                            col1, col2 = st.columns(2)
                            with col1:
                                if len(mean_dist_correct) > 0:
                                    st.metric(
                                        "Correct (avg dist)",
                                        f"{np.mean(mean_dist_correct):.3f}",
                                        help=f"n={len(mean_dist_correct)}"
                                    )
                            with col2:
                                if len(mean_dist_misclass) > 0:
                                    st.metric(
                                        "Misclassified (avg dist)",
                                        f"{np.mean(mean_dist_misclass):.3f}",
                                        help=f"n={len(mean_dist_misclass)}"
                                    )

                            st.caption("Higher distances for misclassified samples suggest they're in ambiguous regions")

                        except Exception as e:
                            st.warning(f"Could not compare distances: {str(e)}")

                        # === K PERFORMANCE CHART (if available) ===
                        st.markdown("##### 📈 k-Value Performance Metrics")

                        try:
                            # Simple alternative: just show metrics instead of k-performance chart
                            col1, col2 = st.columns(2)

                            with col1:
                                accuracy = cv_metrics.get('accuracy', 0)
                                sensitivity = cv_metrics.get('average_sensitivity', 0)
                                st.metric("Accuracy", f"{accuracy:.1f}%")
                                st.metric("Avg Sensitivity", f"{sensitivity:.1f}%")

                            with col2:
                                specificity = cv_metrics.get('average_specificity', 0)
                                n_misclass = len(cv_res.get('misclassified_indices', []))
                                st.metric("Avg Specificity", f"{specificity:.1f}%")
                                st.metric("Misclassified", f"{n_misclass}")

                        except Exception as e:
                            st.info(f"Performance metrics unavailable: {str(e)}")

                        # === MISCLASSIFIED SAMPLES TABLE ===
                        st.markdown("##### ❌ Misclassified Samples Details")

                        try:
                            misclass_indices = cv_res.get('misclassified_indices', [])

                            if len(misclass_indices) == 0:
                                st.success("✅ No misclassified samples!")
                            else:
                                # Get predictions
                                y_true_arr = y_for_cv.values if hasattr(y_for_cv, 'values') else np.asarray(y_for_cv)
                                y_pred_arr = y_pred_cv.values if hasattr(y_pred_cv, 'values') else np.asarray(y_pred_cv)

                                # Build detailed table
                                misclass_data = []
                                for idx in misclass_indices[:20]:  # Show first 20
                                    misclass_data.append({
                                        'Sample Index': idx,
                                        'True Class': y_true_arr[idx],
                                        'Predicted': y_pred_arr[idx]
                                    })

                                if misclass_data:
                                    misclass_df = pd.DataFrame(misclass_data)
                                    st.dataframe(misclass_df, use_container_width=True, hide_index=True)

                                    if len(misclass_indices) > 20:
                                        st.caption(f"Showing first 20 of {len(misclass_indices)} misclassified samples")

                        except Exception as e:
                            st.warning(f"Could not display misclassified samples: {str(e)}")

            except Exception as e:
                st.error(f"Error in kNN analysis: {str(e)}")
                st.info("💡 Try re-running CV in Tab 1 to regenerate model data")

        # --- SIMCA/UNEQ: Coomans Plot ---
        elif cv_classifier in ['SIMCA', 'UNEQ']:
            st.markdown("#### 🎯 Coomans Plot (Class Comparison)")
            st.info(
                "Coomans plot compares distances from samples to two classes. "
                "Good separation shows distinct clusters."
            )

            try:
                # Get available classes for comparison
                classes_list = list(cv_metrics.get('sensitivity_per_class', {}).keys())

                if len(classes_list) >= 2:
                    col_class1, col_class2 = st.columns(2)

                    with col_class1:
                        selected_class1 = st.selectbox(
                            "Class 1",
                            options=classes_list,
                            key="coomans_class1_diag_tab2"
                        )

                    with col_class2:
                        remaining_classes = [c for c in classes_list if c != selected_class1]
                        selected_class2 = st.selectbox(
                            "Class 2",
                            options=remaining_classes if remaining_classes else [selected_class1],
                            key="coomans_class2_diag_tab2"
                        )

                    if selected_class1 != selected_class2:
                        try:
                            fig_coomans = plot_coomans(
                                y_for_cv.values if hasattr(y_for_cv, 'values') else y_for_cv,
                                y_pred_cv,
                                class_1=selected_class1,
                                class_2=selected_class2,
                                cv_results=cv_res,
                                classifier_type=cv_classifier.lower(),
                                title=f"{cv_classifier} - Coomans Plot: {selected_class1} vs {selected_class2}"
                            )
                            st.plotly_chart(fig_coomans, use_container_width=True)
                        except Exception as e:
                            st.warning(f"Could not display Coomans plot: {str(e)}")
                else:
                    st.info("Need at least 2 classes for Coomans plot")

            except Exception as e:
                st.warning(f"Could not prepare Coomans plot: {str(e)}")

        st.divider()

        # ═══════════════════════════════════════════════════════════════════════════
        # GENERAL DIAGNOSTICS: ALL CLASSIFIERS
        # ═══════════════════════════════════════════════════════════════════════════

        st.markdown("### 📋 General Diagnostics")

        # --- Misclassified Samples ---
        st.markdown("#### 🔍 Misclassified Samples")

        try:
            misclass_indices = cv_res.get('misclassified_indices', [])
            n_misclass = len(misclass_indices) if misclass_indices else 0

            if n_misclass == 0:
                st.success("✅ Perfect classification! No misclassified samples.")
            else:
                st.warning(f"⚠️ {n_misclass} misclassified samples out of {len(y_for_cv)}")

                # Show misclassified samples table
                y_true_arr = y_for_cv.values if hasattr(y_for_cv, 'values') else np.asarray(y_for_cv)
                y_pred_arr = y_pred_cv.values if hasattr(y_pred_cv, 'values') else np.asarray(y_pred_cv)

                misclass_data = []
                for idx in (misclass_indices[:20] if len(misclass_indices) > 20 else misclass_indices):
                    misclass_data.append({
                        'Sample Index': idx,
                        'True Class': y_true_arr[idx],
                        'Predicted Class': y_pred_arr[idx]
                    })

                misclass_df = pd.DataFrame(misclass_data)
                st.dataframe(misclass_df, use_container_width=True, hide_index=True)

                if len(misclass_indices) > 20:
                    st.info(f"... and {len(misclass_indices) - 20} more")

        except Exception as e:
            st.info("Misclassified samples not available")

        # ═══════════════════════════════════════════════════════════════════════════
        # LEGACY COOMANS PLOT SECTION (Keep for compatibility)
        # ═══════════════════════════════════════════════════════════════════════════

        # Coomans plot for SIMCA/UNEQ (with debug diagnostics and class selection)
        st.divider()
        st.markdown("#### 📍 Coomans Plot (CV Diagnostics - 2-Class Comparison)")

        # Debug info: Show model and class information
        col1, col2, col3 = st.columns(3)
        with col1:
            model_name = trained['name'] if trained else cv_classifier
            st.metric("Model Type", model_name)
        with col2:
            st.metric("Number of Classes", len(classes))
        with col3:
            classes_str = ', '.join([str(c) for c in classes])
            st.metric("Classes", classes_str)

        # Check 1: Classifier type
        classifier_name = trained['name'] if trained else cv_classifier
        if classifier_name not in ['SIMCA', 'UNEQ']:
            st.info(f"ℹ️ Coomans plot is only available for SIMCA and UNEQ classifiers. Current model: {classifier_name}")
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

        # SAFETY CHECK: Trained model required for Coomans plot
        if trained is None:
            st.info("ℹ️ Coomans plot requires a trained SIMCA or UNEQ model.")
            st.info("📌 To use this feature:")
            st.write("1. Train a SIMCA or UNEQ model in **Tab 1 (Model Training)**")
            st.write("2. Return to **Tab 2 (Diagnostics)** to view Coomans plot")
        elif trained.get('name') not in ['SIMCA', 'UNEQ']:
            st.info(f"ℹ️ Coomans plot is only available for SIMCA and UNEQ. Current model: {trained.get('name')}")
        else:
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
                        st.write(f"- trained['name']: {trained.get('name', 'N/A')}")
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
            title=f"{cv_classifier} CV Performance"
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

        # Check if we have a trained model (required for detailed distance analysis)
        if trained is None:
            st.info("ℹ️ **Distance Distributions** require a trained model.")
            st.caption(
                "These detailed analyses use the trained model to compute distances. "
                "In CV-only mode (with PCA preprocessing), the model is trained internally during CV but not stored. "
                "To view detailed distance distributions and sample-level analysis, train a model in Tab 1 first."
            )
        elif trained['name'] in ['SIMCA', 'UNEQ']:
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

            # ✅ Use classifier_model if available (for PCA models), otherwise use model
            model_to_use = trained.get('classifier_model', trained['model'])

            # ✅ If PCA was used, transform data to PCA space first
            if trained.get('pca_preprocessor') is not None:
                from classification_utils import project_onto_pca
                X_for_prediction = project_onto_pca(X_for_cv_scaled, trained['pca_preprocessor'])
            else:
                X_for_prediction = X_for_cv_scaled

            if trained['name'] == 'LDA':
                y_pred, distances_array = predict_lda(X_for_prediction, model_to_use)
            elif trained['name'] == 'QDA':
                y_pred, distances_array = predict_qda(X_for_prediction, model_to_use)

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

        # Distance distribution for this class (requires trained model)
        if trained is None:
            st.info("ℹ️ **Per-Class Distance Analysis** requires a trained model. Train a model in Tab 1 to view detailed distance distributions.")
        elif trained['name'] in ['LDA', 'QDA', 'SIMCA', 'UNEQ']:
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
                # ✅ Use classifier_model if available (for PCA models), otherwise use model
                model_to_use = trained.get('classifier_model', trained['model'])

                # ✅ If PCA was used, transform data to PCA space first
                if trained.get('pca_preprocessor') is not None:
                    from classification_utils import project_onto_pca
                    X_for_prediction = project_onto_pca(X_for_cv_scaled, trained['pca_preprocessor'])
                else:
                    X_for_prediction = X_for_cv_scaled

                if trained['name'] == 'LDA':
                    y_pred, distances_array = predict_lda(X_for_prediction, model_to_use)
                else:  # QDA
                    y_pred, distances_array = predict_qda(X_for_prediction, model_to_use)

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
        # ✅ Use saved train_sample_names if using 70% split, otherwise use full dataset names
        if st.session_state.get('split_done', False):
            # Using 70-30 split, get train sample names
            saved_train_names = st.session_state.get('train_sample_names')
            if saved_train_names is not None:
                sample_names_dict = {i: saved_train_names[i] for i in range(len(X_for_cv))}
            elif hasattr(X_for_cv, 'index') and not isinstance(X_for_cv.index, pd.RangeIndex):
                sample_names_dict = {i: X_for_cv.index[i] for i in range(len(X_for_cv))}
            else:
                sample_names_dict = {i: str(i+1) for i in range(len(X_for_cv))}
        else:
            # Using full dataset, get names directly from X_for_cv
            if hasattr(X_for_cv, 'index') and not isinstance(X_for_cv.index, pd.RangeIndex):
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

        # Distance to each class (classifier-specific, requires trained model)
        st.markdown("#### Distance to Each Class")

        if trained is None:
            st.info("ℹ️ **Distance to Each Class** analysis requires a trained model. Train a model in Tab 1 to view this analysis.")
        else:
            try:
                distances_to_classes = []

                # ✅ Use classifier_model if available (for PCA models), otherwise use model
                model_to_use = trained.get('classifier_model', trained['model'])

                # ✅ If PCA was used, transform data to PCA space first
                if trained.get('pca_preprocessor') is not None:
                    from classification_utils import project_onto_pca
                    X_for_prediction = project_onto_pca(X_for_cv_scaled, trained['pca_preprocessor'])
                else:
                    X_for_prediction = X_for_cv_scaled

                if trained['name'] == 'LDA':
                    # Get distances for all samples
                    _, distances_array = predict_lda(X_for_prediction, model_to_use)
                    distances_to_classes = distances_array[selected_sample_idx_tab2, :].tolist()

                elif trained['name'] == 'QDA':
                    # Get distances for all samples
                    _, distances_array = predict_qda(X_for_prediction, model_to_use)
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

        # ═══════════════════════════════════════════════════════════════════════
        # GET TRAINED MODEL (with fallback for kNN)
        # ═══════════════════════════════════════════════════════════════════════

        trained = st.session_state.get('trained_model')
        cv_results_info = st.session_state.get('cv_results')

        # Fallback: If trained is None, reconstruct from cv_results (same as Tab 2)
        if trained is None and cv_results_info is not None:
            cv_classifier = st.session_state.get('cv_classifier', 'Unknown')

            if cv_classifier == 'kNN' and 'X_train' in cv_results_info:
                X_train_cv = cv_results_info.get('X_train')
                y_train_cv = cv_results_info.get('y_train')
                metric_cv = cv_results_info.get('metric', 'euclidean')

                # Calculate covariance if using Mahalanobis
                cov_cv = None
                if metric_cv == 'mahalanobis' and X_train_cv is not None:
                    cov_cv = np.cov(X_train_cv, rowvar=False) + np.eye(X_train_cv.shape[1]) * 1e-10

                # ✅ CORRECT: For kNN, the model IS the training data dict
                trained = {
                    'name': 'kNN',
                    'n_features': X_train_cv.shape[1] if X_train_cv is not None else 0,
                    'classes': np.unique(y_train_cv).tolist() if y_train_cv is not None else [],
                    'scaling_method': st.session_state.get('scaling_method', 'autoscale'),
                    'parameters': {
                        'k': cv_results_info.get('k_value', 5),
                        'metric': metric_cv,
                    },
                    'model': {  # ← THIS is the actual kNN "model"
                        'X_train': X_train_cv,
                        'y_train': y_train_cv,
                        'metric': metric_cv,
                        'cov': cov_cv,  # ← REQUIRED by predict_knn
                    }
                }
                st.info("✓ Reconstructed kNN model from cv_results for testing")

        if trained is None:
            st.warning("⚠️ Train a model first in Tab 1 before testing")
            st.info("💡 Go to **Tab 1: Setup & Configuration** and click 'Run Cross-Validation'")
        else:

            # --- CHECK FOR OPTIONAL TRAIN/TEST SPLIT FROM TAB 1 ---
            st.markdown("## 📥 Section 1: Select Test Data")

            split_done = st.session_state.get('split_done', False)

            if split_done:
                # OPTION 1: Use split from Tab 1
                st.success("✅ **Train/test split found from Tab 1!**")

                # ✅ SHOW CORRECT FEATURE COUNT
                use_pca = trained.get('parameters', {}).get('use_pca', False)
                n_components_pca = trained.get('parameters', {}).get('n_components')

                if use_pca and n_components_pca:
                    display_features = f"**{n_components_pca} PCA components**"
                else:
                    display_features = f"{trained['n_features']} features"

                st.info(f"🤖 **Model**: {trained['name']} | Features: {display_features}")

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

                    # ✅ IF MODEL USES PCA, APPLY IT TO TEST DATA
                    use_pca = trained.get('parameters', {}).get('use_pca', False)

                    if use_pca:
                        pca_preprocessor = trained.get('pca_preprocessor')
                        if pca_preprocessor is not None:
                            try:
                                from classification_utils import project_onto_pca
                                X_test_scaled = project_onto_pca(X_test_scaled, pca_preprocessor)
                                st.success("✅ PCA transformation applied to test data")
                            except Exception as e:
                                st.error(f"❌ PCA transformation failed: {str(e)}")
                                return
                        else:
                            st.error("❌ PCA preprocessor not found in trained model")
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

                # ✅ Get the correct model
                model_to_use = trained.get('classifier_model', trained['model'])
                # ✅ X_test_scaled è già nel formato corretto (PCA se usato, altrimenti originale)
                X_test_for_prediction = X_test_scaled

                if trained['name'] == 'LDA':
                    y_pred_test, _ = predict_lda(X_test_for_prediction, model_to_use)
                elif trained['name'] == 'QDA':
                    y_pred_test, _ = predict_qda(X_test_for_prediction, model_to_use)
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

                        # ═══════════════════════════════════════════════════════════════════════════
                        # kNN: SINGLE SAMPLE ANALYSIS (TEST DATA)
                        # ═══════════════════════════════════════════════════════════════════════════
                        if trained['name'] == 'kNN':
                            st.divider()
                            st.markdown("### 📌 Single Sample Analysis (Test) - kNN")

                            try:
                                # Create sample names dictionary for test set
                                saved_test_names = st.session_state.get('test_sample_names')
                                sample_names_dict_test_knn = {}

                                if saved_test_names is not None and len(saved_test_names) == len(X_test):
                                    sample_names_dict_test_knn = {i: str(saved_test_names[i]) for i in range(len(X_test))}
                                    st.caption(f"✓ Using {len(saved_test_names)} sample names from 70-30 split")

                                elif hasattr(X_test, 'index') and not isinstance(X_test.index, pd.RangeIndex):
                                    sample_names_dict_test_knn = {i: str(X_test.index[i]) for i in range(len(X_test))}
                                    st.caption(f"✓ Using sample names from X_test.index")

                                else:
                                    test_start_offset = locals().get('test_start', 1)
                                    sample_names_dict_test_knn = {i: f"Test_{test_start_offset + i}" for i in range(len(X_test))}
                                    st.warning(f"⚠️ Using numeric indices (no sample names found)")

                                # Reorder samples: misclassified first, then correct
                                misclassified_idx_test_knn = np.where(y_test != y_pred_test)[0]
                                correct_idx_test_knn = np.where(y_test == y_pred_test)[0]
                                ordered_indices_test_knn = np.concatenate([misclassified_idx_test_knn, correct_idx_test_knn])

                                # Sample selector
                                selected_sample_idx_tab4_knn = st.selectbox(
                                    "Select sample to analyze",
                                    options=ordered_indices_test_knn,
                                    format_func=lambda x: f"Sample {sample_names_dict_test_knn[x]}: True={y_test[x]}, Pred={y_pred_test[x]}",
                                    key="sample_selector_tab4_knn"
                                )

                                # Sample info metrics
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("True Class", y_test[selected_sample_idx_tab4_knn])
                                with col2:
                                    st.metric("Predicted", y_pred_test[selected_sample_idx_tab4_knn])
                                with col3:
                                    match = "✅ Correct" if y_test[selected_sample_idx_tab4_knn] == y_pred_test[selected_sample_idx_tab4_knn] else "❌ Error"
                                    st.metric("Result", match)

                                # ═══════════════════════════════════════════════════════════════════════
                                # k-NEAREST NEIGHBORS ANALYSIS
                                # ═══════════════════════════════════════════════════════════════════════

                                st.markdown("#### 🔍 k-Nearest Neighbors Analysis")

                                # Get the test sample
                                test_sample = X_test_scaled[selected_sample_idx_tab4_knn].reshape(1, -1)

                                # Get distances to all training samples
                                from classification_utils.calculations import calculate_distance_matrix
                                k_value = cv_results.get('k_value', 5)
                                metric = cv_results.get('metric', 'euclidean')

                                distances_to_train = calculate_distance_matrix(
                                    test_sample,
                                    trained['model']['X_train'],
                                    metric=metric,
                                    cov=trained['model'].get('cov')
                                )[0]

                                # Get k nearest neighbors
                                k_nearest_indices = np.argsort(distances_to_train)[:k_value]
                                k_nearest_distances = distances_to_train[k_nearest_indices]
                                y_train_model = trained['model']['y_train']
                                k_nearest_labels = y_train_model[k_nearest_indices]

                                # Create neighbors table
                                neighbors_data = []
                                train_sample_names = st.session_state.get('train_sample_names')
                                for rank, (neighbor_idx, distance, label) in enumerate(zip(k_nearest_indices, k_nearest_distances, k_nearest_labels), 1):
                                    # Get training sample name
                                    if train_sample_names is not None and neighbor_idx < len(train_sample_names):
                                        sample_name = train_sample_names[neighbor_idx]
                                    else:
                                        sample_name = f"Train_{neighbor_idx}"

                                    neighbors_data.append({
                                        'Rank': rank,
                                        'Training Sample': sample_name,
                                        'Class': label,
                                        'Distance': f"{distance:.4f}",
                                        'Vote': "🗳️"
                                    })

                                neighbors_df = pd.DataFrame(neighbors_data)
                                st.dataframe(neighbors_df, use_container_width=True, hide_index=True)

                                # Voting summary
                                st.markdown("#### 🗳️ Neighbor Voting Summary")

                                vote_counts = {}
                                for label in k_nearest_labels:
                                    vote_counts[label] = vote_counts.get(label, 0) + 1

                                # Create voting display
                                cols = st.columns(len(vote_counts))
                                for col_idx, (class_label, count) in enumerate(sorted(vote_counts.items())):
                                    with cols[col_idx]:
                                        confidence = 100 * count / k_value
                                        winner = "👑 WINNER" if class_label == y_pred_test[selected_sample_idx_tab4_knn] else ""
                                        st.metric(
                                            f"Class {class_label}",
                                            f"{count}/{k_value}",
                                            f"{confidence:.0f}% {winner}",
                                            delta_color="off"
                                        )

                                # Prediction confidence
                                pred_votes = vote_counts.get(y_pred_test[selected_sample_idx_tab4_knn], 0)
                                pred_confidence = 100 * pred_votes / k_value

                                st.markdown("#### 🎯 Prediction Confidence")
                                st.progress(pred_confidence / 100, text=f"{pred_confidence:.0f}% Confidence")

                                if pred_confidence == 100:
                                    st.success("✅ All neighbors vote for predicted class - High confidence!")
                                elif pred_confidence >= 60:
                                    st.info("ℹ️ Most neighbors vote for predicted class - Good confidence")
                                else:
                                    st.warning(f"⚠️ Weak consensus - Only {pred_votes}/{k_value} neighbors vote for {y_pred_test[selected_sample_idx_tab4_knn]}")

                                # Feature values
                                st.markdown("#### 📊 Feature Values (Test Sample)")

                                if isinstance(X_test, pd.DataFrame):
                                    feature_vals_test = X_test.iloc[selected_sample_idx_tab4_knn].values
                                else:
                                    feature_vals_test = X_test[selected_sample_idx_tab4_knn]

                                feature_df_test = pd.DataFrame({
                                    'Feature': x_columns,
                                    'Value': feature_vals_test
                                })
                                st.dataframe(feature_df_test, use_container_width=True, hide_index=True)

                                # Distance distribution visualization
                                st.markdown("#### 📈 Distance to k Neighbors")

                                fig_knn_dist = go.Figure()

                                # Color neighbors by whether they match true class
                                colors = ['green' if k_nearest_labels[i] == y_test[selected_sample_idx_tab4_knn] else 'red'
                                         for i in range(k_value)]

                                fig_knn_dist.add_trace(go.Bar(
                                    x=[f"Neighbor {i+1}<br>{neighbors_data[i]['Training Sample']}" for i in range(k_value)],
                                    y=k_nearest_distances,
                                    marker_color=colors,
                                    text=[f"{d:.3f}" for d in k_nearest_distances],
                                    textposition="outside",
                                    hovertemplate="<b>%{x}</b><br>Distance: %{y:.4f}<extra></extra>"
                                ))
                                fig_knn_dist.update_layout(
                                    title=f"Distance to k-Nearest Neighbors (k={k_value}, metric={metric})",
                                    xaxis_title="Neighbor",
                                    yaxis_title="Distance",
                                    height=400,
                                    showlegend=False,
                                    plot_bgcolor='white',
                                    xaxis=dict(showgrid=False),
                                    yaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray')
                                )
                                st.plotly_chart(fig_knn_dist, use_container_width=True)

                                st.caption("🟢 Green: Neighbor has same class as true label | 🔴 Red: Different class")

                                # Debug expander
                                with st.expander("🔍 Debug: kNN Sample Analysis Details", expanded=False):
                                    st.write(f"**Selected Sample Index:** {selected_sample_idx_tab4_knn}")
                                    st.write(f"**Sample Name:** {sample_names_dict_test_knn.get(selected_sample_idx_tab4_knn, 'N/A')}")
                                    st.write(f"**True Class:** {y_test[selected_sample_idx_tab4_knn]}")
                                    st.write(f"**Predicted Class:** {y_pred_test[selected_sample_idx_tab4_knn]}")
                                    st.write(f"**k Value:** {k_value}")
                                    st.write(f"**Metric:** {metric}")
                                    st.write(f"\n**k Nearest Neighbors:**")
                                    st.write(f"- Indices: {k_nearest_indices.tolist()}")
                                    st.write(f"- Distances: {k_nearest_distances.tolist()}")
                                    st.write(f"- Labels: {k_nearest_labels.tolist()}")
                                    st.write(f"\n**Voting:**")
                                    for cls, count in sorted(vote_counts.items()):
                                        st.write(f"- {cls}: {count} votes ({100*count/k_value:.0f}%)")

                            except Exception as e:
                                st.error(f"❌ Error in kNN sample analysis: {str(e)}")
                                import traceback
                                st.error(traceback.format_exc())

                        # ═══════════════════════════════════════════════════════════════════════════
                        # LDA/QDA: MAHALANOBIS DISTANCE ANALYSIS (TEST DATA)
                        # ═══════════════════════════════════════════════════════════════════════════
                        if trained['name'] in ['LDA', 'QDA']:
                            st.divider()
                            st.markdown("### 📏 Mahalanobis Distance Analysis (Test Data)")
                            st.info(
                                "Mahalanobis distances from each test sample to class centroids. "
                                "Proper class separation shows distinct distribution patterns."
                            )

                            try:
                                # ✅ Get the correct model
                                model_to_use = trained.get('classifier_model', trained['model'])
                                # ✅ X_test_scaled è già nel formato corretto (PCA se usato, altrimenti originale)
                                X_test_for_prediction = X_test_scaled

                                # Get Mahalanobis distances for test set
                                if trained['name'] == 'LDA':
                                    _, mahal_distances_test = predict_lda(X_test_for_prediction, model_to_use)
                                else:  # QDA
                                    _, mahal_distances_test = predict_qda(X_test_for_prediction, model_to_use)

                                # ═══════════════════════════════════════════════════════════════════
                                # CREATE SAMPLE NAMES DICT (before tabs, for all 3 plots)
                                # ═══════════════════════════════════════════════════════════════════

                                saved_test_names = st.session_state.get('test_sample_names')
                                sample_names_dict_test_lda = {}

                                if saved_test_names is not None and len(saved_test_names) == len(X_test):
                                    sample_names_dict_test_lda = {i: str(saved_test_names[i]) for i in range(len(X_test))}
                                    # st.caption(f"✓ Using {len(saved_test_names)} sample names from 70-30 split")

                                elif hasattr(X_test, 'index') and not isinstance(X_test.index, pd.RangeIndex):
                                    sample_names_dict_test_lda = {i: str(X_test.index[i]) for i in range(len(X_test))}
                                    # st.caption(f"✓ Using sample names from X_test.index")

                                else:
                                    sample_names_dict_test_lda = {i: f"Sample_{i}" for i in range(len(X_test))}

                                # Convert to list for passing to plot functions
                                sample_names_list_test = [sample_names_dict_test_lda[i] for i in range(len(X_test))]

                                # Create 3-tab interface for Mahalanobis distance analysis
                                tab_closest_test, tab_category_test, tab_sample_test = st.tabs([
                                    "📊 Closest Category",
                                    "🎯 Distance to Specific Class",
                                    "🔍 Sample Analysis"
                                ])

                                class_names_list_test = [str(c) for c in classes]

                                # TAB 1: Distance to closest category
                                with tab_closest_test:
                                    st.markdown("**Distance from each test sample to its closest class**")
                                    st.caption("Shows the minimum distance to any class model. Bars colored by true class.")

                                    from classification_utils.plots import plot_mahalanobis_distance_closest_category
                                    fig_closest_test = plot_mahalanobis_distance_closest_category(
                                        mahal_distances_test,
                                        y_test,
                                        class_names=class_names_list_test,
                                        sample_names=sample_names_list_test,  # ✅ ADD sample names
                                        title=f"{trained['name']} - Distance to Closest Category (Test Data)"
                                    )
                                    st.plotly_chart(fig_closest_test, use_container_width=True)

                                # TAB 2: Distance to specific class
                                with tab_category_test:
                                    st.markdown("**Distance from all test samples to one specific class model**")
                                    st.caption("Select a class to see how all test samples relate to that class's model.")

                                    col1, col2 = st.columns([1, 3])
                                    with col1:
                                        target_class_test = st.selectbox(
                                            "Select Target Class",
                                            options=classes,
                                            key="mahal_test_target_class"
                                        )

                                    from classification_utils.plots import plot_mahalanobis_distance_category
                                    fig_category_test = plot_mahalanobis_distance_category(
                                        mahal_distances_test,
                                        y_test,
                                        target_class=target_class_test,
                                        class_names=class_names_list_test,
                                        sample_names=sample_names_list_test,  # ✅ ADD sample names
                                        title=f"{trained['name']} - Distance to Class {target_class_test} (Test Data)"
                                    )
                                    st.plotly_chart(fig_category_test, use_container_width=True)

                                # TAB 3: Sample-specific analysis
                                with tab_sample_test:
                                    st.markdown("**Distance from one test sample to all class models**")
                                    st.caption("Examine a specific test sample to understand its classification.")

                                    # ═══════════════════════════════════════════════════════════════════
                                    # USE SELECTBOX INSTEAD OF number_input
                                    # ═══════════════════════════════════════════════════════════════════

                                    col1, col2 = st.columns([1, 3])
                                    with col1:
                                        # Reorder: misclassified first, then correct
                                        misclassified_idx_test_lda = np.where(y_test != y_pred_test)[0]
                                        correct_idx_test_lda = np.where(y_test == y_pred_test)[0]
                                        ordered_indices_test_lda = np.concatenate([misclassified_idx_test_lda, correct_idx_test_lda])

                                        sample_idx_test = st.selectbox(
                                            "Select Sample",
                                            options=ordered_indices_test_lda,
                                            format_func=lambda x: f"{sample_names_dict_test_lda[x]} (T:{y_test[x]}, P:{y_pred_test[x]})",
                                            key="mahal_test_sample_idx"
                                        )

                                    from classification_utils.plots import plot_mahalanobis_distance_object
                                    fig_sample_test = plot_mahalanobis_distance_object(
                                        mahal_distances_test,
                                        sample_idx=sample_idx_test,
                                        y_true=y_test,
                                        class_names=class_names_list_test,
                                        sample_names=sample_names_list_test,  # ✅ ADD sample names
                                        title=f"{trained['name']} - Test Sample {sample_names_dict_test_lda[sample_idx_test]} Distance Analysis"
                                    )
                                    st.plotly_chart(fig_sample_test, use_container_width=True)

                            except Exception as e:
                                st.warning(f"Could not display Mahalanobis distance analysis: {str(e)}")
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
                                # ✅ Get the correct model
                                model_to_use = trained.get('classifier_model', trained['model'])
                                # ✅ X_test_scaled è già nel formato corretto (PCA se usato, altrimenti originale)
                                X_test_for_prediction = X_test_scaled

                                # Get distances array for LDA/QDA
                                if trained['name'] == 'LDA':
                                    _, distances_array = predict_lda(X_test_for_prediction, model_to_use)
                                else:  # QDA
                                    _, distances_array = predict_qda(X_test_for_prediction, model_to_use)

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

                        # ✅ FIX 2: Create sample names dictionary with improved fallbacks
                        saved_test_names = st.session_state.get('test_sample_names')
                        sample_names_dict_test = {}

                        if saved_test_names is not None and len(saved_test_names) == len(X_test):
                            # Use the saved sample names from 70-30 split (BEST)
                            sample_names_dict_test = {i: str(saved_test_names[i]) for i in range(len(X_test))}
                            st.caption(f"✓ Using {len(saved_test_names)} sample names from 70-30 split")

                        elif hasattr(X_test, 'index') and not isinstance(X_test.index, pd.RangeIndex):
                            # X_test is a DataFrame with a non-default index
                            sample_names_dict_test = {i: str(X_test.index[i]) for i in range(len(X_test))}
                            st.caption(f"✓ Using {len(X_test)} sample names from X_test.index")

                        elif hasattr(y_test, 'index') and not isinstance(y_test.index, pd.RangeIndex):
                            # y_test has a non-default index
                            try:
                                sample_names_dict_test = {i: str(y_test.index[i]) for i in range(len(y_test))}
                                st.caption(f"✓ Using {len(y_test)} sample names from y_test.index")
                            except:
                                pass

                        if not sample_names_dict_test:
                            # Fallback: use numeric indexing
                            test_start_offset = locals().get('test_start', 1)
                            sample_names_dict_test = {i: f"Test_{test_start_offset + i}" for i in range(len(X_test))}
                            st.warning(f"⚠️ Using numeric indices (no sample names found)")

                        # Sample selector for Tab4
                        selected_sample_idx_tab4 = st.selectbox(
                            "Select sample to analyze",
                            options=ordered_indices_test,
                            format_func=lambda x: f"Sample {sample_names_dict_test[x]}: True={y_test[x]}, Pred={y_pred_test[x]}",
                            key="sample_selector_tab4"
                        )

                        # ✅ FIX 3: Debug expander to verify sample names
                        with st.expander("🔍 Debug: Sample Names Status", expanded=False):
                            st.write("**Session State:**")
                            saved_names = st.session_state.get('test_sample_names')
                            st.write(f"- test_sample_names exists: {saved_names is not None}")
                            if saved_names:
                                st.write(f"- Length: {len(saved_names)}")
                                st.write(f"- First 5: {saved_names[:5]}")
                            else:
                                st.write("- test_sample_names is None")

                            st.write("\n**X_test Info:**")
                            st.write(f"- Type: {type(X_test)}")
                            if hasattr(X_test, 'index'):
                                st.write(f"- Has index: True")
                                st.write(f"- Index type: {type(X_test.index)}")
                                st.write(f"- Is RangeIndex: {isinstance(X_test.index, pd.RangeIndex)}")
                                st.write(f"- Index values (first 5): {X_test.index[:5].tolist()}")
                            else:
                                st.write(f"- Has index: False (numpy array)")

                            st.write("\n**y_test Info:**")
                            st.write(f"- Type: {type(y_test)}")
                            if hasattr(y_test, 'index'):
                                st.write(f"- Has index: True")
                                st.write(f"- Index type: {type(y_test.index)}")
                                st.write(f"- Index values (first 5): {y_test.index[:5].tolist()}")

                            st.write("\n**Sample Names Dict:**")
                            st.write(f"- Length: {len(sample_names_dict_test)}")
                            st.write(f"- First 5 entries: {dict(list(sample_names_dict_test.items())[:5])}")
                            st.write(f"- Last 5 entries: {dict(list(sample_names_dict_test.items())[-5:])}")

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

                            # ✅ Get the correct model
                            model_to_use = trained.get('classifier_model', trained['model'])
                            # ✅ X_test_scaled è già nel formato corretto (PCA se usato, altrimenti originale)
                            X_test_for_prediction = X_test_scaled

                            if trained['name'] == 'LDA':
                                # Get distances for all test samples
                                _, distances_array = predict_lda(X_test_for_prediction, model_to_use)
                                distances_to_classes_test = distances_array[selected_sample_idx_tab4, :].tolist()

                            elif trained['name'] == 'QDA':
                                # Get distances for all test samples
                                _, distances_array = predict_qda(X_test_for_prediction, model_to_use)
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
