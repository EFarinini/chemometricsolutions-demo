"""
Streamlit Session State Keys - Canonical Definitions
===================================================

This module defines the canonical session state keys used across all pages
in the ChemometricSolutions application. Using constants ensures consistency
and prevents bugs from typos or key mismatches.

Usage:
    from session_state_keys import SESSION_CURRENT_DATA

    # Read data
    if SESSION_CURRENT_DATA in st.session_state:
        df = st.session_state[SESSION_CURRENT_DATA]

    # Write data
    st.session_state[SESSION_CURRENT_DATA] = new_df

Best Practices:
    1. Always import keys from this module
    2. Never use string literals for session state keys
    3. Add new keys here when creating new features
    4. Document the purpose and type of each key
"""

# ============================================================================
# DATA MANAGEMENT
# ============================================================================

SESSION_CURRENT_DATA = 'current_data'
"""
Main active dataset (pd.DataFrame)
This is the primary dataset that all analysis pages should read from.
Updated by: Data Handling, Transformations, Variable Selection, etc.
"""

SESSION_CURRENT_DATASET = 'current_dataset'
"""
Name of the current active dataset (str)
Used for display purposes and workspace management.
"""

SESSION_TRANSFORMATION_HISTORY = 'transformation_history'
"""
Dictionary of all saved datasets (dict[str, pd.DataFrame])
Format: {'dataset_name': dataframe}
Used by: workspace_utils.py
"""

SESSION_SPLIT_DATASETS = 'split_datasets'
"""
Dictionary of split datasets (dict[str, dict])
Format: {'split_name': {'train': df, 'test': df}}
Used by: Data splitting functionality
"""

# ============================================================================
# PCA RESULTS
# ============================================================================

SESSION_PCA_MODEL = 'pca_model'
"""
Fitted PCA model object
Contains: scores, loadings, explained variance, etc.
"""

SESSION_PCA_RESULTS = 'pca_results'
"""
PCA analysis results (dict)
Contains: plots, diagnostics, statistics
"""

SESSION_PCA_MONITORING_MODEL = 'pca_monitoring_model'
"""
PCA monitoring model for quality control (dict)
Contains: control limits, statistics, pretreatment info
"""

# ============================================================================
# MLR/DOE RESULTS
# ============================================================================

SESSION_MLR_MODEL = 'mlr_model'
"""
Multiple linear regression model (dict)
Contains: coefficients, statistics, diagnostics
"""

SESSION_DOE_DESIGN = 'doe_design'
"""
Design of experiments matrix (pd.DataFrame)
"""

SESSION_MLR_RESULTS = 'mlr_results'
"""
MLR analysis results (dict)
Contains: fitted values, residuals, response surfaces
"""

# ============================================================================
# CLASSIFICATION RESULTS
# ============================================================================

SESSION_CLASSIFICATION_MODEL = 'classification_model'
"""
Classification model results (dict)
Contains: model type (LDA/QDA/kNN/SIMCA), parameters, predictions
"""

SESSION_CLASSIFICATION_RESULTS = 'classification_results'
"""
Classification analysis results (dict)
Contains: confusion matrix, accuracy, plots
"""

# ============================================================================
# CALIBRATION RESULTS
# ============================================================================

SESSION_CALIBRATION_MODEL = 'calibration_model'
"""
PLS calibration model (dict)
Contains: components, loadings, coefficients, validation results
"""

SESSION_CALIBRATION_RESULTS = 'calibration_results'
"""
Calibration analysis results (dict)
Contains: RMSECV, predictions, validation plots
"""

# ============================================================================
# GA VARIABLE SELECTION RESULTS
# ============================================================================

SESSION_GA_RESULTS = 'ga_results'
"""
Genetic algorithm results (dict)
Contains: selected_variables, selection_frequency, best_fitness, etc.
"""

SESSION_GA_SELECTED_COLS = 'ga_selected_cols'
"""
List of selected variable names (list[str])
"""

SESSION_GA_X_COLS = 'ga_X_cols'
"""
List of all variable names used in GA (list[str])
"""

SESSION_GA_CONFIG = 'ga_config'
"""
GA configuration parameters (dict)
"""

SESSION_GA_PROBLEM_TYPE = 'ga_problem_type'
"""
GA problem type: 'pls', 'lda', 'fda', 'mahalanobis', 'distance' (str)
"""

SESSION_GA_TARGET_VAR = 'ga_target_var'
"""
Target variable name for supervised GA (str)
"""

# ============================================================================
# PAGE NAVIGATION
# ============================================================================

SESSION_CURRENT_PAGE = 'current_page'
"""
Currently active page (str)
Used by: homepage.py navigation system
"""

SESSION_PAGE = 'page'
"""
Alternative page key (str)
Legacy key, prefer SESSION_CURRENT_PAGE
"""

# ============================================================================
# USER SETTINGS
# ============================================================================

SESSION_THEME = 'theme'
"""
UI theme preference: 'light' or 'dark' (str)
"""

SESSION_PLOT_TEMPLATE = 'plot_template'
"""
Plotly template preference (str)
"""

# ============================================================================
# AUTHENTICATION (if implemented)
# ============================================================================

SESSION_USER_ID = 'user_id'
"""
Authenticated user ID (str)
"""

SESSION_USER_ROLE = 'user_role'
"""
User role/permissions (str)
"""

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_data(session_state) -> 'pd.DataFrame | None':
    """
    Safely retrieve current data from session state.

    Parameters
    ----------
    session_state : st.session_state
        Streamlit session state object

    Returns
    -------
    df : pd.DataFrame or None
        Current dataset if available, None otherwise
    """
    return session_state.get(SESSION_CURRENT_DATA, None)


def set_data(session_state, df: 'pd.DataFrame', name: str = None):
    """
    Safely set current data in session state.

    Parameters
    ----------
    session_state : st.session_state
        Streamlit session state object
    df : pd.DataFrame
        Dataset to set as current
    name : str, optional
        Dataset name for display
    """
    session_state[SESSION_CURRENT_DATA] = df
    if name:
        session_state[SESSION_CURRENT_DATASET] = name


def clear_results(session_state):
    """
    Clear all analysis results from session state.

    Useful when loading new data to ensure old results don't persist.

    Parameters
    ----------
    session_state : st.session_state
        Streamlit session state object
    """
    result_keys = [
        SESSION_PCA_MODEL,
        SESSION_PCA_RESULTS,
        SESSION_MLR_MODEL,
        SESSION_MLR_RESULTS,
        SESSION_CLASSIFICATION_MODEL,
        SESSION_CLASSIFICATION_RESULTS,
        SESSION_CALIBRATION_MODEL,
        SESSION_CALIBRATION_RESULTS,
        SESSION_GA_RESULTS,
        SESSION_GA_SELECTED_COLS,
    ]

    for key in result_keys:
        if key in session_state:
            del session_state[key]


def get_all_session_keys() -> list:
    """
    Get list of all defined session state keys.

    Returns
    -------
    keys : list[str]
        List of all canonical session state keys
    """
    import inspect

    keys = []
    for name, value in globals().items():
        if name.startswith('SESSION_') and isinstance(value, str):
            keys.append(value)

    return sorted(keys)


# ============================================================================
# VALIDATION
# ============================================================================

def validate_session_state(session_state) -> dict:
    """
    Validate session state structure and report issues.

    Parameters
    ----------
    session_state : st.session_state
        Streamlit session state object

    Returns
    -------
    report : dict
        Validation report with keys:
        - 'valid': bool
        - 'issues': list[str]
        - 'warnings': list[str]
    """
    issues = []
    warnings = []

    # Check for required keys
    if SESSION_CURRENT_DATA not in session_state:
        warnings.append("No data loaded (current_data missing)")

    # Check for deprecated keys
    deprecated_keys = {
        'current_df': 'Use SESSION_CURRENT_DATA instead',
        'df': 'Use SESSION_CURRENT_DATA instead',
        'data': 'Use SESSION_CURRENT_DATA instead'
    }

    for old_key, message in deprecated_keys.items():
        if old_key in session_state:
            issues.append(f"Deprecated key '{old_key}' found. {message}")

    # Check data type
    if SESSION_CURRENT_DATA in session_state:
        import pandas as pd
        data = session_state[SESSION_CURRENT_DATA]
        if not isinstance(data, pd.DataFrame):
            issues.append(f"current_data should be DataFrame, got {type(data)}")

    return {
        'valid': len(issues) == 0,
        'issues': issues,
        'warnings': warnings
    }


# ============================================================================
# DOCUMENTATION
# ============================================================================

__doc__ += """

Session State Structure
=======================

The following keys are used across the application:

Data Management:
  - current_data: Main active dataset (pd.DataFrame)
  - current_dataset: Name of active dataset (str)
  - transformation_history: All saved datasets (dict)
  - split_datasets: Train/test splits (dict)

Analysis Results:
  - pca_model: PCA model and results (dict)
  - mlr_model: MLR model and results (dict)
  - classification_model: Classification results (dict)
  - calibration_model: PLS calibration results (dict)
  - ga_results: GA variable selection results (dict)

Navigation:
  - current_page: Active page name (str)

User Settings:
  - theme: UI theme preference (str)
  - plot_template: Plotly template (str)

Usage Guidelines:
1. Always use constants from this module
2. Never hardcode session state keys as strings
3. Use helper functions (get_data, set_data) for common operations
4. Run validate_session_state() during development to catch issues

Example:
    from session_state_keys import SESSION_CURRENT_DATA, get_data, set_data

    # Read data
    df = get_data(st.session_state)

    # Write data
    set_data(st.session_state, new_df, "My Dataset")
"""
