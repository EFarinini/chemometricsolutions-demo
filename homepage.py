"""
ChemometricSolutions Interactive Demos
Homepage - Main navigation and introduction
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Import demo pages
try:
    import data_handling
    DATA_HANDLING_AVAILABLE = True
except ImportError:
    DATA_HANDLING_AVAILABLE = False

try:
    import pca
    PCA_AVAILABLE = True
except ImportError:
    PCA_AVAILABLE = False

try:
    import mlr_doe
    MLR_DOE_AVAILABLE = True
except ImportError:
    MLR_DOE_AVAILABLE = False

try:
    import transformations
    TRANSFORMATIONS_AVAILABLE = True
except ImportError:
    TRANSFORMATIONS_AVAILABLE = False

try:
    import pca_monitoring_page
    PCA_MONITORING_AVAILABLE = True
except ImportError as e:
    PCA_MONITORING_AVAILABLE = False
    # Log the error for debugging
    import sys
    print(f"⚠️ Quality Control module import failed: {e}", file=sys.stderr)
except Exception as e:
    PCA_MONITORING_AVAILABLE = False
    import sys
    print(f"⚠️ Quality Control module error (non-import): {e}", file=sys.stderr)

try:
    import bayesian_optimization_page
    BAYESIAN_OPTIMIZATION_AVAILABLE = True
except ImportError:
    BAYESIAN_OPTIMIZATION_AVAILABLE = False

try:
    import classification_page
    CLASSIFICATION_AVAILABLE = True
except ImportError:
    CLASSIFICATION_AVAILABLE = False

try:
    import calibration_page
    CALIBRATION_AVAILABLE = True
except ImportError:
    CALIBRATION_AVAILABLE = False

try:
    import univariate_page
    UNIVARIATE_AVAILABLE = True
except ImportError:
    UNIVARIATE_AVAILABLE = False

def show_home():
    """Show the main homepage"""
    
    # Hero Section with CSS cube logo
    st.markdown("""
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
    .logo-cube {
        text-align: center;
        margin-bottom: 1rem;
    }
    .logo-cube i {
        font-size: 80px;
        background: linear-gradient(45deg, #2E5293, #1E90FF, #4da3ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        filter: drop-shadow(0 4px 8px rgba(30,144,255,0.3));
    }
    </style>
    
    <div class="logo-cube"><i class="fas fa-cube"></i></div>
    
    <div style='text-align: center; margin-top: 0.2rem;'>
        <h1 style='font-size: 3.5rem; margin-bottom: 0.5rem; background: linear-gradient(45deg, #2E5293, #1E90FF, #4da3ff); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; font-weight: 700;'>
            ChemometricSolutions
        </h1>
        <h2 style='font-size: 1.5rem; color: #666; margin-bottom: 2rem; font-weight: 400;'>
            Interactive Demos for Chemometrics
        </h2>
        <p style='font-size: 1.1rem; max-width: 800px; margin: 0 auto; line-height: 1.6; color: #444;'>
            Explore our chemometric tools and methodologies through interactive demonstrations. 
            These demos showcase the power of multivariate analysis, design of experiments, 
            and process monitoring in chemical and analytical applications.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Available Demos Section
    st.markdown("## Available Interactive Demos")
    
    # Demo cards in columns
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown("""
        ### 📊 Data Handling
        *Import, export, and manage your datasets*
        
        **Features:**
        - Multi-format file support (CSV, Excel, DAT, SAM, RAW)
        - Spectroscopy data conversion
        - Data transformation tools
        - Export to multiple formats
        - Workspace management
        """)
        
        if DATA_HANDLING_AVAILABLE:
            if st.button("🚀 Launch Data Handling Demo", key="launch_data_handling"):
                st.session_state.current_page = "Data Handling"
                st.rerun()
        else:
            st.warning("Data Handling demo not available")
    
    with col2:
        st.markdown("""
        ### 🎯 PCA
        *Principal Component Analysis suite*
        
        **Features:**
        - Complete PCA workflow
        - Interactive visualizations
        - Variance analysis
        - Scores and loadings plots
        - Model diagnostics
        """)
        
        if PCA_AVAILABLE:
            if st.button("🚀 Launch PCA Demo", key="launch_pca_demo"):
                st.session_state.current_page = "PCA"
                st.rerun()
        else:
            st.warning("PCA demo not available")
    
    with col3:
        st.markdown("""
        ### 📊 Quality Control
        *PCA-based Statistical Process Monitoring*

        **Features:**
        - PCA monitoring models
        - T² and Q control charts
        - Automatic pretreatment detection
        - Contribution analysis
        - Fault detection and diagnostics
        """)

        if PCA_MONITORING_AVAILABLE:
            if st.button("🚀 Launch Quality Control", key="launch_qc"):
                st.session_state.current_page = "Quality Control"
                st.rerun()
        else:
            st.info("🚧 Quality Control coming soon")
    
    with col4:
        st.markdown("""
        ### 🧪 MLR/DOE
        *Multiple Linear Regression & Design of Experiments*

        **Features:**
        - Candidate points generation
        - Full factorial designs
        - Model computation with interactions
        - Response surface analysis
        - Lack of fit tests
        """)

        if MLR_DOE_AVAILABLE:
            if st.button("🚀 Launch MLR/DOE Demo", key="launch_mlr_doe"):
                st.session_state.current_page = "MLR/DOE"
                st.rerun()
        else:
            st.info("🚧 MLR/DOE demo coming soon")

    with col5:
        st.markdown("""
        ### 🎯 Bayesian Optimization
        *Experimental Design with BO*

        **Features:**
        - Gaussian Process predictions
        - Acquisition function optimization
        - Automated point suggestion
        - 1D/2D/nD visualization
        - Iterative refinement
        """)

        if BAYESIAN_OPTIMIZATION_AVAILABLE:
            if st.button("🚀 Launch BO Demo", key="launch_bo_demo"):
                st.session_state.current_page = "Bayesian Optimization"
                st.rerun()
        else:
            st.warning("BO demo not available")

    # Additional row for Transformations and Classification
    st.markdown("---")
    col_trans1, col_trans2, col_trans3, col_trans4, col_trans5 = st.columns(5)

    with col_trans1:
        st.markdown("""
        ### 🔬 Transformations
        *Data preprocessing for spectral analysis*

        **Features:**
        - SNV, derivatives, Savitzky-Golay
        - DoE coding, autoscaling
        - Moving averages, binning
        - Visual comparison plots
        - Auto-save to workspace
        """)

        if TRANSFORMATIONS_AVAILABLE:
            if st.button("🚀 Launch Transformations", key="launch_transformations"):
                st.session_state.current_page = "Transformations"
                st.rerun()
        else:
            st.info("🚧 Transformations coming soon")

    with col_trans2:
        st.markdown("""
        ### 🎲 Classification
        *Supervised classification methods*

        **Features:**
        - LDA/QDA discriminant analysis
        - kNN with multiple metrics
        - SIMCA class modeling
        - UNEQ with Mahalanobis
        - Coomans plots & diagnostics
        """)

        if CLASSIFICATION_AVAILABLE:
            if st.button("🚀 Launch Classification", key="launch_classification"):
                st.session_state.current_page = "Classification"
                st.rerun()
        else:
            st.info("🚧 Classification coming soon")

    with col_trans3:
        st.markdown("""
        ### 📊 Univariate Analysis
        *Comprehensive statistical analysis*

        **Features:**
        - Descriptive statistics
        - Dispersion & robust measures
        - Row profile analysis
        - Multiple visualization types
        - Export capabilities
        """)

        if UNIVARIATE_AVAILABLE:
            if st.button("🚀 Launch Univariate Analysis", key="launch_univariate"):
                st.session_state.current_page = "Univariate Analysis"
                st.rerun()
        else:
            st.info("🚧 Univariate Analysis coming soon")

    with col_trans5:
        st.markdown("""
        ### ⚗️ PLS Calibration
        *Partial Least Squares for Multivariate Calibration*

        **Features:**
        - Repeated K-fold cross-validation
        - Intelligent component selection
        - RMSECV analysis
        - Predictions vs observed
        - Test set validation
        - Model export & comparison
        """)

        if CALIBRATION_AVAILABLE:
            if st.button("🚀 Launch PLS Calibration", key="launch_calibration"):
                st.session_state.current_page = "PLS Calibration"
                st.rerun()
        else:
            st.info("🚧 PLS Calibration coming soon")

    st.markdown("---")
    
 

   # About Section
    st.markdown("## About These Demos")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        These interactive demonstrations showcase the capabilities of **ChemometricSolutions** 
        software and methodologies. Each demo is designed to:
        
        - **Demonstrate real-world applications** of chemometric methods
        - **Provide hands-on experience** with multivariate analysis tools
        - **Show best practices** for data handling and analysis
        - **Enable testing** of your own datasets
        
        The demos are built using Python and Streamlit, featuring the same analytical 
        approaches and algorithms used in our commercial software solutions.
        """)
    
    with col2:
        st.markdown("""
        **Key Features:**
        - Interactive visualizations
        - Real-time analysis
        - Export capabilities  
        - Educational content
        - Professional algorithms
        
        **Perfect for:**
        - Method evaluation
        - Training purposes
        - Proof of concept
        - Data exploration
        """)
    
    # Professional Services Section - NUOVO
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; padding: 2rem 0; background: linear-gradient(90deg, rgba(30,144,255,0.1) 0%, rgba(30,144,255,0.05) 100%); border-radius: 10px; margin: 2rem 0; border: 1px solid rgba(30,144,255,0.2);'>
        <h3 style='color: #1E90FF; margin-bottom: 1rem;'>🚀 Need Advanced Features?</h3>
        <p style='font-size: 1.1rem; margin-bottom: 1rem; line-height: 1.6;'>
            These demos showcase core capabilities. For advanced chemometric solutions, 
            custom algorithms, enterprise features, and professional consulting:
        </p>
        <p style='font-size: 1.2rem; font-weight: 600; margin-bottom: 0.5rem;'>
            📞 <a href="https://chemometricsolutions.com" target="_blank" style="color: #1E90FF; text-decoration: none;">Contact us at chemometricsolutions.com</a>
        </p>
        <p style='font-size: 0.9rem; color: #666; margin-top: 1rem;'>
            ✨ Custom solutions • Enterprise support • Method validation • Training programs
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Call to Action
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; padding: 2rem 0; background: linear-gradient(90deg, rgba(30,144,255,0.1) 0%, rgba(30,144,255,0.05) 100%); border-radius: 10px; margin: 2rem 0;'>
        <h3 style='color: #1E90FF; margin-bottom: 1rem;'>Ready to Explore?</h3>
        <p style='font-size: 1.1rem; margin-bottom: 1.5rem;'>
            Start with Data Handling to load your datasets, then explore PCA or MLR/DOE for multivariate methods.
        </p>
        <p style='font-size: 0.9rem; color: #666;'>
            These demos represent a subset of our full chemometric capabilities. 
            Contact us for custom solutions and advanced methodologies.
        </p>
    </div>
    """, unsafe_allow_html=True)

def main():
    """Main application function"""
    
    # Configuration
    st.set_page_config(
        page_title="ChemometricSolutions - Interactive Demos",
        page_icon="🧪",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "Home"
    
    # Sidebar navigation
    # Sidebar with CSS cube logo
    st.sidebar.markdown("""
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
    .sidebar-logo-cube {
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sidebar-logo-cube i {
        font-size: 40px;
        background: linear-gradient(45deg, #2E5293, #1E90FF, #4da3ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        filter: drop-shadow(0 2px 4px rgba(30,144,255,0.3));
    }
    </style>

    <div class="sidebar-logo-cube"><i class="fas fa-cube"></i></div>

    <div style='text-align: center; margin: 0.5rem 0 2rem 0;'>
        <div style='font-size: 1.1rem; font-weight: 700; color: #2E5293; line-height: 1.2;'>
            Chemometric<br>Solutions
        </div>
        <div style='font-size: 0.8rem; color: #666; font-style: italic; margin-top: 0.2rem;'>
            Interactive Demos
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.sidebar.markdown("---")
    
    # Navigation buttons
    if st.sidebar.button("🏠 Home", use_container_width=True, key="nav_home"):
        st.session_state.current_page = "Home"
        st.rerun()
    
    if DATA_HANDLING_AVAILABLE:
        if st.sidebar.button("📊 Data Handling", use_container_width=True, key="nav_data_handling"):
            st.session_state.current_page = "Data Handling"
            st.rerun()
    else:
        st.sidebar.button("📊 Data Handling", disabled=True, use_container_width=True, key="nav_data_handling_disabled")
        st.sidebar.caption("Module not found")
    
    if PCA_AVAILABLE:
        if st.sidebar.button("🎯 PCA", use_container_width=True, key="nav_pca"):
            st.session_state.current_page = "PCA"
            st.rerun()
    else:
        st.sidebar.button("🎯 PCA", disabled=True, use_container_width=True, key="nav_pca_disabled")
        st.sidebar.caption("Module not found")

    if PCA_MONITORING_AVAILABLE:
        if st.sidebar.button("📊 Quality Control", use_container_width=True, key="nav_qc"):
            st.session_state.current_page = "Quality Control"
            st.rerun()
    else:
        st.sidebar.button("📊 Quality Control", disabled=True, use_container_width=True, key="nav_qc_disabled")
        st.sidebar.caption("Module not found")

    if MLR_DOE_AVAILABLE:
        if st.sidebar.button("🧪 MLR/DOE", use_container_width=True, key="nav_mlr_doe"):
            st.session_state.current_page = "MLR/DOE"
            st.rerun()
    else:
        st.sidebar.button("🧪 MLR/DOE", disabled=True, use_container_width=True, key="nav_mlr_doe_disabled")
        st.sidebar.caption("Module not found")

    if TRANSFORMATIONS_AVAILABLE:
        if st.sidebar.button("🔬 Transformations", use_container_width=True, key="nav_transformations"):
            st.session_state.current_page = "Transformations"
            st.rerun()
    else:
        st.sidebar.button("🔬 Transformations", disabled=True, use_container_width=True, key="nav_transformations_disabled")
        st.sidebar.caption("Module not found")

    if BAYESIAN_OPTIMIZATION_AVAILABLE:
        if st.sidebar.button("🎯 Bayesian Optimization", use_container_width=True, key="nav_bayesian_opt"):
            st.session_state.current_page = "Bayesian Optimization"
            st.rerun()
    else:
        st.sidebar.button("🎯 Bayesian Optimization", disabled=True, use_container_width=True, key="nav_bayesian_opt_disabled")
        st.sidebar.caption("Module not found")

    if CLASSIFICATION_AVAILABLE:
        if st.sidebar.button("🎲 Classification", use_container_width=True, key="nav_classification"):
            st.session_state.current_page = "Classification"
            st.rerun()
    else:
        st.sidebar.button("🎲 Classification", disabled=True, use_container_width=True, key="nav_classification_disabled")
        st.sidebar.caption("Module not found")

    if CALIBRATION_AVAILABLE:
        if st.sidebar.button("⚗️ PLS Calibration", use_container_width=True, key="nav_calibration"):
            st.session_state.current_page = "PLS Calibration"
            st.rerun()
    else:
        st.sidebar.button("⚗️ PLS Calibration", disabled=True, use_container_width=True, key="nav_calibration_disabled")
        st.sidebar.caption("Module not found")

    if UNIVARIATE_AVAILABLE:
        if st.sidebar.button("📊 Univariate Analysis", use_container_width=True, key="nav_univariate"):
            st.session_state.current_page = "Univariate Analysis"
            st.rerun()
    else:
        st.sidebar.button("📊 Univariate Analysis", disabled=True, use_container_width=True, key="nav_univariate_disabled")
        st.sidebar.caption("Module not found")

    st.sidebar.markdown("---")
    
    # Current dataset info in sidebar with selector
    with st.sidebar:
        st.markdown("### 📂 Current Dataset")

        # Import workspace utility function
        from workspace_utils import get_workspace_datasets, activate_dataset_in_workspace

        # Get available datasets
        available_datasets = get_workspace_datasets()

        if available_datasets:
            # Create list of dataset names
            dataset_names = list(available_datasets.keys())

            # Get current dataset name
            current_dataset_name = st.session_state.get('dataset_name',
                                                         st.session_state.get('current_dataset', None))

            # Set default to current dataset if available, otherwise first one
            default_index = 0
            if current_dataset_name and current_dataset_name in dataset_names:
                default_index = dataset_names.index(current_dataset_name)

            # Dataset selector dropdown
            selected_dataset = st.selectbox(
                "🔄 Switch Dataset",
                dataset_names,
                index=default_index,
                key="sidebar_dataset_selector"
            )

            # Load selected dataset if changed
            if selected_dataset:
                selected_data = available_datasets[selected_dataset]
                activate_dataset_in_workspace(selected_dataset, selected_data)

                # Update display info
                st.markdown(f"**Name:** `{selected_dataset}`")
                st.markdown(f"**Samples:** {len(selected_data)}")
                st.markdown(f"**Variables:** {len(selected_data.columns)}")
                st.markdown(f"**Memory:** {selected_data.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        else:
            st.info("📊 Load a dataset in Data Handling")
    
# Links and info
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔗 Links")
    st.sidebar.markdown("""
    - [ChemometricSolutions Website](https://chemometricsolutions.com)
    - [GitHub Demos](https://github.com/FarininiChemometricSolutions/chemometricsolutions-demos)
    - [CAT Software](https://gruppochemiometria.it/index.php/software)
    """)
    
    # Professional Services Sidebar - NUOVO
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 💼 Professional Services")
    st.sidebar.info("""
    **Need custom solutions?**
    
    🔬 Advanced algorithms  
    🏢 Enterprise features  
    📊 Method validation  
    🎓 Training & consulting
    
    **[Contact us →](https://chemometricsolutions.com)**
    """)
    
    st.sidebar.markdown("---")
    st.sidebar.caption("© 2025 ChemometricSolutions  \nDeveloped by Dr. Emanuele Farinini, PhD")
    
    # Main content area - ROUTING AGGIORNATO
    if st.session_state.current_page == "Home":
        show_home()
    elif st.session_state.current_page == "Data Handling" and DATA_HANDLING_AVAILABLE:
        data_handling.show()
    elif st.session_state.current_page == "PCA" and PCA_AVAILABLE:
        pca.show()
    elif st.session_state.current_page == "Quality Control" and PCA_MONITORING_AVAILABLE:
        pca_monitoring_page.show()
    elif st.session_state.current_page == "MLR/DOE" and MLR_DOE_AVAILABLE:
        mlr_doe.show()
    elif st.session_state.current_page == "Transformations" and TRANSFORMATIONS_AVAILABLE:
        transformations.show()
    elif st.session_state.current_page == "Bayesian Optimization" and BAYESIAN_OPTIMIZATION_AVAILABLE:
        bayesian_optimization_page.show_bayesian_optimization_page()
    elif st.session_state.current_page == "Classification" and CLASSIFICATION_AVAILABLE:
        classification_page.show()
    elif st.session_state.current_page == "PLS Calibration" and CALIBRATION_AVAILABLE:
        calibration_page.show()
    elif st.session_state.current_page == "Univariate Analysis" and UNIVARIATE_AVAILABLE:
        univariate_page.show()
    else:
        st.error(f"Page '{st.session_state.current_page}' not found or module not available")
        st.session_state.current_page = "Home"
        st.rerun()

if __name__ == "__main__":
    main()