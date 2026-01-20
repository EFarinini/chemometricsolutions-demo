"""
ChemometricSolutions - DEMO VERSION
Homepage - Main navigation and introduction
Workshop Como 2026
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Import demo pages - ONLY 7 MODULES
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
    import pca_monitoring_page
    PCA_MONITORING_AVAILABLE = True
except ImportError as e:
    PCA_MONITORING_AVAILABLE = False
    import sys
    print(f"⚠️ Quality Control module import failed: {e}", file=sys.stderr)

try:
    import mlr_doe
    MLR_DOE_AVAILABLE = True
except ImportError:
    MLR_DOE_AVAILABLE = False

try:
    import univariate_page
    UNIVARIATE_AVAILABLE = True
except ImportError:
    UNIVARIATE_AVAILABLE = False

try:
    import bivariate_page
    BIVARIATE_AVAILABLE = True
except ImportError:
    BIVARIATE_AVAILABLE = False

try:
    import transformations
    TRANSFORMATIONS_AVAILABLE = True
except ImportError:
    TRANSFORMATIONS_AVAILABLE = False

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
            DEMO VERSION - Workshop Como 2026
        </h2>
        <p style='font-size: 1.1rem; max-width: 800px; margin: 0 auto; line-height: 1.6; color: #444;'>
            Explore 7 core chemometric tools through interactive demonstrations.
            These modules showcase the power of multivariate analysis, design of experiments,
            and process monitoring in chemical and analytical applications.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # DEMO VERSION INFO BANNER
    st.info("""
    ### 🎓 DEMO VERSION - Workshop Como 2026

    **Included Modules:**
    ✅ Data Handling & Import
    ✅ PCA Analysis
    ✅ Quality Control (PCA Monitoring)
    ✅ MLR & DoE (Single Response)
    ✅ Univariate Analysis
    ✅ Bivariate Analysis
    ✅ Preprocessing & Transformations

    📧 For full version with 12+ modules: chemometricsolutions@gmail.com
    """)

    st.markdown("---")

    # Hero call-to-action
    st.markdown("""
    ## 📥 Import your data

    Start by loading your datasets in **Data Handling**, then boost your analysis with our chemometric tools.
    """)

    if DATA_HANDLING_AVAILABLE:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("📊 Go to Data Handling", use_container_width=True, key="cta_data_handling"):
                st.session_state.current_page = "Data Handling"
                st.rerun()

    st.markdown("---")
    st.markdown("## 🚀 All Modules")
    st.markdown("*✅ = Included in demo | 🔒 = Full version only*")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Core Analysis:**")

        # PCA - AVAILABLE
        if PCA_AVAILABLE and st.button("✅ 📈 PCA - Principal Component Analysis", use_container_width=True, key="btn_pca"):
            st.session_state.current_page = "PCA"
            st.rerun()

        # Quality Control - AVAILABLE
        if PCA_MONITORING_AVAILABLE and st.button("✅ 📊 Quality Control - PCA Monitoring", use_container_width=True, key="btn_qc"):
            st.session_state.current_page = "Quality Control"
            st.rerun()

        # MLR/DoE - AVAILABLE
        if MLR_DOE_AVAILABLE and st.button("✅ 🧪 MLR & DoE - Single Response", use_container_width=True, key="btn_mlr"):
            st.session_state.current_page = "MLR/DOE"
            st.rerun()

        # Multi-Response DoE - LOCKED
        if st.button("🔒 🎯 Multi-Response DoE - Pareto", use_container_width=True, key="btn_multi_doe"):
            st.warning("""
            ### 🔒 Multi-Response DoE - Full Version Only

            Multi-criteria optimization with Pareto analysis for multiple responses simultaneously.

            **Contact:** chemometricsolutions@gmail.com
            """)

        # Classification - LOCKED
        if st.button("🔒 🎲 Classification - Pattern Recognition", use_container_width=True, key="btn_class"):
            st.warning("""
            ### 🔒 Classification - Full Version Only

            Supervised classification with PLS-DA, LDA, QDA, and other algorithms.

            **Contact:** chemometricsolutions@gmail.com
            """)

    with col2:
        st.markdown("**Statistical & Advanced:**")

        # Univariate - AVAILABLE
        if UNIVARIATE_AVAILABLE and st.button("✅ 📉 Univariate - Statistical Analysis", use_container_width=True, key="btn_uni"):
            st.session_state.current_page = "Univariate Analysis"
            st.rerun()

        # Bivariate - AVAILABLE
        if BIVARIATE_AVAILABLE and st.button("✅ 🔗 Bivariate - Correlation Analysis", use_container_width=True, key="btn_bi"):
            st.session_state.current_page = "Bivariate Analysis"
            st.rerun()

        # Preprocessing - AVAILABLE
        if TRANSFORMATIONS_AVAILABLE and st.button("✅ ⚙️ Preprocessing - Data Transformations", use_container_width=True, key="btn_trans"):
            st.session_state.current_page = "Transformations"
            st.rerun()

        # PLS Calibration - LOCKED
        if st.button("🔒 🔬 PLS Calibration - Quantitative Analysis", use_container_width=True, key="btn_cal"):
            st.warning("""
            ### 🔒 PLS Calibration - Full Version Only

            Partial Least Squares regression for quantitative calibration models.

            **Contact:** chemometricsolutions@gmail.com
            """)

        # GA Variable Selection - LOCKED
        if st.button("🔒 🧬 GA Variable Selection - Optimization", use_container_width=True, key="btn_ga"):
            st.warning("""
            ### 🔒 GA Variable Selection - Full Version Only

            Genetic Algorithm for optimal variable selection in multivariate models.

            **Contact:** chemometricsolutions@gmail.com
            """)

        # Mixture Design - LOCKED
        if st.button("🔒 🧪 Mixture Design - Simplex DoE", use_container_width=True, key="btn_mix"):
            st.warning("""
            ### 🔒 Mixture Design - Full Version Only

            Simplex-based experimental design for mixture formulations.

            **Contact:** chemometricsolutions@gmail.com
            """)

    st.markdown("---")

    # About Section
    st.markdown("## About This Demo")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        This demo version showcases the core capabilities of **ChemometricSolutions**
        software and methodologies. Each module is designed to:

        - **Demonstrate real-world applications** of chemometric methods
        - **Provide hands-on experience** with multivariate analysis tools
        - **Show best practices** for data handling and analysis
        - **Enable testing** with real sample datasets

        The demo includes 7 fully functional modules built using Python and Streamlit,
        featuring the same analytical approaches used in our commercial software.
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

    # Professional Services Section
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; padding: 2rem 0; background: linear-gradient(90deg, rgba(30,144,255,0.1) 0%, rgba(30,144,255,0.05) 100%); border-radius: 10px; margin: 2rem 0; border: 1px solid rgba(30,144,255,0.2);'>
        <h3 style='color: #1E90FF; margin-bottom: 1rem;'>🚀 Need the Full Version?</h3>
        <p style='font-size: 1.1rem; margin-bottom: 1rem; line-height: 1.6;'>
            The full version includes 12+ advanced modules with multi-response optimization,
            classification, calibration (PLS), genetic algorithms, and mixture designs.
        </p>
        <p style='font-size: 1.2rem; font-weight: 600; margin-bottom: 0.5rem;'>
            📧 Contact: <a href="mailto:chemometricsolutions@gmail.com" style="color: #1E90FF; text-decoration: none;">chemometricsolutions@gmail.com</a>
        </p>
        <p style='font-size: 1.0rem; margin-bottom: 0.5rem;'>
            🌐 Website: <a href="https://chemometricsolutions.com" target="_blank" style="color: #1E90FF; text-decoration: none;">chemometricsolutions.com</a>
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
            This demo represents the core capabilities of our chemometric platform.
            Contact us for the full version with advanced methodologies.
        </p>
    </div>
    """, unsafe_allow_html=True)

def main_content():
    """Main application content (without set_page_config)"""

    # Initialize session state
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "Home"

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

    <div style='text-align: center; margin: 0.5rem 0 1rem 0;'>
        <div style='font-size: 1.1rem; font-weight: 700; color: #2E5293; line-height: 1.2;'>
            Chemometric<br>Solutions
        </div>
        <div style='font-size: 0.7rem; color: #666; font-style: italic; margin-top: 0.2rem;'>
            DEMO - Como 2026
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.sidebar.markdown("---")

    # Navigation buttons - ALL MODULES (available + locked)
    if st.sidebar.button("🏠 Home", use_container_width=True, key="nav_home"):
        st.session_state.current_page = "Home"
        st.rerun()

    # AVAILABLE MODULES
    if DATA_HANDLING_AVAILABLE:
        if st.sidebar.button("📊 Data Handling", use_container_width=True, key="nav_data_handling"):
            st.session_state.current_page = "Data Handling"
            st.rerun()

    if PCA_AVAILABLE:
        if st.sidebar.button("📈 PCA", use_container_width=True, key="nav_pca"):
            st.session_state.current_page = "PCA"
            st.rerun()

    if PCA_MONITORING_AVAILABLE:
        if st.sidebar.button("📊 Quality Control", use_container_width=True, key="nav_qc"):
            st.session_state.current_page = "Quality Control"
            st.rerun()

    if MLR_DOE_AVAILABLE:
        if st.sidebar.button("🧪 MLR/DOE", use_container_width=True, key="nav_mlr_doe"):
            st.session_state.current_page = "MLR/DOE"
            st.rerun()

    # LOCKED MODULE - Multi-Response DoE
    if st.sidebar.button("🔒 🎯 Multi-DOE", use_container_width=True, key="nav_multi_doe_locked"):
        st.sidebar.warning("🔒 Full Version Only")

    if UNIVARIATE_AVAILABLE:
        if st.sidebar.button("📉 Univariate", use_container_width=True, key="nav_univariate"):
            st.session_state.current_page = "Univariate Analysis"
            st.rerun()

    if BIVARIATE_AVAILABLE:
        if st.sidebar.button("🔗 Bivariate", use_container_width=True, key="nav_bivariate"):
            st.session_state.current_page = "Bivariate Analysis"
            st.rerun()

    if TRANSFORMATIONS_AVAILABLE:
        if st.sidebar.button("⚙️ Preprocessing", use_container_width=True, key="nav_transformations"):
            st.session_state.current_page = "Transformations"
            st.rerun()

    # LOCKED MODULES
    if st.sidebar.button("🔒 🎲 Classification", use_container_width=True, key="nav_class_locked"):
        st.sidebar.warning("🔒 Full Version Only")

    if st.sidebar.button("🔒 🔬 PLS Calibration", use_container_width=True, key="nav_cal_locked"):
        st.sidebar.warning("🔒 Full Version Only")

    if st.sidebar.button("🔒 🧬 GA Selection", use_container_width=True, key="nav_ga_locked"):
        st.sidebar.warning("🔒 Full Version Only")

    if st.sidebar.button("🔒 🧪 Mixture Design", use_container_width=True, key="nav_mix_locked"):
        st.sidebar.warning("🔒 Full Version Only")

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
    - [ChemometricSolutions](https://chemometricsolutions.com)
    - [GitHub Demo](https://github.com/EFarinini/chemometricsolutions-demo)
    - [CAT Software](https://gruppochemiometria.it/index.php/software)
    """)

    # Professional Services Sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 💼 Full Version")
    st.sidebar.info("""
    **Want more modules?**

    🎯 Multi-response DoE
    🧬 GA Variable Selection
    🎲 Classification
    🔬 PLS Calibration
    🧪 Mixture Design

    📧 **[Contact us →](mailto:chemometricsolutions@gmail.com)**
    """)

    st.sidebar.markdown("---")
    st.sidebar.caption("© 2025 ChemometricSolutions  \nDeveloped by Dr. Emanuele Farinini, PhD")

    # Main content area - ROUTING FOR 7 MODULES ONLY
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
    elif st.session_state.current_page == "Univariate Analysis" and UNIVARIATE_AVAILABLE:
        univariate_page.show()
    elif st.session_state.current_page == "Bivariate Analysis" and BIVARIATE_AVAILABLE:
        bivariate_page.show()
    elif st.session_state.current_page == "Transformations" and TRANSFORMATIONS_AVAILABLE:
        transformations.show()
    else:
        st.error(f"Page '{st.session_state.current_page}' not found or module not available")
        st.session_state.current_page = "Home"
        st.rerun()

def main():
    """Main application function with page config"""
    st.set_page_config(
        page_title="ChemometricSolutions - DEMO VERSION",
        page_icon="🧪",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    main_content()

if __name__ == "__main__":
    main()
