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

def show_home():
    """Show the main homepage"""
    
    # Hero Section
    st.markdown("""
    <div style='text-align: center; padding: 2rem 0;'>
        <h1 style='font-size: 3rem; color: #1E90FF; margin-bottom: 0.5rem;'>
            ChemometricSolutions
        </h1>
        <h2 style='font-size: 1.5rem; color: #666; margin-bottom: 2rem;'>
            Interactive Demos for Chemometrics
        </h2>
        <p style='font-size: 1.1rem; max-width: 800px; margin: 0 auto; line-height: 1.6;'>
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
    col1, col2, col3, col4 = st.columns(4)
    
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
            if st.button("🚀 Launch Data Handling Demo", key="data_handling"):
                st.session_state.current_page = "Data Handling"
                st.rerun()
        else:
            st.warning("Data Handling demo not available")
    
    with col2:
        st.markdown("""
        ### 🎯 PCA Analysis
        *Principal Component Analysis suite*
        
        **Features:**
        - Complete PCA workflow
        - Interactive visualizations
        - Variance analysis
        - Scores and loadings plots
        - Model diagnostics
        """)
        
        if PCA_AVAILABLE:
            if st.button("🚀 Launch PCA Demo", key="pca_demo"):
                st.session_state.current_page = "PCA Analysis"
                st.rerun()
        else:
            st.warning("PCA demo not available")
    
    with col3:
        st.markdown("""
        ### 🧪 MLR/DOE
        *Multiple Linear Regression & Design of Experiments*
        
        **Features:**
        - Candidate points generation
        - Full factorial designs
        - Model computation with interactions
        - Response surface analysis
        - Cross-validation diagnostics
        """)
    
    with col4:
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
            if st.button("🚀 Launch Transformations", key="transformations"):
                st.session_state.current_page = "Transformations"
                st.rerun()
        else:
            st.info("🚧 Transformations coming soon")
        
        if MLR_DOE_AVAILABLE:
            if st.button("🚀 Launch MLR/DOE Demo", key="mlr_doe"):
                st.session_state.current_page = "MLR/DOE"
                st.rerun()
        else:
            st.info("🚧 MLR/DOE demo coming soon")
    
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
    
    # Call to Action
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; padding: 2rem 0; background: linear-gradient(90deg, rgba(30,144,255,0.1) 0%, rgba(30,144,255,0.05) 100%); border-radius: 10px; margin: 2rem 0;'>
        <h3 style='color: #1E90FF; margin-bottom: 1rem;'>Ready to Explore?</h3>
        <p style='font-size: 1.1rem; margin-bottom: 1.5rem;'>
            Start with Data Handling to load your datasets, then explore PCA Analysis or MLR/DOE for multivariate methods.
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
    st.sidebar.title("🧪 ChemometricSolutions")
    st.sidebar.markdown("---")
    
    # Navigation buttons
    if st.sidebar.button("🏠 Home", use_container_width=True):
        st.session_state.current_page = "Home"
        st.rerun()
    
    if DATA_HANDLING_AVAILABLE:
        if st.sidebar.button("📊 Data Handling", use_container_width=True):
            st.session_state.current_page = "Data Handling"
            st.rerun()
    else:
        st.sidebar.button("📊 Data Handling", disabled=True, use_container_width=True)
        st.sidebar.caption("Module not found")
    
    if PCA_AVAILABLE:
        if st.sidebar.button("🎯 PCA Analysis", use_container_width=True):
            st.session_state.current_page = "PCA Analysis"
            st.rerun()
    else:
        st.sidebar.button("🎯 PCA Analysis", disabled=True, use_container_width=True)
        st.sidebar.caption("Module not found")
    
    if MLR_DOE_AVAILABLE:
        if st.sidebar.button("🧪 MLR/DOE", use_container_width=True):
            st.session_state.current_page = "MLR/DOE"
            st.rerun()
    else:
        st.sidebar.button("🧪 MLR/DOE", disabled=True, use_container_width=True)
        st.sidebar.caption("Module not found")

    if TRANSFORMATIONS_AVAILABLE:
        if st.sidebar.button("🔬 Transformations", use_container_width=True):
            st.session_state.current_page = "Transformations"
            st.rerun()
    else:
        st.sidebar.button("🔬 Transformations", disabled=True, use_container_width=True)
        st.sidebar.caption("Module not found")
        
        st.sidebar.markdown("---")
    
    # Current dataset info in sidebar
    if 'current_data' in st.session_state:
        st.sidebar.markdown("### 📂 Current Dataset")
        data = st.session_state.current_data
        dataset_name = st.session_state.get('current_dataset', 'Unknown')
        
        st.sidebar.info(f"""
        **Name:** {dataset_name}  
        **Samples:** {data.shape[0]}  
        **Variables:** {data.shape[1]}  
        **Memory:** {data.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB
        """)
    else:
        st.sidebar.markdown("### 📂 Current Dataset")
        st.sidebar.info("No dataset loaded")
    
    # Links and info
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔗 Links")
    st.sidebar.markdown("""
    - [ChemometricSolutions Website](https://chemometricsolutions.com)
    - [GitHub Demos](https://github.com/FarininiChemometricSolutions/chemometricsolutions-demos)
    - [CAT Software](https://gruppochemiometria.it/index.php/software)
    """)
    
    st.sidebar.markdown("---")
    st.sidebar.caption("© 2025 ChemometricSolutions  \nDeveloped by Dr. Emanuele Farinini, PhD")
    
    # Main content area - ROUTING AGGIORNATO
    if st.session_state.current_page == "Home":
        show_home()
    elif st.session_state.current_page == "Data Handling" and DATA_HANDLING_AVAILABLE:
        data_handling.show()
    elif st.session_state.current_page == "PCA Analysis" and PCA_AVAILABLE:
        pca.show()
    elif st.session_state.current_page == "MLR/DOE" and MLR_DOE_AVAILABLE:
        mlr_doe.show()
    elif st.session_state.current_page == "Transformations" and TRANSFORMATIONS_AVAILABLE:
        transformations.show()
    else:
        st.error(f"Page '{st.session_state.current_page}' not found or module not available")
        st.session_state.current_page = "Home"
        st.rerun()

if __name__ == "__main__":
    main()
