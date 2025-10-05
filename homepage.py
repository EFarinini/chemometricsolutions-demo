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
            if st.button("🚀 Launch Data Handling Demo", key="launch_data_handling"):
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
            if st.button("🚀 Launch PCA Demo", key="launch_pca_demo"):
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
            if st.button("🚀 Launch Transformations", key="launch_transformations"):
                st.session_state.current_page = "Transformations"
                st.rerun()
        else:
            st.info("🚧 Transformations coming soon")
        
        if MLR_DOE_AVAILABLE:
            if st.button("🚀 Launch MLR/DOE Demo", key="launch_mlr_doe"):
                st.session_state.current_page = "MLR/DOE"
                st.rerun()
        else:
            st.info("🚧 MLR/DOE demo coming soon")
    
    st.markdown("---")
    
    # Sample Data Preview
    st.markdown("## Real NIR Data Preview")
    st.markdown("*Actual near-infrared spectroscopy data from pharmaceutical analysis*")
    
    # Real NIR data from the uploaded dataset
    wavelengths = [908.100, 914.294, 920.489, 926.683, 932.877, 939.072, 945.266, 951.460, 957.655, 963.849, 970.044, 976.238, 982.432, 988.627, 994.821, 1001.015, 1007.210, 1013.404, 1019.598, 1025.793, 1031.987, 1038.181, 1044.376, 1050.570, 1056.764, 1062.959, 1069.153, 1075.348, 1081.542, 1087.736, 1093.931, 1100.125, 1106.319, 1112.514, 1118.708, 1124.902, 1131.097, 1137.291, 1143.485, 1149.680, 1155.874, 1162.069, 1168.263, 1174.457, 1180.652, 1186.846, 1193.040, 1199.235, 1205.429, 1211.623, 1217.818, 1224.012, 1230.206, 1236.401, 1242.595, 1248.789, 1254.984, 1261.178, 1267.373, 1273.567, 1279.761, 1285.956, 1292.150, 1298.344, 1304.539, 1310.733, 1316.927, 1323.122, 1329.316, 1335.510, 1341.705, 1347.899, 1354.094, 1360.288, 1366.482, 1372.677, 1378.871, 1385.065, 1391.260, 1397.454, 1403.648, 1409.843, 1416.037, 1422.231, 1428.426, 1434.620, 1440.814, 1447.009, 1453.203, 1459.398, 1465.592, 1471.786, 1477.981, 1484.175, 1490.369, 1496.564, 1502.758, 1508.952, 1515.147, 1521.341, 1527.535, 1533.730, 1539.924, 1546.119, 1552.313, 1558.507, 1564.702, 1570.896, 1577.090, 1583.285, 1589.479, 1595.673, 1601.868, 1608.062, 1614.256, 1620.451, 1626.645, 1632.839, 1639.034, 1645.228, 1651.423, 1657.617, 1663.811, 1670.006, 1676.200]
    
    # Real spectral data for 5 samples with their KF responses
    spectra_data = {
        'Sample_1 (KF=11.63)': [0.755582, 0.726677, 0.699489, 0.668905, 0.640526, 0.621512, 0.612673, 0.611488, 0.616440, 0.623817, 0.632762, 0.634355, 0.632978, 0.631780, 0.630796, 0.628992, 0.626475, 0.622501, 0.616291, 0.609764, 0.603624, 0.598924, 0.595104, 0.591863, 0.588938, 0.586289, 0.584338, 0.583764, 0.585504, 0.592092, 0.601006, 0.607918, 0.613993, 0.624793, 0.643190, 0.669424, 0.697947, 0.728712, 0.760929, 0.791193, 0.817010, 0.837880, 0.854457, 0.870453, 0.891361, 0.902615, 0.905634, 0.911890, 0.921855, 0.922416, 0.907617, 0.872066, 0.835383, 0.807431, 0.787473, 0.773296, 0.762608, 0.753948, 0.747589, 0.743996, 0.743006, 0.743591, 0.745390, 0.748379, 0.752548, 0.757811, 0.765667, 0.777507, 0.791823, 0.809481, 0.831280, 0.855913, 0.878700, 0.897341, 0.915300, 0.936483, 0.963978, 1.001847, 1.041208, 1.066049, 1.088863, 1.120319, 1.153103, 1.173524, 1.179405, 1.177100, 1.169496, 1.159150, 1.149000, 1.138303, 1.126004, 1.113206, 1.100855, 1.089277, 1.078678, 1.068468, 1.059253, 1.052191, 1.045552, 1.039737, 1.035145, 1.031080, 1.026850, 1.021081, 1.013819, 1.006757, 1.000597, 0.994868, 0.989850, 0.986069, 0.983216, 0.981355, 0.981166, 0.982451, 0.984270, 0.984832, 0.984471, 0.984280, 0.986109, 0.991286, 1.000090, 1.012024, 1.023639, 1.034631, 1.047302],
        'Sample_2 (KF=6.47)': [0.801634, 0.771438, 0.742237, 0.709832, 0.678207, 0.653886, 0.638888, 0.632456, 0.631769, 0.634756, 0.639054, 0.642985, 0.644860, 0.646490, 0.647788, 0.648105, 0.647713, 0.645488, 0.640984, 0.635740, 0.631117, 0.627589, 0.624967, 0.623212, 0.622139, 0.620959, 0.619238, 0.618592, 0.619666, 0.624414, 0.631874, 0.638786, 0.645138, 0.654623, 0.672372, 0.701585, 0.735848, 0.764386, 0.793231, 0.821822, 0.846323, 0.867788, 0.884711, 0.900533, 0.918766, 0.924383, 0.921015, 0.917522, 0.917199, 0.912357, 0.894138, 0.864984, 0.836521, 0.813830, 0.796789, 0.784933, 0.775954, 0.768320, 0.762588, 0.759024, 0.757465, 0.757344, 0.758312, 0.760602, 0.763249, 0.766360, 0.772667, 0.783222, 0.797106, 0.814673, 0.837737, 0.865409, 0.891667, 0.912222, 0.930438, 0.949712, 0.971437, 0.999115, 1.025881, 1.042647, 1.060207, 1.084768, 1.109963, 1.126983, 1.135715, 1.138100, 1.134174, 1.126686, 1.119287, 1.112619, 1.105531, 1.098295, 1.090551, 1.083227, 1.076621, 1.070153, 1.064853, 1.060851, 1.055713, 1.050331, 1.046119, 1.042442, 1.039293, 1.035747, 1.031388, 1.027223, 1.023881, 1.021126, 1.019222, 1.017973, 1.016819, 1.016217, 1.016792, 1.019224, 1.022024, 1.024846, 1.027042, 1.028637, 1.030912, 1.035498, 1.044863, 1.060484, 1.077334, 1.092126, 1.106185],
        'Sample_3 (KF=6.45)': [0.801726, 0.772495, 0.744539, 0.713566, 0.683027, 0.659077, 0.644068, 0.637292, 0.636335, 0.639131, 0.643175, 0.646887, 0.648787, 0.650439, 0.651589, 0.652171, 0.651928, 0.649913, 0.645592, 0.640825, 0.636253, 0.632762, 0.630279, 0.628597, 0.627563, 0.626350, 0.624720, 0.624181, 0.625106, 0.629792, 0.636960, 0.643632, 0.649892, 0.658806, 0.675623, 0.703172, 0.735316, 0.761501, 0.788148, 0.814794, 0.838106, 0.858391, 0.874368, 0.888943, 0.906016, 0.911385, 0.908753, 0.906178, 0.906688, 0.902830, 0.885823, 0.857949, 0.830621, 0.808982, 0.792924, 0.781748, 0.773273, 0.766143, 0.760732, 0.757301, 0.755882, 0.755747, 0.756515, 0.758633, 0.760982, 0.763749, 0.769495, 0.779118, 0.791859, 0.808188, 0.829530, 0.855127, 0.879545, 0.898741, 0.915863, 0.933943, 0.954411, 0.980521, 1.005577, 1.021297, 1.037773, 1.061052, 1.084979, 1.101002, 1.109326, 1.111215, 1.107418, 1.100108, 1.093069, 1.086814, 1.080194, 1.073322, 1.066076, 1.059077, 1.052827, 1.046692, 1.041725, 1.038020, 1.033193, 1.027970, 1.023946, 1.020439, 1.017386, 1.013950, 1.009843, 1.005874, 1.002785, 1.000266, 0.998565, 0.997405, 0.996344, 0.995821, 0.996423, 0.998816, 1.001542, 1.004391, 1.006617, 1.008222, 1.010488, 1.014932, 1.023950, 1.039168, 1.055620, 1.070096, 1.083818],
        'Sample_4 (KF=5.09)': [0.794490, 0.762878, 0.733086, 0.700687, 0.669157, 0.644513, 0.627731, 0.619778, 0.618615, 0.621953, 0.626477, 0.628903, 0.630952, 0.632918, 0.635057, 0.636893, 0.638360, 0.637156, 0.633399, 0.629016, 0.624218, 0.619834, 0.617228, 0.615750, 0.614750, 0.613633, 0.612148, 0.611696, 0.612890, 0.616859, 0.624421, 0.632090, 0.640124, 0.650429, 0.667739, 0.697333, 0.730997, 0.755889, 0.781142, 0.809558, 0.838553, 0.863643, 0.879522, 0.893016, 0.909259, 0.911043, 0.908008, 0.905505, 0.904221, 0.899011, 0.881153, 0.852753, 0.825366, 0.803425, 0.786727, 0.775273, 0.766987, 0.759958, 0.754610, 0.751240, 0.749458, 0.748913, 0.749129, 0.751064, 0.753432, 0.756122, 0.762043, 0.771675, 0.784511, 0.801245, 0.823580, 0.850835, 0.877302, 0.897790, 0.915120, 0.933043, 0.953520, 0.979189, 1.003451, 1.017962, 1.032991, 1.056808, 1.082775, 1.100133, 1.105914, 1.103862, 1.096121, 1.085450, 1.076786, 1.071371, 1.067513, 1.062827, 1.056459, 1.049255, 1.043041, 1.038312, 1.036285, 1.034333, 1.028326, 1.020620, 1.014429, 1.009245, 1.004727, 1.000311, 0.996037, 0.992833, 0.991057, 0.989986, 0.989646, 0.989407, 0.988699, 0.988452, 0.989601, 0.992913, 0.996689, 1.000828, 1.005146, 1.008743, 1.012290, 1.017547, 1.027630, 1.044168, 1.061340, 1.075435, 1.088572],
        'Sample_5 (KF=4.88)': [0.808266, 0.775441, 0.745469, 0.713744, 0.682373, 0.657164, 0.640084, 0.632214, 0.631038, 0.634090, 0.638756, 0.641084, 0.643298, 0.645122, 0.647224, 0.648838, 0.649893, 0.648609, 0.644458, 0.640218, 0.635574, 0.631254, 0.628737, 0.627214, 0.626177, 0.625140, 0.623720, 0.623266, 0.624471, 0.628650, 0.636220, 0.643891, 0.651897, 0.662045, 0.679242, 0.708485, 0.741810, 0.766974, 0.792249, 0.820024, 0.847630, 0.871767, 0.887289, 0.900655, 0.916910, 0.919156, 0.915995, 0.913152, 0.912105, 0.906850, 0.889251, 0.861352, 0.834307, 0.812638, 0.796176, 0.784888, 0.776678, 0.769666, 0.764302, 0.760872, 0.759153, 0.758600, 0.758855, 0.760906, 0.763217, 0.765938, 0.771755, 0.781466, 0.794170, 0.810741, 0.832692, 0.859329, 0.885236, 0.905255, 0.922257, 0.940052, 0.960317, 0.985599, 1.009691, 1.024460, 1.039730, 1.063245, 1.088377, 1.105179, 1.111127, 1.109935, 1.103387, 1.093625, 1.085316, 1.079742, 1.075367, 1.070237, 1.063869, 1.056694, 1.050618, 1.045536, 1.042597, 1.040162, 1.034525, 1.027725, 1.022177, 1.017411, 1.013250, 1.009083, 1.004803, 1.001445, 0.999335, 0.997943, 0.997295, 0.996776, 0.995916, 0.995617, 0.996578, 0.999685, 1.003155, 1.006886, 1.010648, 1.013711, 1.016809, 1.021569, 1.031076, 1.047226, 1.064310, 1.078578, 1.091903]
    }
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    fig = go.Figure()
    
    for i, (sample_name, spectrum) in enumerate(spectra_data.items()):
        fig.add_trace(go.Scatter(
            x=wavelengths,
            y=spectrum,
            mode='lines',
            name=sample_name,
            line=dict(color=colors[i], width=2.5),
            hovertemplate='<b>%{fullData.name}</b><br>' +
                         'Wavelength: %{x:.1f} nm<br>' +
                         'Absorbance: %{y:.3f}<extra></extra>'
        ))
    
    fig.update_layout(
        title='Real NIR Spectra - Pharmaceutical Analysis<br><sub>Karl Fischer (KF) Moisture Content Determination</sub>',
        xaxis_title='Wavelength (nm)',
        yaxis_title='Absorbance',
        height=500,
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02
        )
    )
    
    # Add annotation for the absorption peak region
    fig.add_vrect(
        x0=1400, x1=1500, 
        fillcolor="rgba(255,255,0,0.2)", 
        layer="below", line_width=0,
        annotation_text="H₂O absorption region", 
        annotation_position="top left"
    )
    
    st.plotly_chart(fig, width='stretch')
    
    # Real data statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Samples", 5)
    with col2:
        st.metric("Wavelengths", 125)
    with col3:
        st.metric("Range", "908-1676 nm")
    with col4:
        kf_values = [11.63, 6.47, 6.45, 5.09, 4.88]
        st.metric("KF Range", f"{min(kf_values):.2f}-{max(kf_values):.2f}")
    
    # Technical info
    st.info("""
    **📊 Real Dataset Information:**
    - **Application**: Karl Fischer moisture determination in pharmaceutical samples
    - **Technique**: Near-Infrared (NIR) spectroscopy  
    - **Region**: 908-1676 nm (typical NIR analytical window)
    - **Resolution**: ~6.2 nm spacing between data points
    - **Target**: Moisture content prediction using spectral fingerprints
    """)
    
    with st.expander("🔬 About This Analysis"):
        st.markdown("""
        This dataset demonstrates a **quantitative NIR application** where:
        
        - **X-variables**: 125 spectral intensities across NIR range
        - **Y-variable**: Karl Fischer (KF) moisture content (reference method)  
        - **Objective**: Build calibration model to predict moisture from spectra
        - **Challenges**: Spectral overlapping, baseline variations, noise

        **Key spectral features**:
        - 1400-1500 nm: Primary water absorption bands
        - Baseline trends: Scattering effects from particle size
        - Sample variations: Different moisture levels create spectral differences
        """)
    
    # Load data button
    if st.button("📥 Load This Dataset", type="primary", key="load_nir_dataset"):
        # Create DataFrame in proper format for the application
        data_for_app = pd.DataFrame(spectra_data).T
        data_for_app.columns = [f"{wl:.3f}" for wl in wavelengths]
        
        # Add sample metadata
        data_for_app.insert(0, 'Sample_ID', ['Sample_1', 'Sample_2', 'Sample_3', 'Sample_4', 'Sample_5'])
        data_for_app.insert(1, 'KF_Response', [11.63, 6.47, 6.45, 5.09, 4.88])
        
        # Store in session state
        st.session_state.current_data = data_for_app
        st.session_state.current_dataset = "NIR_KF_Dataset.csv"
        
        st.success("✅ **NIR dataset loaded successfully!**")
        st.info("🚀 Go to **Data Handling** to explore, or **PCA Analysis** to start multivariate analysis")
        st.balloons()
    

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
        if st.sidebar.button("🎯 PCA Analysis", use_container_width=True, key="nav_pca"):
            st.session_state.current_page = "PCA Analysis"
            st.rerun()
    else:
        st.sidebar.button("🎯 PCA Analysis", disabled=True, use_container_width=True, key="nav_pca_disabled")
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