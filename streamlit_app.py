"""
ChemometricSolutions Interactive Demos
Main entry point for Streamlit Cloud deployment
"""

import streamlit as st

# Force light theme configuration for online deployment
st.set_page_config(
    page_title="ChemometricSolutions - Interactive Demos",
    page_icon="🧊",  # Using cube emoji as closest to the logo until we can add custom favicon
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://chemometricsolutions.com',
        'Report a bug': 'https://github.com/FarininiChemometricSolutions/chemometricsolutions-demos/issues',
        'About': """
        ## ChemometricSolutions Interactive Demos
        
        Professional chemometric tools for data analysis and multivariate statistics.
        
        **Developed by:** Dr. Emanuele Farinini, PhD  
        **Website:** https://chemometricsolutions.com  
        **GitHub:** https://github.com/FarininiChemometricSolutions
        
        © 2025 ChemometricSolutions - MIT License
        """
    }
)

# CSS for forcing light theme online + logo integration
st.markdown("""
<style>
    /* Force light theme for online deployment */
    .stApp {
        background-color: white !important;
        color: black !important;
    }
    
    /* Sidebar light theme with logo */
    .css-1d391kg {
        background-color: #f0f2f6 !important;
    }
    
    /* Custom logo in sidebar */
    .css-1d391kg::before {
        content: "";
        display: block;
        width: 60px;
        height: 60px;
        margin: 20px auto 10px auto;
        background-image: url("data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjAwIiBoZWlnaHQ9IjIwMCIgdmlld0JveD0iMCAwIDIwMCAyMDAiIGZpbGw9Im5vbmUiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+CjxkZWZzPgo8bGluZWFyR3JhZGllbnQgaWQ9ImdyYWQiIHgxPSIwJSIgeTE9IjAlIiB4Mj0iMTAwJSIgeTI9IjEwMCUiPgo8c3RvcCBvZmZzZXQ9IjAlIiBzdHlsZT0ic3RvcC1jb2xvcjojMkU1MjkzO3N0b3Atb3BhY2l0eToxIiAvPgo8c3RvcCBvZmZzZXQ9IjUwJSIgc3R5bGU9InN0b3AtY29sb3I6IzFFOTBGRjtzdG9wLW9wYWNpdHk6MSIgLz4KPHN0b3Agb2Zmc2V0PSIxMDAlIiBzdHlsZT0ic3RvcC1jb2xvcjojNGRhM2ZmO3N0b3Atb3BhY2l0eToxIiAvPgo8L2xpbmVhckdyYWRpZW50Pgo8L2RlZnM+Cjxwb2x5Z29uIHBvaW50cz0iMTAwLDIwIDIwLDYwIDIwLDE0MCA4MCwxODAgMTgwLDE0MCAyMCw2MCAyMCwxNDAiIGZpbGw9InVybCgjZ3JhZCkiLz4KPHN2Zz4=");
        background-size: contain;
        background-repeat: no-repeat;
        background-position: center;
    }
    
    /* Main content area */
    .block-container {
        background-color: white !important;
    }
    
    /* Metrics and info boxes */
    [data-testid="metric-container"] {
        background-color: white !important;
        border: 1px solid #e0e0e0 !important;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Text elements */
    .stMarkdown, .stText {
        color: black !important;
    }
    
    /* Headers with ChemometricSolutions branding */
    h1 {
        color: #1E90FF !important;
        border-bottom: 3px solid #1E90FF;
        padding-bottom: 10px;
    }
    
    h2, h3, h4, h5, h6 {
        color: #2E5293 !important;
    }
    
    /* Buttons with logo-inspired gradient */
    .stButton > button {
        background: linear-gradient(135deg, #1E90FF, #4da3ff) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #4da3ff, #1E90FF) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 12px rgba(30, 144, 255, 0.3) !important;
    }
    
    /* Info/warning/success boxes */
    .stInfo {
        background-color: #e7f3ff !important;
        border-left: 4px solid #1E90FF !important;
        border-radius: 6px !important;
    }
    
    .stSuccess {
        background-color: #d4edda !important;
        border-left: 4px solid #28a745 !important;
        border-radius: 6px !important;
    }
    
    .stWarning {
        background-color: #fff3cd !important;
        border-left: 4px solid #ffc107 !important;
        border-radius: 6px !important;
    }
    
    .stError {
        background-color: #f8d7da !important;
        border-left: 4px solid #dc3545 !important;
        border-radius: 6px !important;
    }
    
    /* Dataframes */
    .stDataFrame {
        background-color: white !important;
        border-radius: 8px !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
    }
    
    /* Selectboxes and inputs */
    .stSelectbox > div > div {
        background-color: white !important;
        color: black !important;
        border-radius: 6px !important;
    }
    
    .stTextInput > div > div > input {
        background-color: white !important;
        color: black !important;
        border-radius: 6px !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #f8f9fa !important;
        border-radius: 8px !important;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: white !important;
        color: black !important;
        border-radius: 6px !important;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #e7f3ff !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: #f8f9fa !important;
        color: black !important;
        border-radius: 6px !important;
    }
    
    /* CORREZIONE PRINCIPALE: Plotly charts - Background fix più robusto */
    /* Forza background bianco per tutti i grafici Plotly */
    div[data-testid="stPlotlyChart"] > div {
        background-color: white !important;
    }
    
    /* Forza background per l'SVG principale */
    .js-plotly-plot .plotly .main-svg,
    .js-plotly-plot .bg,
    .plotly-graph-div .bg {
        background-color: white !important;
        fill: white !important;
    }
    
    /* Assicura che il contenitore Plotly abbia sfondo bianco */
    .plotly-graph-div {
        background-color: white !important;
    }
    
    /* Forza colore del testo nei grafici */
    .js-plotly-plot .plotly text {
        fill: black !important;
        color: black !important;
    }
    
    /* Assi e labels */
    .js-plotly-plot .plotly .xtick text,
    .js-plotly-plot .plotly .ytick text,
    .js-plotly-plot .plotly .g-xtitle text,
    .js-plotly-plot .plotly .g-ytitle text {
        fill: black !important;
    }
    
    /* Griglia */
    .js-plotly-plot .plotly .gridlayer .xgrid path,
    .js-plotly-plot .plotly .gridlayer .ygrid path {
        stroke: #e0e0e0 !important;
    }
    
    /* Professional branding with logo-inspired gradient */
    .stApp::before {
        content: "";
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #2E5293, #1E90FF, #4da3ff, #1E90FF, #2E5293);
        z-index: 999;
        animation: brandingGlow 3s ease-in-out infinite alternate;
    }
    
    @keyframes brandingGlow {
        0% { opacity: 0.8; }
        100% { opacity: 1; }
    }
    
    /* Sidebar title enhancement */
    .css-1d391kg h1 {
        text-align: center !important;
        font-size: 1.2rem !important;
        color: #2E5293 !important;
        margin-top: 10px !important;
        font-weight: 700 !important;
    }
    
    /* Add subtle 3D effect to main content */
    .main .block-container {
        box-shadow: 0 0 20px rgba(30, 144, 255, 0.1) !important;
        border-radius: 12px !important;
        margin-top: 20px !important;
    }
</style>
""", unsafe_allow_html=True)

# Custom header with logo reference
st.sidebar.markdown("""
<div style="text-align: center; margin: 20px 0;">
    <div style="font-size: 1.1rem; font-weight: 700; color: #2E5293; margin-top: 10px;">
        ChemometricSolutions
    </div>
    <div style="font-size: 0.9rem; color: #666; font-style: italic;">
        Interactive Demos
    </div>
</div>
""", unsafe_allow_html=True)

# Import and run the main application
from homepage import main

if __name__ == "__main__":
    main()