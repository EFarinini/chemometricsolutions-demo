"""
ChemometricSolutions Interactive Demos
Main entry point for Streamlit Cloud deployment
"""

import streamlit as st

# Set page config FIRST - before any other Streamlit command
st.set_page_config(
    page_title="ChemometricSolutions - Interactive Demos",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import authentication utilities
from auth_utils import render_login_page, show_logout_button

if __name__ == "__main__":
    # Password protection - check authentication
    if not render_login_page():
        # User not authenticated, stop here
        st.stop()

    # User is authenticated, show logout button and run app
    show_logout_button()

    # Import and run the main application (without calling set_page_config again)
    from homepage import main_content
    main_content()

