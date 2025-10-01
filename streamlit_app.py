"""
ChemometricSolutions Interactive Demos
Main entry point for Streamlit Cloud deployment
"""

import streamlit as st
import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(__file__))

try:
    from homepage import main
    if __name__ == "__main__":
        main()
except ImportError as e:
    st.error(f"Import error: {e}")
    st.info("Please check that all required files are present in the repository")
except Exception as e:
    st.error(f"Application error: {e}")
    st.info("Please check the application logs for more details")
