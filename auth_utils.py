"""
Authentication utilities for ChemometricSolutions application.
Provides password protection with lockout mechanism.
"""

import streamlit as st
from datetime import datetime, timedelta

# ============================================================================
# CONFIGURATION
# ============================================================================

# Application password (easily configurable)
APP_PASSWORD = "como2026"

# Maximum number of failed login attempts before lockout
MAX_ATTEMPTS = 5

# Lockout duration in minutes
LOCKOUT_DURATION_MINUTES = 5


# ============================================================================
# AUTHENTICATION FUNCTIONS
# ============================================================================

def initialize_session_state():
    """
    Initialize session state variables for authentication.
    Called at the beginning of the authentication flow.
    """
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False

    if 'login_attempts' not in st.session_state:
        st.session_state.login_attempts = 0

    if 'lockout_time' not in st.session_state:
        st.session_state.lockout_time = None


def is_locked_out():
    """
    Check if the user is currently locked out due to too many failed attempts.

    Returns:
        tuple: (is_locked, remaining_seconds)
            - is_locked (bool): True if user is locked out
            - remaining_seconds (int): Seconds remaining in lockout, or 0
    """
    if st.session_state.lockout_time is None:
        return False, 0

    now = datetime.now()
    if now < st.session_state.lockout_time:
        remaining = (st.session_state.lockout_time - now).total_seconds()
        return True, int(remaining)
    else:
        # Lockout expired, reset attempts
        st.session_state.lockout_time = None
        st.session_state.login_attempts = 0
        return False, 0


def verify_password(password):
    """
    Verify the entered password against the configured password.

    Args:
        password (str): Password entered by the user

    Returns:
        bool: True if password is correct, False otherwise
    """
    return password == APP_PASSWORD


def handle_login_attempt(password):
    """
    Handle a login attempt with password verification and lockout logic.

    Args:
        password (str): Password entered by the user

    Returns:
        bool: True if login successful, False otherwise
    """
    if verify_password(password):
        # Successful login
        st.session_state.authenticated = True
        st.session_state.login_attempts = 0
        st.session_state.lockout_time = None
        return True
    else:
        # Failed login
        st.session_state.login_attempts += 1

        if st.session_state.login_attempts >= MAX_ATTEMPTS:
            # Trigger lockout
            lockout_end = datetime.now() + timedelta(minutes=LOCKOUT_DURATION_MINUTES)
            st.session_state.lockout_time = lockout_end

        return False


def logout():
    """
    Log out the current user by resetting session state.
    """
    st.session_state.authenticated = False
    st.session_state.login_attempts = 0
    st.session_state.lockout_time = None


def render_login_page():
    """
    Render the login page with password input form.
    Handles lockout display and login attempts.

    Returns:
        bool: True if user is authenticated, False otherwise
    """
    # Initialize session state
    initialize_session_state()

    # Check if already authenticated
    if st.session_state.authenticated:
        return True

    # Check lockout status
    locked, remaining_seconds = is_locked_out()

    # Display login form
    st.title("🔐 ChemometricSolutions")
    st.subheader("Authentication Required")

    if locked:
        # Show lockout message
        minutes = remaining_seconds // 60
        seconds = remaining_seconds % 60
        st.error(
            f"Too many failed login attempts. "
            f"Please wait {minutes}m {seconds}s before trying again."
        )
        st.info("The page will refresh automatically. Please wait...")

        # Auto-refresh to update countdown
        st.rerun()
        return False

    # Show login form
    with st.form("login_form"):
        st.write("Please enter the password to access the application.")

        password = st.text_input(
            "Password",
            type="password",
            placeholder="Enter password",
            help="Contact administrator for password"
        )

        submit_button = st.form_submit_button("Login")

        if submit_button:
            if not password:
                st.warning("Please enter a password.")
            else:
                success = handle_login_attempt(password)

                if success:
                    st.success("Login successful! Redirecting...")
                    st.rerun()
                else:
                    remaining_attempts = MAX_ATTEMPTS - st.session_state.login_attempts

                    if remaining_attempts > 0:
                        st.error(
                            f"Incorrect password. "
                            f"{remaining_attempts} attempt(s) remaining."
                        )
                    else:
                        # This will trigger lockout
                        st.rerun()

    # Show attempt counter
    if st.session_state.login_attempts > 0 and not locked:
        attempts_used = st.session_state.login_attempts
        st.caption(f"Login attempts: {attempts_used}/{MAX_ATTEMPTS}")

    # Add some styling and information
    st.markdown("---")
    st.markdown(
        """
        ### About ChemometricSolutions

        This application provides interactive chemometric analysis tools including:
        - Principal Component Analysis (PCA)
        - Multiple Linear Regression (MLR) and Design of Experiments (DOE)
        - Data preprocessing and transformations
        - And more...

        **Default password:** `demo123`
        """
    )

    return False


def show_logout_button():
    """
    Logout button disabled - not needed.
    Users stay logged in until browser is closed or session expires.
    """
    pass
