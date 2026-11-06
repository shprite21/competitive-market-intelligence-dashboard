"""Shared-password access gate for hosted Streamlit deployments."""

import hmac
import os


def _enabled(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def require_auth() -> None:
    """Require a configured password on hosted deployments and protect each page."""
    import streamlit as st

    password = os.environ.get("APP_AUTH_PASSWORD", "")
    required = (
        _enabled(os.environ.get("APP_AUTH_REQUIRED"))
        or _enabled(os.environ.get("RENDER"))
        or os.environ.get("APP_ENV", "").strip().lower() in {"prod", "production"}
    )
    if not password:
        if required:
            st.error("Access is locked: configure APP_AUTH_PASSWORD for this deployment.")
            st.stop()
        return

    if not st.session_state.get("app_authenticated", False):
        st.title("Sign in")
        st.caption("Enter the access password for this Market Intelligence workspace.")
        with st.form("app_login_form"):
            submitted_password = st.text_input(
                "Access password", type="password", key="app_login_password"
            )
            submitted = st.form_submit_button("Sign in", type="primary")
        if submitted:
            if hmac.compare_digest(submitted_password, password):
                st.session_state.app_authenticated = True
                st.rerun()
            st.error("The password is incorrect.")
        st.stop()

    if st.sidebar.button("Log out", key="app_logout"):
        st.session_state.app_authenticated = False
        st.rerun()
