"""Shared two-position workspace navigation."""

import streamlit as st


def render_workspace_toggle(current: str) -> None:
    columns = st.columns([1, 2, 1])
    with columns[1]:
        selected = st.segmented_control(
            "Workspace",
            ["Overview", "Financial intelligence"],
            default=current,
            required=True,
            selection_mode="single",
            label_visibility="collapsed",
            width="stretch",
        )
    if selected != current:
        st.switch_page("app.py" if selected == "Overview" else "pages/2_Financial_Intelligence.py")
