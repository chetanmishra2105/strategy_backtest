"""Playbook page — renders playbook.md in-app."""

from __future__ import annotations

import os

import streamlit as st

from nsewing import config, ui

st.set_page_config(page_title="Playbook", page_icon="📖", layout="wide")
ui.sidebar_controls()

st.title("📖 Trading Playbook")

path = os.path.join(config.PROJECT_DIR, "playbook.md")
if os.path.exists(path):
    with open(path, "r", encoding="utf-8") as f:
        st.markdown(f.read())
else:
    st.error("playbook.md not found.")
