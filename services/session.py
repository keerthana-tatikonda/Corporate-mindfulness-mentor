# services/session.py
import time
import streamlit as st
from typing import Optional, Dict, Any, List

SESSION_KEY = "relax_session"

def init_session():
    if SESSION_KEY not in st.session_state:
        st.session_state[SESSION_KEY] = {
            "active_item_id": None,
            "step_index": 0,
            "started_at": None,
            "running": False
        }

def start_session(item_id: str):
    s = st.session_state[SESSION_KEY]
    s["active_item_id"] = item_id
    s["step_index"] = 0
    s["started_at"] = time.time()
    s["running"] = True

def stop_session():
    st.session_state[SESSION_KEY]["running"] = False

def next_step():
    st.session_state[SESSION_KEY]["step_index"] += 1

def get_state() -> Dict[str, Any]:
    return st.session_state[SESSION_KEY]
