"""
IndicAI Pro - Enterprise Neural Machine Translation & Voice Intelligence
Primary Visual Design Reference: OpsPilot Pro Executive Dark Dashboard & Voice Conversation UI
Deep Navy (#08162D) | Turquoise Accent (#20C9B7) | Dark Blue Cards (#142B49) | Borders (#1E3A5F)
"""

import os
import io
import time
import pandas as pd
import streamlit as st

# Configure Layout & Page Setup
st.set_page_config(
    page_title="IndicAI Pro | Multilingual Translation System",
    page_icon="IT",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Executive Theme CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');
    
    /* Global Reset & Base */
    html, body, [class*="css"], .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
        background-color: #08162D !important;
        color: #F8FAFC !important;
    }

    /* Hide default menu and footer only - DO NOT hide header or toolbar */
    #MainMenu {
        visibility: hidden !important;
    }
    footer {
        visibility: hidden !important;
    }
    
    header[data-testid="stHeader"] {
        background: transparent !important;
    }
    
    /* Ensure the mobile/desktop sidebar expand toggle button is ALWAYS visible & styled in executive turquoise */
    [data-testid="stSidebarCollapsedControl"],
    button[data-testid="stSidebarCollapsedControl"],
    .st-emotion-cache-12fmjuu,
    .st-emotion-cache-15ecox0,
    [data-testid="collapsedControl"] {
        display: flex !important;
        visibility: visible !important;
        opacity: 1 !important;
        position: fixed !important;
        top: 14px !important;
        left: 14px !important;
        z-index: 99999999 !important;
        background-color: #142B49 !important;
        border: 1.5px solid #20C9B7 !important;
        border-radius: 8px !important;
        color: #20C9B7 !important;
        padding: 6px 8px !important;
        box-shadow: 0 4px 18px rgba(0, 0, 0, 0.6) !important;
        transition: all 0.2s ease !important;
        cursor: pointer !important;
    }
    [data-testid="stSidebarCollapsedControl"]:hover,
    button[data-testid="stSidebarCollapsedControl"]:hover,
    [data-testid="collapsedControl"]:hover {
        background-color: #1B365D !important;
        border-color: #20C9B7 !important;
        box-shadow: 0 0 12px rgba(32, 201, 183, 0.3) !important;
    }
    [data-testid="stSidebarCollapsedControl"] svg,
    button[data-testid="stSidebarCollapsedControl"] svg,
    [data-testid="collapsedControl"] svg {
        fill: #20C9B7 !important;
        stroke: #20C9B7 !important;
        color: #20C9B7 !important;
        width: 1.4rem !important;
        height: 1.4rem !important;
    }
    [data-testid="stSidebarCollapseButton"] {
        color: #94A3B8 !important;
    }
    [data-testid="stSidebarCollapseButton"]:hover {
        color: #20C9B7 !important;
    }

    /* Executive Pills Styling for Top Navigation */
    div[data-testid="stPills"] {
        background: transparent !important;
        margin-bottom: 0.85rem !important;
    }
    div[data-testid="stPills"] button {
        background-color: #142B49 !important;
        border: 1px solid #1E3A5F !important;
        color: #94A3B8 !important;
        border-radius: 8px !important;
        font-weight: 500 !important;
        font-size: 0.88rem !important;
        padding: 0.4rem 1rem !important;
        transition: all 0.15s ease !important;
    }
    div[data-testid="stPills"] button:hover {
        background-color: #1B365D !important;
        border-color: #20C9B7 !important;
        color: #F8FAFC !important;
    }
    div[data-testid="stPills"] button[aria-selected="true"] {
        background-color: #20C9B7 !important;
        border-color: #20C9B7 !important;
        color: #08162D !important;
        font-weight: 600 !important;
    }
    div[data-testid="stPills"] button[aria-selected="true"] span {
        color: #08162D !important;
        font-weight: 600 !important;
    }
    
    .block-container {
        padding-top: 2.2rem !important;
        padding-bottom: 3rem !important;
        max-width: 1380px !important;
    }

    @media (max-width: 768px) {
        .block-container {
            padding-top: 3.8rem !important;
            padding-left: 0.85rem !important;
            padding-right: 0.85rem !important;
        }
        .page-main-title {
            font-size: 1.4rem !important;
        }
        [data-testid="stSidebarCollapsedControl"],
        button[data-testid="stSidebarCollapsedControl"] {
            top: 0.5rem !important;
            left: 0.5rem !important;
            z-index: 9999999 !important;
        }
    }

    /* Sidebar Executive Styling */
    section[data-testid="stSidebar"] {
        background-color: #0B1B36 !important;
        border-right: 1px solid #1E3A5F !important;
    }
    section[data-testid="stSidebar"] .block-container {
        padding-top: 1.75rem !important;
        padding-left: 1.25rem !important;
        padding-right: 1.25rem !important;
    }

    /* Navigation Radio Items in Sidebar */
    div[data-testid="stRadio"] > div {
        background: transparent;
        gap: 5px;
    }
    div[data-testid="stRadio"] label {
        background-color: transparent !important;
        border: 1px solid transparent !important;
        border-radius: 8px !important;
        padding: 0.65rem 0.9rem !important;
        margin: 0 !important;
        cursor: pointer !important;
        transition: all 0.15s ease-in-out !important;
        color: #94A3B8 !important;
        font-size: 0.9rem !important;
        font-weight: 500 !important;
    }
    div[data-testid="stRadio"] label:hover {
        background-color: #142B49 !important;
        color: #F8FAFC !important;
    }
    div[data-testid="stRadio"] label[data-checked="true"],
    div[data-testid="stRadio"] label:has(input:checked) {
        background-color: #20C9B7 !important;
        color: #08162D !important;
        font-weight: 600 !important;
    }
    div[data-testid="stRadio"] label[data-checked="true"] span,
    div[data-testid="stRadio"] label:has(input:checked) span {
        color: #08162D !important;
        font-weight: 600 !important;
    }

    /* Executive Breadcrumbs */
    .top-breadcrumb-bar {
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 0.5rem;
    }
    .top-breadcrumb {
        font-size: 0.78rem;
        color: #64748B;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    .breadcrumb-pill {
        font-size: 0.7rem;
        font-weight: 600;
        color: #20C9B7;
        background: rgba(32, 201, 183, 0.1);
        border: 1px solid rgba(32, 201, 183, 0.25);
        padding: 2px 7px;
        border-radius: 4px;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }

    .page-main-title {
        font-size: 1.85rem;
        font-weight: 700;
        color: #F8FAFC;
        letter-spacing: -0.025em;
        margin: 0 0 0.4rem 0;
    }
    .page-main-desc {
        font-size: 0.92rem;
        color: #94A3B8;
        line-height: 1.55;
        margin-bottom: 1.5rem;
    }

    /* Executive Metric & KPI Cards */
    .kpi-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
        gap: 16px;
        margin-bottom: 1.5rem;
    }
    .kpi-card {
        background-color: #142B49;
        border: 1px solid #1E3A5F;
        border-radius: 10px;
        padding: 1.25rem;
    }
    .kpi-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.65rem;
    }
    .kpi-label {
        font-size: 0.8rem;
        font-weight: 500;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .kpi-badge {
        font-size: 0.75rem;
        font-weight: 600;
        color: #20C9B7;
        background: rgba(32, 201, 183, 0.12);
        border: 1px solid rgba(32, 201, 183, 0.25);
        padding: 2px 7px;
        border-radius: 4px;
    }
    .kpi-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #F8FAFC;
        letter-spacing: -0.03em;
        margin-bottom: 0.3rem;
    }
    .kpi-subtext {
        font-size: 0.8rem;
        color: #64748B;
    }

    /* Structured Dark Card */
    .app-card {
        background-color: #142B49;
        border: 1px solid #1E3A5F;
        border-radius: 10px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1.25rem;
    }
    .app-card-title {
        font-size: 1.05rem;
        font-weight: 600;
        color: #F8FAFC;
        margin-bottom: 0.35rem;
    }
    .app-card-sub {
        font-size: 0.85rem;
        color: #94A3B8;
        margin-bottom: 1rem;
    }

    /* Translation Panel Containers */
    .panel-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.5rem;
    }
    .panel-title {
        font-size: 0.82rem;
        font-weight: 600;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    .panel-tag {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        background: #0E223D;
        color: #20C9B7;
        padding: 2px 8px;
        border-radius: 4px;
        border: 1px solid #1E3A5F;
    }

    .output-box {
        background-color: #0E223D;
        border: 1px solid #1E3A5F;
        border-radius: 8px;
        padding: 1.15rem 1.25rem;
        min-height: 220px;
        font-size: 1.1rem;
        line-height: 1.7;
        color: #F8FAFC;
        white-space: pre-wrap;
        word-break: break-word;
    }
    .output-placeholder {
        background-color: #0E223D;
        border: 1px dashed #1E3A5F;
        border-radius: 8px;
        padding: 1rem;
        min-height: 220px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        color: #64748B;
        font-size: 0.92rem;
        text-align: center;
    }

    /* Audio Waveform Animation */
    .waveform-container {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 4px;
        height: 52px;
        padding: 6px 14px;
        background: #0E223D;
        border: 1px solid #1E3A5F;
        border-radius: 8px;
        margin: 0.75rem 0;
    }
    .wave-bar {
        width: 3.5px;
        background-color: #20C9B7;
        border-radius: 2px;
        animation: waveAnim 1.2s ease-in-out infinite alternate;
    }
    @keyframes waveAnim {
        0% { height: 6px; opacity: 0.35; }
        50% { height: 38px; opacity: 1; }
        100% { height: 10px; opacity: 0.55; }
    }
    .wave-bar:nth-child(1) { animation-delay: 0.05s; }
    .wave-bar:nth-child(2) { animation-delay: 0.15s; }
    .wave-bar:nth-child(3) { animation-delay: 0.3s; }
    .wave-bar:nth-child(4) { animation-delay: 0.45s; }
    .wave-bar:nth-child(5) { animation-delay: 0.2s; }
    .wave-bar:nth-child(6) { animation-delay: 0.35s; }
    .wave-bar:nth-child(7) { animation-delay: 0.55s; }
    .wave-bar:nth-child(8) { animation-delay: 0.25s; }
    .wave-bar:nth-child(9) { animation-delay: 0.4s; }
    .wave-bar:nth-child(10) { animation-delay: 0.1s; }
    .wave-bar:nth-child(11) { animation-delay: 0.3s; }
    .wave-bar:nth-child(12) { animation-delay: 0.5s; }
    .wave-bar:nth-child(13) { animation-delay: 0.18s; }
    .wave-bar:nth-child(14) { animation-delay: 0.38s; }
    .wave-bar:nth-child(15) { animation-delay: 0.22s; }
    .wave-bar:nth-child(16) { animation-delay: 0.42s; }
    .wave-bar:nth-child(17) { animation-delay: 0.12s; }
    .wave-bar:nth-child(18) { animation-delay: 0.32s; }

    /* Conversation Chat Stream */
    .convo-bubble-a {
        background-color: #142B49;
        border-left: 3px solid #20C9B7;
        border-top: 1px solid #1E3A5F;
        border-right: 1px solid #1E3A5F;
        border-bottom: 1px solid #1E3A5F;
        border-radius: 8px;
        padding: 1rem 1.25rem;
        margin-bottom: 0.85rem;
    }
    .convo-bubble-b {
        background-color: #0E223D;
        border-left: 3px solid #38BDF8;
        border-top: 1px solid #1E3A5F;
        border-right: 1px solid #1E3A5F;
        border-bottom: 1px solid #1E3A5F;
        border-radius: 8px;
        padding: 1rem 1.25rem;
        margin-bottom: 0.85rem;
    }
    .bubble-meta {
        display: flex;
        justify-content: space-between;
        font-size: 0.78rem;
        font-weight: 600;
        margin-bottom: 0.4rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .bubble-orig {
        font-size: 0.95rem;
        color: #94A3B8;
        margin-bottom: 0.35rem;
    }
    .bubble-trans {
        font-size: 1.1rem;
        font-weight: 500;
        color: #F8FAFC;
        line-height: 1.5;
    }

    /* Streamlit Control Overrides */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea,
    .stSelectbox > div > div {
        background-color: #0E223D !important;
        border: 1px solid #1E3A5F !important;
        color: #F8FAFC !important;
        border-radius: 8px !important;
    }
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: #20C9B7 !important;
        box-shadow: 0 0 0 1px #20C9B7 !important;
    }

    /* Primary Action Buttons (Turquoise) */
    button[kind="primary"] {
        background-color: #20C9B7 !important;
        color: #08162D !important;
        font-weight: 600 !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.55rem 1.25rem !important;
        transition: opacity 0.15s ease !important;
    }
    button[kind="primary"]:hover {
        opacity: 0.9 !important;
    }

    /* Secondary Buttons */
    button[kind="secondary"] {
        background-color: #142B49 !important;
        color: #E2E8F0 !important;
        border: 1px solid #1E3A5F !important;
        border-radius: 8px !important;
    }
    button[kind="secondary"]:hover {
        background-color: #1B365D !important;
        border-color: #20C9B7 !important;
    }

    /* Mobile Voice Smartphone Mockup & Card */
    .mobile-phone-frame {
        max-width: 480px;
        margin: 0 auto 2rem auto;
        background: #0B1B36;
        border: 1px solid #1E3A5F;
        border-radius: 28px;
        padding: 1.5rem 1.25rem;
        box-shadow: 0 20px 45px rgba(0, 0, 0, 0.55);
    }
    .mobile-dock {
        background: #08162D;
        border: 1px solid #1E3A5F;
        border-radius: 22px;
        padding: 1.25rem 1rem;
        margin-top: 1.25rem;
        text-align: center;
        position: relative;
    }
    .mobile-speaker-btn {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 60px;
        height: 60px;
        border-radius: 50%;
        background: radial-gradient(circle, #20C9B7 0%, #0D9488 100%);
        border: 2px solid #20C9B7;
        box-shadow: 0 0 20px rgba(32, 201, 183, 0.4);
        color: #08162D;
        font-size: 1.5rem;
        margin: 0.75rem auto;
        cursor: pointer;
        transition: transform 0.15s ease;
    }
    .mobile-speaker-btn:hover {
        transform: scale(1.05);
    }

    /* Style Streamlit Audio Input into Large Glowing Circular Mic Button */
    div[data-testid="stAudioInput"] {
        background: transparent !important;
        border: none !important;
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
        justify-content: center !important;
        padding: 0 !important;
    }
    div[data-testid="stAudioInput"] button[data-testid="stAudioInputRecordButton"],
    div[data-testid="stAudioInput"] button {
        width: 76px !important;
        height: 76px !important;
        border-radius: 50% !important;
        background: radial-gradient(circle, #20C9B7 0%, #0F766E 100%) !important;
        border: 3px solid rgba(32, 201, 183, 0.4) !important;
        box-shadow: 0 0 25px rgba(32, 201, 183, 0.5) !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
        cursor: pointer !important;
        margin: 0.5rem auto !important;
    }
    div[data-testid="stAudioInput"] button:hover {
        transform: scale(1.04) !important;
        box-shadow: 0 0 30px rgba(32, 201, 183, 0.7) !important;
    }
    div[data-testid="stAudioInput"] button:active {
        transform: scale(0.92) !important;
        background: radial-gradient(circle, #F43F5E 0%, #BE123C 100%) !important;
        border-color: #FDA4AF !important;
        box-shadow: 0 0 35px rgba(244, 63, 94, 0.8) !important;
    }
    div[data-testid="stAudioInput"] svg {
        width: 32px !important;
        height: 32px !important;
        fill: #08162D !important;
        color: #08162D !important;
    }

    /* Sub-status Indicator Pills */
    .status-pill-subtle {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-size: 0.78rem;
        font-weight: 500;
        background: #0E223D;
        color: #94A3B8;
        border: 1px solid #1E3A5F;
        padding: 4px 10px;
        border-radius: 6px;
    }
    .status-indicator-dot {
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background-color: #20C9B7;
    }
    
    .state-badge-active {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-size: 0.75rem;
        font-weight: 600;
        color: #20C9B7;
        background: rgba(32, 201, 183, 0.12);
        border: 1px solid rgba(32, 201, 183, 0.3);
        padding: 3px 8px;
        border-radius: 4px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
</style>
""", unsafe_allow_html=True)

# Imports from modular engine
from model_engine.config import (
    SUPPORTED_LANGUAGES,
    LANGUAGE_NATIVE_NAMES,
    LANGUAGE_SPEECH_CAPABILITIES,
    get_model_for_pair,
)
from model_engine.translator import TranslationEngine
from model_engine.speech_recognition import SpeechRecognizer, FLORES_TO_WHISPER_LANG
from model_engine.text_to_speech import TextToSpeechEngine, FLORES_TO_TTS_LANG
from evaluation.evaluator import compute_metrics
from evaluation.test_benchmark import get_saved_benchmark

# Initialize Session State
if "history" not in st.session_state:
    st.session_state.history = []

if "dialogue_history" not in st.session_state:
    st.session_state.dialogue_history = []

if "hf_token" not in st.session_state:
    secret_tok = ""
    try:
        if "HF_TOKEN" in st.secrets:
            secret_tok = st.secrets["HF_TOKEN"]
    except Exception:
        pass
    st.session_state.hf_token = os.environ.get("HF_TOKEN", "") or secret_tok

if "engine" not in st.session_state:
    st.session_state.engine = TranslationEngine(hf_token=st.session_state.hf_token)

if "asr_engine" not in st.session_state:
    st.session_state.asr_engine = SpeechRecognizer(model_size="base", device="cpu", compute_type="int8")

if "tts_engine" not in st.session_state:
    st.session_state.tts_engine = TextToSpeechEngine()

# Language helper with clean professional labeling (No emoji clutter)
def format_lang_clean(name: str) -> str:
    code = SUPPORTED_LANGUAGES.get(name, "")
    native = LANGUAGE_NATIVE_NAMES.get(name, name)
    cap = LANGUAGE_SPEECH_CAPABILITIES.get(code, {})
    if cap.get("asr") and cap.get("tts"):
        tag = "[Voice + Audio]"
    elif cap.get("asr"):
        tag = "[Voice In]"
    else:
        tag = "[Text Only]"
    return f"{name} ({native}) · {tag}"

all_lang_names = list(SUPPORTED_LANGUAGES.keys())

# Preset testing prompts
PRESETS = [
    ("Governance & AI", "Artificial intelligence is transforming public administration, healthcare, and education across India."),
    ("Digital Infrastructure", "Robust digital infrastructure and high-speed connectivity are vital for regional economic development."),
    ("Multilingual NLP", "Neural machine translation enables seamless communication across diverse linguistic communities."),
    ("Academic Research", "Scholars and researchers require accurate translation systems to preserve and disseminate scientific knowledge."),
]

NAV_OPTIONS = [
    "Translate",
    "Voice Conversation",
    "Quality Benchmarks",
    "History",
    "Settings"
]

if "nav_choice" not in st.session_state:
    st.session_state.nav_choice = "Translate"

# ==============================================================================
# SIDEBAR NAVIGATION
# ==============================================================================
with st.sidebar:
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 1.75rem;">
        <div style="width: 38px; height: 38px; border-radius: 8px; background: #20C9B7; display: flex; align-items: center; justify-content: center; color: #08162D; font-weight: 800; font-size: 1.1rem; letter-spacing: -0.03em;">
            IT
        </div>
        <div>
            <div style="font-size: 1.1rem; font-weight: 700; color: #F8FAFC; letter-spacing: -0.01em;">IndicAI Pro</div>
            <div style="font-size: 0.72rem; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.05em;">Protected Workspace</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div style="font-size: 0.72rem; font-weight: 600; color: #64748B; text-transform: uppercase; letter-spacing: 0.08em; margin: 1.25rem 0 0.5rem 0.25rem;">Workspace Navigation</div>', unsafe_allow_html=True)
    
    current_idx = NAV_OPTIONS.index(st.session_state.nav_choice) if st.session_state.nav_choice in NAV_OPTIONS else 0
    sidebar_choice = st.radio(
        "Navigation",
        options=NAV_OPTIONS,
        index=current_idx,
        label_visibility="collapsed",
        key="sidebar_nav_radio"
    )
    if sidebar_choice != st.session_state.nav_choice:
        st.session_state.nav_choice = sidebar_choice
        st.rerun()

    st.markdown('<div style="margin-top: 2.25rem; border-top: 1px solid #1E3A5F; padding-top: 1.25rem;"></div>', unsafe_allow_html=True)
    
    # Engine Telemetry Footer in Sidebar
    st.markdown("""
    <div style="font-size: 0.72rem; color: #64748B; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 0.65rem;">System Engine</div>
    <div style="display: flex; flex-direction: column; gap: 6px;">
        <div class="status-pill-subtle">
            <span class="status-indicator-dot"></span>
            <span>22 Scheduled Languages Active</span>
        </div>
        <div class="status-pill-subtle">
            <span class="status-indicator-dot"></span>
            <span>IndicTrans2 (Dense Checkpoints)</span>
        </div>
        <div class="status-pill-subtle">
            <span class="status-indicator-dot"></span>
            <span>faster-whisper ASR (int8 CPU)</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ==============================================================================
# TOP EXECUTIVE NAVIGATION BAR (ALWAYS ACCESSIBLE ON MOBILE & DESKTOP)
# ==============================================================================
top_nav_col1, top_nav_col2 = st.columns([10, 2])
with top_nav_col1:
    top_pills_val = st.pills(
        "Workspace Navigation",
        options=NAV_OPTIONS,
        default=st.session_state.nav_choice,
        label_visibility="collapsed",
        key="top_nav_pills_selector"
    )
    if top_pills_val and top_pills_val != st.session_state.nav_choice:
        st.session_state.nav_choice = top_pills_val
        st.rerun()

with top_nav_col2:
    st.markdown("""
    <div style="display:flex; justify-content:flex-end; align-items:center; height:100%; padding-top:4px;">
        <span class="breadcrumb-pill">22 Languages</span>
    </div>
    """, unsafe_allow_html=True)

nav_choice = st.session_state.nav_choice

# ==============================================================================
# VIEW 1: TRANSLATE (TEXT-TO-TEXT WORKSPACE)
# ==============================================================================
if nav_choice == "Translate":
    st.markdown("""
    <div class="top-breadcrumb-bar">
        <span class="top-breadcrumb">IndicAI Pro &gt; Workspace &gt; Translate</span>
        <span class="breadcrumb-pill">Protected Preview</span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<h1 class="page-main-title">Neural Machine Translation Workspace</h1>', unsafe_allow_html=True)
    st.markdown('<div class="page-main-desc">Bidirectional translation across all 22 Eighth Schedule Indian languages and English utilizing specialized IndicTrans2 dense transformer checkpoints.</div>', unsafe_allow_html=True)

    # Top Control Bar (Language Selector & Routing)
    ctrl_col1, ctrl_swap, ctrl_col2, ctrl_meta = st.columns([4, 1, 4, 3])

    def do_swap_languages():
        src = st.session_state.get("src_lang_sel", "English")
        tgt = st.session_state.get("tgt_lang_sel", "Hindi")
        st.session_state["src_lang_sel"] = tgt
        st.session_state["tgt_lang_sel"] = src

    with ctrl_col1:
        if "src_lang_sel" not in st.session_state:
            st.session_state["src_lang_sel"] = "English"
        txt_src = st.selectbox(
            "Source Language",
            options=all_lang_names,
            key="src_lang_sel",
            format_func=format_lang_clean
        )

    with ctrl_swap:
        st.write("")
        st.write("")
        st.button("⇄", help="Swap Source and Target Languages", key="swap_btn", use_container_width=True, on_click=do_swap_languages)

    with ctrl_col2:
        if "tgt_lang_sel" not in st.session_state:
            st.session_state["tgt_lang_sel"] = "Hindi"
        txt_tgt = st.selectbox(
            "Target Language",
            options=all_lang_names,
            key="tgt_lang_sel",
            format_func=format_lang_clean
        )

    code_s = SUPPORTED_LANGUAGES[txt_src]
    code_t = SUPPORTED_LANGUAGES[txt_tgt]
    model_routed = get_model_for_pair(code_s, code_t)

    with ctrl_meta:
        st.write("")
        st.write("")
        if model_routed == "IDENTITY":
            st.markdown('<div style="background:#0E223D; border:1px solid #1E3A5F; padding:8px 12px; border-radius:8px; font-size:0.8rem; color:#38BDF8; font-weight:600;">Identity Translation (Source == Target)</div>', unsafe_allow_html=True)
        else:
            model_short = model_routed.split("/")[-1]
            st.markdown(f'<div style="background:#0E223D; border:1px solid #1E3A5F; padding:8px 12px; border-radius:8px; font-size:0.8rem; color:#20C9B7; font-family:\'JetBrains Mono\'; font-weight:500;">Model: {model_short}</div>', unsafe_allow_html=True)

    # Preset Sample Chips
    st.write("")
    p_cols = st.columns([1.2] + [2.5] * len(PRESETS))
    with p_cols[0]:
        st.markdown("<p style='font-size:0.8rem; font-weight:600; color:#94A3B8; margin-top:8px; text-transform:uppercase;'>Samples:</p>", unsafe_allow_html=True)
    for idx, (label, sample_text) in enumerate(PRESETS):
        with p_cols[idx + 1]:
            if st.button(label, key=f"t_sample_{idx}", use_container_width=True):
                st.session_state["text_input_area"] = sample_text
                st.rerun()

    st.write("")

    # Translation Panels (Source & Target)
    tp_1, tp_2 = st.columns(2)
    with tp_1:
        st.markdown(f"""
        <div class="panel-header">
            <span class="panel-title">Source Input ({txt_src})</span>
            <span class="panel-tag">{code_s}</span>
        </div>
        """, unsafe_allow_html=True)
        t_user_in = st.text_area(
            "Source Text",
            height=260,
            placeholder=f"Enter or paste text in {txt_src}...",
            label_visibility="collapsed",
            key="text_input_area"
        )
        t_c = len(t_user_in) if t_user_in else 0
        t_w = len(t_user_in.split()) if t_user_in and t_user_in.strip() else 0
        st.markdown(f'<div style="font-size:0.75rem; color:#64748B; margin-top:4px;">Characters: <span style="color:#94A3B8;">{t_c}</span> | Words: <span style="color:#94A3B8;">{t_w}</span></div>', unsafe_allow_html=True)

    with tp_2:
        st.markdown(f"""
        <div class="panel-header">
            <span class="panel-title">Translated Output ({txt_tgt})</span>
            <span class="panel-tag">{code_t}</span>
        </div>
        """, unsafe_allow_html=True)
        
        out_text = st.session_state.get("text_out_val", "")
        if out_text:
            st.markdown(f'<div class="output-box">{out_text}</div>', unsafe_allow_html=True)
            
            # Action Controls (Copy, Download, Audio)
            act1, act2, act3 = st.columns([2, 2, 2])
            with act1:
                st.download_button(
                    "Download Output (.txt)",
                    data=out_text,
                    file_name=f"translation_{code_s}_to_{code_t}.txt",
                    mime="text/plain",
                    use_container_width=True
                )
            with act2:
                if LANGUAGE_SPEECH_CAPABILITIES.get(code_t, {}).get("tts"):
                    with st.popover("Listen to Audio"):
                        try:
                            t_audio, _ = st.session_state.tts_engine.synthesize(out_text, target_lang_code=code_t)
                            st.audio(t_audio, format="audio/mp3")
                        except Exception as e:
                            st.caption(f"Audio engine notice: {e}")
                else:
                    st.caption("Audio speech unavailable for this dialect")
            with act3:
                if st.button("Clear Output", key="clear_out_btn", use_container_width=True):
                    st.session_state["text_out_val"] = ""
                    st.rerun()
        else:
            st.markdown("""
            <div class="output-placeholder">
                <div style="font-weight: 500; color: #94A3B8; margin-bottom: 4px;">Translation Output Panel</div>
                <div style="font-size: 0.82rem; color: #64748B;">Enter source text and click Translate to execute neural inference</div>
            </div>
            """, unsafe_allow_html=True)

    st.write("")
    btn_c1, btn_c2, btn_c3 = st.columns([1, 2, 1])
    with btn_c2:
        if st.button("Translate Text", key="btn_trans_run", type="primary", use_container_width=True):
            if not t_user_in or not t_user_in.strip():
                st.warning("Please enter source text before translating.")
            else:
                with st.spinner(f"Translating to {txt_tgt}..."):
                    t_st = time.time()
                    res_t = st.session_state.engine.translate(t_user_in, src_lang=code_s, tgt_lang=code_t)
                    lat = time.time() - t_st
                    st.session_state["text_out_val"] = res_t
                    st.session_state.history.append({
                        "Time": time.strftime("%H:%M:%S"),
                        "Source": txt_src,
                        "Target": txt_tgt,
                        "Input": t_user_in,
                        "Translated": res_t,
                        "Model": model_routed.split("/")[-1],
                        "Latency (s)": round(lat, 2)
                    })
                    st.rerun()


# ==============================================================================
# VIEW 2: VOICE CONVERSATION (SPEECH-TO-SPEECH WORKSPACE)
# ==============================================================================
elif nav_choice == "Voice Conversation":
    st.markdown("""
    <div class="top-breadcrumb-bar">
        <span class="top-breadcrumb">IndicAI Pro &gt; Workspace &gt; Voice Conversation</span>
        <span class="breadcrumb-pill">Protected Preview</span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<h1 class="page-main-title">Voice Conversation &amp; Speech Translator</h1>', unsafe_allow_html=True)
    st.markdown('<div class="page-main-desc">Turn-by-turn Speech-to-Speech translation enabling natural conversation between two speakers of different Indian languages, or direct audio file translation.</div>', unsafe_allow_html=True)

    voice_mode = st.radio(
        "Voice Translator Layout",
        ["Mobile Voice Translator (Hold & Record)", "Two-Speaker Dialogue Console", "Audio File Upload"],
        horizontal=True,
        key="voice_mode_radio"
    )

    if voice_mode == "Mobile Voice Translator (Hold & Record)":
        # ----------------------------------------------------------------------
        # MOBILE PHONE TRANSLATOR UI (EXACT MATCH TO UPLOADED REFERENCE SCREENSHOT)
        # ----------------------------------------------------------------------
        st.markdown("""
        <div class="mobile-phone-frame">
            <div class="mobile-voice-header">
                <div style="display:flex; align-items:center; gap:8px;">
                    <span style="font-size:0.85rem; color:#64748B; font-weight:600;">◀ VOICE</span>
                    <span class="mobile-voice-title">Voice Translator</span>
                </div>
                <div class="state-badge-active">● Listening Ready</div>
            </div>
        """, unsafe_allow_html=True)

        # Language Pair Selector (From / To)
        mv_c1, mv_swap, mv_c2 = st.columns([5, 1, 5])
        
        def do_mv_swap():
            s = st.session_state.get("mv_src_lang", "English")
            t = st.session_state.get("mv_tgt_lang", "Hindi")
            st.session_state["mv_src_lang"] = t
            st.session_state["mv_tgt_lang"] = s

        with mv_c1:
            if "mv_src_lang" not in st.session_state:
                st.session_state["mv_src_lang"] = "English"
            m_src = st.selectbox(
                "From",
                options=["Auto Detect"] + all_lang_names,
                format_func=lambda x: "Auto Detect" if x == "Auto Detect" else format_lang_clean(x),
                key="mv_src_lang"
            )

        with mv_swap:
            st.write("")
            st.write("")
            st.button("⇄", key="btn_mv_swap", on_click=do_mv_swap, help="Swap Languages")

        with mv_c2:
            if "mv_tgt_lang" not in st.session_state:
                st.session_state["mv_tgt_lang"] = "Hindi"
            m_tgt = st.selectbox(
                "To",
                options=all_lang_names,
                key="mv_tgt_lang",
                format_func=format_lang_clean
            )

        m_tgt_code = SUPPORTED_LANGUAGES[m_tgt]
        m_tgt_cap = LANGUAGE_SPEECH_CAPABILITIES.get(m_tgt_code, {})

        # Optional Manual Text Input / Correction Field
        m_typed_text = st.text_input(
            "Write here | (Optional text message)",
            placeholder="Write here | (or hold the mic button below to record)",
            key="mv_typed_text",
            label_visibility="visible"
        )

        # Curved Recording Dock with Waveform & Hold-to-Record Mic Button
        st.markdown("""
        <div class="mobile-dock">
            <div style="font-size:0.75rem; text-transform:uppercase; letter-spacing:0.06em; color:#94A3B8; margin-bottom:0.25rem;">
                Audio Frequency Waveform
            </div>
            <div class="waveform-container" style="background:#08162D; border:none; height:46px;">
                <div class="wave-bar"></div><div class="wave-bar"></div><div class="wave-bar"></div>
                <div class="wave-bar"></div><div class="wave-bar"></div><div class="wave-bar"></div>
                <div class="wave-bar"></div><div class="wave-bar"></div><div class="wave-bar"></div>
                <div class="wave-bar"></div><div class="wave-bar"></div><div class="wave-bar"></div>
                <div class="wave-bar"></div><div class="wave-bar"></div><div class="wave-bar"></div>
                <div class="wave-bar"></div><div class="wave-bar"></div><div class="wave-bar"></div>
            </div>
            <div style="font-size:0.78rem; font-weight:600; color:#20C9B7; margin-bottom:0.5rem;">
                Hold to Record &bull; Tap to Speak
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Streamlit Native Mic Button (Transformed via CSS into circular hold-and-record button)
        mv_audio = st.audio_input("Record Voice Input", key="mv_mic_input", label_visibility="collapsed")

        # Action Button to Execute Pipeline
        mv_run = st.button("Translate Speech", key="btn_mv_run", type="primary", use_container_width=True)

        if mv_run:
            mv_rec_text = ""
            if mv_audio is not None:
                with st.spinner("Transcribing speech with faster-whisper..."):
                    arg_asr = None
                    if m_src != "Auto Detect":
                        arg_asr = LANGUAGE_SPEECH_CAPABILITIES.get(SUPPORTED_LANGUAGES[m_src], {}).get("asr")
                    mv_rec_text, m_meta = st.session_state.asr_engine.transcribe(mv_audio, source_language_code=arg_asr)
            elif m_typed_text.strip():
                mv_rec_text = m_typed_text.strip()
            else:
                st.warning("Please record voice or type a message above.")

            if mv_rec_text.strip():
                if m_src != "Auto Detect":
                    m_src_code = SUPPORTED_LANGUAGES[m_src]
                    m_src_disp = m_src
                else:
                    det = m_meta.get("detected_language", "en") if "m_meta" in locals() else "en"
                    from model_engine.speech_recognition import WHISPER_TO_FLORES_LANG
                    m_src_code = WHISPER_TO_FLORES_LANG.get(det, "eng_Latn")
                    matched = [k for k, v in SUPPORTED_LANGUAGES.items() if v == m_src_code]
                    m_src_disp = matched[0] if matched else "English"

                with st.spinner(f"Translating ({m_src_disp} → {m_tgt})..."):
                    mv_trans = st.session_state.engine.translate(mv_rec_text, src_lang=m_src_code, tgt_lang=m_tgt_code)

                mv_audio_bytes = None
                if m_tgt_cap.get("tts"):
                    with st.spinner(f"Generating {m_tgt} audio output..."):
                        try:
                            mv_audio_bytes, _ = st.session_state.tts_engine.synthesize(mv_trans, target_lang_code=m_tgt_code)
                        except Exception as e:
                            st.caption(f"Speech synthesis notice: {e}")

                st.session_state["last_mv_result"] = {
                    "src_lang": m_src_disp,
                    "tgt_lang": m_tgt,
                    "original": mv_rec_text,
                    "translated": mv_trans,
                    "audio": mv_audio_bytes,
                    "time": time.strftime("%H:%M:%S")
                }
                st.session_state.dialogue_history.append({
                    "speaker": f"{m_src_disp}",
                    "recipient": f"{m_tgt}",
                    "original": mv_rec_text,
                    "translated": mv_trans,
                    "audio": mv_audio_bytes,
                    "time": time.strftime("%H:%M:%S")
                })
                st.rerun()

        # Display Latest Translation Result inside Mobile Card
        if "last_mv_result" in st.session_state and st.session_state["last_mv_result"]:
            res = st.session_state["last_mv_result"]
            st.markdown(f"""
            <div style="background:#142B49; border:1px solid #1E3A5F; border-radius:18px; padding:1.2rem; margin-top:1.25rem;">
                <div style="display:flex; justify-content:space-between; font-size:0.75rem; color:#94A3B8; text-transform:uppercase; margin-bottom:0.35rem;">
                    <span>{res['src_lang']}</span>
                    <span>{res['time']}</span>
                </div>
                <div style="font-size:0.95rem; color:#94A3B8; margin-bottom:0.85rem;">"{res['original']}"</div>
                
                <div style="display:flex; justify-content:space-between; font-size:0.75rem; color:#20C9B7; text-transform:uppercase; margin-bottom:0.35rem; font-weight:600;">
                    <span>{res['tgt_lang']} Translated</span>
                </div>
                <div style="font-size:1.15rem; font-weight:600; color:#F8FAFC; margin-bottom:0.85rem; line-height:1.5;">{res['translated']}</div>
            </div>
            """, unsafe_allow_html=True)
            if res.get("audio"):
                st.audio(res["audio"], format="audio/mp3")

        # Translation History List in Mobile Phone
        if st.session_state.dialogue_history:
            st.markdown('<div style="font-size:0.8rem; font-weight:600; color:#94A3B8; text-transform:uppercase; letter-spacing:0.05em; margin:1.5rem 0 0.5rem 0;">Recent Translations</div>', unsafe_allow_html=True)
            for item in reversed(st.session_state.dialogue_history[-3:]):
                st.markdown(f"""
                <div style="background:#08162D; border:1px solid #1E3A5F; border-radius:12px; padding:0.85rem 1rem; margin-bottom:0.5rem;">
                    <div style="font-size:0.72rem; color:#20C9B7; font-weight:600;">{item['speaker']} → {item['recipient']}</div>
                    <div style="font-size:0.92rem; color:#F8FAFC; margin-top:2px;">{item['translated']}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    elif voice_mode == "Two-Speaker Dialogue Console":
        # ----------------------------------------------------------------------
        # TWO-SPEAKER DIALOGUE CONSOLE (CHANNEL 1 & CHANNEL 2)
        # ----------------------------------------------------------------------
        p_col1, p_col2 = st.columns(2)
        with p_col1:
            st.markdown("""
            <div class="panel-header">
                <span class="panel-title">Speaker A Configuration</span>
                <span class="panel-tag">Channel 1</span>
            </div>
            """, unsafe_allow_html=True)
            lang_a = st.selectbox(
                "Speaker A Language",
                options=all_lang_names,
                index=all_lang_names.index("Hindi"),
                format_func=format_lang_clean,
                key="convo_lang_a",
                label_visibility="collapsed"
            )
            code_a = SUPPORTED_LANGUAGES[lang_a]
            cap_a = LANGUAGE_SPEECH_CAPABILITIES.get(code_a, {})

        with p_col2:
            st.markdown("""
            <div class="panel-header">
                <span class="panel-title">Speaker B Configuration</span>
                <span class="panel-tag">Channel 2</span>
            </div>
            """, unsafe_allow_html=True)
            lang_b = st.selectbox(
                "Speaker B Language",
                options=all_lang_names,
                index=all_lang_names.index("Tamil"),
                format_func=format_lang_clean,
                key="convo_lang_b",
                label_visibility="collapsed"
            )
            code_b = SUPPORTED_LANGUAGES[lang_b]
            cap_b = LANGUAGE_SPEECH_CAPABILITIES.get(code_b, {})

        # Speech-to-Speech Consoles
        st.write("")
        spk_col1, spk_col2 = st.columns(2)

        # Speaker A Console
        with spk_col1:
            st.markdown(f"""
            <div class="app-card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                    <div class="app-card-title">Speaker A · {lang_a}</div>
                    <span class="state-badge-active">Listening Ready</span>
                </div>
                <div class="app-card-sub">Record speech in {lang_a} to translate and speak in {lang_b}</div>
                <div class="waveform-container">
                    <div class="wave-bar"></div><div class="wave-bar"></div><div class="wave-bar"></div>
                    <div class="wave-bar"></div><div class="wave-bar"></div><div class="wave-bar"></div>
                    <div class="wave-bar"></div><div class="wave-bar"></div><div class="wave-bar"></div>
                    <div class="wave-bar"></div><div class="wave-bar"></div><div class="wave-bar"></div>
                    <div class="wave-bar"></div><div class="wave-bar"></div><div class="wave-bar"></div>
                    <div class="wave-bar"></div><div class="wave-bar"></div><div class="wave-bar"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            audio_a = st.audio_input(f"Record Voice ({lang_a})", key="mic_person_a")
            text_input_a = st.text_input(
                "Text Input (Alternative / Correction)",
                placeholder=f"Type in {lang_a}...",
                key="text_person_a",
                label_visibility="collapsed"
            )

            send_a = st.button(f"Transmit to Speaker B ({lang_b})", key="btn_send_a", type="primary", use_container_width=True)
            if send_a:
                orig_text = ""
                if audio_a is not None:
                    with st.spinner(f"Transcribing {lang_a} speech..."):
                        asr_lang = cap_a.get("asr")
                        orig_text, meta = st.session_state.asr_engine.transcribe(audio_a, source_language_code=asr_lang)
                elif text_input_a.strip():
                    orig_text = text_input_a.strip()
                else:
                    st.warning(f"Please record audio or provide text for Speaker A.")

                if orig_text.strip():
                    with st.spinner(f"Translating ({lang_a} to {lang_b})..."):
                        trans_text = st.session_state.engine.translate(orig_text, src_lang=code_a, tgt_lang=code_b)

                    audio_b_bytes = None
                    if cap_b.get("tts"):
                        with st.spinner(f"Generating audio in {lang_b}..."):
                            try:
                                audio_b_bytes, _ = st.session_state.tts_engine.synthesize(trans_text, target_lang_code=code_b)
                            except Exception as e:
                                st.caption(f"Speech synthesis notice: {e}")

                    st.session_state.dialogue_history.append({
                        "speaker": f"Speaker A ({lang_a})",
                        "recipient": f"Speaker B ({lang_b})",
                        "original": orig_text,
                        "translated": trans_text,
                        "audio": audio_b_bytes,
                        "time": time.strftime("%H:%M:%S")
                    })
                    st.rerun()

        # Speaker B Console
        with spk_col2:
            st.markdown(f"""
            <div class="app-card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                    <div class="app-card-title">Speaker B · {lang_b}</div>
                    <span class="state-badge-active">Listening Ready</span>
                </div>
                <div class="app-card-sub">Record speech in {lang_b} to translate and speak in {lang_a}</div>
                <div class="waveform-container">
                    <div class="wave-bar"></div><div class="wave-bar"></div><div class="wave-bar"></div>
                    <div class="wave-bar"></div><div class="wave-bar"></div><div class="wave-bar"></div>
                    <div class="wave-bar"></div><div class="wave-bar"></div><div class="wave-bar"></div>
                    <div class="wave-bar"></div><div class="wave-bar"></div><div class="wave-bar"></div>
                    <div class="wave-bar"></div><div class="wave-bar"></div><div class="wave-bar"></div>
                    <div class="wave-bar"></div><div class="wave-bar"></div><div class="wave-bar"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            audio_b = st.audio_input(f"Record Voice ({lang_b})", key="mic_person_b")
            text_input_b = st.text_input(
                "Text Input (Alternative / Correction)",
                placeholder=f"Type in {lang_b}...",
                key="text_person_b",
                label_visibility="collapsed"
            )

            send_b = st.button(f"Transmit to Speaker A ({lang_a})", key="btn_send_b", type="primary", use_container_width=True)
            if send_b:
                orig_text = ""
                if audio_b is not None:
                    with st.spinner(f"Transcribing {lang_b} speech..."):
                        asr_lang = cap_b.get("asr")
                        orig_text, meta = st.session_state.asr_engine.transcribe(audio_b, source_language_code=asr_lang)
                elif text_input_b.strip():
                    orig_text = text_input_b.strip()
                else:
                    st.warning(f"Please record audio or provide text for Speaker B.")

                if orig_text.strip():
                    with st.spinner(f"Translating ({lang_b} to {lang_a})..."):
                        trans_text = st.session_state.engine.translate(orig_text, src_lang=code_b, tgt_lang=code_a)

                    audio_a_bytes = None
                    if cap_a.get("tts"):
                        with st.spinner(f"Generating audio in {lang_a}..."):
                            try:
                                audio_a_bytes, _ = st.session_state.tts_engine.synthesize(trans_text, target_lang_code=code_a)
                            except Exception as e:
                                st.caption(f"Speech synthesis notice: {e}")

                    st.session_state.dialogue_history.append({
                        "speaker": f"Speaker B ({lang_b})",
                        "recipient": f"Speaker A ({lang_a})",
                        "original": orig_text,
                        "translated": trans_text,
                        "audio": audio_a_bytes,
                        "time": time.strftime("%H:%M:%S")
                    })
                    st.rerun()

        # Conversation Dialogue Stream
        st.write("")
        st.markdown("""
        <div class="panel-header" style="margin-top: 1.5rem;">
            <span class="panel-title">Conversation Stream</span>
            <span class="panel-tag">Live Dialogue</span>
        </div>
        """, unsafe_allow_html=True)

        if not st.session_state.dialogue_history:
            st.markdown("""
            <div class="output-placeholder" style="min-height: 120px;">
                <div style="color: #94A3B8; font-weight: 500;">No dialogue recorded yet</div>
                <div style="font-size: 0.82rem; color: #64748B;">Speakers can record voice or submit text above to begin the turn-by-turn conversation</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            for turn in reversed(st.session_state.dialogue_history):
                is_a = "Speaker A" in turn["speaker"]
                bubble_class = "convo-bubble-a" if is_a else "convo-bubble-b"
                acc_color = "#20C9B7" if is_a else "#38BDF8"
                
                st.markdown(f"""
                <div class="{bubble_class}">
                    <div class="bubble-meta">
                        <span style="color: {acc_color};">{turn['speaker']} → {turn['recipient']}</span>
                        <span style="color: #64748B;">{turn['time']}</span>
                    </div>
                    <div class="bubble-orig">"{turn['original']}"</div>
                    <div class="bubble-trans">→ {turn['translated']}</div>
                </div>
                """, unsafe_allow_html=True)
                if turn.get("audio"):
                    st.audio(turn["audio"], format="audio/mp3")

            st.write("")
            if st.button("Clear Conversation Dialogue", key="clear_convo_btn"):
                st.session_state.dialogue_history = []
                st.rerun()

    else:
        # Single Voice / File Translator
        st.markdown('<div class="app-card-title" style="margin-top:0.5rem;">Single-Speaker Audio File &amp; Voice Translator</div>', unsafe_allow_html=True)
        st.markdown('<div class="page-main-desc">Process recorded speech or upload existing audio files (.wav, .mp3, .m4a) for automatic speech recognition, neural translation, and synthesized speech reproduction.</div>', unsafe_allow_html=True)

        sv_col1, sv_col2 = st.columns(2)
        with sv_col1:
            st.markdown('<div class="panel-title" style="margin-bottom:6px;">Input Speech Language</div>', unsafe_allow_html=True)
            s_src_lang = st.selectbox(
                "Spoken / Input Language",
                options=["Auto Detect"] + all_lang_names,
                format_func=lambda x: "Auto Detect (Whisper Multilingual)" if x == "Auto Detect" else format_lang_clean(x),
                key="sv_src_lang",
                label_visibility="collapsed"
            )
        with sv_col2:
            st.markdown('<div class="panel-title" style="margin-bottom:6px;">Target Audio Language</div>', unsafe_allow_html=True)
            s_tgt_lang = st.selectbox(
                "Target Spoken Language",
                options=all_lang_names,
                index=all_lang_names.index("Hindi"),
                format_func=format_lang_clean,
                key="sv_tgt_lang",
                label_visibility="collapsed"
            )

        s_code_tgt = SUPPORTED_LANGUAGES[s_tgt_lang]
        s_cap_tgt = LANGUAGE_SPEECH_CAPABILITIES.get(s_code_tgt, {})

        st.write("")
        sv_mode = st.radio("Input Modality", ["Microphone Recording", "Audio File Upload"], horizontal=True)
        
        sv_audio = None
        if "Microphone" in sv_mode:
            sv_audio = st.audio_input("Record Voice", key="sv_mic")
        else:
            sv_audio = st.file_uploader("Upload Audio File", type=["wav", "mp3", "m4a", "ogg"], key="sv_file")

        if sv_audio is not None:
            st.audio(sv_audio)

        st.write("")
        if st.button("Process Speech Translation", key="btn_sv_run", type="primary"):
            if sv_audio is None:
                st.warning("Please record voice or upload an audio file first.")
            else:
                with st.spinner("Transcribing speech with faster-whisper..."):
                    whisper_arg = None
                    if s_src_lang != "Auto Detect":
                        whisper_arg = LANGUAGE_SPEECH_CAPABILITIES.get(SUPPORTED_LANGUAGES[s_src_lang], {}).get("asr")
                    txt_rec, meta = st.session_state.asr_engine.transcribe(sv_audio, source_language_code=whisper_arg)

                if not txt_rec.strip():
                    st.warning("No speech detected in audio input.")
                else:
                    if s_src_lang != "Auto Detect":
                        s_code_src = SUPPORTED_LANGUAGES[s_src_lang]
                        disp_src = s_src_lang
                    else:
                        det = meta.get("detected_language", "en")
                        from model_engine.speech_recognition import WHISPER_TO_FLORES_LANG
                        s_code_src = WHISPER_TO_FLORES_LANG.get(det, "eng_Latn")
                        matched = [k for k, v in SUPPORTED_LANGUAGES.items() if v == s_code_src]
                        disp_src = matched[0] if matched else "English"

                    with st.spinner(f"Translating ({disp_src} → {s_tgt_lang})..."):
                        txt_trans = st.session_state.engine.translate(txt_rec, src_lang=s_code_src, tgt_lang=s_code_tgt)

                    voice_bytes = None
                    if s_cap_tgt.get("tts"):
                        with st.spinner(f"Generating spoken audio for {s_tgt_lang}..."):
                            try:
                                voice_bytes, _ = st.session_state.tts_engine.synthesize(txt_trans, target_lang_code=s_code_tgt)
                            except Exception as e:
                                st.caption(f"Speech synthesis notice: {e}")

                    st.write("")
                    o_c1, o_c2 = st.columns(2)
                    with o_c1:
                        st.markdown("""
                        <div class="panel-header">
                            <span class="panel-title">Transcribed Speech</span>
                        </div>
                        """, unsafe_allow_html=True)
                        st.markdown(f'<div class="output-box" style="min-height:120px;">{txt_rec}</div>', unsafe_allow_html=True)
                        st.caption(f"Detected: {meta.get('detected_language')} | Duration: {meta.get('duration')}s")
                    with o_c2:
                        st.markdown(f"""
                        <div class="panel-header">
                            <span class="panel-title">Translated Output ({s_tgt_lang})</span>
                        </div>
                        """, unsafe_allow_html=True)
                        st.markdown(f'<div class="output-box" style="min-height:120px;">{txt_trans}</div>', unsafe_allow_html=True)
                        if voice_bytes:
                            st.audio(voice_bytes, format="audio/mp3")
                        else:
                            st.caption(f"Speech output is unavailable for {s_tgt_lang}.")


# ==============================================================================
# VIEW 3: QUALITY BENCHMARKS & SACREBLEU EVALUATION
# ==============================================================================
elif nav_choice == "Quality Benchmarks":
    st.markdown("""
    <div class="top-breadcrumb-bar">
        <span class="top-breadcrumb">IndicAI Pro &gt; Analytics &gt; Quality Benchmarks</span>
        <span class="breadcrumb-pill">Protected Preview</span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<h1 class="page-main-title">Quantitative Quality Benchmarks</h1>', unsafe_allow_html=True)
    st.markdown('<div class="page-main-desc">Automated evaluation metrics using SacreBLEU and chrF++ across benchmark translation pairs on AI4Bharat IndicTrans2 architectures.</div>', unsafe_allow_html=True)

    df_saved = get_saved_benchmark()
    if not df_saved.empty:
        avg_bleu = df_saved["BLEU"].mean()
        avg_chrf = df_saved["chrF++"].mean()
        avg_time = df_saved["Latency (s)"].mean()

        st.markdown(f"""
        <div class="kpi-grid">
            <div class="kpi-card">
                <div class="kpi-header">
                    <span class="kpi-label">Average SacreBLEU</span>
                    <span class="kpi-badge">↗ High Quality</span>
                </div>
                <div class="kpi-value">{avg_bleu:.1f}</div>
                <div class="kpi-subtext">Word n-gram precision metric (0-100)</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-header">
                    <span class="kpi-label">Average chrF++</span>
                    <span class="kpi-badge">↗ Morphology Aware</span>
                </div>
                <div class="kpi-value">{avg_chrf:.1f}</div>
                <div class="kpi-subtext">Character n-gram score for agglutinative languages</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-header">
                    <span class="kpi-label">Inference Latency</span>
                    <span class="kpi-badge">CPU Float32</span>
                </div>
                <div class="kpi-value">{avg_time:.2f}s</div>
                <div class="kpi-subtext">Average end-to-end translation latency</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-header">
                    <span class="kpi-label">Total Matrix Pairs</span>
                    <span class="kpi-badge">23x23 Active</span>
                </div>
                <div class="kpi-value">506</div>
                <div class="kpi-subtext">Complete coverage of 22 languages + English</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="panel-header">
            <span class="panel-title">Standardized Evaluation Set (7 Language Pairs)</span>
            <span class="panel-tag">IndicTrans2 Benchmark</span>
        </div>
        """, unsafe_allow_html=True)
        st.dataframe(
            df_saved[["Pair", "Model Used", "Input Text", "Human Reference", "Model Prediction", "BLEU", "chrF++", "Latency (s)"]],
            use_container_width=True
        )

    st.write("")
    st.markdown('<div class="app-card-title" style="margin-top: 1rem;">Metric Calculation Sandbox</div>', unsafe_allow_html=True)
    st.markdown('<div class="app-card-sub">Evaluate arbitrary candidate translation output against human reference sentences in real time.</div>', unsafe_allow_html=True)
    
    ev1, ev2 = st.columns(2)
    with ev1:
        test_cand = st.text_area("Hypothesis / Candidate Text", "कृत्रिम बुद्धिमत्ता आधुनिक सॉफ्टवेयर विकास को गति देती है।", height=90)
    with ev2:
        test_ref = st.text_area("Reference / Ground Truth Text", "आर्टिफिशियल इंटेलिजेंस आधुनिक सॉफ्टवेयर विकास को गति प्रदान करता है।", height=90)

    if st.button("Compute SacreBLEU & chrF++", type="primary"):
        scores = compute_metrics(test_cand, test_ref)
        m1, m2 = st.columns(2)
        m1.metric("SacreBLEU Score", f"{scores['bleu']} / 100")
        m2.metric("chrF++ Score", f"{scores['chrf_plus_plus']} / 100")


# ==============================================================================
# VIEW 4: SESSION HISTORY
# ==============================================================================
elif nav_choice == "History":
    st.markdown("""
    <div class="top-breadcrumb-bar">
        <span class="top-breadcrumb">IndicAI Pro &gt; Data &gt; Session History</span>
        <span class="breadcrumb-pill">Protected Preview</span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<h1 class="page-main-title">Translation Session History</h1>', unsafe_allow_html=True)
    st.markdown('<div class="page-main-desc">Complete chronological record of all text translations performed during the current session, including latency and neural model routing metadata.</div>', unsafe_allow_html=True)

    if not st.session_state.history:
        st.markdown("""
        <div class="output-placeholder">
            <div style="color: #94A3B8; font-weight: 500;">Session history is currently empty</div>
            <div style="font-size: 0.82rem; color: #64748B;">Perform translations in the workspace to accumulate exportable query logs</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        df_h = pd.DataFrame(st.session_state.history)
        st.dataframe(df_h, use_container_width=True)

        st.write("")
        h1, h2 = st.columns(2)
        with h1:
            csv_bytes = df_h.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Export History (.csv)",
                data=csv_bytes,
                file_name="indicai_translation_history.csv",
                mime="text/csv",
                use_container_width=True
            )
        with h2:
            if st.button("Clear Session History", use_container_width=True):
                st.session_state.history = []
                st.rerun()


# ==============================================================================
# VIEW 5: SETTINGS & SYSTEM CONFIGURATION
# ==============================================================================
elif nav_choice == "Settings":
    st.markdown("""
    <div class="top-breadcrumb-bar">
        <span class="top-breadcrumb">IndicAI Pro &gt; System &gt; Settings</span>
        <span class="breadcrumb-pill">Protected Preview</span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<h1 class="page-main-title">System Settings &amp; Model Engine</h1>', unsafe_allow_html=True)
    st.markdown('<div class="page-main-desc">Configure authentication credentials, verify runtime hardware acceleration, and inspect cached transformer model weights.</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="app-card">
        <div class="app-card-title">Hugging Face Access Token</div>
        <div class="app-card-sub">Required if accessing gated model repositories or running in cloud environments with rate limits</div>
    </div>
    """, unsafe_allow_html=True)

    hf_tok_in = st.text_input(
        "Hugging Face Token",
        value=st.session_state.hf_token,
        type="password",
        placeholder="hf_xxxxxxxxxxxxxxxxxxxxxx",
        key="hf_tok_input"
    )
    if st.button("Save Access Token", type="primary"):
        st.session_state.hf_token = hf_tok_in.strip()
        st.session_state.engine = TranslationEngine(hf_token=st.session_state.hf_token)
        st.success("Token saved and Translation Engine updated.")

    st.write("")
    st.markdown("""
    <div class="app-card">
        <div class="app-card-title">Engine Runtime Specifications</div>
        <div class="app-card-sub">Active neural translation and speech pipeline configuration</div>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; margin-top: 1rem;">
            <div style="background:#0E223D; padding:12px; border-radius:8px; border:1px solid #1E3A5F;">
                <div style="font-size:0.75rem; color:#64748B; text-transform:uppercase;">NMT Architecture</div>
                <div style="font-size:0.95rem; font-weight:600; color:#F8FAFC; margin-top:4px;">IndicTrans2 (Dense)</div>
            </div>
            <div style="background:#0E223D; padding:12px; border-radius:8px; border:1px solid #1E3A5F;">
                <div style="font-size:0.75rem; color:#64748B; text-transform:uppercase;">Execution Device</div>
                <div style="font-size:0.95rem; font-weight:600; color:#20C9B7; margin-top:4px;">CPU (Float32 Precision)</div>
            </div>
            <div style="background:#0E223D; padding:12px; border-radius:8px; border:1px solid #1E3A5F;">
                <div style="font-size:0.75rem; color:#64748B; text-transform:uppercase;">Speech Recognition</div>
                <div style="font-size:0.95rem; font-weight:600; color:#F8FAFC; margin-top:4px;">faster-whisper (int8)</div>
            </div>
            <div style="background:#0E223D; padding:12px; border-radius:8px; border:1px solid #1E3A5F;">
                <div style="font-size:0.75rem; color:#64748B; text-transform:uppercase;">Speech Synthesis</div>
                <div style="font-size:0.95rem; font-weight:600; color:#F8FAFC; margin-top:4px;">Indic-TTS Engine</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
