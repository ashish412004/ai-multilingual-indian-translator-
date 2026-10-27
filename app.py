"""
IndicAI Pro - AI-Based Multilingual Indian Language Translator
Ultra-Modern, Production-Grade SaaS Interface
"""

import os
import time
import pandas as pd
import streamlit as st

# Configure Layout & Meta
st.set_page_config(
    page_title="IndicAI Pro — Neural Machine Translation",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Ultra-Modern SaaS CSS (DeepL / Linear / Vercel Aesthetic)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');
    
    /* Global Reset & Typography */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
        color: #0f172a;
        background-color: #f8fafc;
    }

    #MainMenu, footer, header {visibility: hidden !important;}
    .block-container {
        padding-top: 1.25rem !important;
        padding-bottom: 2.5rem !important;
        max-width: 1280px !important;
    }

    /* Top Global Navbar */
    .saas-navbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0.85rem 1.5rem;
        background: rgba(255, 255, 255, 0.85);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(226, 232, 240, 0.8);
        border-radius: 16px;
        margin-bottom: 1.75rem;
        box-shadow: 0 4px 20px -2px rgba(15, 23, 42, 0.04);
    }
    .brand-group {
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .brand-logo {
        width: 36px;
        height: 36px;
        border-radius: 10px;
        background: linear-gradient(135deg, #2563eb, #7c3aed);
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: 800;
        font-size: 1.1rem;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
    }
    .brand-name {
        font-size: 1.2rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        background: linear-gradient(135deg, #0f172a 0%, #334155 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .brand-tag {
        font-size: 0.72rem;
        font-weight: 700;
        background: #ede9fe;
        color: #6d28d9;
        padding: 2px 8px;
        border-radius: 9999px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-size: 0.8rem;
        font-weight: 600;
        background: #ecfdf5;
        color: #059669;
        border: 1px solid #a7f3d0;
        padding: 4px 12px;
        border-radius: 9999px;
    }
    .status-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background-color: #10b981;
        box-shadow: 0 0 0 2px rgba(16, 185, 129, 0.2);
    }

    /* Main Dual-Pane Workspace Card */
    .workspace-wrapper {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 20px;
        box-shadow: 0 10px 30px -5px rgba(15, 23, 42, 0.05), 0 20px 25px -5px rgba(15, 23, 42, 0.02);
        overflow: hidden;
        margin-bottom: 1.5rem;
    }

    /* Workspace Top Toolbar */
    .workspace-toolbar {
        background: #f8fafc;
        border-bottom: 1px solid #e2e8f0;
        padding: 0.85rem 1.25rem;
    }

    /* Output Card */
    .output-container {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 1.2rem;
        min-height: 240px;
        font-size: 1.15rem;
        line-height: 1.75;
        color: #0f172a;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.02);
    }
    .output-empty {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        min-height: 240px;
        color: #94a3b8;
        background: #f8fafc;
        border: 1px dashed #cbd5e1;
        border-radius: 14px;
        gap: 8px;
    }

    /* Preset Chips */
    .preset-pill {
        display: inline-block;
        background: #f1f5f9;
        color: #475569;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 6px 14px;
        font-size: 0.85rem;
        font-weight: 500;
        cursor: pointer;
        transition: all 0.2s ease;
    }

    /* Custom Streamlit Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #f1f5f9;
        padding: 5px;
        border-radius: 14px;
        border: 1px solid #e2e8f0;
        margin-bottom: 1.5rem;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 8px 20px;
        font-weight: 600;
        font-size: 0.92rem;
        color: #64748b;
        border: none !important;
        background-color: transparent !important;
        transition: all 0.2s ease;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ffffff !important;
        color: #2563eb !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06) !important;
    }

    /* Stats & Benchmark Cards */
    .kpi-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 1.25rem 1.5rem;
        box-shadow: 0 4px 15px -2px rgba(15, 23, 42, 0.04);
        position: relative;
        overflow: hidden;
    }
    .kpi-card::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 4px;
        height: 100%;
        background: linear-gradient(180deg, #2563eb, #7c3aed);
    }
    .kpi-title {
        font-size: 0.85rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .kpi-val {
        font-size: 2rem;
        font-weight: 800;
        color: #0f172a;
        margin-top: 0.25rem;
        letter-spacing: -0.03em;
    }
    .kpi-sub {
        font-size: 0.8rem;
        color: #10b981;
        font-weight: 600;
        margin-top: 0.2rem;
    }

    /* Technical Badge */
    .tech-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        color: #475569;
        padding: 3px 10px;
        border-radius: 6px;
    }
</style>
""", unsafe_allow_html=True)

# Imports from modular engine
from model_engine.config import (
    SUPPORTED_LANGUAGES,
    LANGUAGE_NATIVE_NAMES,
    get_model_for_pair,
)
from model_engine.translator import TranslationEngine
from evaluation.evaluator import compute_metrics
from evaluation.test_benchmark import get_saved_benchmark

# Initialize Session State
if "history" not in st.session_state:
    st.session_state.history = []

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

if "src_lang_select" not in st.session_state:
    st.session_state["src_lang_select"] = "English"

if "tgt_lang_select" not in st.session_state:
    st.session_state["tgt_lang_select"] = "Hindi"

if "main_input_area" not in st.session_state:
    st.session_state["main_input_area"] = ""

if "output_text" not in st.session_state:
    st.session_state.output_text = ""

if "last_latency" not in st.session_state:
    st.session_state.last_latency = 0.0

# Callbacks for reliable UI state manipulation
def do_swap():
    cur_src = st.session_state.get("src_lang_select", "English")
    cur_tgt = st.session_state.get("tgt_lang_select", "Hindi")
    st.session_state["src_lang_select"] = cur_tgt
    st.session_state["tgt_lang_select"] = cur_src

    cur_out = st.session_state.get("output_text", "")
    if cur_out.strip():
        cur_in = st.session_state.get("main_input_area", "")
        st.session_state["main_input_area"] = cur_out
        st.session_state["output_text"] = cur_in

def set_preset(text):
    st.session_state["main_input_area"] = text

def do_clear():
    st.session_state["main_input_area"] = ""
    st.session_state["output_text"] = ""

# Presets for Quick Testing
PRESETS = [
    ("Artificial Intelligence", "Artificial intelligence is transforming education and governance across India."),
    ("Digital Development", "Education and technology play a vital role in national development."),
    ("Multilingual NLP", "Neural machine translation bridges communication barriers between languages."),
    ("Student Innovation", "Students should study innovative technology and develop their skills."),
]

# Top Global Navbar
st.markdown("""
<div class="saas-navbar">
    <div class="brand-group">
        <div class="brand-logo">IA</div>
        <div>
            <div style="display:flex; align-items:center; gap:8px;">
                <span class="brand-name">IndicAI Pro</span>
                <span class="brand-tag">Neural MT v2.0</span>
            </div>
        </div>
    </div>
    <div style="display: flex; align-items: center; gap: 14px;">
        <div class="status-pill">
            <span class="status-dot"></span>
            <span>AI4Bharat IndicTrans2 Ready</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Main Navigation Tabs
tab_translate, tab_benchmarks, tab_history = st.tabs([
    "✨ Neural Translator",
    "📈 Quality Benchmarks & BLEU",
    "📋 Session History"
])

# ==============================================================================
# TAB 1: NEURAL TRANSLATOR (WORLD-CLASS INTERFACE)
# ==============================================================================
with tab_translate:
    lang_names = list(SUPPORTED_LANGUAGES.keys())

    # Language Selector Ribbon
    with st.container():
        ctrl_col1, ctrl_swap, ctrl_col2, ctrl_meta = st.columns([4, 1, 4, 3])

        with ctrl_col1:
            src_choice = st.selectbox(
                "Source Language",
                options=lang_names,
                format_func=lambda x: LANGUAGE_NATIVE_NAMES.get(x, x),
                key="src_lang_select"
            )

        with ctrl_swap:
            st.write("")
            st.write("")
            st.button("⇄", help="Swap Source and Target Languages", on_click=do_swap, use_container_width=True)

        with ctrl_col2:
            tgt_choice = st.selectbox(
                "Target Language",
                options=lang_names,
                format_func=lambda x: LANGUAGE_NATIVE_NAMES.get(x, x),
                key="tgt_lang_select"
            )

        src_code = SUPPORTED_LANGUAGES[src_choice]
        tgt_code = SUPPORTED_LANGUAGES[tgt_choice]
        routed_model = get_model_for_pair(src_code, tgt_code)

        with ctrl_meta:
            st.write("")
            st.write("")
            if routed_model == "IDENTITY":
                st.markdown('<div class="tech-pill" style="color:#0284c7; background:#e0f2fe; border-color:#bae6fd;">⚡ Direct Same-Language Pass</div>', unsafe_allow_html=True)
            else:
                is_loaded = st.session_state.engine.is_model_loaded(routed_model)
                badge_style = "color:#15803d; background:#dcfce7; border-color:#bbf7d0;" if is_loaded else "color:#b45309; background:#fef3c7; border-color:#fde68a;"
                badge_lbl = "● In-Memory Cache" if is_loaded else "○ Auto-Load on Run"
                st.markdown(f'<div class="tech-pill" style="{badge_style}"><b>{badge_lbl}</b></div>', unsafe_allow_html=True)

    # Quick One-Click Presets Ribbon
    st.write("")
    p_cols = st.columns([1.2] + [2.5] * len(PRESETS))
    with p_cols[0]:
        st.markdown("<p style='font-size:0.85rem; font-weight:700; color:#64748b; margin-top:8px;'>💡 Try Sample:</p>", unsafe_allow_html=True)
    for idx, (label, text) in enumerate(PRESETS):
        with p_cols[idx + 1]:
            st.button(f"✨ {label}", key=f"sample_{idx}", on_click=set_preset, args=(text,), use_container_width=True)

    st.write("")

    # Unified Translation Workspace Card
    pane_col1, pane_col2 = st.columns(2)

    with pane_col1:
        st.markdown(f"**From:** `{src_choice}` ({src_code})")
        user_input = st.text_area(
            "Source Text Input",
            height=260,
            placeholder=f"Enter or paste text in {src_choice} to translate...",
            label_visibility="collapsed",
            key="main_input_area"
        )
        
        # Meta footer under input
        chars = len(user_input)
        words = len(user_input.split()) if user_input.strip() else 0
        foot_in1, foot_in2 = st.columns([3, 1])
        with foot_in1:
            st.markdown(f'<span class="tech-pill">Chars: {chars}</span> <span class="tech-pill">Words: {words}</span>', unsafe_allow_html=True)
        with foot_in2:
            st.button("🧹 Clear", on_click=do_clear, use_container_width=True)

    with pane_col2:
        st.markdown(f"**To:** `{tgt_choice}` ({tgt_code})")
        
        if st.session_state.output_text:
            st.markdown(f'<div class="output-container">{st.session_state.output_text}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="output-empty">
                <span style="font-size: 1.8rem;">🌐</span>
                <span style="font-weight: 500;">Translation will appear here instantly</span>
                <span style="font-size: 0.8rem; color:#94a3b8;">Select language and click 'Translate'</span>
            </div>
            """, unsafe_allow_html=True)

        # Meta footer under output
        foot_out1, foot_out2, foot_out3 = st.columns([2, 1, 1])
        with foot_out1:
            if st.session_state.last_latency > 0:
                st.markdown(f'<span class="tech-pill" style="color:#2563eb; background:#eff6ff;">⚡ {st.session_state.last_latency:.2f}s latency</span>', unsafe_allow_html=True)
        with foot_out2:
            if st.session_state.output_text:
                st.download_button(
                    "📥 TXT",
                    data=st.session_state.output_text,
                    file_name=f"indic_translation_{src_code}_to_{tgt_code}.txt",
                    mime="text/plain",
                    use_container_width=True
                )
        with foot_out3:
            if st.session_state.output_text:
                with st.popover("📋 Copy"):
                    st.code(st.session_state.output_text, language="")

    # Action Banner
    st.write("")
    act_col1, act_center, act_col2 = st.columns([1, 2, 1])
    with act_center:
        run_btn = st.button("🚀 Translate Text", type="primary", use_container_width=True)

    if run_btn:
        if not user_input.strip():
            st.warning("⚠️ Please enter or paste some text to translate.")
        else:
            with st.spinner(f"Translating to {tgt_choice} via {routed_model.split('/')[-1]}..."):
                t0 = time.time()
                try:
                    res = st.session_state.engine.translate(
                        user_input,
                        src_lang=src_code,
                        tgt_lang=tgt_code,
                    )
                    lat = time.time() - t0
                    st.session_state.output_text = res
                    st.session_state.last_latency = lat

                    st.session_state.history.append({
                        "Time": time.strftime("%H:%M:%S"),
                        "Source": src_choice,
                        "Target": tgt_choice,
                        "Input": user_input,
                        "Translated": res,
                        "Model": routed_model.split("/")[-1],
                        "Latency (s)": round(lat, 2)
                    })
                    st.rerun()

                except PermissionError as pe:
                    st.error("🔒 Hugging Face Authentication Required")
                    st.info(f"{pe}")
                except Exception as ex:
                    st.error(f"Inference error: {ex}")

    # Subtle Settings Drawer at Bottom
    st.write("")
    with st.expander("⚙️ System Engine & Hugging Face Authentication"):
        sc1, sc2 = st.columns([3, 1])
        with sc1:
            new_tok = st.text_input(
                "Hugging Face User Access Token",
                value=st.session_state.hf_token,
                type="password",
                help="Enter token if models require gated access on Hugging Face."
            )
        with sc2:
            st.write("")
            st.write("")
            if st.button("Save Token"):
                st.session_state.hf_token = new_tok.strip()
                st.session_state.engine.set_hf_token(st.session_state.hf_token)
                st.success("Token saved!")
        st.caption(f"Backend Hardware: **{st.session_state.engine.device}** | Torch Dtype: **{st.session_state.engine.torch_dtype}** | Routing Model: **{routed_model}**")


# ==============================================================================
# TAB 2: QUALITY BENCHMARKS & EVALUATION
# ==============================================================================
with tab_benchmarks:
    st.markdown("### 📈 SacreBLEU & chrF++ Quantitative Evaluation")
    st.markdown("Real computed metrics across the 7 required translation pairs using AI4Bharat IndicTrans2 models:")

    df_saved = get_saved_benchmark()
    if not df_saved.empty:
        b1, b2, b3 = st.columns(3)
        with b1:
            avg_bleu = df_saved["BLEU"].mean()
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Average SacreBLEU</div>
                <div class="kpi-val">{avg_bleu:.1f}</div>
                <div class="kpi-sub">✓ Word n-gram Precision (0-100)</div>
            </div>
            """, unsafe_allow_html=True)
        with b2:
            avg_chrf = df_saved["chrF++"].mean()
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Average chrF++</div>
                <div class="kpi-val">{avg_chrf:.1f}</div>
                <div class="kpi-sub">✓ Morphology-Aware Indic Metric</div>
            </div>
            """, unsafe_allow_html=True)
        with b3:
            avg_time = df_saved["Latency (s)"].mean()
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Average Latency</div>
                <div class="kpi-val">{avg_time:.2f}s</div>
                <div class="kpi-sub">✓ CPU Inference Speed</div>
            </div>
            """, unsafe_allow_html=True)

        st.write("")
        st.dataframe(
            df_saved[["Pair", "Model Used", "Input Text", "Human Reference", "Model Prediction", "BLEU", "chrF++", "Latency (s)"]],
            use_container_width=True
        )

    st.markdown("---")
    st.subheader("🧪 Live Metric Evaluation Sandbox")
    st.caption("Compute SacreBLEU and chrF++ scores on custom candidate and reference pairs:")

    ev1, ev2 = st.columns(2)
    with ev1:
        test_cand = st.text_area("Model Output / Candidate", "कृत्रिम बुद्धिमत्ता आधुनिक सॉफ्टवेयर विकास को गति देती है।", height=90)
    with ev2:
        test_ref = st.text_area("Ground Truth Reference", "आर्टिफिशियल इंटेलिजेंस आधुनिक सॉफ्टवेयर विकास को गति प्रदान करता है।", height=90)

    if st.button("📊 Calculate Metrics", type="primary"):
        scores = compute_metrics(test_cand, test_ref)
        m1, m2 = st.columns(2)
        m1.metric("BLEU Score", f"{scores['bleu']} / 100")
        m2.metric("chrF++ Score", f"{scores['chrf_plus_plus']} / 100")


# ==============================================================================
# TAB 3: SESSION HISTORY
# ==============================================================================
with tab_history:
    st.markdown("### 📋 Translation Session History")

    if not st.session_state.history:
        st.info("No translations performed in this session yet.")
    else:
        df_h = pd.DataFrame(st.session_state.history)
        st.dataframe(df_h, use_container_width=True)

        h1, h2 = st.columns(2)
        with h1:
            csv_bytes = df_h.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Export History as CSV",
                data=csv_bytes,
                file_name="indicai_translation_history.csv",
                mime="text/csv",
                use_container_width=True
            )
        with h2:
            if st.button("🗑️ Clear History", use_container_width=True):
                st.session_state.history = []
                st.rerun()
