"""
AI-Based Multilingual Indian Language Translator Using NLP
Modern, High-End Streamlit Application
"""

import os
import time
import pandas as pd
import streamlit as st

# Configure layout and favicon
st.set_page_config(
    page_title="IndicAI Translator | Neural MT",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom High-End SaaS / DeepL-style Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
        color: #0f172a;
    }
    
    /* Hide default Streamlit clutter */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1250px;
    }

    /* Top Brand Hero Banner */
    .hero-container {
        text-align: center;
        margin-bottom: 2rem;
    }
    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(99, 102, 241, 0.08);
        border: 1px solid rgba(99, 102, 241, 0.2);
        color: #4f46e5;
        padding: 6px 16px;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
        letter-spacing: 0.02em;
        margin-bottom: 0.8rem;
    }
    .hero-title {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #1e293b 0%, #3b82f6 50%, #6366f1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        line-height: 1.2;
    }
    .hero-subtitle {
        color: #64748b;
        font-size: 1.05rem;
        margin-top: 0.5rem;
        font-weight: 400;
    }

    /* Translator Box Container */
    .translator-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 20px;
        box-shadow: 0 12px 36px -8px rgba(15, 23, 42, 0.08);
        overflow: hidden;
        margin-bottom: 2rem;
    }
    
    /* Control Bar */
    .control-bar {
        background: #f8fafc;
        border-bottom: 1px solid #e2e8f0;
        padding: 1rem 1.5rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }

    /* Output Box */
    .output-area {
        background: #ffffff;
        border: 1px solid #cbd5e1;
        border-radius: 12px;
        padding: 1.25rem;
        min-height: 220px;
        font-size: 1.15rem;
        line-height: 1.7;
        color: #1e293b;
        font-weight: 400;
        white-space: pre-wrap;
    }
    .output-placeholder {
        color: #94a3b8;
        font-size: 1.1rem;
        display: flex;
        align-items: center;
        justify-content: center;
        min-height: 220px;
        background: #f8fafc;
        border: 1px dashed #cbd5e1;
        border-radius: 12px;
    }

    /* Meta Pills & Badges */
    .meta-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: #f1f5f9;
        color: #475569;
        padding: 4px 10px;
        border-radius: 8px;
        font-size: 0.8rem;
        font-weight: 500;
        font-family: 'JetBrains Mono', monospace;
    }
    .model-active-pill {
        background: #eef2ff;
        color: #4338ca;
        border: 1px solid #c7d2fe;
        padding: 4px 12px;
        border-radius: 8px;
        font-size: 0.82rem;
        font-weight: 600;
    }

    /* Preset Chips */
    .preset-label {
        font-size: 0.85rem;
        font-weight: 600;
        color: #64748b;
        margin-bottom: 0.4rem;
    }

    /* Metric Stat Card */
    .stat-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 1.25rem;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.03);
    }
    .stat-val {
        font-size: 1.8rem;
        font-weight: 700;
        color: #4f46e5;
    }
    .stat-lbl {
        font-size: 0.85rem;
        color: #64748b;
        margin-top: 0.2rem;
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
from evaluation.test_benchmark import BENCHMARK_PAIRS, get_saved_benchmark

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

# Callbacks for reliable state mutations
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

# Quick Example Presets
PRESETS = [
    "Artificial intelligence is transforming education and governance across India.",
    "Welcome to the next-generation multilingual neural translation portal.",
    "Education and technology play a vital role in national development.",
    "Machine learning models empower regional Indian languages with high accuracy.",
]

# Hero Header
st.markdown("""
<div class="hero-container">
    <div class="hero-badge">
        <span>⚡ Powered by AI4Bharat IndicTrans2</span> • <span>Transformer Seq2Seq</span>
    </div>
    <h1 class="hero-title">Multilingual Indian Language Translator</h1>
    <p class="hero-subtitle">High-fidelity neural machine translation across English and 9 major Indian languages</p>
</div>
""", unsafe_allow_html=True)

# Top Tab Navigation (Clean & Modern)
tab_translator, tab_benchmark, tab_history = st.tabs([
    "🚀 Instant Translator",
    "📊 Model Benchmarks & Metrics",
    "📜 Session History",
])

# ==============================================================================
# TAB 1: INSTANT TRANSLATOR
# ==============================================================================
with tab_translator:
    lang_names = list(SUPPORTED_LANGUAGES.keys())

    # Language Selector Header
    sel_col1, sel_swap, sel_col2, sel_status = st.columns([4, 1, 4, 3])

    with sel_col1:
        src_choice = st.selectbox(
            "Source Language",
            options=lang_names,
            format_func=lambda x: LANGUAGE_NATIVE_NAMES.get(x, x),
            key="src_lang_select"
        )

    with sel_swap:
        st.write("")
        st.write("")
        st.button("⇄", help="Swap Languages", on_click=do_swap, use_container_width=True)

    with sel_col2:
        tgt_choice = st.selectbox(
            "Target Language",
            options=lang_names,
            format_func=lambda x: LANGUAGE_NATIVE_NAMES.get(x, x),
            key="tgt_lang_select"
        )

    src_code = SUPPORTED_LANGUAGES[src_choice]
    tgt_code = SUPPORTED_LANGUAGES[tgt_choice]
    routed_model = get_model_for_pair(src_code, tgt_code)

    with sel_status:
        st.write("")
        st.write("")
        if routed_model == "IDENTITY":
            st.markdown('<div class="model-active-pill">⚡ Direct Pass (Same Language)</div>', unsafe_allow_html=True)
        else:
            is_loaded = st.session_state.engine.is_model_loaded(routed_model)
            badge_text = "🟢 In-Memory Cached" if is_loaded else "🟡 Ready on Demand"
            st.markdown(f'<div class="model-active-pill">{badge_text}</div>', unsafe_allow_html=True)

    # Preset Quick-Click Chips
    st.markdown('<div class="preset-label">💡 Quick Example Sentences:</div>', unsafe_allow_html=True)
    chip_cols = st.columns(len(PRESETS))
    for i, p in enumerate(PRESETS):
        with chip_cols[i]:
            st.button(f'"{p[:34]}..."', key=f"chip_{i}", on_click=set_preset, args=(p,), use_container_width=True)

    st.write("")

    # Dual Translation Workstation
    work_col1, work_col2 = st.columns(2)

    with work_col1:
        st.markdown(f"**{src_choice}** (`{src_code}`)")
        user_input = st.text_area(
            "Source Text",
            height=240,
            placeholder=f"Type or paste text in {src_choice} here...",
            label_visibility="collapsed",
            key="main_input_area"
        )

        # Bottom Bar under Input
        c_count = len(user_input)
        w_count = len(user_input.split()) if user_input.strip() else 0
        
        in_b1, in_b2 = st.columns([2, 1])
        with in_b1:
            st.markdown(f'<span class="meta-pill">Characters: {c_count}</span> <span class="meta-pill">Words: {w_count}</span>', unsafe_allow_html=True)
        with in_b2:
            st.button("🧹 Clear", on_click=do_clear, use_container_width=True)

    with work_col2:
        st.markdown(f"**{tgt_choice}** (`{tgt_code}`)")
        
        # Display Output Box
        if st.session_state.output_text:
            st.markdown(f'<div class="output-area">{st.session_state.output_text}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="output-placeholder">Translation in {tgt_choice} will appear here...</div>', unsafe_allow_html=True)

        # Bottom Bar under Output
        out_b1, out_b2, out_b3 = st.columns([2, 1, 1])
        with out_b1:
            if st.session_state.last_latency > 0:
                st.markdown(f'<span class="meta-pill">⚡ Latency: {st.session_state.last_latency:.2f}s</span>', unsafe_allow_html=True)
        with out_b2:
            if st.session_state.output_text:
                st.download_button(
                    "📥 TXT",
                    data=st.session_state.output_text,
                    file_name=f"translation_{src_code}_to_{tgt_code}.txt",
                    mime="text/plain",
                    use_container_width=True
                )
        with out_b3:
            if st.session_state.output_text:
                with st.popover("📋 Copy"):
                    st.code(st.session_state.output_text, language="")

    # Large Hero Translate Action Button
    st.write("")
    btn_l, btn_center, btn_r = st.columns([1, 2, 1])
    with btn_center:
        translate_clicked = st.button("🚀 Translate Now", type="primary", use_container_width=True)

    if translate_clicked:
        if not user_input.strip():
            st.warning("⚠️ Please enter text to translate.")
        else:
            with st.spinner(f"Translating to {tgt_choice} via IndicTrans2..."):
                t_start = time.time()
                try:
                    res = st.session_state.engine.translate(
                        user_input,
                        src_lang=src_code,
                        tgt_lang=tgt_code,
                    )
                    t_elapsed = time.time() - t_start
                    st.session_state.output_text = res
                    st.session_state.last_latency = t_elapsed

                    # Log to History
                    st.session_state.history.append({
                        "Time": time.strftime("%H:%M:%S"),
                        "Source": src_choice,
                        "Target": tgt_choice,
                        "Input": user_input,
                        "Translated": res,
                        "Model": routed_model.split("/")[-1],
                        "Latency (s)": round(t_elapsed, 2)
                    })
                    st.rerun()

                except PermissionError as pe:
                    st.error("🔒 Hugging Face Authentication Required")
                    st.info(f"{pe}")
                except Exception as ex:
                    st.error(f"Error during translation: {ex}")

    # Collapsible Model Architecture & Settings Drawer
    st.write("")
    with st.expander("⚙️ System Settings & Hugging Face Access Token"):
        token_col1, token_col2 = st.columns([3, 1])
        with token_col1:
            token_val = st.text_input(
                "Hugging Face Access Token",
                value=st.session_state.hf_token,
                type="password",
                help="Gated AI4Bharat models unlock automatically when authenticated."
            )
        with token_col2:
            st.write("")
            st.write("")
            if st.button("Update Token"):
                st.session_state.hf_token = token_val.strip()
                st.session_state.engine.set_hf_token(st.session_state.hf_token)
                st.success("Token updated!")

        st.caption(f"Hardware Compute: **{st.session_state.engine.device}** | PyTorch Type: **{st.session_state.engine.torch_dtype}** | Active Model: **{routed_model}**")


# ==============================================================================
# TAB 2: MODEL BENCHMARKS & EVALUATION
# ==============================================================================
with tab_benchmark:
    st.markdown("### 📊 SacreBLEU & chrF++ Empirical Benchmarks")
    st.markdown("Actual translation accuracy evaluated on the standard 7-pair test suite using genuine pretrained models:")

    # Top Metric Summary Cards
    df_saved = get_saved_benchmark()
    if not df_saved.empty:
        stat1, stat2, stat3 = st.columns(3)
        with stat1:
            avg_bleu = df_saved["BLEU"].mean()
            st.markdown(f'<div class="stat-card"><div class="stat-val">{avg_bleu:.1f}</div><div class="stat-lbl">Average SacreBLEU Score (0-100)</div></div>', unsafe_allow_html=True)
        with stat2:
            avg_chrf = df_saved["chrF++"].mean()
            st.markdown(f'<div class="stat-card"><div class="stat-val">{avg_chrf:.1f}</div><div class="stat-lbl">Average chrF++ Score (Morphology-Aware)</div></div>', unsafe_allow_html=True)
        with stat3:
            avg_lat = df_saved["Latency (s)"].mean()
            st.markdown(f'<div class="stat-card"><div class="stat-val">{avg_lat:.2f}s</div><div class="stat-lbl">Average Inference Latency (CPU)</div></div>', unsafe_allow_html=True)

        st.write("")
        st.dataframe(
            df_saved[["Pair", "Model Used", "Input Text", "Human Reference", "Model Prediction", "BLEU", "chrF++", "Latency (s)"]],
            use_container_width=True
        )

    st.markdown("---")
    st.subheader("🧪 Live Interactive BLEU & chrF++ Tester")
    st.write("Compare any candidate translation against human ground-truth to compute real metrics:")
    
    t_c1, t_c2 = st.columns(2)
    with t_c1:
        test_pred = st.text_area("Model Prediction / Candidate Text", "कृत्रिम बुद्धिमत्ता आधुनिक सॉफ्टवेयर विकास को गति देती है।", height=100)
    with t_c2:
        test_ref = st.text_area("Human Reference / Ground Truth", "आर्टिफिशियल इंटेलिजेंस आधुनिक सॉफ्टवेयर विकास को गति प्रदान करता है।", height=100)

    if st.button("📊 Compute SacreBLEU Scores", type="primary"):
        m = compute_metrics(test_pred, test_ref)
        r1, r2 = st.columns(2)
        r1.metric("BLEU Score", f"{m['bleu']} / 100")
        r2.metric("chrF++ Score", f"{m['chrf_plus_plus']} / 100")


# ==============================================================================
# TAB 3: SESSION HISTORY
# ==============================================================================
with tab_history:
    st.markdown("### 📜 Current Session Translations")

    if not st.session_state.history:
        st.info("No translations performed in this session yet. Translate text in the first tab to view records here.")
    else:
        df_hist = pd.DataFrame(st.session_state.history)
        st.dataframe(df_hist, use_container_width=True)

        h_c1, h_c2 = st.columns(2)
        with h_c1:
            csv_bytes = df_hist.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download History as CSV",
                data=csv_bytes,
                file_name="translation_history.csv",
                mime="text/csv",
                use_container_width=True
            )
        with h_c2:
            if st.button("🗑️ Clear History", use_container_width=True):
                st.session_state.history = []
                st.rerun()
