"""
IndicAI Pro - AI-Based Multilingual Indian Language Translator
Ultra-Modern, Production-Grade SaaS Interface with Full Speech-to-Speech Translation
"""

import os
import io
import time
import pandas as pd
import streamlit as st

# Configure Layout & Meta
st.set_page_config(
    page_title="IndicAI Pro — Neural MT & Speech-to-Speech",
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
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 20px -2px rgba(15, 23, 42, 0.04);
    }
    .brand-group {
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .brand-logo {
        width: 38px;
        height: 38px;
        border-radius: 11px;
        background: linear-gradient(135deg, #2563eb, #7c3aed);
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: 800;
        font-size: 1.15rem;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
    }
    .brand-name {
        font-size: 1.25rem;
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

    /* Speech Pipeline Flow Card */
    .pipeline-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 1rem 1.25rem;
        margin-bottom: 1.25rem;
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.03);
    }
    .flow-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: #f1f5f9;
        color: #334155;
        font-weight: 600;
        font-size: 0.8rem;
        padding: 4px 10px;
        border-radius: 8px;
    }

    /* Audio Player Box */
    .audio-playback-box {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1rem;
        margin-top: 0.75rem;
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
from model_engine.speech_recognition import SpeechRecognizer, FLORES_TO_WHISPER_LANG
from model_engine.text_to_speech import TextToSpeechEngine, FLORES_TO_TTS_LANG
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

# Engines (Cached in Session State)
if "engine" not in st.session_state:
    st.session_state.engine = TranslationEngine(hf_token=st.session_state.hf_token)

if "asr_engine" not in st.session_state:
    st.session_state.asr_engine = SpeechRecognizer(model_size="base", device="cpu", compute_type="int8")

if "tts_engine" not in st.session_state:
    st.session_state.tts_engine = TextToSpeechEngine()

# State variables for Text Translator
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

# State variables for Speech-to-Speech
if "speech_src_lang" not in st.session_state:
    st.session_state["speech_src_lang"] = "Auto Detect"

if "speech_tgt_lang" not in st.session_state:
    st.session_state["speech_tgt_lang"] = "Hindi"

if "speech_recognized_text" not in st.session_state:
    st.session_state["speech_recognized_text"] = ""

if "speech_translated_text" not in st.session_state:
    st.session_state["speech_translated_text"] = ""

if "speech_audio_bytes" not in st.session_state:
    st.session_state["speech_audio_bytes"] = None

if "speech_detected_meta" not in st.session_state:
    st.session_state["speech_detected_meta"] = None

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

def do_clear_speech():
    st.session_state["speech_recognized_text"] = ""
    st.session_state["speech_translated_text"] = ""
    st.session_state["speech_audio_bytes"] = None
    st.session_state["speech_detected_meta"] = None

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
                <span class="brand-tag">Speech & Text MT v2.5</span>
            </div>
        </div>
    </div>
    <div style="display: flex; align-items: center; gap: 14px;">
        <div class="status-pill">
            <span class="status-dot"></span>
            <span>Speech & IndicTrans2 Ready</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Main Navigation Tabs (Preserving existing tabs + Adding Speech-to-Speech)
tab_speech, tab_translate, tab_benchmarks, tab_history = st.tabs([
    "🎙️ Speech-to-Speech Translation",
    "✨ Text Translator",
    "📈 Quality Benchmarks & BLEU",
    "📋 Session History"
])

# ==============================================================================
# TAB 1: SPEECH-TO-SPEECH TRANSLATION PIPELINE
# ==============================================================================
with tab_speech:
    # Pipeline Overview Banner
    st.markdown("""
    <div class="pipeline-card">
        <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 10px;">
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="font-size: 1.1rem; font-weight: 700; color: #1e293b;">Neural Speech-to-Speech Pipeline</span>
            </div>
            <div style="display: flex; align-items: center; gap: 6px; flex-wrap: wrap;">
                <span class="flow-badge">1. Voice Input (Mic / File)</span> ➔
                <span class="flow-badge">2. Whisper ASR</span> ➔
                <span class="flow-badge">3. IndicTrans2 NMT</span> ➔
                <span class="flow-badge">4. Multilingual TTS</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    lang_names = list(SUPPORTED_LANGUAGES.keys())
    speech_src_options = ["Auto Detect"] + lang_names

    # Language Selection Controls
    s_col1, s_col2, s_col3 = st.columns([4, 4, 3])
    
    with s_col1:
        speech_src_choice = st.selectbox(
            "Spoken / Input Language",
            options=speech_src_options,
            format_func=lambda x: "🌐 Auto Detect (faster-whisper)" if x == "Auto Detect" else LANGUAGE_NATIVE_NAMES.get(x, x),
            key="speech_src_lang"
        )
    
    with s_col2:
        speech_tgt_choice = st.selectbox(
            "Target Spoken Language",
            options=lang_names,
            format_func=lambda x: LANGUAGE_NATIVE_NAMES.get(x, x),
            key="speech_tgt_lang"
        )

    with s_col3:
        st.write("")
        st.write("")
        st.button("🧹 Reset Voice Session", on_click=do_clear_speech, use_container_width=True)

    # 1. AUDIO INPUT INTERFACE (Microphone or File Upload)
    st.write("")
    st.subheader("Step 1: Record or Upload Speech")

    input_mode = st.radio(
        "Choose Speech Input Source",
        ["🎙️ Record via Microphone", "📁 Upload Audio File (WAV, MP3, M4A, OGG)"],
        horizontal=True,
        label_visibility="collapsed"
    )

    audio_source = None
    audio_format = "audio/wav"

    if "Microphone" in input_mode:
        # Native Streamlit audio_input
        mic_audio = st.audio_input("Record voice through microphone")
        if mic_audio is not None:
            audio_source = mic_audio
            audio_format = "audio/wav"
    else:
        # File upload option
        uploaded_audio = st.file_uploader(
            "Upload speech recording",
            type=["wav", "mp3", "m4a", "ogg", "flac"],
            help="Upload an audio file in English or any supported Indian language."
        )
        if uploaded_audio is not None:
            audio_source = uploaded_audio
            audio_format = uploaded_audio.type or "audio/wav"

    # Audio Preview
    if audio_source is not None:
        st.audio(audio_source, format=audio_format)

    # 2. TRANSCRIBE & TRANSLATE ACTION
    st.write("")
    transcribe_col1, transcribe_col2, transcribe_col3 = st.columns([1, 2, 1])
    with transcribe_col2:
        start_speech_btn = st.button("🚀 Transcribe & Translate Voice", type="primary", use_container_width=True)

    if start_speech_btn:
        if audio_source is None:
            st.warning("⚠️ Please record a voice clip or upload an audio file first.")
        else:
            with st.spinner("Step 1/3: Recognizing speech using faster-whisper..."):
                t_speech_start = time.time()
                
                # Determine language code for Whisper
                whisper_lang_arg = None
                if speech_src_choice != "Auto Detect":
                    whisper_lang_arg = SUPPORTED_LANGUAGES[speech_src_choice]

                try:
                    recognized_text, meta = st.session_state.asr_engine.transcribe(
                        audio_source,
                        source_language_code=whisper_lang_arg
                    )
                    st.session_state["speech_recognized_text"] = recognized_text
                    st.session_state["speech_detected_meta"] = meta

                    if not recognized_text.strip():
                        st.warning("Could not detect any speech in the audio. Please speak clearly into the microphone.")
                    else:
                        # Determine source FLORES code
                        if speech_src_choice != "Auto Detect":
                            effective_src_code = SUPPORTED_LANGUAGES[speech_src_choice]
                            effective_src_lang = speech_src_choice
                        else:
                            # Map detected 2-letter code back to FLORES
                            det_code = meta.get("detected_language", "en")
                            from model_engine.speech_recognition import WHISPER_TO_FLORES_LANG
                            effective_src_code = WHISPER_TO_FLORES_LANG.get(det_code, "eng_Latn")
                            effective_src_lang = [k for k, v in SUPPORTED_LANGUAGES.items() if v == effective_src_code][0]

                        effective_tgt_code = SUPPORTED_LANGUAGES[speech_tgt_choice]
                        
                        # Step 2/3: Neural Translation via IndicTrans2
                        with st.spinner(f"Step 2/3: Translating ({effective_src_lang} ➔ {speech_tgt_choice}) via IndicTrans2..."):
                            translated_text = st.session_state.engine.translate(
                                recognized_text,
                                src_lang=effective_src_code,
                                tgt_lang=effective_tgt_code,
                            )
                            st.session_state["speech_translated_text"] = translated_text

                        # Step 3/3: Synthesize Speech via TTS
                        with st.spinner(f"Step 3/3: Generating speech audio in {speech_tgt_choice}..."):
                            audio_bytes, mime = st.session_state.tts_engine.synthesize(
                                translated_text,
                                target_lang_code=effective_tgt_code,
                            )
                            st.session_state["speech_audio_bytes"] = audio_bytes

                        t_speech_total = time.time() - t_speech_start
                        
                        # Record in history
                        st.session_state.history.append({
                            "Time": time.strftime("%H:%M:%S"),
                            "Source": f"{effective_src_lang} (Voice)",
                            "Target": f"{speech_tgt_choice} (Audio)",
                            "Input": recognized_text,
                            "Translated": translated_text,
                            "Model": "Whisper + IndicTrans2 + TTS",
                            "Latency (s)": round(t_speech_total, 2)
                        })
                        st.rerun()

                except PermissionError as pe:
                    st.error("🔒 Hugging Face Authentication Required")
                    st.info(f"{pe}")
                except Exception as ex:
                    st.error(f"Speech processing error: {ex}")

    # 3. DISPLAY EDITABLE RECOGNIZED TEXT & TRANSLATED SPEECH OUTPUT
    if st.session_state["speech_recognized_text"]:
        st.write("---")
        res_col1, res_col2 = st.columns(2)

        with res_col1:
            st.markdown("#### 🗣️ Recognized Speech (Editable)")
            meta = st.session_state.get("speech_detected_meta", {})
            if meta:
                st.caption(f"Audio Duration: **{meta.get('duration', 0)}s** | Detected Language: **{meta.get('detected_language', 'N/A').upper()}** (prob: {meta.get('language_probability', 1.0)})")
            
            # Editable text area allowing user corrections before re-translation
            edited_asr = st.text_area(
                "Recognized speech text",
                value=st.session_state["speech_recognized_text"],
                height=160,
                key="asr_editable_field"
            )

            # Re-translate button if user edits text
            if st.button("🔄 Re-Translate Edited Text", use_container_width=True):
                if edited_asr.strip():
                    with st.spinner("Re-translating edited text..."):
                        if speech_src_choice != "Auto Detect":
                            re_src_code = SUPPORTED_LANGUAGES[speech_src_choice]
                        else:
                            det_code = meta.get("detected_language", "en")
                            from model_engine.speech_recognition import WHISPER_TO_FLORES_LANG
                            re_src_code = WHISPER_TO_FLORES_LANG.get(det_code, "eng_Latn")
                        
                        re_tgt_code = SUPPORTED_LANGUAGES[speech_tgt_choice]

                        new_trans = st.session_state.engine.translate(
                            edited_asr,
                            src_lang=re_src_code,
                            tgt_lang=re_tgt_code
                        )
                        new_audio, _ = st.session_state.tts_engine.synthesize(
                            new_trans,
                            target_lang_code=re_tgt_code
                        )
                        st.session_state["speech_recognized_text"] = edited_asr
                        st.session_state["speech_translated_text"] = new_trans
                        st.session_state["speech_audio_bytes"] = new_audio
                        st.rerun()

        with res_col2:
            st.markdown(f"#### 🔊 Translated Voice Output ({speech_tgt_choice})")
            
            trans_text = st.session_state.get("speech_translated_text", "")
            if trans_text:
                st.markdown(f'<div class="output-container" style="min-height: 160px;">{trans_text}</div>', unsafe_allow_html=True)
                
                audio_bytes = st.session_state.get("speech_audio_bytes")
                if audio_bytes:
                    st.write("")
                    st.markdown("**Listen to Output Speech:**")
                    st.audio(audio_bytes, format="audio/mp3")

                    down_c1, down_c2 = st.columns(2)
                    with down_c1:
                        st.download_button(
                            "📥 Download Audio (MP3)",
                            data=audio_bytes,
                            file_name=f"translated_speech_{speech_tgt_choice.lower()}.mp3",
                            mime="audio/mp3",
                            use_container_width=True
                        )
                    with down_c2:
                        st.download_button(
                            "📄 Download Text (TXT)",
                            data=trans_text,
                            file_name=f"translated_text_{speech_tgt_choice.lower()}.txt",
                            mime="text/plain",
                            use_container_width=True
                        )


# ==============================================================================
# TAB 2: TEXT TRANSLATOR (ORIGINAL PRESERVED WITH WORLD-CLASS INTERFACE)
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
        foot_out1, foot_out2, foot_out3, foot_out4 = st.columns([1.5, 1, 1, 1])
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
        with foot_out4:
            # Voice speak button for Text Translator
            if st.session_state.output_text:
                with st.popover("🔊 Speak"):
                    try:
                        voice_bytes, _ = st.session_state.tts_engine.synthesize(
                            st.session_state.output_text,
                            target_lang_code=tgt_code
                        )
                        st.audio(voice_bytes, format="audio/mp3")
                    except Exception as ve:
                        st.error(f"TTS error: {ve}")

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
# TAB 3: QUALITY BENCHMARKS & EVALUATION
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
# TAB 4: SESSION HISTORY
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
