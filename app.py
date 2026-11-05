"""
IndicAI Pro - AI-Based Multilingual Indian Language Translator
Supports All 22 Scheduled Indian Languages + English
Includes:
- Two-Person Voice Conversation (Turn-by-turn Speech-to-Speech)
- Single Voice Translator (Mic/Upload + ASR + NMT + TTS)
- Instant Text Translator (All 22 languages)
- Quality Benchmarks & BLEU Evaluation
- Session History
"""

import os
import io
import time
import pandas as pd
import streamlit as st

# Configure Layout & Meta
st.set_page_config(
    page_title="IndicAI Pro — 22 Indian Languages Voice & Neural MT",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Ultra-Modern SaaS CSS (DeepL / Linear Aesthetic)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
        color: #0f172a;
        background-color: #f8fafc;
    }

    #MainMenu, footer, header {visibility: hidden !important;}
    .block-container {
        padding-top: 1.25rem !important;
        padding-bottom: 2.5rem !important;
        max-width: 1320px !important;
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
        min-height: 220px;
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
        min-height: 220px;
        color: #94a3b8;
        background: #f8fafc;
        border: 1px dashed #cbd5e1;
        border-radius: 14px;
        gap: 8px;
    }

    /* Conversation Chat Bubbles */
    .chat-bubble-a {
        background: #eff6ff;
        border: 1px solid #bfdbfe;
        border-radius: 16px 16px 16px 4px;
        padding: 1rem 1.25rem;
        margin-bottom: 1rem;
    }
    .chat-bubble-b {
        background: #faf5ff;
        border: 1px solid #e9d5ff;
        border-radius: 16px 16px 4px 16px;
        padding: 1rem 1.25rem;
        margin-bottom: 1rem;
    }
    .bubble-speaker {
        font-size: 0.82rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.35rem;
    }
    .bubble-orig {
        font-size: 1.05rem;
        font-weight: 600;
        color: #0f172a;
    }
    .bubble-trans {
        font-size: 1.15rem;
        color: #1e3a8a;
        margin-top: 0.4rem;
        line-height: 1.6;
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
    .capability-tag {
        font-size: 0.72rem;
        font-weight: 600;
        padding: 2px 7px;
        border-radius: 4px;
        margin-left: 6px;
    }
    .cap-full { background: #dcfce7; color: #15803d; }
    .cap-in { background: #fef3c7; color: #b45309; }
    .cap-text { background: #e0f2fe; color: #0369a1; }
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

# Engines (Cached in Session State)
if "engine" not in st.session_state:
    st.session_state.engine = TranslationEngine(hf_token=st.session_state.hf_token)

if "asr_engine" not in st.session_state:
    st.session_state.asr_engine = SpeechRecognizer(model_size="base", device="cpu", compute_type="int8")

if "tts_engine" not in st.session_state:
    st.session_state.tts_engine = TextToSpeechEngine()

# Helpers for capability tagging
def format_lang_label(name: str) -> str:
    code = SUPPORTED_LANGUAGES.get(name, "")
    native = LANGUAGE_NATIVE_NAMES.get(name, name)
    cap = LANGUAGE_SPEECH_CAPABILITIES.get(code, {})
    if cap.get("asr") and cap.get("tts"):
        badge = " [🗣️ Voice + 🔊 Audio]"
    elif cap.get("asr"):
        badge = " [🗣️ Voice Input Only]"
    else:
        badge = " [📝 Text Only]"
    return f"{native}{badge}"

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
                <span class="brand-tag">All 22 Languages</span>
            </div>
        </div>
    </div>
    <div style="display: flex; align-items: center; gap: 14px;">
        <div class="status-pill">
            <span class="status-dot"></span>
            <span>22 Scheduled Languages + English</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Main Navigation Tabs
tab_convo, tab_single_voice, tab_translate, tab_benchmarks, tab_history = st.tabs([
    "👥 Two-Person Voice Conversation",
    "🎙️ Single Voice Translator",
    "✨ Instant Text Translator",
    "📈 Quality Benchmarks & BLEU",
    "📋 Session History"
])

all_lang_names = list(SUPPORTED_LANGUAGES.keys())

# ==============================================================================
# TAB 1: TWO-PERSON VOICE CONVERSATION TRANSLATOR (TURN-BY-TURN)
# ==============================================================================
with tab_convo:
    st.markdown("""
    <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 16px; padding: 1.25rem 1.5rem; margin-bottom: 1.5rem; box-shadow: 0 4px 12px rgba(15,23,42,0.03);">
        <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 10px;">
            <div>
                <h3 style="margin: 0; font-size: 1.35rem; font-weight: 800; color: #1e293b;">Bilingual Voice Conversation Mode</h3>
                <p style="margin: 0.25rem 0 0 0; color: #64748b; font-size: 0.95rem;">
                    Speak turn-by-turn in any two Indian languages. The system transcribes, translates with IndicTrans2, and speaks the translated response aloud!
                </p>
            </div>
            <div>
                <span class="tech-pill" style="background:#eff6ff; color:#2563eb; font-weight:700;">Automatic Turn-by-Turn</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Participant Language Configuration
    p_col1, p_col2 = st.columns(2)
    with p_col1:
        st.markdown("#### 👤 Person A")
        lang_a = st.selectbox(
            "Person A Language",
            options=all_lang_names,
            index=all_lang_names.index("Hindi"),
            format_func=format_lang_label,
            key="convo_lang_a"
        )
        code_a = SUPPORTED_LANGUAGES[lang_a]
        cap_a = LANGUAGE_SPEECH_CAPABILITIES.get(code_a, {})

    with p_col2:
        st.markdown("#### 👤 Person B")
        lang_b = st.selectbox(
            "Person B Language",
            options=all_lang_names,
            index=all_lang_names.index("Tamil"),
            format_func=format_lang_label,
            key="convo_lang_b"
        )
        code_b = SUPPORTED_LANGUAGES[lang_b]
        cap_b = LANGUAGE_SPEECH_CAPABILITIES.get(code_b, {})

    # Live Capability Alert Banner for Conversation
    warnings = []
    if not cap_a.get("asr"):
        warnings.append(f"Person A ({lang_a}) does not have speech recognition support (type text instead).")
    if not cap_b.get("tts"):
        warnings.append(f"Person B ({lang_b}) will receive translated text, but audio speech output is unavailable.")
    if not cap_b.get("asr"):
        warnings.append(f"Person B ({lang_b}) does not have speech recognition support (type text instead).")
    if not cap_a.get("tts"):
        warnings.append(f"Person A ({lang_a}) will receive translated text, but audio speech output is unavailable.")

    if warnings:
        with st.expander("ℹ️ Capability Notes for Selected Conversation Pair"):
            for w in warnings:
                st.caption(f"• {w}")

    st.write("---")

    # Turn-by-Turn Speaking Consoles
    spk_col1, spk_col2 = st.columns(2)

    # ---------------- PERSON A'S CONSOLE ----------------
    with spk_col1:
        st.markdown(f"### 🎤 Speak as Person A ({lang_a})")
        audio_a = st.audio_input(f"Record Person A ({lang_a})", key="mic_person_a")
        
        # Optional Text Fallback / Correction
        text_input_a = st.text_input(
            "Or type message for Person A:",
            placeholder=f"Type in {lang_a}...",
            key="text_person_a"
        )

        send_a = st.button(f"➡️ Send to Person B ({lang_b})", key="btn_send_a", type="primary", use_container_width=True)

        if send_a:
            orig_text = ""
            if audio_a is not None:
                with st.spinner(f"Transcribing {lang_a} speech..."):
                    asr_lang = cap_a.get("asr")
                    orig_text, meta = st.session_state.asr_engine.transcribe(audio_a, source_language_code=asr_lang)
            elif text_input_a.strip():
                orig_text = text_input_a.strip()
            else:
                st.warning(f"Please record voice or type a message for Person A.")

            if orig_text.strip():
                with st.spinner(f"Translating ({lang_a} ➔ {lang_b})..."):
                    trans_text = st.session_state.engine.translate(orig_text, src_lang=code_a, tgt_lang=code_b)

                audio_b_bytes = None
                if cap_b.get("tts"):
                    with st.spinner(f"Speaking translated audio for Person B ({lang_b})..."):
                        try:
                            audio_b_bytes, _ = st.session_state.tts_engine.synthesize(trans_text, target_lang_code=code_b)
                        except Exception as e:
                            st.caption(f"TTS Notice: {e}")

                st.session_state.dialogue_history.append({
                    "speaker": "Person A",
                    "src_lang": lang_a,
                    "tgt_lang": lang_b,
                    "original": orig_text,
                    "translated": trans_text,
                    "audio": audio_b_bytes,
                    "time": time.strftime("%H:%M:%S")
                })
                st.rerun()

    # ---------------- PERSON B'S CONSOLE ----------------
    with spk_col2:
        st.markdown(f"### 🎤 Speak as Person B ({lang_b})")
        audio_b = st.audio_input(f"Record Person B ({lang_b})", key="mic_person_b")

        text_input_b = st.text_input(
            "Or type message for Person B:",
            placeholder=f"Type in {lang_b}...",
            key="text_person_b"
        )

        send_b = st.button(f"⬅️ Send to Person A ({lang_a})", key="btn_send_b", type="primary", use_container_width=True)

        if send_b:
            orig_text = ""
            if audio_b is not None:
                with st.spinner(f"Transcribing {lang_b} speech..."):
                    asr_lang = cap_b.get("asr")
                    orig_text, meta = st.session_state.asr_engine.transcribe(audio_b, source_language_code=asr_lang)
            elif text_input_b.strip():
                orig_text = text_input_b.strip()
            else:
                st.warning(f"Please record voice or type a message for Person B.")

            if orig_text.strip():
                with st.spinner(f"Translating ({lang_b} ➔ {lang_a})..."):
                    trans_text = st.session_state.engine.translate(orig_text, src_lang=code_b, tgt_lang=code_a)

                audio_a_bytes = None
                if cap_a.get("tts"):
                    with st.spinner(f"Speaking translated audio for Person A ({lang_a})..."):
                        try:
                            audio_a_bytes, _ = st.session_state.tts_engine.synthesize(trans_text, target_lang_code=code_a)
                        except Exception as e:
                            st.caption(f"TTS Notice: {e}")

                st.session_state.dialogue_history.append({
                    "speaker": "Person B",
                    "src_lang": lang_b,
                    "tgt_lang": lang_a,
                    "original": orig_text,
                    "translated": trans_text,
                    "audio": audio_a_bytes,
                    "time": time.strftime("%H:%M:%S")
                })
                st.rerun()

    # ---------------- LIVE CONVERSATION DIALOGUE STREAM ----------------
    st.write("---")
    st.markdown("### 💬 Live Conversation Dialogue")

    if not st.session_state.dialogue_history:
        st.info("No messages in this conversation yet. Person A or Person B can speak above to begin.")
    else:
        for turn in reversed(st.session_state.dialogue_history):
            if turn["speaker"] == "Person A":
                st.markdown(f"""
                <div class="chat-bubble-a">
                    <div class="bubble-speaker" style="color: #2563eb;">👤 Person A ({turn['src_lang']}) ➔ Person B ({turn['tgt_lang']}) • <span style="font-weight:400; color:#64748b;">{turn['time']}</span></div>
                    <div class="bubble-orig">"{turn['original']}"</div>
                    <div class="bubble-trans">➔ {turn['translated']}</div>
                </div>
                """, unsafe_allow_html=True)
                if turn.get("audio"):
                    st.audio(turn["audio"], format="audio/mp3")
            else:
                st.markdown(f"""
                <div class="chat-bubble-b">
                    <div class="bubble-speaker" style="color: #7c3aed;">👤 Person B ({turn['src_lang']}) ➔ Person A ({turn['tgt_lang']}) • <span style="font-weight:400; color:#64748b;">{turn['time']}</span></div>
                    <div class="bubble-orig">"{turn['original']}"</div>
                    <div class="bubble-trans">➔ {turn['translated']}</div>
                </div>
                """, unsafe_allow_html=True)
                if turn.get("audio"):
                    st.audio(turn["audio"], format="audio/mp3")

        if st.button("🗑️ Clear Conversation Chat"):
            st.session_state.dialogue_history = []
            st.rerun()


# ==============================================================================
# TAB 2: SINGLE VOICE TRANSLATOR
# ==============================================================================
with tab_single_voice:
    st.markdown("### 🎙️ Voice & Audio File Translator")
    st.caption("Record voice or upload an audio file to transcribe with faster-whisper, translate with IndicTrans2, and hear spoken output.")

    sv_col1, sv_col2 = st.columns(2)
    with sv_col1:
        s_src_lang = st.selectbox(
            "Spoken / Input Language",
            options=["Auto Detect"] + all_lang_names,
            format_func=lambda x: "🌐 Auto Detect" if x == "Auto Detect" else format_lang_label(x),
            key="sv_src_lang"
        )
    with sv_col2:
        s_tgt_lang = st.selectbox(
            "Target Spoken Language",
            options=all_lang_names,
            index=all_lang_names.index("Hindi"),
            format_func=format_lang_label,
            key="sv_tgt_lang"
        )

    s_code_tgt = SUPPORTED_LANGUAGES[s_tgt_lang]
    s_cap_tgt = LANGUAGE_SPEECH_CAPABILITIES.get(s_code_tgt, {})

    st.write("")
    sv_mode = st.radio("Input Source", ["🎙️ Microphone", "📁 File Upload"], horizontal=True)
    
    sv_audio = None
    if "Microphone" in sv_mode:
        sv_audio = st.audio_input("Record voice", key="sv_mic")
    else:
        sv_audio = st.file_uploader("Upload audio file", type=["wav", "mp3", "m4a", "ogg"], key="sv_file")

    if sv_audio is not None:
        st.audio(sv_audio)

    if st.button("🚀 Transcribe & Translate", key="btn_sv_run", type="primary"):
        if sv_audio is None:
            st.warning("Please record voice or upload audio.")
        else:
            with st.spinner("Transcribing speech..."):
                t_sv_start = time.time()
                whisper_arg = None
                if s_src_lang != "Auto Detect":
                    whisper_arg = LANGUAGE_SPEECH_CAPABILITIES.get(SUPPORTED_LANGUAGES[s_src_lang], {}).get("asr")
                
                txt_rec, meta = st.session_state.asr_engine.transcribe(sv_audio, source_language_code=whisper_arg)

            if not txt_rec.strip():
                st.warning("No speech detected.")
            else:
                if s_src_lang != "Auto Detect":
                    s_code_src = SUPPORTED_LANGUAGES[s_src_lang]
                    disp_src = s_src_lang
                else:
                    det = meta.get("detected_language", "en")
                    from model_engine.speech_recognition import WHISPER_TO_FLORES_LANG
                    s_code_src = WHISPER_TO_FLORES_LANG.get(det, "eng_Latn")
                    disp_src = [k for k, v in SUPPORTED_LANGUAGES.items() if v == s_code_src][0]

                with st.spinner(f"Translating ({disp_src} ➔ {s_tgt_lang})..."):
                    txt_trans = st.session_state.engine.translate(txt_rec, src_lang=s_code_src, tgt_lang=s_code_tgt)

                voice_bytes = None
                if s_cap_tgt.get("tts"):
                    with st.spinner(f"Generating voice output in {s_tgt_lang}..."):
                        try:
                            voice_bytes, _ = st.session_state.tts_engine.synthesize(txt_trans, target_lang_code=s_code_tgt)
                        except Exception as e:
                            st.caption(f"TTS Notice: {e}")

                st.write("---")
                o_c1, o_c2 = st.columns(2)
                with o_c1:
                    st.markdown("#### 🗣️ Recognized Text")
                    st.write(txt_rec)
                    st.caption(f"Language: **{meta.get('detected_language')}** | Duration: **{meta.get('duration')}s**")
                with o_c2:
                    st.markdown(f"#### 🔊 Translated Output ({s_tgt_lang})")
                    st.write(txt_trans)
                    if voice_bytes:
                        st.audio(voice_bytes, format="audio/mp3")
                    else:
                        st.info(f"Audio voice generation is unavailable for {s_tgt_lang}. Translation is displayed above.")


# ==============================================================================
# TAB 3: INSTANT TEXT TRANSLATOR (ALL 22 LANGUAGES PRESERVED)
# ==============================================================================
with tab_translate:
    with st.container():
        ctrl_col1, ctrl_swap, ctrl_col2, ctrl_meta = st.columns([4, 1, 4, 3])

        with ctrl_col1:
            txt_src = st.selectbox(
                "Source Language",
                options=all_lang_names,
                index=all_lang_names.index("English"),
                format_func=format_lang_label,
                key="text_src_select"
            )

        with ctrl_swap:
            st.write("")
            st.write("")
            if st.button("⇄", help="Swap Languages", key="swap_text_btn", use_container_width=True):
                tmp = st.session_state["text_src_select"]
                st.session_state["text_src_select"] = st.session_state["text_tgt_select"]
                st.session_state["text_tgt_select"] = tmp
                st.rerun()

        with ctrl_col2:
            txt_tgt = st.selectbox(
                "Target Language",
                options=all_lang_names,
                index=all_lang_names.index("Hindi"),
                format_func=format_lang_label,
                key="text_tgt_select"
            )

        code_s = SUPPORTED_LANGUAGES[txt_src]
        code_t = SUPPORTED_LANGUAGES[txt_tgt]
        model_routed = get_model_for_pair(code_s, code_t)

        with ctrl_meta:
            st.write("")
            st.write("")
            if model_routed == "IDENTITY":
                st.markdown('<div class="tech-pill" style="color:#0284c7; background:#e0f2fe;">⚡ Direct Same-Language Pass</div>', unsafe_allow_html=True)
            else:
                is_ld = st.session_state.engine.is_model_loaded(model_routed)
                st.markdown(f'<div class="tech-pill"><b>{model_routed.split("/")[-1]}</b></div>', unsafe_allow_html=True)

    # Preset sample chips
    st.write("")
    p_cols = st.columns([1.2] + [2.5] * len(PRESETS))
    with p_cols[0]:
        st.markdown("<p style='font-size:0.85rem; font-weight:700; color:#64748b; margin-top:8px;'>💡 Sample:</p>", unsafe_allow_html=True)
    for idx, (label, text) in enumerate(PRESETS):
        with p_cols[idx + 1]:
            if st.button(f"✨ {label}", key=f"t_sample_{idx}", use_container_width=True):
                st.session_state["text_input_area"] = text
                st.rerun()

    # Workstation Panes
    tp_1, tp_2 = st.columns(2)
    with tp_1:
        st.markdown(f"**From:** `{txt_src}` ({code_s})")
        t_user_in = st.text_area(
            "Source Text",
            height=250,
            placeholder=f"Enter text in {txt_src}...",
            label_visibility="collapsed",
            key="text_input_area"
        )
        t_c = len(t_user_in)
        t_w = len(t_user_in.split()) if t_user_in.strip() else 0
        st.markdown(f'<span class="tech-pill">Chars: {t_c}</span> <span class="tech-pill">Words: {t_w}</span>', unsafe_allow_html=True)

    with tp_2:
        st.markdown(f"**To:** `{txt_tgt}` ({code_t})")
        if "text_out_val" in st.session_state and st.session_state["text_out_val"]:
            st.markdown(f'<div class="output-container">{st.session_state["text_out_val"]}</div>', unsafe_allow_html=True)
            
            # Speak Button if TTS supported
            if LANGUAGE_SPEECH_CAPABILITIES.get(code_t, {}).get("tts"):
                with st.popover("🔊 Speak Audio"):
                    try:
                        t_audio, _ = st.session_state.tts_engine.synthesize(st.session_state["text_out_val"], target_lang_code=code_t)
                        st.audio(t_audio, format="audio/mp3")
                    except Exception as e:
                        st.caption(f"Notice: {e}")
        else:
            st.markdown('<div class="output-empty">Translation will appear here instantly</div>', unsafe_allow_html=True)

    st.write("")
    _, act_btn, _ = st.columns([1, 2, 1])
    with act_btn:
        if st.button("🚀 Translate Text", key="btn_trans_run", type="primary", use_container_width=True):
            if not t_user_in.strip():
                st.warning("Please enter text to translate.")
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
# TAB 4: QUALITY BENCHMARKS & EVALUATION
# ==============================================================================
with tab_benchmarks:
    st.markdown("### 📈 SacreBLEU & chrF++ Quantitative Evaluation")
    st.markdown("Real computed metrics across the 7 required translation pairs using AI4Bharat IndicTrans2 models:")

    df_saved = get_saved_benchmark()
    if not df_saved.empty:
        b1, b2, b3 = st.columns(3)
        with b1:
            avg_bleu = df_saved["BLEU"].mean()
            st.markdown(f'<div class="kpi-card"><div class="kpi-title">Average SacreBLEU</div><div class="kpi-val">{avg_bleu:.1f}</div><div class="kpi-sub">✓ Word n-gram Precision (0-100)</div></div>', unsafe_allow_html=True)
        with b2:
            avg_chrf = df_saved["chrF++"].mean()
            st.markdown(f'<div class="kpi-card"><div class="kpi-title">Average chrF++</div><div class="kpi-val">{avg_chrf:.1f}</div><div class="kpi-sub">✓ Morphology-Aware Indic Metric</div></div>', unsafe_allow_html=True)
        with b3:
            avg_time = df_saved["Latency (s)"].mean()
            st.markdown(f'<div class="kpi-card"><div class="kpi-title">Average Latency</div><div class="kpi-val">{avg_time:.2f}s</div><div class="kpi-sub">✓ CPU Inference Speed</div></div>', unsafe_allow_html=True)

        st.write("")
        st.dataframe(
            df_saved[["Pair", "Model Used", "Input Text", "Human Reference", "Model Prediction", "BLEU", "chrF++", "Latency (s)"]],
            use_container_width=True
        )

    st.markdown("---")
    st.subheader("🧪 Live Metric Evaluation Sandbox")
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
# TAB 5: SESSION HISTORY
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
