"""
AI-Based Multilingual Indian Language Translator Using NLP
Streamlit Web Application
"""

import os
import time
import pandas as pd
import streamlit as st

# Configure page layout and meta
st.set_page_config(
    page_title="AI-Based Multilingual Indian Language Translator",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling (Clean Blue, White, Subtle Purple Theme)
st.markdown("""
<style>
    /* Global Styles */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Header Gradient & Card */
    .main-header {
        background: linear-gradient(135deg, #1e3a8a 0%, #2563eb 50%, #7c3aed 100%);
        padding: 2.2rem 2rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px -5px rgba(37, 99, 235, 0.25);
    }
    .main-header h1 {
        color: #ffffff;
        font-weight: 700;
        font-size: 2.1rem;
        margin: 0;
        letter-spacing: -0.02em;
    }
    .main-header p {
        color: #e0e7ff;
        font-size: 1.05rem;
        margin-top: 0.5rem;
        margin-bottom: 0;
    }

    /* Cards */
    .metric-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.2rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        margin-bottom: 1rem;
    }
    .output-card {
        background: #f8fafc;
        border: 1px solid #cbd5e1;
        border-radius: 12px;
        padding: 1.25rem;
        min-height: 180px;
        font-size: 1.1rem;
        line-height: 1.6;
        color: #0f172a;
    }
    .badge-ready {
        display: inline-block;
        background-color: #dcfce7;
        color: #15803d;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 600;
    }
    .badge-warn {
        display: inline-block;
        background-color: #fef3c7;
        color: #b45309;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 600;
    }
    .badge-model {
        background-color: #ede9fe;
        color: #6d28d9;
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        font-family: monospace;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

# Imports from modular engine
from model_engine.config import (
    SUPPORTED_LANGUAGES,
    CODE_TO_LANGUAGE,
    LANGUAGE_NATIVE_NAMES,
    MODEL_EN_INDIC,
    MODEL_INDIC_EN,
    MODEL_INDIC_INDIC,
    get_model_for_pair,
)
from model_engine.translator import TranslationEngine
from evaluation.evaluator import compute_metrics, evaluate_batch
from evaluation.test_benchmark import BENCHMARK_PAIRS, run_benchmark, get_saved_benchmark

# Initialize Session State
if "history" not in st.session_state:
    st.session_state.history = []

if "hf_token" not in st.session_state:
    st.session_state.hf_token = os.environ.get("HF_TOKEN", "")

if "engine" not in st.session_state:
    st.session_state.engine = TranslationEngine(hf_token=st.session_state.hf_token)

# Sidebar Configuration
st.sidebar.image("https://img.icons8.com/fluency/96/translate.png", width=64)
st.sidebar.title("Translator Control")

navigation = st.sidebar.radio(
    "Navigation",
    [
        "🌐 Neural Translator",
        "📋 Project Overview",
        "🧠 AI Model Architecture",
        "📊 SacreBLEU Evaluation",
        "📜 Translation History",
    ],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.subheader("🔑 Hugging Face Access")
hf_token_input = st.sidebar.text_input(
    "HF Access Token (Read Permission)",
    value=st.session_state.hf_token,
    type="password",
    help="AI4Bharat IndicTrans2 is a gated model. Accepting terms on HF and providing a free Read token unlocks downloading.",
)

if hf_token_input != st.session_state.hf_token:
    st.session_state.hf_token = hf_token_input.strip()
    st.session_state.engine.set_hf_token(st.session_state.hf_token)
    st.sidebar.success("Token updated in memory!")

if not st.session_state.hf_token:
    st.sidebar.markdown("""
    <div style="font-size: 0.8rem; color: #64748b; line-height: 1.4;">
    ⚠️ AI4Bharat models are <b>gated</b>.<br>
    1. Visit <a href="https://huggingface.co/ai4bharat/indictrans2-en-indic-dist-200M" target="_blank">HF Repo</a> and click <i>Agree and access</i>.<br>
    2. Get free token from <a href="https://huggingface.co/settings/tokens" target="_blank">HF Settings</a>.<br>
    3. Paste token above.
    </div>
    """, unsafe_allow_html=True)
else:
    st.sidebar.markdown('<span class="badge-ready">✓ Token Configured</span>', unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.subheader("💻 System & Inference")
device_name = "CUDA GPU" if st.session_state.engine.device.type == "cuda" else "CPU (Float32)"
st.sidebar.write(f"**Compute Device:** `{device_name}`")
st.sidebar.write(f"**Precision:** `{'float16' if st.session_state.engine.device.type == 'cuda' else 'float32'}`")

# -------------------------------------------------------------
# PAGE 1: NEURAL TRANSLATOR
# -------------------------------------------------------------
if navigation == "🌐 Neural Translator":
    st.markdown("""
    <div class="main-header">
        <h1>AI-Based Multilingual Indian Language Translator</h1>
        <p>Real-Time Neural Machine Translation using AI4Bharat IndicTrans2 Pretrained Transformer Models</p>
    </div>
    """, unsafe_allow_html=True)

    # Language Selectors with Swap
    col_src, col_swap, col_tgt = st.columns([5, 1, 5])
    
    lang_names = list(SUPPORTED_LANGUAGES.keys())

    if "src_idx" not in st.session_state:
        st.session_state.src_idx = 0  # English
    if "tgt_idx" not in st.session_state:
        st.session_state.tgt_idx = 1  # Hindi

    with col_src:
        src_choice = st.selectbox(
            "Source Language",
            options=lang_names,
            index=st.session_state.src_idx,
            format_func=lambda x: LANGUAGE_NATIVE_NAMES.get(x, x),
            key="src_select"
        )
        st.session_state.src_idx = lang_names.index(src_choice)

    with col_swap:
        st.write("")
        st.write("")
        if st.button("⇄", help="Swap Source and Target Languages", use_container_width=True):
            tmp = st.session_state.src_idx
            st.session_state.src_idx = st.session_state.tgt_idx
            st.session_state.tgt_idx = tmp
            st.rerun()

    with col_tgt:
        tgt_choice = st.selectbox(
            "Target Language",
            options=lang_names,
            index=st.session_state.tgt_idx,
            format_func=lambda x: LANGUAGE_NATIVE_NAMES.get(x, x),
            key="tgt_select"
        )
        st.session_state.tgt_idx = lang_names.index(tgt_choice)

    src_code = SUPPORTED_LANGUAGES[src_choice]
    tgt_code = SUPPORTED_LANGUAGES[tgt_choice]
    routed_model = get_model_for_pair(src_code, tgt_code)

    # Active Route Banner
    if routed_model == "IDENTITY":
        st.info("ℹ️ Same source and target language selected: Input text will be returned directly without model inference.")
    else:
        is_cached = st.session_state.engine.is_model_loaded(routed_model)
        status_badge = '<span class="badge-ready">● In Memory</span>' if is_cached else '<span class="badge-warn">○ Will Load on Demand</span>'
        st.markdown(f"""
        <div style="font-size: 0.9rem; color: #475569; margin-bottom: 1rem;">
            Routing: <b>{src_choice}</b> (<code>{src_code}</code>) ➔ <b>{tgt_choice}</b> (<code>{tgt_code}</code>) | 
            Model: <span class="badge-model">{routed_model}</span> | Status: {status_badge}
        </div>
        """, unsafe_allow_html=True)

    # Text Input & Translation
    col_in, col_out = st.columns(2)

    with col_in:
        st.subheader("Source Text")
        input_text = st.text_area(
            "Enter text to translate",
            height=200,
            placeholder=f"Type or paste text in {src_choice}...",
            label_visibility="collapsed",
            key="input_box"
        )
        
        char_count = len(input_text)
        word_count = len(input_text.split()) if input_text.strip() else 0
        st.caption(f"Characters: **{char_count}** | Words: **{word_count}**")

        btn_col1, btn_col2 = st.columns([1, 1])
        with btn_col1:
            translate_btn = st.button("🚀 Translate Text", type="primary", use_container_width=True)
        with btn_col2:
            clear_btn = st.button("🧹 Clear Input", use_container_width=True)
            if clear_btn:
                st.session_state.translated_output = ""
                st.rerun()

    with col_out:
        st.subheader("Translated Output")
        
        if "translated_output" not in st.session_state:
            st.session_state.translated_output = ""

        if translate_btn:
            if not input_text.strip():
                st.warning("Please enter text to translate.")
            else:
                progress_container = st.empty()
                with progress_container.container():
                    st.info(f"Translating via IndicTrans2 ({routed_model})... Please wait.")
                
                start_time = time.time()
                try:
                    translated = st.session_state.engine.translate(
                        input_text,
                        src_lang=src_code,
                        tgt_lang=tgt_code,
                    )
                    elapsed = round(time.time() - start_time, 2)
                    progress_container.empty()
                    st.session_state.translated_output = translated
                    
                    # Log translation to session history
                    st.session_state.history.append({
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "source_lang": src_choice,
                        "target_lang": tgt_choice,
                        "source_code": src_code,
                        "target_code": tgt_code,
                        "model": routed_model.split("/")[-1],
                        "source_text": input_text,
                        "translated_text": translated,
                        "latency_sec": elapsed,
                    })
                    st.toast(f"Translated in {elapsed}s", icon="✅")

                except PermissionError as pe:
                    progress_container.empty()
                    st.error("🔒 Hugging Face Authentication Required")
                    st.markdown(f"""
                    **Details:** {pe}
                    
                    **Steps to resolve:**
                    1. Create a free account at [Hugging Face](https://huggingface.co).
                    2. Go to [{routed_model}](https://huggingface.co/{routed_model}) and click **"Agree and access"**.
                    3. Generate a token at [HF Access Tokens](https://huggingface.co/settings/tokens) with `Read` permission.
                    4. Enter your token in the left sidebar and click Translate again.
                    """)
                except MemoryError as me:
                    progress_container.empty()
                    st.error(f"⚠️ Memory Exhausted: {me}")
                except Exception as ex:
                    progress_container.empty()
                    st.error(f"Translation Error: {ex}")

        # Output Display Card
        if st.session_state.translated_output:
            st.markdown(f"""
            <div class="output-card">
                {st.session_state.translated_output}
            </div>
            """, unsafe_allow_html=True)

            st.write("")
            out_col1, out_col2 = st.columns(2)
            with out_col1:
                st.download_button(
                    label="📥 Download as TXT",
                    data=st.session_state.translated_output,
                    file_name=f"translation_{src_code}_to_{tgt_code}.txt",
                    mime="text/plain",
                    use_container_width=True
                )
            with out_col2:
                # Code block facilitates 1-click clipboard copying in Streamlit
                with st.expander("📋 Copy Translation"):
                    st.code(st.session_state.translated_output, language="")
        else:
            st.markdown("""
            <div class="output-card" style="display: flex; align-items: center; justify-content: center; color: #94a3b8;">
                Translation result will appear here.
            </div>
            """, unsafe_allow_html=True)


# -------------------------------------------------------------
# PAGE 2: PROJECT OVERVIEW
# -------------------------------------------------------------
elif navigation == "📋 Project Overview":
    st.markdown("""
    <div class="main-header">
        <h1>Project Overview</h1>
        <p>AI-Based Multilingual Indian Language Translator Using NLP</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ### 📄 Abstract
        India is home to over 1.4 billion people speaking 22 constitutionally recognized languages across 12 distinct scripts. 
        Language barriers significantly impede seamless communication, equitable access to digital governance, healthcare, 
        and high-quality education. 
        
        This project presents a genuine, deep-learning-driven **AI-Based Multilingual Indian Language Translator** built with 
        state-of-the-art **IndicTrans2** neural machine translation models developed by AI4Bharat and IIT Madras. 
        The system executes real Transformer encoder-decoder inference locally, supporting translations across English and 
        9 major Indian languages with script-aware tokenization and entity-preserving normalization.
        """)
        
        st.markdown("""
        ### 🎯 Problem Statement
        - Traditional statistical and rule-based machine translation systems fail on morphologically rich Indian languages.
        - Existing commercial solutions are proprietary, closed-source cloud APIs that compromise privacy and cannot operate offline.
        - Complex Indic scripts (Devanagari, Dravidian, Gurmukhi, Bengali, etc.) require specialized Unicode transliteration and numeral mapping that standard Western NLP models lack.
        """)

    with col2:
        st.markdown("""
        ### 🚀 Project Objectives
        1. **Genuine Neural Inference:** Run real pretrained Transformer sequence-to-sequence models without mock responses or web wrappers.
        2. **Intelligent Model Routing:** Dynamically dispatch translation queries across English-Indic (200M), Indic-English (200M), and Indic-Indic (320M) distilled models.
        3. **Advanced Script Preprocessing:** Integrate `IndicProcessor` for script transliteration, entity placeholders, and punctuation normalization.
        4. **Standard NLP Metric Evaluation:** Provide an evaluation pipeline using **SacreBLEU** and **chrF++** to rigorously benchmark translation quality.
        5. **Clean & Responsive Interface:** Deliver a production-grade Streamlit dashboard for interactive translation, history tracking, and metric analysis.
        """)

    st.markdown("---")
    st.markdown("### 🌍 Real-World Applications")
    app1, app2, app3, app4 = st.columns(4)
    with app1:
        st.info("🏛️ **Digital Governance**\n\nTranslating official government notifications and welfare schemes into regional languages.")
    with app2:
        st.success("🏥 **Healthcare Delivery**\n\nEnabling medical professionals to communicate diagnosis and prescriptions in the patient's native tongue.")
    with app3:
        st.warning("📚 **Inclusive Education**\n\nConverting English STEM textbooks and courseware into Indian regional languages.")
    with app4:
        st.error("🛒 **Rural E-Commerce**\n\nEmpowering vernacular buyers and artisans to access digital marketplaces in their local language.")

    st.markdown("---")
    st.markdown("### 🇮🇳 Supported Languages and Script Metadata")
    
    table_data = [
        {"Language": "English", "Script": "Latin", "IndicTrans Code": "eng_Latn", "ISO Code": "en", "Sample": "Hello World"},
        {"Language": "Hindi", "Script": "Devanagari", "IndicTrans Code": "hin_Deva", "ISO Code": "hi", "Sample": "नमस्ते दुनिया"},
        {"Language": "Gujarati", "Script": "Gujarati", "IndicTrans Code": "guj_Gujr", "ISO Code": "gu", "Sample": "નમસ્તે દુનિયા"},
        {"Language": "Marathi", "Script": "Devanagari", "IndicTrans Code": "mar_Deva", "ISO Code": "mr", "Sample": "नमस्कार जग"},
        {"Language": "Tamil", "Script": "Tamil", "IndicTrans Code": "tam_Taml", "ISO Code": "ta", "Sample": "வணக்கம் உலகம்"},
        {"Language": "Telugu", "Script": "Telugu", "IndicTrans Code": "tel_Telu", "ISO Code": "te", "Sample": "నమస్కారం ప్రపంచం"},
        {"Language": "Bengali", "Script": "Bengali", "IndicTrans Code": "ben_Beng", "ISO Code": "bn", "Sample": "নমস্কার বিশ্ব"},
        {"Language": "Punjabi", "Script": "Gurmukhi", "IndicTrans Code": "pan_Guru", "ISO Code": "pa", "Sample": "ਸਤਿ ਸ੍ਰੀ ਅਕਾਲ ਦੁਨਿਆ"},
        {"Language": "Kannada", "Script": "Kannada", "IndicTrans Code": "kan_Knda", "ISO Code": "kn", "Sample": "ನಮಸ್ಕಾರ ಪ್ರಪಂಚ"},
        {"Language": "Malayalam", "Script": "Malayalam", "IndicTrans Code": "mal_Mlym", "ISO Code": "ml", "Sample": "നമസ്കാരം ലോകം"},
    ]
    st.dataframe(pd.DataFrame(table_data), use_container_width=True)


# -------------------------------------------------------------
# PAGE 3: AI MODEL EXPLANATION
# -------------------------------------------------------------
elif navigation == "🧠 AI Model Architecture":
    st.markdown("""
    <div class="main-header">
        <h1>AI Model Architecture</h1>
        <p>In-Depth Breakdown of IndicTrans2 Transformer Seq2Seq Architecture & NLP Pipeline</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    ### 🔬 IndicTrans2 Model Family Overview
    IndicTrans2 is the state-of-the-art multilingual machine translation model series created by **AI4Bharat**, **IIT Madras**, 
    and published in major NLP venues. It supports all 22 scheduled Indian languages plus English, training on over 1 billion 
    high-quality parallel sentence tokens with knowledge distillation.
    """)

    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown("""
        <div class="metric-card">
            <h4>1. English-to-Indic Model</h4>
            <p><code>ai4bharat/indictrans2-en-indic-dist-200M</code></p>
            <ul>
                <li><b>Parameters:</b> ~200 Million</li>
                <li><b>Architecture:</b> 12-layer Encoder, 12-layer Decoder</li>
                <li><b>Hidden Size:</b> 1024, 16 Attention Heads</li>
                <li><b>Source:</b> English (<code>eng_Latn</code>)</li>
                <li><b>Target:</b> 22 Indian Languages</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with m2:
        st.markdown("""
        <div class="metric-card">
            <h4>2. Indic-to-English Model</h4>
            <p><code>ai4bharat/indictrans2-indic-en-dist-200M</code></p>
            <ul>
                <li><b>Parameters:</b> ~200 Million</li>
                <li><b>Architecture:</b> 12-layer Encoder, 12-layer Decoder</li>
                <li><b>Hidden Size:</b> 1024, 16 Attention Heads</li>
                <li><b>Source:</b> 22 Indian Languages</li>
                <li><b>Target:</b> English (<code>eng_Latn</code>)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with m3:
        st.markdown("""
        <div class="metric-card">
            <h4>3. Indic-to-Indic Model</h4>
            <p><code>ai4bharat/indictrans2-indic-indic-dist-320M</code></p>
            <ul>
                <li><b>Parameters:</b> ~320 Million</li>
                <li><b>Architecture:</b> 18-layer Encoder, 18-layer Decoder</li>
                <li><b>Hidden Size:</b> 1024, 16 Attention Heads</li>
                <li><b>Source:</b> 22 Indian Languages</li>
                <li><b>Target:</b> 22 Indian Languages</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🔄 End-to-End NLP Processing Pipeline")
    
    st.markdown("""
    The translation system executes a 7-stage neural pipeline to guarantee high fidelity across morphologically rich languages:
    """)

    pipeline_code = """
+---------------------------------------------------------------------------------------------------------+
|                                    INDIC NLP INFERENCE PIPELINE                                         |
+---------------------------------------------------------------------------------------------------------+
   [ Raw Input Text ]
          │
          ▼
   1. Punctuation & Digit Normalization: Convert native script numerals (०-९, ૦-૯, etc.) to standard digits.
          │
          ▼
   2. Entity & Placeholder Masking: Protect URLs, emails, dates, and numbers as <ID1>, <ID2> placeholders.
          │
          ▼
   3. Script Transliteration: Convert native Brahmic scripts to Devanagari representation for shared vocabulary.
          │
          ▼
   4. Language Tagging & Tokenization: Prepend `__src__ __tgt__` tokens and run subword BPE tokenization.
          │
          ▼
   5. Transformer Encoder-Decoder Inference:
         • Multi-Head Self-Attention layers encode contextual representation.
         • Cross-Attention decoder computes autoregressive target token probabilities via Beam Search (beams=5).
          │
          ▼
   6. Subword Detokenization & De-transliteration: Map pivot script tokens back into native target script.
          │
          ▼
   7. Placeholder Restoration & Postprocessing: Restore original masked entities (<ID1> -> URL/email/number).
          │
          ▼
   [ Final Translated Text ]
    """
    st.code(pipeline_code, language="text")

    st.markdown("""
    ### ⚙️ Why Pivot Transliteration is Crucial for Indic Languages
    Most Indian languages share significant phonetic, lexical, and grammatical structures derived from ancient Sanskrit 
    and Dravidian language families, but they use visually distinct alphabets (scripts). 
    
    By mapping phonetically equivalent characters to a single pivot script (Devanagari) during preprocessing:
    1. The model shares subword vocabulary across languages instead of maintaining isolated vocabularies per script.
    2. Cognates and loan words (e.g., 'Bharat', 'Shanti', 'Vidyalaya') share weights in the embedding matrix.
    3. The model parameters required are reduced by **over 60%** while improving cross-lingual zero-shot transfer!
    """)


# -------------------------------------------------------------
# PAGE 4: EVALUATION (SacreBLEU & chrF++)
# -------------------------------------------------------------
elif navigation == "📊 SacreBLEU Evaluation":
    st.markdown("""
    <div class="main-header">
        <h1>Translation Evaluation</h1>
        <p>Quantitative Benchmarking Using SacreBLEU and chrF++ Standard NLP Metrics</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    ### 📐 Machine Translation Metric Fundamentals
    To avoid subjective or fabricated claims, this application calculates standard industry metrics via **SacreBLEU**:
    
    - **BLEU (Bilingual Evaluation Understudy):** Measures geometric mean of precision of $n$-grams (typically $n=1$ to $4$) with brevity penalty. 
      Best for word-level matches.
    - **chrF++ (Character n-gram F-score + word order):** Measures character-level $n$-gram overlap combined with word $n$-grams. 
      **Highly recommended for Indian languages** because Dravidian and Indo-Aryan languages are agglutinative and morphologically complex, where small grammatical inflections alter word endings without breaking root meaning.
    """)

    st.markdown("---")
    st.subheader("🧪 Live Interactive Evaluation")
    st.write("Compare any candidate translation against a human reference to compute genuine SacreBLEU metrics:")

    ev_col1, ev_col2 = st.columns(2)
    with ev_col1:
        eval_cand = st.text_area("Model Prediction / Candidate Text", "कृत्रिम बुद्धिमत्ता आधुनिक सॉफ्टवेयर विकास को गति देती है।", height=100)
    with ev_col2:
        eval_ref = st.text_area("Ground Truth / Human Reference", "आर्टिफिशियल इंटेलिजेंस आधुनिक सॉफ्टवेयर विकास को गति प्रदान करता है।", height=100)

    if st.button("📊 Calculate SacreBLEU Metrics", type="primary"):
        metrics = compute_metrics(eval_cand, eval_ref)
        m_col1, m_col2 = st.columns(2)
        with m_col1:
            st.metric("BLEU Score", f"{metrics['bleu']} / 100")
            st.caption("Standard word-level n-gram precision metric.")
        with m_col2:
            st.metric("chrF++ Score", f"{metrics['chrf_plus_plus']} / 100")
            st.caption("Character n-gram F-score with word order 2 (morphology-aware).")

    st.markdown("---")
    st.subheader("📋 Academic Benchmark Suite (7 Test Pairs)")
    st.write("The benchmark suite includes the required test translation pairs across all 3 IndicTrans2 model categories:")

    if st.button("⚡ Run Full Benchmark Suite Through Model Engine"):
        with st.spinner("Executing benchmark translations and computing real metrics..."):
            df_results = run_benchmark(st.session_state.engine)
            st.dataframe(df_results, use_container_width=True)
            
            evaluated_only = df_results[df_results["Status"] == "Evaluated (Real Model)"]
            if not evaluated_only.empty:
                avg_bleu = evaluated_only["BLEU"].mean()
                avg_chrf = evaluated_only["chrF++"].mean()
                c1, c2 = st.columns(2)
                c1.metric("Average Benchmark BLEU", f"{round(avg_bleu, 2)} / 100")
                c2.metric("Average Benchmark chrF++", f"{round(avg_chrf, 2)} / 100")
    else:
        # Display verified benchmark results from genuine model execution
        df_saved = get_saved_benchmark()
        if not df_saved.empty:
            st.markdown("#### ✅ Empirical Results from Real Model Execution")
            st.dataframe(df_saved, use_container_width=True)

            avg_bleu = df_saved["BLEU"].mean()
            avg_chrf = df_saved["chrF++"].mean()
            avg_time = df_saved["Latency (s)"].mean()

            c1, c2, c3 = st.columns(3)
            c1.metric("Average Real BLEU", f"{round(avg_bleu, 2)} / 100")
            c2.metric("Average Real chrF++", f"{round(avg_chrf, 2)} / 100")
            c3.metric("Avg Inference Latency", f"{round(avg_time, 2)}s")
        else:
            df_bench = pd.DataFrame(BENCHMARK_PAIRS)
            st.dataframe(
                df_bench[["pair_name", "src_code", "tgt_code", "model_id", "input_text", "reference"]],
                use_container_width=True
            )


# -------------------------------------------------------------
# PAGE 5: TRANSLATION HISTORY
# -------------------------------------------------------------
elif navigation == "📜 Translation History":
    st.markdown("""
    <div class="main-header">
        <h1>Translation Session History</h1>
        <p>Review and Export Translations Generated During the Current Session</p>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.history:
        st.info("No translations generated yet in this session. Go to the 'Neural Translator' tab to translate sentences!")
    else:
        st.write(f"Total translations recorded: **{len(st.session_state.history)}**")

        df_hist = pd.DataFrame(st.session_state.history)
        st.dataframe(df_hist, use_container_width=True)

        col_h1, col_h2 = st.columns(2)
        with col_h1:
            csv_data = df_hist.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Export History as CSV",
                data=csv_data,
                file_name="translation_history.csv",
                mime="text/csv",
                use_container_width=True
            )
        with col_h2:
            if st.button("🗑️ Clear Session History", use_container_width=True):
                st.session_state.history = []
                st.rerun()
