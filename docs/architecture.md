# Architecture & Technical Design: AI-Based Multilingual Indian Language Translator

## 1. Executive Summary
This document provides the technical architecture and neural engineering design of the **AI-Based Multilingual Indian Language Translator Using NLP**. The system implements real sequence-to-sequence translation across English and 9 major scheduled Indian languages (Hindi, Gujarati, Marathi, Tamil, Telugu, Bengali, Punjabi, Kannada, Malayalam) using state-of-the-art pretrained **IndicTrans2** Transformer models from AI4Bharat (IIT Madras).

---

## 2. IndicTrans2 Neural Architecture

IndicTrans2 is a multilingual sequence-to-sequence Transformer architecture based on the encoder-decoder paradigm. Rather than training a generic multilingual model, IndicTrans2 utilizes dedicated distilled models specialized for specific translation directions:

```
+-------------------------------------------------------------+
|                     ROUTING CONTROLLER                      |
+-------------------------------------------------------------+
                              │
             ┌────────────────┼────────────────┐
             ▼                ▼                ▼
     [English -> Indic]  [Indic -> English] [Indic -> Indic]
       (200M Params)       (200M Params)     (320M Params)
             │                │                │
             └────────────────┼────────────────┘
                              ▼
           [Transformer Seq2Seq Encoder-Decoder]
```

### 2.1 Model Specifications

| Property | English-to-Indic | Indic-to-English | Indic-to-Indic |
| :--- | :--- | :--- | :--- |
| **Model ID** | `ai4bharat/indictrans2-en-indic-dist-200M` | `ai4bharat/indictrans2-indic-en-dist-200M` | `ai4bharat/indictrans2-indic-indic-dist-320M` |
| **Parameters** | ~200 Million | ~200 Million | ~320 Million |
| **Encoder Layers**| 12 | 12 | 18 |
| **Decoder Layers**| 12 | 12 | 18 |
| **Hidden Dim** | 1024 | 1024 | 1024 |
| **Attention Heads**| 16 | 16 | 16 |
| **Feedforward Dim**| 4096 | 4096 | 4096 |
| **Precision** | Float32 (CPU) / Float16 (CUDA) | Float32 (CPU) / Float16 (CUDA) | Float32 (CPU) / Float16 (CUDA) |

---

## 3. End-to-End NLP Processing Pipeline

Machine translation between morphologically rich Indian languages requires specialized preprocessing to preserve named entities, handle diverse Brahmic scripts, and maintain phonetic equivalences.

### 3.1 Pipeline Stages

```
   Raw User Input Sentence
              │
              ▼
   [Stage 1: Punctuation Normalization]
   - Replaces smart quotes, non-breaking spaces, em-dashes, and irregular whitespace
              │
              ▼
   [Stage 2: Indic Numeral Translation]
   - Unifies native script numerals (०-९, ૦-૯, ౦-౯, etc.) into ASCII digits
              │
              ▼
   [Stage 3: Entity Masking & Placeholder Injection]
   - Detects URLs, email addresses, phone numbers, and time expressions
   - Replaces them with sequential placeholders: <ID1>, <ID2>...
              │
              ▼
   [Stage 4: Pivot Transliteration]
   - Uses Brahmic phonetic equivalence to transliterate non-Devanagari scripts
     into a shared pivot representation (Hindi/Devanagari)
              │
              ▼
   [Stage 5: Language Tagging & BPE Tokenization]
   - Prepends source and target tokens: e.g. "eng_Latn hin_Deva <tokens>"
   - Subword tokenization using SentencePiece
              │
              ▼
   [Stage 6: Autoregressive Beam Search Generation]
   - 12/18 layer Encoder computes contextual representations
   - 12/18 layer Decoder generates target token IDs with beam_width=5
              │
              ▼
   [Stage 7: Detokenization, De-transliteration & Entity Restoration]
   - Decodes subwords into text string
   - Re-transliterates pivot Devanagari into native target script (e.g. Tamil, Gujarati)
   - Restores <ID1>, <ID2> placeholders with original uncorrupted entities
              │
              ▼
   Final Translated Sentence
```

---

## 4. Evaluation Methodology

Translation quality is objectively evaluated using **SacreBLEU**, avoiding subjective or hardcoded accuracy numbers:

### 4.1 Metric 1: BLEU (Bilingual Evaluation Understudy)
Standard corpus and sentence-level metric computing modified $n$-gram precision:
$$\text{BLEU} = \text{BP} \cdot \exp \left( \sum_{n=1}^{4} w_n \log p_n \right)$$
where $\text{BP} = \min(1, e^{1 - r/c})$ is the Brevity Penalty.

### 4.2 Metric 2: chrF++ (Character n-gram F-score + word order)
Indian languages exhibit rich inflectional morphology and compounding. Standard word BLEU can penalize correct translations that vary only by small case or gender suffixes. **chrF++** computes character $n$-gram overlaps supplemented by word $n$-grams (word order $\beta=2$):
$$\text{chrF++} = (1 + \beta^2) \frac{\text{CHRP} \cdot \text{CHRR}}{\beta^2 \text{CHRP} + \text{CHRR}}$$
This correlates significantly higher with human translation judgments for Indic languages.

---

## 5. Security & Hugging Face Gated Repository Model

The AI4Bharat IndicTrans2 model family is hosted under a gated license on Hugging Face to encourage responsible AI attribution.
- The project **never hardcodes credentials** or access tokens in source code.
- Tokens can be provided via environment variable `HF_TOKEN`, the Hugging Face CLI (`huggingface-cli login`), or dynamically typed into the Streamlit UI.
- The application provides clear error notifications and direct links to model acknowledgement pages if access has not yet been accepted.
