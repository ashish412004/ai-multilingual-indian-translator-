"""
Configuration and constants for the AI-Based Multilingual Indian Language Translator.
Supports all 22 scheduled Indian languages (with official script variants) plus English.
"""

from typing import Dict, List, Tuple

# Pretrained Model IDs from Hugging Face AI4Bharat
MODEL_EN_INDIC = "ai4bharat/indictrans2-en-indic-dist-200M"
MODEL_INDIC_EN = "ai4bharat/indictrans2-indic-en-dist-200M"
MODEL_INDIC_INDIC = "ai4bharat/indictrans2-indic-indic-dist-320M"

# All 22 Scheduled Indian Languages + English and their official FLORES codes
SUPPORTED_LANGUAGES: Dict[str, str] = {
    "English": "eng_Latn",
    "Assamese": "asm_Beng",
    "Bengali": "ben_Beng",
    "Bodo": "brx_Deva",
    "Dogri": "doi_Deva",
    "Gujarati": "guj_Gujr",
    "Hindi": "hin_Deva",
    "Kannada": "kan_Knda",
    "Kashmiri (Arabic)": "kas_Arab",
    "Kashmiri (Devanagari)": "kas_Deva",
    "Konkani": "gom_Deva",
    "Maithili": "mai_Deva",
    "Malayalam": "mal_Mlym",
    "Manipuri (Bengali)": "mni_Beng",
    "Manipuri (Meitei)": "mni_Mtei",
    "Marathi": "mar_Deva",
    "Nepali": "npi_Deva",
    "Odia": "ory_Orya",
    "Punjabi": "pan_Guru",
    "Sanskrit": "san_Deva",
    "Santali": "sat_Olck",
    "Sindhi (Arabic)": "snd_Arab",
    "Sindhi (Devanagari)": "snd_Deva",
    "Tamil": "tam_Taml",
    "Telugu": "tel_Telu",
    "Urdu": "urd_Arab",
}

# Reverse mapping: FLORES code -> Display name
CODE_TO_LANGUAGE = {v: k for k, v in SUPPORTED_LANGUAGES.items()}

# Native script titles for all 22 languages
LANGUAGE_NATIVE_NAMES: Dict[str, str] = {
    "English": "English",
    "Assamese": "অসমীয়া (Assamese)",
    "Bengali": "বাংলা (Bengali)",
    "Bodo": "बड़ो (Bodo)",
    "Dogri": "डोगरी (Dogri)",
    "Gujarati": "ગુજરાતી (Gujarati)",
    "Hindi": "हिन्दी (Hindi)",
    "Kannada": "ಕನ್ನಡ (Kannada)",
    "Kashmiri (Arabic)": "کٲشُر (Kashmiri - Arabic)",
    "Kashmiri (Devanagari)": "कॉशुर (Kashmiri - Devanagari)",
    "Konkani": "कोंकणी (Konkani)",
    "Maithili": "मैथिली (Maithili)",
    "Malayalam": "മലയാളം (Malayalam)",
    "Manipuri (Bengali)": "মৈতৈলোন্ (Manipuri - Bengali)",
    "Manipuri (Meitei)": "ꯃꯤꯇꯩꯂꯣꯟ (Manipuri - Meitei)",
    "Marathi": "मराठी (Marathi)",
    "Nepali": "नेपाली (Nepali)",
    "Odia": "ଓଡ଼ିଆ (Odia)",
    "Punjabi": "ਪੰਜਾਬੀ (Punjabi)",
    "Sanskrit": "संस्कृतम् (Sanskrit)",
    "Santali": "ᱥᱟᱱᱛᱟᱲᱤ (Santali)",
    "Sindhi (Arabic)": "سنڌي (Sindhi - Arabic)",
    "Sindhi (Devanagari)": "सिन्धी (Sindhi - Devanagari)",
    "Tamil": "தமிழ் (Tamil)",
    "Telugu": "తెలుగు (Telugu)",
    "Urdu": "اردو (Urdu)",
}

# Verified speech capability matrix
# asr_code: ISO code for Whisper or None
# tts_code: ISO code for TTS or None
LANGUAGE_SPEECH_CAPABILITIES: Dict[str, Dict[str, any]] = {
    "eng_Latn": {"asr": "en", "tts": "en", "status": "Full S2ST"},
    "hin_Deva": {"asr": "hi", "tts": "hi", "status": "Full S2ST"},
    "ben_Beng": {"asr": "bn", "tts": "bn", "status": "Full S2ST"},
    "guj_Gujr": {"asr": "gu", "tts": "gu", "status": "Full S2ST"},
    "kan_Knda": {"asr": "kn", "tts": "kn", "status": "Full S2ST"},
    "mal_Mlym": {"asr": "ml", "tts": "ml", "status": "Full S2ST"},
    "mar_Deva": {"asr": "mr", "tts": "mr", "status": "Full S2ST"},
    "npi_Deva": {"asr": "ne", "tts": "ne", "status": "Full S2ST"},
    "pan_Guru": {"asr": "pa", "tts": "pa", "status": "Full S2ST"},
    "tam_Taml": {"asr": "ta", "tts": "ta", "status": "Full S2ST"},
    "tel_Telu": {"asr": "te", "tts": "te", "status": "Full S2ST"},
    "urd_Arab": {"asr": "ur", "tts": "ur", "status": "Full S2ST"},
    "asm_Beng": {"asr": "as", "tts": None, "status": "Voice Input -> Text"},
    "san_Deva": {"asr": "sa", "tts": None, "status": "Voice Input -> Text"},
    "snd_Arab": {"asr": "sd", "tts": None, "status": "Voice Input -> Text"},
    "snd_Deva": {"asr": "sd", "tts": None, "status": "Voice Input -> Text"},
    "brx_Deva": {"asr": None, "tts": None, "status": "Text Translation Only"},
    "doi_Deva": {"asr": None, "tts": None, "status": "Text Translation Only"},
    "gom_Deva": {"asr": None, "tts": None, "status": "Text Translation Only"},
    "kas_Arab": {"asr": None, "tts": None, "status": "Text Translation Only"},
    "kas_Deva": {"asr": None, "tts": None, "status": "Text Translation Only"},
    "mai_Deva": {"asr": None, "tts": None, "status": "Text Translation Only"},
    "mni_Beng": {"asr": None, "tts": None, "status": "Text Translation Only"},
    "mni_Mtei": {"asr": None, "tts": None, "status": "Text Translation Only"},
    "ory_Orya": {"asr": None, "tts": None, "status": "Text Translation Only"},
    "sat_Olck": {"asr": None, "tts": None, "status": "Text Translation Only"},
}


def get_model_for_pair(src_code: str, tgt_code: str) -> str:
    """
    Determine which IndicTrans2 model to route the translation request to.
    
    Returns:
        Model ID string, or 'IDENTITY' if src == tgt
    """
    if src_code == tgt_code:
        return "IDENTITY"
    
    is_src_en = (src_code == "eng_Latn")
    is_tgt_en = (tgt_code == "eng_Latn")
    
    if is_src_en and not is_tgt_en:
        return MODEL_EN_INDIC
    elif not is_src_en and is_tgt_en:
        return MODEL_INDIC_EN
    else:
        return MODEL_INDIC_INDIC
