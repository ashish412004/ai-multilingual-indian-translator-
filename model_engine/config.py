"""
Configuration and constants for the AI-Based Multilingual Indian Language Translator.
"""

from typing import Dict

# Model IDs from Hugging Face AI4Bharat
MODEL_EN_INDIC = "ai4bharat/indictrans2-en-indic-dist-200M"
MODEL_INDIC_EN = "ai4bharat/indictrans2-indic-en-dist-200M"
MODEL_INDIC_INDIC = "ai4bharat/indictrans2-indic-indic-dist-320M"

# Supported 10 languages and official FLORES/IndicTrans2 codes
SUPPORTED_LANGUAGES: Dict[str, str] = {
    "English": "eng_Latn",
    "Hindi": "hin_Deva",
    "Gujarati": "guj_Gujr",
    "Marathi": "mar_Deva",
    "Tamil": "tam_Taml",
    "Telugu": "tel_Telu",
    "Bengali": "ben_Beng",
    "Punjabi": "pan_Guru",
    "Kannada": "kan_Knda",
    "Malayalam": "mal_Mlym",
}

# Reverse mapping: code -> Display name
CODE_TO_LANGUAGE = {v: k for k, v in SUPPORTED_LANGUAGES.items()}

# Language script representation samples for UI
LANGUAGE_NATIVE_NAMES = {
    "English": "English",
    "Hindi": "हिन्दी (Hindi)",
    "Gujarati": "ગુજરાતી (Gujarati)",
    "Marathi": "मराठी (Marathi)",
    "Tamil": "தமிழ் (Tamil)",
    "Telugu": "తెలుగు (Telugu)",
    "Bengali": "বাংলা (Bengali)",
    "Punjabi": "ਪੰਜਾਬੀ (Punjabi)",
    "Kannada": "ಕನ್ನಡ (Kannada)",
    "Malayalam": "മലയാളം (Malayalam)",
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
