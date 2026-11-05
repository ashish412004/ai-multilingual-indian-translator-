"""
Model Engine package for AI-Based Multilingual Indian Language Translator.
Includes Text Translation, Speech Recognition (ASR), and Text-to-Speech (TTS).
"""
from .config import (
    SUPPORTED_LANGUAGES,
    CODE_TO_LANGUAGE,
    LANGUAGE_NATIVE_NAMES,
    MODEL_EN_INDIC,
    MODEL_INDIC_EN,
    MODEL_INDIC_INDIC,
    get_model_for_pair,
)
from .translator import TranslationEngine
from .speech_recognition import SpeechRecognizer, FLORES_TO_WHISPER_LANG, WHISPER_TO_FLORES_LANG
from .text_to_speech import TextToSpeechEngine, FLORES_TO_TTS_LANG

__all__ = [
    "SUPPORTED_LANGUAGES",
    "CODE_TO_LANGUAGE",
    "LANGUAGE_NATIVE_NAMES",
    "MODEL_EN_INDIC",
    "MODEL_INDIC_EN",
    "MODEL_INDIC_INDIC",
    "get_model_for_pair",
    "TranslationEngine",
    "SpeechRecognizer",
    "FLORES_TO_WHISPER_LANG",
    "WHISPER_TO_FLORES_LANG",
    "TextToSpeechEngine",
    "FLORES_TO_TTS_LANG",
]
