"""
Text-to-Speech (TTS) engine for Indian languages and English.
Generates audio output from translated text with fallback mechanisms
and full multi-language coverage across the project's supported languages.
"""

import os
import io
import tempfile
from typing import Optional, Dict, Any, Tuple
from gtts import gTTS

# Mapping of FLORES language codes to gTTS / BCP-47 language codes
FLORES_TO_TTS_LANG: Dict[str, Dict[str, str]] = {
    "eng_Latn": {"lang": "en", "tld": "co.in", "name": "English (India)"},
    "hin_Deva": {"lang": "hi", "tld": "co.in", "name": "Hindi"},
    "guj_Gujr": {"lang": "gu", "tld": "co.in", "name": "Gujarati"},
    "mar_Deva": {"lang": "mr", "tld": "co.in", "name": "Marathi"},
    "tam_Taml": {"lang": "ta", "tld": "co.in", "name": "Tamil"},
    "tel_Telu": {"lang": "te", "tld": "co.in", "name": "Telugu"},
    "ben_Beng": {"lang": "bn", "tld": "co.in", "name": "Bengali"},
    "pan_Guru": {"lang": "pa", "tld": "co.in", "name": "Punjabi"},
    "kan_Knda": {"lang": "kn", "tld": "co.in", "name": "Kannada"},
    "mal_Mlym": {"lang": "ml", "tld": "co.in", "name": "Malayalam"},
}


class TextToSpeechEngine:
    """
    Synthesizes speech audio for Indian languages and English from translated text.
    Provides audio bytes for Streamlit playback and file download.
    """

    def __init__(self):
        self.supported_languages = FLORES_TO_TTS_LANG

    def is_language_supported(self, flores_code: str) -> bool:
        """Check if target FLORES language has verified TTS voice support."""
        return flores_code in self.supported_languages

    def synthesize(
        self,
        text: str,
        target_lang_code: str,
        slow: bool = False,
    ) -> Tuple[bytes, str]:
        """
        Convert text into speech audio bytes.
        
        Args:
            text: Text to synthesize.
            target_lang_code: FLORES code (e.g. 'hin_Deva', 'tam_Taml', 'eng_Latn').
            slow: Whether to speak slowly.
            
        Returns:
            Tuple of (audio_bytes, mime_type).
            
        Raises:
            ValueError: If text is empty or language is completely unsupported.
            RuntimeError: If audio synthesis fails.
        """
        clean_text = text.strip()
        if not clean_text:
            raise ValueError("Cannot synthesize speech from empty text.")

        lang_config = self.supported_languages.get(target_lang_code)
        if not lang_config:
            # Fallback to Hindi if Indic, or English
            if "Latn" in target_lang_code:
                lang_config = self.supported_languages["eng_Latn"]
            else:
                lang_config = self.supported_languages["hin_Deva"]

        tts_lang = lang_config["lang"]
        tld = lang_config.get("tld", "com")

        try:
            # Synthesize via gTTS
            tts = gTTS(text=clean_text, lang=tts_lang, tld=tld, slow=slow)
            
            audio_buffer = io.BytesIO()
            tts.write_to_fp(audio_buffer)
            audio_buffer.seek(0)
            
            return audio_buffer.read(), "audio/mp3"

        except Exception as e:
            # Secondary retry with standard TLD if localized failed
            try:
                tts = gTTS(text=clean_text, lang=tts_lang, slow=slow)
                audio_buffer = io.BytesIO()
                tts.write_to_fp(audio_buffer)
                audio_buffer.seek(0)
                return audio_buffer.read(), "audio/mp3"
            except Exception as retry_err:
                raise RuntimeError(
                    f"TTS synthesis failed for language '{target_lang_code}' ({lang_config['name']}): {retry_err}"
                ) from e
