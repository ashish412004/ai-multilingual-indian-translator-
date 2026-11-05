"""
Text-to-Speech (TTS) engine for Indian languages and English.
Generates audio output from translated text with verified voice mapping,
explicit capability checking, and clear feedback when TTS is unavailable.
"""

import io
from typing import Optional, Dict, Any, Tuple
from gtts import gTTS
from .config import LANGUAGE_SPEECH_CAPABILITIES

# Verified language configurations for TTS synthesis
FLORES_TO_TTS_LANG: Dict[str, Dict[str, str]] = {
    "eng_Latn": {"lang": "en", "tld": "co.in", "name": "English"},
    "hin_Deva": {"lang": "hi", "tld": "co.in", "name": "Hindi"},
    "ben_Beng": {"lang": "bn", "tld": "co.in", "name": "Bengali"},
    "guj_Gujr": {"lang": "gu", "tld": "co.in", "name": "Gujarati"},
    "kan_Knda": {"lang": "kn", "tld": "co.in", "name": "Kannada"},
    "mal_Mlym": {"lang": "ml", "tld": "co.in", "name": "Malayalam"},
    "mar_Deva": {"lang": "mr", "tld": "co.in", "name": "Marathi"},
    "npi_Deva": {"lang": "ne", "tld": "co.in", "name": "Nepali"},
    "pan_Guru": {"lang": "pa", "tld": "co.in", "name": "Punjabi"},
    "tam_Taml": {"lang": "ta", "tld": "co.in", "name": "Tamil"},
    "tel_Telu": {"lang": "te", "tld": "co.in", "name": "Telugu"},
    "urd_Arab": {"lang": "ur", "tld": "co.in", "name": "Urdu"},
}


class TextToSpeechEngine:
    """
    Synthesizes speech audio for supported Indian languages and English from translated text.
    Provides audio bytes for Streamlit playback and file download.
    Never silently swaps one language for another.
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
            ValueError: If text is empty or TTS is not supported for this language.
            RuntimeError: If audio synthesis fails.
        """
        clean_text = text.strip()
        if not clean_text:
            raise ValueError("Cannot synthesize speech from empty text.")

        if not self.is_language_supported(target_lang_code):
            raise ValueError(
                f"Text-to-Speech voice is currently unavailable for language code '{target_lang_code}'. "
                f"Translation has been completed in text format."
            )

        lang_config = self.supported_languages[target_lang_code]
        tts_lang = lang_config["lang"]
        tld = lang_config.get("tld", "com")

        try:
            tts = gTTS(text=clean_text, lang=tts_lang, tld=tld, slow=slow)
            audio_buffer = io.BytesIO()
            tts.write_to_fp(audio_buffer)
            audio_buffer.seek(0)
            return audio_buffer.read(), "audio/mp3"

        except Exception as e:
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
