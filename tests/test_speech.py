"""
Unit tests for Speech Recognition (faster-whisper) and Text-to-Speech (TTS) engines.
Verifies all 22 scheduled Indian languages + English speech coverage.
"""

import unittest
import io
import wave
import struct
from model_engine.speech_recognition import (
    SpeechRecognizer,
    FLORES_TO_WHISPER_LANG,
    WHISPER_TO_FLORES_LANG,
)
from model_engine.text_to_speech import TextToSpeechEngine, FLORES_TO_TTS_LANG
from model_engine.config import SUPPORTED_LANGUAGES, LANGUAGE_SPEECH_CAPABILITIES


class TestSpeechPipeline(unittest.TestCase):
    def setUp(self):
        self.tts = TextToSpeechEngine()
        self.asr = SpeechRecognizer(model_size="base", device="cpu", compute_type="int8")

    def test_all_22_languages_defined(self):
        """Ensure all 22 scheduled languages are defined with speech capabilities."""
        self.assertGreaterEqual(len(SUPPORTED_LANGUAGES), 23)
        for name, code in SUPPORTED_LANGUAGES.items():
            self.assertIn(code, LANGUAGE_SPEECH_CAPABILITIES, f"Missing capability record for {name} ({code})")

    def test_tts_supported_languages(self):
        """Test generating real speech audio bytes for supported languages."""
        for code in ["hin_Deva", "eng_Latn", "tam_Taml", "ben_Beng"]:
            self.assertTrue(self.tts.is_language_supported(code))
            audio_bytes, mime = self.tts.synthesize("test", code)
            self.assertIsInstance(audio_bytes, bytes)
            self.assertGreater(len(audio_bytes), 100)
            self.assertEqual(mime, "audio/mp3")

    def test_tts_unsupported_language_raises_error(self):
        """Languages without verified TTS (e.g. Odia or Bodo) should raise clear error, never fake audio."""
        self.assertFalse(self.tts.is_language_supported("ory_Orya"))
        with self.assertRaises(ValueError):
            self.tts.synthesize("ଓଡ଼ିଆ", "ory_Orya")

    def test_asr_supported_languages(self):
        """Verify Whisper supports key Indic languages including Assamese, Sanskrit, Sindhi."""
        for code in ["eng_Latn", "hin_Deva", "asm_Beng", "san_Deva", "snd_Arab", "urd_Arab"]:
            self.assertTrue(self.asr.is_language_supported(code))

    def test_asr_transcribe_synthetic_audio(self):
        """Test ASR transcription with valid 1-second silence WAV buffer."""
        sample_rate = 16000
        duration = 1.0
        num_samples = int(sample_rate * duration)
        
        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            data = struct.pack(f'<{num_samples}h', *[0] * num_samples)
            wav.writeframes(data)
        
        buffer.seek(0)
        text, meta = self.asr.transcribe(buffer, source_language_code="eng_Latn")
        self.assertIsInstance(text, str)
        self.assertIn("duration", meta)


if __name__ == "__main__":
    unittest.main()
