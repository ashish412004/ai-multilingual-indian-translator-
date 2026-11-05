"""
Unit tests for Speech Recognition (faster-whisper) and Text-to-Speech (TTS) engines.
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
from model_engine.config import SUPPORTED_LANGUAGES


class TestSpeechPipeline(unittest.TestCase):
    def setUp(self):
        self.tts = TextToSpeechEngine()
        self.asr = SpeechRecognizer(model_size="base", device="cpu", compute_type="int8")

    def test_language_mappings_coverage(self):
        """Ensure all 10 project languages are mapped for ASR and TTS."""
        for lang_name, code in SUPPORTED_LANGUAGES.items():
            self.assertIn(code, FLORES_TO_WHISPER_LANG, f"Missing ASR mapping for {lang_name} ({code})")
            self.assertIn(code, FLORES_TO_TTS_LANG, f"Missing TTS mapping for {lang_name} ({code})")

    def test_tts_synthesis_hindi(self):
        """Test generating real speech audio bytes for Hindi."""
        text = "नमस्ते, यह एक परीक्षण है।"
        audio_bytes, mime = self.tts.synthesize(text, "hin_Deva")
        self.assertIsInstance(audio_bytes, bytes)
        self.assertGreater(len(audio_bytes), 1000)
        self.assertEqual(mime, "audio/mp3")

    def test_tts_synthesis_english(self):
        """Test generating real speech audio bytes for English."""
        text = "Hello, this is a test audio generation."
        audio_bytes, mime = self.tts.synthesize(text, "eng_Latn")
        self.assertIsInstance(audio_bytes, bytes)
        self.assertGreater(len(audio_bytes), 1000)
        self.assertEqual(mime, "audio/mp3")

    def test_tts_empty_text_error(self):
        """Empty text should raise ValueError."""
        with self.assertRaises(ValueError):
            self.tts.synthesize("", "hin_Deva")

    def test_asr_model_initialization(self):
        """Verify Whisper model loads successfully."""
        model = self.asr.load_model()
        self.assertIsNotNone(model)
        self.assertTrue(self.asr.is_loaded())

    def test_asr_transcribe_synthetic_audio(self):
        """Test ASR transcription with valid 1-second silence WAV buffer."""
        # Create 1 second 16kHz mono WAV file in memory
        sample_rate = 16000
        duration = 1.0
        num_samples = int(sample_rate * duration)
        
        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            # 16-bit silence
            data = struct.pack(f'<{num_samples}h', *[0] * num_samples)
            wav.writeframes(data)
        
        buffer.seek(0)
        text, meta = self.asr.transcribe(buffer, source_language_code="eng_Latn")
        self.assertIsInstance(text, str)
        self.assertIn("duration", meta)
        self.assertAlmostEqual(meta["duration"], 1.0, delta=0.2)


if __name__ == "__main__":
    unittest.main()
