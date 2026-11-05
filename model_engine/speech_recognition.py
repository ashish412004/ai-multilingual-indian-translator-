"""
Speech recognition (ASR) engine powered by faster-whisper.
Converts spoken audio (microphone recordings or audio uploads) into text
with automatic or guided language detection across supported Indian languages and English.
"""

import os
import io
import tempfile
from typing import Optional, Tuple, Dict, Any
from faster_whisper import WhisperModel
from .config import LANGUAGE_SPEECH_CAPABILITIES

# Map IndicTrans2/FLORES codes to ISO 639-1 two-letter codes supported by Whisper
FLORES_TO_WHISPER_LANG: Dict[str, str] = {
    code: meta["asr"]
    for code, meta in LANGUAGE_SPEECH_CAPABILITIES.items()
    if meta.get("asr") is not None
}

# Reverse map: 2-letter ISO code to primary FLORES code
WHISPER_TO_FLORES_LANG: Dict[str, str] = {
    "en": "eng_Latn",
    "hi": "hin_Deva",
    "bn": "ben_Beng",
    "gu": "guj_Gujr",
    "kn": "kan_Knda",
    "ml": "mal_Mlym",
    "mr": "mar_Deva",
    "ne": "npi_Deva",
    "pa": "pan_Guru",
    "ta": "tam_Taml",
    "te": "tel_Telu",
    "ur": "urd_Arab",
    "as": "asm_Beng",
    "sa": "san_Deva",
    "sd": "snd_Arab",
}


class SpeechRecognizer:
    """
    Manages faster-whisper model loading, caching, audio processing,
    and speech-to-text inference.
    """

    def __init__(self, model_size: str = "base", device: str = "cpu", compute_type: str = "int8"):
        """
        Initialize the speech recognizer.
        
        Args:
            model_size: Whisper model size ('tiny', 'base', 'small', 'medium'). Default 'base'.
            device: 'cpu' or 'cuda'.
            compute_type: Quantization precision ('int8', 'float32', 'float16'). Default 'int8'.
        """
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self._model: Optional[WhisperModel] = None

    def load_model(self) -> WhisperModel:
        """Load and cache the Whisper model in memory."""
        if self._model is None:
            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
                cpu_threads=4,
            )
        return self._model

    def is_loaded(self) -> bool:
        """Check if model is already loaded in memory."""
        return self._model is not None

    def is_language_supported(self, flores_code: str) -> bool:
        """Check whether speech recognition is verified for this language."""
        return flores_code in FLORES_TO_WHISPER_LANG

    def transcribe(
        self,
        audio_data: Any,
        source_language_code: Optional[str] = None,
        beam_size: int = 5,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Transcribe audio input into text.
        
        Args:
            audio_data: Bytes, BytesIO, UploadedFile, or file path.
            source_language_code: Optional FLORES code (e.g. 'hin_Deva') or ISO ('hi').
            beam_size: Beam search size for decoding.
            
        Returns:
            Tuple of (transcribed_text, info_dict)
            where info_dict contains 'detected_language', 'language_probability', 'duration'.
        """
        model = self.load_model()
        
        # Determine language parameter for Whisper
        whisper_lang = None
        if source_language_code:
            if source_language_code in FLORES_TO_WHISPER_LANG:
                whisper_lang = FLORES_TO_WHISPER_LANG[source_language_code]
            elif len(source_language_code) == 2:
                whisper_lang = source_language_code

        # If audio_data is bytes or file-like, write to a temporary file
        temp_path = None
        try:
            if isinstance(audio_data, str) and os.path.isfile(audio_data):
                audio_input = audio_data
            else:
                if hasattr(audio_data, "read"):
                    data_bytes = audio_data.read()
                    if hasattr(audio_data, "seek"):
                        audio_data.seek(0)
                elif isinstance(audio_data, (bytes, bytearray)):
                    data_bytes = audio_data
                elif hasattr(audio_data, "getbuffer"):
                    data_bytes = bytes(audio_data.getbuffer())
                else:
                    raise ValueError(f"Unsupported audio data type: {type(audio_data)}")

                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    tmp.write(data_bytes)
                    temp_path = tmp.name
                audio_input = temp_path

            # Run transcription with VAD (voice activity detection) filtering
            segments, info = model.transcribe(
                audio_input,
                language=whisper_lang,
                beam_size=beam_size,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500),
            )

            # Assemble transcribed text from segments
            text_segments = [seg.text.strip() for seg in segments if seg.text.strip()]
            full_text = " ".join(text_segments).strip()

            meta = {
                "detected_language": info.language,
                "language_probability": round(info.language_probability, 3),
                "duration": round(info.duration, 2),
            }

            return full_text, meta

        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass
