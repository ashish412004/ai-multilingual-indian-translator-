"""
Core Translation Engine for Multilingual Indian Language Translation using IndicTrans2.
"""

import os
import sys
import torch
from typing import Optional, Dict, Any, Tuple
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from .config import (
    MODEL_EN_INDIC,
    MODEL_INDIC_EN,
    MODEL_INDIC_INDIC,
    get_model_for_pair,
)

# Import IndicProcessor from IndicTransToolkit
try:
    from IndicTransToolkit.processor import IndicProcessor
except ImportError:
    # Fallback to local import if needed
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from IndicTransToolkit_src.IndicTransToolkit.processor import IndicProcessor


class TranslationEngine:
    """
    Manages model loading, caching, device allocation, and neural sequence-to-sequence
    translation using AI4Bharat IndicTrans2 models.
    """

    def __init__(self, hf_token: Optional[str] = None):
        """
        Initialize the translation engine.
        
        Args:
            hf_token: Optional Hugging Face user access token for gated models.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        # Processor handles language-specific scripts, transliteration, placeholders
        self.processor = IndicProcessor(inference=True)
        
        # Cache for loaded models: {model_name: {"model": model, "tokenizer": tokenizer}}
        self._loaded_models: Dict[str, Dict[str, Any]] = {}
        
        # Hugging Face auth token (supports env vars and Streamlit secrets)
        token = (
            hf_token
            or os.environ.get("HF_TOKEN")
            or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        )
        if not token:
            try:
                import streamlit as st
                if "HF_TOKEN" in st.secrets:
                    token = st.secrets["HF_TOKEN"]
            except Exception:
                pass
        self.hf_token = token

    def set_hf_token(self, token: str):
        """Update Hugging Face token dynamically."""
        self.hf_token = token.strip() if token else None

    def is_model_loaded(self, model_name: str) -> bool:
        """Check if a specific model is already in memory."""
        return model_name in self._loaded_models

    def load_model(self, model_name: str) -> Tuple[AutoTokenizer, AutoModelForSeq2SeqLM]:
        """
        Load tokenizer and model weights into memory with caching.
        
        Args:
            model_name: Hugging Face repo ID (e.g. ai4bharat/indictrans2-en-indic-dist-200M)
            
        Returns:
            Tuple of (tokenizer, model)
            
        Raises:
            PermissionError: If model is gated and token is missing or unauthorized
            MemoryError: If system RAM/VRAM is insufficient
            RuntimeError: For other loading errors
        """
        if model_name in self._loaded_models:
            return (
                self._loaded_models[model_name]["tokenizer"],
                self._loaded_models[model_name]["model"],
            )

        token = self.hf_token
        
        try:
            # 1. Load Tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                token=token,
            )

            # 2. Load Model Seq2Seq weights
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=self.torch_dtype,
                low_cpu_mem_usage=True,
                token=token,
            ).to(self.device)

            model.eval()

            # Cache the initialized objects
            self._loaded_models[model_name] = {
                "tokenizer": tokenizer,
                "model": model,
            }

            return tokenizer, model

        except Exception as e:
            err_str = str(e).lower()
            if "gated repo" in err_str or "unauthorized" in err_str or "401" in err_str or "403" in err_str:
                raise PermissionError(
                    f"Authentication required for gated model '{model_name}'.\n"
                    f"Please agree to the model terms on Hugging Face (https://huggingface.co/{model_name}) "
                    f"and provide a valid Hugging Face Access Token."
                ) from e
            elif "out of memory" in err_str or "cuda error: out of memory" in err_str:
                raise MemoryError(
                    f"System or GPU memory was exhausted while loading '{model_name}'. "
                    f"Free up system RAM or run in CPU mode."
                ) from e
            else:
                raise RuntimeError(f"Failed to load model '{model_name}': {e}") from e

    def translate(
        self,
        text: str,
        src_lang: str,
        tgt_lang: str,
        num_beams: int = 5,
        max_length: int = 256,
    ) -> str:
        """
        Translate input text from src_lang to tgt_lang using genuine IndicTrans2 neural inference.
        
        Args:
            text: Input text string
            src_lang: Source FLORES language code (e.g., 'eng_Latn', 'hin_Deva')
            tgt_lang: Target FLORES language code (e.g., 'hin_Deva', 'guj_Gujr')
            num_beams: Beam search size for generation (default 5)
            max_length: Maximum tokens to generate
            
        Returns:
            Translated text string
        """
        stripped_text = text.strip()
        if not stripped_text:
            return ""

        # Identity check: If source and target are identical, return unchanged input
        if src_lang == tgt_lang:
            return stripped_text

        # 1. Determine model routing
        model_name = get_model_for_pair(src_lang, tgt_lang)
        if model_name == "IDENTITY":
            return stripped_text

        # 2. Ensure model is loaded
        tokenizer, model = self.load_model(model_name)

        # 3. Preprocess input text through IndicProcessor
        # Adds language tags, normalizes Unicode scripts, numerals, placeholders
        preprocessed_batch = self.processor.preprocess_batch(
            [stripped_text],
            src_lang=src_lang,
            tgt_lang=tgt_lang,
        )

        # 4. Tokenize preprocessed text
        inputs = tokenizer(
            preprocessed_batch,
            truncation=True,
            padding="longest",
            return_tensors="pt",
            return_attention_mask=True,
        ).to(self.device)

        # 5. Transformer Sequence-to-Sequence Generation
        with torch.no_grad():
            generated_tokens = model.generate(
                **inputs,
                use_cache=True,
                min_length=0,
                max_length=max_length,
                num_beams=num_beams,
                num_return_sequences=1,
            )

        # 6. Decode token IDs back into string
        decoded_tokens = tokenizer.batch_decode(
            generated_tokens.detach().cpu().tolist(),
            skip_special_tokens=True,
        )

        # 7. Postprocess output through IndicProcessor (detokenize, restore entities, fix scripts)
        postprocessed = self.processor.postprocess_batch(
            decoded_tokens,
            lang=tgt_lang,
        )

        return postprocessed[0]
