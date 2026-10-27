"""
Unit tests for model routing logic.
"""

import unittest
from model_engine.config import (
    MODEL_EN_INDIC,
    MODEL_INDIC_EN,
    MODEL_INDIC_INDIC,
    get_model_for_pair,
    SUPPORTED_LANGUAGES,
)


class TestModelRouting(unittest.TestCase):
    def test_english_to_indic_routing(self):
        indic_langs = ["hin_Deva", "guj_Gujr", "mar_Deva", "tam_Taml", "tel_Telu", "ben_Beng"]
        for lang in indic_langs:
            model = get_model_for_pair("eng_Latn", lang)
            self.assertEqual(model, MODEL_EN_INDIC, f"Expected {MODEL_EN_INDIC} for eng_Latn -> {lang}")

    def test_indic_to_english_routing(self):
        indic_langs = ["hin_Deva", "guj_Gujr", "mar_Deva", "tam_Taml", "tel_Telu", "ben_Beng"]
        for lang in indic_langs:
            model = get_model_for_pair(lang, "eng_Latn")
            self.assertEqual(model, MODEL_INDIC_EN, f"Expected {MODEL_INDIC_EN} for {lang} -> eng_Latn")

    def test_indic_to_indic_routing(self):
        pairs = [
            ("hin_Deva", "guj_Gujr"),
            ("tam_Taml", "hin_Deva"),
            ("mar_Deva", "ben_Beng"),
            ("pan_Guru", "kan_Knda"),
        ]
        for src, tgt in pairs:
            model = get_model_for_pair(src, tgt)
            self.assertEqual(model, MODEL_INDIC_INDIC, f"Expected {MODEL_INDIC_INDIC} for {src} -> {tgt}")

    def test_identity_routing(self):
        for name, code in SUPPORTED_LANGUAGES.items():
            model = get_model_for_pair(code, code)
            self.assertEqual(model, "IDENTITY", f"Expected IDENTITY for {code} -> {code}")


if __name__ == "__main__":
    unittest.main()
