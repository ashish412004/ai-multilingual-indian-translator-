"""
Unit tests for translation engine and evaluation metrics.
"""

import unittest
from model_engine.translator import TranslationEngine
from evaluation.evaluator import compute_metrics, evaluate_batch


class TestEngineAndEvaluation(unittest.TestCase):
    def setUp(self):
        self.engine = TranslationEngine()

    def test_empty_string_handling(self):
        res = self.engine.translate("", "eng_Latn", "hin_Deva")
        self.assertEqual(res, "")
        
        res_spaces = self.engine.translate("   ", "eng_Latn", "hin_Deva")
        self.assertEqual(res_spaces, "")

    def test_identity_translation(self):
        # When source and target are the same, should return text untouched without loading model
        text = "Machine learning is fascinating."
        res = self.engine.translate(text, "eng_Latn", "eng_Latn")
        self.assertEqual(res, text)

        hindi_text = "नमस्ते दुनिया"
        res_hi = self.engine.translate(hindi_text, "hin_Deva", "hin_Deva")
        self.assertEqual(res_hi, hindi_text)

    def test_sacrebleu_perfect_match(self):
        ref = "यह एक परीक्षण वाक्य है।"
        hyp = "यह एक परीक्षण वाक्य है।"
        metrics = compute_metrics(hyp, ref)
        self.assertAlmostEqual(metrics["bleu"], 100.0, places=1)
        self.assertAlmostEqual(metrics["chrf_plus_plus"], 100.0, places=1)

    def test_sacrebleu_mismatch(self):
        ref = "यह एक परीक्षण वाक्य है।"
        hyp = "कंप्यूटर विज्ञान एक रोचक विषय है।"
        metrics = compute_metrics(hyp, ref)
        self.assertLess(metrics["bleu"], 50.0)

    def test_batch_evaluation(self):
        refs = ["Good morning", "How are you?"]
        hyps = ["Good morning", "How are you?"]
        batch_res = evaluate_batch(hyps, refs)
        self.assertEqual(batch_res["pairs_evaluated"], 2)
        self.assertAlmostEqual(batch_res["corpus_bleu"], 100.0, places=1)


if __name__ == "__main__":
    unittest.main()
