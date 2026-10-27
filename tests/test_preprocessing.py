"""
Unit tests for IndicProcessor preprocessing and postprocessing pipeline.
"""

import unittest
from IndicTransToolkit.processor import IndicProcessor


class TestPreprocessing(unittest.TestCase):
    def setUp(self):
        self.processor = IndicProcessor(inference=True)

    def test_en_to_indic_prefix_format(self):
        sample = "Good morning to everyone."
        preprocessed = self.processor.preprocess_batch([sample], src_lang="eng_Latn", tgt_lang="hin_Deva")
        self.assertEqual(len(preprocessed), 1)
        self.assertTrue(preprocessed[0].startswith("eng_Latn hin_Deva "))
        self.assertIn("Good morning to everyone", preprocessed[0])

    def test_placeholder_wrapping(self):
        sample = "Please send the file to test@example.com before 5:00 PM."
        preprocessed = self.processor.preprocess_batch([sample], src_lang="eng_Latn", tgt_lang="hin_Deva")
        # Ensure email pattern is replaced with ID placeholder in preprocessed stream
        self.assertTrue("ID1" in preprocessed[0])

    def test_numeral_and_postprocessing(self):
        sample = "The meeting is at 10:30 AM."
        preprocessed = self.processor.preprocess_batch([sample], src_lang="eng_Latn", tgt_lang="hin_Deva")
        
        # Simulate model passing the placeholder through
        simulated_model_output = preprocessed[0].replace("eng_Latn hin_Deva ", "")
        postprocessed = self.processor.postprocess_batch([simulated_model_output], lang="eng_Latn")
        
        self.assertEqual(len(postprocessed), 1)
        self.assertIn("10:30", postprocessed[0])


if __name__ == "__main__":
    unittest.main()
