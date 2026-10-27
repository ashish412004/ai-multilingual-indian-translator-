"""
Script to execute the 7 required translation evaluations using actual AI models.
Outputs real computed translations, BLEU scores, and chrF++ metrics.
"""

import os
import sys
import time

# Ensure UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from model_engine.translator import TranslationEngine
from evaluation.test_benchmark import BENCHMARK_PAIRS
from evaluation.evaluator import compute_metrics

def main():
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("ERROR: HF_TOKEN environment variable is not set.")
        sys.exit(1)

    print("=" * 80)
    print("AI-BASED MULTILINGUAL INDIAN LANGUAGE TRANSLATOR USING NLP")
    print("GENUINE NEURAL INFERENCE BENCHMARK RUNNER")
    print("=" * 80)
    
    engine = TranslationEngine(hf_token=token)
    print(f"Device: {engine.device} | PyTorch Precision: {engine.torch_dtype}\n")

    results = []

    for idx, item in enumerate(BENCHMARK_PAIRS, 1):
        print("-" * 80)
        print(f"TEST {idx}/7: {item['pair_name']} ({item['src_code']} -> {item['tgt_code']})")
        print(f"Active Model: {item['model_id']}")
        print(f"Input Text:   {item['input_text']}")
        print(f"Reference:    {item['reference']}")
        
        start_t = time.time()
        try:
            prediction = engine.translate(
                item['input_text'],
                src_lang=item['src_code'],
                tgt_lang=item['tgt_code']
            )
            elapsed = time.time() - start_t
            metrics = compute_metrics(prediction, item['reference'])
            
            print(f"Prediction:   {prediction}")
            print(f"Latency:      {elapsed:.2f}s")
            print(f"Real BLEU:    {metrics['bleu']} / 100")
            print(f"Real chrF++:  {metrics['chrf_plus_plus']} / 100")
            
            results.append({
                "pair": item['pair_name'],
                "prediction": prediction,
                "metrics": metrics,
                "latency": elapsed
            })
        except Exception as e:
            print(f"FAILED: {e}")
            results.append({
                "pair": item['pair_name'],
                "prediction": f"ERROR: {e}",
                "metrics": {"bleu": 0.0, "chrf_plus_plus": 0.0},
                "latency": 0.0
            })

    print("\n" + "=" * 80)
    print("BENCHMARK EXECUTION SUMMARY")
    print("=" * 80)
    for r in results:
        print(f"• {r['pair']:<22} | BLEU: {r['metrics']['bleu']:>5.2f} | chrF++: {r['metrics']['chrf_plus_plus']:>5.2f} | Time: {r['latency']:>5.2f}s")
    print("=" * 80)

if __name__ == "__main__":
    main()
