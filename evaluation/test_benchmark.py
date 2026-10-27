"""
College demonstration benchmark script.
Evaluates real neural machine translation pairs using SacreBLEU metrics.
"""

from typing import List, Dict
import pandas as pd
from .evaluator import compute_metrics, evaluate_batch

# Standard reference test suite for Indian language translation demonstration
BENCHMARK_PAIRS = [
    {
        "pair_id": "EN-HI",
        "pair_name": "English -> Hindi",
        "src_code": "eng_Latn",
        "tgt_code": "hin_Deva",
        "model_id": "ai4bharat/indictrans2-en-indic-dist-200M",
        "input_text": "Antigravity is an advanced AI assistant designed to accelerate modern software development.",
        "reference": "एंटीग्रेविटी एक उन्नत एआई सहायक है जिसे आधुनिक सॉफ्टवेयर विकास में तेजी लाने के लिए डिज़ाइन किया गया है।",
    },
    {
        "pair_id": "HI-EN",
        "pair_name": "Hindi -> English",
        "src_code": "hin_Deva",
        "tgt_code": "eng_Latn",
        "model_id": "ai4bharat/indictrans2-indic-en-dist-200M",
        "input_text": "कृत्रिम बुद्धिमत्ता और प्राकृतिक भाषा प्रसंस्करण भारतीय भाषाओं को सशक्त बना रहे हैं।",
        "reference": "Artificial intelligence and natural language processing are empowering Indian languages.",
    },
    {
        "pair_id": "EN-GU",
        "pair_name": "English -> Gujarati",
        "src_code": "eng_Latn",
        "tgt_code": "guj_Gujr",
        "model_id": "ai4bharat/indictrans2-en-indic-dist-200M",
        "input_text": "Education and technology play a vital role in nation building.",
        "reference": "શિક્ષણ અને ટેકનોલોજી રાષ્ટ્ર નિર્માણમાં મહત્વપૂર્ણ ભૂમિકા ભજવે છે.",
    },
    {
        "pair_id": "EN-MR",
        "pair_name": "English -> Marathi",
        "src_code": "eng_Latn",
        "tgt_code": "mar_Deva",
        "model_id": "ai4bharat/indictrans2-en-indic-dist-200M",
        "input_text": "Machine translation helps bridge communication barriers across diverse linguistic communities.",
        "reference": "मशीन भाषांतर विविध भाषिक समुदायांमध्ये संपर्कातील अडथळे दूर करण्यास मदत करते.",
    },
    {
        "pair_id": "HI-GU",
        "pair_name": "Hindi -> Gujarati",
        "src_code": "hin_Deva",
        "tgt_code": "guj_Gujr",
        "model_id": "ai4bharat/indictrans2-indic-indic-dist-320M",
        "input_text": "यह अनुसंधान परियोजना भारतीय भाषाओं के लिए तंत्रिका अनुवाद की शक्ति प्रदर्शित करती है।",
        "reference": "આ સંશોધન પ્રોજેક્ટ ભારતીય ભાષાઓ માટે ન્યુરલ ટ્રાન્સલેશનની શક્તિ દર્શાવે છે.",
    },
    {
        "pair_id": "MR-EN",
        "pair_name": "Marathi -> English",
        "src_code": "mar_Deva",
        "tgt_code": "eng_Latn",
        "model_id": "ai4bharat/indictrans2-indic-en-dist-200M",
        "input_text": "विद्यार्थ्यांनी नवनवीन तंत्रज्ञानाचा अभ्यास करून स्वतःचा विकास केला पाहिजे.",
        "reference": "Students should study innovative technology and develop themselves.",
    },
    {
        "pair_id": "TA-HI",
        "pair_name": "Tamil -> Hindi",
        "src_code": "tam_Taml",
        "tgt_code": "hin_Deva",
        "model_id": "ai4bharat/indictrans2-indic-indic-dist-320M",
        "input_text": "தமிழ்நாட்டின் கலாச்சாரம் மற்றும் பாரம்பரியம் மிகவும் பழமையானது.",
        "reference": "तमिलनाडु की संस्कृति और परंपरा बहुत प्राचीन है।",
    },
]


def get_saved_benchmark() -> pd.DataFrame:
    """
    Load verified benchmark results from local file if available.
    """
    import json
    import os
    json_path = os.path.join(os.path.dirname(__file__), "benchmark_results.json")
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return pd.DataFrame([
            {
                "Pair": item["pair_name"],
                "Model Used": item["model_id"].split("/")[-1],
                "Source Code": item["src_code"],
                "Target Code": item["tgt_code"],
                "Input Text": item["input_text"],
                "Human Reference": item["human_reference"],
                "Model Prediction": item["model_prediction"],
                "BLEU": item["bleu_score"],
                "chrF++": item["chrf_plus_plus"],
                "Latency (s)": item["latency_seconds"],
                "Status": "Verified Execution"
            }
            for item in data
        ])
    return pd.DataFrame()


def run_benchmark(engine=None) -> pd.DataFrame:
    """
    Run evaluation against the benchmark test suite.
    If engine is provided and accessible, it generates real predictions and computes genuine SacreBLEU scores.
    """
    records = []
    
    for item in BENCHMARK_PAIRS:
        pred_text = ""
        bleu = 0.0
        chrf = 0.0
        status = "Pending Model"

        if engine is not None:
            try:
                pred_text = engine.translate(
                    item["input_text"],
                    src_lang=item["src_code"],
                    tgt_lang=item["tgt_code"]
                )
                metrics = compute_metrics(pred_text, item["reference"])
                bleu = metrics["bleu"]
                chrf = metrics["chrf_plus_plus"]
                status = "Evaluated (Real Model)"
            except Exception as e:
                pred_text = f"Error: {e}"
                status = "Failed"

        records.append({
            "Pair": item["pair_name"],
            "Model Used": item["model_id"].split("/")[-1],
            "Source Code": item["src_code"],
            "Target Code": item["tgt_code"],
            "Input Text": item["input_text"],
            "Human Reference": item["reference"],
            "Model Prediction": pred_text,
            "BLEU": bleu,
            "chrF++": chrf,
            "Status": status
        })

    return pd.DataFrame(records)
