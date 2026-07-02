---
title: Malaysian Telco Sentiment Analyzer
emoji: 📶
colorFrom: blue
colorTo: indigo
sdk: gradio
python_version: "3.12"
app_file: app.py
pinned: false
license: mit
---

# Malaysian Telco & Broadband Sentiment Analyzer

TNL6323 group project. Aspect-aware 3-class sentiment analysis for Malaysian telco / broadband
user feedback (Play Store, Trustpilot, forums), in English, Bahasa Melayu and Manglish.

- **Sentiment:** XLM-RoBERTa fine-tuned for negative / neutral / positive.
- **Aspects:** rule-based extractor over five locked service aspects — coverage, speed, billing,
  customer support, app usability — with a light per-aspect sentiment read.
- **Language:** best-effort detection (note: langdetect conflates Bahasa Melayu with Indonesian).

## Configuration

The model is loaded at runtime from a HF model repo, set via the **`HF_MODEL_ID`** Space secret
(e.g. `your-username/xlmr-telco-sentiment`). The ~1.1 GB weights are not committed into this Space.
See `deploy/DEPLOY.md` in the project repo for the full upload + deploy steps.
