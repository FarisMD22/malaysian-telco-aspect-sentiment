# Submission Checklist — TNL6323 Group Project

**Project:** Aspect-Aware Sentiment Analysis of Malaysian Telco & Broadband Feedback
**Repository:** https://github.com/FarisMD22/malaysian-telco-aspect-sentiment (public)
**Live demo:** https://huggingface.co/spaces/FarisTheCoder/telco-sentiment

## Deliverables

- [x] **Source code** — `src/` (pipeline), `scrapers/` (data collection), `app/` + `deploy/` (web app), `labeling/` (labelling tooling), `notebooks/` (GPU fine-tuning)
- [x] **Data** — `data/raw/`, `data/cleaned/`, `data/labeled/` (labelled training + three eval tiers); see `data/DATASET_SUMMARY.md`
- [x] **User manual** — `UserManual.pdf` (with editable `UserManual.docx` source)
- [x] **Documentation** — `README.md`, `data/DATASET_SUMMARY.md`, `deploy/DEPLOY.md`, `labeling/rubric.md`, `data/labeled/kappa.md`

## Models

- [x] **Baseline** (TF-IDF + LogReg / Linear SVM) — trained files committed under `models/`
- [x] **Transformer** (fine-tuned XLM-RoBERTa, ~1.1 GB) — hosted on Hugging Face (`FarisTheCoder/xlmr-telco-sentiment`); loaded automatically by the app (too large to commit)

## How to run (CPU, from repo root)

```bash
python -m venv .venv && .venv\Scripts\activate     # Windows
pip install -r requirements.txt
python src/baseline.py      # train the baseline        -> models/
python src/evaluate.py      # three-tier accuracy table -> models/eval_results.csv
python app/app.py           # launch the web app (auto-loads the hosted model)
```

Full pipeline and options: `UserManual.pdf`.

## Verification

- [x] Full CPU pipeline runs end-to-end (preprocess → baseline → aspect tagging → evaluate → degradation → ablations)
- [x] Web application launches and returns predictions; hosted Space is live
- [x] Three-tier result reproduced: macro-F1 **0.594 (in-domain) → 0.428 (cross-platform) → 0.333 (cross-domain)**
- [x] A GPU is required **only** for fine-tuning the transformer; everything else runs on CPU
