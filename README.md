# Aspect-Aware Sentiment Analysis of Malaysian Telco & Broadband Feedback

An end-to-end natural-language-processing system that analyses customer feedback about
Malaysian telecommunications and broadband services. Given a review or forum post in
English, Bahasa Melayu, or Manglish (code-switched Malay–English), it predicts the
overall sentiment (negative / neutral / positive) and identifies which of five service
aspects the text discusses — **coverage, speed, billing, customer support, and app
usability** — reporting a sentiment for each.

Two models are provided and compared: a classical **TF-IDF + Logistic Regression / Linear
SVM** baseline, and a fine-tuned **XLM-RoBERTa** transformer. Each is evaluated across
three tiers of increasingly out-of-domain data — Play Store (in-domain), Trustpilot
(cross-platform), and Reddit/Lowyat forums (cross-domain) — and the accuracy *degradation*
across those tiers, measured per aspect, is the project's principal finding.

Built for **TNL6323 — Natural Language Processing**.

> **Full instructions:** see **`UserManual.pdf`** for step-by-step installation, pipeline,
> and application usage. This README is a short orientation.

## Live demo

The full application — fine-tuned XLM-RoBERTa sentiment + per-aspect analysis — is deployed and
running (no setup, no GPU, no download required):

> **🚀 https://huggingface.co/spaces/FarisTheCoder/telco-sentiment**

The transformer weights (~1.1 GB) are hosted separately as a HuggingFace model repo,
**`FarisTheCoder/xlmr-telco-sentiment`**, and loaded by the app at runtime. To run the same app
locally against that hosted model (CPU-only; downloads the model from HF on first launch):

```bash
python app/app.py            # auto-loads FarisTheCoder/xlmr-telco-sentiment if no local model
```

## Quick start

All commands are run from the repository root, with the virtual environment active.

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # macOS / Linux
pip install -r requirements.txt

python src/baseline.py            # train the TF-IDF baseline  -> models/
python src/evaluate.py            # three-tier accuracy table  -> models/eval_results.csv
python app/app.py                 # launch the web app at http://127.0.0.1:7860
```

The trained XLM-RoBERTa model (~1.1 GB) is **not** committed to this repo (it exceeds the
submission size limit). It is hosted on HuggingFace as **`FarisTheCoder/xlmr-telco-sentiment`**
and the app loads it automatically at runtime — see **Live demo** above. You can also regenerate
it yourself by running the pipeline (`src/finetune_xlmr.py`, GPU required — the Colab notebook in
`notebooks/` provides a GPU path). The baseline, evaluation, and app all run on CPU. If neither the
hosted model nor a local one is reachable, the web app still opens and the evaluation reports the
baseline results only.

## Reproduce the full pipeline

```bash
python src/preprocess.py           # clean/deduplicate the raw data  -> data/cleaned/
python src/baseline.py             # TF-IDF baseline (LogReg + SVM)  -> models/
python src/finetune_xlmr.py        # fine-tune XLM-RoBERTa (GPU)     -> models/xlmr_final/
python src/aspect_sa.py            # tag rows with the five aspects
python src/evaluate.py             # three-tier comparison table
python src/aspect_degradation.py   # per-aspect degradation + heatmaps
```

## Repository layout

| Path | Contents |
|------|----------|
| `src/` | Pipeline: preprocessing, baseline, transformer fine-tune, aspect extractor, evaluation, degradation experiment, ablations |
| `scrapers/` | Data collection (Google Play, Reddit, Trustpilot, Lowyat) |
| `app/` | Local launcher for the Gradio web application |
| `deploy/` | Hugging Face Spaces deployment (`hf_space/`) and `DEPLOY.md` |
| `notebooks/` | Colab notebook for GPU fine-tuning |
| `labeling/` | Labelling rubric and inter-annotator-agreement tooling |
| `data/` | `labeled/` (used for training/evaluation), plus `raw/` and `cleaned/` |
| `models/` | Output directory for trained models and result files (generated) |
| `UserManual.pdf` | Full user manual |
| `METHODOLOGY.md`, `EXPERIMENT_aspect_degradation.md`, `DATASET_SUMMARY.md` | Technical documentation |

## Requirements

Python 3.10 or later. All dependencies are listed in `requirements.txt`. A GPU is required
only for fine-tuning the transformer; everything else runs on CPU. See `UserManual.pdf`,
Section 2, for full system requirements.

## Mapping to the submission rubric

For assessment, this repository corresponds to the required submission folders as follows:
**Source Codes** = `src/`, `scrapers/`, `app/`, `deploy/`, `labeling/`, and `notebooks/`;
**Data** = `data/`. The user manual is `UserManual.pdf`.
