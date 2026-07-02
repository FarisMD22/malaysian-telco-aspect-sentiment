"""
Gradio app — Malaysian Telco & Broadband Sentiment Analyzer (HuggingFace Space).

TNL6323 group project. Self-contained: this folder (app.py + aspect_sa.py + requirements.txt +
README.md) is exactly what gets pushed to the Space.

Model loading
-------------
The fine-tuned XLM-R model (~1.1 GB) is NOT committed into the Space. Instead it lives in a HF
model repo and is loaded by id at runtime via the HF_MODEL_ID environment variable / Space secret:
    HF_MODEL_ID = <your-username>/xlmr-telco-sentiment
Locally (with the model on disk) it falls back to ./models/xlmr_final, so `python deploy/hf_space/app.py`
works from the repo root for development. See deploy/DEPLOY.md.
"""
import os

import gradio as gr

from aspect_sa import ASPECTS, extract_aspects, sentences_for_aspect

MODEL_ID = os.environ.get("HF_MODEL_ID", "models/xlmr_final")
MAX_LEN = 128

# --- language detection (best-effort; langdetect conflates Bahasa Melayu with 'id') ---
try:
    from langdetect import DetectorFactory, LangDetectException, detect
    DetectorFactory.seed = 42
    _LANGDETECT = True
except Exception:
    _LANGDETECT = False

_LANG_NAMES = {"en": "English", "ms": "Bahasa Melayu", "id": "Indonesian", "zh-cn": "Chinese",
               "zh-tw": "Chinese", "ta": "Tamil"}

# --- model ---
LOAD_ERROR = ""
try:
    from transformers import pipeline
    clf = pipeline("text-classification", model=MODEL_ID, tokenizer=MODEL_ID,
                   truncation=True, max_length=MAX_LEN, top_k=None)
    MODEL_READY = True
except Exception as e:  # model missing / not yet uploaded
    clf, MODEL_READY, LOAD_ERROR = None, False, str(e)


def detect_language(text: str) -> str:
    if not _LANGDETECT:
        return "—"
    try:
        code = detect(text)
    except LangDetectException:
        return "unknown"
    name = _LANG_NAMES.get(code, code)
    if code == "id":
        name += " — note: likely Bahasa Melayu (langdetect conflates BM with Indonesian)"
    return name


def _scores(text: str) -> dict:
    """{label: prob} over the 3 classes for gr.Label bars."""
    out = clf(text)[0]  # top_k=None -> list of {label, score}
    return {d["label"]: float(d["score"]) for d in out}


def analyze(text: str):
    text = (text or "").strip()
    if not text:
        return {}, "", []
    if not MODEL_READY:
        msg = (f"⚠️ Model not loaded ({LOAD_ERROR}). Set the HF_MODEL_ID secret to your uploaded "
               "model repo, or place the model at ./models/xlmr_final for local runs.")
        return {}, msg, []

    overall = _scores(text)
    lang = f"**Detected language:** {detect_language(text)}"

    # Light per-aspect sentiment: classify only the sentences mentioning each detected aspect.
    rows = []
    for aspect in extract_aspects(text):
        snippet = sentences_for_aspect(text, aspect)
        s = _scores(snippet)
        top = max(s, key=s.get)
        rows.append([aspect.replace("_", " "), top, f"{s[top]:.0%}"])
    if not rows:
        rows = [["(no aspect keywords detected)", "—", "—"]]
    return overall, lang, rows


EXAMPLES = [
    "The 5G coverage in KL is decent now, but the signal is patchy indoors in Subang.",
    "Maxis customer service took 3 hours to respond and the agent was rude. Useless support.",
    "Unifi fibre speed has been consistent for 6 months, but the bill is too expensive.",
    "Line celcom memang lembab, internet slow gila and the app selalu crash bila nak bayar.",
]

with gr.Blocks(title="Malaysian Telco Sentiment Analyzer") as demo:
    gr.Markdown(
        "# 📶 Malaysian Telco & Broadband Sentiment Analyzer\n"
        "TNL6323 group project — 3-class sentiment (XLM-RoBERTa) + rule-based aspect analysis "
        "over five service aspects: **coverage, speed, billing, customer support, app usability**. "
        "Handles English, Bahasa Melayu and Manglish (code-switched) text."
    )
    if not MODEL_READY:
        gr.Markdown(f"> ⚠️ **Model not loaded.** {LOAD_ERROR}")
    with gr.Row():
        with gr.Column():
            inp = gr.Textbox(lines=5, label="Review or forum post",
                             placeholder="Paste a Play Store / Trustpilot review or a forum post…")
            btn = gr.Button("Analyze", variant="primary")
            gr.Examples(EXAMPLES, inputs=inp)
        with gr.Column():
            out_sent = gr.Label(label="Overall sentiment", num_top_classes=3)
            out_lang = gr.Markdown()
            out_aspects = gr.Dataframe(headers=["aspect", "sentiment", "confidence"],
                                       label="Per-aspect sentiment (sentences mentioning each aspect)",
                                       interactive=False, wrap=True)
    btn.click(analyze, inputs=inp, outputs=[out_sent, out_lang, out_aspects])
    inp.submit(analyze, inputs=inp, outputs=[out_sent, out_lang, out_aspects])

if __name__ == "__main__":
    demo.launch()
