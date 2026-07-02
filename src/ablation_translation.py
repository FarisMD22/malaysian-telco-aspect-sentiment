"""
Translation ablation.

Input:  the eval tiers + models/xlmr_final
Output: models/ablation_translation.csv, models/translation_cache.json + console table
Run:    python src/ablation_translation.py            (forum + trustpilot tiers)
        python src/ablation_translation.py --tiers forum

XLM-RoBERTa is natively multilingual, so the production path feeds cleaned Bahasa Melayu / Manglish
straight to the model. This ablation tests whether an English pivot helps: each eval item is
machine-translated to English (deep-translator / Google backend) and the same fine-tuned model is
re-scored on the translated text, then macro-F1 is compared to the original. It is an eval-time
ablation (no retrain). Translations are cached to disk so reruns are free and offline; network
failures fall back to the original text per item and are counted, so a flaky connection degrades
gracefully rather than crashing.
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import LABEL_MAP, xlmr_preds_fn  # reuse the exact XLM-R predict wrapper

# The non-English (BM/Manglish, mostly tagged 'id') content is concentrated in the in-domain
# test set (64/240); Trustpilot is 100% English and the forum sample 57/60 English, so the
# ablation is only informative in-domain. We still allow the cross-tiers for completeness.
TIERS = {
    "test": ("Play Store (in-domain)", "data/labeled/labeled_main_test.csv"),
    "forum": ("Reddit/Lowyat (cross-domain)", "data/labeled/forum_eval.csv"),
    "trustpilot": ("Trustpilot (cross-platform)", "data/labeled/trustpilot_eval.csv"),
}
CACHE_PATH = Path("models/translation_cache.json")
XLMR_PATH = "models/xlmr_final"


def load_cache() -> dict:
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    return {}


def translate_all(texts, cache) -> tuple[list, int]:
    """Translate to English with per-item fallback to the original. Returns (texts, n_failed)."""
    from deep_translator import GoogleTranslator
    translator = GoogleTranslator(source="auto", target="en")
    out, failed = [], 0
    for t in texts:
        t = (t or "").strip()
        if not t:
            out.append("")
            continue
        if t in cache:
            out.append(cache[t])
            continue
        try:
            en = translator.translate(t[:4900]) or t  # Google caps ~5k chars/request
        except Exception:
            en, failed = t, failed + 1            # fall back to original, keep going
        cache[t] = en
        out.append(en)
    CACHE_PATH.parent.mkdir(exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    return out, failed


def macro(df, preds_fn, texts):
    y_true = df["sentiment_label"].map(LABEL_MAP).astype(int).tolist()
    y_pred = preds_fn([t or "" for t in texts])
    return accuracy_score(y_true, y_pred), f1_score(y_true, y_pred, average="macro")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiers", nargs="+", default=["test"], choices=list(TIERS),
                    help="default 'test', the only tier with meaningful non-English content")
    a = ap.parse_args()

    if not Path(XLMR_PATH, "model.safetensors").exists():
        sys.exit(f"{XLMR_PATH}/model.safetensors not found; train XLM-R first "
                 "(notebooks/finetune_xlmr_colab.ipynb).")

    preds_fn = xlmr_preds_fn(XLMR_PATH)
    cache = load_cache()
    rows = []
    for key in a.tiers:
        label, path = TIERS[key]
        df = pd.read_csv(path)
        orig = df["cleaned_text"].fillna("").tolist()
        trans, failed = translate_all(orig, cache)
        if failed:
            print(f"  [{key}] {failed}/{len(df)} translations failed -> used original text for those")

        # Full-tier delta.
        acc_o, f1_o = macro(df, preds_fn, orig)
        acc_t, f1_t = macro(df, preds_fn, trans)

        # Subset: rows where translation actually changed the text (i.e. genuinely
        # non-English). Identity rows (already-English) only dilute the signal.
        changed = [o.strip() != t.strip() for o, t in zip(orig, trans)]
        sub = df[pd.Series(changed, index=df.index)]
        n_sub = len(sub)
        if n_sub:
            sub_orig = [o for o, c in zip(orig, changed) if c]
            sub_trans = [t for t, c in zip(trans, changed) if c]
            _, f1_sub_o = macro(sub, preds_fn, sub_orig)
            _, f1_sub_t = macro(sub, preds_fn, sub_trans)
        else:
            f1_sub_o = f1_sub_t = float("nan")

        print(f"[{label}]  full tier (n={len(df)}): orig F1={f1_o:.3f} -> translated F1={f1_t:.3f} "
              f"(delta {f1_t - f1_o:+.3f})")
        print(f"    non-English subset actually translated (n={n_sub}): "
              f"orig F1={f1_sub_o:.3f} -> translated F1={f1_sub_t:.3f} (delta {f1_sub_t - f1_sub_o:+.3f})")
        rows.append({"tier": label, "n": len(df), "n_translation_failed": failed,
                     "macro_f1_original": f1_o, "macro_f1_translated": f1_t,
                     "macro_f1_delta": f1_t - f1_o,
                     "n_nonenglish_translated": n_sub,
                     "macro_f1_original_sub": f1_sub_o, "macro_f1_translated_sub": f1_sub_t,
                     "macro_f1_delta_sub": f1_sub_t - f1_sub_o})

    res = pd.DataFrame(rows)
    Path("models").mkdir(exist_ok=True)
    res.to_csv("models/ablation_translation.csv", index=False)
    print("\nSaved models/ablation_translation.csv")
    print("Interpretation: delta <= 0 supports the multilingual-model design "
          "(no English pivot needed); a large positive delta would argue for a translation step.")


if __name__ == "__main__":
    main()
