# Methodology — Malaysian Telco & Broadband Sentiment Analysis

This document is the technical specification of the project's approach. It complements [README.md](README.md).

## 1. Problem statement

Classify short user-generated text about Malaysian telco and broadband services into one of three sentiment classes — **positive, neutral, negative** — and identify which of five service aspects the text discusses. Inputs include English, Bahasa Melayu (BM), and Manglish code-switched text; across three distinct text genres (app store reviews, third-party review site, informal forum threads).

## 2. Data strategy

### 2.1 Sources

| Source | Genre | Role |
|---|---|---|
| Google Play Store | Star-rated app reviews | Primary training + in-domain test |
| Trustpilot (MY telco pages) | Star-rated reviews on third-party site | Cross-platform eval |
| Reddit — r/malaysia, r/bolehland | Informal forum threads, no ratings | Cross-domain eval |
| Lowyat.NET — Internet Access, Mobile Telco subforums | Informal forum threads, no ratings | Cross-domain eval (optional) |

Play Store is the only training source. Trustpilot, Reddit, and Lowyat are evaluation-only and kept strictly held out from training. This design gives the three-tier evaluation story (in-domain → cross-platform → cross-domain).

### 2.2 Target apps for Play Store scrape

Seven apps, collectively representing the Malaysian telco/broadband market:

1. MyCelcomDigi
2. MyMaxis
3. Hotlink
4. MyUMobile
5. Yes 5G
6. MyUnifi
7. TIME Self Care

Scraper uses `google-play-scraper` Python library, locale `en_MY`, country `MY`. Target: ≥20,000 raw reviews (minimum 5,000).

### 2.3 Unified schema

Every scraper emits CSV rows with the same columns:

```
id, text, rating, date, source, app_or_subforum
```

`rating` is `null` for Reddit/Lowyat (no stars). `source` is one of `{playstore, trustpilot, reddit, lowyat}`. `app_or_subforum` names the specific app or thread origin.

## 3. Labeling

### 3.1 Three-label scheme

- **positive** — endorses, praises, recommends, or expresses satisfaction
- **negative** — complains, criticises, expresses frustration, or calls out a problem
- **neutral** — factual, balanced-mixed, a question, or too bland to carry sentiment

Full decision rules, worked examples (including Manglish and BM), and off-topic handling are in [`labeling/rubric.md`](labeling/rubric.md).

### 3.2 Auto-labeling — Play Store and Trustpilot only

Both platforms have star ratings. The rule is:

- 1★ or 2★ → `negative`
- 3★ → `neutral`
- 4★ or 5★ → `positive`

### 3.3 Stratified sampling — main training set

From the auto-labeled Play Store pool, stratified-sample **360 rows** with exactly **120 per class** (120 pos / 120 neu / 120 neg). This is the `labeled_main.csv` file that drives training.

We manually audit **50 random rows** against the rubric. If manual agreement with auto-label is below 90% (equivalent to κ < ~0.85), re-sample with tightened text-length filters.

### 3.4 Manual double-labeling — eval sets

Reddit, Lowyat, and Trustpilot eval items get labeled by **two** members independently. A third member arbitrates disagreements. Cohen's κ is computed per tier. Target **κ ≥ 0.6**.

Per tier: 60 items. Per person workload across both tiers: ~60 items.

### 3.5 Optional expansion

If the audit on the 360-set returns κ ≥ 0.9, the training set can expand to 500–800 rows using the same auto-labeling rule, but only as a secondary robustness experiment, not as a replacement for the 360-row core.

## 4. Preprocessing

File: `src/preprocess.py`.

Transformations applied in order:

1. Lowercase
2. Strip URLs (regex `https?://\S+`)
3. HTML strip (for any residual markup from Trustpilot/Lowyat)
4. Normalise elongations (`loool` → `lol`, `besttttt` → `best`) — character repeats > 2 collapsed to 2
5. Whitespace collapse
6. Language detection via `langdetect` → column `language` ∈ {en, ms, id, zh, ta, mixed, unk}
7. Drop reviews shorter than 5 whitespace-tokenised words
8. Deduplicate on exact `text` match within source

Outputs: `data/cleaned/<source>_clean.csv` per source, plus a merged `data/cleaned/all_clean.csv`.

**Translation is NOT applied here.** XLM-R handles multilingual and code-switched input natively. See §8 for the optional translation ablation.

### 4.1 On stemming and translation (rubric note)

The guideline lists pre-processing as "cleaning, tokenization, stemming, translation, etc. (whichever applicable)." Two of these are handled deliberately rather than by default:

- **Stemming / lemmatisation — omitted from the transformer pipeline, by design.** XLM-RoBERTa operates on SentencePiece sub-word units and already decomposes inflected and agglutinative forms internally; an upstream stemmer would discard surface signal the model is built to exploit and empirically degrades transformer accuracy. The multilingual setting reinforces this: no robust, validated stemmer exists for Bahasa Melayu or for Manglish code-switched text, and an English stemmer would corrupt the non-English portion of the corpus (see §4.2). **For rubric coverage, Porter stemming is applied within the TF-IDF baseline only** (`src/ablation_stemming.py`), and we report both stemmed and unstemmed baseline scores. The measured effect is a net wash (Δ macro-F1 −0.006 in-domain / −0.010 cross-domain / +0.016 cross-platform): on this mixed-language data Porter mis-stems the BM/Manglish tokens, cancelling the vocabulary-shrinkage benefit — which is itself evidence for keeping stemming out of both pipelines.
- **Translation — implemented as a controlled ablation, not a default stage.** See §8. The point is to *test* translation's value rather than assume it; the reported accuracy delta ticks the guideline's translation item while doubling as evidence for the multilingual-model design choice.

### 4.2 Language-detection caveat (report in Discussion)

`langdetect` cannot reliably separate Bahasa Melayu from Indonesian: on the 1,200-row training set it labels ~256 rows `id` (Indonesian), the large majority of which are actually BM. This is a genuine limitation of language-ID-based routing — and precisely why the pipeline feeds raw multilingual text to a model that does not depend on a correct language tag. It is reportable both as a data-quality limitation and as justification for the XLM-R design.

## 5. Knowledge representation

Double output:

- **CSV** — human-readable, used for labeling and for the report's data section.
- **HDF5** — compact binary, used as the training input for the XLM-R pipeline and for faster pandas reads during evaluation.

Both outputs are required by the rubric's Knowledge Representation criterion (5 marks). Files live in `data/cleaned/` and `data/labeled/`.

## 6. Models

### 6.1 Baseline — TF-IDF + linear classifiers

File: `src/baseline.py`.

- Vectoriser: `TfidfVectorizer(ngram_range=(1,2), min_df=3, max_features=30000)`
- Classifiers: `LogisticRegression(max_iter=1000, class_weight='balanced')` and `LinearSVC(class_weight='balanced')`
- Split: 80/20 stratified within `labeled_main.csv` (seed 42)
- **Stemming variant:** `src/ablation_stemming.py` retrains the baseline without stemming and with Porter stemming (`nltk.PorterStemmer`) applied to tokens before vectorising, reporting both macro-F1 scores on all three tiers. This is where the guideline's "stemming" step is satisfied (see §4.1); result is a net wash (numbers in `models/ablation_stemming.csv`).
- Evaluation: accuracy + macro-F1 on all three tiers (in-domain held-out, Trustpilot eval, forum eval)
- Saved artifact: `models/baseline_lr.pkl` (scikit-learn Pipeline)

The baseline's purpose is twofold: it is a rubric-compliant "traditional ML" option, and it sets the bar the XLM-R fine-tune must beat.

### 6.2 Advanced — XLM-RoBERTa fine-tune

File: `src/finetune_xlmr.py`.

- Base model: `xlm-roberta-base` (HuggingFace)
- Tokenizer: `XLMRobertaTokenizer`, `max_length=128`, truncation on
- Optimiser: `AdamW`, `lr=2e-5`, `weight_decay=0.01`
- Training: 5 epochs, batch size 16, warmup steps 100
- Loss: cross-entropy on three classes
- Trainer: HuggingFace `Trainer` with `load_best_model_at_end=True`, `metric_for_best_model='macro_f1'`
- Compute: Colab free-tier GPU (T4) is sufficient; falls back to CPU (~2 hours) if needed
- Saved artifact: `models/xlmr_final/` (full HF model folder, gitignored)

Why XLM-RoBERTa: it is multilingual, handles English + BM + Manglish code-switching natively, and is a strong SOTA baseline for short-text sentiment. This model choice also buys the "Transformer model" advanced-feature rubric point.

### 6.3 Aspect extraction — keyword-rule

File: `src/aspect_sa.py`. **Aspect keys are fixed.**

| Aspect | Example keywords (non-exhaustive) |
|---|---|
| `coverage` | coverage, signal, reception, no service, dropped call, liputan, kawasan |
| `speed` | speed, slow, fast, lag, buffer, ping, mbps, bandwidth, laju, lembab |
| `billing` | bill, invoice, price, plan, charge, overcharge, promo, mahal, murah, bayar |
| `customer_support` | support, customer service, agent, helpline, ticket, live chat, rude, helpful |
| `app_usability` | app, interface, ui, ux, bug, crash, login, glitch, freeze, error |

Matching: case-insensitive word-boundary regex on cleaned text. A review can match multiple aspects.

Keyword lists can be extended but keys **cannot** change. Keywords must be disjoint across aspects — if a word could plausibly belong to two aspects, assign it to the more specific one.

### 6.4 Emoji handling (advanced feature add-on)

Convert emoji to descriptive text via the `emoji` library (`emoji.demojize(text, delimiters=(' ', ' '))`) before tokenisation. Run evaluation with and without; report the delta in the report. This adds robustness for social-media-style text where sentiment is carried by emoji (🙄, 😡, 🔥).

## 7. Evaluation — the three-way table

File: `src/evaluate.py`.

Every classifier reports accuracy and macro-F1 on three evaluation sets:

| Tier | Description | Size |
|---|---|---|
| **In-domain** | Held-out 20% of `labeled_main.csv` | 240 rows (`labeled_main_test.csv`; the team built a 1,200-row balanced set, not the 360-row minimum) |
| **Cross-platform** | `trustpilot_eval.csv` double-labeled | 60 rows |
| **Cross-domain** | `forum_eval.csv` (Reddit + Lowyat, double-labeled) | 60 rows |

Expected pattern: `in-domain > cross-platform > cross-domain`. The gap sizes become the report's discussion material. The output is a CSV at `models/eval_results.csv` which the report pivots into a 2×3 table (LogReg vs XLM-R × three tiers).

`src/aspect_degradation.py` extends this table *per aspect* (advanced feature #5 — see §12 and `EXPERIMENT_aspect_degradation.md`): it is a thin analysis layer over the same models and aspect rules that asks whether the degradation is uniform across the five aspects. It runs independently and degrades gracefully when models/tiers are absent.

If Lowyat is dropped at Checkpoint A, the cross-domain tier becomes Reddit-only but still targets 60 items.

## 8. Translation — optional ablation only

**Not** part of the default pipeline. The default is: feed cleaned text directly to XLM-R, which handles BM/Manglish natively.

If time allows after Phase 5, run a small ablation:

1. Select ~100 BM-heavy items from `labeled_main.csv` (identified by langdetect = `ms` or by manual scan)
2. Translate to English using `deep-translator` (GoogleTranslator)
3. Re-evaluate XLM-R on translated inputs
4. Report the accuracy delta

Findings: typically either unchanged (XLM-R is already multilingual) or slightly worse (translation introduces noise). Either outcome is a reportable experimental result.

The report **must not** describe translation as part of the main pipeline. It appears only in the discussion or appendix.

## 9. Deployment

Files: `app/app.py`, plus a HuggingFace Spaces repo.

- Framework: `gradio` (SDK: Gradio on HF Spaces)
- Input: free-text textbox + source-type dropdown (Play Store, Trustpilot, Forum, Auto-detect)
- Output: Markdown block with sentiment, confidence, detected aspects, detected source
- Model loaded: `models/xlmr_final` (fallback message if model artifact absent)
- Launch target: `demo.launch(share=False)` locally; HF Spaces handles public URL

The HF Spaces URL and a screenshot of the live app go into the report's Deployment section.

## 10. Reproducibility

- Fixed seed `42` in `baseline.py` and `finetune_xlmr.py`
- All scrapers write raw CSVs with a `scrape_date` column so a re-run can be distinguished from the original run
- `requirements.txt` pins library versions
- `models/` and `data/raw/` gitignored — anyone cloning fresh re-runs scrapers and re-trains. Labeled data is committed so labeling effort is not redone.

## 11. Known limitations (report in Discussion)

- Play Store reviews skew toward recent dissatisfied users (selection bias in star ratings)
- Auto-labeling by stars is a proxy; sarcasm and mixed-sentiment reviews can be mis-labeled
- 3★ → `neutral` is a weak rule; many 3★ reviews carry strong complaints
- 360 training rows is small for a transformer; XLM-R may overfit — monitor dev-set loss carefully
- Keyword-rule aspect extraction misses implicit references ("my bill shot up" would match `billing` but "they charged me again" without the word `charge` would not — see if the list captures it)
- Cross-domain gap may be large: Reddit/Lowyat users write very differently from app reviewers, and the model has not seen this genre in training

These limitations are expected and make for good Discussion-section content. They are **not** project failures.

## 12. Advanced features (rubric mapping)

The guideline names three example advanced features **"including but not limited to"** and allocates 3 marks (system) + 2 marks (report). There is **no "choose 2" cap**, and the open-ended phrasing lets us count analysis-level contributions as advanced features. We ship five, the first three being the named examples and the last two drawn from our own analysis:

| # | Advanced feature | Type | Where it lives | Reportable result |
|---|---|---|---|---|
| 1 | **Aspect-based SA** | Named (rubric) | `src/aspect_sa.py` | Per-brand × per-aspect complaint composition; the project's centerpiece. |
| 2 | **Transformer model (XLM-RoBERTa)** | Named (rubric) | `src/finetune_xlmr.py` | Beats the TF-IDF baseline; handles BM/Manglish natively. |
| 3 | **Emoji handling** | Named (rubric) | `src/preprocess.py` (demojize) | ~7.8% of reviews carry an emoji; report the with/without accuracy delta honestly (expect modest). |
| 4 | **Multilingual / code-switching handling** | Ours (badged) | XLM-R + langdetect tagging | ~27% of the corpus is non-English; model needs no correct language tag (see §4.2). |
| 5 | **Cross-domain robustness — aspect-conditioned degradation** | Ours (badged) | `src/aspect_degradation.py` (analysis layer over `src/evaluate.py`) | Tests *whether* the in-domain → cross-platform → cross-domain degradation is uniform across aspects. **Hypothesis:** coverage/speed transfer, app_usability degrades most — to be confirmed by the run, not pre-asserted. See `EXPERIMENT_aspect_degradation.md`. |

Marks for the Advanced-Features criterion cap at 3% + 2%; features 4–5 do not add raw marks but materially strengthen the report's novelty narrative and provide redundancy if any single feature underperforms. Features 4–5 are the same work as the project's novelty plan — badging them here means they are presented twice (as advanced features *and* as findings) at no extra build cost.

---

_Last substantive revision: 2026-06-23._
