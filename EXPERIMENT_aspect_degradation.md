# Experiment — Aspect-conditioned cross-domain degradation

**Script:** `src/aspect_degradation.py` · **Feature:** advanced #5 (METHODOLOGY §12)

This experiment turns the project's headline degradation table (`src/evaluate.py`) into a
more defensible novelty result. It is a thin **analysis layer** over the existing evaluator
and the locked aspect rules — it duplicates no model logic.

## Hypothesis

- **H1:** cross-tier accuracy degradation is **not uniform across service aspects**. We
  *hypothesise* that `coverage` and `speed` (universal grievances, expressed similarly on
  every channel) degrade least from in-domain → cross-domain, while `app_usability`
  (an artefact of the app-store genre — forum users rarely discuss "the app crashed")
  degrades most.

This is a hypothesis to be **tested**, not an asserted result. No present-tense "collapses"
claim is made anywhere until the run produces the numbers.

## Method

- **Models:** LogReg baseline and XLM-R (whichever are trained; missing ones skip).
- **Tiers (degradation order):** Play Store in-domain → Trustpilot cross-platform →
  Reddit/Lowyat cross-domain (whichever eval sets exist).
- **Unit of analysis:** for each `(model, tier, aspect)`, the subset of that tier's rows
  whose `extract_aspects(cleaned_text)` contains the aspect. **Subsets overlap** — a row
  mentioning both billing and speed is counted in both slices. This is intended.
- **Aspect tagging** is always recomputed from `cleaned_text` (never read from a stored
  `aspects` column, which a CSV would serialise as the string `"[]"`).

## Metric

- **Primary: accuracy** per aspect slice — *not* macro-F1, because small slices rarely
  contain all three sentiment classes, which makes macro-F1 undefined/misleading.
- Every cell reports **support `n`** and a **Wilson 95% confidence interval** on accuracy,
  so the report shows uncertainty honestly instead of over-reading a noisy 5-row slice.
- **Small-sample rule:** slices with `n < MIN_SUPPORT` (= 8) are flagged `low_support=True`,
  greyed in the heatmap, and **excluded from the headline claim** (stated openly, not hidden).
- **`macro_f1`** is a nullable column: populated only on the `__overall__` reference row,
  `NaN` on per-aspect slices.

### Reconciliation with `evaluate.py`

The `__overall__` `macro_f1` uses the **exact same call** as `evaluate.py`:
`f1_score(y_true, y_pred, average="macro")` (no `labels=`/`zero_division=`). So for any
(model, tier) the two scripts produce identical headline numbers — use this as a check.

> Future option (not used now): switching both scripts together to
> `labels=[0,1,2], zero_division=0` would report more honestly when a tier is single-class
> (the Trustpilot tier is ~all-negative). Changing only this script would break reconciliation,
> so it is deferred as a joint change.

## Outputs

- `models/aspect_degradation.csv` — **always written** (header-only if nothing could be
  scored), columns: `model,tier,aspect,n,accuracy,ci_low,ci_high,macro_f1,low_support`.
- `models/aspect_degradation_<model_slug>.png` — one heatmap per model (aspect rows × tier
  columns; `__overall__` excluded). Best-effort: requires matplotlib (in `requirements.txt`).

## Reading the result

For each aspect, the **slope of accuracy across the three ordered tiers** is the finding:
a flat slope means the aspect transfers across channels; a steep negative slope means it is
genre-bound. H1 is supported if `coverage`/`speed` slopes are flatter than `app_usability`'s,
on slices that clear `MIN_SUPPORT`.

## Threats to validity (report in Discussion)

- Keyword-rule aspect tagging is lossy — implicit mentions ("charged me again" without the
  word *charge*) are missed; slices are a proxy for true aspect membership.
- The Trustpilot tier is ~all-negative (single-class), so accuracy there is high by default
  and must be read with caution — lean on the in-domain → cross-domain contrast.
- Small per-aspect `n` in 60-row eval tiers — hence the Wilson CI and `MIN_SUPPORT` gate.
- These are already listed in METHODOLOGY §11.

## How to run

```bash
python src/aspect_degradation.py          # from repo root
# or
python -m src.aspect_degradation
```

Graceful by design: with no models trained yet it tags aspects, prints which models/tiers
are missing, and writes a header-only CSV without crashing. As `models/baseline_lr.pkl`, the
eval sets, and `models/xlmr_final` arrive, rerun to fill in the full 2×3×5 picture.
