# Labeling Rubric — TNL6323 Project

**Apply this rubric consistently to every item you label** — consistent application is what keeps the labels reliable.

## Three labels

- **positive** — endorses, praises, recommends, or expresses satisfaction with the service
- **negative** — complains, criticises, expresses frustration, or calls out a problem
- **neutral** — factual, mixed (both pos and neg roughly balanced), a question, or too bland to carry sentiment

## Decision rules (apply in order)

1. **Is the text about the telco/broadband service itself?** If not (pure off-topic chit-chat, meme, spam), mark as neutral and add `off_topic` in the notes column.
2. **Is one polarity clearly dominant?** If yes → that label. If both appear roughly equal → neutral.
3. **Sarcasm and irony count.** Read the whole sentence. "Thanks Maxis for nothing 🙄" = negative.
4. **Code-switching is fine.** Label based on your understanding — do not translate first. Native Bahasa Melayu, Manglish, and Chinese-flavoured English are all valid inputs.
5. **Questions without opinion → neutral.** "How do I change my plan?" is neutral.
6. **If genuinely in doubt, choose neutral.** Don't force polarity.

## Worked examples

| Text | Label | Why |
|---|---|---|
| "Best network in Malaysia, keep it up!" | positive | Clear endorsement |
| "Buffering every 5 min, so laggy." | negative | Clear complaint |
| "How do I change my plan?" | neutral | Question, no opinion |
| "The 5G is fast but coverage is still patchy." | neutral | Balanced mixed |
| "Slow but at least it works." | negative | Negative dominates |
| "Thanks Maxis for nothing 🙄" | negative | Sarcasm |
| "Just signed up today lah." | neutral | Factual, no sentiment |
| "Unifi signal kat area aku okay je" | positive | "okay je" = fine/good in Manglish |
| "Rm50 sebulan too mahal dah" | negative | "mahal" = expensive; pricing complaint |

## Eval-set labeling

The Trustpilot and Reddit/Lowyat eval items are labeled by a single annotator using this rubric.
For Trustpilot, agreement is checked against a star-derived proxy label — a human-vs-star validity
check, not inter-annotator agreement; see `data/labeled/kappa.md`. Lowyat has no star proxy.

## Workflow

1. Open the eval sheet (`labeling/*_eval_sheet.csv`).
2. Fill `label_1` for each row using this rubric.
3. If the text is unclear, leave a note in `notes`.
4. Run `python labeling/compute_kappa.py` to export the final eval sets and the agreement report.
