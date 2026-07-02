# Labeling Rubric — TNL6323 Project

**Apply this rubric to every item you label.** Consistency across members is how we score a high Cohen's κ.

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

## For double-labeled eval items

Every Trustpilot and Reddit/Lowyat eval item is labeled by **two** members independently. A third member arbitrates disagreements. Track Cohen's κ per tier. Target κ ≥ 0.6.

## Workflow (Rain coordinates)

1. Open the shared Google Sheet (link in group chat).
2. Find rows assigned to you. Fill `your_label` using this rubric.
3. If you disagree with a pre-label or find the text unclear, leave a note in `notes`.
4. Done labeling? Mark your name in the "finished" tab. Rain calculates κ.
