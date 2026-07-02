# Labeling agreement (kappa.md)

**Single annotator.** True inter-annotator Cohen's kappa requires two independent human labelers (METHODOLOGY.md sec 3.4); that is not available here. Trustpilot reports a human-vs-star **proxy** kappa instead, from a **blind** re-label (the annotator did not see the star rating or any pre-label, so the agreement is genuine, not anchored), a validity check, *not* inter-annotator agreement. Lowyat has no proxy and no second annotator, so kappa is N/A.

## trustpilot

- rows: 60 | labeled: 60
- class balance: {'negative': 56, 'positive': 4}
- labeling: **blind** (annotator did not see star rating or pre_label)
- raw agreement (blind human vs star-proxy): 98.3%
- **proxy kappa = 0.881** (blind human vs star; compared against the 0.6 reference threshold, but **not** counted as an inter-annotator pass/fail)
- caveat: only 4/60 non-negative -> proxy kappa and macro-F1 are low-support, **directional only**.

- exported: `data/labeled/trustpilot_eval.csv`

## forum

- rows: 60 | labeled: 60
- class balance: {'neutral': 52, 'negative': 4, 'positive': 4}
- **kappa: N/A** (single annotator, no star proxy)

- exported: `data/labeled/forum_eval.csv`

## Optional follow-up

A genuine consistency number is still obtainable via **test-retest**: relabel a sample days apart and report intra-annotator kappa. Not done by default.
