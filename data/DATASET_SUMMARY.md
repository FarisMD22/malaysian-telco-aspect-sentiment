# Dataset Summary — Malaysian Telco & Broadband Sentiment

**TNL6323 Natural Language Processing · Group Project (40%)**

This is a one-page overview of the data collected for our Malaysia-focused telco &
broadband sentiment analysis system. It covers what we gathered, from where, how it
is labelled, and how it maps onto our three-tier evaluation design.

---

## 1. At a glance

- **17,392 raw records** collected across **three independent sources**.
- **9,619** Play Store reviews cleaned (deduped, language-detected, short reviews dropped).
- **1,200-row labelled training set**, perfectly balanced **400 / 400 / 400** across
  positive / neutral / negative.
- **Multilingual & code-switched**: English, Bahasa Melayu, and Manglish, including emoji.
- **Three evaluation tiers** designed to measure how accuracy degrades as we move away
  from the training domain — this degradation pattern is the analytical headline of the report.

---

## 2. Sources and the three-tier design

| Source | Role / Evaluation tier | Raw rows | Why this source |
|---|---|---|---|
| **Google Play Store** (7 telco apps) | In-domain — **training + in-domain test** | 15,946 | Primary labelled training data; star ratings give weak sentiment labels |
| **Trustpilot** (3 brand pages) | **Cross-platform** eval | 170 | Same brands, different platform & writing style — tests platform shift |
| **Lowyat.NET forum** (telco sub-forums) | **Cross-domain** eval | 1,276 | Long-form Malaysian forum discussion — tests domain shift |
| | **Total raw** | **17,392** | |

**The research question the tiers answer:** a model trained on Play Store reviews is
evaluated on (1) held-out Play Store data, (2) Trustpilot, and (3) Lowyat. We expect
accuracy to drop from tier 1 → 3; quantifying that drop is the main finding.

> Note on Reddit: Reddit was the original cross-domain plan, but Reddit's 2023+ API
> changes now route app registration through their on-platform "Devvit" framework,
> which does not issue scraping credentials. We pivoted the cross-domain tier to
> **Lowyat.NET**, a Malaysian forum — arguably a better fit, as it is Malaysia-specific
> and richer in code-switched language.

---

## 3. Play Store breakdown (the training source)

Reviews per app (raw), scraped live with `google-play-scraper` (country = MY):

| App | Raw reviews |
|---|---|
| MyCelcomDigi | 2,600 |
| MyMaxis | 2,600 |
| Hotlink | 2,600 |
| MyUMobile | 2,600 |
| MyUnifi | 2,600 |
| Yes 5G | 2,600 |
| TIME Self Care | 346 |
| **Total** | **15,946** |

After cleaning (dedupe, language detection, drop reviews < 5 words): **9,619 reviews**,
spanning **Dec 2021 – Jun 2026**, median length **16 words**.

---

## 4. The labelled training set (`labeled_main.csv`)

- **1,200 rows**, **balanced 400 / 400 / 400** (positive / neutral / negative).
- **Auto-labelled by star rating**: 1–2★ → negative, 3★ → neutral, 4–5★ → positive.
- Split into a **960-row train / 240-row test** held-out set.
- Unified schema (one row = one review):

  `id · text · rating · date · source · app · cleaned_text · word_count · language · sentiment_label`

**Language mix (auto-detected):** ~73% English, ~21% Bahasa Melayu/Manglish*, remainder
mixed. *(Note: the `langdetect` library tags Bahasa Melayu as `id`/Indonesian because the
two languages are very close — these are Malaysian reviews, not Indonesian.)*

**Sample rows (verbatim):**

| Sentiment | Rating | App | Review |
|---|---|---|---|
| negative | 1★ | MyMaxis | *Menyesal pulak update.. Selepas update terus tak boleh bukak* |
| positive | 5★ | Yes 5G | *Yes prepaid is the best 👍🏻 1000/100* |
| neutral | 3★ | MyUMobile | *Bil bulan lepas saya dah bayar rm140...tetapi kenapa bulan ni tiada penolakan amaun..* |

These illustrate the core challenge: sentiment must be read across **English, Bahasa
Melayu, Manglish, and emoji** — which is why we use a multilingual transformer (XLM-RoBERTa)
rather than an English-only model.

---

## 5. Aspect annotation (five locked aspects)

Beyond sentiment, each review is tagged for which service aspect it discusses, using a
keyword-rule extractor (`labeled_main_with_aspects.csv`). The five aspects are:

`coverage · speed · billing · customer_support · app_usability`

`app_usability` and `speed` are the most frequently discussed.

---

## 6. Evaluation-set characteristics (honest limitations)

- **Trustpilot is heavily negative**: of 170 reviews, ~156 are 1★ (people visit Trustpilot
  mostly to complain). The cross-platform eval sample will therefore be negative-dominated —
  a real class-balance limitation we discuss in the report.
- **Lowyat is long-form**: 1,276 posts, 1,188 usable (≥ 5 words), median 21 words — a strong
  contrast to short app-store reviews, which is exactly the domain shift we want to measure.
- **Neutral is the scarce class everywhere**: telco reviews skew to strong opinions, so
  genuine neutral examples are rare. We cap the balanced training set at 400/class to keep
  a neutral buffer. This skew is documented as a limitation.

---

## 7. Status

All three sources are collected; Play Store is cleaned and labelled; the balanced 1,200-row
training set is built; baseline models are trained; and aspects are extracted. The two 60-row
evaluation samples (Trustpilot cross-platform, Lowyat cross-domain) are labelled by a single
annotator per `labeling/rubric.md`; Trustpilot additionally reports a human-vs-star proxy
agreement (see `data/labeled/kappa.md`).

---

*Files: raw data in `data/raw/`, cleaned corpus in `data/cleaned/all_sources.csv`,
labelled set in `data/labeled/labeled_main.csv`.*
