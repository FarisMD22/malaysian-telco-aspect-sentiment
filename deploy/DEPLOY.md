# Deploying the Gradio app to HuggingFace Spaces

The app is `deploy/hf_space/` (app.py + aspect_sa.py + requirements.txt + README.md). The ~1.1 GB
XLM-R model is **not** bundled into the Space — it lives in a separate HF **model repo** and the
Space loads it by id at runtime via the `HF_MODEL_ID` variable. This keeps the Space small and fast
to build.

> These final steps need **your** HuggingFace account, so you run them (I can't push to your
> account). Everything else (the app, the Space scaffolding) is already done. There are two paths:
> **Option A — website (no CLI, easiest)** below, or **Option B — CLI** further down.

---

## Option A — via the website (no CLI)

You only need a free HuggingFace account (https://huggingface.co). No token, no install.

### A1. Upload the model to a public model repo
1. https://huggingface.co/new → **Model**. Owner = you, name = `xlmr-telco-sentiment`,
   visibility = **Public**. Create.
2. On the repo → **Files and versions** → **Add file** → **Upload files**.
3. Drag in **all 5 files from your local `models/xlmr_final/`**: `config.json`,
   `model.safetensors` (~1.1 GB — the upload bar will sit on this one a while; let it finish),
   `tokenizer.json`, `tokenizer_config.json`, `training_args.bin`.
4. **Commit changes**. Confirm `model.safetensors` shows ~1.1 GB and a "LFS" tag.

### A2. Create the Space and upload the app
1. https://huggingface.co/new-space → name = `telco-sentiment`, **SDK = Gradio**,
   visibility = Public. Create.
2. On the Space → **Files** → **Add file** → **Upload files**.
3. Drag in **the 4 files from `deploy/hf_space/`**: `app.py`, `aspect_sa.py`,
   `requirements.txt`, `README.md`. (Uploading our `README.md` replaces the auto-generated one —
   that's intended; it carries the correct Gradio `sdk_version`.)
4. **Commit changes**. The Space starts building automatically.

### A3. Point the Space at the model
Space → **Settings** → **Variables and secrets** → **New variable** (plain variable, not secret —
the model repo is public):
```
Name:  HF_MODEL_ID
Value: <your-username>/xlmr-telco-sentiment
```
The Space restarts and loads the model. Then jump to **Step 4 — Verify** at the bottom.

---

## Option B — via the CLI

### Prerequisites (once)

1. Create a free account at https://huggingface.co.
2. Create a **write** access token: https://huggingface.co/settings/tokens → "New token" → role
   **Write**.
3. Install + log in (PowerShell, from the repo root):
   ```powershell
   pip install -U "huggingface_hub[cli]"
   huggingface-cli login        # paste the WRITE token
   ```
   Replace `<user>` below with your HF username.

### B1. Upload the model to a HF model repo (public)

A public model repo means the Space can download it with no runtime token.
```powershell
huggingface-cli repo create xlmr-telco-sentiment --type model -y
huggingface-cli upload <user>/xlmr-telco-sentiment models/xlmr_final . --repo-type model
```
This uploads the contents of local `models/xlmr_final/` (config, `model.safetensors`, tokenizer) to
the repo root; safetensors is handled via LFS automatically. Verify at
`https://huggingface.co/<user>/xlmr-telco-sentiment` that `model.safetensors` is present.

### B2. Create the Space and push the app

```powershell
huggingface-cli repo create telco-sentiment --type space --space_sdk gradio -y
huggingface-cli upload <user>/telco-sentiment deploy/hf_space . --repo-type space
```
This pushes `app.py`, `aspect_sa.py`, `requirements.txt`, `README.md` to the Space root. The Space
starts building immediately.

### B3. Point the Space at the model

On the Space page → **Settings** → **Variables and secrets** → **New variable** (a plain variable
is fine since the model repo is public):
```
Name:  HF_MODEL_ID
Value: <user>/xlmr-telco-sentiment
```
The Space restarts and loads the model. (If you instead kept the model repo **private**, add an
`HF_TOKEN` *secret* with a read token so the Space can download it.)

---

## Step 4 — Verify + capture for the report (both options)

1. Open `https://huggingface.co/spaces/<user>/telco-sentiment`, wait for "Running".
2. Try the built-in examples (incl. the Manglish one) and confirm sentiment + per-aspect + language
   render.
3. **Screenshot** the running app for the report's Deployment section, and put the live URL in the
   report.

## Updating the app later

Re-run only the Space upload (model unchanged):
```powershell
huggingface-cli upload <user>/telco-sentiment deploy/hf_space . --repo-type space
```

## Local run (no deploy)

```powershell
python deploy/hf_space/app.py      # uses ./models/xlmr_final, opens http://127.0.0.1:7860
```

## Troubleshooting

- **"Model not loaded" banner in the app** → `HF_MODEL_ID` is unset/typo'd, or the model repo is
  private without an `HF_TOKEN` secret.
- **`ModuleNotFoundError: audioop`** → the Space ran on Python 3.13 (which removed stdlib
  `audioop`). Fixed by `python_version: "3.12"` in `deploy/hf_space/README.md`.
- **`ImportError: cannot import name 'HfFolder'` / `TypeError: unhashable type: 'dict'` (jinja2)** →
  both come from forcing the old Gradio 4.44.1 onto HF's modern dependency image. The fix is to NOT
  pin old Gradio: `deploy/hf_space/README.md` has **no `sdk_version`** (so the Space uses the current
  Gradio) and `requirements.txt` leaves transformers/huggingface_hub **unpinned** so pip resolves a
  consistent modern set. The app's API (Blocks, Label, Dataframe, Examples) is verified on Gradio 6.
- **Want a reproducible Gradio version** → add `sdk_version: <a current release>` back to
  `README.md` once you see which version built successfully (check the Space build log).
- **Slow first response** → CPU Space loads the 1.1 GB model on cold start; subsequent calls are
  fast. Fine for a demo; no GPU hardware needed.
