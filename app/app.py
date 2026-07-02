"""
Local launcher for the Gradio app.

The canonical app now lives in `deploy/hf_space/app.py` (the same file that is pushed to the
HuggingFace Space — see deploy/DEPLOY.md), so there is a single source of truth. This shim just
makes `python app/app.py` keep working from the repo root for local development; it loads the
model from ./models/xlmr_final unless HF_MODEL_ID is set.
"""
import importlib.util
import sys
from pathlib import Path

_SPACE = Path(__file__).resolve().parents[1] / "deploy" / "hf_space"
sys.path.insert(0, str(_SPACE))  # so the app's `import aspect_sa` resolves

_spec = importlib.util.spec_from_file_location("space_app", _SPACE / "app.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

if __name__ == "__main__":
    _mod.demo.launch()
