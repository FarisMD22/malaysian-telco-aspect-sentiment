"""
Local launcher for the Gradio app.

The app lives in `deploy/hf_space/app.py` (the same file pushed to the HuggingFace Space).
This shim makes `python app/app.py` work from the repo root; model resolution is handled in
deploy/hf_space/app.py (HF_MODEL_ID, then a local ./models/xlmr_final, then the hosted model).
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
