import platform
import sys
from typing import Dict, Any
import numpy as np

def _safe_import_version(pkg: str):
    try:
        mod = __import__(pkg)
        return getattr(mod, "__version__", "unknown")
    except Exception:
        return None

def collect_env_info() -> Dict[str, Any]:
    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "torch_version": _safe_import_version("torch"),
        "tensorflow_version": _safe_import_version("tensorflow"),
        "art_version": _safe_import_version("art"),
        "gradio_version": _safe_import_version("gradio"),
    }