import json
import os
import zipfile
from typing import Any, Dict

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def save_json(path: str, obj: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def zip_dir(dir_path: str) -> str:
    zip_path = dir_path.rstrip("/\\") + ".zip"
    if os.path.exists(zip_path):
        os.remove(zip_path)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(dir_path):
            for fn in files:
                full = os.path.join(root, fn)
                rel = os.path.relpath(full, dir_path)
                zf.write(full, rel)
    return zip_path