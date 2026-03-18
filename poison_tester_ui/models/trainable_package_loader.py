"""Load a trainable PyTorch model package from a zip file.

Expected zip content:
- model.py  – must expose ``build_model(num_classes: int) -> nn.Module``
- weights.pth – PyTorch ``state_dict``
"""
from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


class TrainablePackageError(Exception):
    """Raised when the zip package is invalid or cannot be loaded."""


@dataclass
class TrainablePackageInfo:
    model_py_path: str
    weights_pth_path: str
    extract_dir: str
    device: str
    input_shape_chw: Tuple[int, int, int]
    num_classes: int


def load_trainable_package(
    zip_path: str,
    num_classes: int,
    device: str = "cpu",
    input_shape_chw: Optional[Tuple[int, int, int]] = None,
    run_dir: Optional[str] = None,
) -> Tuple[object, TrainablePackageInfo]:
    """Load a trainable PyTorch model from a zip package and wrap it in an
    ART ``PyTorchClassifier``.

    Args:
        zip_path: Path to the ``.zip`` file containing ``model.py`` and
            ``weights.pth``.
        num_classes: Number of output classes (must match the model and the
            NPZ dataset).
        device: PyTorch device string, e.g. ``"cpu"`` or ``"cuda"``.
        input_shape_chw: Channel-first input shape ``(C, H, W)``.  Defaults
            to ``(3, 224, 224)`` when *None*.
        run_dir: If provided, the zip is extracted into
            ``{run_dir}/trainable_pkg/``; otherwise a temporary directory is
            used.

    Returns:
        ``(classifier, info)`` where *classifier* is an ART
        ``PyTorchClassifier`` and *info* contains metadata about the loaded
        package.

    Raises:
        TrainablePackageError: If required files are missing or the model
            cannot be loaded / validated.
    """
    import torch
    import torch.nn as nn

    try:
        from art.estimators.classification import PyTorchClassifier
    except ImportError as exc:
        raise TrainablePackageError(
            "adversarial-robustness-toolbox is not installed."
        ) from exc

    if input_shape_chw is None:
        input_shape_chw = (3, 224, 224)

    # ---- extract zip -------------------------------------------------------
    if run_dir is not None:
        extract_dir = os.path.join(run_dir, "trainable_pkg")
        os.makedirs(extract_dir, exist_ok=True)
    else:
        extract_dir = tempfile.mkdtemp(prefix="trainable_pkg_")

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            names = zf.namelist()
            missing = [f for f in ("model.py", "weights.pth") if f not in names]
            if missing:
                raise TrainablePackageError(
                    f"zip 內缺少檔案：{', '.join(missing)}"
                )
            zf.extractall(extract_dir)
    except zipfile.BadZipFile as exc:
        raise TrainablePackageError(f"無法解壓 zip：{exc}") from exc

    model_py_path = os.path.join(extract_dir, "model.py")
    weights_pth_path = os.path.join(extract_dir, "weights.pth")

    # ---- dynamic import model.py -------------------------------------------
    module_name = f"_trainable_pkg_{os.path.basename(extract_dir)}"
    spec = importlib.util.spec_from_file_location(module_name, model_py_path)
    if spec is None or spec.loader is None:
        raise TrainablePackageError("無法建立 model.py 的 module spec。")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)  # type: ignore[union-attr]
    except Exception as exc:
        raise TrainablePackageError(f"執行 model.py 失敗：{exc}") from exc

    if not hasattr(module, "build_model"):
        raise TrainablePackageError("model.py 缺少 build_model(num_classes) 函式。")

    # ---- build model and load weights --------------------------------------
    try:
        model = module.build_model(num_classes)
    except Exception as exc:
        raise TrainablePackageError(f"build_model({num_classes}) 失敗：{exc}") from exc

    try:
        state_dict = torch.load(weights_pth_path, map_location=device)
        model.load_state_dict(state_dict)
    except Exception as exc:
        raise TrainablePackageError(f"載入 weights.pth 失敗：{exc}") from exc

    model.eval()
    model = model.to(torch.device(device))

    # ---- wrap in ART PyTorchClassifier -------------------------------------
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    classifier = PyTorchClassifier(
        model=model,
        loss=criterion,
        optimizer=optimizer,
        input_shape=tuple(input_shape_chw),
        nb_classes=num_classes,
        clip_values=(0.0, 1.0),
        channels_first=True,
        device_type=device,
    )

    # ---- smoke test: forward pass ------------------------------------------
    dummy = np.zeros((2,) + tuple(input_shape_chw), dtype=np.float32)
    try:
        preds = classifier.predict(dummy, batch_size=2)
    except Exception as exc:
        raise TrainablePackageError(f"forward smoke test 失敗：{exc}") from exc

    if preds.shape != (2, num_classes):
        raise TrainablePackageError(
            f"模型輸出 shape {preds.shape} 與預期 (2, {num_classes}) 不符。"
        )

    info = TrainablePackageInfo(
        model_py_path=model_py_path,
        weights_pth_path=weights_pth_path,
        extract_dir=extract_dir,
        device=device,
        input_shape_chw=tuple(input_shape_chw),  # type: ignore[arg-type]
        num_classes=num_classes,
    )
    return classifier, info