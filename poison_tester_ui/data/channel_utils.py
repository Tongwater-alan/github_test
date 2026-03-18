"""Channel-count alignment utilities for NHWC float32 arrays."""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def align_channels_nhwc(
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
    model_channels: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[str]]:
    """Auto-align NHWC float32 arrays to *model_channels* (1 or 3).

    Supported conversions:

    - data C=3 → model C=1: luminance-weighted RGB→grayscale
    - data C=1 → model C=3: repeat single channel three times

    Returns the (possibly converted) arrays and an error string (or ``None``
    on success).  When conversion is not possible the original arrays are
    returned unchanged together with a descriptive error string.
    """
    data_channels = x_train.shape[-1] if x_train.ndim == 4 else None
    if data_channels is None or data_channels == model_channels:
        return x_train, x_val, x_test, None

    if model_channels == 1 and data_channels == 3:
        # RGB → grayscale using ITU-R BT.709 luminance weights
        w_rgb = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
        x_train = (x_train * w_rgb).sum(axis=-1, keepdims=True).astype(np.float32)
        x_val = (x_val * w_rgb).sum(axis=-1, keepdims=True).astype(np.float32)
        x_test = (x_test * w_rgb).sum(axis=-1, keepdims=True).astype(np.float32)
        return x_train, x_val, x_test, None

    if model_channels == 3 and data_channels == 1:
        # Grayscale → RGB by repeating the single channel
        x_train = np.repeat(x_train, 3, axis=-1).astype(np.float32)
        x_val = np.repeat(x_val, 3, axis=-1).astype(np.float32)
        x_test = np.repeat(x_test, 3, axis=-1).astype(np.float32)
        return x_train, x_val, x_test, None

    err = (
        f"Channel mismatch: model expects {model_channels} channels but data has "
        f"{data_channels} channels. Auto-conversion is only supported for 1↔3 channels."
    )
    return x_train, x_val, x_test, err
