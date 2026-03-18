from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Any

import numpy as np

class SimpleImageMaskBackdoor:
    """
    最小相容 backdoor 物件，提供 .poison(x, y, broadcast=True) -> (x_poisoned, y_out)
    - x: NHWC float32 or uint8
    - y: target label (one-hot) or indices; HiddenTriggerBackdoorKeras 會傳 self.target (one-hot)
    """
    def __init__(self, trigger_image: np.ndarray, trigger_mask: np.ndarray, location: str, alpha: float):
        self.trigger_image = self._ensure_hwc_uint8(trigger_image)
        self.trigger_mask = self._ensure_hw_uint8(trigger_mask)
        self.location = location
        self.alpha = float(alpha)

    def _ensure_hwc_uint8(self, img: np.ndarray) -> np.ndarray:
        if img.dtype != np.uint8:
            img = np.clip(img, 0.0, 1.0)
            img = (img * 255.0).astype(np.uint8)
        if img.ndim == 2:
            img = np.stack([img,img,img], axis=-1)
        if img.shape[-1] == 4:
            img = img[..., :3]
        if img.shape[-1] == 1:
            img = np.repeat(img, 3, axis=-1)
        return img

    def _ensure_hw_uint8(self, m: np.ndarray) -> np.ndarray:
        if m.ndim == 3:
            m = m[..., 0]
        if m.dtype != np.uint8:
            m = np.clip(m, 0.0, 1.0)
            m = (m * 255.0).astype(np.uint8)
        return m

    def _paste(self, img_hwc_uint8: np.ndarray) -> np.ndarray:
        H, W, _ = img_hwc_uint8.shape
        th, tw = self.trigger_image.shape[0], self.trigger_image.shape[1]
        if self.location == "top-left":
            top, left = 0, 0
        elif self.location == "top-right":
            top, left = 0, W - tw
        elif self.location == "bottom-left":
            top, left = H - th, 0
        elif self.location == "random":
            top = np.random.randint(0, max(1, H - th + 1))
            left = np.random.randint(0, max(1, W - tw + 1))
        else:
            top, left = H - th, W - tw

        out = img_hwc_uint8.copy()
        patch = out[top:top+th, left:left+tw, :].astype(np.float32)
        trig = self.trigger_image.astype(np.float32)
        mask = (self.trigger_mask > 127).astype(np.float32)[..., None]
        blended = (1 - self.alpha * mask) * patch + (self.alpha * mask) * trig
        out[top:top+th, left:left+tw, :] = np.clip(blended, 0, 255).astype(np.uint8)
        return out

    def poison(self, x: np.ndarray, y: Any, broadcast: bool = True, **kwargs):
        data = np.copy(x)
        # accept NHWC float or uint8
        x_uint8 = data
        if x_uint8.dtype != np.uint8:
            x_uint8 = np.clip(x_uint8, 0.0, 1.0)
            x_uint8 = (x_uint8 * 255.0).astype(np.uint8)

        for i in range(x_uint8.shape[0]):
            x_uint8[i] = self._paste(x_uint8[i])

        # return in same dtype as input
        if data.dtype == np.uint8:
            x_out = x_uint8
        else:
            x_out = (x_uint8.astype(np.float32) / 255.0).astype(np.float32)

        return x_out, y