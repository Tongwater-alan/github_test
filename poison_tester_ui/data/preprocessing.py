from dataclasses import dataclass
from typing import Tuple, Literal, Optional

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

from poison_tester_ui.data.npz_loader import DatasetSplits

NormalizeMode = Literal["imagenet_default", "estimate_from_train"]

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

@dataclass
class PreprocessConfig:
    resize_hw: Tuple[int, int] = (224, 224)
    normalize_mode: NormalizeMode = "imagenet_default"
    estimate_n: int = 1024

class Preprocess:
    """
    提供兩種輸出：
    - to_tensor_batch: PyTorch 用（NCHW, torch.Tensor, normalize）
    - to_numpy_nhwc: TF/ART(KerasClassifier) 用（NHWC, np.float32, normalize or not）
      *注意*：ART KerasClassifier 內部還有 preprocessing=(sub,div)，我們預設(0,1)，所以 normalize 在外面做即可。
    """
    def __init__(self, cfg: PreprocessConfig, mean, std, fmt: str):
        self.cfg = cfg
        self.mean = mean
        self.std = std
        self.fmt = fmt  # "NHWC" or "NCHW"

    def _to_pil(self, img: np.ndarray) -> Image.Image:
        if img.dtype != np.uint8:
            img = np.clip(img, 0.0, 1.0)
            img = (img * 255.0).astype(np.uint8)
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        if img.shape[-1] == 1:
            img = np.repeat(img, 3, axis=-1)
        if img.shape[-1] == 4:
            img = img[..., :3]
        return Image.fromarray(img)

    def _as_hwc_uint8(self, img: np.ndarray) -> np.ndarray:
        if self.fmt == "NCHW":
            img = np.transpose(img, (1, 2, 0))
        if img.dtype != np.uint8:
            img = np.clip(img, 0.0, 1.0)
            img = (img * 255.0).astype(np.uint8)
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        if img.shape[-1] == 1:
            img = np.repeat(img, 3, axis=-1)
        if img.shape[-1] == 4:
            img = img[..., :3]
        return img

    def to_tensor_batch(self, x: np.ndarray) -> torch.Tensor:
        imgs = []
        for i in range(x.shape[0]):
            img = self._as_hwc_uint8(x[i])
            pil = Image.fromarray(img)
            pil = pil.resize(self.cfg.resize_hw[::-1], resample=Image.BILINEAR)
            t = TF.to_tensor(pil)  # [0,1], CHW
            t = TF.normalize(t, mean=self.mean, std=self.std)
            imgs.append(t)
        return torch.stack(imgs, dim=0)

    def to_numpy_nhwc(self, x: np.ndarray, *, normalize: bool = True) -> np.ndarray:
        arr = []
        for i in range(x.shape[0]):
            img = self._as_hwc_uint8(x[i])
            pil = Image.fromarray(img)
            pil = pil.resize(self.cfg.resize_hw[::-1], resample=Image.BILINEAR)
            t = TF.to_tensor(pil)  # CHW float32 [0,1]
            if normalize:
                t = TF.normalize(t, mean=self.mean, std=self.std)
            nhwc = t.permute(1, 2, 0).contiguous().numpy().astype(np.float32)
            arr.append(nhwc)
        return np.stack(arr, axis=0)


def _estimate_mean_std(splits: DatasetSplits, cfg: PreprocessConfig):
    x = splits.x_train
    n = min(int(cfg.estimate_n), x.shape[0])
    if n <= 0:
        return IMAGENET_MEAN, IMAGENET_STD
    idx = np.random.choice(x.shape[0], size=n, replace=False)

    imgs = []
    tmp = Preprocess(cfg, IMAGENET_MEAN, IMAGENET_STD, splits.format)
    for i in range(n):
        img = tmp._as_hwc_uint8(x[idx[i]])
        pil = Image.fromarray(img).resize(cfg.resize_hw[::-1], resample=Image.BILINEAR)
        t = TF.to_tensor(pil)  # [0,1]
        imgs.append(t)
    xb = torch.stack(imgs, dim=0)
    mean = xb.mean(dim=[0,2,3]).tolist()
    std = xb.std(dim=[0,2,3]).tolist()
    std = [s if s > 1e-6 else 1e-6 for s in std]
    return (float(mean[0]), float(mean[1]), float(mean[2])), (float(std[0]), float(std[1]), float(std[2]))


def build_preprocess(splits: DatasetSplits, cfg: PreprocessConfig) -> Preprocess:
    if cfg.normalize_mode == "imagenet_default":
        mean, std = IMAGENET_MEAN, IMAGENET_STD
    else:
        mean, std = _estimate_mean_std(splits, cfg)
    return Preprocess(cfg, mean, std, splits.format)