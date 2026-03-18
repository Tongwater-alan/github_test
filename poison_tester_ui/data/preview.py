from typing import Optional, Tuple, List
import numpy as np

from poison_tester_ui.data.npz_loader import DatasetSplits

def _to_uint8_hwc(img: np.ndarray, fmt: str) -> np.ndarray:
    if fmt == "NCHW":
        img = np.transpose(img, (1,2,0))
    if img.ndim == 2:
        img = np.stack([img,img,img], axis=-1)
    if img.shape[-1] == 1:
        img = np.repeat(img, 3, axis=-1)
    if img.dtype != np.uint8:
        img = np.clip(img, 0.0, 1.0)
        img = (img * 255.0).astype(np.uint8)
    if img.shape[-1] == 4:
        img = img[..., :3]
    return img

def sample_preview_items(
    splits: DatasetSplits,
    split: str = "train",
    label_filter: Optional[int] = None,
    k: int = 16,
) -> Tuple[List[Tuple[np.ndarray, str]], List[int]]:
    if split == "train":
        x, y = splits.x_train, splits.y_train
    elif split == "val":
        x, y = splits.x_val, splits.y_val
    else:
        x, y = splits.x_test, splits.y_test

    idx_all = np.arange(x.shape[0])
    if label_filter is not None:
        idx_all = idx_all[y == label_filter]
    if idx_all.size == 0:
        return [], []
    sel = np.random.choice(idx_all, size=min(k, idx_all.size), replace=False)
    items = []
    for idx in sel:
        img = _to_uint8_hwc(x[idx], splits.format)
        items.append((img, f"idx={int(idx)} label={int(y[idx])}"))
    return items, [int(i) for i in sel]