from dataclasses import dataclass
from typing import Literal, Dict, Any, Tuple

import numpy as np

FormatType = Literal["NHWC", "NCHW"]

@dataclass
class DatasetSplits:
    schema_used: str
    format: FormatType
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    num_classes: int

    def to_summary_dict(self) -> Dict[str, Any]:
        return {
            "schema_used": self.schema_used,
            "format": self.format,
            "x_train_shape": list(self.x_train.shape),
            "y_train_shape": list(self.y_train.shape),
            "x_val_shape": list(self.x_val.shape),
            "y_val_shape": list(self.y_val.shape),
            "x_test_shape": list(self.x_test.shape),
            "y_test_shape": list(self.y_test.shape),
            "num_classes": self.num_classes,
        }


def _infer_format(x: np.ndarray) -> FormatType:
    if x.ndim != 4:
        raise ValueError(f"x must be 4D, got shape {x.shape}")
    if x.shape[-1] in (1, 3):
        return "NHWC"
    if x.shape[1] in (1, 3):
        return "NCHW"
    return "NHWC"


def _ensure_labels_1d(y: np.ndarray) -> np.ndarray:
    if y.ndim == 1:
        return y.astype(np.int64)
    if y.ndim == 2:
        return np.argmax(y, axis=1).astype(np.int64)
    raise ValueError(f"Unsupported y shape: {y.shape}")


def _infer_num_classes(*ys: np.ndarray) -> int:
    mx = 0
    for y in ys:
        y1 = _ensure_labels_1d(y)
        mx = max(mx, int(y1.max()) if y1.size > 0 else 0)
    return mx + 1


def _split_train_into_train_val(
    x_train: np.ndarray,
    y_train: np.ndarray,
    val_ratio: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not (0.0 < val_ratio < 1.0):
        raise ValueError(f"val_ratio must be in (0,1), got {val_ratio}")
    n = x_train.shape[0]
    n_val = max(1, int(round(n * val_ratio)))
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    val_idx = perm[:n_val]
    tr_idx = perm[n_val:]
    return x_train[tr_idx], y_train[tr_idx], x_train[val_idx], y_train[val_idx]


def load_npz_splits(npz_path: str, *, val_ratio: float = 0.1, seed: int = 1234) -> DatasetSplits:
    """
    Supported schemas:
    - schema-1a (train/val/test): x_train,y_train,x_val,y_val,x_test,y_test
    - schema-1b (train/test only): x_train,y_train,x_test,y_test  -> auto split train into train/val
    - schema-2 (indices): x,y,train_idx,val_idx,test_idx
    """
    data = np.load(npz_path, allow_pickle=False)
    keys = set(data.files)

    if {"x_train","y_train","x_val","y_val","x_test","y_test"}.issubset(keys):
        schema_used = "schema-1a"
        x_train = data["x_train"]
        y_train = _ensure_labels_1d(data["y_train"])
        x_val = data["x_val"]
        y_val = _ensure_labels_1d(data["y_val"])
        x_test = data["x_test"]
        y_test = _ensure_labels_1d(data["y_test"])

    elif {"x_train","y_train","x_test","y_test"}.issubset(keys):
        schema_used = "schema-1b(auto-val)"
        x_train_full = data["x_train"]
        y_train_full = _ensure_labels_1d(data["y_train"])
        x_test = data["x_test"]
        y_test = _ensure_labels_1d(data["y_test"])
        x_train, y_train, x_val, y_val = _split_train_into_train_val(
            x_train_full, y_train_full, val_ratio=val_ratio, seed=seed
        )

    elif {"x","y","train_idx","val_idx","test_idx"}.issubset(keys):
        schema_used = "schema-2"
        x = data["x"]
        y = _ensure_labels_1d(data["y"])
        train_idx = data["train_idx"].astype(np.int64)
        val_idx = data["val_idx"].astype(np.int64)
        test_idx = data["test_idx"].astype(np.int64)
        x_train, y_train = x[train_idx], y[train_idx]
        x_val, y_val = x[val_idx], y[val_idx]
        x_test, y_test = x[test_idx], y[test_idx]

    else:
        raise ValueError(f"NPZ keys not recognized. Found keys: {sorted(list(keys))}")

    fmt = _infer_format(x_train)
    num_classes = _infer_num_classes(y_train, y_val, y_test)

    return DatasetSplits(
        schema_used=schema_used,
        format=fmt,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
        num_classes=num_classes,
    )