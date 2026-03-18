from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

@dataclass
class KerasModelInfo:
    framework: str = "tensorflow.keras"
    input_shape: Optional[Tuple[int, ...]] = None
    output_shape: Optional[Tuple[int, ...]] = None

def load_keras_h5_model(path: str):
    """
    Load a Keras .h5 model.
    Note: requires tensorflow installed.
    """
    import tensorflow as tf  # lazy import

    model = tf.keras.models.load_model(path, compile=False)
    # Try best-effort shapes
    in_shape = None
    out_shape = None
    try:
        in_shape = tuple(model.input_shape)  # type: ignore[attr-defined]
    except Exception:
        pass
    try:
        out_shape = tuple(model.output_shape)  # type: ignore[attr-defined]
    except Exception:
        pass
    info = KerasModelInfo(input_shape=in_shape, output_shape=out_shape)
    return model, info

def keras_predict_logits(model, x_nchw: np.ndarray) -> np.ndarray:
    """
    Our preprocessing produces NCHW float tensors for PyTorch.
    Keras typically expects NHWC. Convert NCHW->NHWC.
    Returns numpy logits of shape (N, num_classes).
    """
    x_nhwc = np.transpose(x_nchw, (0, 2, 3, 1)).astype(np.float32)
    y = model(x_nhwc, training=False)
    # y could be tf.Tensor
    try:
        y = y.numpy()
    except Exception:
        y = np.array(y)
    return y