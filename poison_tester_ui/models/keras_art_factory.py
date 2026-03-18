from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any

import numpy as np

from art.estimators.classification.keras import KerasClassifier


@dataclass
class KerasArtModelInfo:
    h5_path: str
    input_shape: Optional[Tuple[int, ...]]
    output_shape: Optional[Tuple[int, ...]]
    nb_classes: int
    layer_names: List[str]
    is_compiled: bool
    channels_first: bool
    clip_values: Optional[Tuple[float, float]]
    use_logits: bool


def load_keras_h5_model(path: str):
    import tensorflow as tf
    model = tf.keras.models.load_model(path, compile=True)  # 你保證可訓練/compiled
    return model


def infer_nb_classes_from_model(model) -> int:
    out_shape = getattr(model, "output_shape", None)
    if out_shape is None:
        raise ValueError("Keras model has no output_shape.")
    if isinstance(out_shape, list):
        nb = out_shape[0][-1]
    else:
        nb = out_shape[-1]
    if nb == 1:
        nb = 2
    return int(nb)


def get_layer_names(model) -> List[str]:
    try:
        return [layer.name for layer in model.layers if hasattr(layer, "name")]
    except Exception:
        return []


def is_model_compiled(model) -> bool:
    # compiled_loss / optimizer are strong hints
    return getattr(model, "optimizer", None) is not None and getattr(model, "loss", None) is not None


def build_art_keras_classifier(
    *,
    h5_path: str,
    clip_values: Optional[Tuple[float, float]] = (0.0, 1.0),
    channels_first: bool = False,
    use_logits: bool = True,
    preprocessing: Tuple[float, float] = (0.0, 1.0),
    input_layer: int = 0,
    output_layer: int = 0,
) -> tuple[KerasClassifier, KerasArtModelInfo]:
    """
    Create ART KerasClassifier from a compiled Keras model (.h5).
    - channels_first should be False for TF/Keras (NHWC).
    - preprocessing follows ART convention: (subtrahend, divisor), applied inside estimator.
      我們會在外面把資料整理成 float [0,1]，因此預設 preprocessing=(0,1)。
    """
    model = load_keras_h5_model(h5_path)
    nb_classes = infer_nb_classes_from_model(model)
    layer_names = get_layer_names(model)
    compiled = is_model_compiled(model)

    if not compiled:
        raise ValueError(
            "你的 .h5 似乎不是 compiled 模型（缺 optimizer/loss）。ART KerasClassifier 需要 compiled model。"
        )

    classifier = KerasClassifier(
        model=model,
        use_logits=use_logits,
        channels_first=channels_first,
        clip_values=clip_values,
        preprocessing=preprocessing,
        input_layer=input_layer,
        output_layer=output_layer,
    )

    info = KerasArtModelInfo(
        h5_path=h5_path,
        input_shape=getattr(model, "input_shape", None),
        output_shape=getattr(model, "output_shape", None),
        nb_classes=nb_classes,
        layer_names=layer_names,
        is_compiled=compiled,
        channels_first=channels_first,
        clip_values=clip_values,
        use_logits=use_logits,
    )
    return classifier, info