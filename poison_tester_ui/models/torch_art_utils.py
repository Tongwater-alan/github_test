"""Utilities for listing and filtering PyTorch model layers for ART PyTorchClassifier."""
from __future__ import annotations

from typing import List, Tuple

import numpy as np


def list_named_module_layers(model) -> List[str]:
    """Return a list of all named submodule layer names from a PyTorch model.

    Only non-empty names are returned (the top-level '' entry is excluded).
    """
    return [name for name, _ in model.named_modules() if name]


def filter_supported_activation_layers(
    classifier,
    candidate_layers: List[str],
    input_shape_chw: Tuple[int, int, int],
) -> List[str]:
    """Return layers from *candidate_layers* that are supported by
    ``classifier.get_activations`` (ART PyTorchClassifier).

    A layer is considered *supported* when ``get_activations`` returns a
    non-None result without raising an exception.

    Args:
        classifier: ART ``PyTorchClassifier`` instance.
        candidate_layers: Layer names to probe.
        input_shape_chw: Channel-first shape ``(C, H, W)`` used to build a
            dummy input batch.

    Returns:
        Ordered list of layer names that produced valid activations.
    """
    c, h, w = input_shape_chw
    dummy = np.zeros((2, c, h, w), dtype=np.float32)

    supported: List[str] = []
    for layer in candidate_layers:
        try:
            acts = classifier.get_activations(
                dummy, layer_name=layer, batch_size=2, framework=False
            )
            if acts is not None:
                supported.append(layer)
        except Exception:
            pass
    return supported
