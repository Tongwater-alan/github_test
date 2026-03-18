from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Tuple, List, Optional

import numpy as np

from art.attacks.poisoning.hidden_trigger_backdoor.hidden_trigger_backdoor import HiddenTriggerBackdoor

from poison_tester_ui.attacks.simple_image_mask_backdoor import SimpleImageMaskBackdoor


@dataclass
class HiddenTriggerParams:
    # ART params
    eps: float = 0.1
    learning_rate: float = 0.001
    decay_coeff: float = 0.95
    decay_iter: int = 2000
    stopping_threshold: float = 10.0
    max_iter: int = 2000
    batch_size: int = 32
    poison_percent: float = 0.1
    verbose: bool = True
    print_iter: int = 100

    # backdoor params (image/mask)
    location: str = "bottom-right"
    alpha: float = 1.0


def one_hot(label: int, nb_classes: int) -> np.ndarray:
    y = np.zeros((1, nb_classes), dtype=np.float32)
    y[0, int(label)] = 1.0
    return y


def run_hidden_trigger_backdoor_art(
    *,
    classifier,  # ART classifier (KerasClassifier)
    x_train_nhwc: np.ndarray,
    y_train_int: np.ndarray,  # (N,)
    target_label: int,
    source_label: int,
    feature_layer: str,
    trigger_image: np.ndarray,
    trigger_mask: np.ndarray,
    params: HiddenTriggerParams,
    seed: int = 1234,
) -> tuple[np.ndarray, np.ndarray, List[int], Dict[str, Any]]:
    """
    Returns:
    - x_train_poisoned (NHWC float32)
    - y_train_poisoned_int (N,)
    - poisoned_indices
    - meta
    """
    rng = np.random.RandomState(seed)
    np.random.seed(seed)

    nb_classes = classifier.nb_classes

    # target/source as arrays (per ART HiddenTriggerBackdoor._check_params)
    target = one_hot(target_label, nb_classes)
    source = one_hot(source_label, nb_classes)

    backdoor = SimpleImageMaskBackdoor(
        trigger_image=trigger_image,
        trigger_mask=trigger_mask,
        location=params.location,
        alpha=params.alpha,
    )

    attack = HiddenTriggerBackdoor(
        classifier=classifier,
        target=target,
        source=source,
        feature_layer=feature_layer,
        backdoor=backdoor,  # type: ignore[arg-type]
        eps=params.eps,
        learning_rate=params.learning_rate,
        decay_coeff=params.decay_coeff,
        decay_iter=params.decay_iter,
        stopping_threshold=params.stopping_threshold,
        max_iter=params.max_iter,
        batch_size=params.batch_size,
        poison_percent=params.poison_percent,
        is_index=False,
        verbose=params.verbose,
        print_iter=params.print_iter,
    )

    # Need y in one-hot for KerasClassifier predict/label handling; we have ints, convert:
    y_onehot = np.eye(nb_classes, dtype=np.float32)[y_train_int]

    poisoned_x, poisoned_indices = attack.poison(x_train_nhwc, y_onehot)
    poisoned_indices = list(map(int, poisoned_indices))

    x_out = np.copy(x_train_nhwc)
    x_out[poisoned_indices] = poisoned_x

    y_out = np.copy(y_train_int)
    y_out[poisoned_indices] = int(target_label)

    meta = {
        "poisoned_count": len(poisoned_indices),
        "poisoned_fraction": float(len(poisoned_indices) / max(1, x_train_nhwc.shape[0])),
        "target_label": int(target_label),
        "source_label": int(source_label),
        "feature_layer": feature_layer,
        "params": params.__dict__,
    }
    return x_out, y_out, poisoned_indices, meta