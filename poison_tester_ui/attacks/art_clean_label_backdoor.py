from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple

import numpy as np

from art.attacks.poisoning.clean_label_backdoor_attack import PoisoningAttackCleanLabelBackdoor
from poison_tester_ui.attacks.simple_image_mask_backdoor import SimpleImageMaskBackdoor


@dataclass
class CleanLabelBackdoorParams:
    pp_poison: float = 0.1
    norm: float | int | str = np.inf
    eps: float = 0.3
    eps_step: float = 0.1
    max_iter: int = 20
    num_random_init: int = 0

    location: str = "bottom-right"
    alpha: float = 1.0


def one_hot(label: int, nb_classes: int) -> np.ndarray:
    y = np.zeros((1, nb_classes), dtype=np.float32)
    y[0, int(label)] = 1.0
    return y


def run_clean_label_backdoor_art(
    *,
    proxy_classifier,  # ART classifier with loss gradients (KerasClassifier)
    x_train_nhwc: np.ndarray,
    y_train_int: np.ndarray,
    target_label: int,
    trigger_image: np.ndarray,
    trigger_mask: np.ndarray,
    params: CleanLabelBackdoorParams,
    seed: int = 1234,
) -> tuple[np.ndarray, np.ndarray, List[int], Dict[str, Any]]:
    """
    ART 的 PoisoningAttackCleanLabelBackdoor:
    - 只會在 "target label" 的資料子集合中挑 pp_poison 比例做 PGD + backdoor（labels 不變，但我們後續用 backdoor 成功率評估）
    - 它的 poison() 會回傳整個 data 與 labels（estimated_labels）；
      我們用 diff 來找 indices（保守做法：np.any(x_changed)）
    """
    np.random.seed(seed)

    nb_classes = proxy_classifier.nb_classes

    backdoor = SimpleImageMaskBackdoor(
        trigger_image=trigger_image,
        trigger_mask=trigger_mask,
        location=params.location,
        alpha=params.alpha,
    )

    attack = PoisoningAttackCleanLabelBackdoor(
        backdoor=backdoor,  # type: ignore[arg-type]
        proxy_classifier=proxy_classifier,
        target=one_hot(target_label, nb_classes),
        pp_poison=float(params.pp_poison),
        norm=params.norm,
        eps=float(params.eps),
        eps_step=float(params.eps_step),
        max_iter=int(params.max_iter),
        num_random_init=int(params.num_random_init),
    )

    # y for this attack: uses proxy_classifier.predict if y None, but we pass y one-hot to be stable
    y_onehot = np.eye(nb_classes, dtype=np.float32)[y_train_int]
    x_poisoned_all, y_est = attack.poison(x_train_nhwc, y_onehot, broadcast=True)

    # find changed indices (works for float32 arrays)
    diff = np.abs(x_poisoned_all - x_train_nhwc)
    changed = (diff.reshape(diff.shape[0], -1).max(axis=1) > 1e-8)
    poisoned_indices = np.where(changed)[0].astype(int).tolist()

    meta = {
        "poisoned_count": len(poisoned_indices),
        "poisoned_fraction": float(len(poisoned_indices) / max(1, x_train_nhwc.shape[0])),
        "target_label": int(target_label),
        "params": params.__dict__,
    }

    # labels 不改（clean-label）
    return x_poisoned_all, np.copy(y_train_int), poisoned_indices, meta