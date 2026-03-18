from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Tuple, List

import numpy as np
from art.defences.detector.poison.spectral_signature_defense import SpectralSignatureDefense


@dataclass
class SpectralParams:
    expected_pp_poison: float = 0.1
    batch_size: int = 128
    eps_multiplier: float = 1.5


def run_spectral_signature_defense(
    *,
    classifier,
    x_train: np.ndarray,
    y_train_int: np.ndarray,
    is_clean_gt: np.ndarray,  # 1 clean, 0 poison
    params: SpectralParams,
) -> Dict[str, Any]:
    defence = SpectralSignatureDefense(
        classifier=classifier,
        x_train=x_train,
        y_train=y_train_int,
        expected_pp_poison=float(params.expected_pp_poison),
        batch_size=int(params.batch_size),
        eps_multiplier=float(params.eps_multiplier),
    )
    report, is_clean_lst = defence.detect_poison()
    conf_json = defence.evaluate_defence(is_clean=is_clean_gt)

    # build scores array
    scores = np.zeros((x_train.shape[0],), dtype=np.float32)
    for k, v in report.items():
        scores[int(k)] = float(v)
    flagged = [i for i, c in enumerate(is_clean_lst) if c == 0]

    return {
        "name": "SpectralSignatureDefense",
        "params": params.__dict__,
        "report": report,
        "is_clean_lst": is_clean_lst,
        "flagged_indices": flagged,
        "scores": scores.tolist(),
        "confusion_matrix_json": conf_json,
    }