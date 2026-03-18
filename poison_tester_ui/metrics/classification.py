from __future__ import annotations
from typing import Tuple, Dict, Any, Optional

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def accuracy_from_probs_or_logits(pred: np.ndarray, y_true_int: np.ndarray) -> float:
    pred_label = np.argmax(pred, axis=1)
    return float((pred_label == y_true_int).mean())


def asr_from_probs_or_logits(pred_triggered: np.ndarray, target_label: int) -> float:
    pred_label = np.argmax(pred_triggered, axis=1)
    return float((pred_label == int(target_label)).mean())


def roc_auc_from_scores(is_poison_gt: np.ndarray, scores: np.ndarray) -> Dict[str, Any]:
    """
    is_poison_gt: 1=poison, 0=clean
    scores: higher means more suspicious (poison)
    """
    y = is_poison_gt.astype(int)
    if len(np.unique(y)) < 2:
        return {"auc": None, "fpr": [], "tpr": [], "thresholds": []}
    auc = float(roc_auc_score(y, scores))
    fpr, tpr, thr = roc_curve(y, scores)
    return {"auc": auc, "fpr": fpr.tolist(), "tpr": tpr.tolist(), "thresholds": thr.tolist()}