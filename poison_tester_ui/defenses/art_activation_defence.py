from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List

import numpy as np
from art.defences.detector.poison.activation_defence import ActivationDefence


@dataclass
class ActivationParams:
    nb_clusters: int = 2
    clustering_method: str = "KMeans"
    nb_dims: int = 10
    reduce: str = "PCA"
    cluster_analysis: str = "smaller"
    ex_re_threshold: float | None = None


def run_activation_defence(
    *,
    classifier,
    x_train: np.ndarray,
    y_train_int: np.ndarray,
    is_clean_gt: np.ndarray,
    params: ActivationParams,
) -> Dict[str, Any]:
    defence = ActivationDefence(
        classifier=classifier,
        x_train=x_train,
        y_train=y_train_int,
        generator=None,
        ex_re_threshold=params.ex_re_threshold,
    )
    # set params
    report, is_clean_lst = defence.detect_poison(
        nb_clusters=int(params.nb_clusters),
        clustering_method=params.clustering_method,
        nb_dims=int(params.nb_dims),
        reduce=params.reduce,
        cluster_analysis=params.cluster_analysis,
    )
    conf_json = defence.evaluate_defence(is_clean=is_clean_gt)

    flagged = [i for i, c in enumerate(is_clean_lst) if c == 0]

    # ActivationDefence doesn't return per-index score in a standard way; report is cluster-level.
    return {
        "name": "ActivationDefence",
        "params": params.__dict__,
        "report": report,
        "is_clean_lst": is_clean_lst,
        "flagged_indices": flagged,
        "confusion_matrix_json": conf_json,
    }