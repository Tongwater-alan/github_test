import os
import numpy as np
import matplotlib.pyplot as plt

def plot_metrics_bar(out_path: str, baseline_acc: float, attacked_acc: float, defended_acc: float,
                     baseline_asr: float, attacked_asr: float, defended_asr: float) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    labels = ["baseline", "attacked", "defended"]
    acc = [baseline_acc, attacked_acc, defended_acc]
    asr = [baseline_asr, attacked_asr, defended_asr]

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].bar(labels, acc)
    ax[0].set_title("Clean Accuracy (test)")
    ax[0].set_ylim(0, 1)

    ax[1].bar(labels, asr)
    ax[1].set_title("ASR (triggered test)")
    ax[1].set_ylim(0, 1)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

def plot_roc(out_path: str, roc_dicts: list[dict], title: str = "ROC (defenses)") -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 5))
    for rd in roc_dicts:
        name = rd.get("name", "defense")
        fpr = rd.get("fpr", [])
        tpr = rd.get("tpr", [])
        auc = rd.get("auc", None)
        if fpr and tpr:
            ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})" if auc is not None else name)
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set_title(title)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

def plot_reject_vs_drop(out_path: str, reject_rates: list[float], acc_drops: list[float]) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(reject_rates, acc_drops, marker="o")
    ax.set_title("Reject rate vs Accuracy drop")
    ax.set_xlabel("Reject rate")
    ax.set_ylabel("Accuracy drop")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)