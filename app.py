import os
from dataclasses import asdict  # <-- FIX: add asdict
from typing import Optional, List, Dict, Any

import gradio as gr
import numpy as np

from poison_tester_ui.data.npz_loader import load_npz_splits
from poison_tester_ui.data.preprocessing import PreprocessConfig, build_preprocess
from poison_tester_ui.data.preview import sample_preview_items

from poison_tester_ui.models.keras_art_factory import build_art_keras_classifier, KerasArtModelInfo

from poison_tester_ui.attacks.art_hidden_trigger import HiddenTriggerParams, run_hidden_trigger_backdoor_art
from poison_tester_ui.attacks.art_clean_label_backdoor import CleanLabelBackdoorParams, run_clean_label_backdoor_art
from poison_tester_ui.defenses.art_spectral_signature import SpectralParams, run_spectral_signature_defense
from poison_tester_ui.defenses.art_activation_defence import ActivationParams, run_activation_defence

from poison_tester_ui.metrics.classification import (
    accuracy_from_probs_or_logits,
    asr_from_probs_or_logits,
    roc_auc_from_scores,
)
from poison_tester_ui.reporting.report_writer import RunReport, write_run_report
from poison_tester_ui.plotting.plots import plot_metrics_bar, plot_roc, plot_reject_vs_drop

from poison_tester_ui.utils.env_info import collect_env_info
from poison_tester_ui.utils.seeding import seed_everything
from poison_tester_ui.utils.time_id import make_run_id
from poison_tester_ui.utils.io import ensure_dir, save_json, zip_dir


class SessionState:
    def __init__(self):
        self.npz_path: Optional[str] = None
        self.splits = None
        self.preprocess_cfg: Optional[PreprocessConfig] = None
        self.preprocess = None

        self.h5_path: Optional[str] = None
        self.art_classifier = None
        self.keras_info: Optional[KerasArtModelInfo] = None

        self.last_run_dir: Optional[str] = None


STATE = SessionState()


def _npz_summary_text(splits) -> str:
    return (
        f"- schema_used: `{splits.schema_used}`\n"
        f"- x_train: {tuple(splits.x_train.shape)} dtype={splits.x_train.dtype}\n"
        f"- y_train: {tuple(splits.y_train.shape)} dtype={splits.y_train.dtype}\n"
        f"- x_val: {tuple(splits.x_val.shape)} dtype={splits.x_val.dtype}\n"
        f"- y_val: {tuple(splits.y_val.shape)} dtype={splits.y_val.dtype}\n"
        f"- x_test: {tuple(splits.x_test.shape)} dtype={splits.x_test.dtype}\n"
        f"- y_test: {tuple(splits.y_test.shape)} dtype={splits.y_test.dtype}\n"
        f"- num_classes (inferred): {splits.num_classes}\n"
        f"- format: {splits.format}\n"
    )


def ui_load_npz(npz_file, normalize_mode: str, estimate_n: int, val_ratio: float, split_seed: int):
    if npz_file is None:
        return gr.Markdown("**錯誤**：請上傳 .npz 檔"), None, None

    npz_path = npz_file.name
    STATE.npz_path = npz_path

    splits = load_npz_splits(npz_path, val_ratio=float(val_ratio), seed=int(split_seed))
    STATE.splits = splits

    preprocess_cfg = PreprocessConfig(
        resize_hw=(224, 224),
        normalize_mode=normalize_mode,
        estimate_n=int(estimate_n),
    )
    preprocess = build_preprocess(splits, preprocess_cfg)
    STATE.preprocess_cfg = preprocess_cfg
    STATE.preprocess = preprocess

    summary = _npz_summary_text(splits)
    label_choices = ["All"] + [str(i) for i in range(splits.num_classes)]
    gallery, _ = sample_preview_items(splits, split="train", label_filter=None, k=16)
    return gr.Markdown("**NPZ 載入成功**\n\n" + summary), gr.Dropdown(choices=label_choices, value="All"), gallery


def ui_refresh_preview(preview_split: str, filter_label: str):
    splits = STATE.splits
    if splits is None:
        return []
    label_filter = None if filter_label == "All" else int(filter_label)
    gallery, _ = sample_preview_items(splits, split=preview_split, label_filter=label_filter, k=16)
    return gallery


def ui_load_h5(h5_file, use_logits: bool):
    if h5_file is None:
        return gr.Markdown("**錯誤**：請上傳 .h5"), None, None
    if STATE.splits is None:
        return gr.Markdown("**錯誤**：請先載入 NPZ"), None, None

    h5_path = h5_file.name
    STATE.h5_path = h5_path

    try:
        classifier, info = build_art_keras_classifier(
            h5_path=h5_path,
            clip_values=(0.0, 1.0),
            channels_first=False,
            use_logits=bool(use_logits),
            preprocessing=(0.0, 1.0),
        )
    except Exception as e:
        return gr.Markdown(f"**載入/建立 ART KerasClassifier 失敗**：{type(e).__name__}: {e}"), None, None

    if info.nb_classes != STATE.splits.num_classes:
        md = (
            f"**載入成功，但類別數不一致**\n\n"
            f"- model nb_classes: `{info.nb_classes}`\n"
            f"- dataset num_classes: `{STATE.splits.num_classes}`\n"
            f"\n請確認你的 .h5 與 NPZ label 定義一致。"
        )
    else:
        md = (
            f"**Keras .h5 + ART KerasClassifier 建立成功**\n\n"
            f"- path: `{h5_path}`\n"
            f"- input_shape: `{info.input_shape}`\n"
            f"- output_shape: `{info.output_shape}`\n"
            f"- nb_classes: `{info.nb_classes}`\n"
            f"- layers: `{len(info.layer_names)}`\n"
        )

    STATE.art_classifier = classifier
    STATE.keras_info = info

    layer_choices = info.layer_names if info.layer_names else []
    default_feature = layer_choices[-2] if len(layer_choices) >= 2 else (layer_choices[-1] if layer_choices else "")
    return gr.Markdown(md), gr.Dropdown(choices=layer_choices, value=default_feature), gr.Markdown(f"預設 feature_layer: `{default_feature}`")


def _subsample_train(x: np.ndarray, y: np.ndarray, mode: str, cap: str, seed: int):
    if mode != "Quick":
        return x, y
    if cap == "all":
        return x, y
    if cap.endswith("k"):
        ncap = int(cap[:-1]) * 1000
    else:
        ncap = int(cap)
    ncap = min(ncap, x.shape[0])
    rng = np.random.RandomState(seed)
    idx = rng.choice(x.shape[0], size=ncap, replace=False)
    return x[idx], y[idx]


def _eval_art_classifier(classifier, x_nhwc: np.ndarray, y_int: np.ndarray, batch_size: int) -> float:
    pred = classifier.predict(x_nhwc, batch_size=batch_size)
    return accuracy_from_probs_or_logits(pred, y_int)


def _eval_asr(classifier, x_triggered_nhwc: np.ndarray, target_label: int, batch_size: int) -> float:
    pred = classifier.predict(x_triggered_nhwc, batch_size=batch_size)
    return asr_from_probs_or_logits(pred, target_label)


def _apply_trigger_to_dataset_nhwc(x_nhwc: np.ndarray, trigger_image: np.ndarray, trigger_mask: np.ndarray, location: str, alpha: float):
    from poison_tester_ui.attacks.simple_image_mask_backdoor import SimpleImageMaskBackdoor
    bd = SimpleImageMaskBackdoor(trigger_image, trigger_mask, location, alpha)
    x_out, _ = bd.poison(x_nhwc, y=None, broadcast=True)
    return x_out


def ui_run_full_tf(
    train_mode: str,
    max_train_samples: str,
    epochs: int,
    batch_size: int,
    seed: int,
    feature_layer: str,
    use_logits: bool,
    attack_name: str,
    target_label: int,
    source_label: int,
    poison_fraction: float,
    trigger_image,
    trigger_mask,
    trigger_location: str,
    trigger_alpha: float,
    cl_eps: float,
    cl_eps_step: float,
    cl_max_iter: int,
    do_spectral: bool,
    do_activation: bool,
    reject_rate_target: float,
):
    if STATE.splits is None or STATE.preprocess is None:
        return "請先載入 NPZ", None, None, None, None, None
    if STATE.h5_path is None:
        return "請先上傳 .h5", None, None, None, None, None
    if STATE.art_classifier is None or STATE.keras_info is None:
        return "請先建立 ART KerasClassifier（Model tab Validate）", None, None, None, None, None
    if not feature_layer:
        return "請選擇 feature_layer", None, None, None, None, None
    if trigger_image is None or trigger_mask is None:
        return "請上傳 trigger image + mask", None, None, None, None, None
    if attack_name == "Feature collision (TF disabled)":
        return "Feature collision：TF 在 ART 1.20.1 不支援（僅 PyTorchClassifier）。", None, None, None, None, None

    seed_everything(int(seed))
    np.random.seed(int(seed))

    splits = STATE.splits
    preprocess = STATE.preprocess

    run_id = make_run_id()
    run_dir = os.path.join(".", "run", run_id)
    ensure_dir(run_dir)
    ensure_dir(os.path.join(run_dir, "plots"))
    ensure_dir(os.path.join(run_dir, "artifacts"))
    ensure_dir(os.path.join(run_dir, "artifacts", "attack"))
    ensure_dir(os.path.join(run_dir, "artifacts", "defense"))

    env = collect_env_info()
    save_json(os.path.join(run_dir, "env.json"), env)

    # NHWC float32 normalized (for Keras/ART)
    x_train = preprocess.to_numpy_nhwc(splits.x_train, normalize=True)
    x_val = preprocess.to_numpy_nhwc(splits.x_val, normalize=True)
    x_test = preprocess.to_numpy_nhwc(splits.x_test, normalize=True)

    y_train = splits.y_train.astype(np.int64)
    y_val = splits.y_val.astype(np.int64)
    y_test = splits.y_test.astype(np.int64)

    x_train_fit, y_train_fit = _subsample_train(x_train, y_train, train_mode, max_train_samples, int(seed))

    clf = STATE.art_classifier

    # baseline
    base_acc_test = _eval_art_classifier(clf, x_test, y_test, batch_size=int(batch_size))
    x_test_triggered = _apply_trigger_to_dataset_nhwc(x_test, trigger_image, trigger_mask, trigger_location, float(trigger_alpha))
    base_asr = _eval_asr(clf, x_test_triggered, int(target_label), batch_size=int(batch_size))

    log_lines: List[str] = []
    def log(s: str):
        log_lines.append(s)

    log("== TF Baseline (ART KerasClassifier) ==")
    log(f"baseline test acc={base_acc_test:.4f}, baseline ASR={base_asr:.4f}")

    poisoned_indices: List[int] = []
    attack_meta: Dict[str, Any] = {}

    if attack_name == "Hidden Trigger Backdoor (ART)":
        params = HiddenTriggerParams(
            max_iter=800 if train_mode == "Quick" else 2000,
            batch_size=int(batch_size),
            poison_percent=float(poison_fraction),
            location=trigger_location,
            alpha=float(trigger_alpha),
        )
        x_train_poisoned, y_train_poisoned, poisoned_indices, attack_meta = run_hidden_trigger_backdoor_art(
            classifier=clf,
            x_train_nhwc=x_train_fit,
            y_train_int=y_train_fit,
            target_label=int(target_label),
            source_label=int(source_label),
            feature_layer=feature_layer,
            trigger_image=trigger_image,
            trigger_mask=trigger_mask,
            params=params,
            seed=int(seed),
        )

    elif attack_name == "Clean-label Backdoor (ART)":
        params = CleanLabelBackdoorParams(
            pp_poison=float(poison_fraction),
            eps=float(cl_eps),
            eps_step=float(cl_eps_step),
            max_iter=int(cl_max_iter),
            num_random_init=0,
            location=trigger_location,
            alpha=float(trigger_alpha),
        )
        x_train_poisoned, y_train_poisoned, poisoned_indices, attack_meta = run_clean_label_backdoor_art(
            proxy_classifier=clf,
            x_train_nhwc=x_train_fit,
            y_train_int=y_train_fit,
            target_label=int(target_label),
            trigger_image=trigger_image,
            trigger_mask=trigger_mask,
            params=params,
            seed=int(seed),
        )
    else:
        return "未知 attack", None, None, None, None, None

    save_json(os.path.join(run_dir, "artifacts", "attack", "poisoned_indices.json"), {"poisoned_indices": poisoned_indices})
    save_json(os.path.join(run_dir, "artifacts", "attack", "attack_meta.json"), attack_meta)

    is_clean_gt = np.ones((x_train_fit.shape[0],), dtype=np.int64)
    is_clean_gt[np.array(poisoned_indices, dtype=np.int64)] = 0
    is_poison_gt = 1 - is_clean_gt

    log(f"== Attack done: {attack_name} ==")
    log(f"poisoned_count={len(poisoned_indices)}")

    # refit attacked model
    attacked_clf = clf.clone_for_refitting()
    attacked_clf.fit(
        x_train_poisoned,
        y_train_poisoned,  # FIX: pass int labels; ART will transform if needed
        batch_size=int(batch_size),
        nb_epochs=int(epochs),
        verbose=True,
    )

    attacked_acc_test = _eval_art_classifier(attacked_clf, x_test, y_test, batch_size=int(batch_size))
    attacked_asr = _eval_asr(attacked_clf, x_test_triggered, int(target_label), batch_size=int(batch_size))
    log(f"attacked test acc={attacked_acc_test:.4f}, attacked ASR={attacked_asr:.4f}")

    defense_results = []
    roc_list = []
    reject_rates = []

    filtered_idx_keep = np.arange(x_train_poisoned.shape[0])

    if do_spectral:
        sp = SpectralParams(expected_pp_poison=float(poison_fraction), batch_size=int(batch_size), eps_multiplier=1.5)
        out = run_spectral_signature_defense(
            classifier=attacked_clf,
            x_train=x_train_poisoned,
            y_train_int=y_train_poisoned,
            is_clean_gt=is_clean_gt,
            params=sp,
        )
        scores = np.array(out["scores"], dtype=np.float32)
        roc = roc_auc_from_scores(is_poison_gt=is_poison_gt, scores=scores)
        roc["name"] = "SpectralSignatureDefense"
        roc_list.append(roc)

        defense_results.append({
            "name": out["name"],
            "flagged_count": len(out["flagged_indices"]),
            "auc": roc.get("auc"),
            "confusion_matrix_json": out["confusion_matrix_json"],
        })

        n_drop = int(round(float(reject_rate_target) * x_train_poisoned.shape[0]))
        drop_idx = np.argsort(-scores)[:n_drop] if n_drop > 0 else np.array([], dtype=int)
        keep_mask = np.ones((x_train_poisoned.shape[0],), dtype=bool)
        keep_mask[drop_idx] = False
        filtered_idx_keep = filtered_idx_keep[keep_mask]
        reject_rates.append(float(n_drop / max(1, x_train_poisoned.shape[0])))

    if do_activation:
        ap = ActivationParams(nb_clusters=2, clustering_method="KMeans", nb_dims=10, reduce="PCA", cluster_analysis="smaller", ex_re_threshold=None)
        out = run_activation_defence(
            classifier=attacked_clf,
            x_train=x_train_poisoned,
            y_train_int=y_train_poisoned,
            is_clean_gt=is_clean_gt,
            params=ap,
        )
        defense_results.append({
            "name": out["name"],
            "flagged_count": len(out["flagged_indices"]),
            "auc": None,
            "confusion_matrix_json": out["confusion_matrix_json"],
        })

        is_clean_lst = np.array(out["is_clean_lst"], dtype=np.int64)
        flagged = np.where(is_clean_lst == 0)[0]
        n_drop = int(round(float(reject_rate_target) * x_train_poisoned.shape[0]))
        if n_drop > 0 and flagged.size > 0:
            drop_idx = flagged[: min(n_drop, flagged.size)]
            keep_mask = np.ones((x_train_poisoned.shape[0],), dtype=bool)
            keep_mask[drop_idx] = False
            filtered_idx_keep = np.intersect1d(filtered_idx_keep, np.where(keep_mask)[0])
        reject_rates.append(float(n_drop / max(1, x_train_poisoned.shape[0])))

    x_train_def = x_train_poisoned[filtered_idx_keep]
    y_train_def = y_train_poisoned[filtered_idx_keep]

    defended_clf = attacked_clf.clone_for_refitting()
    defended_clf.fit(
        x_train_def,
        y_train_def,  # FIX: int labels
        batch_size=int(batch_size),
        nb_epochs=int(epochs),
        verbose=True,
    )

    defended_acc_test = _eval_art_classifier(defended_clf, x_test, y_test, batch_size=int(batch_size))
    defended_asr = _eval_asr(defended_clf, x_test_triggered, int(target_label), batch_size=int(batch_size))
    log(f"defended test acc={defended_acc_test:.4f}, defended ASR={defended_asr:.4f}")

    acc_drop = float(attacked_acc_test - defended_acc_test)
    rr = float(max(reject_rates)) if reject_rates else 0.0

    save_json(os.path.join(run_dir, "artifacts", "defense", "defense_results.json"), {"defenses": defense_results})
    save_json(os.path.join(run_dir, "artifacts", "defense", "kept_indices.json"), {"kept_indices": filtered_idx_keep.tolist()})

    plot_metrics_bar(
        out_path=os.path.join(run_dir, "plots", "metrics_bar.png"),
        baseline_acc=base_acc_test,
        attacked_acc=attacked_acc_test,
        defended_acc=defended_acc_test,
        baseline_asr=base_asr,
        attacked_asr=attacked_asr,
        defended_asr=defended_asr,
    )
    plot_roc(os.path.join(run_dir, "plots", "roc.png"), roc_list, title="ROC (Spectral)")
    plot_reject_vs_drop(os.path.join(run_dir, "plots", "reject_curve.png"), [rr], [acc_drop])

    report = RunReport(
        meta={"run_id": run_id, "output_dir": run_dir},
        environment=env,
        seed=int(seed),
        data=splits.to_summary_dict(),
        preprocess=asdict(STATE.preprocess_cfg) if STATE.preprocess_cfg else {},
        model={"h5_path": STATE.h5_path, "nb_classes": STATE.keras_info.nb_classes, "feature_layer": feature_layer},
        results={
            "baseline": {"acc_test": base_acc_test, "asr": base_asr},
            "attack": {
                "name": attack_name,
                "poisoned_count": len(poisoned_indices),
                "poison_fraction": float(poison_fraction),
                "attacked_acc_test": attacked_acc_test,
                "attacked_asr": attacked_asr,
                "meta": attack_meta,
            },
            "defense": defense_results,
            "post_defense": {
                "acc_test": defended_acc_test,
                "asr": defended_asr,
                "reject_rate": rr,
                "accuracy_drop": acc_drop,
            },
        },
        plots={
            "metrics_bar": os.path.join("plots", "metrics_bar.png"),
            "roc": os.path.join("plots", "roc.png"),
            "reject_curve": os.path.join("plots", "reject_curve.png"),
        },
    )
    write_run_report(report, run_dir)

    zip_path = zip_dir(run_dir)
    STATE.last_run_dir = run_dir

    metrics_md = (
        f"### Summary (TF)\n"
        f"- baseline test acc: **{base_acc_test:.4f}**, ASR: **{base_asr:.4f}**\n"
        f"- attacked test acc: **{attacked_acc_test:.4f}**, ASR: **{attacked_asr:.4f}**\n"
        f"- defended test acc: **{defended_acc_test:.4f}**, ASR: **{defended_asr:.4f}**\n"
        f"- reject_rate: **{rr:.4f}**, accuracy_drop: **{acc_drop:.4f}**\n"
        f"- output: `{run_dir}`\n"
    )

    return "\n".join(log_lines), metrics_md, os.path.join(run_dir, "plots", "metrics_bar.png"), os.path.join(run_dir, "plots", "roc.png"), os.path.join(run_dir, "plots", "reject_curve.png"), zip_path


def main():
    with gr.Blocks(title="Poison Tester UI (TF ART 1.20.1)") as demo:
        gr.Markdown(
            "# Poison Tester UI (TF/Keras + ART 1.20.1)\n"
            "- Hidden Trigger + Clean-label Backdoor + Defenses\n"
            "- Feature collision TF：禁用（ART 1.20.1 僅支援 PyTorchClassifier）"
        )

        with gr.Tab("Data (NPZ)"):
            npz_file = gr.File(label="Upload NPZ")
            normalize_mode = gr.Radio(choices=["imagenet_default", "estimate_from_train"], value="imagenet_default", label="Normalize mode")
            estimate_n = gr.Slider(128, 4096, value=1024, step=128, label="Estimate N (if enabled)")
            val_ratio = gr.Slider(0.01, 0.5, value=0.1, step=0.01, label="val_ratio (if NPZ has no val split)")
            split_seed = gr.Number(value=1234, precision=0, label="split seed (for auto val split)")
            load_npz_btn = gr.Button("Load NPZ")
            data_status = gr.Markdown()

            preview_split = gr.Radio(choices=["train", "val", "test"], value="train", label="Preview split")
            filter_label = gr.Dropdown(choices=["All"], value="All", label="Filter label")
            preview_btn = gr.Button("Refresh preview")
            preview_gallery = gr.Gallery(label="Preview (sample 16)", columns=4, rows=4, height=360)

            load_npz_btn.click(fn=ui_load_npz, inputs=[npz_file, normalize_mode, estimate_n, val_ratio, split_seed], outputs=[data_status, filter_label, preview_gallery])
            preview_btn.click(fn=ui_refresh_preview, inputs=[preview_split, filter_label], outputs=[preview_gallery])

        with gr.Tab("Model (Keras .h5)"):
            h5_file = gr.File(label="Upload Keras .h5 (compiled)")
            use_logits = gr.Checkbox(value=True, label="use_logits (recommended True)")
            validate_h5 = gr.Button("Validate .h5 & Build ART KerasClassifier")
            h5_status = gr.Markdown()
            feature_layer = gr.Dropdown(choices=[], value="", label="feature_layer (required for Hidden Trigger)")
            feature_hint = gr.Markdown()
            validate_h5.click(fn=ui_load_h5, inputs=[h5_file, use_logits], outputs=[h5_status, feature_layer, feature_hint])

        with gr.Tab("Training"):
            train_mode = gr.Radio(choices=["Quick", "Normal"], value="Quick", label="Mode")
            max_train_samples = gr.Dropdown(choices=["5k", "10k", "20k", "all"], value="10k", label="Max train samples (Quick)")
            epochs = gr.Slider(1, 20, value=2, step=1, label="Epochs (refit)")
            batch_size = gr.Dropdown(choices=[8, 16, 32, 64], value=32, label="Batch size")
            seed = gr.Number(value=1234, precision=0, label="Seed")

        with gr.Tab("Attack"):
            attack_name = gr.Radio(
                choices=["Hidden Trigger Backdoor (ART)", "Clean-label Backdoor (ART)", "Feature collision (TF disabled)"],
                value="Hidden Trigger Backdoor (ART)",
                label="Attack",
            )
            target_label = gr.Number(value=0, precision=0, label="Target label (int)")
            source_label = gr.Number(value=1, precision=0, label="Source label (int, for Hidden Trigger)")
            poison_fraction = gr.Slider(0.01, 0.5, value=0.1, step=0.01, label="poison fraction")

            trigger_image = gr.Image(type="numpy", label="Trigger image (upload)")
            trigger_mask = gr.Image(type="numpy", label="Trigger mask (upload, white=apply)")
            trigger_location = gr.Dropdown(choices=["bottom-right", "bottom-left", "top-right", "top-left", "random"], value="bottom-right", label="Trigger location")
            trigger_alpha = gr.Slider(0.0, 1.0, value=1.0, step=0.05, label="Alpha")

            gr.Markdown("### Clean-label Backdoor params (ART PGD)")
            cl_eps = gr.Number(value=0.3, label="eps")
            cl_eps_step = gr.Number(value=0.1, label="eps_step")
            cl_max_iter = gr.Slider(1, 200, value=20, step=1, label="max_iter")

        with gr.Tab("Defense"):
            do_spectral = gr.Checkbox(value=True, label="Spectral Signature Defense (ART)")
            do_activation = gr.Checkbox(value=True, label="Activation Defence (ART)")
            reject_rate_target = gr.Slider(0.0, 0.5, value=0.1, step=0.01, label="Reject/Drop rate target")

        with gr.Tab("Run & Report"):
            run_full_btn = gr.Button("Run FULL TF pipeline (baseline -> attack -> refit -> defense -> refit -> eval)")
            run_log = gr.Textbox(label="Log", lines=18)
            metrics_md = gr.Markdown()
            plot_a = gr.Image(label="Plot (a) metrics bar")
            plot_b = gr.Image(label="Plot (b) ROC")
            plot_c = gr.Image(label="Plot (c) reject curve")
            download_zip = gr.File(label="Download run folder (.zip)")

            run_full_btn.click(
                fn=ui_run_full_tf,
                inputs=[
                    train_mode, max_train_samples, epochs, batch_size, seed,
                    feature_layer, use_logits,
                    attack_name, target_label, source_label, poison_fraction,
                    trigger_image, trigger_mask, trigger_location, trigger_alpha,
                    cl_eps, cl_eps_step, cl_max_iter,
                    do_spectral, do_activation, reject_rate_target,
                ],
                outputs=[run_log, metrics_md, plot_a, plot_b, plot_c, download_zip],
            )

    demo.launch()


if __name__ == "__main__":
    main()