"""
DINOv3 Thalassaemia — Binary Linear Probe (Positives vs Negatives)
==================================================================
Trains a *binary* linear probe on top of a frozen DINOv3 backbone, using
ONLY the `Positives` and `Negatives` classes — the `Other` class is dropped.

What it does
------------
1.  Loads the teacher backbone (DCP `ckpt/NNN` dir or legacy `.pth`) — reuses
    the loaders from `evaluate_checkpoint.py`.
2.  Extracts [CLS] features for train / val / test.
3.  Keeps only Negatives (label 0) and Positives (label 2), remapped to a
    binary target  {Negatives: 0, Positives: 1}.
4.  Fits a logistic-regression linear probe on standardised features.
    The regularisation strength C is selected on the VAL split by AUC.
5.  Reports a full clinical metrics panel **with confidence intervals**:
       • AUC-ROC               — 95% CI via stratified bootstrap
       • Accuracy / Sens / Spec / PPV / NPV / F1  — 95% Wilson CIs
       • At threshold 0.5 AND the Youden-J optimal threshold
       • Brier score
    Plus per-sample predicted probabilities (CSV).
6.  Saves light-theme figures: ROC curve, confusion matrix, t-SNE.

Usage (Colab, from repo root)
-----------------------------
  !python scripts/run_binary_probe.py \\
      --ckpt-path   outputs/thalassaemia_scratch/ckpt/999 \\
      --data-root   data/preprocessed \\
      --output-dir  outputs/binary_probe_999 \\
      --run-tsne

Outputs (in --output-dir)
-------------------------
  binary_probe_results.json   ← all metrics + CIs
  probabilities_test.csv      ← per-sample y_true, p_positive, y_pred
  probabilities_val.csv
  roc_curve_test.png          ← light theme
  confusion_matrix_test.png   ← light theme
  tsne_binary_all.png         ← light theme (train+val+test)
  tsne_binary_test.png        ← light theme (test only)
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
from pathlib import Path

import numpy as np
import torch

# ── reuse the heavy lifting from evaluate_checkpoint.py ─────────────────────
# (importing it also installs the torch.load weights_only patch and puts the
#  repo root on sys.path)
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from evaluate_checkpoint import (  # noqa: E402
    load_backbone,
    make_dataloaders,
    extract_features,
)

logger = logging.getLogger("binary_probe")

# Original 3-class indices (see dinov3/data/datasets/thalassaemia.py):
#   0 = Negatives, 1 = Other, 2 = Positives
NEG_IDX = 0
POS_IDX = 2
BINARY_CLASS_NAMES = ["Negatives", "Positives"]   # index 0 / 1 after remap


# ─────────────────────────────────────────────────────────────────────────────
# Binary filtering
# ─────────────────────────────────────────────────────────────────────────────
def to_binary(feats: torch.Tensor, labels: torch.Tensor):
    """Keep only Negatives(0) and Positives(2); remap to {0,1}. Drop 'Other'."""
    labels = labels.long()
    keep = (labels == NEG_IDX) | (labels == POS_IDX)
    f = feats[keep].numpy().astype(np.float64)
    y = (labels[keep] == POS_IDX).long().numpy()   # Positives -> 1, Negatives -> 0
    n_drop = int((~keep).sum())
    return f, y, n_drop


# ─────────────────────────────────────────────────────────────────────────────
# Confidence-interval helpers
# ─────────────────────────────────────────────────────────────────────────────
def wilson_ci(k: int, n: int, z: float = 1.959963985):
    """Wilson score interval for a binomial proportion k/n."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def bootstrap_auc_ci(y_true: np.ndarray, scores: np.ndarray,
                     n_boot: int = 2000, seed: int = 42, alpha: float = 0.05):
    """Stratified bootstrap 95% CI for ROC-AUC."""
    from sklearn.metrics import roc_auc_score

    rng = np.random.RandomState(seed)
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    boots = []
    for _ in range(n_boot):
        bp = rng.choice(pos_idx, size=len(pos_idx), replace=True)
        bn = rng.choice(neg_idx, size=len(neg_idx), replace=True)
        idx = np.concatenate([bp, bn])
        yt = y_true[idx]
        sc = scores[idx]
        if len(np.unique(yt)) < 2:
            continue
        boots.append(roc_auc_score(yt, sc))
    if not boots:
        return (float("nan"), float("nan"))
    lo = float(np.percentile(boots, 100 * alpha / 2))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return (lo, hi)


def threshold_metrics(y_true: np.ndarray, prob: np.ndarray, thr: float):
    """Compute the full confusion-derived panel at a given probability threshold."""
    pred = (prob >= thr).astype(int)
    tp = int(np.sum((pred == 1) & (y_true == 1)))
    fp = int(np.sum((pred == 1) & (y_true == 0)))
    tn = int(np.sum((pred == 0) & (y_true == 0)))
    fn = int(np.sum((pred == 0) & (y_true == 1)))
    n = tp + fp + tn + fn

    def safe(num, den):
        return (num / den) if den else float("nan")

    sens = safe(tp, tp + fn)          # recall / TPR
    spec = safe(tn, tn + fp)          # TNR
    ppv = safe(tp, tp + fp)           # precision
    npv = safe(tn, tn + fn)
    acc = safe(tp + tn, n)
    f1 = safe(2 * tp, 2 * tp + fp + fn)

    return {
        "threshold": float(thr),
        "confusion_matrix": {"TP": tp, "FP": fp, "TN": tn, "FN": fn},
        "accuracy":    {"value": acc,  "ci95": wilson_ci(tp + tn, n)},
        "sensitivity": {"value": sens, "ci95": wilson_ci(tp, tp + fn)},
        "specificity": {"value": spec, "ci95": wilson_ci(tn, tn + fp)},
        "ppv":         {"value": ppv,  "ci95": wilson_ci(tp, tp + fp)},
        "npv":         {"value": npv,  "ci95": wilson_ci(tn, tn + fn)},
        "f1":          {"value": f1,   "ci95": (float("nan"), float("nan"))},
    }


# ─────────────────────────────────────────────────────────────────────────────
# Linear probe (logistic regression) with C selection on val
# ─────────────────────────────────────────────────────────────────────────────
def fit_probe(Xtr, ytr, Xva, yva, C_grid, class_weight, max_iter):
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    scaler = StandardScaler().fit(Xtr)
    Xtr_s, Xva_s = scaler.transform(Xtr), scaler.transform(Xva)

    best = None
    logger.info("Selecting C on the validation split (by AUC) …")
    for C in C_grid:
        clf = LogisticRegression(
            C=C, max_iter=max_iter, class_weight=class_weight, solver="lbfgs",
        )
        clf.fit(Xtr_s, ytr)
        val_auc = roc_auc_score(yva, clf.predict_proba(Xva_s)[:, 1])
        logger.info(f"  C={C:<8g}  val_AUC={val_auc:.4f}")
        if best is None or val_auc > best[0]:
            best = (val_auc, C, clf)
    best_auc, best_C, best_clf = best
    logger.info(f"  → best C={best_C}  (val AUC={best_auc:.4f})")
    return scaler, best_clf, best_C, best_auc


def evaluate_split(clf, scaler, X, y, split_name, n_boot, seed):
    from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss

    prob = clf.predict_proba(scaler.transform(X))[:, 1]
    auc = float(roc_auc_score(y, prob))
    auc_ci = bootstrap_auc_ci(y, prob, n_boot=n_boot, seed=seed)
    brier = float(brier_score_loss(y, prob))

    # Youden-J optimal threshold
    fpr, tpr, thr = roc_curve(y, prob)
    youden = tpr - fpr
    j_idx = int(np.argmax(youden))
    j_thr = float(thr[j_idx]) if np.isfinite(thr[j_idx]) else 0.5

    results = {
        "n": int(len(y)),
        "n_positive": int(np.sum(y == 1)),
        "n_negative": int(np.sum(y == 0)),
        "auc_roc": {"value": auc, "ci95": auc_ci, "ci_method": f"stratified bootstrap ({n_boot})"},
        "brier_score": brier,
        "at_threshold_0.5":     threshold_metrics(y, prob, 0.5),
        "at_threshold_youden":  threshold_metrics(y, prob, j_thr),
        "youden_threshold": j_thr,
        "roc_curve": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
    }
    _log_split(split_name, results)
    return results, prob


def _fmt(m):
    lo, hi = m["ci95"]
    return f"{m['value']:.4f}  [{lo:.4f}, {hi:.4f}]"


def _log_split(name, r):
    logger.info("\n" + "=" * 60)
    logger.info(f"  {name.upper()}  —  n={r['n']}  (pos={r['n_positive']}, neg={r['n_negative']})")
    logger.info("=" * 60)
    a = r["auc_roc"]
    logger.info(f"  AUC-ROC      : {a['value']:.4f}  95% CI [{a['ci95'][0]:.4f}, {a['ci95'][1]:.4f}]")
    logger.info(f"  Brier score  : {r['brier_score']:.4f}")
    for tkey in ("at_threshold_0.5", "at_threshold_youden"):
        t = r[tkey]
        logger.info(f"\n  ── {tkey}  (thr={t['threshold']:.4f}) ──")
        cm = t["confusion_matrix"]
        logger.info(f"     TP={cm['TP']}  FP={cm['FP']}  TN={cm['TN']}  FN={cm['FN']}")
        logger.info(f"     Accuracy    : {_fmt(t['accuracy'])}")
        logger.info(f"     Sensitivity : {_fmt(t['sensitivity'])}")
        logger.info(f"     Specificity : {_fmt(t['specificity'])}")
        logger.info(f"     PPV         : {_fmt(t['ppv'])}")
        logger.info(f"     NPV         : {_fmt(t['npv'])}")
        logger.info(f"     F1          : {t['f1']['value']:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# Light-theme visualisations
# ─────────────────────────────────────────────────────────────────────────────
LIGHT_NEG = "#1f77b4"   # Negatives — blue
LIGHT_POS = "#d62728"   # Positives — red


def _light_ax(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=13, fontweight="bold", color="#222222", pad=12)
    ax.set_xlabel(xlabel, color="#333333", fontsize=11)
    ax.set_ylabel(ylabel, color="#333333", fontsize=11)
    ax.tick_params(colors="#444444")
    for spine in ax.spines.values():
        spine.set_edgecolor("#cccccc")
    ax.grid(True, color="#e8e8e8", linewidth=0.8)
    ax.set_axisbelow(True)


def plot_roc(fpr, tpr, auc, auc_ci, save_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.plot(fpr, tpr, color=LIGHT_POS, linewidth=2.2,
            label=f"AUC = {auc:.3f}  [{auc_ci[0]:.3f}, {auc_ci[1]:.3f}]")
    ax.plot([0, 1], [0, 1], color="#999999", linestyle="--", linewidth=1.2)
    _light_ax(ax, "ROC — Positives vs Negatives (Test)",
              "False Positive Rate (1 − Specificity)", "True Positive Rate (Sensitivity)")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="lower right", fontsize=11, framealpha=0.95,
              facecolor="white", edgecolor="#cccccc", labelcolor="#222222")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, facecolor="white")
    plt.close()
    logger.info(f"  Saved ROC curve → {save_path}")


def plot_confusion(cm_dict, save_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = np.array([[cm_dict["TN"], cm_dict["FP"]],
                   [cm_dict["FN"], cm_dict["TP"]]])
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    fig.patch.set_facecolor("white")
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(BINARY_CLASS_NAMES)
    ax.set_yticklabels(BINARY_CLASS_NAMES)
    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "#222222",
                    fontsize=14, fontweight="bold")
    ax.set_title("Confusion Matrix (Test)", fontsize=13, fontweight="bold", color="#222222", pad=12)
    ax.set_xlabel("Predicted", color="#333333", fontsize=11)
    ax.set_ylabel("True", color="#333333", fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, facecolor="white")
    plt.close()
    logger.info(f"  Saved confusion matrix → {save_path}")


def plot_tsne_light(feats: np.ndarray, y: np.ndarray, title: str, save_path: Path,
                    perplexity: int = 30, max_samples: int = 3000, seed: int = 42):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    if len(feats) > max_samples:
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(feats), max_samples, replace=False)
        feats, y = feats[idx], y[idx]
        logger.info(f"  t-SNE: sub-sampled to {max_samples} points")

    norms = np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8
    feats = feats / norms

    logger.info(f"  Running t-SNE on {len(feats)} samples …")
    emb = TSNE(
        n_components=2,
        perplexity=min(perplexity, max(5, len(feats) // 4)),
        init="pca",
        random_state=seed,
        verbose=0,
    ).fit_transform(feats)

    fig, ax = plt.subplots(figsize=(9, 7.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    for lbl, name, color in [(0, "Negatives", LIGHT_NEG), (1, "Positives", LIGHT_POS)]:
        mask = y == lbl
        ax.scatter(emb[mask, 0], emb[mask, 1], c=color, label=name,
                   s=20, alpha=0.7, linewidths=0.3, edgecolors="white")
    _light_ax(ax, title, "t-SNE dim 1", "t-SNE dim 2")
    ax.legend(fontsize=11, markerscale=1.8, framealpha=0.95,
              facecolor="white", edgecolor="#cccccc", labelcolor="#222222")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, facecolor="white")
    plt.close()
    logger.info(f"  Saved t-SNE → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CSV export
# ─────────────────────────────────────────────────────────────────────────────
def save_probabilities(path: Path, y_true, prob, thr=0.5):
    pred = (prob >= thr).astype(int)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["index", "y_true", "true_name", "p_positive", "y_pred", "pred_name"])
        for i, (yt, p, yp) in enumerate(zip(y_true, prob, pred)):
            w.writerow([i, int(yt), BINARY_CLASS_NAMES[int(yt)],
                        f"{p:.6f}", int(yp), BINARY_CLASS_NAMES[int(yp)]])
    logger.info(f"  Saved per-sample probabilities → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Binary linear probe (Positives vs Negatives) for DINOv3 thalassaemia")
    p.add_argument("--ckpt-path",   required=True, help="DCP ckpt dir (ckpt/999) or legacy .pth")
    p.add_argument("--config-path", default=None, help="config.yaml — auto-detected if omitted")
    p.add_argument("--data-root",   default="data/preprocessed")
    p.add_argument("--output-dir",  default="outputs/binary_probe")

    p.add_argument("--run-tsne",    action="store_true", help="Generate light-theme t-SNE plots")
    p.add_argument("--tsne-max-samples", type=int, default=3000)

    p.add_argument("--class-weight", choices=["none", "balanced"], default="none",
                   help="LogisticRegression class_weight")
    p.add_argument("--max-iter",    type=int, default=2000)
    p.add_argument("--n-boot",      type=int, default=2000, help="Bootstrap iterations for AUC CI")
    p.add_argument("--seed",        type=int, default=42)

    p.add_argument("--batch-size",  type=int, default=64)
    p.add_argument("--num-workers", type=int, default=2)
    return p.parse_args()


def _autodetect_config(ckpt_path: str, explicit: str | None) -> str:
    if explicit:
        return explicit
    for c in [
        Path(ckpt_path).parent.parent / "config.yaml",
        Path(ckpt_path).parent.parent.parent / "config.yaml",
        Path(ckpt_path) / "config.yaml",
    ]:
        if c.exists():
            logger.info(f"Auto-detected config: {c}")
            return str(c)
    raise FileNotFoundError("Could not auto-detect config.yaml — pass --config-path.")


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    config_path = _autodetect_config(args.ckpt_path, args.config_path)

    # ── Backbone ────────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60 + "\n Loading backbone\n" + "=" * 60)
    backbone, embed_dim = load_backbone(args.ckpt_path, config_path, device)

    # ── Data + features ─────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60 + "\n Building datasets\n" + "=" * 60)
    loaders, _class_names, _num_classes = make_dataloaders(
        args.data_root, args.batch_size, args.num_workers
    )

    logger.info("\n" + "=" * 60 + "\n Extracting features\n" + "=" * 60)
    tr_f, tr_l = extract_features(backbone, loaders["train"], device, "train")
    va_f, va_l = extract_features(backbone, loaders["val"],   device, "val  ")
    te_f, te_l = extract_features(backbone, loaders["test"],  device, "test ")

    # ── Drop 'Other', remap to binary ───────────────────────────────────────
    Xtr, ytr, d_tr = to_binary(tr_f, tr_l)
    Xva, yva, d_va = to_binary(va_f, va_l)
    Xte, yte, d_te = to_binary(te_f, te_l)
    logger.info("\n" + "=" * 60)
    logger.info(" Binary subset (Negatives vs Positives — 'Other' dropped)")
    logger.info("=" * 60)
    logger.info(f"  train: {len(ytr)} kept (dropped {d_tr} Other)  pos={int(ytr.sum())} neg={int((ytr==0).sum())}")
    logger.info(f"  val  : {len(yva)} kept (dropped {d_va} Other)  pos={int(yva.sum())} neg={int((yva==0).sum())}")
    logger.info(f"  test : {len(yte)} kept (dropped {d_te} Other)  pos={int(yte.sum())} neg={int((yte==0).sum())}")

    # ── Fit probe ───────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60 + "\n Fitting logistic-regression linear probe\n" + "=" * 60)
    class_weight = None if args.class_weight == "none" else "balanced"
    C_grid = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    scaler, clf, best_C, best_val_auc = fit_probe(
        Xtr, ytr, Xva, yva, C_grid, class_weight, args.max_iter
    )

    # ── Evaluate ────────────────────────────────────────────────────────────
    val_res,  val_prob  = evaluate_split(clf, scaler, Xva, yva, "val",  args.n_boot, args.seed)
    test_res, test_prob = evaluate_split(clf, scaler, Xte, yte, "test", args.n_boot, args.seed)

    # ── Per-sample probabilities ────────────────────────────────────────────
    save_probabilities(out / "probabilities_val.csv",  yva, val_prob)
    save_probabilities(out / "probabilities_test.csv", yte, test_prob)

    # ── Figures (light theme) ───────────────────────────────────────────────
    logger.info("\n" + "=" * 60 + "\n Plots (light theme)\n" + "=" * 60)
    roc = test_res["roc_curve"]
    plot_roc(np.array(roc["fpr"]), np.array(roc["tpr"]),
             test_res["auc_roc"]["value"], test_res["auc_roc"]["ci95"],
             out / "roc_curve_test.png")
    plot_confusion(test_res["at_threshold_0.5"]["confusion_matrix"],
                   out / "confusion_matrix_test.png")

    if args.run_tsne:
        all_f = np.concatenate([Xtr, Xva, Xte], axis=0)
        all_y = np.concatenate([ytr, yva, yte], axis=0)
        plot_tsne_light(all_f, all_y,
                        "DINOv3 Features — t-SNE (Negatives vs Positives, all splits)",
                        out / "tsne_binary_all.png",
                        max_samples=args.tsne_max_samples, seed=args.seed)
        plot_tsne_light(Xte, yte,
                        "DINOv3 Features — t-SNE (Negatives vs Positives, test)",
                        out / "tsne_binary_test.png",
                        max_samples=args.tsne_max_samples, seed=args.seed)

    # ── Save JSON ───────────────────────────────────────────────────────────
    # strip the verbose roc_curve arrays out of the saved summary
    def _slim(r):
        return {k: v for k, v in r.items() if k != "roc_curve"}

    summary = {
        "ckpt_path": str(args.ckpt_path),
        "config": config_path,
        "embed_dim": embed_dim,
        "classes": BINARY_CLASS_NAMES,
        "dropped_class": "Other",
        "probe": {
            "type": "logistic_regression",
            "standardised": True,
            "class_weight": args.class_weight,
            "best_C": best_C,
            "val_auc_at_selection": best_val_auc,
            "C_grid": C_grid,
        },
        "counts": {
            "train": {"n": int(len(ytr)), "pos": int(ytr.sum()), "neg": int((ytr == 0).sum()), "dropped_other": d_tr},
            "val":   {"n": int(len(yva)), "pos": int(yva.sum()), "neg": int((yva == 0).sum()), "dropped_other": d_va},
            "test":  {"n": int(len(yte)), "pos": int(yte.sum()), "neg": int((yte == 0).sum()), "dropped_other": d_te},
        },
        "val":  _slim(val_res),
        "test": _slim(test_res),
    }
    res_path = out / "binary_probe_results.json"
    with open(res_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"\n✓ Results saved → {res_path}")

    # ── Headline summary ────────────────────────────────────────────────────
    t = test_res
    t05 = t["at_threshold_0.5"]
    logger.info("\n" + "=" * 60)
    logger.info(" SUMMARY  (TEST, Positives vs Negatives)")
    logger.info("=" * 60)
    logger.info(f"  AUC-ROC     : {t['auc_roc']['value']:.4f}  95% CI [{t['auc_roc']['ci95'][0]:.4f}, {t['auc_roc']['ci95'][1]:.4f}]")
    logger.info(f"  Accuracy    : {_fmt(t05['accuracy'])}   (thr=0.5)")
    logger.info(f"  Sensitivity : {_fmt(t05['sensitivity'])}")
    logger.info(f"  Specificity : {_fmt(t05['specificity'])}")
    logger.info(f"  Output dir  : {out}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
