"""
DINOv3 Thalassaemia — Checkpoint Evaluation
============================================
Evaluates a DINOv3 SSL checkpoint (DCP / .distcp format) saved during training.

Supports:
  • Linear Probe  (frozen backbone + trained linear head)
  • KNN Probe     (k-nearest-neighbor on raw features)
  • t-SNE / UMAP  (2-D feature visualisation coloured by class)
  • Confusion Matrix (seaborn heatmap)
  • Full metrics JSON export

Works with:
  - PyTorch Distributed Checkpoint dirs  (ckpt/999  containing __0_0.distcp)
  - Legacy .pth files  (teacher_checkpoint.pth)

Usage (in Colab):
─────────────────
  # From dinov3/ directory:

  # Evaluate checkpoint at iteration 999
  !python scripts/evaluate_checkpoint.py \\
      --ckpt-path   outputs/thalassaemia_scratch/ckpt/999 \\
      --config-path outputs/thalassaemia_scratch/config.yaml \\
      --data-root   ../preprocessed \\
      --output-dir  outputs/eval_999 \\
      --run-tsne

  # Quick KNN-only evaluation (no linear probe training):
  !python scripts/evaluate_checkpoint.py \\
      --ckpt-path   outputs/thalassaemia_scratch/ckpt/999 \\
      --config-path outputs/thalassaemia_scratch/config.yaml \\
      --data-root   ../preprocessed \\
      --knn-only

Notes
─────
• The DCP loader used here works on a single GPU / CPU — no distributed init needed.
• Config is auto-detected from  <ckpt_path>/../../config.yaml  if not provided.
• Output images (t-SNE, confusion matrix) are saved as PNG in --output-dir.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# ── ensure repo root on PYTHONPATH ────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(name)-12s  %(message)s",
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger("evaluate")

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint loading  (DCP + legacy .pth)
# ─────────────────────────────────────────────────────────────────────────────

def _load_dcp_backbone(ckpt_dir: Path, cfg, device: torch.device) -> nn.Module:
    """Load backbone from a PyTorch Distributed Checkpoint directory."""
    import torch.distributed.checkpoint as dcp
    import torch.distributed.checkpoint.filesystem as dcpfs
    from dinov3.models import build_model_from_cfg

    logger.info(f"DCP checkpoint detected: {ckpt_dir}")

    # build_model_from_cfg always creates on 'meta' device — use to_empty() not to()
    teacher, embed_dim = build_model_from_cfg(cfg, only_teacher=True)
    teacher = teacher.to_empty(device="cpu")   # materialize from meta → CPU
    teacher.eval()

    # Inspect the checkpoint keys first so we can map them correctly
    try:
        reader = dcpfs.FileSystemReader(str(ckpt_dir))
        metadata = reader.read_metadata()
        all_keys = list(metadata.state_dict_metadata.keys())
        logger.info(f"  DCP keys (first 5): {all_keys[:5]}")
    except Exception as e:
        logger.warning(f"  Could not read DCP metadata: {e}")
        all_keys = []

    # Determine the key prefix used in the checkpoint
    # Common patterns: "model.<layer>", "teacher.<layer>", plain "<layer>"
    prefixes_seen = set(k.split(".")[0] for k in all_keys)
    logger.info(f"  Top-level DCP key prefixes: {prefixes_seen}")

    # Build a state-dict container matching exactly what's in the checkpoint
    teacher_sd = teacher.state_dict()

    # Try direct load first (keys match model keys)
    load_sd = {"model": teacher_sd}
    try:
        dcp.load(load_sd, storage_reader=dcpfs.FileSystemReader(str(ckpt_dir)))
        missing, unexpected = teacher.load_state_dict(load_sd["model"], strict=False)
        logger.info(f"✓ DCP weights loaded — missing={len(missing)}, unexpected={len(unexpected)}")
    except Exception as e1:
        logger.warning(f"Direct DCP load failed ({e1}), trying prefix-stripped fallback …")
        try:
            # Load the raw tensors and strip common prefixes
            raw: dict = {}
            dcp.load({"model": raw}, storage_reader=dcpfs.FileSystemReader(str(ckpt_dir)))
            cleaned = {}
            for k, v in raw.items():
                for prefix in ("teacher.", "backbone.", "module.", "model."):
                    k = k.removeprefix(prefix)
                cleaned[k] = v
            missing, unexpected = teacher.load_state_dict(cleaned, strict=False)
            logger.info(f"✓ Prefix-stripped load — missing={len(missing)}, unexpected={len(unexpected)}")
        except Exception as e2:
            logger.error(f"Both DCP load strategies failed: {e2}")
            raise

    teacher = teacher.to(device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    logger.info(f"Backbone embed_dim={embed_dim}")
    return teacher, embed_dim


def _load_pth_backbone(pth_path: Path, cfg, device: torch.device) -> nn.Module:
    """Load backbone from a legacy .pth checkpoint."""
    from dinov3.models import build_model_from_cfg

    logger.info(f"Legacy .pth checkpoint: {pth_path}")
    ckpt = torch.load(str(pth_path), map_location="cpu")

    if "teacher" in ckpt:
        state = ckpt["teacher"]
    elif "model" in ckpt:
        state = ckpt["model"]
    else:
        state = ckpt

    # Strip prefixes
    cleaned = {}
    for k, v in state.items():
        for prefix in ("backbone.", "module.", "teacher."):
            k = k.removeprefix(prefix)
        cleaned[k] = v

    teacher, embed_dim = build_model_from_cfg(cfg, only_teacher=True)
    teacher = teacher.to("cpu")
    missing, unexpected = teacher.load_state_dict(cleaned, strict=False)
    if missing:
        logger.warning(f"Missing keys: {missing[:5]}")
    teacher = teacher.to(device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    logger.info(f"Backbone embed_dim={embed_dim}")
    return teacher, embed_dim


def load_backbone(ckpt_path: str, config_path: str, device: torch.device):
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(config_path)

    p = Path(ckpt_path)
    if p.is_dir():
        return _load_dcp_backbone(p, cfg, device)
    else:
        return _load_pth_backbone(p, cfg, device)


# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────

def make_transform(image_size: int = 224):
    from torchvision.transforms import v2
    return v2.Compose([
        v2.ToImage(),
        v2.Resize((image_size, image_size), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def make_dataloaders(data_root: str, batch_size: int, num_workers: int):
    from dinov3.data.datasets.thalassaemia import (
        ThalassaemiaDataset,
        THALASSAEMIA_CLASSES,
        THALASSAEMIA_NUM_CLASSES,
    )

    tf = make_transform()
    splits = {}
    for split_name, split_enum in [
        ("train", ThalassaemiaDataset.Split.TRAIN),
        ("val",   ThalassaemiaDataset.Split.VAL),
        ("test",  ThalassaemiaDataset.Split.TEST),
    ]:
        ds = ThalassaemiaDataset(split=split_enum, root=data_root, transform=tf)
        loader = torch.utils.data.DataLoader(
            ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, drop_last=False, pin_memory=True,
        )
        splits[split_name] = loader
        logger.info(f"  {split_name}: {len(ds)} samples")

    return splits, THALASSAEMIA_CLASSES, THALASSAEMIA_NUM_CLASSES


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def extract_features(backbone: nn.Module, loader, device: torch.device, desc: str = ""):
    all_feats, all_labels = [], []
    total = len(loader.dataset)
    seen = 0
    for images, labels in loader:
        images = images.to(device)
        out = backbone(images)
        if isinstance(out, dict):
            feats = out.get("x_norm_clstoken", out.get("cls_token", next(iter(out.values()))))
        elif isinstance(out, (list, tuple)):
            feats = out[0]
        else:
            feats = out
        all_feats.append(feats.float().cpu())
        all_labels.append(labels)
        seen += len(images)
        if seen % 500 == 0 or seen == total:
            logger.info(f"  {desc}  {seen}/{total} extracted")
    feats  = torch.cat(all_feats,  dim=0)
    labels = torch.cat(all_labels, dim=0)
    logger.info(f"  → feature shape: {feats.shape}")
    return feats, labels


# ─────────────────────────────────────────────────────────────────────────────
# KNN evaluation
# ─────────────────────────────────────────────────────────────────────────────

def knn_eval(
    train_feats, train_labels,
    test_feats,  test_labels,
    k: int = 20,
    num_classes: int = 3,
    class_names: list[str] = None,
):
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

    logger.info(f"\nKNN evaluation  (k={k}) …")
    # L2-normalise
    tf = nn.functional.normalize(train_feats, dim=1)
    qf = nn.functional.normalize(test_feats,  dim=1)

    # Cosine similarity matrix  (chunk to avoid OOM)
    preds = []
    chunk = 256
    for i in range(0, len(qf), chunk):
        sim = qf[i:i+chunk] @ tf.T          # (chunk, N_train)
        topk = sim.topk(k, dim=1).indices    # (chunk, k)
        nn_labels = train_labels[topk]       # (chunk, k)
        vote = torch.zeros(len(nn_labels), num_classes)
        for j in range(k):
            vote.scatter_add_(1, nn_labels[:, j:j+1], sim[i:i+chunk, topk[:, j]:topk[:, j]+1] * 0 + 1)
        preds.append(vote.argmax(dim=1))
    preds = torch.cat(preds).numpy()
    true  = test_labels.numpy()

    acc = accuracy_score(true, preds)
    report = classification_report(true, preds, target_names=class_names or [str(i) for i in range(num_classes)], zero_division=0)
    cm = confusion_matrix(true, preds)

    logger.info(f"  KNN-{k} Accuracy: {acc:.4f}")
    logger.info(f"\nKNN Classification Report:\n{report}")
    return {"knn_accuracy": acc, "k": k, "report": report, "confusion_matrix": cm.tolist(), "preds": preds, "true": true}


# ─────────────────────────────────────────────────────────────────────────────
# Linear probe
# ─────────────────────────────────────────────────────────────────────────────

def linear_probe_eval(
    train_feats, train_labels,
    val_feats,   val_labels,
    test_feats,  test_labels,
    num_classes: int,
    class_names: list[str],
    epochs: int = 30,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: torch.device = torch.device("cpu"),
    batch_size: int = 256,
):
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support

    feat_dim = train_feats.shape[1]
    head = nn.Linear(feat_dim, num_classes).to(device)
    opt  = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit  = nn.CrossEntropyLoss()

    tf = train_feats.to(device)
    tl = train_labels.to(device)
    vf = val_feats.to(device)
    vl = val_labels.to(device)

    best_val, best_state = 0.0, None

    logger.info(f"\nLinear probe  (epochs={epochs}, lr={lr}, wd={weight_decay}) …")
    for epoch in range(epochs):
        head.train()
        perm  = torch.randperm(len(tf), device=device)
        losses = []
        for i in range(0, len(tf), batch_size):
            idx    = perm[i:i+batch_size]
            logits = head(tf[idx])
            loss   = crit(logits, tl[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        sched.step()

        head.eval()
        with torch.no_grad():
            val_acc = (head(vf).argmax(1) == vl).float().mean().item()
        logger.info(f"  epoch {epoch+1:3d}/{epochs}  loss={np.mean(losses):.4f}  val_acc={val_acc:.4f}")

        if val_acc > best_val:
            best_val  = val_acc
            best_state = {k: v.clone() for k, v in head.state_dict().items()}

    head.load_state_dict(best_state)
    logger.info(f"  Best val accuracy: {best_val:.4f}")

    # Evaluate on test
    head.eval()
    with torch.no_grad():
        preds = head(test_feats.to(device)).argmax(1).cpu().numpy()
    true = test_labels.numpy()

    acc = accuracy_score(true, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(true, preds, average="macro", zero_division=0)
    cm = confusion_matrix(true, preds)
    report = classification_report(true, preds, target_names=class_names, zero_division=0)

    logger.info(f"\n{'='*55}")
    logger.info(f"LINEAR PROBE — TEST RESULTS")
    logger.info(f"{'='*55}")
    logger.info(f"  Accuracy:  {acc:.4f}")
    logger.info(f"  Precision: {prec:.4f}")
    logger.info(f"  Recall:    {rec:.4f}")
    logger.info(f"  F1:        {f1:.4f}")
    logger.info(f"\n{report}")

    return {
        "best_val_accuracy": best_val,
        "test_accuracy": acc,
        "test_precision": prec,
        "test_recall": rec,
        "test_f1": f1,
        "confusion_matrix": cm.tolist(),
        "report": report,
        "preds": preds,
        "true": true,
        "head": head,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Visualisations
# ─────────────────────────────────────────────────────────────────────────────

# Colour palette — one per class, visually distinct
CLASS_PALETTE = ["#4FC3F7", "#FF7043", "#66BB6A", "#AB47BC", "#FFA726", "#26C6DA"]


def plot_confusion_matrix(cm: np.ndarray, class_names: list[str], title: str, save_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(max(5, len(class_names) * 1.5), max(4, len(class_names) * 1.4)))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        linewidths=0.5, ax=ax,
    )
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True", fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info(f"  Saved confusion matrix → {save_path}")


def plot_tsne(
    feats: torch.Tensor,
    labels: torch.Tensor,
    class_names: list[str],
    title: str,
    save_path: Path,
    method: str = "tsne",         # "tsne" or "umap"
    perplexity: int = 30,
    max_samples: int = 3000,
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe

    feats_np  = feats.numpy()
    labels_np = labels.numpy()

    # Sub-sample if large dataset
    if len(feats_np) > max_samples:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(feats_np), max_samples, replace=False)
        feats_np  = feats_np[idx]
        labels_np = labels_np[idx]
        logger.info(f"  t-SNE: sub-sampled to {max_samples} points")

    # L2-normalise before dimensionality reduction
    norms = np.linalg.norm(feats_np, axis=1, keepdims=True) + 1e-8
    feats_np = feats_np / norms

    logger.info(f"  Running {method.upper()} on {len(feats_np)} samples …")
    if method == "umap":
        try:
            import umap
            reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
            embedding = reducer.fit_transform(feats_np)
            xlabel, ylabel = "UMAP-1", "UMAP-2"
        except ImportError:
            logger.warning("umap-learn not installed, falling back to t-SNE")
            method = "tsne"

    if method == "tsne":
        from sklearn.manifold import TSNE
        tsne = TSNE(
            n_components=2,
            perplexity=min(perplexity, len(feats_np) // 4),
            n_iter=1000,
            random_state=42,
            verbose=1,
            n_jobs=-1,
        )
        embedding = tsne.fit_transform(feats_np)
        xlabel, ylabel = "t-SNE dim 1", "t-SNE dim 2"

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")

    unique_labels = sorted(np.unique(labels_np))
    for i, lbl in enumerate(unique_labels):
        mask = labels_np == lbl
        color = CLASS_PALETTE[i % len(CLASS_PALETTE)]
        ax.scatter(
            embedding[mask, 0], embedding[mask, 1],
            c=color, label=class_names[lbl],
            s=18, alpha=0.75, linewidths=0,
        )

    # Legend
    legend = ax.legend(
        fontsize=11, markerscale=2.5, framealpha=0.3,
        facecolor="#0f3460", edgecolor="white",
        labelcolor="white",
    )

    ax.set_title(title, fontsize=14, fontweight="bold", color="white", pad=12)
    ax.set_xlabel(xlabel, color="#aaaacc", fontsize=10)
    ax.set_ylabel(ylabel, color="#aaaacc", fontsize=10)
    ax.tick_params(colors="#888899")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333355")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close()
    logger.info(f"  Saved {method.upper()} plot → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a DINOv3 Thalassaemia checkpoint")

    # Checkpoint
    p.add_argument("--ckpt-path",    required=True,
                   help="Path to DCP checkpoint dir (ckpt/999) OR legacy .pth file")
    p.add_argument("--config-path",  default=None,
                   help="Path to config.yaml — auto-detected if not given")

    # Data
    p.add_argument("--data-root",    default="../preprocessed",
                   help="Root of preprocessed data (contains train/ val/ test/ metadata/)")

    # Output
    p.add_argument("--output-dir",   default="outputs/eval",
                   help="Where to save results and plots")

    # Eval options
    p.add_argument("--knn-only",     action="store_true",
                   help="Skip linear probe; run KNN evaluation only")
    p.add_argument("--no-linear",    action="store_true",
                   help="Skip linear probe")
    p.add_argument("--no-knn",       action="store_true",
                   help="Skip KNN evaluation")
    p.add_argument("--run-tsne",     action="store_true",
                   help="Generate t-SNE visualisation")
    p.add_argument("--tsne-method",  default="tsne", choices=["tsne", "umap"],
                   help="Dimensionality reduction method for visualisation")
    p.add_argument("--tsne-max-samples", type=int, default=3000,
                   help="Max points to plot in t-SNE (sub-samples if larger)")

    # Linear probe hyper-params
    p.add_argument("--lp-epochs",    type=int,   default=30)
    p.add_argument("--lp-lr",        type=float, default=1e-3)
    p.add_argument("--lp-wd",        type=float, default=1e-4)

    # KNN
    p.add_argument("--knn-k",        type=int,   default=20)

    # Dataloader
    p.add_argument("--batch-size",   type=int,   default=64)
    p.add_argument("--num-workers",  type=int,   default=2)

    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Auto-detect config ──────────────────────────────────────────────────
    config_path = args.config_path
    if config_path is None:
        # Try <ckpt>/../../../config.yaml  (outputs/thalassaemia_scratch/ckpt/999 → outputs/.../config.yaml)
        candidates = [
            Path(args.ckpt_path).parent.parent / "config.yaml",
            Path(args.ckpt_path).parent.parent.parent / "config.yaml",
            Path(args.ckpt_path) / "config.yaml",
        ]
        for c in candidates:
            if c.exists():
                config_path = str(c)
                logger.info(f"Auto-detected config: {config_path}")
                break
        if config_path is None:
            raise FileNotFoundError(
                "Could not auto-detect config.yaml. "
                "Please pass --config-path explicitly."
            )

    # ── Load backbone ───────────────────────────────────────────────────────
    logger.info("\n" + "="*60)
    logger.info(" Loading backbone")
    logger.info("="*60)
    backbone, embed_dim = load_backbone(args.ckpt_path, config_path, device)

    # ── Datasets ────────────────────────────────────────────────────────────
    logger.info("\n" + "="*60)
    logger.info(" Building datasets")
    logger.info("="*60)
    loaders, class_names, num_classes = make_dataloaders(
        args.data_root, args.batch_size, args.num_workers
    )

    # ── Extract features ────────────────────────────────────────────────────
    logger.info("\n" + "="*60)
    logger.info(" Extracting features")
    logger.info("="*60)
    train_feats, train_labels = extract_features(backbone, loaders["train"], device, "train")
    val_feats,   val_labels   = extract_features(backbone, loaders["val"],   device, "val  ")
    test_feats,  test_labels  = extract_features(backbone, loaders["test"],  device, "test ")

    results = {
        "ckpt_path": str(args.ckpt_path),
        "config": config_path,
        "embed_dim": embed_dim,
        "num_train": len(train_labels),
        "num_val":   len(val_labels),
        "num_test":  len(test_labels),
    }

    # ── KNN ─────────────────────────────────────────────────────────────────
    run_knn = not args.no_knn
    if args.knn_only:
        run_knn = True

    if run_knn:
        logger.info("\n" + "="*60)
        logger.info(" KNN Evaluation")
        logger.info("="*60)
        knn_results = knn_eval(
            train_feats, train_labels,
            test_feats,  test_labels,
            k=args.knn_k,
            num_classes=num_classes,
            class_names=class_names,
        )
        results["knn"] = {k: v for k, v in knn_results.items() if k not in ("preds", "true")}

        # KNN confusion matrix
        cm_knn = np.array(knn_results["confusion_matrix"])
        plot_confusion_matrix(
            cm_knn, class_names,
            title=f"KNN (k={args.knn_k}) — Confusion Matrix (Test)",
            save_path=output_dir / "confusion_matrix_knn.png",
        )

    # ── Linear probe ────────────────────────────────────────────────────────
    lp_results = None
    run_lp = not args.no_linear and not args.knn_only
    if run_lp:
        logger.info("\n" + "="*60)
        logger.info(" Linear Probe")
        logger.info("="*60)
        lp_results = linear_probe_eval(
            train_feats, train_labels,
            val_feats,   val_labels,
            test_feats,  test_labels,
            num_classes=num_classes,
            class_names=class_names,
            epochs=args.lp_epochs,
            lr=args.lp_lr,
            weight_decay=args.lp_wd,
            device=device,
        )

        # Save linear head
        torch.save({
            "state_dict": lp_results["head"].state_dict(),
            "feat_dim": embed_dim,
            "num_classes": num_classes,
            "class_names": class_names,
        }, output_dir / "linear_head.pth")

        # LP confusion matrix
        cm_lp = np.array(lp_results["confusion_matrix"])
        plot_confusion_matrix(
            cm_lp, class_names,
            title="Linear Probe — Confusion Matrix (Test)",
            save_path=output_dir / "confusion_matrix_linear.png",
        )

        results["linear_probe"] = {k: v for k, v in lp_results.items()
                                    if k not in ("preds", "true", "head")}

    # ── t-SNE ───────────────────────────────────────────────────────────────
    if args.run_tsne:
        logger.info("\n" + "="*60)
        logger.info(f" {args.tsne_method.upper()} Visualisation")
        logger.info("="*60)

        # Combine all splits for a richer plot
        all_feats  = torch.cat([train_feats, val_feats, test_feats], dim=0)
        all_labels = torch.cat([train_labels, val_labels, test_labels], dim=0)

        plot_tsne(
            all_feats, all_labels,
            class_names=class_names,
            title=f"DINOv3 Features — {args.tsne_method.upper()} (All Splits)",
            save_path=output_dir / f"{args.tsne_method}_all.png",
            method=args.tsne_method,
            max_samples=args.tsne_max_samples,
        )

        # Test-only t-SNE
        plot_tsne(
            test_feats, test_labels,
            class_names=class_names,
            title=f"DINOv3 Features — {args.tsne_method.upper()} (Test Split)",
            save_path=output_dir / f"{args.tsne_method}_test.png",
            method=args.tsne_method,
            max_samples=args.tsne_max_samples,
        )

    # ── Save results JSON ───────────────────────────────────────────────────
    results_path = output_dir / "eval_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"\n✓ Results saved → {results_path}")

    # ── Summary ─────────────────────────────────────────────────────────────
    logger.info("\n" + "="*60)
    logger.info(" SUMMARY")
    logger.info("="*60)
    if run_knn:
        logger.info(f"  KNN-{args.knn_k} Test Accuracy : {results['knn']['knn_accuracy']:.4f}")
    if lp_results:
        logger.info(f"  Linear Probe Test Accuracy  : {lp_results['test_accuracy']:.4f}")
        logger.info(f"  Linear Probe Test F1 (macro): {lp_results['test_f1']:.4f}")
    logger.info(f"\n  Output directory: {output_dir}")
    logger.info("="*60)


if __name__ == "__main__":
    main()
