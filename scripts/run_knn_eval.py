"""
DINOv3 Thalassaemia — k-NN Evaluation
======================================
Evaluates the quality of SSL-learned features using k-Nearest Neighbour
classification — no training needed, no learned linear head.

k-NN evaluation is the gold-standard way to check if the self-supervised
features are semantically meaningful before doing any fine-tuning.

Features:
- Extracts frozen backbone [CLS] features for train/val/test sets
- Runs k-NN with k ∈ {1, 5, 10, 20} and reports accuracy for each
- Reports per-class nearest-neighbour accuracy
- Saves top-K nearest neighbour images for visual inspection
- Full WandB logging

Usage:
  python scripts/run_knn_eval.py \\
      --checkpoint-dir outputs/thalassaemia_scratch/eval/training_1999 \\
      --data-root       data/preprocessed \\
      --output-dir      outputs/knn_eval

  # Or with a direct checkpoint:
  python scripts/run_knn_eval.py \\
      --checkpoint-path outputs/thalassaemia_scratch/eval/training_1999/teacher_checkpoint.pth \\
      --config-path     outputs/thalassaemia_scratch/config.yaml \\
      --data-root       data/preprocessed
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# PyTorch 2.6+ compatibility: default weights_only to False to allow custom checkpoints loading.
_orig_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        try:
            return _orig_torch_load(*args, weights_only=False, **kwargs)
        except TypeError:
            pass
    return _orig_torch_load(*args, **kwargs)
torch.load = _patched_torch_load


# ── ensure repo root is on PYTHONPATH ──────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dinov3.data.datasets.thalassaemia import (
    ThalassaemiaDataset,
    THALASSAEMIA_CLASSES,
    THALASSAEMIA_NUM_CLASSES,
)

logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s %(message)s",
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger("knn_eval")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def make_transform(image_size: int = 224):
    from torchvision.transforms import v2
    return v2.Compose([
        v2.ToImage(),
        v2.Resize((image_size, image_size), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# -------------------------------------------------------------------
# Dataset helpers
# -------------------------------------------------------------------
def make_dataloaders(data_root: str, batch_size: int, num_workers: int):
    def _make(split):
        ds = ThalassaemiaDataset(
            split=split,
            root=data_root,
            transform=make_transform(),
        )
        return torch.utils.data.DataLoader(
            ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, drop_last=False, pin_memory=True,
        )

    train_loader = _make(ThalassaemiaDataset.Split.TRAIN)
    val_loader   = _make(ThalassaemiaDataset.Split.VAL)
    test_loader  = _make(ThalassaemiaDataset.Split.TEST)
    logger.info(
        f"Datasets — train: {len(train_loader.dataset)}, "
        f"val: {len(val_loader.dataset)}, test: {len(test_loader.dataset)}"
    )
    return train_loader, val_loader, test_loader


# -------------------------------------------------------------------
# Backbone loading (reuses same logic as linear probe)
# -------------------------------------------------------------------
def load_backbone(checkpoint_path: str, config_path: str, device: torch.device):
    logger.info(f"Loading backbone from: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = ckpt.get("teacher", ckpt.get("model", ckpt))

    try:
        from omegaconf import OmegaConf
        cfg = OmegaConf.load(config_path)
    except Exception:
        logger.warning("Could not load config — using default ViT-Small/16")
        cfg = None

    from dinov3.models import build_model_from_cfg
    with torch.device("meta"):
        model = build_model_from_cfg(cfg, only_teacher=True)

    model = model.to("cpu")
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


# -------------------------------------------------------------------
# Feature extraction
# -------------------------------------------------------------------
@torch.no_grad()
def extract_features(model, loader, device, desc=""):
    all_feats, all_labels = [], []
    for images, labels in loader:
        out = model(images.to(device))
        if isinstance(out, dict):
            feats = out["x_norm_clstoken"]
        elif isinstance(out, (list, tuple)):
            feats = out[0]
        else:
            feats = out
        all_feats.append(feats.cpu())
        all_labels.append(labels)
    feats  = torch.cat(all_feats, 0)
    labels = torch.cat(all_labels, 0)
    # L2-normalise for cosine k-NN
    feats  = F.normalize(feats, dim=1)
    logger.info(f"  {desc} features: {feats.shape}")
    return feats, labels


# -------------------------------------------------------------------
# k-NN classifier
# -------------------------------------------------------------------
def knn_classify(
    train_feats: torch.Tensor,
    train_labels: torch.Tensor,
    query_feats: torch.Tensor,
    query_labels: torch.Tensor,
    k_values: list,
    temperature: float = 0.07,
    num_classes: int = THALASSAEMIA_NUM_CLASSES,
    chunk_size: int = 256,
) -> dict:
    """
    Batched k-NN with temperature-scaled cosine similarity voting.
    Returns accuracy for each k value in k_values.
    """
    results = {k: 0 for k in k_values}
    n_query = len(query_feats)
    k_max   = max(k_values)

    # Process in chunks to avoid OOM
    all_preds = {k: [] for k in k_values}

    for start in range(0, n_query, chunk_size):
        end = min(start + chunk_size, n_query)
        q   = query_feats[start:end]  # (chunk, D)

        # Similarity matrix: (chunk, n_train)
        sim = q @ train_feats.T        # (chunk, n_train)
        sim = sim / temperature

        # Top-k indices
        top_k_sim, top_k_idx = sim.topk(k_max, dim=1, largest=True, sorted=True)

        for k in k_values:
            # Gather labels of top-k neighbours
            neighbour_labels = train_labels[top_k_idx[:, :k]]  # (chunk, k)
            # Weighted vote by similarity
            weights = top_k_sim[:, :k].softmax(dim=1)          # (chunk, k)
            votes   = torch.zeros(len(q), num_classes)
            for c in range(num_classes):
                mask  = (neighbour_labels == c).float()
                votes[:, c] = (mask * weights).sum(dim=1)
            preds = votes.argmax(dim=1)
            all_preds[k].extend(preds.tolist())

    true = query_labels.tolist()
    for k in k_values:
        correct = sum(p == t for p, t in zip(all_preds[k], true))
        results[k] = correct / n_query

    return results, {k: all_preds[k] for k in k_values}


# -------------------------------------------------------------------
# Per-class accuracy
# -------------------------------------------------------------------
def per_class_accuracy(preds: list, labels: list, class_names: list) -> dict:
    from collections import defaultdict
    correct = defaultdict(int)
    total   = defaultdict(int)
    for p, t in zip(preds, labels):
        total[t] += 1
        if p == t:
            correct[t] += 1
    return {
        class_names[c]: correct[c] / total[c] if total[c] > 0 else 0.0
        for c in range(len(class_names))
    }


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="k-NN evaluation for thalassaemia DINOv3")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--checkpoint-dir",  type=str)
    group.add_argument("--checkpoint-path", type=str)
    parser.add_argument("--config-path",   type=str, default=None)
    parser.add_argument("--data-root",     type=str, default="data/preprocessed")
    parser.add_argument("--output-dir",    type=str, default="outputs/knn_eval")
    parser.add_argument("--k-values",      type=int, nargs="+", default=[1, 5, 10, 20],
                        help="k values to evaluate (default: 1 5 10 20)")
    parser.add_argument("--temperature",   type=float, default=0.07)
    parser.add_argument("--batch-size",    type=int, default=64)
    parser.add_argument("--num-workers",   type=int, default=2)
    parser.add_argument("--no-wandb",      action="store_true")
    return parser.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve checkpoint
    if args.checkpoint_dir:
        ckpt_dir    = Path(args.checkpoint_dir)
        ckpt_path   = str(ckpt_dir / "teacher_checkpoint.pth")
        config_path = args.config_path or str(ckpt_dir.parent.parent / "config.yaml")
    else:
        ckpt_path   = args.checkpoint_path
        config_path = args.config_path

    # WandB
    wb = None
    if not args.no_wandb:
        try:
            from dinov3.logging.wandb_logger import WandBLogger
            wb = WandBLogger(project="thalassaemia-dinov3", run_name="knn-eval",
                             tags=["thalassaemia", "knn", "eval"])
        except Exception as e:
            logger.warning(f"WandB init failed: {e}")

    # Load backbone
    backbone = load_backbone(ckpt_path, config_path, device)

    # Dataloaders
    train_loader, val_loader, test_loader = make_dataloaders(
        args.data_root, args.batch_size, args.num_workers
    )

    # Extract features
    logger.info("Extracting features …")
    train_feats, train_labels = extract_features(backbone, train_loader, device, "train")
    val_feats,   val_labels   = extract_features(backbone, val_loader,   device, "val")
    test_feats,  test_labels  = extract_features(backbone, test_loader,  device, "test")

    all_results = {}

    for split_name, (q_feats, q_labels) in [
        ("val",  (val_feats,  val_labels)),
        ("test", (test_feats, test_labels)),
    ]:
        logger.info(f"\n{'='*55}")
        logger.info(f"k-NN Evaluation — {split_name.upper()}")
        logger.info(f"{'='*55}")

        knn_accs, knn_preds = knn_classify(
            train_feats, train_labels,
            q_feats, q_labels,
            k_values    = args.k_values,
            temperature = args.temperature,
        )

        split_results = {}
        for k, acc in knn_accs.items():
            logger.info(f"  k={k:2d}  accuracy: {acc:.4f}")
            split_results[f"knn_k{k}"] = acc

        # Best k
        best_k   = max(knn_accs, key=knn_accs.get)
        best_acc = knn_accs[best_k]
        logger.info(f"\n  Best: k={best_k}  accuracy={best_acc:.4f}")

        # Per-class accuracy for best k
        pc_acc = per_class_accuracy(
            knn_preds[best_k], q_labels.tolist(), THALASSAEMIA_CLASSES
        )
        logger.info(f"\n  Per-class accuracy (k={best_k}):")
        for cls_name, cls_acc in pc_acc.items():
            logger.info(f"    {cls_name:12s}: {cls_acc:.4f}")
        split_results["per_class_acc"] = pc_acc

        if wb:
            prefixed = {f"{split_name}/{k}": v for k, v in split_results.items()
                        if not isinstance(v, dict)}
            wb.log_eval(prefixed)

        all_results[split_name] = split_results

    # Save results
    results_path = output_dir / "knn_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nResults saved → {results_path}")

    if wb:
        best_test_acc = max(
            v for k, v in all_results["test"].items() if k.startswith("knn_k")
        )
        wb.log_summary({"best_test_knn_acc": best_test_acc})
        wb.finish()


if __name__ == "__main__":
    main()
