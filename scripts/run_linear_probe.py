"""
DINOv3 Thalassaemia — Linear Probe Evaluation
==============================================
Runs a linear probe (frozen backbone + trained linear head) on the
thalassaemia dataset after SSL pretraining.

Wraps DINOv3's `dinov3/eval/linear.py` with:
- Thalassaemia-specific dataset paths
- WandB logging of all evaluation metrics
- Confusion matrix generation and logging
- Per-class precision / recall / F1 reporting
- Results saved to a JSON summary file

Usage:
  python scripts/run_linear_probe.py \\
      --checkpoint-dir outputs/thalassaemia_scratch/eval/training_1999 \\
      --data-root       data/preprocessed \\
      --output-dir      outputs/linear_probe \\
      --epochs          20 \\
      --lr              0.001

  # Or using a specific teacher checkpoint .pth:
  python scripts/run_linear_probe.py \\
      --checkpoint-path outputs/thalassaemia_scratch/eval/training_1999/teacher_checkpoint.pth \\
      --config-path     outputs/thalassaemia_scratch/config.yaml \\
      --data-root       data/preprocessed \\
      --output-dir      outputs/linear_probe
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

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
logger = logging.getLogger("linear_probe")

# -------------------------------------------------------------------
# ImageNet-style normalisation (matches DINOv3 training)
# -------------------------------------------------------------------
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
    train_ds = ThalassaemiaDataset(
        split=ThalassaemiaDataset.Split.TRAIN,
        root=data_root,
        transform=make_transform(),
    )
    val_ds = ThalassaemiaDataset(
        split=ThalassaemiaDataset.Split.VAL,
        root=data_root,
        transform=make_transform(),
    )
    test_ds = ThalassaemiaDataset(
        split=ThalassaemiaDataset.Split.TEST,
        root=data_root,
        transform=make_transform(),
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, drop_last=False, pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, drop_last=False, pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, drop_last=False, pin_memory=True,
    )
    logger.info(
        f"Datasets — train: {len(train_ds)}, val: {len(val_ds)}, test: {len(test_ds)}"
    )
    return train_loader, val_loader, test_loader


# -------------------------------------------------------------------
# Load backbone from checkpoint
# -------------------------------------------------------------------
def load_backbone(checkpoint_path: str, config_path: str, device: torch.device):
    """
    Load the teacher backbone from a DINOv3 checkpoint.
    Returns the backbone model in eval mode with frozen weights.
    """
    logger.info(f"Loading backbone from: {checkpoint_path}")

    # Load the teacher checkpoint dict
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    # DINOv3 saves teacher weights under 'teacher' key
    if "teacher" in ckpt:
        state_dict = ckpt["teacher"]
    elif "model" in ckpt:
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt

    # Build ViT backbone using DINOv3's hub
    import dinov3.hub as hub
    try:
        # Try loading from config
        from omegaconf import OmegaConf
        cfg = OmegaConf.load(config_path)
        arch = cfg.student.arch          # e.g. "vit_small"
        patch_size = cfg.student.patch_size  # e.g. 16
    except Exception:
        arch = "vit_small"
        patch_size = 16
        logger.warning(f"Could not read config — defaulting to {arch}/patch{patch_size}")

    # Import the model builder
    from dinov3.models import build_model_from_cfg
    from omegaconf import OmegaConf

    cfg_full = OmegaConf.load(config_path)
    with torch.device("meta"):
        model = build_model_from_cfg(cfg_full, only_teacher=True)

    # Move to real device and load weights
    model = model.to("cpu")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        logger.warning(f"Missing keys in checkpoint: {missing[:5]} ...")
    model = model.to(device)
    model.eval()

    # Freeze all backbone parameters
    for param in model.parameters():
        param.requires_grad_(False)

    logger.info(f"Backbone loaded: {arch}/patch{patch_size}")
    return model


# -------------------------------------------------------------------
# Feature extraction
# -------------------------------------------------------------------
@torch.no_grad()
def extract_features(model, loader, device):
    """Extract [CLS] features from a frozen backbone."""
    all_features, all_labels = [], []
    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device)
        # DINOv3 returns a dict; we want the [CLS] token
        out = model(images)
        if isinstance(out, dict):
            feats = out["x_norm_clstoken"]
        elif isinstance(out, (list, tuple)):
            feats = out[0]
        else:
            feats = out
        all_features.append(feats.cpu())
        all_labels.append(labels)
        if (batch_idx + 1) % 20 == 0:
            logger.info(f"  Extracted {(batch_idx + 1) * loader.batch_size} samples …")

    features = torch.cat(all_features, dim=0)
    labels   = torch.cat(all_labels,   dim=0)
    logger.info(f"  Feature shape: {features.shape}")
    return features, labels


# -------------------------------------------------------------------
# Linear probe training
# -------------------------------------------------------------------
def train_linear_probe(
    train_feats, train_labels,
    val_feats,   val_labels,
    num_classes: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
    wandb_logger=None,
):
    """Train a single linear layer on top of frozen features."""
    feat_dim = train_feats.shape[1]
    linear_head = nn.Linear(feat_dim, num_classes).to(device)
    optimizer = torch.optim.AdamW(linear_head.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    best_val_acc  = 0.0
    best_state    = None

    train_feats  = train_feats.to(device)
    train_labels = train_labels.to(device)
    val_feats    = val_feats.to(device)
    val_labels   = val_labels.to(device)

    for epoch in range(epochs):
        # ── Train epoch ──────────────────────────────────────────────
        linear_head.train()
        perm   = torch.randperm(len(train_feats))
        losses = []
        batch_size = 64
        for i in range(0, len(train_feats), batch_size):
            idx    = perm[i : i + batch_size]
            logits = linear_head(train_feats[idx])
            loss   = criterion(logits, train_labels[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        scheduler.step()

        # ── Val eval ─────────────────────────────────────────────────
        linear_head.eval()
        with torch.no_grad():
            val_logits = linear_head(val_feats)
            val_preds  = val_logits.argmax(dim=1)
            val_acc    = (val_preds == val_labels).float().mean().item()

        train_loss = float(np.mean(losses))
        current_lr = scheduler.get_last_lr()[0]
        logger.info(
            f"Epoch {epoch+1:3d}/{epochs}  "
            f"loss={train_loss:.4f}  val_acc={val_acc:.4f}  lr={current_lr:.2e}"
        )

        if wandb_logger is not None:
            wandb_logger.log({
                "linear_probe/train_loss": train_loss,
                "linear_probe/val_acc":    val_acc,
                "linear_probe/lr":         current_lr,
            }, step=epoch)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state   = {k: v.clone() for k, v in linear_head.state_dict().items()}

    linear_head.load_state_dict(best_state)
    logger.info(f"Best val accuracy: {best_val_acc:.4f}")
    return linear_head, best_val_acc


# -------------------------------------------------------------------
# Test evaluation + metrics
# -------------------------------------------------------------------
@torch.no_grad()
def evaluate(model_head, feats, labels, class_names, split_name, wandb_logger=None):
    """Evaluate linear probe and return full metrics."""
    from sklearn.metrics import (
        accuracy_score, precision_recall_fscore_support,
        confusion_matrix, classification_report,
    )

    model_head.eval()
    device = next(model_head.parameters()).device
    logits = model_head(feats.to(device))
    preds  = logits.argmax(dim=1).cpu().numpy()
    true   = labels.numpy()

    acc     = accuracy_score(true, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        true, preds, average="macro", zero_division=0
    )
    cm = confusion_matrix(true, preds)
    report = classification_report(true, preds, target_names=class_names, zero_division=0)

    logger.info(f"\n{'='*55}")
    logger.info(f"{split_name.upper()} RESULTS")
    logger.info(f"{'='*55}")
    logger.info(f"  Accuracy (macro): {acc:.4f}")
    logger.info(f"  Precision:        {prec:.4f}")
    logger.info(f"  Recall:           {rec:.4f}")
    logger.info(f"  F1 (macro):       {f1:.4f}")
    logger.info(f"\nClassification Report:\n{report}")

    if wandb_logger is not None:
        wandb_logger.log_eval({
            f"{split_name}_accuracy":  acc,
            f"{split_name}_precision": prec,
            f"{split_name}_recall":    rec,
            f"{split_name}_f1":        f1,
        })
        try:
            wandb_logger.log_confusion_matrix(
                y_true=true.tolist(),
                y_pred=preds.tolist(),
                class_names=class_names,
                title=f"{split_name.capitalize()} Confusion Matrix",
            )
        except Exception:
            pass

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
            "confusion_matrix": cm.tolist(), "report": report}


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Linear probe evaluation for thalassaemia DINOv3")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--checkpoint-dir",  type=str,
                       help="Path to eval checkpoint directory (contains teacher_checkpoint.pth + config.yaml)")
    group.add_argument("--checkpoint-path", type=str,
                       help="Path directly to teacher_checkpoint.pth")

    parser.add_argument("--config-path",  type=str, default=None,
                        help="Path to config.yaml (auto-detected if --checkpoint-dir)")
    parser.add_argument("--data-root",    type=str, default="data/preprocessed",
                        help="Root of preprocessed data")
    parser.add_argument("--output-dir",   type=str, default="outputs/linear_probe")
    parser.add_argument("--epochs",       type=int,   default=20)
    parser.add_argument("--lr",           type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size",   type=int,   default=64)
    parser.add_argument("--num-workers",  type=int,   default=2)
    parser.add_argument("--no-wandb",     action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve checkpoint + config paths
    if args.checkpoint_dir:
        ckpt_dir   = Path(args.checkpoint_dir)
        ckpt_path  = str(ckpt_dir / "teacher_checkpoint.pth")
        config_path = args.config_path or str(ckpt_dir.parent.parent / "config.yaml")
    else:
        ckpt_path   = args.checkpoint_path
        config_path = args.config_path

    # WandB
    wb = None
    if not args.no_wandb:
        try:
            from dinov3.logging.wandb_logger import WandBLogger
            wb = WandBLogger(project="thalassaemia-dinov3", run_name="linear-probe",
                             tags=["thalassaemia", "linear-probe", "eval"])
        except Exception as e:
            logger.warning(f"WandB init failed: {e}")

    # Load backbone
    backbone = load_backbone(ckpt_path, config_path, device)

    # Dataloaders
    train_loader, val_loader, test_loader = make_dataloaders(
        args.data_root, args.batch_size, args.num_workers
    )

    # Extract features
    logger.info("Extracting train features …")
    train_feats, train_labels = extract_features(backbone, train_loader, device)
    logger.info("Extracting val features …")
    val_feats,   val_labels   = extract_features(backbone, val_loader,   device)
    logger.info("Extracting test features …")
    test_feats,  test_labels  = extract_features(backbone, test_loader,  device)

    # Train linear head
    logger.info(f"\nTraining linear probe  (epochs={args.epochs}, lr={args.lr}) …")
    linear_head, best_val_acc = train_linear_probe(
        train_feats, train_labels,
        val_feats,   val_labels,
        num_classes  = THALASSAEMIA_NUM_CLASSES,
        epochs       = args.epochs,
        lr           = args.lr,
        weight_decay = args.weight_decay,
        device       = device,
        wandb_logger = wb,
    )

    # Evaluate on val and test
    val_results  = evaluate(linear_head, val_feats,  val_labels,
                            THALASSAEMIA_CLASSES, "val",  wb)
    test_results = evaluate(linear_head, test_feats, test_labels,
                            THALASSAEMIA_CLASSES, "test", wb)

    # Save linear head
    head_path = output_dir / "linear_head.pth"
    torch.save({"state_dict": linear_head.state_dict(),
                "feat_dim": train_feats.shape[1],
                "num_classes": THALASSAEMIA_NUM_CLASSES,
                "class_names": THALASSAEMIA_CLASSES}, head_path)

    # Save results JSON
    results = {
        "checkpoint": ckpt_path,
        "val":  {k: v for k, v in val_results.items()  if k != "report"},
        "test": {k: v for k, v in test_results.items() if k != "report"},
    }
    results_path = output_dir / "linear_probe_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved → {results_path}")

    if wb:
        wb.log_summary({
            "best_val_acc":  best_val_acc,
            "test_accuracy": test_results["accuracy"],
            "test_f1":       test_results["f1"],
        })
        wb.finish()


if __name__ == "__main__":
    main()
