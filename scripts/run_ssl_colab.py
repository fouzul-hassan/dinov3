"""
DINOv3 Thalassaemia — Colab-Friendly SSL Training Launcher
============================================================
Replaces the SLURM-based `dinov3.run.submit` with a direct single-GPU
launcher suitable for Google Colab free tier (T4 GPU).

Features:
- Initialises torch.distributed for single-GPU mode
- Sets up WandB logging from config or environment variables
- Supports both from-scratch and pretrained ViT-S/16 training
- Provides clean checkpoint resumption
- Adds Colab-specific optimisations (compile=false, etc.)

Usage (run from repo root on Colab):
  # From scratch:
  !PYTHONPATH=${PWD} python scripts/run_ssl_colab.py \\
      --config-file dinov3/configs/train/thalassaemia_vits16_scratch.yaml \\
      --output-dir outputs/thalassaemia_scratch

  # From pretrained:
  !PYTHONPATH=${PWD} python scripts/run_ssl_colab.py \\
      --config-file dinov3/configs/train/thalassaemia_vits16_pretrained.yaml \\
      --output-dir outputs/thalassaemia_pretrained \\
      student.pretrained_weights=checkpoints/dinov3_vits16_pretrained/model.safetensors

  # Resume from checkpoint:
  !PYTHONPATH=${PWD} python scripts/run_ssl_colab.py \\
      --config-file dinov3/configs/train/thalassaemia_vits16_scratch.yaml \\
      --output-dir outputs/thalassaemia_scratch
      # (no --no-resume flag = auto-resumes from last checkpoint)
"""

import os
import sys
import logging
import argparse
from pathlib import Path

# ── ensure repo root is on PYTHONPATH ──────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

logger = logging.getLogger("dinov3")


def setup_single_gpu_distributed():
    """
    Initialise torch.distributed for single-GPU operation.
    DINOv3's training code requires distributed to be initialised,
    even with a single GPU.
    """
    import torch
    import torch.distributed as dist

    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29500")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("LOCAL_RANK", "0")

        dist.init_process_group(
            backend="nccl" if torch.cuda.is_available() else "gloo",
            init_method="env://",
            world_size=1,
            rank=0,
        )
        logger.info("Initialised torch.distributed (single-GPU mode)")


def print_gpu_info():
    """Print GPU memory info — useful for Colab monitoring."""
    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            total = props.total_memory / 1024**3
            logger.info(f"GPU: {props.name}  |  VRAM: {total:.1f} GB")
        else:
            logger.warning("No GPU detected — training will be very slow on CPU!")
    except Exception:
        pass


def check_prerequisites(output_dir: str):
    """Sanity-check requirements before training starts."""
    issues = []

    # Check preprocessed data exists
    preprocessed = _REPO_ROOT / "data" / "preprocessed"
    if not preprocessed.exists():
        issues.append(
            "Preprocessed data not found at data/preprocessed/\n"
            "Run:  python scripts/preprocess_data.py"
        )
    else:
        train_dir = preprocessed / "train"
        if not train_dir.exists() or not any(train_dir.iterdir()):
            issues.append("data/preprocessed/train/ is empty. Re-run preprocess_data.py")

    # Check metadata
    meta_dir = preprocessed / "metadata"
    for f in ["entries-TRAIN.npy", "labels.txt"]:
        if not (meta_dir / f).exists():
            issues.append(f"Missing metadata file: {meta_dir / f}")

    if issues:
        logger.error("Prerequisites check FAILED:")
        for issue in issues:
            logger.error(f"  ✗ {issue}")
        sys.exit(1)

    logger.info("✓ Prerequisites check passed")


def setup_wandb(cfg, output_dir: str):
    """Initialise WandB from config + environment."""
    try:
        from dinov3.logging.wandb_logger import make_wandb_logger
        wb = make_wandb_logger(cfg)
        if wb.enabled:
            logger.info("✓ WandB logging active")
        return wb
    except Exception as e:
        logger.warning(f"WandB setup failed: {e}. Training will continue without W&B.")
        return None


def run_training(args):
    """Main training entry point."""
    # Setup distributed (required even for single GPU)
    setup_single_gpu_distributed()
    print_gpu_info()

    # Check prerequisites
    check_prerequisites(args.output_dir)

    # Import DINOv3 training components
    from dinov3.configs import setup_config, setup_job
    from dinov3.logging import setup_logging

    # Setup job (sets seeds, creates output dir, etc.)
    setup_job(output_dir=args.output_dir, seed=getattr(args, "seed", 0))

    # Setup config (merges YAML + CLI overrides)
    cfg = setup_config(args, strict_cfg=False)

    # Log config
    from dinov3.logging import setup_logging as _sl
    _sl(output=os.path.join(os.path.abspath(args.output_dir), "nan_logs"), name="nan_logger")
    logger.info(f"Config:\n{cfg}")

    # ── WandB ──────────────────────────────────────────────────────────────
    wb = setup_wandb(cfg, args.output_dir)

    # ── Model ──────────────────────────────────────────────────────────────
    import math
    import torch
    from dinov3.train.ssl_meta_arch import SSLMetaArch

    with torch.device("meta"):
        model = SSLMetaArch(cfg)
    model.prepare_for_distributed_training()
    model._apply(
        lambda t: torch.full_like(
            t,
            fill_value=math.nan if t.dtype.is_floating_point
                        else (2 ** (t.dtype.itemsize * 8 - 1)),
            device="cuda",
        ),
        recurse=True,
    )
    logger.info(f"Model: {model.__class__.__name__}")

    # ── Training ───────────────────────────────────────────────────────────
    from dinov3.train.train import do_train
    try:
        do_train(cfg, model, resume=not args.no_resume)
    finally:
        # Always cleanly finish WandB run
        if wb is not None:
            wb.finish()

    logger.info("Training complete!")
    logger.info(f"Checkpoints saved to: {args.output_dir}")
    logger.info(
        "Next steps:\n"
        "  1. Linear probe:  python scripts/run_linear_probe.py "
        "--checkpoint-dir <output_dir>/eval/<iteration>\n"
        "  2. k-NN eval:     python scripts/run_knn_eval.py "
        "--checkpoint-dir <output_dir>/eval/<iteration>"
    )


def get_args_parser():
    """Extend DINOv3's training arg parser with Colab-specific options."""
    from dinov3.train.train import get_args_parser as _base_parser
    parser = _base_parser(add_help=False)
    parser.add_argument(
        "--help", "-h", action="help", help="Show this help message and exit."
    )
    parser.add_argument(
        "--skip-prerequisites",
        action="store_true",
        help="Skip the prerequisites check (not recommended)",
    )
    return parser


def main():
    parser = get_args_parser()
    args = parser.parse_args()

    # Basic logging setup (before DINOv3's logging takes over)
    logging.basicConfig(
        format="%(asctime)s  %(levelname)-8s %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        stream=sys.stdout,
    )

    logger.info("=" * 65)
    logger.info("DINOv3 Thalassaemia SSL Training — Colab Launcher")
    logger.info("=" * 65)
    logger.info(f"Config : {args.config_file}")
    logger.info(f"Output : {args.output_dir}")

    run_training(args)


if __name__ == "__main__":
    main()
