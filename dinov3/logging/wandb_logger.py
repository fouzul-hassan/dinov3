# Copyright (c) Meta Platforms, Inc. and affiliates.
# WandB integration for DINOv3 thalassaemia research pipeline.

"""
WandB Logger for DINOv3 Training & Evaluation
==============================================
Provides a clean WandB integration that hooks into DINOv3's existing
MetricLogger without modifying core training code.

Usage — automatic (via environment variables):
    export WANDB_ENABLED=true
    export WANDB_PROJECT=thalassaemia-dinov3
    export WANDB_RUN_NAME=vits16-scratch
    export WANDB_TAGS=thalassaemia,ssl,vit-small

Usage — programmatic:
    from dinov3.logging.wandb_logger import WandBLogger
    wb = WandBLogger(project="thalassaemia-dinov3", run_name="my-run", cfg=cfg)
    wb.log({"loss": 0.5, "lr": 1e-4}, step=100)
    wb.log_eval({"knn_acc": 0.82, "linear_acc": 0.85}, step=100)
    wb.finish()
"""

import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger("dinov3")

# Global WandBLogger singleton (set up once in run_ssl_colab.py)
_global_wandb_logger: Optional["WandBLogger"] = None


def get_wandb_logger() -> Optional["WandBLogger"]:
    return _global_wandb_logger


def set_wandb_logger(wb: "WandBLogger") -> None:
    global _global_wandb_logger
    _global_wandb_logger = wb


class WandBLogger:
    """
    Lightweight WandB logger for DINOv3 experiments.

    Handles:
    - Training metrics (loss, lr, momentum, grad norms)
    - Eval metrics (kNN accuracy, linear probe accuracy)
    - Config logging as W&B artifact
    - Graceful no-op when wandb is not installed or WANDB_ENABLED=false
    """

    def __init__(
        self,
        *,
        project: str = "thalassaemia-dinov3",
        run_name: Optional[str] = None,
        tags: Optional[list] = None,
        cfg: Optional[Any] = None,
        enabled: bool = True,
        entity: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> None:
        self.enabled = enabled and self._check_wandb_available()
        self._run = None

        if not self.enabled:
            logger.info("WandB logging disabled (wandb not installed or WANDB_ENABLED=false)")
            return

        try:
            import wandb

            # Flatten OmegaConf config to a plain dict for W&B
            config_dict = {}
            if cfg is not None:
                try:
                    from omegaconf import OmegaConf
                    config_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False)
                except Exception:
                    config_dict = {}

            # Build tags list
            final_tags = list(tags or [])
            if "thalassaemia" not in final_tags:
                final_tags.insert(0, "thalassaemia")

            self._run = wandb.init(
                project=project,
                name=run_name,
                entity=entity,
                tags=final_tags,
                notes=notes,
                config=config_dict,
                resume="allow",
            )
            logger.info(f"WandB run initialised: {self._run.url}")

        except Exception as e:
            logger.warning(f"WandB initialisation failed: {e}. Disabling W&B.")
            self.enabled = False

    # ------------------------------------------------------------------
    # Core logging methods
    # ------------------------------------------------------------------
    def log(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """Log training metrics (call every iteration or every N iterations)."""
        if not self.enabled or self._run is None:
            return
        try:
            import wandb
            wandb.log(metrics, step=step)
        except Exception as e:
            logger.debug(f"WandB log failed: {e}")

    def log_eval(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """Log evaluation metrics (kNN, linear probe, etc.)."""
        if not self.enabled or self._run is None:
            return
        try:
            import wandb
            prefixed = {f"eval/{k}": v for k, v in metrics.items()}
            wandb.log(prefixed, step=step)
        except Exception as e:
            logger.debug(f"WandB eval log failed: {e}")

    def log_confusion_matrix(
        self,
        y_true,
        y_pred,
        class_names: list,
        title: str = "Confusion Matrix",
    ) -> None:
        """Log a confusion matrix as a W&B artifact."""
        if not self.enabled or self._run is None:
            return
        try:
            import wandb
            cm = wandb.plot.confusion_matrix(
                probs=None,
                y_true=y_true,
                preds=y_pred,
                class_names=class_names,
                title=title,
            )
            wandb.log({title: cm})
        except Exception as e:
            logger.debug(f"WandB confusion matrix log failed: {e}")

    def log_image_samples(
        self,
        images,
        captions: Optional[list] = None,
        step: Optional[int] = None,
        key: str = "samples",
    ) -> None:
        """Log a batch of images to W&B."""
        if not self.enabled or self._run is None:
            return
        try:
            import wandb
            wb_images = [
                wandb.Image(img, caption=cap if captions else None)
                for img, cap in zip(images, captions or [None] * len(images))
            ]
            wandb.log({key: wb_images}, step=step)
        except Exception as e:
            logger.debug(f"WandB image log failed: {e}")

    def log_summary(self, metrics: Dict[str, Any]) -> None:
        """Log final summary metrics (shown prominently in W&B dashboard)."""
        if not self.enabled or self._run is None:
            return
        try:
            import wandb
            for k, v in metrics.items():
                wandb.run.summary[k] = v
        except Exception as e:
            logger.debug(f"WandB summary log failed: {e}")

    def finish(self) -> None:
        """Mark the run as finished."""
        if not self.enabled or self._run is None:
            return
        try:
            import wandb
            wandb.finish()
            logger.info("WandB run finished.")
        except Exception as e:
            logger.debug(f"WandB finish failed: {e}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _check_wandb_available() -> bool:
        # Check environment variable override
        env_enabled = os.environ.get("WANDB_ENABLED", "true").lower()
        if env_enabled in ("false", "0", "no"):
            return False
        try:
            import wandb  # noqa: F401
            return True
        except ImportError:
            return False

    @property
    def run(self):
        return self._run

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------
    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.finish()


# ---------------------------------------------------------------------------
# Convenience factory — reads from cfg or environment variables
# ---------------------------------------------------------------------------
def make_wandb_logger(cfg=None) -> WandBLogger:
    """
    Create a WandBLogger from a DINOv3 config or environment variables.

    Priority: cfg.wandb > environment variables > defaults
    """
    # Defaults
    project  = "thalassaemia-dinov3"
    run_name = None
    tags     = ["thalassaemia", "dinov3"]
    enabled  = True

    # Read from config if available
    if cfg is not None and hasattr(cfg, "wandb"):
        wcfg = cfg.wandb
        project  = getattr(wcfg, "project",  project)
        run_name = getattr(wcfg, "run_name", run_name)
        tags     = list(getattr(wcfg, "tags",    tags))
        enabled  = bool(getattr(wcfg, "enabled", enabled))

    # Environment variable overrides
    project  = os.environ.get("WANDB_PROJECT",  project)
    run_name = os.environ.get("WANDB_RUN_NAME", run_name)
    env_tags = os.environ.get("WANDB_TAGS", "")
    if env_tags:
        tags = [t.strip() for t in env_tags.split(",") if t.strip()]

    wb = WandBLogger(
        project=project,
        run_name=run_name,
        tags=tags,
        cfg=cfg,
        enabled=enabled,
    )
    set_wandb_logger(wb)
    return wb
