"""
Thalassaemia Microscopy Image Preprocessing Pipeline
======================================================
Prepares raw patient-level image data for DINOv3 SSL training.

Data structure expected:
  raw-data/
    Positives/  <patient_id>/  *.jpg
    Negatives/  <patient_id>/  *.jpg
    Other/      <patient_id>/  *.heic or *.jpg

Output structure:
  preprocessed/
    train/  Positives/  Negatives/  Other/
    val/    Positives/  Negatives/  Other/
    test/   Positives/  Negatives/  Other/
    metadata/
      split_manifest.json
      labels.txt
      entries-{TRAIN,VAL,TEST}.npy
      class-ids-{TRAIN,VAL}.npy
      class-names-{TRAIN,VAL}.npy
      dataset_stats.json   <- mean/std computed from train images

Usage (run from repo root):
  python scripts/preprocess_data.py \\
      --raw-data-dir data/raw-data \\
      --output-dir   data/preprocessed \\
      --train-ratio  0.70 \\
      --val-ratio    0.15 \\
      --seed         42
"""

import argparse
import json
import logging
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger("preprocess")


# ---------------------------------------------------------------------------
# HEIC conversion helper
# ---------------------------------------------------------------------------
def _try_import_heif():
    """Try to import pillow-heif; return converter or None."""
    try:
        import pillow_heif
        pillow_heif.register_heif_opener()
        logger.info("pillow-heif found — HEIC images will be converted automatically.")
        return True
    except ImportError:
        logger.warning(
            "pillow-heif NOT found.  HEIC images in the 'Other' class will be skipped.\n"
            "Install it with:  pip install pillow-heif"
        )
        return False


def convert_and_copy_image(src: Path, dst: Path, heif_available: bool) -> bool:
    """
    Copy a JPEG image, or convert a HEIC image to JPEG.
    Returns True on success, False if the image was skipped.
    """
    from PIL import Image

    suffix = src.suffix.lower()
    try:
        if suffix in (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"):
            shutil.copy2(src, dst.with_suffix(".jpg"))
            return True
        elif suffix == ".heic":
            if not heif_available:
                return False
            # pillow-heif registers itself as a PIL opener, so just use PIL
            img = Image.open(src).convert("RGB")
            img.save(dst.with_suffix(".jpg"), "JPEG", quality=95)
            return True
        else:
            logger.warning(f"Unknown extension {suffix}, skipping: {src}")
            return False
    except Exception as e:
        logger.warning(f"Failed to process {src}: {e}")
        return False


# ---------------------------------------------------------------------------
# Dataset discovery
# ---------------------------------------------------------------------------
CLASS_NAMES = ["Negatives", "Other", "Positives"]   # sorted for determinism
VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".heic"}


def discover_patients(raw_data_dir: Path) -> Dict[str, List[Tuple[str, Path]]]:
    """
    Returns a dict: class_name -> list of (patient_id, patient_dir).
    Only classes that exist as subdirectories are included.
    """
    patients_by_class: Dict[str, List[Tuple[str, Path]]] = {}
    for class_name in CLASS_NAMES:
        class_dir = raw_data_dir / class_name
        if not class_dir.is_dir():
            logger.warning(f"Class directory not found: {class_dir}")
            continue
        patient_dirs = sorted(
            [d for d in class_dir.iterdir() if d.is_dir()],
            key=lambda d: d.name,
        )
        patients_by_class[class_name] = [(d.name, d) for d in patient_dirs]
        logger.info(f"  {class_name}: {len(patient_dirs)} patients")
    return patients_by_class


def collect_images(patient_dir: Path) -> List[Path]:
    """Return all valid image files in a patient directory."""
    images = [
        f
        for f in sorted(patient_dir.iterdir())
        if f.is_file() and f.suffix.lower() in VALID_IMAGE_EXTENSIONS
    ]
    return images


# ---------------------------------------------------------------------------
# Patient-level stratified split
# ---------------------------------------------------------------------------
SPLITS = ["train", "val", "test"]


def stratified_patient_split(
    patients_by_class: Dict[str, List[Tuple[str, Path]]],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Dict[str, Dict[str, List[Tuple[str, Path]]]]:
    """
    Performs a stratified split at the patient level.
    Returns: {split_name: {class_name: [(patient_id, patient_dir)]}}
    """
    rng = random.Random(seed)
    split_assignment: Dict[str, Dict[str, List[Tuple[str, Path]]]] = {
        s: {} for s in SPLITS
    }

    for class_name, patients in patients_by_class.items():
        shuffled = patients.copy()
        rng.shuffle(shuffled)

        n = len(shuffled)
        n_train = max(1, round(n * train_ratio))
        n_val   = max(1, round(n * val_ratio))
        n_test  = n - n_train - n_val

        # Safety guard for tiny classes
        if n_test < 1:
            n_test = 1
            n_train = n - n_val - 1

        split_assignment["train"][class_name] = shuffled[:n_train]
        split_assignment["val"][class_name]   = shuffled[n_train : n_train + n_val]
        split_assignment["test"][class_name]  = shuffled[n_train + n_val :]

        logger.info(
            f"  {class_name}: {n_train} train | {n_val} val | {n_test} test patients"
        )

    return split_assignment


# ---------------------------------------------------------------------------
# Copy files into split folders
# ---------------------------------------------------------------------------
def build_split_directories(
    split_assignment: Dict[str, Dict[str, List[Tuple[str, Path]]]],
    output_dir: Path,
    heif_available: bool,
) -> Dict[str, Dict[str, List[Tuple[str, str, int]]]]:
    """
    Copy/convert images into output_dir/{split}/{class}/.
    Returns image records: {split: {class_name: [(patient_id, filename, actual_index)]}}
    """
    records: Dict[str, Dict[str, List[Tuple[str, str, int]]]] = {
        s: {} for s in SPLITS
    }

    for split_name, classes in split_assignment.items():
        for class_name, patients in classes.items():
            dst_class_dir = output_dir / split_name / class_name
            dst_class_dir.mkdir(parents=True, exist_ok=True)
            class_records = []
            actual_index = 0

            for patient_id, patient_dir in patients:
                images = collect_images(patient_dir)
                if not images:
                    logger.warning(f"No images found in {patient_dir}")
                    continue

                for img_path in images:
                    # Unique filename encodes patient and image name
                    stem = f"{patient_id.replace(' ', '_')}__{img_path.stem}"
                    dst = dst_class_dir / stem  # extension added in convert_and_copy_image

                    success = convert_and_copy_image(img_path, dst, heif_available)
                    if success:
                        final_name = stem + ".jpg"
                        class_records.append((patient_id, final_name, actual_index))
                        actual_index += 1

            records[split_name][class_name] = class_records
            logger.info(
                f"  [{split_name}/{class_name}] {actual_index} images from "
                f"{len(patients)} patients"
            )

    return records


# ---------------------------------------------------------------------------
# DINOv3 metadata (.npy arrays)
# ---------------------------------------------------------------------------
def build_metadata(
    records: Dict[str, Dict[str, List[Tuple[str, str, int]]]],
    class_names: List[str],
    metadata_dir: Path,
) -> None:
    """
    Generate DINOv3-compatible metadata files:
      entries-{TRAIN,VAL,TEST}.npy
      class-ids-{TRAIN,VAL}.npy
      class-names-{TRAIN,VAL}.npy
      labels.txt
    """
    metadata_dir.mkdir(parents=True, exist_ok=True)

    # Stable class → index mapping
    class_to_idx = {cn: i for i, cn in enumerate(class_names)}

    # Write labels.txt  (class_id,class_name  per line)
    labels_path = metadata_dir / "labels.txt"
    with open(labels_path, "w") as f:
        for cn in class_names:
            f.write(f"{cn},{cn}\n")
    logger.info(f"Wrote {labels_path}")

    for split_name in SPLITS:
        split_records = records[split_name]

        # Flatten all images into a single list for this split
        all_entries = []
        for class_name in class_names:
            class_index = class_to_idx[class_name]
            for patient_id, filename, actual_index in split_records.get(class_name, []):
                all_entries.append(
                    (actual_index, class_index, class_name, class_name, filename)
                )

        if not all_entries:
            continue

        # entries array
        max_class_id_len   = max(len(e[2]) for e in all_entries)
        max_class_name_len = max(len(e[3]) for e in all_entries)
        max_filename_len   = max(len(e[4]) for e in all_entries)

        dtype = np.dtype(
            [
                ("actual_index", "<u4"),
                ("class_index",  "<u4"),
                ("class_id",     f"U{max_class_id_len}"),
                ("class_name",   f"U{max_class_name_len}"),
                ("filename",     f"U{max_filename_len}"),
            ]
        )
        entries_array = np.array(
            [(e[0], e[1], e[2], e[3], e[4]) for e in all_entries], dtype=dtype
        )
        entries_path = metadata_dir / f"entries-{split_name.upper()}.npy"
        np.save(entries_path, entries_array)
        logger.info(f"Wrote {entries_path}  ({len(entries_array)} entries)")

        # class-ids and class-names arrays (only for TRAIN/VAL, not TEST)
        if split_name != "test":
            class_ids_array   = np.array(class_names, dtype=f"U{max_class_id_len}")
            class_names_array = np.array(class_names, dtype=f"U{max_class_name_len}")
            np.save(metadata_dir / f"class-ids-{split_name.upper()}.npy",   class_ids_array)
            np.save(metadata_dir / f"class-names-{split_name.upper()}.npy", class_names_array)


# ---------------------------------------------------------------------------
# Dataset statistics (mean / std) — computed on TRAIN images
# ---------------------------------------------------------------------------
def compute_dataset_stats(
    train_dir: Path,
    class_names: List[str],
    max_samples: int = 2000,
    resize: int = 224,
) -> Dict:
    """
    Estimate per-channel mean and std from a random subset of training images.
    Uses torchvision to load images so the values match what the model sees.
    """
    logger.info("Computing dataset statistics from training images …")

    try:
        import torch
        from torchvision import transforms
        from PIL import Image
    except ImportError:
        logger.warning("torch/torchvision not available — skipping stats computation.")
        return {}

    transform = transforms.Compose([
        transforms.Resize((resize, resize)),
        transforms.ToTensor(),           # [0,1]
    ])

    all_image_paths: List[Path] = []
    for class_name in class_names:
        class_dir = train_dir / class_name
        if class_dir.is_dir():
            all_image_paths.extend(sorted(class_dir.glob("*.jpg")))

    random.shuffle(all_image_paths)
    sample_paths = all_image_paths[:max_samples]
    logger.info(f"  Sampling {len(sample_paths)} / {len(all_image_paths)} train images")

    channel_sums   = torch.zeros(3, dtype=torch.float64)
    channel_sq_sum = torch.zeros(3, dtype=torch.float64)
    n_pixels = 0

    for p in sample_paths:
        try:
            img = Image.open(p).convert("RGB")
            t = transform(img)           # C x H x W
            channel_sums   += t.sum(dim=[1, 2]).double()
            channel_sq_sum += (t ** 2).sum(dim=[1, 2]).double()
            n_pixels       += t.shape[1] * t.shape[2]
        except Exception as e:
            logger.warning(f"Skipping {p}: {e}")

    mean = (channel_sums / n_pixels).tolist()
    std  = ((channel_sq_sum / n_pixels - torch.tensor(mean, dtype=torch.float64) ** 2)
            .clamp(min=0).sqrt().tolist())

    stats = {
        "n_samples": len(sample_paths),
        "mean": mean,
        "std":  std,
        "imagenet_mean": [0.485, 0.456, 0.406],
        "imagenet_std":  [0.229, 0.224, 0.225],
        "note": "Use 'mean'/'std' for domain-adapted training, or 'imagenet_mean'/'imagenet_std' when fine-tuning from pretrained ImageNet weights."
    }
    logger.info(f"  Mean: {[round(v, 4) for v in mean]}")
    logger.info(f"  Std:  {[round(v, 4) for v in std]}")
    return stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# Repo root is the parent of the directory this script lives in
_SCRIPT_DIR = Path(__file__).resolve().parent   # .../dinov3/scripts/
_REPO_ROOT   = _SCRIPT_DIR.parent               # .../dinov3/


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess thalassaemia microscopy data for DINOv3")
    parser.add_argument(
        "--raw-data-dir", type=str,
        default=str(_REPO_ROOT / "data" / "raw-data"),
        help="Path to raw-data directory (default: <repo-root>/data/raw-data)"
    )
    parser.add_argument(
        "--output-dir", type=str,
        default=str(_REPO_ROOT / "data" / "preprocessed"),
        help="Path to write preprocessed data (default: <repo-root>/data/preprocessed)"
    )
    parser.add_argument("--train-ratio",  type=float, default=0.70,
                        help="Fraction of patients for training (default 0.70)")
    parser.add_argument("--val-ratio",    type=float, default=0.15,
                        help="Fraction of patients for validation (default 0.15)")
    parser.add_argument("--seed",         type=int,   default=42,
                        help="Random seed for patient split reproducibility")
    parser.add_argument("--skip-stats",   action="store_true",
                        help="Skip computing dataset mean/std statistics")
    return parser.parse_args()


def main():
    args = parse_args()

    raw_data_dir = Path(args.raw_data_dir).resolve()
    output_dir   = Path(args.output_dir).resolve()

    logger.info("=" * 60)
    logger.info("DINOv3 Thalassaemia Preprocessing Pipeline")
    logger.info("=" * 60)
    logger.info(f"Repo root : {_REPO_ROOT}")
    logger.info(f"Raw data  : {raw_data_dir}")
    logger.info(f"Output    : {output_dir}")
    logger.info(f"Split     : {args.train_ratio:.0%} train / {args.val_ratio:.0%} val / "
                f"{1 - args.train_ratio - args.val_ratio:.0%} test")
    logger.info(f"Seed      : {args.seed}")

    # ── Validate raw data directory exists ────────────────────────────────
    if not raw_data_dir.exists():
        logger.error(f"Raw data directory NOT FOUND: {raw_data_dir}")
        logger.error("Please pass the correct path with --raw-data-dir, for example:")
        logger.error(f"  python scripts/preprocess_data.py --raw-data-dir data/raw-data")
        logger.error(f"  (run from the repo root: {_REPO_ROOT})")
        sys.exit(1)

    expected_classes = ["Negatives", "Other", "Positives"]
    found_classes = [c for c in expected_classes if (raw_data_dir / c).is_dir()]
    if not found_classes:
        logger.error(
            f"No expected class directories found in {raw_data_dir}\n"
            f"Expected: {expected_classes}\n"
            f"Found:    {[d.name for d in raw_data_dir.iterdir() if d.is_dir()]}"
        )
        sys.exit(1)
    logger.info(f"Found class dirs: {found_classes}")

    # 0. Check HEIC support
    heif_available = _try_import_heif()

    # 1. Discover patients
    logger.info("\n[Step 1/5] Discovering patients …")
    patients_by_class = discover_patients(raw_data_dir)
    total_patients = sum(len(v) for v in patients_by_class.values())
    logger.info(f"Total patients: {total_patients}")

    # 2. Split at patient level
    logger.info("\n[Step 2/5] Stratified patient-level split …")
    split_assignment = stratified_patient_split(
        patients_by_class,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    # 3. Copy / convert images
    logger.info("\n[Step 3/5] Copying / converting images …")
    records = build_split_directories(split_assignment, output_dir, heif_available)

    # 4. Build DINOv3 metadata
    logger.info("\n[Step 4/5] Building DINOv3 metadata arrays …")
    existing_classes = list(patients_by_class.keys())
    metadata_dir = output_dir / "metadata"
    build_metadata(records, existing_classes, metadata_dir)

    # 5. Compute dataset stats
    if not args.skip_stats:
        logger.info("\n[Step 5/5] Computing dataset statistics …")
        stats = compute_dataset_stats(output_dir / "train", existing_classes)
        if stats:
            stats_path = metadata_dir / "dataset_stats.json"
            with open(stats_path, "w") as f:
                json.dump(stats, f, indent=2)
            logger.info(f"Wrote {stats_path}")
    else:
        logger.info("\n[Step 5/5] Skipping stats computation (--skip-stats)")

    # 6. Save split manifest
    manifest = {}
    for split_name, classes in split_assignment.items():
        manifest[split_name] = {}
        for class_name, patients in classes.items():
            manifest[split_name][class_name] = [pid for pid, _ in patients]

    manifest_path = metadata_dir / "split_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"\nWrote split manifest → {manifest_path}")

    # 7. Print summary
    logger.info("\n" + "=" * 60)
    logger.info("PREPROCESSING COMPLETE — Summary")
    logger.info("=" * 60)
    for split_name in SPLITS:
        total_imgs = sum(
            len(records[split_name].get(cn, [])) for cn in existing_classes
        )
        logger.info(f"  {split_name:6s}: {total_imgs:4d} images")
        for cn in existing_classes:
            n = len(records[split_name].get(cn, []))
            logger.info(f"          {cn}: {n}")
    logger.info(f"\nData ready at: {output_dir}")
    logger.info("Next step: Register the dataset and start SSL training!")


if __name__ == "__main__":
    main()
