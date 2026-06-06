"""
Thalassaemia Metadata Regeneration Script
==========================================
Regenerates the DINOv3 metadata (.npy) files from an already-preprocessed
directory that contains the split image folders but is missing the metadata/.

Run this INSTEAD of preprocess_data.py when:
  - You already have the preprocessed/ images (train/ val/ test/) on Google Drive
  - But the metadata/ .npy files are missing (e.g. after a Drive sync issue)
  - You do NOT have the original raw-data/ any more

Usage (in Colab):
─────────────────
  !python scripts/regenerate_metadata.py \
      --preprocessed-dir ../preprocessed

  # Or if it's at a different path:
  !python scripts/regenerate_metadata.py \
      --preprocessed-dir /content/drive/MyDrive/3.ResearchWorks/Thalassaemia/preprocessed

Output:
  preprocessed/metadata/
    entries-TRAIN.npy
    entries-VAL.npy
    entries-TEST.npy
    class-ids-TRAIN.npy
    class-ids-VAL.npy
    class-names-TRAIN.npy
    class-names-VAL.npy
    labels.txt
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger("regen_meta")

CLASS_NAMES = ["Negatives", "Other", "Positives"]
SPLITS      = ["train", "val", "test"]
VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


def scan_split(split_dir: Path, class_names: list) -> dict:
    """
    Walk split_dir/{class_name}/ and collect all image filenames.
    Returns: {class_name: [filename, ...]}
    """
    records = {}
    for class_name in class_names:
        class_dir = split_dir / class_name
        if not class_dir.is_dir():
            logger.warning(f"  Class dir not found: {class_dir} — skipping")
            records[class_name] = []
            continue
        files = sorted(
            f.name
            for f in class_dir.iterdir()
            if f.is_file() and f.suffix.lower() in VALID_IMAGE_EXTENSIONS
        )
        records[class_name] = files
        logger.info(f"  {class_name}: {len(files)} images")
    return records


def build_entries_array(split_records: dict, class_names: list) -> np.ndarray:
    """
    Build the structured entries numpy array for one split.
    Schema: actual_index (u4), class_index (u4), class_id (U*), class_name (U*), filename (U*)
    """
    class_to_idx = {cn: i for i, cn in enumerate(class_names)}

    all_entries = []
    actual_index = 0
    for class_name in class_names:
        class_index = class_to_idx[class_name]
        for filename in split_records.get(class_name, []):
            all_entries.append((actual_index, class_index, class_name, class_name, filename))
            actual_index += 1

    if not all_entries:
        return np.array([], dtype=[
            ("actual_index", "<u4"),
            ("class_index",  "<u4"),
            ("class_id",     "U1"),
            ("class_name",   "U1"),
            ("filename",     "U1"),
        ])

    max_class_id_len   = max(len(e[2]) for e in all_entries)
    max_class_name_len = max(len(e[3]) for e in all_entries)
    max_filename_len   = max(len(e[4]) for e in all_entries)

    dtype = np.dtype([
        ("actual_index", "<u4"),
        ("class_index",  "<u4"),
        ("class_id",     f"U{max_class_id_len}"),
        ("class_name",   f"U{max_class_name_len}"),
        ("filename",     f"U{max_filename_len}"),
    ])
    return np.array([(e[0], e[1], e[2], e[3], e[4]) for e in all_entries], dtype=dtype)


def regenerate_metadata(preprocessed_dir: Path) -> None:
    metadata_dir = preprocessed_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output metadata dir: {metadata_dir}")

    # Determine which classes actually exist
    # (use first split that exists to detect classes present)
    existing_classes = []
    for cn in CLASS_NAMES:
        for split in SPLITS:
            if (preprocessed_dir / split / cn).is_dir():
                if cn not in existing_classes:
                    existing_classes.append(cn)
                break

    if not existing_classes:
        logger.error(
            f"No class directories found under any split in {preprocessed_dir}.\n"
            f"Expected structure: preprocessed/train/Negatives, preprocessed/train/Other, etc."
        )
        sys.exit(1)

    logger.info(f"Detected classes: {existing_classes}")

    # Write labels.txt
    labels_path = metadata_dir / "labels.txt"
    with open(labels_path, "w") as f:
        for cn in existing_classes:
            f.write(f"{cn},{cn}\n")
    logger.info(f"Wrote {labels_path}")

    # Process each split
    all_records = {}
    for split_name in SPLITS:
        split_dir = preprocessed_dir / split_name
        if not split_dir.is_dir():
            logger.warning(f"Split directory not found: {split_dir} — skipping {split_name}")
            all_records[split_name] = {}
            continue

        logger.info(f"\nScanning {split_name}/...")
        split_records = scan_split(split_dir, existing_classes)
        all_records[split_name] = split_records

        # entries .npy
        entries_array = build_entries_array(split_records, existing_classes)
        entries_path  = metadata_dir / f"entries-{split_name.upper()}.npy"
        np.save(entries_path, entries_array)
        logger.info(f"  → Wrote {entries_path}  ({len(entries_array)} entries)")

        # class-ids and class-names (not for TEST)
        if split_name != "test":
            max_cn_len = max(len(cn) for cn in existing_classes)
            class_ids_arr   = np.array(existing_classes, dtype=f"U{max_cn_len}")
            class_names_arr = np.array(existing_classes, dtype=f"U{max_cn_len}")
            np.save(metadata_dir / f"class-ids-{split_name.upper()}.npy",   class_ids_arr)
            np.save(metadata_dir / f"class-names-{split_name.upper()}.npy", class_names_arr)
            logger.info(f"  → Wrote class-ids/names for {split_name.upper()}")

    # Summary
    logger.info("\n" + "=" * 55)
    logger.info("METADATA REGENERATION COMPLETE")
    logger.info("=" * 55)
    for split_name in SPLITS:
        total = sum(len(v) for v in all_records.get(split_name, {}).values())
        logger.info(f"  {split_name:6s}: {total:5d} images")
        for cn in existing_classes:
            n = len(all_records.get(split_name, {}).get(cn, []))
            logger.info(f"         {cn}: {n}")
    logger.info(f"\n✓ Metadata ready at: {metadata_dir}")
    logger.info("You can now run evaluate_checkpoint.py")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Regenerate DINOv3 metadata .npy files from existing preprocessed/ folder"
    )
    parser.add_argument(
        "--preprocessed-dir",
        type=str,
        required=True,
        help="Path to the preprocessed/ directory (must contain train/ val/ test/ subdirs)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    preprocessed_dir = Path(args.preprocessed_dir).resolve()

    logger.info("=" * 55)
    logger.info("Thalassaemia Metadata Regeneration")
    logger.info("=" * 55)
    logger.info(f"Preprocessed dir: {preprocessed_dir}")

    if not preprocessed_dir.is_dir():
        logger.error(f"Preprocessed directory not found: {preprocessed_dir}")
        sys.exit(1)

    regenerate_metadata(preprocessed_dir)


if __name__ == "__main__":
    main()
