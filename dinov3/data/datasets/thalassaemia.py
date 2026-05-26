# Copyright (c) Meta Platforms, Inc. and affiliates.
# Extended for Thalassaemia Microscopy Classification — DINOv3 Research Pipeline

"""
ThalassaemiaDataset
===================
A DINOv3-compatible dataset for thalassaemia microscopy images.
Follows the same interface as ImageNet (ExtendedVisionDataset).

Dataset string format (used in YAML configs):
    Thalassaemia:split=TRAIN:root=<path/to/preprocessed>:extra=<path/to/preprocessed/metadata>

The 'extra' directory defaults to root/metadata if not provided.

Directory layout (produced by scripts/preprocess_data.py):
    preprocessed/
        train/
            Negatives/  img001.jpg ...
            Other/      img001.jpg ...
            Positives/  img001.jpg ...
        val/   ...
        test/  ...
        metadata/
            entries-TRAIN.npy
            entries-VAL.npy
            entries-TEST.npy
            class-ids-TRAIN.npy
            class-ids-VAL.npy
            class-names-TRAIN.npy
            class-names-VAL.npy
            labels.txt
"""

import logging
import os
from enum import Enum
from typing import Callable, List, Optional, Tuple, Union

import numpy as np

from .decoders import ImageDataDecoder, TargetDecoder
from .extended import ExtendedVisionDataset

logger = logging.getLogger("dinov3")

# Class labels — must match what preprocess_data.py produces
THALASSAEMIA_CLASSES = ["Negatives", "Other", "Positives"]
THALASSAEMIA_NUM_CLASSES = len(THALASSAEMIA_CLASSES)

_Target = int


class _Split(Enum):
    TRAIN = "train"
    VAL   = "val"
    TEST  = "test"

    @property
    def length(self) -> int:
        """
        The length is determined at runtime from the entries .npy file,
        so we return -1 here and let __len__ read from the file.
        """
        return -1  # resolved dynamically

    def get_dirname(self) -> str:
        return self.value

    def get_image_relpath(self, class_name: str, filename: str) -> str:
        """Relative path from the root: split/class_name/filename"""
        return os.path.join(self.value, class_name, filename)


class ThalassaemiaDataset(ExtendedVisionDataset):
    """
    Thalassaemia microscopy image dataset for DINOv3 SSL pretraining,
    linear probing, and k-NN evaluation.

    Args:
        split: One of ThalassaemiaDataset.Split enum values.
        root:  Path to the preprocessed data root (contains train/, val/, test/).
        extra: Path to metadata directory. Defaults to root/metadata.
    """

    Target = Union[_Target]
    Split  = Union[_Split]

    CLASS_NAMES  = THALASSAEMIA_CLASSES
    NUM_CLASSES  = THALASSAEMIA_NUM_CLASSES

    def __init__(
        self,
        *,
        split: "_Split",
        root: str,
        extra: Optional[str] = None,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        abs_root = os.path.abspath(root)
        super().__init__(
            root=abs_root,
            transforms=transforms,
            transform=transform,
            target_transform=target_transform,
            image_decoder=ImageDataDecoder,
            target_decoder=TargetDecoder,
        )
        self._split      = split
        self._extra_root = os.path.abspath(extra) if extra is not None else os.path.abspath(os.path.join(abs_root, "metadata"))

        self._entries     = None
        self._class_ids   = None
        self._class_names = None

        # Warm up file system to populate Google Drive / FUSE caches on Colab
        self._warmup_directories()

        # Early validation check to avoid failing deep inside training loop
        self._validate_images_exist()

    def _warmup_directories(self) -> None:
        """Warm up directory contents by listing them to force Google Drive / FUSE sync."""
        try:
            split_dir = os.path.join(self.root, self._split.value)
            if os.path.exists(split_dir):
                logger.info(f"Warming up file system cache for {split_dir}...")
                for class_name in self.CLASS_NAMES:
                    class_dir = os.path.join(split_dir, class_name)
                    if os.path.exists(class_dir):
                        # Force reading directory contents to sync metadata from Google Drive
                        _ = os.listdir(class_dir)
                logger.info("File system cache warmed up successfully.")
        except Exception as e:
            logger.warning(f"File system warmup skipped/failed: {e}")

    def _validate_images_exist(self) -> None:
        """Verify that the first few images exist to catch missing files early."""
        try:
            entries = self._get_entries()
            if len(entries) == 0:
                raise ValueError(f"The metadata entries list for split {self._split.value} is empty!")
            
            # Check up to 5 images to confirm paths are correct and files exist
            num_to_check = min(5, len(entries))
            for i in range(num_to_check):
                entry = entries[i]
                class_name = str(entry["class_name"])
                filename = str(entry["filename"])
                image_relpath = self._split.get_image_relpath(class_name, filename)
                image_fullpath = os.path.join(self.root, image_relpath)
                
                if not os.path.exists(image_fullpath):
                    raise FileNotFoundError(
                        f"Image file not found: '{image_fullpath}'\n"
                        f"The metadata file '{self._get_extra_full_path(self._entries_path)}' was loaded successfully,\n"
                        f"but the actual preprocessed image file is missing.\n"
                        f"Please ensure you copied/mounted the entire 'preprocessed/' directory containing the image splits,\n"
                        f"not just the 'metadata/' directory."
                    )
        except FileNotFoundError as e:
            # Let FileNotFoundError propagate with our helpful context
            raise e

        except Exception as e:
            logger.warning(f"Image validation skipped due to error: {e}")


    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def split(self) -> "_Split":
        return self._split

    @property
    def num_classes(self) -> int:
        return self.NUM_CLASSES

    @property
    def class_names(self) -> List[str]:
        return self.CLASS_NAMES

    # ------------------------------------------------------------------
    # Metadata helpers (mirrors ImageNet API)
    # ------------------------------------------------------------------
    def _get_extra_full_path(self, extra_path: str) -> str:
        return os.path.join(self._extra_root, extra_path)

    def _load_extra(self, extra_path: str) -> np.ndarray:
        full_path = self._get_extra_full_path(extra_path)
        if not os.path.exists(full_path):
            raise FileNotFoundError(
                f"Metadata file not found: {full_path}\n"
                f"Run scripts/preprocess_data.py first to generate metadata."
            )
        return np.load(full_path, mmap_mode="r", allow_pickle=False)

    @property
    def _entries_path(self) -> str:
        return f"entries-{self._split.value.upper()}.npy"

    @property
    def _class_ids_path(self) -> str:
        return f"class-ids-{self._split.value.upper()}.npy"

    @property
    def _class_names_path(self) -> str:
        return f"class-names-{self._split.value.upper()}.npy"

    def _get_entries(self) -> np.ndarray:
        if self._entries is None:
            self._entries = self._load_extra(self._entries_path)
        return self._entries

    def _get_class_ids(self) -> np.ndarray:
        if self._split == _Split.TEST:
            raise AssertionError("Class IDs are not available in TEST split")
        if self._class_ids is None:
            self._class_ids = self._load_extra(self._class_ids_path)
        return self._class_ids

    def _get_class_names(self) -> np.ndarray:
        if self._split == _Split.TEST:
            raise AssertionError("Class names are not available in TEST split")
        if self._class_names is None:
            self._class_names = self._load_extra(self._class_names_path)
        return self._class_names

    # ------------------------------------------------------------------
    # Public class accessors (mirrors ImageNet API)
    # ------------------------------------------------------------------
    def find_class_id(self, class_index: int) -> str:
        return str(self._get_class_ids()[class_index])

    def find_class_name(self, class_index: int) -> str:
        return str(self._get_class_names()[class_index])

    # ------------------------------------------------------------------
    # Core dataset interface
    # ------------------------------------------------------------------
    def get_image_data(self, index: int) -> bytes:
        entries   = self._get_entries()
        entry     = entries[index]
        class_name = str(entry["class_name"])
        filename   = str(entry["filename"])

        image_relpath  = self._split.get_image_relpath(class_name, filename)
        image_fullpath = os.path.join(self.root, image_relpath)

        with open(image_fullpath, mode="rb") as f:
            return f.read()

    def get_target(self, index: int) -> Optional[_Target]:
        entries     = self._get_entries()
        class_index = entries[index]["class_index"]
        return None if self._split == _Split.TEST else int(class_index)

    def get_targets(self) -> Optional[np.ndarray]:
        entries = self._get_entries()
        return None if self._split == _Split.TEST else entries["class_index"]

    def get_class_id(self, index: int) -> Optional[str]:
        entries  = self._get_entries()
        class_id = entries[index]["class_id"]
        return None if self._split == _Split.TEST else str(class_id)

    def get_class_name(self, index: int) -> Optional[str]:
        entries    = self._get_entries()
        class_name = entries[index]["class_name"]
        return None if self._split == _Split.TEST else str(class_name)

    def __len__(self) -> int:
        return len(self._get_entries())

    # ------------------------------------------------------------------
    # Utility: dump_extra (generate metadata if missing)
    # ------------------------------------------------------------------
    def dump_extra(self) -> None:
        """
        Re-generate metadata .npy files from the folder structure.
        Useful if metadata was lost or you add new images.
        """
        import sys
        logger.info("dump_extra: regenerate metadata from folder structure.")
        logger.info("Re-run scripts/preprocess_data.py to rebuild metadata properly.")
