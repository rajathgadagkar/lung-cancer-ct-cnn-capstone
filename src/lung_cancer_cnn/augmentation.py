"""Deterministic 14-fold data augmentation pipeline for lung CT scans.

Expands the 100 preprocessed CT slices into 1,400 images (14x) by composing
Shear, Scale, Horizontal Flip, and Vertical Flip transformations as documented
in Section 6.2 and Appendix B of the Capstone Report.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple

from .config import AugmentationConfig

AUGMENTATION_SET_NAMES: Tuple[str, ...] = (
    "01_original",
    "02_sheared",
    "03_scaled",
    "04_hflip",
    "05_vflip",
    "06_sheared_hflip",
    "07_scaled_hflip",
    "08_vflip_hflip",
    "09_sheared_vflip",
    "10_scaled_vflip",
    "11_scaled_hflip_sheared",
    "12_hflip_scaled_sheared_vflip",
    "13_scaled_sheared_vflip",
    "14_scaled_sheared",
)


def _require_aug_deps() -> Tuple[Any, Any]:
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise ImportError(
            "Data augmentation requires numpy and opencv-python. "
            "Install them with: pip install -r requirements.txt"
        ) from exc
    return np, cv2


def horizontal_flip(images: Sequence[Any]) -> List[Any]:
    """Mirror each 2D image left-to-right."""
    np, _ = _require_aug_deps()
    return [np.ascontiguousarray(np.asarray(img)[:, ::-1]) for img in images]


def vertical_flip(images: Sequence[Any]) -> List[Any]:
    """Mirror each 2D image top-to-bottom."""
    np, _ = _require_aug_deps()
    return [np.ascontiguousarray(np.asarray(img)[::-1, :]) for img in images]


def random_scale(
    images: Sequence[Any],
    scale_range: Tuple[float, float] = (0.9, 1.0),
    rng: Optional[Any] = None,
) -> List[Any]:
    """Apply centered affine scaling within ``scale_range`` while preserving image dimensions."""
    np, cv2 = _require_aug_deps()
    generator = rng if rng is not None else np.random.default_rng(42)

    results: List[Any] = []
    for img in images:
        arr = np.asarray(img)
        h, w = arr.shape[:2]
        sx = float(generator.uniform(scale_range[0], scale_range[1]))
        sy = float(generator.uniform(scale_range[0], scale_range[1]))
        matrix = np.array(
            [
                [sx, 0.0, (1.0 - sx) * w * 0.5],
                [0.0, sy, (1.0 - sy) * h * 0.5],
            ],
            dtype=np.float32,
        )
        warped = cv2.warpAffine(arr, matrix, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        results.append(warped)
    return results


def random_shear(
    images: Sequence[Any],
    shear_degrees_range: Tuple[float, float] = (0.0, 20.0),
    rng: Optional[Any] = None,
) -> List[Any]:
    """Apply horizontal affine shear in degrees within ``shear_degrees_range``."""
    np, cv2 = _require_aug_deps()
    generator = rng if rng is not None else np.random.default_rng(42)

    results: List[Any] = []
    for img in images:
        arr = np.asarray(img)
        h, w = arr.shape[:2]
        angle_deg = float(generator.uniform(shear_degrees_range[0], shear_degrees_range[1]))
        shx = float(np.tan(np.deg2rad(angle_deg)))
        matrix = np.array(
            [
                [1.0, shx, -shx * h * 0.5],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        )
        warped = cv2.warpAffine(arr, matrix, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        results.append(warped)
    return results


def augment_dataset_14x(
    images: Sequence[Any],
    labels: Sequence[int],
    config: Optional[AugmentationConfig] = None,
) -> Tuple[Any, Any]:
    """Generate the exact 14-fold augmented dataset (14 * N images and duplicated labels)."""
    np, _ = _require_aug_deps()
    cfg = config or AugmentationConfig()
    rng = np.random.default_rng(cfg.random_seed)

    base = [np.asarray(img) for img in images]

    set_01 = base
    set_02 = random_shear(base, cfg.shear_degrees_range, rng=rng)
    set_03 = random_scale(base, cfg.scale_range, rng=rng)
    set_04 = horizontal_flip(base)
    set_05 = vertical_flip(base)
    set_06 = horizontal_flip(set_02)
    set_07 = horizontal_flip(set_03)
    set_08 = horizontal_flip(set_05)
    set_09 = vertical_flip(set_02)
    set_10 = vertical_flip(set_03)
    set_11 = random_shear(set_07, cfg.shear_degrees_range, rng=rng)
    set_12 = vertical_flip(
        random_shear(
            random_scale(set_04, cfg.scale_range, rng=rng),
            cfg.shear_degrees_range,
            rng=rng,
        )
    )
    set_13 = vertical_flip(
        random_shear(
            random_scale(base, cfg.scale_range, rng=rng),
            cfg.shear_degrees_range,
            rng=rng,
        )
    )
    set_14 = random_shear(set_03, cfg.shear_degrees_range, rng=rng)

    all_sets = [
        set_01,
        set_02,
        set_03,
        set_04,
        set_05,
        set_06,
        set_07,
        set_08,
        set_09,
        set_10,
        set_11,
        set_12,
        set_13,
        set_14,
    ]

    augmented_images = np.concatenate([np.stack(s, axis=0) for s in all_sets], axis=0)
    augmented_labels = np.tile(np.asarray(labels, dtype=np.int64), len(all_sets))
    return augmented_images, augmented_labels
