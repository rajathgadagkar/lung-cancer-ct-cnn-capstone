"""Dataset metadata parsing, DICOM loading, and 80/10/10 dataset splitting."""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .config import DatasetConfig

FILENAME_PATTERN = re.compile(
    r"ID_(?P<case_id>\d+)_AGE_(?P<age>\d+)_CONTRAST_(?P<label>[01])_CT"
)


@dataclass(frozen=True)
class CaseRecord:
    """Single CT scan metadata record parsed from ``overview.csv`` or DICOM filename."""

    case_id: int
    age: int
    contrast: bool
    label: int
    dicom_name: str
    tiff_name: str = ""
    contrast_tag: str = ""


def parse_ct_filename(filename: str) -> Tuple[int, int, int]:
    """Extract ``(case_id, age, label)`` from a Kaggle Lung CT Challenge filename.

    Example:
        ``ID_0000_AGE_0060_CONTRAST_1_CT.dcm`` -> ``(0, 60, 1)``
    """
    match = FILENAME_PATTERN.search(filename)
    if not match:
        raise ValueError(
            f"Filename {filename!r} does not match expected pattern "
            "'ID_<id>_AGE_<age>_CONTRAST_<0|1>_CT'."
        )
    return (
        int(match.group("case_id")),
        int(match.group("age")),
        int(match.group("label")),
    )


def load_overview_metadata(csv_path: Path | str) -> List[CaseRecord]:
    """Load and parse ``overview.csv`` using only Python's standard library."""
    path = Path(csv_path)
    records: List[CaseRecord] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            contrast_bool = str(row["Contrast"]).strip().lower() in {"true", "1", "yes"}
            records.append(
                CaseRecord(
                    case_id=int(row["id"]),
                    age=int(row["Age"]),
                    contrast=contrast_bool,
                    label=1 if contrast_bool else 0,
                    dicom_name=row["dicom_name"].strip(),
                    tiff_name=row.get("tiff_name", "").strip(),
                    contrast_tag=row.get("ContrastTag", "").strip(),
                )
            )
    return records


def summarize_overview_metadata(records: List[CaseRecord]) -> Dict[str, Any]:
    """Compute descriptive summary statistics for the CT dataset metadata."""
    if not records:
        return {
            "total_cases": 0,
            "positive_cases": 0,
            "negative_cases": 0,
            "age_min": None,
            "age_max": None,
            "age_mean": None,
            "age_60_plus_pct": 0.0,
        }
    ages = [r.age for r in records]
    positives = sum(1 for r in records if r.label == 1)
    negatives = len(records) - positives
    age_60_plus = sum(1 for a in ages if a >= 60)
    return {
        "total_cases": len(records),
        "positive_cases": positives,
        "negative_cases": negatives,
        "age_min": min(ages),
        "age_max": max(ages),
        "age_mean": round(sum(ages) / len(ages), 2),
        "age_60_plus_pct": round(100.0 * age_60_plus / len(ages), 1),
    }


def load_dicom_dataset(
    dicom_dir: Path | str,
    image_size: int = 256,
) -> Tuple[Any, Any, List[str]]:
    """Load preprocessed ``.dcm`` slices, resize to ``(image_size, image_size)``, and extract labels."""
    try:
        import cv2
        import numpy as np
        import pydicom
    except ImportError as exc:
        raise ImportError(
            "Loading DICOM datasets requires numpy, opencv-python, and pydicom. "
            "Install them with: pip install -r requirements.txt"
        ) from exc

    dcm_paths = sorted(Path(dicom_dir).glob("*.dcm"))
    if not dcm_paths:
        raise FileNotFoundError(f"No .dcm files found in {dicom_dir}")

    images: List[Any] = []
    labels: List[int] = []
    filenames: List[str] = []

    for path in dcm_paths:
        _, _, label = parse_ct_filename(path.name)
        pixel_array = pydicom.dcmread(str(path)).pixel_array
        resized = cv2.resize(pixel_array, (image_size, image_size))
        images.append(resized)
        labels.append(label)
        filenames.append(path.name)

    return np.asarray(images), np.asarray(labels, dtype=np.int64), filenames


def prepare_train_val_test_splits(
    images: Any,
    labels: Any,
    config: Optional[DatasetConfig] = None,
) -> Dict[str, Any]:
    """Split augmented images into normalized float32 80% train, 10% val, 10% test tensors."""
    try:
        import numpy as np
        from sklearn.model_selection import train_test_split
    except ImportError as exc:
        raise ImportError(
            "Dataset splitting requires numpy and scikit-learn. "
            "Install them with: pip install -r requirements.txt"
        ) from exc

    cfg = config or DatasetConfig()

    x_train_val, x_test, y_train_val, y_test = train_test_split(
        images,
        labels,
        test_size=cfg.test_size,
        random_state=cfg.test_random_state,
        stratify=labels,
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_val,
        y_train_val,
        test_size=cfg.val_fraction_of_trainval,
        random_state=cfg.val_random_state,
        stratify=y_train_val,
    )

    def _normalize_reshape(arr: Any) -> Any:
        out = arr.astype("float32").reshape((arr.shape[0], *cfg.input_shape))
        max_val = float(np.max(out))
        return out / max_val if max_val > 0 else out

    def _one_hot(y: Any) -> Any:
        eye = np.eye(cfg.num_classes, dtype="float32")
        return eye[np.asarray(y, dtype=np.int64)]

    return {
        "x_train": _normalize_reshape(x_train),
        "y_train": _one_hot(y_train),
        "x_val": _normalize_reshape(x_val),
        "y_val": _one_hot(y_val),
        "x_test": _normalize_reshape(x_test),
        "y_test": _one_hot(y_test),
    }
