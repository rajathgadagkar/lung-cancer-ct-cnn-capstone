"""CT scan image preprocessing and marker-controlled watershed segmentation.

Python implementation faithful to the MATLAB preprocessing pipeline in Appendix A
of the PES University Capstone Report:
    1. Median filter (salt-and-pepper noise removal)
    2. Gaussian filter (high-frequency noise suppression, sigma=2)
    3. Adaptive Histogram Equalization / CLAHE (local contrast restoration)
    4. Global Otsu thresholding -> binary mask
    5. Complement + negative Euclidean distance transform
    6. Extended-minima transform + minima imposition -> Watershed segmentation
    7. Composite fusion of watershed-delimited binary mask and enhanced CT slice
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import PreprocessingConfig


def _require_imaging_deps() -> dict[str, Any]:
    try:
        import cv2
        import numpy as np
        from scipy import ndimage
        from skimage import filters, morphology, segmentation
    except ImportError as exc:
        raise ImportError(
            "Image preprocessing requires numpy, opencv-python, scipy, and scikit-image. "
            "Install them with: pip install -r requirements.txt"
        ) from exc
    return {
        "cv2": cv2,
        "np": np,
        "ndimage": ndimage,
        "filters": filters,
        "morphology": morphology,
        "segmentation": segmentation,
    }


@dataclass
class PreprocessingArtifacts:
    """Intermediate and final outputs from the CT preprocessing pipeline."""

    original: Any
    median_filtered: Any
    gaussian_filtered: Any
    clahe_enhanced: Any
    binary_mask: Any
    distance_transform: Any
    watershed_labels: Any
    watershed_binary: Any
    fused_output: Any


def normalize_to_uint8(image: Any) -> Any:
    """Scale an arbitrary numeric 2D array into [0, 255] uint8."""
    deps = _require_imaging_deps()
    np = deps["np"]

    arr = np.asarray(image, dtype=np.float64)
    min_val = float(np.min(arr))
    max_val = float(np.max(arr))
    if max_val <= min_val:
        return np.zeros(arr.shape, dtype=np.uint8)
    scaled = (arr - min_val) / (max_val - min_val)
    return (scaled * 255.0).clip(0, 255).astype(np.uint8)


def preprocess_ct_slice(
    image: Any,
    config: PreprocessingConfig | None = None,
) -> PreprocessingArtifacts:
    """Run the full denoising, contrast enhancement, and watershed pipeline on one 2D CT slice."""
    deps = _require_imaging_deps()
    cv2 = deps["cv2"]
    np = deps["np"]
    ndimage = deps["ndimage"]
    filters = deps["filters"]
    morphology = deps["morphology"]
    segmentation = deps["segmentation"]

    cfg = config or PreprocessingConfig()
    original_u8 = normalize_to_uint8(image)

    # 1. Median filter (salt-and-pepper noise removal)
    ksize = cfg.median_kernel_size
    if ksize % 2 == 0:
        ksize += 1
    median_img = cv2.medianBlur(original_u8, ksize)

    # 2. Gaussian filter (high-frequency noise suppression, sigma=2.0)
    gaussian_img = ndimage.gaussian_filter(
        median_img.astype(np.float64),
        sigma=cfg.gaussian_sigma,
    )
    gaussian_u8 = normalize_to_uint8(gaussian_img)

    # 3. Contrast Limited Adaptive Histogram Equalization (AHE / CLAHE)
    clahe = cv2.createCLAHE(
        clipLimit=cfg.clahe_clip_limit,
        tileGridSize=cfg.clahe_tile_grid_size,
    )
    clahe_img = clahe.apply(gaussian_u8)

    # 4. Global Otsu thresholding
    thresh = filters.threshold_otsu(clahe_img)
    binary_mask = clahe_img > thresh

    # 5. Complement + negative Euclidean distance transform
    distance = -ndimage.distance_transform_edt(~binary_mask)
    distance[~binary_mask] = -np.inf

    # 6. Extended-minima filtering and marker-controlled watershed
    finite_dist = np.where(np.isfinite(distance), distance, 0.0)
    local_min_mask = morphology.h_minima(finite_dist, h=cfg.extended_minima_h)
    markers, _ = ndimage.label(local_min_mask)
    if np.max(markers) == 0:
        markers = ndimage.label(binary_mask)[0]

    watershed_labels = segmentation.watershed(
        finite_dist,
        markers=markers,
        mask=binary_mask,
        watershed_line=True,
    )

    watershed_binary = binary_mask.copy()
    watershed_binary[watershed_labels == 0] = False

    # 7. Difference composite (MATLAB imfuse(..., 'diff') equivalent)
    fused = np.abs(
        watershed_binary.astype(np.float32) - (clahe_img.astype(np.float32) / 255.0)
    )
    fused_resized = cv2.resize(
        fused,
        cfg.output_size,
        interpolation=cv2.INTER_AREA,
    )

    return PreprocessingArtifacts(
        original=original_u8,
        median_filtered=median_img,
        gaussian_filtered=gaussian_u8,
        clahe_enhanced=clahe_img,
        binary_mask=binary_mask,
        distance_transform=distance,
        watershed_labels=watershed_labels,
        watershed_binary=watershed_binary,
        fused_output=fused_resized,
    )


def preprocess_dicom_directory(
    input_dir: Path | str,
    output_dir: Path | str,
    config: PreprocessingConfig | None = None,
) -> int:
    """Batch-preprocess all DICOM slices in ``input_dir`` and save PNG outputs."""
    try:
        import cv2
        import pydicom
    except ImportError as exc:
        raise ImportError(
            "Batch DICOM preprocessing requires pydicom and opencv-python. "
            "Install them with: pip install -r requirements.txt"
        ) from exc

    in_path = Path(input_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    dicom_files = sorted(in_path.glob("*.dcm"))
    for dcm_file in dicom_files:
        ds = pydicom.dcmread(str(dcm_file))
        artifacts = preprocess_ct_slice(ds.pixel_array, config=config)
        out_img = normalize_to_uint8(artifacts.fused_output)
        cv2.imwrite(str(out_path / f"{dcm_file.stem}.png"), out_img)

    return len(dicom_files)
