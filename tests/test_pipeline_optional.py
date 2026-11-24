"""Integration tests for preprocessing, 14x augmentation, and data splitting.

Automatically skipped when optional scientific imaging packages (numpy, cv2, scipy,
scikit-image, scikit-learn) are not installed in the local Python environment.
Executed automatically in GitHub Actions CI.
"""

from __future__ import annotations

import importlib.util
import unittest

HAS_IMAGING = all(
    importlib.util.find_spec(pkg) is not None
    for pkg in ("numpy", "cv2", "scipy", "skimage", "sklearn")
)


@unittest.skipUnless(HAS_IMAGING, "Optional imaging/ML dependencies not installed locally")
class TestImagingAndAugmentationPipeline(unittest.TestCase):
    def test_preprocess_ct_slice_synthetic(self) -> None:
        import numpy as np

        from lung_cancer_cnn.config import PreprocessingConfig
        from lung_cancer_cnn.preprocessing import preprocess_ct_slice

        # Synthetic 128x128 CT-like phantom with lung cavities and a bright nodule
        img = np.zeros((128, 128), dtype=np.uint8)
        yy, xx = np.ogrid[:128, :128]
        chest = ((xx - 64) ** 2 + (yy - 64) ** 2) <= 50**2
        lung_l = ((xx - 44) ** 2 + (yy - 64) ** 2) <= 18**2
        lung_r = ((xx - 84) ** 2 + (yy - 64) ** 2) <= 18**2
        nodule = ((xx - 88) ** 2 + (yy - 68) ** 2) <= 5**2
        img[chest] = 160
        img[lung_l | lung_r] = 20
        img[nodule] = 230

        artifacts = preprocess_ct_slice(img, PreprocessingConfig(output_size=(128, 128)))
        self.assertEqual(artifacts.fused_output.shape, (128, 128))
        self.assertEqual(artifacts.median_filtered.shape, (128, 128))
        self.assertEqual(artifacts.clahe_enhanced.shape, (128, 128))

    def test_augment_dataset_14x_and_split(self) -> None:
        import numpy as np

        from lung_cancer_cnn.augmentation import augment_dataset_14x
        from lung_cancer_cnn.config import DatasetConfig
        from lung_cancer_cnn.data import prepare_train_val_test_splits

        rng = np.random.default_rng(0)
        images = rng.integers(0, 255, size=(10, 64, 64), dtype=np.uint8)
        labels = np.array([0, 1] * 5, dtype=np.int64)

        aug_images, aug_labels = augment_dataset_14x(images, labels)
        self.assertEqual(aug_images.shape, (140, 64, 64))
        self.assertEqual(aug_labels.shape, (140,))

        splits = prepare_train_val_test_splits(
            aug_images, aug_labels, config=DatasetConfig(image_size=64)
        )
        self.assertEqual(splits["x_train"].shape[1:], (64, 64, 1))
        self.assertEqual(splits["y_test"].shape[1], 2)
        self.assertLessEqual(float(np.max(splits["x_train"])), 1.0)


if __name__ == "__main__":
    unittest.main()
