"""Configuration dataclasses and verified capstone benchmark constants."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PreprocessingConfig:
    """Parameters for CT scan denoising, contrast enhancement, and watershed segmentation."""

    median_kernel_size: int = 3
    gaussian_sigma: float = 2.0
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: tuple[int, int] = (8, 8)
    extended_minima_h: float = 2.0
    output_size: tuple[int, int] = (512, 512)


@dataclass(frozen=True)
class AugmentationConfig:
    """Parameters for the 14-fold deterministic CT data augmentation pipeline."""

    scale_range: tuple[float, float] = (0.9, 1.0)
    shear_degrees_range: tuple[float, float] = (0.0, 20.0)
    random_seed: int = 42
    fold_multiplier: int = 14


@dataclass(frozen=True)
class DatasetConfig:
    """Dataset dimensions and 80/10/10 train/validation/test split configuration."""

    image_size: int = 256
    channels: int = 1
    num_classes: int = 2
    test_size: float = 0.10
    val_fraction_of_trainval: float = 0.111  # 0.111 * 0.90 ≈ 0.10 of total dataset
    test_random_state: int = 23
    val_random_state: int = 21
    class_names: tuple[str, str] = ("Benign / Non-Contrast (0)", "Malignant / Contrast (1)")

    @property
    def input_shape(self) -> tuple[int, int, int]:
        return (self.image_size, self.image_size, self.channels)


@dataclass(frozen=True)
class ModelConfig:
    """Architecture selection and regularization hyperparameters."""

    architecture: str = "proposed"
    input_shape: tuple[int, int, int] = (256, 256, 1)
    num_classes: int = 2
    dropout_rate: float = 0.7


@dataclass(frozen=True)
class TrainingConfig:
    """Model optimization and training schedule hyperparameters."""

    batch_size: int = 20
    epochs: int = 100
    learning_rate: float = 1e-3
    early_stopping_patience: int = 30
    early_stopping_monitor: str = "val_accuracy"
    loss: str = "binary_crossentropy"
    metrics: list[str] = field(default_factory=lambda: ["accuracy"])


@dataclass(frozen=True)
class BenchmarkRecord:
    """Verified evaluation metrics reported in Table 7.1 of the Capstone Report."""

    model_name: str
    layers: int
    total_params: int
    test_accuracy: float
    precision_class_0: float
    precision_class_1: float
    recall_class_0: float
    recall_class_1: float
    f1_class_0: float
    f1_class_1: float
    train_loss: float
    val_loss: float
    wrong_predictions_out_of_140: int
    early_stopped_epoch: int


BENCHMARK_RESULTS: dict[str, BenchmarkRecord] = {
    "lenet": BenchmarkRecord(
        model_name="LeNet",
        layers=5,
        total_params=7_391_874,
        test_accuracy=88.57,
        precision_class_0=0.84,
        precision_class_1=0.91,
        recall_class_0=0.91,
        recall_class_1=0.85,
        f1_class_0=0.87,
        f1_class_1=0.88,
        train_loss=2.4172e-05,
        val_loss=0.9127,
        wrong_predictions_out_of_140=17,
        early_stopped_epoch=38,
    ),
    "alexnet": BenchmarkRecord(
        model_name="AlexNet",
        layers=8,
        total_params=56_305_730,
        test_accuracy=89.99,
        precision_class_0=0.88,
        precision_class_1=0.92,
        recall_class_0=0.91,
        recall_class_1=0.89,
        f1_class_0=0.89,
        f1_class_1=0.91,
        train_loss=0.2752,
        val_loss=0.9657,
        wrong_predictions_out_of_140=14,
        early_stopped_epoch=38,
    ),
    "vgg16": BenchmarkRecord(
        model_name="VGG16",
        layers=16,
        total_params=165_724_866,
        test_accuracy=51.43,
        precision_class_0=0.46,
        precision_class_1=0.00,
        recall_class_0=1.00,
        recall_class_1=0.00,
        f1_class_0=0.63,
        f1_class_1=0.00,
        train_loss=0.6931,
        val_loss=0.6930,
        wrong_predictions_out_of_140=75,
        early_stopped_epoch=51,
    ),
    "proposed": BenchmarkRecord(
        model_name="Proposed 12-Layer CNN",
        layers=12,
        total_params=52_843_458,
        test_accuracy=99.28,
        precision_class_0=0.98,
        precision_class_1=1.00,
        recall_class_0=1.00,
        recall_class_1=0.99,
        f1_class_0=0.99,
        f1_class_1=0.99,
        train_loss=0.1812,
        val_loss=0.0235,
        wrong_predictions_out_of_140=1,
        early_stopped_epoch=100,
    ),
}
