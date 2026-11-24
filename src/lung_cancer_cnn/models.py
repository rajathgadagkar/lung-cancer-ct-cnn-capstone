"""Modular CNN architecture definitions and pure-Python layer specifications.

Implements all four deep learning architectures evaluated in Chapter 6 of the
Capstone Report:
  - ``lenet``:    5-layer LeNet baseline (7,391,874 params)
  - ``alexnet``:  8-layer AlexNet with BatchNorm & Dropout (56,305,730 params)
  - ``vgg16``:    16-layer VGG16 baseline (165,724,866 params)
  - ``proposed``: Custom 12-layer CNN with 8 Conv2D, 8 BatchNorm, 4 MaxPool2D,
                  6 Dropout(0.7), and 4 Dense layers (52,843,458 params)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LayerSpec:
    """Pure-Python specification of a single CNN layer for documentation and verification."""

    name: str
    layer_type: str
    output_shape: tuple[int, ...]
    params: int
    details: str


PROPOSED_CNN_SPEC: tuple[LayerSpec, ...] = (
    LayerSpec("conv2d", "Conv2D", (62, 62, 96), 11_712, "96 filters, 11x11, s=4, valid, ReLU"),
    LayerSpec("batch_normalization", "BatchNormalization", (62, 62, 96), 384, "96 channels"),
    LayerSpec("max_pooling2d", "MaxPool2D", (30, 30, 96), 0, "Pool 3x3, stride 2"),
    LayerSpec("conv2d_1", "Conv2D", (30, 30, 256), 614_656, "256 filters, 5x5, s=1, same, ReLU"),
    LayerSpec("batch_normalization_1", "BatchNormalization", (30, 30, 256), 1_024, "256 ch"),
    LayerSpec("max_pooling2d_1", "MaxPool2D", (14, 14, 256), 0, "Pool 3x3, stride 2"),
    LayerSpec("dropout", "Dropout", (14, 14, 256), 0, "Rate = 0.7"),
    LayerSpec("conv2d_2", "Conv2D", (14, 14, 384), 885_120, "384 filters, 3x3, s=1, same, ReLU"),
    LayerSpec("batch_normalization_2", "BatchNormalization", (14, 14, 384), 1_536, "384 ch"),
    LayerSpec("conv2d_3", "Conv2D", (14, 14, 384), 1_327_488, "384 filters, 3x3, s=1, same, ReLU"),
    LayerSpec("batch_normalization_3", "BatchNormalization", (14, 14, 384), 1_536, "384 ch"),
    LayerSpec("conv2d_4", "Conv2D", (14, 14, 256), 98_560, "256 filters, 1x1, s=1, same, ReLU"),
    LayerSpec("batch_normalization_4", "BatchNormalization", (14, 14, 256), 1_024, "256 ch"),
    LayerSpec("max_pooling2d_2", "MaxPool2D", (6, 6, 256), 0, "Pool 3x3, stride 2"),
    LayerSpec("dropout_1", "Dropout", (6, 6, 256), 0, "Rate = 0.7"),
    LayerSpec("conv2d_5", "Conv2D", (6, 6, 512), 3_277_312, "512 filters, 5x5, s=1, same, ReLU"),
    LayerSpec("batch_normalization_5", "BatchNormalization", (6, 6, 512), 2_048, "512 ch"),
    LayerSpec("conv2d_6", "Conv2D", (6, 6, 512), 6_554_112, "512 filters, 5x5, s=1, same, ReLU"),
    LayerSpec("batch_normalization_6", "BatchNormalization", (6, 6, 512), 2_048, "512 ch"),
    LayerSpec("conv2d_7", "Conv2D", (6, 6, 384), 196_992, "384 filters, 1x1, s=1, same, ReLU"),
    LayerSpec("batch_normalization_7", "BatchNormalization", (6, 6, 384), 1_536, "384 ch"),
    LayerSpec("max_pooling2d_3", "MaxPool2D", (2, 2, 384), 0, "Pool 3x3, stride 2"),
    LayerSpec("dropout_2", "Dropout", (2, 2, 384), 0, "Rate = 0.7"),
    LayerSpec("flatten", "Flatten", (1_536,), 0, "Flattens 2x2x384 -> 1536"),
    LayerSpec("dense", "Dense", (4_096,), 6_295_552, "4096 units, ReLU"),
    LayerSpec("dropout_3", "Dropout", (4_096,), 0, "Rate = 0.7"),
    LayerSpec("dense_1", "Dense", (4_096,), 16_781_312, "4096 units, ReLU"),
    LayerSpec("dropout_4", "Dropout", (4_096,), 0, "Rate = 0.7"),
    LayerSpec("dense_2", "Dense", (4_096,), 16_781_312, "4096 units, ReLU"),
    LayerSpec("dropout_5", "Dropout", (4_096,), 0, "Rate = 0.7"),
    LayerSpec("dense_3", "Dense", (2,), 8_194, "2 units, Softmax (Benign vs. Malignant)"),
)


def compute_conv2d_params(kernel_h: int, kernel_w: int, in_channels: int, out_filters: int) -> int:
    """Compute trainable parameter count for a Conv2D layer with bias."""
    return (kernel_h * kernel_w * in_channels + 1) * out_filters


def compute_dense_params(in_features: int, units: int) -> int:
    """Compute trainable parameter count for a Dense layer with bias."""
    return (in_features + 1) * units


def get_proposed_spec_summary() -> dict[str, Any]:
    """Return total parameters and layer counts for the proposed 12-layer CNN."""
    total_params = sum(layer.params for layer in PROPOSED_CNN_SPEC)
    non_trainable = sum(
        layer.params // 2
        for layer in PROPOSED_CNN_SPEC
        if layer.layer_type == "BatchNormalization"
    )
    return {
        "name": "Proposed 12-Layer CNN",
        "conv_layers": sum(1 for layer in PROPOSED_CNN_SPEC if layer.layer_type == "Conv2D"),
        "batchnorm_layers": sum(
            1 for layer in PROPOSED_CNN_SPEC if layer.layer_type == "BatchNormalization"
        ),
        "pooling_layers": sum(1 for layer in PROPOSED_CNN_SPEC if layer.layer_type == "MaxPool2D"),
        "dropout_layers": sum(1 for layer in PROPOSED_CNN_SPEC if layer.layer_type == "Dropout"),
        "dense_layers": sum(1 for layer in PROPOSED_CNN_SPEC if layer.layer_type == "Dense"),
        "total_params": total_params,
        "trainable_params": total_params - non_trainable,
        "non_trainable_params": non_trainable,
    }


def _require_keras() -> Any:
    try:
        from tensorflow import keras
    except ImportError as exc:
        raise ImportError(
            "Building or training CNN models requires TensorFlow/Keras. "
            "Install it with: pip install tensorflow"
        ) from exc
    return keras


def build_lenet(input_shape: tuple[int, int, int] = (256, 256, 1), num_classes: int = 2) -> Any:
    """Build the 5-layer LeNet baseline (7,391,874 params)."""
    keras = _require_keras()
    layers = keras.layers
    return keras.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Conv2D(6, kernel_size=(3, 3), strides=(1, 1), activation="relu"),
            layers.AveragePooling2D(pool_size=(2, 2)),
            layers.Conv2D(16, kernel_size=(3, 3), strides=(1, 1), activation="relu"),
            layers.AveragePooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dense(120, activation="relu"),
            layers.Dense(84, activation="relu"),
            layers.Dense(num_classes, activation="softmax"),
        ],
        name="lenet_baseline",
    )


def build_alexnet(
    input_shape: tuple[int, int, int] = (256, 256, 1),
    num_classes: int = 2,
    dropout_rate: float = 0.5,
) -> Any:
    """Build the 8-layer AlexNet baseline with BatchNormalization (56,305,730 params)."""
    keras = _require_keras()
    layers = keras.layers
    return keras.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Conv2D(96, kernel_size=(11, 11), strides=(4, 4), activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
            layers.Conv2D(
                256, kernel_size=(5, 5), strides=(1, 1), activation="relu", padding="same"
            ),
            layers.BatchNormalization(),
            layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
            layers.Conv2D(
                384, kernel_size=(3, 3), strides=(1, 1), activation="relu", padding="same"
            ),
            layers.BatchNormalization(),
            layers.Conv2D(
                384, kernel_size=(1, 1), strides=(1, 1), activation="relu", padding="same"
            ),
            layers.BatchNormalization(),
            layers.Conv2D(
                256, kernel_size=(1, 1), strides=(1, 1), activation="relu", padding="same"
            ),
            layers.BatchNormalization(),
            layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
            layers.Flatten(),
            layers.Dense(4096, activation="relu"),
            layers.Dropout(dropout_rate),
            layers.Dense(4096, activation="relu"),
            layers.Dropout(dropout_rate),
            layers.Dense(num_classes, activation="softmax"),
        ],
        name="alexnet_baseline",
    )


def build_vgg16(input_shape: tuple[int, int, int] = (256, 256, 1), num_classes: int = 2) -> Any:
    """Build the 16-layer VGG16 baseline (165,724,866 params)."""
    keras = _require_keras()
    layers = keras.layers
    return keras.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            layers.MaxPool2D((2, 2), strides=(2, 2)),
            layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
            layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
            layers.MaxPool2D((2, 2), strides=(2, 2)),
            layers.Conv2D(256, (3, 3), activation="relu", padding="same"),
            layers.Conv2D(256, (3, 3), activation="relu", padding="same"),
            layers.Conv2D(256, (3, 3), activation="relu", padding="same"),
            layers.MaxPool2D((2, 2), strides=(2, 2)),
            layers.Conv2D(512, (3, 3), activation="relu", padding="same"),
            layers.Conv2D(512, (3, 3), activation="relu", padding="same"),
            layers.Conv2D(512, (3, 3), activation="relu", padding="same"),
            layers.MaxPool2D((2, 2), strides=(2, 2)),
            layers.Conv2D(512, (3, 3), activation="relu", padding="same"),
            layers.Conv2D(512, (3, 3), activation="relu", padding="same"),
            layers.Conv2D(512, (3, 3), activation="relu", padding="same"),
            layers.MaxPool2D((2, 2), strides=(2, 2)),
            layers.Flatten(),
            layers.Dense(4096, activation="relu"),
            layers.Dense(4096, activation="relu"),
            layers.Dense(num_classes, activation="softmax"),
        ],
        name="vgg16_baseline",
    )


def build_proposed_cnn(
    input_shape: tuple[int, int, int] = (256, 256, 1),
    num_classes: int = 2,
    dropout_rate: float = 0.7,
) -> Any:
    """Build the proposed 12-layer CNN architecture (52,843,458 params)."""
    keras = _require_keras()
    layers = keras.layers
    return keras.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Conv2D(96, kernel_size=(11, 11), strides=(4, 4), activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
            layers.Conv2D(
                256, kernel_size=(5, 5), strides=(1, 1), activation="relu", padding="same"
            ),
            layers.BatchNormalization(),
            layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
            layers.Dropout(dropout_rate),
            layers.Conv2D(
                384, kernel_size=(3, 3), strides=(1, 1), activation="relu", padding="same"
            ),
            layers.BatchNormalization(),
            layers.Conv2D(
                384, kernel_size=(3, 3), strides=(1, 1), activation="relu", padding="same"
            ),
            layers.BatchNormalization(),
            layers.Conv2D(
                256, kernel_size=(1, 1), strides=(1, 1), activation="relu", padding="same"
            ),
            layers.BatchNormalization(),
            layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
            layers.Dropout(dropout_rate),
            layers.Conv2D(
                512, kernel_size=(5, 5), strides=(1, 1), activation="relu", padding="same"
            ),
            layers.BatchNormalization(),
            layers.Conv2D(
                512, kernel_size=(5, 5), strides=(1, 1), activation="relu", padding="same"
            ),
            layers.BatchNormalization(),
            layers.Conv2D(
                384, kernel_size=(1, 1), strides=(1, 1), activation="relu", padding="same"
            ),
            layers.BatchNormalization(),
            layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
            layers.Dropout(dropout_rate),
            layers.Flatten(),
            layers.Dense(4096, activation="relu"),
            layers.Dropout(dropout_rate),
            layers.Dense(4096, activation="relu"),
            layers.Dropout(dropout_rate),
            layers.Dense(4096, activation="relu"),
            layers.Dropout(dropout_rate),
            layers.Dense(num_classes, activation="softmax"),
        ],
        name="proposed_lung_cancer_cnn",
    )


MODEL_BUILDERS = {
    "lenet": build_lenet,
    "alexnet": build_alexnet,
    "vgg16": build_vgg16,
    "proposed": build_proposed_cnn,
}


def build_model(
    architecture: str = "proposed",
    input_shape: tuple[int, int, int] = (256, 256, 1),
    num_classes: int = 2,
    dropout_rate: float = 0.7,
) -> Any:
    """Factory function to construct any of the four benchmarked CNN architectures."""
    key = architecture.lower().strip()
    if key not in MODEL_BUILDERS:
        valid = ", ".join(sorted(MODEL_BUILDERS))
        raise ValueError(f"Unknown architecture {architecture!r}. Choose from: {valid}")
    builder = MODEL_BUILDERS[key]
    if key in {"alexnet", "proposed"}:
        return builder(input_shape=input_shape, num_classes=num_classes, dropout_rate=dropout_rate)
    return builder(input_shape=input_shape, num_classes=num_classes)


def list_architectures() -> list[str]:
    """Return sorted list of supported CNN architecture identifiers."""
    return sorted(MODEL_BUILDERS.keys())
