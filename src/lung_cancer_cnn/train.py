"""End-to-end training pipeline for lung CT cancer classification."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from .augmentation import augment_dataset_14x
from .config import AugmentationConfig, DatasetConfig, ModelConfig, TrainingConfig
from .data import load_dicom_dataset, prepare_train_val_test_splits
from .evaluate import evaluate_model, save_training_curves
from .models import build_model


def train_pipeline(
    dicom_dir: Path | str,
    output_dir: Path | str = "outputs",
    model_config: Optional[ModelConfig] = None,
    dataset_config: Optional[DatasetConfig] = None,
    augmentation_config: Optional[AugmentationConfig] = None,
    training_config: Optional[TrainingConfig] = None,
) -> Dict[str, Any]:
    """Load preprocessed DICOM scans, apply 14x augmentation, train CNN, and evaluate."""
    try:
        from tensorflow import keras
    except ImportError as exc:
        raise ImportError(
            "Training requires TensorFlow/Keras. Install it with: pip install tensorflow"
        ) from exc

    m_cfg = model_config or ModelConfig()
    d_cfg = dataset_config or DatasetConfig()
    a_cfg = augmentation_config or AugmentationConfig()
    t_cfg = training_config or TrainingConfig()

    images, labels, _ = load_dicom_dataset(dicom_dir, image_size=d_cfg.image_size)
    aug_images, aug_labels = augment_dataset_14x(images, labels, config=a_cfg)
    splits = prepare_train_val_test_splits(aug_images, aug_labels, config=d_cfg)

    model = build_model(
        architecture=m_cfg.architecture,
        input_shape=m_cfg.input_shape,
        num_classes=m_cfg.num_classes,
        dropout_rate=m_cfg.dropout_rate,
    )
    model.compile(
        loss=t_cfg.loss,
        optimizer=keras.optimizers.Adam(learning_rate=t_cfg.learning_rate),
        metrics=list(t_cfg.metrics),
    )

    early_stopping = keras.callbacks.EarlyStopping(
        monitor=t_cfg.early_stopping_monitor,
        min_delta=0,
        patience=t_cfg.early_stopping_patience,
        verbose=1,
        mode="auto",
        restore_best_weights=True,
    )

    history = model.fit(
        splits["x_train"],
        splits["y_train"],
        batch_size=t_cfg.batch_size,
        epochs=t_cfg.epochs,
        validation_data=(splits["x_val"], splits["y_val"]),
        callbacks=[early_stopping],
        verbose=1,
    )

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    model_path = out_path / f"{m_cfg.architecture}_lung_cancer_cnn.keras"
    model.save(model_path)

    metrics = evaluate_model(model, splits["x_test"], splits["y_test"])
    curves_path = save_training_curves(history.history, out_path, prefix=m_cfg.architecture)

    return {
        "model_path": str(model_path),
        "curves_path": str(curves_path) if curves_path else None,
        "test_metrics": metrics,
        "epochs_trained": len(history.history.get("loss", [])),
    }
