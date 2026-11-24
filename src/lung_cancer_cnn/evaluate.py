"""Evaluation metrics, classification reporting, and training history visualization."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from .config import BENCHMARK_RESULTS


def format_benchmark_table() -> str:
    """Format the verified capstone benchmark comparison table in Markdown."""
    header = (
        "| Model | Layers | Parameters | Accuracy (%) | Precision (0/1) | "
        "Recall (0/1) | F1-Score (0/1) | Train Loss | Val Loss | Errors (/140) |"
    )
    sep = "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    rows = [header, sep]
    for key in ("lenet", "alexnet", "vgg16", "proposed"):
        r = BENCHMARK_RESULTS[key]
        rows.append(
            f"| {r.model_name} | {r.layers} | {r.total_params:,} | {r.test_accuracy:.2f} | "
            f"{r.precision_class_0:.2f} / {r.precision_class_1:.2f} | "
            f"{r.recall_class_0:.2f} / {r.recall_class_1:.2f} | "
            f"{r.f1_class_0:.2f} / {r.f1_class_1:.2f} | "
            f"{r.train_loss:.4g} | {r.val_loss:.4f} | {r.wrong_predictions_out_of_140} |"
        )
    return "\n".join(rows)


def evaluate_model(model: Any, x_test: Any, y_test: Any) -> Dict[str, Any]:
    """Evaluate a trained Keras model on a test split and compute classification metrics."""
    try:
        import numpy as np
        from sklearn.metrics import classification_report, confusion_matrix
    except ImportError as exc:
        raise ImportError(
            "Model evaluation requires numpy and scikit-learn. "
            "Install them with: pip install -r requirements.txt"
        ) from exc

    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    probabilities = model.predict(x_test, verbose=0)
    y_pred = np.argmax(probabilities, axis=1)
    y_true = np.argmax(y_test, axis=1)
    wrong_predictions = int(np.sum(y_pred != y_true))

    return {
        "loss": float(loss),
        "accuracy": float(accuracy),
        "wrong_predictions": wrong_predictions,
        "total_samples": int(len(y_true)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, labels=[0, 1], output_dict=True
        ),
        "classification_report_text": classification_report(y_true, y_pred, labels=[0, 1]),
    }


def save_training_curves(
    history: Dict[str, Any],
    output_dir: Path | str,
    prefix: str = "proposed",
) -> Optional[Path]:
    """Save accuracy and loss curves from a Keras training history dictionary."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "Plotting training curves requires matplotlib. "
            "Install it with: pip install -r requirements.txt"
        ) from exc

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    fig_path = out_path / f"{prefix}_training_curves.png"

    fig, (ax_acc, ax_loss) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax_acc.plot(history.get("accuracy", []), label="Train")
    ax_acc.plot(history.get("val_accuracy", []), label="Validation")
    ax_acc.set_title("Model Accuracy")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.legend(loc="lower right")
    ax_acc.grid(alpha=0.3)

    ax_loss.plot(history.get("loss", []), label="Train")
    ax_loss.plot(history.get("val_loss", []), label="Validation")
    ax_loss.set_title("Model Loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.legend(loc="upper right")
    ax_loss.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    return fig_path
