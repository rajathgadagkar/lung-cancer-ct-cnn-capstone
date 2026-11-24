"""Command-line interface for the Lung Cancer Detection CNN Capstone Project."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __author__, __email__, __version__
from .config import DatasetConfig, ModelConfig, PreprocessingConfig, TrainingConfig
from .data import load_overview_metadata, summarize_overview_metadata
from .evaluate import format_benchmark_table
from .models import PROPOSED_CNN_SPEC, get_proposed_spec_summary, list_architectures


def _cmd_info(args: argparse.Namespace) -> int:
    payload = {
        "project": "Lung Cancer Detection and Classification using Image Processing and CNN",
        "institution": "PES University, Bengaluru — Department of ECE (B.Tech Capstone, 2020–2021)",
        "author": __author__,
        "email": __email__,
        "version": __version__,
        "architectures": list_architectures(),
        "proposed_cnn": get_proposed_spec_summary(),
    }
    if args.csv and Path(args.csv).exists():
        payload["dataset_summary"] = summarize_overview_metadata(load_overview_metadata(args.csv))

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"{payload['project']} (v{__version__})")
        print(f"Institution : {payload['institution']}")
        print(f"Author      : {__author__} <{__email__}>")
        spec = payload["proposed_cnn"]
        print(
            f"Proposed CNN: {spec['conv_layers']} Conv2D + {spec['batchnorm_layers']} BatchNorm + "
            f"{spec['pooling_layers']} MaxPool2D + {spec['dropout_layers']} Dropout(0.7) + "
            f"{spec['dense_layers']} Dense = {spec['total_params']:,} params"
        )
        if "dataset_summary" in payload:
            ds = payload["dataset_summary"]
            print(
                f"Dataset CSV : {ds['total_cases']} cases "
                f"({ds['positive_cases']} positive / {ds['negative_cases']} negative, "
                f"ages {ds['age_min']}–{ds['age_max']}, mean {ds['age_mean']})"
            )
    return 0


def _cmd_benchmarks(_args: argparse.Namespace) -> int:
    print(format_benchmark_table())
    return 0


def _cmd_architecture(_args: argparse.Namespace) -> int:
    summary = get_proposed_spec_summary()
    print(f"Architecture: {summary['name']}")
    print(f"{'Layer':<24} {'Type':<20} {'Output Shape':<16} {'Params':>12}  Details")
    print("-" * 100)
    for layer in PROPOSED_CNN_SPEC:
        shape_str = str(layer.output_shape)
        print(
            f"{layer.name:<24} {layer.layer_type:<20} {shape_str:<16} "
            f"{layer.params:>12,}  {layer.details}"
        )
    print("-" * 100)
    print(
        f"Total params: {summary['total_params']:,} | "
        f"Trainable: {summary['trainable_params']:,} | "
        f"Non-trainable: {summary['non_trainable_params']:,}"
    )
    return 0


def _cmd_preprocess(args: argparse.Namespace) -> int:
    from .preprocessing import preprocess_dicom_directory

    cfg = PreprocessingConfig(
        median_kernel_size=args.median_kernel,
        gaussian_sigma=args.gaussian_sigma,
    )
    count = preprocess_dicom_directory(args.input_dir, args.output_dir, config=cfg)
    print(f"Preprocessed {count} DICOM scan(s) -> {args.output_dir}")
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    from .train import train_pipeline

    result = train_pipeline(
        dicom_dir=args.dicom_dir,
        output_dir=args.output_dir,
        model_config=ModelConfig(architecture=args.arch, dropout_rate=args.dropout),
        dataset_config=DatasetConfig(image_size=args.image_size),
        training_config=TrainingConfig(batch_size=args.batch_size, epochs=args.epochs),
    )
    print(json.dumps(result["test_metrics"], indent=2))
    print(f"Saved trained model to: {result['model_path']}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lung-cancer-cnn",
        description="Lung Cancer Detection & Classification using 12-Layer CNN",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_info = sub.add_parser("info", help="Display project, model, and dataset metadata")
    p_info.add_argument("--csv", default="data/overview.csv", help="Path to overview.csv")
    p_info.add_argument("--json", action="store_true", help="Output as JSON")
    p_info.set_defaults(func=_cmd_info)

    p_bench = sub.add_parser("benchmarks", help="Print capstone model benchmark table")
    p_bench.set_defaults(func=_cmd_benchmarks)

    p_arch = sub.add_parser("architecture", help="Print layer-by-layer Proposed 12-Layer CNN spec")
    p_arch.set_defaults(func=_cmd_architecture)

    p_prep = sub.add_parser("preprocess", help="Run denoising and watershed segmentation on DICOMs")
    p_prep.add_argument("--input-dir", required=True, help="Directory containing raw .dcm scans")
    p_prep.add_argument("--output-dir", required=True, help="Directory to save preprocessed images")
    p_prep.add_argument("--median-kernel", type=int, default=3, help="Median filter kernel size")
    p_prep.add_argument("--gaussian-sigma", type=float, default=2.0, help="Gaussian filter sigma")
    p_prep.set_defaults(func=_cmd_preprocess)

    p_train = sub.add_parser("train", help="Train a benchmarked CNN architecture on DICOM scans")
    p_train.add_argument(
        "--dicom-dir", required=True, help="Directory with preprocessed .dcm files"
    )
    p_train.add_argument("--output-dir", default="outputs", help="Directory for model & plots")
    p_train.add_argument(
        "--arch",
        default="proposed",
        choices=list_architectures(),
        help="CNN architecture to train",
    )
    p_train.add_argument("--epochs", type=int, default=100, help="Maximum training epochs")
    p_train.add_argument("--batch-size", type=int, default=20, help="Mini-batch size")
    p_train.add_argument("--image-size", type=int, default=256, help="Input spatial resolution")
    p_train.add_argument("--dropout", type=float, default=0.7, help="Dropout rate")
    p_train.set_defaults(func=_cmd_train)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
