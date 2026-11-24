"""Lung Cancer Detection and Classification using Image Processing and CNN.

Official B.Tech Capstone Project — Department of Electronics & Communication Engineering,
PES University, Bengaluru (August 2020 – May 2021).

Author: Rajath Gadagkar <rajathgadagkar@gmail.com>
License: MIT
"""

from .config import (
    BENCHMARK_RESULTS,
    AugmentationConfig,
    DatasetConfig,
    ModelConfig,
    PreprocessingConfig,
    TrainingConfig,
)

__version__ = "1.0.0"
__author__ = "Rajath Gadagkar"
__email__ = "rajathgadagkar@gmail.com"
__license__ = "MIT"

__all__ = [
    "__version__",
    "__author__",
    "__email__",
    "__license__",
    "BENCHMARK_RESULTS",
    "PreprocessingConfig",
    "AugmentationConfig",
    "DatasetConfig",
    "ModelConfig",
    "TrainingConfig",
]
