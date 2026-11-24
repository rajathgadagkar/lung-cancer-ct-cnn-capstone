"""Unit tests for configuration dataclasses, benchmark records, and CNN layer math."""

from __future__ import annotations

import unittest

from lung_cancer_cnn.config import (
    BENCHMARK_RESULTS,
    AugmentationConfig,
    DatasetConfig,
    ModelConfig,
    PreprocessingConfig,
    TrainingConfig,
)
from lung_cancer_cnn.evaluate import format_benchmark_table
from lung_cancer_cnn.models import (
    PROPOSED_CNN_SPEC,
    compute_conv2d_params,
    compute_dense_params,
    get_proposed_spec_summary,
    list_architectures,
)


class TestConfigAndSpecs(unittest.TestCase):
    def test_default_configs_match_capstone_report(self) -> None:
        prep = PreprocessingConfig()
        aug = AugmentationConfig()
        ds = DatasetConfig()
        model = ModelConfig()
        train = TrainingConfig()

        self.assertEqual(prep.gaussian_sigma, 2.0)
        self.assertEqual(prep.output_size, (512, 512))
        self.assertEqual(aug.fold_multiplier, 14)
        self.assertEqual(ds.input_shape, (256, 256, 1))
        self.assertEqual(model.dropout_rate, 0.7)
        self.assertEqual(train.batch_size, 20)
        self.assertEqual(train.epochs, 100)

    def test_proposed_cnn_parameter_count_matches_report_fig_6_9(self) -> None:
        summary = get_proposed_spec_summary()
        self.assertEqual(summary["conv_layers"], 8)
        self.assertEqual(summary["batchnorm_layers"], 8)
        self.assertEqual(summary["pooling_layers"], 4)
        self.assertEqual(summary["dropout_layers"], 6)
        self.assertEqual(summary["dense_layers"], 4)
        self.assertEqual(summary["total_params"], 52_843_458)
        self.assertEqual(summary["trainable_params"], 52_837_890)
        self.assertEqual(summary["non_trainable_params"], 5_568)

    def test_individual_layer_parameter_formulas(self) -> None:
        # Conv1: 11x11, 1 channel -> 96 filters
        self.assertEqual(compute_conv2d_params(11, 11, 1, 96), 11_712)
        # Conv2: 5x5, 96 channels -> 256 filters
        self.assertEqual(compute_conv2d_params(5, 5, 96, 256), 614_656)
        # Conv8: 1x1, 512 channels -> 384 filters
        self.assertEqual(compute_conv2d_params(1, 1, 512, 384), 196_992)
        # Dense1: 1536 -> 4096
        self.assertEqual(compute_dense_params(1_536, 4_096), 6_295_552)
        # Output Dense: 4096 -> 2
        self.assertEqual(compute_dense_params(4_096, 2), 8_194)
        self.assertEqual(PROPOSED_CNN_SPEC[-1].output_shape, (2,))

    def test_benchmark_records_and_markdown_table(self) -> None:
        self.assertEqual(set(list_architectures()), {"alexnet", "lenet", "proposed", "vgg16"})
        proposed = BENCHMARK_RESULTS["proposed"]
        self.assertAlmostEqual(proposed.test_accuracy, 99.28)
        self.assertEqual(proposed.total_params, 52_843_458)
        table = format_benchmark_table()
        self.assertIn("Proposed 12-Layer CNN", table)
        self.assertIn("99.28", table)
        self.assertIn("VGG16", table)


if __name__ == "__main__":
    unittest.main()
