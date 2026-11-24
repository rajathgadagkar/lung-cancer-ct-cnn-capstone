"""Unit tests for dataset filename parsing and overview.csv metadata loading."""

from __future__ import annotations

import unittest
from pathlib import Path

from lung_cancer_cnn.data import (
    load_overview_metadata,
    parse_ct_filename,
    summarize_overview_metadata,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
OVERVIEW_CSV = REPO_ROOT / "data" / "overview.csv"


class TestDataUtils(unittest.TestCase):
    def test_parse_ct_filename_valid(self) -> None:
        case_id, age, label = parse_ct_filename("ID_0000_AGE_0060_CONTRAST_1_CT.dcm")
        self.assertEqual((case_id, age, label), (0, 60, 1))

        case_id2, age2, label2 = parse_ct_filename("/tmp/ID_0099_AGE_0074_CONTRAST_0_CT.dcm")
        self.assertEqual((case_id2, age2, label2), (99, 74, 0))

    def test_parse_ct_filename_invalid_raises(self) -> None:
        with self.assertRaises(ValueError):
            parse_ct_filename("random_scan.dcm")

    def test_load_and_summarize_overview_csv(self) -> None:
        self.assertTrue(OVERVIEW_CSV.exists(), f"Missing dataset CSV: {OVERVIEW_CSV}")
        records = load_overview_metadata(OVERVIEW_CSV)
        summary = summarize_overview_metadata(records)

        self.assertEqual(summary["total_cases"], 100)
        self.assertEqual(summary["positive_cases"], 50)
        self.assertEqual(summary["negative_cases"], 50)
        self.assertEqual(summary["age_min"], 39)
        self.assertEqual(summary["age_max"], 83)
        self.assertAlmostEqual(summary["age_mean"], 67.5, places=1)


if __name__ == "__main__":
    unittest.main()
