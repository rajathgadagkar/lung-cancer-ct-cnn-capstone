"""End-to-end tests for the lung-cancer-cnn CLI."""

from __future__ import annotations

import io
import json
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from lung_cancer_cnn.cli import main

REPO_ROOT = Path(__file__).resolve().parents[1]


class TestCLI(unittest.TestCase):
    def test_info_json(self) -> None:
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = main(["info", "--csv", str(REPO_ROOT / "data" / "overview.csv"), "--json"])
        self.assertEqual(code, 0)
        data = json.loads(buf.getvalue())
        self.assertEqual(data["author"], "Rajath Gadagkar")
        self.assertEqual(data["email"], "rajathgadagkar@gmail.com")
        self.assertEqual(data["proposed_cnn"]["total_params"], 52_843_458)
        self.assertEqual(data["dataset_summary"]["total_cases"], 100)

    def test_benchmarks_command(self) -> None:
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = main(["benchmarks"])
        self.assertEqual(code, 0)
        out = buf.getvalue()
        self.assertIn("Proposed 12-Layer CNN", out)
        self.assertIn("99.28", out)

    def test_architecture_command(self) -> None:
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = main(["architecture"])
        self.assertEqual(code, 0)
        out = buf.getvalue()
        self.assertIn("52,843,458", out)
        self.assertIn("conv2d_7", out)


if __name__ == "__main__":
    unittest.main()
