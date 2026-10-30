"""Regression checks for incomplete or misleading reproduction comparisons.

Run with: python -m unittest discover -s tests -v
"""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src import compare_results as compare


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.ref = Path(self.tmp.name) / "reference"
        self.run = Path(self.tmp.name) / "run"
        self.ref.mkdir()
        self.run.mkdir()

    def write(self, folder, rel, frame):
        path = folder / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
        return path

    def predictions(self):
        return pd.DataFrame({"filename": ["a.csv", "b.csv"], "y_true": [1, 0],
                             "y_pred": [1, 0], "score": [0.8, 0.1]})

    def complete_main_run(self):
        # Small complete archives let CLI checks distinguish absence from disagreement.
        for _, rel, key, columns, required in compare.TABLES:
            if required:
                self.write(self.ref, rel, pd.DataFrame([
                    {key: "example", **{col: 0.8 for col in columns}}
                ]))
        for _, rel in compare.PREDICTIONS:
            self.write(self.ref, rel, self.predictions())
        model_dir = self.ref / "prauc_protocol/xgboost_prauc"
        (model_dir / "results.json").write_text(json.dumps({"threshold": 0.53}))
        shutil.copytree(self.ref, self.run, dirs_exist_ok=True)

    def cli(self, *args):
        output = io.StringIO()
        argv = ["compare_results.py", "--run", str(self.run), "--reference", str(self.ref), *args]
        with patch("sys.argv", argv), redirect_stdout(output):
            code = compare.main()
        return code, output.getvalue()

    def test_metrics_respect_tolerance(self):
        ref = pd.DataFrame({"model": ["XGB"], "pr_auc": [0.929]})
        run = pd.DataFrame({"model": ["XGB"], "pr_auc": [0.9291]})
        result = compare.compare_frames("Table 3", "results.csv", ref, run, "model", ["pr_auc"], 0.0005)
        self.assertEqual(result[0]["status"], "within tolerance")

    def test_missing_metric_columns_fail_on_either_side(self):
        full = pd.DataFrame({"model": ["XGB"], "pr_auc": [0.929]})
        for side in ("reference", "run"):
            with self.subTest(side=side):
                ref, run = (full.drop(columns="pr_auc"), full) if side == "reference" else (full, full.drop(columns="pr_auc"))
                result = compare.compare_frames("Table 3", "results.csv", ref, run, "model", ["pr_auc"], 0.0005)
                self.assertEqual(result[0]["status"], f"columns missing in {side}")

    def test_missing_and_extra_rows_fail(self):
        full = pd.DataFrame({"model": ["LR", "XGB"], "pr_auc": [0.908, 0.929]})
        short = full.iloc[:1]
        for ref, run, status in ((full, short, "row only in reference"), (short, full, "row only in run")):
            with self.subTest(status=status):
                result = compare.compare_frames("Table 3", "results.csv", ref, run, "model", ["pr_auc"], 0.0005)
                self.assertIn(status, [r["status"] for r in result])

    def test_empty_or_duplicate_tables_fail(self):
        frame = pd.DataFrame({"model": ["XGB"], "pr_auc": [0.929]})
        for invalid, status in ((frame.iloc[:0], "empty in run"),
                                (pd.concat([frame, frame]), "invalid keys in run")):
            with self.subTest(status=status):
                result = compare.compare_frames("Table 3", "results.csv", frame, invalid, "model", ["pr_auc"], 0.0005)
                self.assertEqual(result[0]["status"], status)

    def test_changed_true_labels_fail_even_if_predictions_agree(self):
        rel = compare.PREDICTIONS[0][1]
        frame = self.predictions()
        self.write(self.ref, rel, frame)
        frame.loc[0, "y_true"] = 0
        self.write(self.run, rel, frame)
        result = compare.compare_predictions(self.ref, self.run, 0.0005)
        self.assertEqual(result[0]["status"], "different")
        self.assertIn("true labels differing: 1", result[0]["column"])

    def test_duplicate_prediction_tracks_fail(self):
        rel = compare.PREDICTIONS[0][1]
        frame = self.predictions()
        self.write(self.ref, rel, frame)
        self.write(self.run, rel, pd.concat([frame, frame.iloc[:1]]))
        result = compare.compare_predictions(self.ref, self.run, 0.0005)
        self.assertEqual(result[0]["status"], "invalid keys in run")

    def test_missing_or_nonfinite_prediction_scores_fail(self):
        rel = compare.PREDICTIONS[0][1]
        frame = self.predictions()
        self.write(self.ref, rel, frame)
        for invalid in (frame.drop(columns="score"), frame.assign(score=[np.nan, 0.1])):
            with self.subTest(columns=list(invalid.columns)):
                self.write(self.run, rel, invalid)
                result = compare.compare_predictions(self.ref, self.run, 0.0005)
                self.assertNotIn(result[0]["status"], ["identical", "within tolerance"])

    def test_missing_feature_columns_and_changed_labels_fail(self):
        frame = pd.DataFrame({"filename": ["a.csv"], "label": [1], "mean_speed": [4.0]})
        self.write(self.ref, "track_features.csv", frame)
        for invalid in (frame.drop(columns="mean_speed"), frame.assign(label=0)):
            with self.subTest(columns=list(invalid.columns)):
                self.write(self.run, "track_features.csv", invalid)
                result = compare.compare_features(self.ref, self.run)
                self.assertNotEqual(result[0]["status"], "identical")

    def test_matching_missing_features_and_relative_roundoff_are_allowed(self):
        frame = pd.DataFrame({"filename": ["a.csv", "b.csv"], "label": [1, 0], "mean_speed": [4.0, np.nan]})
        self.write(self.ref, "track_features.csv", frame)
        frame.loc[0, "mean_speed"] += 1e-10
        self.write(self.run, "track_features.csv", frame)
        self.assertEqual(compare.compare_features(self.ref, self.run)[0]["status"], "identical")

    def test_changed_folds_fail(self):
        frame = pd.DataFrame({"filename": ["a.csv"], "fold": [4]})
        ref_csv = self.write(self.ref, "splits.csv", frame)
        run_csv = self.write(self.run, "splits.csv", frame.assign(fold=3))
        self.assertEqual(compare.compare_splits(ref_csv, run_csv)[0]["status"], "different")

    def test_complete_main_run_matches_without_separate_studies(self):
        self.complete_main_run()
        code, output = self.cli()
        self.assertEqual(code, 0)
        self.assertIn("Result: MATCH", output)
        self.assertIn("Tables 9 and 11 are not requested", output)

    def test_missing_required_file_in_reference_or_run_fails(self):
        self.complete_main_run()
        for folder in (self.ref, self.run):
            path = folder / "cross_farm/cross_farm_results.csv"
            content = path.read_bytes()
            path.unlink()
            with self.subTest(folder=folder.name):
                code, output = self.cli()
                self.assertEqual(code, 1)
                self.assertIn("Result: INCOMPLETE", output)
                self.assertIn("cross_farm/cross_farm_results.csv", output)
            path.write_bytes(content)

    def test_missing_prediction_file_cannot_match(self):
        self.complete_main_run()
        (self.run / compare.PREDICTIONS[0][1]).unlink()
        code, output = self.cli()
        self.assertEqual(code, 1)
        self.assertNotIn("Result: MATCH", output)

    def test_empty_archives_fail_without_crashing(self):
        code, output = self.cli()
        self.assertEqual(code, 1)
        self.assertIn("Result: INCOMPLETE", output)

    def test_requested_features_must_exist(self):
        self.complete_main_run()
        code, output = self.cli("--features")
        self.assertEqual(code, 1)
        self.assertIn("track_features.csv: missing in", output)

    def test_supplementary_studies_are_required_when_requested(self):
        self.complete_main_run()
        code, output = self.cli("--include-supplementary")
        self.assertEqual(code, 1)
        self.assertIn("Supp. Table 9", output)
        self.assertIn("Supp. Table 11", output)
        for folder in (self.ref, self.run):
            self.write(folder, "cross_farm_threshold_study/cross_farm_threshold_summary.csv",
                       pd.DataFrame({"evaluation": ["A_to_B"], "deployed_f1": [0.8],
                                     "local_optimal_threshold": [0.5], "local_optimal_f1": [0.8],
                                     "f1_gap_deployed_vs_local_optimal": [0.0]}))
            self.write(folder, "convtran_batch_size_study/convtran_batch_size_summary.csv",
                       pd.DataFrame({"batch_size": [128], "pr_auc": [0.85], "pr_auc_lo": [0.82],
                                     "pr_auc_hi": [0.88], "f1": [0.78], "selected_epochs": [27]}))
        code, output = self.cli("--include-supplementary")
        self.assertEqual(code, 0)
        self.assertIn("Result: MATCH", output)

    def test_unpaired_split_flag_or_invalid_tolerance_is_rejected(self):
        for args in (("--splits-run", "splits.csv"), ("--tol", "-1"), ("--tol", "nan")):
            with self.subTest(args=args), patch("sys.stderr", io.StringIO()):
                with self.assertRaises(SystemExit) as err:
                    self.cli(*args)
                self.assertEqual(err.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
