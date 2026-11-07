"""CPU-only fixtures for the standalone validator; never rebuild real data."""

import csv
import json
import tempfile
import unittest
from pathlib import Path

from dataset.validate_dataset import FILES, REQUIRED, main, validate_dataset


class DatasetValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.processed = self.root / "processed"
        self.rows = []
        for word, label, score in [("alpha", "[1, 3]", "2"), ("beta", "[1, 3]", "3"),
                                   ("gamma", "(3, 5]", "4"), ("delta", "(3, 5]", "5"),
                                   ("epsilon", "(5, 7]", "6"), ("zeta", "(5, 7]", "7")]:
            self.rows.append(dict(word=word, length=str(len(word)), pos="Noun", pos_share="1.0",
                                 score_bin=label, glasgow_imageability_mean=score,
                                 glasgow_imageability_sd="1.2", glasgow_imageability_n="35"))
        self.write("noun_words.csv", self.rows)
        self.write("sampled_words.csv", self.rows[::2])
        verb = {**self.rows[0], "word": "think", "length": "5", "pos": "Verb"}
        self.write("cleaned_words.csv", self.rows + [verb])
        for i, row in enumerate(self.rows, 1):
            self.write(f"noun_words_split/{i}.csv", [row])

    def write(self, name, rows, columns=REQUIRED):
        path = self.processed / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

    def validate(self):
        return validate_dataset(self.processed, expected_sample_size=3)

    def test_valid_dataset_counts_and_no_input_changes(self):
        before = {p: p.read_bytes() for p in self.processed.rglob("*.csv")}
        summary = self.validate()
        self.assertTrue(summary["valid"])
        self.assertEqual(summary["error_count"], 0)
        self.assertEqual(summary["files"]["noun_words.csv"]["group_counts"], {"low": 2, "medium": 2, "high": 2})
        self.assertEqual(summary["files"]["cleaned_words.csv"]["rows"], 7)
        self.assertTrue(all(check["status"] == "passed" for check in summary["checks"].values()))
        self.assertTrue(all(p.read_bytes() == content for p, content in before.items()))
        json.dumps(summary, allow_nan=False)

    def test_missing_cleaned_is_incomplete_and_related_checks_blocked(self):
        (self.processed / "cleaned_words.csv").unlink()
        summary = self.validate()
        self.assertFalse(summary["valid"])
        self.assertEqual(summary["status"], "incomplete")
        self.assertIsNone(summary["files"]["cleaned_words.csv"]["rows"])
        self.assertEqual(summary["checks"]["nouns_match_cleaned_noun_subset"]["status"], "blocked")
        self.assertEqual(summary["checks"]["split_union_matches_nouns"]["status"], "passed")
        self.assertEqual(summary["checks"]["cross_file_metadata_consistency"]["available_files_status"], "passed")
        self.assertFalse((self.processed / "cleaned_words.csv").exists())

    def test_overlap_and_union_inconsistencies_reported(self):
        self.write("noun_words_split/2.csv", [self.rows[0]])
        summary = self.validate()
        self.assertEqual(summary["checks"]["split_disjointness"]["status"], "failed")
        self.assertIn("alpha", summary["checks"]["split_disjointness"]["overlapping_words"])
        self.assertIn("beta", summary["checks"]["split_union_matches_nouns"]["missing_from_splits"])

    def test_missing_sample_and_non_noun_records(self):
        extra = {**self.rows[0], "word": "extra", "length": "5", "pos": "Verb"}
        self.write("sampled_words.csv", [extra, *self.rows[2::2]])
        summary = self.validate()
        self.assertIn("non_noun", {i["code"] for i in summary["issues"]})
        self.assertEqual(summary["checks"]["sample_subset_of_nouns"]["missing_sampled_words"], ["extra"])

    def test_duplicate_canonical_words_and_wrong_sample_size(self):
        self.write("sampled_words.csv", [self.rows[0], {**self.rows[0], "word": " ALPHA "}, self.rows[4]])
        summary = self.validate()
        self.assertEqual(summary["files"]["sampled_words.csv"]["duplicate_words"], ["alpha"])
        self.assertEqual(summary["checks"]["sample_size"]["status"], "failed")

    def test_required_fields_missing_and_numeric_metadata_invalid(self):
        for field, value in [("score_bin", ""), ("glasgow_imageability_mean", "NaN"),
                             ("glasgow_imageability_n", "0"), ("glasgow_imageability_sd", "-1"),
                             ("pos_share", "1.1"), ("length", "99")]:
            with self.subTest(field=field):
                self.write("noun_words_split/1.csv", [{**self.rows[0], field: value}])
                self.assertFalse(self.validate()["valid"])
        self.write("noun_words_split/1.csv", [self.rows[0]], columns=[c for c in REQUIRED if c != "pos"])
        self.assertIn("invalid_schema", {i["code"] for i in self.validate()["issues"]})

    def test_cross_file_score_and_rating_count_mismatch(self):
        self.write("noun_words_split/1.csv", [{**self.rows[0], "glasgow_imageability_mean": "2.5", "glasgow_imageability_n": "34"}])
        summary = self.validate()
        mismatch = next(i for i in summary["issues"] if i["code"] == "cross_file_metadata_mismatch")
        self.assertEqual(set(mismatch["columns"]), {"glasgow_imageability_mean", "glasgow_imageability_n"})

    def test_labels_and_split_groups_not_inferred_from_scores(self):
        self.write("noun_words_split/1.csv", [self.rows[2]])
        self.assertIn("wrong_split_group", {i["code"] for i in self.validate()["issues"]})
        self.write("noun_words_split/1.csv", [{**self.rows[0], "score_bin": "[1, 3]", "glasgow_imageability_mean": "4"}])
        self.assertIn("invalid_metadata", {i["code"] for i in self.validate()["issues"]})

    def test_numeric_format_equivalence_and_cleaned_subset_mismatch(self):
        self.write("noun_words_split/1.csv", [{**self.rows[0], "glasgow_imageability_mean": "2.000", "glasgow_imageability_n": "35.0"}])
        self.assertTrue(self.validate()["valid"])
        self.write("cleaned_words.csv", self.rows[1:])
        self.assertEqual(self.validate()["checks"]["nouns_match_cleaned_noun_subset"]["status"], "failed")

    def test_unreadable_file_and_optional_missing_values_are_reported(self):
        (self.processed / "noun_words_split/1.csv").write_bytes(b"\xff\xff")
        summary = self.validate()
        self.assertIn("unreadable_file", {i["code"] for i in summary["issues"]})
        self.assertEqual(summary["checks"]["split_union_matches_nouns"]["status"], "blocked")
        self.write("noun_words_split/1.csv", [self.rows[0]], columns=[*REQUIRED, "extra_optional"])
        summary = self.validate()
        self.assertTrue(summary["valid"])
        self.assertEqual(summary["warning_count"], 1)

    def test_cli_summary_exit_status_and_output_safety(self):
        from unittest.mock import patch
        output = self.root / "validation.json"
        with patch("builtins.print"):
            self.assertEqual(main(["--processed-dir", str(self.processed), "--expected-sample-size", "3", "--output", str(output)]), 0)
        self.assertTrue(json.loads(output.read_text())["valid"])
        with patch("sys.stderr"), self.assertRaises(SystemExit):
            main(["--processed-dir", str(self.processed), "--output", str(output)])
        with patch("sys.stderr"), self.assertRaises(SystemExit):
            main(["--processed-dir", str(self.processed), "--output", str(self.processed / "new.json")])
        (self.processed / "cleaned_words.csv").unlink()
        with patch("builtins.print"):
            self.assertEqual(main(["--processed-dir", str(self.processed), "--expected-sample-size", "3"]), 1)
        self.assertFalse((self.processed / "new.json").exists())


if __name__ == "__main__":
    unittest.main()
