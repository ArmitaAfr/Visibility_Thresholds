"""Strict metadata joins and descriptive statistics; no model dependencies."""

import copy
import csv
import json
import tempfile
import unittest
from pathlib import Path

from judges.analysis.imageability import (
    GROUP_BINS, ImageabilityMetadataError, load_imageability,
    stratify_imageability, imageability_markdown,
)
from judges.analysis.patterns import caption_patterns


class ImageabilityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.metadata = self.root / "metadata.csv"
        self.records = [dict(word="cat", score_bin="[1, 3]", glasgow_imageability_mean="2.5"),
                        dict(word="dog", score_bin="(3, 5]", glasgow_imageability_mean="4"),
                        dict(word="idea", score_bin="(5, 7]", glasgow_imageability_mean="6")]
        self.save()
        self.rows = []
        # Arbitrary synthetic ratings/labels test joining, not real word ratings.
        for i, (word, rank, caption) in enumerate([
                ("CAT", 1, "A cat with a written word."),
                (" cat ", 7, "A glass. No text."),
                ("dog", 2, "An abstract idea."),
                ("idea", 6, "A bird."),
                ("idea", 4, "Freedom written on a sign.")]):
            p = caption_patterns(caption)
            self.rows.append(dict(image_id=str(i), target_word=word, siglip2_target_rank=rank,
                florence2_original_caption=caption, florence2_mentions_apparent_text=bool(p["apparent_text"]),
                florence2_text_mention_cues=p["apparent_text"], florence2_object_cues=p["objects"],
                florence2_abstract_concept_cues=p["abstract_concepts"]))
        self.summary = {"matched_images": 5, "siglip2": {"vocabulary_size": 150, "vocabulary_sha256": "a" * 64},
                        "models": {"siglip2": "siglip@commit", "florence2": "flo@commit"},
                        "configuration_hashes": {"siglip2": "one", "florence2": "two"}}

    def save(self):
        columns = list(dict.fromkeys(k for row in self.records for k in row))
        with self.metadata.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(self.records)

    def stratify(self, rows=None):
        return stratify_imageability(self.rows if rows is None else rows, self.summary, self.metadata)

    def test_repeated_words_are_separate_images_with_mean_and_median(self):
        rows, summary = self.stratify()
        self.assertEqual(len(rows), 5)
        groups = summary["imageability"]["groups"]
        low = groups["low"]
        self.assertEqual(low["evaluated_images"], 2)
        self.assertEqual(low["unique_target_words"], 1)
        self.assertEqual(low["siglip2"]["top1_target_hits"], 1)
        self.assertEqual(low["siglip2"]["top1_target_hit_rate"], .5)
        self.assertEqual(low["siglip2"]["top5_target_hits"], 1)
        self.assertEqual(low["siglip2"]["mean_target_rank"], 4)
        self.assertEqual(low["siglip2"]["median_target_rank"], 4)
        self.assertEqual(groups["medium"]["siglip2"]["median_target_rank"], 2)
        self.assertEqual(groups["high"]["siglip2"]["median_target_rank"], 5)
        self.assertEqual(groups["high"]["unique_target_words"], 1)
        self.assertEqual(summary["imageability"]["coverage"]["matched_unique_target_words"], 3)
        self.assertEqual([row["canonical_target_word"] for row in rows[:2]], ["cat", "cat"])

    def test_existing_heuristic_counts_and_percentages(self):
        _, summary = self.stratify()
        groups = summary["imageability"]["groups"]
        low = groups["low"]["florence2_patterns"]
        self.assertEqual(low["apparent_text"]["captions_with_cues"], 2)  # Includes 'no text'.
        self.assertEqual(low["apparent_text"]["percentage"], 100)
        self.assertEqual(low["objects"]["percentage"], 100)
        self.assertEqual(groups["high"]["florence2_patterns"]["abstract_concepts"]["percentage"], 50)

    def test_overall_summary_rows_and_metadata_preserved(self):
        before_rows, before_summary, before_bytes = copy.deepcopy(self.rows), copy.deepcopy(self.summary), self.metadata.read_bytes()
        enriched, summary = self.stratify()
        self.assertEqual(self.rows, before_rows)
        self.assertEqual(self.summary, before_summary)
        self.assertEqual(self.metadata.read_bytes(), before_bytes)
        for key in self.summary:
            self.assertEqual(summary[key], self.summary[key])
        self.assertEqual(enriched[0]["florence2_original_caption"], self.rows[0]["florence2_original_caption"])
        self.assertEqual(enriched[0]["score_bin"], "[1, 3]")
        self.assertEqual(enriched[0]["glasgow_imageability_mean"], 2.5)
        self.assertEqual(summary["imageability"]["metadata"]["optional_columns_absent"],
                         ["glasgow_imageability_sd", "glasgow_imageability_n"])

    def test_empty_groups_and_missing_dataset_categories_are_explicit(self):
        self.records = self.records[:1]
        self.save()
        _, summary = self.stratify(self.rows[:2])
        data = summary["imageability"]
        self.assertEqual(data["metadata"]["categories_absent_from_metadata"], ["medium", "high"])
        self.assertEqual(data["coverage"]["empty_evaluated_groups"], ["medium", "high"])
        for group in ("medium", "high"):
            stats = data["groups"][group]
            self.assertEqual(stats["status"], "empty")
            self.assertEqual(stats["evaluated_images"], 0)
            self.assertIsNone(stats["siglip2"]["top1_target_hit_rate"])
            self.assertIsNone(stats["siglip2"]["mean_target_rank"])
            self.assertIsNone(stats["siglip2"]["median_target_rank"])
            self.assertIsNone(stats["florence2_patterns"]["objects"]["percentage"])
        text = "\n".join(imageability_markdown(data))
        self.assertIn("N/A", text)
        self.assertIn("not verified OCR", text)
        json.dumps(summary, allow_nan=False)

    def test_all_missing_targets_reported_without_partial_results(self):
        self.records = self.records[:1]
        self.save()
        with self.assertRaises(ImageabilityMetadataError) as error:
            self.stratify()
        self.assertEqual({issue["word"] for issue in error.exception.issues}, {"dog", "idea"})

    def test_duplicate_and_ambiguous_word_matches_are_errors(self):
        self.records.append({**self.records[0], "word": " CAT "})
        self.save()
        with self.assertRaises(ImageabilityMetadataError) as error:
            self.stratify()
        self.assertEqual(error.exception.issues[0]["code"], "duplicate_word")
        self.records[-1].update(score_bin="(3, 5]", glasgow_imageability_mean="4")
        self.save()
        with self.assertRaises(ImageabilityMetadataError) as error:
            self.stratify()
        self.assertEqual(error.exception.issues[0]["code"], "ambiguous_word")

    def test_missing_or_inconsistent_required_metadata_never_inferred(self):
        for value in ("", "low", "(1, 3]"):
            with self.subTest(label=value):
                self.records[0]["score_bin"] = value
                self.save()
                with self.assertRaises(ImageabilityMetadataError):
                    self.stratify()
        self.records[0]["score_bin"] = "[1, 3]"
        for value in ("", "NaN", "inf", "4.0"):
            with self.subTest(score=value):
                self.records[0]["glasgow_imageability_mean"] = value
                self.save()
                with self.assertRaises(ImageabilityMetadataError):
                    self.stratify()
        del self.records[0]["glasgow_imageability_mean"]
        self.records = self.records[:1]
        self.save()
        with self.assertRaisesRegex(ImageabilityMetadataError, "invalid_columns"):
            self.stratify()

    def test_present_optional_columns_must_be_complete_and_valid(self):
        for row in self.records:
            row.update(glasgow_imageability_sd="1.2", glasgow_imageability_n="35")
        self.save()
        enriched, _ = self.stratify()
        self.assertEqual(enriched[0]["glasgow_imageability_n"], 35)
        self.records[0]["glasgow_imageability_n"] = "0"
        self.save()
        with self.assertRaises(ImageabilityMetadataError):
            self.stratify()

    def test_source_interval_boundaries_and_actual_dataset(self):
        for row, score in zip(self.records, ("3", "5", "7")):
            row["glasgow_imageability_mean"] = score
        self.save()
        self.stratify()
        self.records[1]["glasgow_imageability_mean"] = "3"
        self.save()
        with self.assertRaises(ImageabilityMetadataError):
            self.stratify()
        index, audit = load_imageability(Path(__file__).resolve().parents[2] / "dataset/processed/sampled_words.csv")
        self.assertEqual(len(index), 150)
        self.assertEqual(audit["category_word_counts"], {"low": 50, "medium": 50, "high": 50})
        self.assertEqual(audit["validation_issues"], [])
        self.assertEqual(audit["group_bin_mapping"], GROUP_BINS)

    def test_duplicate_headers_and_incomplete_rows_rejected(self):
        self.metadata.write_text('word,word,score_bin,glasgow_imageability_mean\ncat,cat,"[1, 3]",2\n')
        with self.assertRaises(ImageabilityMetadataError):
            self.stratify()
        self.metadata.write_text('word,score_bin,glasgow_imageability_mean\ncat,"[1, 3]"\n')
        with self.assertRaises(ImageabilityMetadataError):
            self.stratify()


if __name__ == "__main__":
    unittest.main()
