"""Saved-artifact reporting fixtures; no model imports or inference."""

import base64
import csv
import hashlib
import json
import tempfile
import unittest
import zlib
from pathlib import Path
from unittest.mock import patch

import unified_colab as vt
from judges.analysis.combined import MODELS, read
from text_score import run_manifest

PNG = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=")


class CombinedTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.base = Path(temp.name)
        words = self.base / "words.csv"
        words.write_text('word,score_bin,glasgow_imageability_mean\ncat,"[1, 3]",2\ndog,"(5, 7]",6\n')
        root = self.base / "inputs"
        # Two different metadata records for cat; a valid ancillary PNG comment distinguishes dog.
        chunk = b"tEXtComment\0fixture"
        extra = (len(chunk)-4).to_bytes(4, "big") + chunk + zlib.crc32(chunk).to_bytes(4, "big")
        for i, word in enumerate(["cat", "dog", "cat"]):
            folder = root / f"part{i}" / "001_study"
            folder.mkdir(parents=True)
            (folder / "clipdraw_final.png").write_bytes(PNG if word == "cat" else PNG[:-12] + extra + PNG[-12:])
            vt.create(folder / "experiment.json", {"word": word, "seed": i})
        self.flow = vt.Workflow(drive_root=self.base, run_id="run1", cohort_id="pilot", generation_csv=words,
                                vocabulary_csv=words, metadata_csv=words, input_roots=[root], source_hash="a"*64)
        self.flow.bind("MANIFEST")
        self.manifest = self.flow.manifest(publish=True)
        self.text = self.flow.artifacts / "text_score"
        self.judges = {}
        for model in vt.PINS:
            cfg, directory = self.flow.job(model, 1, self.manifest)
            self.judges[model] = directory
            job = {"configuration": cfg, "mode": "run", "selected_image_ids": cfg["expected_image_ids"],
                   "vocabulary_file_sha256": self.manifest["csv_sha256"]["vocabulary_csv"]}
            ranking = [{"word": "cat", "score": 1.0}, {"word": "dog", "score": 0.0}]
            rows = []
            for image in reversed(self.manifest["images"]):
                row = {k: image[k] for k in ("image_id", "image_path", "metadata_path", "target_word")}
                profile = {"resolved_revision": cfg["revision"]}
                if model == "siglip2":
                    rank = 1 if row["target_word"] == "cat" else 2
                    profile.update(candidate_vocabulary_size=2, candidate_vocabulary_sha256=
                                   hashlib.sha256(json.dumps(["cat", "dog"]).encode()).hexdigest())
                    row.update(raw_output={"candidate_scores": ranking, "top5": ranking, "configuration": profile},
                               normalized_prediction="cat", target_rank=rank, target_score=ranking[rank-1]["score"])
                else:
                    row.update(raw_output={"caption": "No text is visible", "configuration": profile},
                               normalized_prediction="no text is visible", target_rank=None, target_score=None)
                rows.append({**row, "model_name": MODELS[model]+"@"+cfg["revision"], "inference_time_seconds": 1})
            vt.create(directory / "job_manifest.json", job)
            vt.create(directory / "results.json", rows)
            vt.create(directory / "complete.json", {"export_sha256": vt.file_digest(directory / "results.json"),
                                                    "job_manifest_sha256": vt.digest(job)})

    def score(self, fail=False):
        revision = "b"*40

        def stub(png, word):
            if fail and word == "dog":
                raise RuntimeError("fixture OCR failure")
            return {"word": word, "ocr_strings": "", "ocr_max_conf": 0, "ocr_n_boxes": 0,
                    "text_area_frac": 0, "target_sim": 0, "target_partial": 0, "zs_p_text": .25,
                    "zs_typo_margin": -.1, "text_level_pred": 1, "text_score": .25 if word == "cat" else 1.0}

        provenance = {"model": MODELS["text_score"], "revision": revision, "upscale": 3}
        with patch.object(run_manifest, "_load_scorer", return_value=(stub, provenance)):
            run_manifest.run(self.flow.manifest_path, self.text, revision=revision)

    def test_three_way_join_aggregates_and_reproducibility(self):
        self.score()
        dest = self.flow.analyze()
        summary = read(dest / "combined_summary.json")
        self.assertEqual(summary["status"], "complete")
        overall = summary["overall"]
        self.assertEqual((overall["images"], overall["unique_targets"]), (3, 2))
        self.assertEqual(overall["siglip2"]["rank_counts"], {"1": 2, "2": 1})
        self.assertEqual(overall["siglip2"]["top1_accuracy"], 2/3)
        self.assertEqual(overall["siglip2"]["top5_accuracy"], 1)
        self.assertEqual(overall["text_score"]["mean"], .5)
        self.assertEqual(summary["by_imageability"]["high"]["text_score"]["median"], 1)
        self.assertIsNone(summary["by_imageability"]["medium"]["siglip2"]["top5_accuracy"])
        self.assertEqual(summary["by_text_score_interval"]["[0.25, 0.5)"]["siglip2"]["denominator"], 2)
        with (dest / "combined.csv").open() as f:
            rows = list(csv.DictReader(f))
        for row, image in zip(rows, self.manifest["images"]):
            self.assertEqual(row["image_id"], image["image_id"])
            self.assertEqual(row["image_sha256"], image["image_sha256"])
            self.assertEqual(float(row["text_score"]), 1 if row["target_word"] == "dog" else .25)
            self.assertEqual(row["florence2_mentions_apparent_text"], "True")
            self.assertEqual(row["ocr_strings"], "")
        archived = self.base / "previous"
        dest.rename(archived)
        self.flow.analyze()
        with self.assertRaises(FileExistsError):
            self.flow.analyze()
        for path in archived.iterdir():
            self.assertEqual(path.read_bytes(), (dest/path.name).read_bytes())

    def test_reject_wrong_id_hash_target_cohort_duplicates_and_missing(self):
        self.score()
        p = self.text / "results.json"
        original = p.read_text()
        for field in ("image_id", "image_sha256", "target_word", "word", "cohort_id", "manifest_sha256",
                      "duplicate", "missing"):
            with self.subTest(field=field):
                data = json.loads(original)
                if field == "duplicate":
                    data["results"].append(data["results"][0])
                elif field == "missing":
                    data["results"].pop()
                elif field == "manifest_sha256":
                    data[field] = "bad"
                else:
                    data["results"][0][field] = "bad"
                p.write_text(json.dumps(data))
                with self.assertRaises(ValueError):
                    self.flow.analyze(partial=True)
        self.assertFalse((self.flow.artifacts / "analysis_exploratory").exists())

    def test_partial_sources_and_failed_attempt(self):
        with self.assertRaisesRegex(ValueError, "1 image"):
            self.score(fail=True)
        (self.judges["florence2"] / "complete.json").unlink()
        with self.assertRaisesRegex(ValueError, "Incomplete production analysis"):
            self.flow.analyze()
        dest = self.flow.analyze(partial=True)
        summary = read(dest / "combined_summary.json")
        self.assertEqual(summary["status"], "partial")
        self.assertEqual(summary["sources"]["florence2"]["shards"][0]["status"], "unfinished")
        self.assertEqual(len(summary["coverage"]["florence2"]["missing_ids"]), 3)
        self.assertEqual(summary["sources"]["text_score"]["last_recorded_errors"][0]["target_word"], "dog")
        self.assertEqual(summary["overall"]["text_score"]["n"], 2)
        self.assertEqual(summary["overall"]["siglip2"]["denominator"], 3)
        self.assertEqual(summary["by_imageability"]["high"]["text_score"]["n"], 0)
        self.assertFalse((dest / "comparison.csv").exists())

    def test_reject_corrupt_checkpoint_and_conflicting_judge_identity(self):
        self.score()
        path = next((self.text/"checkpoints").glob("*.json"))
        envelope = read(path)
        envelope["result"]["text_score"] = .19
        path.write_text(json.dumps(envelope))
        with self.assertRaisesRegex(ValueError, "checkpoint"):
            self.flow.analyze()
        directory = self.judges["siglip2"]
        export, receipt = directory/"results.json", directory/"complete.json"
        rows = read(export)
        rows[0]["image_sha256"] = "0"*64
        export.write_text(json.dumps(rows))
        receipt.write_text(json.dumps({**read(receipt), "export_sha256": vt.file_digest(export)}))
        with self.assertRaisesRegex(ValueError, "image_sha256 mismatch"):
            self.flow.analyze(partial=True)
