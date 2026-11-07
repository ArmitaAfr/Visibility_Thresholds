"""Synthetic exports and temporary source artifacts; standard-library CPU only."""

import copy
import csv
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from judges.analysis import analyze_exports, write_report
from judges.analysis.patterns import caption_patterns
from judges.analyze import main
from judges.discovery import discover_experiments


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.sources = self.root / "sources"
        self.sig, self.flo = [], []
        words = ["cat", "dog", "fox", "bird", "cup", "idea"]
        for index, rank in enumerate([1, 2, 6]):
            folder = self.sources / str(index)
            folder.mkdir(parents=True)
            target = words[rank - 1]
            (folder / "experiment.json").write_text(json.dumps({"word": target, "seed": index}))
            (folder / "clipdraw_final.png").write_bytes(f"opaque-fixture-{index}".encode())
            experiment = discover_experiments(folder).experiments[0]
            common = dict(image_id=experiment.image_id, target_word=target,
                          image_path=str(experiment.image_path), metadata_path=str(experiment.metadata_path),
                          inference_time_seconds=1.25)
            ranking = [{"word": word, "score": float(6 - i)} for i, word in enumerate(words)]
            self.sig.append({**common, "model_name": "google/siglip2@" + "a" * 40,
                "normalized_prediction": "cat", "target_rank": rank, "target_score": ranking[rank - 1]["score"],
                "raw_output": {"top5": ranking[:5], "candidate_scores": ranking,
                    "configuration": {"candidate_vocabulary_size": 6, "candidate_vocabulary_sha256": "a" * 64,
                                      "prompt_template": "A drawing of {}.", "image_batch_size": index + 1}}})
            caption = [" A glass with the word CAT written on it.\n", "An abstract symbol of freedom.",
                       "A bird beside a cup; no text."][index]
            self.flo.append({**common, "model_name": "microsoft/Florence-2@" + "b" * 40,
                "normalized_prediction": caption.strip().lower(), "target_rank": None, "target_score": None,
                "raw_output": {"caption": caption, "configuration": {"task_prompt": "<CAPTION>"}}})
        self.sig_path, self.flo_path = self.root / "sig.json", self.root / "flo.json"
        self.save()

    def save(self):
        self.sig_path.write_text(json.dumps(self.sig), encoding="utf-8")
        self.flo_path.write_text(json.dumps(self.flo), encoding="utf-8")

    def analyze(self, **kwargs):
        return analyze_exports(self.sig_path, self.flo_path, **kwargs)

    def test_metrics_caption_preservation_patterns_and_csv(self):
        rows, summary = self.analyze()
        metrics = summary["siglip2"]
        self.assertEqual(metrics["vocabulary_size"], 6)
        self.assertEqual(metrics["denominator"], 3)
        self.assertEqual(metrics["top1_target_hit_rate"], 1 / 3)
        self.assertEqual(metrics["top5_target_hit_rate"], 2 / 3)
        self.assertEqual(metrics["mean_target_rank"], 3)
        by_id = {row["image_id"]: row for row in rows}
        self.assertEqual(by_id[self.sig[0]["image_id"]]["florence2_original_caption"], self.flo[0]["raw_output"]["caption"])
        self.assertTrue(by_id[self.sig[2]["image_id"]]["florence2_mentions_apparent_text"])
        self.assertEqual(summary["florence2_patterns"]["apparent_text"]["captions_with_cues"], 2)
        self.assertEqual(summary["florence2_patterns"]["objects"]["captions_with_cues"], 2)
        self.assertEqual(summary["florence2_patterns"]["abstract_concepts"]["captions_with_cues"], 1)
        out = write_report(rows, summary, self.root / "report")
        with (out / "comparison.csv").open(encoding="utf-8", newline="") as handle:
            exported = {r["image_id"]: r for r in csv.DictReader(handle)}
        self.assertEqual(exported[self.sig[0]["image_id"]]["florence2_original_caption"], self.flo[0]["raw_output"]["caption"])
        self.assertEqual(json.loads(exported[self.sig[0]["image_id"]]["siglip2_top5_predictions"]), ["cat", "dog", "fox", "bird", "cup"])
        self.assertIn("not actual human recognition", (out / "report.md").read_text())
        with self.assertRaises(FileExistsError):
            write_report(rows, summary, out)

    def test_determinism_and_source_preservation(self):
        before = {p: p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        self.assertEqual(self.analyze(), self.analyze())
        self.assertTrue(all(p.read_bytes() == value for p, value in before.items()))

    def test_duplicate_and_unmatched_records_are_explicit(self):
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            analyze_exports([self.sig_path, self.sig_path], self.flo_path)
        self.flo.pop()
        self.save()
        with self.assertRaisesRegex(ValueError, "Unmatched"):
            self.analyze()
        rows, summary = self.analyze(allow_unmatched=True)
        self.assertEqual(len(rows), 2)
        self.assertEqual(summary["siglip2"]["denominator"], 2)
        self.assertEqual(len(summary["unmatched_ids"]["siglip2_only"]), 1)

    def test_changed_source_and_wrong_exported_image_rejected(self):
        source = Path(self.sig[0]["image_path"])
        original = source.read_bytes()
        source.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "no longer matches"):
            self.analyze()
        source.write_bytes(original)
        self.flo[0]["image_path"] = self.sig[1]["image_path"]
        self.save()
        with self.assertRaisesRegex(ValueError, "disagrees with metadata"):
            self.analyze()

    def test_relocated_sources_verified_by_ids(self):
        relocated = self.root / "relocated"
        shutil.copytree(self.sources, relocated)
        for row in self.sig + self.flo:
            row["image_path"] = str(self.root / "unavailable.png")
            row["metadata_path"] = str(self.root / "unavailable/experiment.json")
        self.save()
        rows, _ = self.analyze(source_roots=[relocated])
        self.assertTrue(all(str(relocated) in row["image_path"] for row in rows))

    def test_inconsistent_configuration_vocabulary_ranks_and_scores(self):
        baseline = copy.deepcopy(self.sig)
        for mutate in [lambda: self.sig[0]["raw_output"]["configuration"].update(prompt_template="a photo of {}"),
                       lambda: self.sig[0].update(target_rank=2),
                       lambda: self.sig[0].update(target_score=99),
                       lambda: self.sig[0]["raw_output"].update(top5=[]),
                       lambda: self.sig[0].update(inference_time_seconds=float("nan"))]:
            self.sig = copy.deepcopy(baseline)
            mutate()
            self.save()
            with self.assertRaises(ValueError):
                self.analyze()

    def test_target_disagreement_and_missing_caption_rejected(self):
        self.flo[0]["target_word"] = "different"
        self.save()
        with self.assertRaisesRegex(ValueError, "target"):
            self.analyze()
        self.flo[0]["target_word"] = self.sig[0]["target_word"]
        del self.flo[0]["raw_output"]["caption"]
        self.save()
        with self.assertRaisesRegex(ValueError, "original caption"):
            self.analyze()

    def test_patterns_are_whole_word_mentions_not_ocr(self):
        self.assertEqual(caption_patterns("A texture.")["apparent_text"], [])
        self.assertEqual(caption_patterns("No text is visible.")["apparent_text"], ["text"])

    def test_cli_creates_report_files_without_inference(self):
        out = self.root / "cli-report"
        main(["--siglip2", str(self.sig_path), "--florence2", str(self.flo_path), "--output-dir", str(out)])
        self.assertEqual({p.name for p in out.iterdir()}, {"comparison.csv", "summary.json", "report.md"})

    def test_merge_nonoverlapping_exports_from_separate_sessions(self):
        sig2, flo2 = self.root / "sig-session2.json", self.root / "flo-session2.json"
        sig2.write_text(json.dumps(self.sig[1:]))
        flo2.write_text(json.dumps(self.flo[1:]))
        self.sig, self.flo = self.sig[:1], self.flo[:1]
        self.save()
        rows, summary = analyze_exports([self.sig_path, sig2], [self.flo_path, flo2])
        self.assertEqual(len(rows), 3)
        self.assertEqual(len(summary["inputs"]["siglip2"]), 2)
        self.assertEqual(summary["siglip2"]["top1_target_hits"], 1)


if __name__ == "__main__":
    unittest.main()
