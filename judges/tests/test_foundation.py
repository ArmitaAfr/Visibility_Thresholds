import csv
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from judges import (
    DiscoveryError, JudgeOutput, discover_experiments, evaluate,
    export_results, load_candidates,
)


class FoundationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def record(self, folder="part1/001_misleading", word="dog", **extra):
        path = self.root / folder
        path.mkdir(parents=True)
        (path / "experiment.json").write_text(
            json.dumps(dict(word=word, seed=0, run_mode="full", **extra)), encoding="utf-8")
        # No image decoding/inference in Phase 1; opaque bytes are sufficient.
        (path / "clipdraw_final.png").write_bytes(b"fixture-image")
        return path

    def test_multiple_roots_metadata_label_and_move_stable_id(self):
        first = self.record()
        self.record("part2/001_other", "cat")
        report = discover_experiments([self.root / "part1", self.root / "part2"])
        self.assertEqual([e.target_word for e in report.experiments], ["dog", "cat"])
        image_id = report.experiments[0].image_id
        shutil.move(str(first), str(self.root / "moved"))
        self.assertEqual(discover_experiments(self.root / "moved").experiments[0].image_id, image_id)

    def test_missing_empty_images_and_orphans(self):
        path = self.record()
        image = path / "clipdraw_final.png"
        image.unlink()
        with self.assertRaises(DiscoveryError) as error:
            discover_experiments(self.root)
        self.assertEqual(error.exception.report.issues[0].code, "missing_image")
        image.touch()
        self.assertEqual(discover_experiments(self.root, strict=False).issues[0].code, "missing_image")
        (path / "experiment.json").unlink()
        self.assertEqual(discover_experiments(self.root, strict=False).issues[0].code, "missing_metadata")

    def test_malformed_metadata(self):
        path = self.record()
        for value in ['{', '[]', '{}', '{"word": 3}', '{"word": " "}', '{"word":"dog","seed":NaN}']:
            with self.subTest(value=value):
                (path / "experiment.json").write_text(value)
                report = discover_experiments(self.root, strict=False)
                self.assertEqual(report.experiments, [])
                self.assertEqual(report.issues[0].code, "malformed_metadata")

    def test_duplicate_copies_and_overlapping_roots(self):
        path = self.record()
        shutil.copytree(path, self.root / "copy")
        report = discover_experiments(self.root, strict=False)
        self.assertEqual(len(report.experiments), 1)
        self.assertEqual(report.issues[0].code, "duplicate_record")
        with self.assertRaises(DiscoveryError):
            discover_experiments([path, path])

    def test_same_word_with_different_settings_is_not_duplicate(self):
        self.record()
        self.record("other", prompt="a different prompt")
        self.assertEqual(len(discover_experiments(self.root).experiments), 2)

    def test_invalid_and_empty_roots(self):
        for root in [self.root, self.root / "missing"]:
            with self.assertRaises(DiscoveryError):
                discover_experiments(root)
        with self.assertRaises(DiscoveryError):
            discover_experiments([])

    def test_vocabulary_cleanup_and_official_dataset(self):
        path = self.root / "words.csv"
        path.write_text('\ufeffWords,score\n Dog ,1\ndog,2\n,3\nNA,4\nnull,5\ncat,6\n', encoding="utf-8")
        self.assertEqual(load_candidates(path), ["Dog", "NA", "null", "cat"])
        words = load_candidates()
        self.assertEqual(len(words), 150)
        self.assertEqual(words[0], "alimony")
        for contents in ["other\ndog\n", "word\n\n"]:
            path.write_text(contents)
            with self.assertRaises(ValueError):
                load_candidates(path)

    def test_ranking_caption_exports_and_read_only_inputs(self):
        self.record()
        experiments = discover_experiments(self.root).experiments
        before = {p: p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        ranking = Mock(model_name="mock-ranking", predict=Mock(return_value=JudgeOutput(
            {"scores": [0.8, 0.2]}, "cat", ["cat", "dog"])))
        caption = Mock(model_name="mock-caption", predict=Mock(return_value=JudgeOutput(
            {"caption": "A dog, sitting.\n"}, "a dog, sitting.")))
        rows = evaluate(experiments, ["dog", "cat"], ranking)
        rows += evaluate(experiments, ["dog", "cat"], caption)
        self.assertEqual(rows[0].target_rank, 2)
        self.assertIsNone(rows[1].target_rank)
        self.assertGreaterEqual(rows[0].inference_time_seconds, 0)
        ranking.predict.assert_called_once_with(experiments[0].image_path, ("dog", "cat"))
        export_results(rows, self.root / "judge-results/results.json")
        export_results(rows, self.root / "judge-results/results.csv")
        data = json.loads((self.root / "judge-results/results.json").read_text())
        with (self.root / "judge-results/results.csv").open(newline="") as handle:
            csv_rows = list(csv.DictReader(handle))
        self.assertEqual(json.loads(csv_rows[1]["raw_output"]), data[1]["raw_output"])
        self.assertEqual(csv_rows[1]["target_rank"], "")
        self.assertTrue(all(p.read_bytes() == content for p, content in before.items()))
        with self.assertRaises(FileExistsError):
            export_results(rows, experiments[0].metadata_path)
        self.assertTrue(all(p.read_bytes() == content for p, content in before.items()))

    def test_invalid_adapter_outputs_and_vocabulary(self):
        self.record()
        experiments = discover_experiments(self.root).experiments
        judge = Mock(model_name="mock", predict=Mock(return_value=JudgeOutput({}, "dog")))
        for candidates in [[], ["cat"], ["dog", "DOG"], ["dog", ""]]:
            with self.assertRaises(ValueError):
                evaluate(experiments, candidates, judge)
        with self.assertRaises(ValueError):
            evaluate(experiments * 2, ["dog", "cat"], judge)
        for output in [JudgeOutput({}, "dog", ["dog", "dog"]),
                       JudgeOutput({}, "dog", ["dog"]),
                       JudgeOutput({"score": float("nan")}, "dog"),
                       JudgeOutput({}, None)]:
            judge.predict.return_value = output
            with self.assertRaises(ValueError):
                evaluate(experiments, ["dog", "cat"], judge)

    def test_adapter_errors_propagate_without_export(self):
        self.record()
        judge = Mock(model_name="mock", predict=Mock(side_effect=RuntimeError("failure")))
        with self.assertRaisesRegex(RuntimeError, "failure"):
            evaluate(discover_experiments(self.root).experiments, ["dog"], judge)

    def test_empty_exports_and_invalid_extension(self):
        export_results([], self.root / "empty.csv")
        export_results([], self.root / "empty.json")
        self.assertEqual(json.loads((self.root / "empty.json").read_text()), [])
        self.assertIn("image_id", (self.root / "empty.csv").read_text())
        with self.assertRaises(ValueError):
            export_results([], self.root / "bad.txt")
        self.assertFalse((self.root / "bad.txt").exists())


if __name__ == "__main__":
    unittest.main()
