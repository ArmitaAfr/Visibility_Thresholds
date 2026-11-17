"""Exercise persistence and joins without importing OCR, Torch or Transformers."""

import base64
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from judges.experiment_manifest import ExperimentConfig, save_manifest
from text_score.run_manifest import run

PNG = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=")
REV = "a" * 40


class TextScoreTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        base = Path(tmp.name)
        self.root = base / "parts"
        for part in ("one", "two"):
            folder = self.root / part / "001_cat"
            folder.mkdir(parents=True)
            (folder / "clipdraw_final.png").write_bytes(PNG)
            (folder / "experiment.json").write_text(json.dumps(dict(word="cat", run=part)))
        words = base / "words.csv"
        words.write_text("word\ncat\n")
        config = ExperimentConfig("pilot", words, words, (self.root,), base / "outputs")
        self.manifest = save_manifest(config)
        self.data = json.loads(self.manifest.read_text())
        self.assertEqual(self.data["issues"], [])
        self.out = config.output_root / "test"
        self.score = Mock(return_value={"word": "cat", "text_level_pred": 1, "text_score": 0.3})
        loader = patch("text_score.run_manifest._load_scorer",
                       side_effect=lambda rev, scale: (self.score, dict(revision=rev, upscale=scale)))
        loader.start()
        self.addCleanup(loader.stop)

    def test_join_and_nonoverwrite(self):
        path = run(self.manifest, self.out, revision=REV)
        rows = json.loads(path.read_text())["results"]
        self.assertEqual(len(rows), 2)
        for row, original in zip(rows, self.data["images"]):
            for field in ("image_id", "image_sha256", "target_word", "part", "folder", "image_path"):
                self.assertEqual(row[field], original[field])
            self.assertEqual(row["word"], row["target_word"])
            self.assertEqual(row["text_score"], 0.3)
        with self.assertRaises(FileExistsError):
            run(self.manifest, self.out, revision=REV)
        with self.assertRaisesRegex(ValueError, "unfinished"):
            run(self.manifest, self.out, revision=REV, resume=True)

    def test_partial_and_resume(self):
        self.score.side_effect = [{"word": "cat", "text_score": .2}, RuntimeError("ocr failed")]
        with self.assertRaisesRegex(ValueError, "1 image"):
            run(self.manifest, self.out, revision=REV)
        failed = json.loads((self.out / "attempt_errors.json").read_text())["errors"][0]
        self.assertEqual(failed["image_path"], self.data["images"][1]["image_path"])
        self.assertEqual(failed["image_id"], self.data["images"][1]["image_id"])
        self.assertFalse((self.out / "results.json").exists())
        with self.assertRaisesRegex(ValueError, "configuration changed"):
            run(self.manifest, self.out, revision=REV, resume=True, upscale=4)
        self.score.side_effect = None
        cached = Path(self.data["images"][0]["image_path"])
        cached.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "1 image"):
            run(self.manifest, self.out, revision=REV, resume=True)
        cached.write_bytes(PNG)
        rows = json.loads(run(self.manifest, self.out, revision=REV, resume=True).read_text())["results"]
        self.assertEqual(len({row["image_id"] for row in rows}), 2)
        self.assertEqual(self.score.call_count, 3)
        self.assertEqual(self.score.call_args.args[0].getvalue(), PNG)

    def test_invalid_manifests(self):
        for change in [dict(schema_version=2), dict(issues=["incomplete"]), dict(images=[]),
                       dict(images=self.data["images"] * 2), dict(images=[{"image_id": "bad"}])]:
            with self.subTest(change=list(change)):
                self.manifest.write_text(json.dumps({**self.data, **change}))
                with self.assertRaises(ValueError):
                    run(self.manifest, self.out, revision=REV)
                self.assertFalse(self.out.exists())

    def test_source_verification_and_paths(self):
        with self.assertRaisesRegex(ValueError, "output_root"):
            run(self.manifest, self.root / "new", revision=REV)
        first, second = self.data["images"]
        Path(first["metadata_path"]).write_text('{"word":"dog"}')
        Path(second["image_path"]).unlink()
        with self.assertRaisesRegex(ValueError, "2 image"):
            run(self.manifest, self.out, revision=REV)
        self.score.assert_not_called()
