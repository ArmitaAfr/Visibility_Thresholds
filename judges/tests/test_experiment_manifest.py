"""CPU-only manifest tests. All generated artifacts live in temporary directories."""

import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from PIL import Image

from judges.discovery import discover_experiments
from judges.experiment_manifest import ExperimentConfig, build_manifest, save_manifest

DATA = Path(__file__).resolve().parents[2] / "dataset/processed"


class ManifestTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.base = Path(temp.name)
        self.root = self.base / "runs"
        self.root.mkdir()
        self.words = self.base / "words.csv"
        self.words.write_text("word\ncat\ndog\n")
        self.config = ExperimentConfig("pilot", self.words, self.words, (self.root,), self.base / "reports")

    def image(self, part, word, color="red", **extra):
        folder = self.root / part / "001_cat"
        folder.mkdir(parents=True)
        (folder / "experiment.json").write_text(json.dumps({"word": word, **extra}))
        png = folder / "clipdraw_final.png"
        Image.new("RGB", (3, 3), color).save(png)
        return png

    def test_dataset_scope_and_assignment(self):
        for filename, size in [("sampled_words.csv", 150), ("noun_words.csv", 2476)]:
            csv = DATA / filename
            config = replace(self.config, generation_csv=csv, vocabulary_csv=csv, imageability_csv=csv,
                             person=2, num_people=3)
            assigned, vocab, metadata = config.validate()
            self.assertEqual(len(vocab), size)
            self.assertEqual(assigned, vocab[1::3])
            self.assertIn(assigned[0], metadata)
            self.assertEqual(ExperimentConfig(**json.loads(json.dumps(config.to_dict()))), config)
        with self.assertRaisesRegex(ValueError, "missing from judge vocabulary"):
            replace(config, vocabulary_csv=DATA / "sampled_words.csv").validate()

    def test_identity_and_order(self):
        a = self.image("part1", "cat")
        b = self.image("part2", "cat", seed=2)
        config = replace(self.config, input_roots=(b.parent.parent, a.parent.parent))
        manifest = build_manifest(config)
        ids = sorted(e.image_id for e in discover_experiments(self.root).experiments)
        self.assertEqual([row["image_id"] for row in manifest["images"]], ids)
        self.assertEqual(len(set(ids)), 2)
        self.assertEqual({r["part"] for r in manifest["images"]}, {"part1", "part2"})
        self.assertEqual({r["image_sha256"] for r in manifest["images"]},
                         {hashlib.sha256(a.read_bytes()).hexdigest()})
        self.assertEqual(manifest["issues"], [])
        self.assertEqual(build_manifest(replace(config, input_roots=config.input_roots[::-1])), manifest)
        path = save_manifest(config)
        self.assertEqual(json.loads(path.read_text()), manifest)
        with self.assertRaises(FileExistsError):
            save_manifest(config)

    def test_duplicate_and_conflict(self):
        self.image("part1", "cat")
        self.image("part2", "cat")
        manifest = build_manifest(self.config)
        self.assertEqual(len(manifest["images"]), 1)
        self.assertEqual(manifest["issues"][0]["code"], "duplicate_record")
        self.image("part3", "dog")
        manifest = build_manifest(self.config)
        self.assertEqual(manifest["images"], [])
        self.assertIn("conflicting_targets", {i["code"] for i in manifest["issues"]})

    def test_invalid_artifacts(self):
        self.image("missing_png", "cat").unlink()
        self.image("missing_metadata", "cat").with_name("experiment.json").unlink()
        self.image("invalid_metadata", " ")
        self.image("bad_png", "cat").write_bytes(b"not a PNG")
        self.image("unlisted", "coyote")
        (self.root / "005_unfinished").mkdir()
        manifest = build_manifest(self.config)
        self.assertEqual(manifest["images"], [])
        self.assertEqual({i["code"] for i in manifest["issues"]}, {
            "incomplete_folder", "missing_image", "missing_metadata", "malformed_metadata",
            "invalid_source", "target_outside_vocabulary"})

    def test_metadata_join_and_preflight(self):
        metadata = self.base / "metadata.csv"
        metadata.write_text('word,score_bin,glasgow_imageability_mean\ncat,"[1, 3]",2\ndog,"(5, 7]",6\n')
        self.image("part1", "Cat")
        config = replace(self.config, imageability_csv=metadata)
        row = build_manifest(config)["images"][0]
        self.assertEqual(row["imageability"]["imageability_group"], "low")
        metadata.write_text("word\ncat\n")
        with self.assertRaisesRegex(ValueError, "invalid_columns"):
            config.validate()
        with self.assertRaises(FileNotFoundError):
            replace(config, vocabulary_csv=self.base / "missing.csv").validate()
        with self.assertRaisesRegex(ValueError, "Missing image input"):
            replace(config, input_roots=(self.base / "absent",)).validate()
        with self.assertRaisesRegex(ValueError, "inside image inputs"):
            replace(config, output_root=self.root / "out").validate()
