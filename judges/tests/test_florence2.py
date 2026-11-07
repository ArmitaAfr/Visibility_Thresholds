"""Offline captioning fixtures: no pretrained weights or network required."""

import importlib.util
import importlib.metadata
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from judges import EvaluationFailure, Experiment, evaluate
from judges.florence2 import Florence2Judge, load_florence, normalize_caption
from judges.run_florence2 import main

HAS_DEPS = all(importlib.util.find_spec(name) for name in ("torch", "PIL", "transformers"))
if HAS_DEPS:
    import torch
    from PIL import Image


class NormalizationTests(unittest.TestCase):
    def test_caption_normalization_is_separate_text_cleanup(self):
        self.assertEqual(normalize_caption("  A CAT,\n sitting.  "), "a cat, sitting.")
        self.assertEqual(normalize_caption("\uff21 bird"), "a bird")


@unittest.skipUnless(HAS_DEPS, "Optional model dependencies not installed.")
class FlorenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.image = self.root / "target_must_not_leak.png"
        Image.new("RGB", (10, 8), "white").save(self.image)
        self.model = Mock()
        self.model.to.return_value = self.model
        self.model.eval.return_value = self.model
        self.model.config = SimpleNamespace(_commit_hash="a" * 40)
        self.model.language_model.generation_config.to_dict.return_value = {"eos_token_id": 2}
        self.model.generate.side_effect = self.generate
        self.processor = Mock()
        self.processor.image_processor.to_dict.return_value = {"size": {"height": 768, "width": 768}}
        self.processor.side_effect = self.process
        self.raw_text = "<s>  A CAT,\n sitting.  </s>"
        self.caption = "  A CAT,\n sitting.  "
        self.processor.batch_decode.side_effect = lambda ids, **kw: [self.raw_text] * len(ids)
        self.processor.post_process_generation.return_value = {"<CAPTION>": self.caption}

    def process(self, *, text, images, return_tensors, padding):
        self.assertEqual(text, ["<CAPTION>"] * len(images))
        self.assertTrue(all(isinstance(image, Image.Image) for image in images))
        self.assertTrue(all(image.mode == "RGB" for image in images))
        self.assertEqual(return_tensors, "pt")
        return {"input_ids": torch.ones(len(images), 2, dtype=torch.long),
                "pixel_values": torch.ones(len(images), 3, 8, 10),
                "attention_mask": torch.ones(len(images), 2, dtype=torch.long)}

    def generate(self, *, input_ids, pixel_values, **kwargs):
        self.assertFalse(torch.is_grad_enabled())
        self.assertEqual(input_ids.dtype, torch.long)
        self.assertEqual(pixel_values.dtype, torch.float32)
        self.assertNotIn("attention_mask", kwargs)
        self.assertFalse(kwargs["do_sample"])
        return torch.ones(len(input_ids), 3, dtype=torch.long)

    def judge(self, **kwargs):
        return Florence2Judge(model=self.model, processor=self.processor, device="cpu", **kwargs)

    def test_exact_raw_caption_and_no_target_vocabulary_or_filename_context(self):
        class ForbiddenCandidates:
            def __iter__(self):
                raise AssertionError("Caption adapter must not inspect vocabulary")
        judge = self.judge()
        output = judge.predict(self.image, ForbiddenCandidates())
        self.assertEqual(output.raw_output["caption"], self.caption)
        self.assertEqual(output.raw_output["generated_text"], self.raw_text)
        self.assertEqual(output.normalized_prediction, "a cat, sitting.")
        self.assertIsNone(output.ranked_candidates)
        self.assertIsNone(output.candidate_scores)
        self.processor.post_process_generation.assert_called_once_with(
            self.raw_text, task="<CAPTION>", image_size=(10, 8))
        json.dumps(output.raw_output, allow_nan=False)
        self.model.eval.assert_called_once()
        self.model.requires_grad_.assert_called_once_with(False)

    def test_batching_null_target_fields_and_source_preservation(self):
        before = self.image.read_bytes()
        experiments = [Experiment(str(i), "cat", self.image, self.root / "experiment.json") for i in range(3)]
        rows = evaluate(experiments, ["cat"], self.judge(), batch_size=2)
        self.assertEqual(self.model.generate.call_count, 2)
        self.assertTrue(all(row.target_rank is None and row.target_score is None for row in rows))
        self.assertTrue(all(row.inference_time_seconds >= 0 for row in rows))
        self.assertEqual(before, self.image.read_bytes())

    def test_checkpoint_resume_and_changed_generation_rejected(self):
        experiment = Experiment("image-id", "cat", self.image, self.root / "experiment.json")
        checkpoint = self.root / "checkpoint"
        judge = self.judge()
        rows = evaluate([experiment], ["cat"], judge, checkpoint_dir=checkpoint)
        self.model.generate.reset_mock()
        self.assertEqual(evaluate([experiment], ["cat"], judge, checkpoint_dir=checkpoint, resume=True), rows)
        self.model.generate.assert_not_called()
        for kwargs in ({"num_beams": 1}, {"max_new_tokens": 64}):
            with self.assertRaisesRegex(ValueError, "configuration changed"):
                evaluate([experiment], ["cat"], self.judge(**kwargs), checkpoint_dir=checkpoint, resume=True)
        self.model.config._commit_hash = "b" * 40
        with self.assertRaisesRegex(ValueError, "configuration changed"):
            evaluate([experiment], ["cat"], self.judge(), checkpoint_dir=checkpoint, resume=True)

    def test_failed_generation_unreadable_and_invalid_captions(self):
        judge = self.judge()
        with self.assertRaisesRegex(RuntimeError, "Cannot read image"):
            judge.predict(self.root / "missing.png", [])
        self.model.generate.side_effect = RuntimeError("generation failure")
        with self.assertRaisesRegex(RuntimeError, "generation failure"):
            judge.predict(self.image, [])
        self.model.generate.side_effect = self.generate
        for result in ({}, {"<CAPTION>": " "}, {"<CAPTION>": []}):
            self.processor.post_process_generation.return_value = result
            with self.assertRaisesRegex(RuntimeError, "invalid caption"):
                judge.predict(self.image, [])

    def test_loading_pins_model_processor_and_custom_code_once(self):
        self.model.config = SimpleNamespace()  # Missing private metadata is allowed.
        with patch("judges.florence2.SUPPORTED_TRANSFORMERS", importlib.metadata.version("transformers")), \
             patch("judges.florence2.resolve_model_revision", return_value="a" * 40) as resolve, \
             patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=self.model) as model_load, \
             patch("transformers.AutoProcessor.from_pretrained", return_value=self.processor) as processor_load:
            judge = Florence2Judge(device="cpu", local_files_only=True, cache_dir=self.root)
            judge.predict(self.image, [])
            judge.predict(self.image, [])
            model_load.assert_called_once()
            processor_load.assert_called_once()
            for loader in (model_load, processor_load):
                self.assertEqual(loader.call_args.kwargs["revision"], "a" * 40)
                self.assertEqual(loader.call_args.kwargs["code_revision"], "a" * 40)
                self.assertTrue(loader.call_args.kwargs["trust_remote_code"])
                self.assertTrue(loader.call_args.kwargs["local_files_only"])
            self.assertEqual(judge.checkpoint_configuration()["resolved_revision"], "a" * 40)
            resolve.assert_called_once()

    def test_compatibility_guard_and_loading_errors_actionable(self):
        with patch("transformers.__version__", "5.3.0"), \
             patch("judges.florence2.resolve_model_revision") as resolve:
            with self.assertRaisesRegex(RuntimeError, "separate Python 3.12"):
                load_florence("repo", "main", cache_dir=None, local_files_only=False, dtype=torch.float32)
            resolve.assert_not_called()
        with patch("judges.florence2.SUPPORTED_TRANSFORMERS", importlib.metadata.version("transformers")), \
             patch("judges.florence2.resolve_model_revision", return_value="a" * 40), \
             patch("transformers.AutoProcessor.from_pretrained", side_effect=RuntimeError("missing custom code")):
            with self.assertRaisesRegex(RuntimeError, "Cannot load.*missing custom code"):
                load_florence("repo", "main", cache_dir=None, local_files_only=True, dtype=torch.float32)

    def test_cli_fresh_and_resume_keeps_null_ranks(self):
        folder = self.root / "part1/001_cat"
        folder.mkdir(parents=True)
        (folder / "experiment.json").write_text('{"word":"cat"}')
        Image.new("RGB", (10, 8), "white").save(folder / "clipdraw_final.png")
        words = self.root / "words.csv"
        words.write_text("word\ncat\n")
        checkpoint = self.root / "cli.checkpoint"
        common = [str(self.root / "part1"), "--limit", "3", "--words", str(words),
                  "--checkpoint", str(checkpoint)]
        with patch("judges.run_florence2.Florence2Judge", return_value=self.judge()):
            main(common + ["--output", str(self.root / "captions.json")])
            self.model.generate.reset_mock()
            main(common + ["--resume", "--output", str(self.root / "captions.csv")])
            self.model.generate.assert_not_called()
        row = json.loads((self.root / "captions.json").read_text())[0]
        self.assertIsNone(row["target_rank"])
        self.assertIsNone(row["target_score"])

    def test_validation_cpu_fallback_and_unresolved_commit(self):
        for kwargs in ({"max_new_tokens": 0}, {"num_beams": 0}, {"dtype": "float16"}):
            with self.assertRaises(ValueError):
                self.judge(**kwargs)
        with patch("torch.cuda.is_available", return_value=False):
            judge = Florence2Judge(model=self.model, processor=self.processor)
            self.assertEqual(str(judge.device), "cpu")
            self.assertEqual(judge.configuration["dtype"], "float32")
        self.model.config._commit_hash = None
        with self.assertRaisesRegex(ValueError, "immutable 40-character"):
            self.judge()


if __name__ == "__main__":
    unittest.main()
