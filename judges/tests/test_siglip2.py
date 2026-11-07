"""Offline CPU tests: no from_pretrained call is allowed to reach the network."""

import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from judges import EvaluationResult, Experiment, JudgeOutput, evaluate, export_results
from judges.siglip2 import SigLIP2Judge
from judges.run_siglip2 import main

HAS_DEPS = all(importlib.util.find_spec(name) for name in ("torch", "PIL", "transformers"))
if HAS_DEPS:
    import torch
    from PIL import Image


@unittest.skipUnless(HAS_DEPS, "Install judges/requirements-siglip2.txt for adapter tests.")
class SigLIP2Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.image = self.root / "image.png"
        Image.new("RGB", (8, 8), "white").save(self.image)
        self.words = ("ant", "bear", "cat", "dog", "eel", "fox")
        self.processor = Mock()
        self.processor.image_processor.to_dict.return_value = {"size": {"height": 224, "width": 224}}
        self.processor.side_effect = self.process
        self.model = Mock()
        self.model.float.return_value = self.model
        self.model.to.return_value = self.model
        self.model.eval.return_value = self.model
        self.model.config = SimpleNamespace(_commit_hash="a" * 40)
        self.model.logit_scale = torch.tensor(math.log(2.0))
        self.model.logit_bias = torch.tensor(-1.0)
        self.model.get_text_features.side_effect = self.text_features
        self.model.get_image_features.side_effect = self.image_features

    def process(self, *, text=None, images=None, **kwargs):
        if text is not None:
            self.assertEqual(kwargs, dict(padding="max_length", max_length=64,
                                         truncation=True, return_tensors="pt"))
            ids = [[self.words.index(prompt.removeprefix("a drawing of a ").removesuffix("."))]
                   for prompt in text]
            return {"input_ids": torch.tensor(ids)}
        self.assertTrue(all(image.mode == "RGB" for image in images))
        return {"pixel_values": torch.ones(len(images), 2)}

    def text_features(self, input_ids):
        self.assertFalse(torch.is_grad_enabled())
        return torch.cat([input_ids.float() + 1, torch.ones_like(input_ids)], dim=1)

    def image_features(self, pixel_values):
        self.assertFalse(torch.is_grad_enabled())
        return torch.tensor([[1.0, 0.0]]).repeat(len(pixel_values), 1)

    def judge(self, **kwargs):
        return SigLIP2Judge(self.words, model=self.model, processor=self.processor,
                           device="cpu", text_batch_size=2, **kwargs)

    def test_ranking_top5_scores_and_cached_text(self):
        judge = self.judge()
        self.assertEqual(self.model.get_text_features.call_count, 3)
        output = judge.predict(self.image, self.words)
        judge.predict_batch([self.image, self.image], self.words)
        self.assertEqual(self.model.get_text_features.call_count, 3)
        self.assertEqual(output.ranked_candidates, list(reversed(self.words)))
        self.assertEqual(output.normalized_prediction, "fox")
        top = output.raw_output["top5"]
        self.assertEqual([entry["word"] for entry in top], ["fox", "eel", "dog", "cat", "bear"])
        self.assertAlmostEqual(top[0]["score"], 2 * 6 / math.sqrt(37) - 1, places=6)
        self.assertEqual(len(output.raw_output["candidate_scores"]), 6)
        self.model.eval.assert_called_once()
        self.model.requires_grad_.assert_called_once_with(False)
        self.assertEqual(output.raw_output["configuration"]["candidate_vocabulary_size"], 6)
        json.dumps(output.raw_output, allow_nan=False)

    def test_pipeline_target_rank_score_batching_exports_and_sources(self):
        before = self.image.read_bytes()
        experiments = [Experiment(str(i), "CaT", self.image, self.root / "experiment.json") for i in range(5)]
        results = evaluate(experiments, self.words, self.judge(), batch_size=2)
        self.assertEqual(self.model.get_image_features.call_count, 3)
        self.assertTrue(all(isinstance(row, EvaluationResult) for row in results))
        self.assertTrue(all(row.target_rank == 4 for row in results))
        self.assertAlmostEqual(results[0].target_score, 2 * 3 / math.sqrt(10) - 1, places=6)
        self.assertTrue(all(row.inference_time_seconds >= 0 for row in results))
        for extension in ("csv", "json"):
            export_results(results, self.root / f"results.{extension}")
        self.assertEqual(self.image.read_bytes(), before)

    def test_tie_breaking_original_order_and_fewer_than_five(self):
        self.model.get_text_features.side_effect = lambda **kwargs: torch.ones(len(kwargs["input_ids"]), 2)
        judge = self.judge()
        self.assertEqual(judge.predict(self.image, self.words).ranked_candidates, list(self.words))
        short = SigLIP2Judge(self.words[:2], model=self.model, processor=self.processor, device="cpu")
        self.assertEqual(len(short.predict(self.image, self.words[:2]).raw_output["top5"]), 2)

    def test_vocabulary_cannot_change_and_validation(self):
        judge = self.judge()
        for words in (self.words[::-1], self.words[:-1], tuple(w.upper() for w in self.words)):
            with self.assertRaisesRegex(ValueError, "vocabulary/order"):
                judge.predict(self.image, words)
        for kwargs in (dict(text_batch_size=0), dict(prompt_template="no placeholder"),
                       dict(prompt_template="{} {}"), dict(device="mps")):
            with self.assertRaises(ValueError):
                SigLIP2Judge(self.words, model=self.model, processor=self.processor, **kwargs)
        for words in ([], ["dog", "DOG"], [" dog"], [""]):
            with self.assertRaises(ValueError):
                SigLIP2Judge(words, model=self.model, processor=self.processor)

    def test_unreadable_images_model_errors_and_nonfinite_scores(self):
        judge = self.judge()
        bad = self.root / "bad.png"
        bad.write_bytes(b"not-an-image")
        for path in (bad, self.root / "missing.png"):
            with self.assertRaisesRegex(RuntimeError, "Cannot read image"):
                judge.predict(path, self.words)
        self.model.get_image_features.side_effect = RuntimeError("mock model failure")
        with self.assertRaisesRegex(RuntimeError, "mock model failure"):
            judge.predict(self.image, self.words)
        self.model.get_image_features.side_effect = self.image_features
        self.model.logit_bias = torch.tensor(float("nan"))
        with self.assertRaisesRegex(RuntimeError, "invalid compatibility scores"):
            judge.predict(self.image, self.words)

    def test_pooled_feature_api_and_invalid_text_features(self):
        self.model.get_text_features.side_effect = lambda **kw: SimpleNamespace(pooler_output=self.text_features(**kw))
        self.model.get_image_features.side_effect = lambda **kw: SimpleNamespace(pooler_output=self.image_features(**kw))
        self.assertEqual(self.judge().predict(self.image, self.words).normalized_prediction, "fox")
        self.model.get_text_features.side_effect = lambda **kw: torch.zeros(len(kw["input_ids"]), 2)
        with self.assertRaisesRegex(ValueError, "invalid feature"):
            self.judge()

    def test_model_processor_load_once_matching_revision_cpu_fallback(self):
        with patch("transformers.AutoModel.from_pretrained", return_value=self.model) as load_model, \
             patch("transformers.AutoProcessor.from_pretrained", return_value=self.processor) as load_processor, \
             patch("judges.siglip2.resolve_model_revision", return_value="a" * 40), \
             patch("torch.cuda.is_available", return_value=False):
            judge = SigLIP2Judge(self.words)
            judge.predict(self.image, self.words)
            judge.predict(self.image, self.words)
            load_model.assert_called_once()
            load_processor.assert_called_once_with("google/siglip2-base-patch16-224", revision="a" * 40,
                                                   cache_dir=None, local_files_only=False)
            self.assertEqual(str(judge.device), "cpu")
            with self.assertRaisesRegex(RuntimeError, "CUDA was requested"):
                SigLIP2Judge(self.words, device="cuda")

    def test_missing_private_commit_resolved_before_loading_and_resume_compatible(self):
        from judges.checkpoints import CheckpointStore

        self.model.config = SimpleNamespace()  # Mirrors the real reported failure.
        options = dict(revision="main", device="cpu", cache_dir=self.root / "hub",
                       local_files_only=True)
        with patch("judges.siglip2.resolve_model_revision", return_value="a" * 40) as resolve, \
             patch("transformers.AutoModel.from_pretrained", return_value=self.model) as load_model, \
             patch("transformers.AutoProcessor.from_pretrained", return_value=self.processor) as load_processor:
            judge = SigLIP2Judge(self.words, **options)
            expected = dict(revision="a" * 40, cache_dir=self.root / "hub", local_files_only=True)
            load_model.assert_called_once_with("google/siglip2-base-patch16-224",
                                               attn_implementation="eager", **expected)
            load_processor.assert_called_once_with("google/siglip2-base-patch16-224", **expected)
            resolve.assert_called_once_with("google/siglip2-base-patch16-224", "main",
                                             cache_dir=self.root / "hub", local_files_only=True)
            configuration = judge.checkpoint_configuration()
            self.assertEqual(configuration["resolved_revision"], "a" * 40)
            self.assertTrue(judge.model_name.endswith("@" + "a" * 40))
            checkpoint = self.root / "revision-checkpoint"
            CheckpointStore(checkpoint, configuration, resume=False)
            # The same pinned files resolve identically even without _commit_hash.
            again = SigLIP2Judge(self.words, **options)
            CheckpointStore(checkpoint, again.checkpoint_configuration(), resume=True)
            with patch("judges.siglip2.resolve_model_revision", return_value="b" * 40):
                changed = SigLIP2Judge(self.words, **options)
            with self.assertRaisesRegex(ValueError, "configuration changed"):
                CheckpointStore(checkpoint, changed.checkpoint_configuration(), resume=True)

    def test_conflicting_reported_revision_and_unverified_injection_rejected(self):
        with patch("judges.siglip2.resolve_model_revision", return_value="b" * 40), \
             patch("transformers.AutoModel.from_pretrained", return_value=self.model), \
             patch("transformers.AutoProcessor.from_pretrained") as processor:
            with self.assertRaisesRegex(ValueError, "expected pinned commit"):
                SigLIP2Judge(self.words, device="cpu")
            processor.assert_not_called()
        self.model.config = SimpleNamespace(_commit_hash="main")
        with self.assertRaisesRegex(ValueError, "verified 40-character"):
            self.judge().checkpoint_configuration()

    def test_cli_uses_discovery_limit_fixed_vocab_and_exports(self):
        for i, word in enumerate(self.words[:3]):
            folder = self.root / f"part1/{i}"
            folder.mkdir(parents=True)
            Image.new("RGB", (8, 8), "white").save(folder / "clipdraw_final.png")
            (folder / "experiment.json").write_text(json.dumps({"word": word}))
        words_path = self.root / "words.csv"
        words_path.write_text("word\n" + "\n".join(self.words))
        with patch("judges.run_siglip2.SigLIP2Judge", return_value=self.judge()) as factory:
            main([str(self.root / "part1"), "--words", str(words_path), "--output",
                  str(self.root / "cli.json"), "--limit", "2", "--batch-size", "2"])
            self.assertEqual(tuple(factory.call_args.args[0]), self.words)
        rows = json.loads((self.root / "cli.json").read_text())
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["target_rank"], 6)
        self.assertEqual(rows[0]["raw_output"]["configuration"]["candidate_vocabulary_size"], 6)
        self.model.get_image_features.reset_mock()
        with patch("judges.run_siglip2.SigLIP2Judge", return_value=factory.return_value):
            main([str(self.root / "part1"), "--words", str(words_path), "--output",
                  str(self.root / "cli-resumed.json"), "--limit", "2", "--batch-size", "2",
                  "--checkpoint", str(self.root / "cli.json.checkpoint"), "--resume"])
        self.model.get_image_features.assert_not_called()
        self.assertEqual(json.loads((self.root / "cli-resumed.json").read_text()), rows)

    def test_bad_batch_output_and_scores_rejected(self):
        experiment = Experiment("id", "cat", self.image, self.root / "experiment.json")
        judge = self.judge()
        with patch.object(judge, "predict_batch", return_value=[]):
            with self.assertRaisesRegex(ValueError, "wrong number"):
                evaluate([experiment], self.words, judge, batch_size=2)
        for size in (0, -1, True):
            with self.assertRaisesRegex(ValueError, "batch_size"):
                evaluate([experiment], self.words, judge, batch_size=size)
        for scores in ([0.0], [float("nan")] * 6):
            with patch.object(judge, "predict", return_value=JudgeOutput(
                    {}, "ant", list(self.words), scores)):
                with self.assertRaisesRegex(ValueError, "Candidate scores"):
                    evaluate([experiment], self.words, judge)

    def test_cli_inference_failure_does_not_export(self):
        folder = self.root / "part1/001_cat"
        folder.mkdir(parents=True)
        (folder / "experiment.json").write_text('{"word":"cat"}')
        (folder / "clipdraw_final.png").write_bytes(b"broken PNG")
        words_path = self.root / "words.csv"
        words_path.write_text("word\n" + "\n".join(self.words))
        output = self.root / "never-written.json"
        with patch("judges.run_siglip2.SigLIP2Judge", return_value=self.judge()):
            with self.assertRaisesRegex(RuntimeError, "Cannot read image"):
                main([str(self.root / "part1"), "--words", str(words_path),
                      "--output", str(output)])
        self.assertFalse(output.exists())

    def test_cached_logits_match_tiny_transformers_forward_without_download(self):
        # The fixed-resolution checkpoint uses the SiglipModel architecture.
        # A tiny random instance tests HF feature APIs and logit math offline.
        from transformers import SiglipConfig, SiglipModel

        config = SiglipConfig(
            text_config=dict(vocab_size=16, hidden_size=8, intermediate_size=16,
                             num_hidden_layers=1, num_attention_heads=2, max_position_embeddings=64),
            vision_config=dict(hidden_size=8, intermediate_size=16, num_hidden_layers=1,
                               num_attention_heads=2, image_size=8, patch_size=4),
        )
        model = SiglipModel(config).eval()
        inputs = {"input_ids": torch.arange(6).view(6, 1).repeat(1, 64),
                  "pixel_values": torch.ones(2, 3, 8, 8)}
        with torch.inference_mode():
            expected = model(**inputs).logits_per_image
        processor = Mock()
        processor.image_processor.to_dict.return_value = {"size": 8}

        def process(*, text=None, images=None, **kwargs):
            if text is not None:
                indices = [self.words.index(p.removeprefix("a drawing of a ").removesuffix(".")) for p in text]
                return {"input_ids": inputs["input_ids"][indices]}
            return {"pixel_values": inputs["pixel_values"][:len(images)]}

        processor.side_effect = process
        judge = SigLIP2Judge(self.words, model=model, processor=processor, device="cpu", text_batch_size=2)
        outputs = judge.predict_batch([self.image, self.image], self.words)
        for row, output in zip(expected, outputs):
            by_word = {entry["word"]: entry["score"] for entry in output.raw_output["candidate_scores"]}
            actual = torch.tensor([by_word[w] for w in self.words])
            torch.testing.assert_close(actual, row, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
