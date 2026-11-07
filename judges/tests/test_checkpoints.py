"""Checkpoint/resume tests use only standard-library mock adapters."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from judges import EvaluationFailure, Experiment, JudgeOutput, evaluate, export_results
from judges.checkpoints import atomic_write


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.checkpoint = self.root / "checkpoints"
        self.experiments = []
        for index in range(4):
            folder = self.root / f"source{index}"
            folder.mkdir()
            image = folder / "clipdraw_final.png"
            image.write_bytes(f"image{index}".encode())
            metadata = folder / "experiment.json"
            metadata.write_text('{"word":"cat"}')
            self.experiments.append(Experiment(str(index), "cat", image, metadata))
        self.output = JudgeOutput({"score": 1.0}, "cat", ["cat", "dog"], [1.0, 0.0])
        self.judge = Mock(model_name="mock@commit", predict=Mock(return_value=self.output))
        self.judge.predict_batch.side_effect = lambda paths, candidates: [self.output for _ in paths]
        self.config = {"checkpoint": "commit", "prompt_template": "A drawing of {}.", "dtype": "float32"}

    def run_evaluation(self, *, resume=False, experiments=None, candidates=("cat", "dog"), **kwargs):
        return evaluate(self.experiments if experiments is None else experiments,
                        candidates, self.judge, batch_size=2, checkpoint_dir=self.checkpoint,
                        resume=resume, checkpoint_config=self.config, **kwargs)

    def records(self):
        return [p for p in self.checkpoint.glob("*.json") if p.name != "manifest.json"]

    def test_interruption_saves_prior_batch_and_resume_skips_completed(self):
        self.judge.predict_batch.side_effect = [[self.output, self.output], KeyboardInterrupt()]
        with self.assertRaises(KeyboardInterrupt):
            self.run_evaluation()
        self.assertEqual(len(self.records()), 2)
        self.judge.predict_batch.reset_mock(side_effect=True)
        self.judge.predict_batch.return_value = [self.output, self.output]
        rows = self.run_evaluation(resume=True)
        self.judge.predict_batch.assert_called_once()
        self.assertEqual([str(p) for p in self.judge.predict_batch.call_args.args[0]],
                         [str(e.image_path) for e in self.experiments[2:]])
        self.assertEqual(len(rows), 4)
        self.judge.predict_batch.reset_mock()
        self.assertEqual(self.run_evaluation(resume=True), rows)
        self.judge.predict_batch.assert_not_called()

    def test_changed_configuration_vocabulary_and_model_rejected(self):
        self.run_evaluation()
        self.config["prompt_template"] = "A photo of {}."
        with self.assertRaisesRegex(ValueError, "configuration changed"):
            self.run_evaluation(resume=True)
        self.config["prompt_template"] = "A drawing of {}."
        for words in [("dog", "cat"), ("cat", "dog", "fox")]:
            with self.assertRaisesRegex(ValueError, "configuration changed"):
                self.run_evaluation(resume=True, candidates=words)
        self.judge.model_name = "mock@new-commit"
        with self.assertRaisesRegex(ValueError, "configuration changed"):
            self.run_evaluation(resume=True)

    def test_image_and_metadata_changes_recompute_only_changed_record(self):
        self.run_evaluation()
        for path in (self.experiments[0].image_path, self.experiments[0].metadata_path):
            path.write_bytes(path.read_bytes() + b" ")
            self.judge.predict_batch.reset_mock()
            self.judge.predict_batch.side_effect = lambda paths, candidates: [self.output for _ in paths]
            self.run_evaluation(resume=True)
            self.assertEqual(self.judge.predict_batch.call_count, 1)
            self.assertEqual(self.judge.predict_batch.call_args.args[0], [self.experiments[0].image_path])

    def test_failed_batch_retries_individuals_saves_good_and_retries_failure_on_resume(self):
        self.judge.predict_batch.side_effect = RuntimeError("batch failure")

        def predict(path, candidates):
            if path == self.experiments[1].image_path:
                raise ValueError("unreadable image")
            return self.output

        self.judge.predict.side_effect = predict
        with self.assertRaises(EvaluationFailure) as failure:
            self.run_evaluation()
        self.assertEqual(len(failure.exception.results), 3)
        self.assertIn("unreadable image", str(failure.exception))
        self.assertEqual(len(self.records()), 3)
        self.judge.predict_batch.side_effect = lambda paths, candidates: [self.output for _ in paths]
        self.judge.predict_batch.reset_mock()
        self.assertEqual(len(self.run_evaluation(resume=True)), 4)
        self.assertEqual(self.judge.predict_batch.call_args.args[0], [self.experiments[1].image_path])

    def test_partial_valid_batch_saves_valid_outputs(self):
        bad = JudgeOutput({}, "cat", ["cat"])
        self.judge.predict_batch.side_effect = [[self.output, bad], [self.output, self.output]]
        with self.assertRaises(EvaluationFailure):
            self.run_evaluation()
        self.assertEqual(len(self.records()), 3)

    def test_corrupt_record_or_manifest_never_used(self):
        self.run_evaluation()
        record = self.records()[0]
        original = record.read_bytes()
        payload = json.loads(original)
        payload["payload"]["result"]["normalized_prediction"] = "tampered"
        record.write_text(json.dumps(payload))
        with self.assertRaisesRegex(ValueError, "Damaged checkpoint"):
            self.run_evaluation(resume=True)
        record.write_bytes(original)
        (self.checkpoint / "manifest.json").write_text("{")
        with self.assertRaisesRegex(ValueError, "manifest"):
            self.run_evaluation(resume=True)

    def test_atomic_replace_interruption_preserves_previous_file(self):
        path = self.root / "atomic.json"
        atomic_write(path, {"old": True})
        with patch("judges.checkpoints.os.replace", side_effect=OSError("interrupted replace")):
            with self.assertRaises(OSError):
                atomic_write(path, {"new": True})
        self.assertEqual(json.loads(path.read_text()), {"old": True})
        self.assertEqual(list(self.root.glob(".tmp-*")), [])

    def test_interruption_between_record_writes_resumes_remaining_records(self):
        from judges.checkpoints import CheckpointStore
        save = CheckpointStore.save
        count = 0

        def interrupted(store, *args):
            nonlocal count
            count += 1
            if count == 2:
                raise KeyboardInterrupt()
            save(store, *args)

        with patch.object(CheckpointStore, "save", interrupted):
            with self.assertRaises(KeyboardInterrupt):
                self.run_evaluation()
        self.assertEqual(len(self.records()), 1)
        self.assertEqual(len(self.run_evaluation(resume=True)), 4)

    def test_sources_preserved_exports_and_explicit_resume_required(self):
        before = {p: p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        rows = self.run_evaluation()
        for extension in ("csv", "json"):
            export_results(rows, self.root / f"results.{extension}")
        self.assertTrue(all(path.read_bytes() == contents for path, contents in before.items()))
        with self.assertRaises(FileExistsError):
            self.run_evaluation()
        with self.assertRaises(ValueError):
            evaluate(self.experiments, ("cat", "dog"), self.judge, resume=True)

    def test_source_change_during_inference_is_not_checkpointed(self):
        def predict(paths, candidates):
            paths[0].write_bytes(b"changed during inference")
            return [self.output for _ in paths]
        self.judge.predict_batch.side_effect = predict
        with self.assertRaisesRegex(ValueError, "Source changed during inference"):
            self.run_evaluation()
        self.assertEqual(self.records(), [])


if __name__ == "__main__":
    unittest.main()
