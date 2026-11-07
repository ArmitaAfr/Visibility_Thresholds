"""No GPUs, downloads, subprocess installs or model initialization in these tests."""

import json
import hashlib
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import Mock, patch

from judges import EvaluationResult, JudgeOutput, discover_experiments
from judges.colab_setup import CUDA_INDEX, PYTHON_VERSION, environment_spec, install_commands, validate_lock
from judges.prepare_colab import prepare_bundle
from judges.scaling import benchmark_summary, run_job, select_experiments


class ScalingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.inputs = self.root / "inputs"
        for i in range(4):
            folder = self.inputs / f"part{i % 2 + 1}" / str(i)
            folder.mkdir(parents=True)
            (folder / "experiment.json").write_text(json.dumps({"word": "cat", "seed": i}))
            (folder / "clipdraw_final.png").write_bytes(f"fixture-{i}".encode())
        self.words = self.root / "words.csv"
        self.words.write_text("word\ncat\ndog\n")
        self.config = dict(model="siglip2", revision="a" * 40, job_name="pilot_v1",
                           benchmark_tag="benchmark_01", input_roots=[str(self.inputs / "part1"), str(self.inputs / "part2")],
                           output_root=str(self.root / "results"), words_source=str(self.words),
                           device="cpu", batch_size=2, benchmark_images=3, warmup_images=1)
        output = JudgeOutput({"configuration": {"revision": "a" * 40}}, "cat", ["cat", "dog"], [1.0, 0.0])
        self.judge = Mock(model_name="siglip2@" + "a" * 40)
        self.judge.checkpoint_configuration.return_value = {"revision": "a" * 40}
        self.judge.predict.return_value = output
        self.judge.predict_batch.side_effect = lambda paths, candidates: [output for _ in paths]

    def run_job(self, mode="run", resume=False):
        with patch("judges.scaling.build_judge", return_value=self.judge), patch("builtins.print"):
            return run_job(self.config, mode, resume=resume)

    def test_multiple_roots_production_resume_skips_images(self):
        directory = self.run_job()
        self.assertEqual(len(json.loads((directory / "results.json").read_text())), 4)
        self.judge.predict_batch.reset_mock()
        self.run_job(resume=True)
        self.judge.predict_batch.assert_not_called()
        with self.assertRaises(FileExistsError):
            self.run_job()

    def test_changed_cohort_config_or_source_rejected(self):
        self.run_job()
        self.config["batch_size"] = 1
        with self.assertRaisesRegex(ValueError, "Job/source/environment/cohort"):
            self.run_job(resume=True)
        self.config["batch_size"] = 2
        with patch("judges.scaling.source_fingerprint", return_value={"changed.py": "new"}):
            with self.assertRaisesRegex(ValueError, "Job/source/environment/cohort"):
                self.run_job(resume=True)

    def test_production_interruption_resumes_only_remaining_batch(self):
        good = self.judge.predict.return_value
        self.judge.predict_batch.side_effect = [[good, good], KeyboardInterrupt()]
        with self.assertRaises(KeyboardInterrupt):
            self.run_job()
        self.judge.predict_batch.reset_mock()
        self.judge.predict_batch.side_effect = lambda paths, candidates: [good for _ in paths]
        directory = self.run_job(resume=True)
        self.assertEqual(self.judge.predict_batch.call_count, 1)
        self.assertEqual(len(json.loads((directory / "results.json").read_text())), 4)

    def test_benchmark_warmup_is_excluded_and_outputs_separate(self):
        directory = self.run_job("benchmark")
        report = json.loads((directory / "benchmark.json").read_text())
        self.assertEqual(report["sample_images"], 3)
        self.assertEqual(set(report["estimates"]), {"150", "2000", "2500"})
        self.assertGreater(report["evaluation_images_per_second"], 0)
        self.assertEqual(self.judge.predict_batch.call_count, 3)  # Warmup plus two measured batches.
        production = self.run_job()
        self.assertNotEqual(production, directory)
        with self.assertRaisesRegex(ValueError, "Benchmarks cannot resume"):
            self.run_job("benchmark", resume=True)

    def test_runtime_estimate_math(self):
        row = EvaluationResult("id", "cat", "image", "meta", "mock", {}, "cat", 1, 2.0)
        report = benchmark_summary([row, row], 5.0, 10.0, 1.0, [150, 2000])
        self.assertEqual(report["inference_images_per_second"], .5)
        self.assertEqual(report["estimates"]["150"]["inference_seconds"], 300)
        self.assertEqual(report["estimates"]["2000"]["evaluation_seconds"], 5000)
        self.assertEqual(report["estimates"]["2000"]["evaluation_plus_model_load_seconds"], 5010)
        with self.assertRaises(ValueError):
            benchmark_summary([], 1, 0, 0, [150])

    def test_shards_disjoint_complete_and_root_order_independent(self):
        images = discover_experiments(self.inputs).experiments
        one = select_experiments(images, 1, 2)
        two = select_experiments(list(reversed(images)), 2, 2)
        self.assertEqual({e.image_id for e in one + two}, {e.image_id for e in images})
        self.assertFalse({e.image_id for e in one} & {e.image_id for e in two})
        with self.assertRaises(ValueError):
            select_experiments(images, 0, 2)

    def test_pinned_revisions_required(self):
        self.config["revision"] = "main"
        with self.assertRaisesRegex(ValueError, "immutable commit"):
            self.run_job()

    def test_environment_command_plan_isolated_and_cuda_matched(self):
        commands = install_commands("uv", "/content/judge_envs/florence2/bin/python", "requirements.txt")
        self.assertIn("--python", commands[0])
        self.assertIn("torch==2.6.0", commands[0])
        self.assertIn("torchvision==0.21.0", commands[0])
        self.assertIn(CUDA_INDEX, commands[0])
        self.assertNotIn("--system", commands[0])
        spec = environment_spec("florence2", Path(__file__).resolve().parents[2])
        self.assertEqual(spec["python"], PYTHON_VERSION)

    def test_source_bundle_contains_current_code_and_no_results(self):
        output = prepare_bundle(self.root / "source.zip")
        with zipfile.ZipFile(output) as archive:
            names = archive.namelist()
            self.assertIn("judges_colab.ipynb", names)
            self.assertIn("judges/florence2.py", names)
            self.assertIn("dataset/processed/sampled_words.csv", names)
            self.assertFalse(any(name.startswith(("judge_results/", "batch/")) for name in names))
            manifest = json.loads(archive.read("source_manifest.json"))
            self.assertEqual(set(names), set(manifest) | {"source_manifest.json"})
        with self.assertRaises(FileExistsError):
            prepare_bundle(output)

    def test_environment_lock_validated_without_installing_packages(self):
        spec = {"model": "florence2", "python": PYTHON_VERSION}
        freeze = "torch==2.6.0+cu124\ntransformers==4.49.0\n"
        lock = {"spec": spec, "freeze": freeze, "freeze_sha256": hashlib.sha256(freeze.encode()).hexdigest()}
        validate_lock(lock, spec)
        with self.assertRaisesRegex(ValueError, "changed or damaged"):
            validate_lock(lock, {**spec, "python": "3.13.0"})
        lock["freeze"] += "tampered==1\n"
        with self.assertRaisesRegex(ValueError, "changed or damaged"):
            validate_lock(lock, spec)

    def test_notebook_json_and_python_cells_valid_and_no_outputs(self):
        path = Path(__file__).resolve().parents[2] / "judges_colab.ipynb"
        notebook = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(notebook["nbformat"], 4)
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                compile(cell["source"], cell["id"], "exec")
                self.assertEqual(cell["outputs"], [])
                self.assertIsNone(cell["execution_count"])


if __name__ == "__main__":
    unittest.main()
