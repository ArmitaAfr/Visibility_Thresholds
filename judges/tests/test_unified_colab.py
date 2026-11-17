"""CPU orchestration checks. Source bundles and run folders are temporary."""
import base64
import hashlib
import json
import subprocess
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import Mock, patch

import unified_colab as vt
from judges import JudgeOutput, prepare_colab, scaling

NOTEBOOK = vt.SRC / "visibility_thresholds_colab.ipynb"
CELLS = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
CODE = ["".join(c["source"]) for c in CELLS if c["cell_type"] == "code"]
PNG = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=")


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.drive = Path(tmp.name)
        self.csv = self.drive / "words.csv"
        self.csv.write_text('word,score_bin,glasgow_imageability_mean\ncat,"[1, 3]",2\ndog,"[1, 3]",2\n')
        self.kw = dict(drive_root=self.drive, run_id="smoke", cohort_id="pilot", source_hash="a"*64,
                       generation_csv=self.csv, vocabulary_csv=self.csv, metadata_csv=self.csv)

    def test_notebook_cpu_stages_really_skip_compute(self):
        for code in CODE:
            compile(code, str(NOTEBOOK), "exec")
        self.assertEqual(len([c for c in CELLS if c["cell_type"] == "markdown"]), 12)
        self.assertTrue(all(c["execution_count"] is None and c["outputs"] == []
                            for c in CELLS if c["cell_type"] == "code"))
        for stage in ("PREFLIGHT", "ALL"):
            ns = dict(vt=vt, SRC=vt.SRC, DRIVE_ROOT=self.drive, ARCHIVE_SHA256="a"*64,
                      Path=Path, re=vt.re)
            exec(CODE[0], ns)
            self.assertEqual(ns["STAGE"], "PREFLIGHT")
            ns["STAGE"] = stage
            with patch("subprocess.run", side_effect=AssertionError("unexpected subprocess")):
                for code in CODE[3:]:
                    exec(code, ns)
            self.assertFalse((self.drive / "runs").exists())
        for stage in ("GENERATE", "JUDGE_SIGLIP2", "JUDGE_FLORENCE2", "TEXT_SCORE", "unknown"):
            with self.assertRaises(ValueError):
                vt.select_stage(stage, "")

    def test_source_setup_requires_the_reviewed_release_commit(self):
        instructions = "".join(next(c["source"] for c in CELLS if c["id"] == "qmind-05"))
        self.assertIn("--unified-base (git rev-parse HEAD)", instructions)
        ns = dict(DRIVE_ROOT=self.drive)
        exec(CODE[2].split("SRC = verify_bundle(")[0], ns)
        self.assertEqual(ns["BASE_COMMIT"], "")
        with self.assertRaisesRegex(ValueError, "reviewed release commit"):
            ns["verify_bundle"](self.drive / "unused.zip", "a"*64, ns["BASE_COMMIT"], self.drive)

    def test_archive_cli_and_notebook_verification(self):
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=vt.SRC, text=True).strip()
        archive = self.drive / "src.zip"
        with self.assertRaises(ValueError):
            prepare_colab.main(["--unified-base", "0"*40, "--output", str(archive)])
        prepare_colab.main(["--unified-base", head, "--output", str(archive)])
        with zipfile.ZipFile(archive) as z:
            self.assertIn("judges/experiment_manifest.py", z.namelist())
            self.assertIn("text_score/run_manifest.py", z.namelist())
            self.assertIn("judges/analysis/combined.py", z.namelist())
            self.assertFalse(any(n.startswith("judge_results/") or n.endswith(".png") or
                                 n.startswith("text_score/") and n.endswith(".csv") for n in z.namelist()))
        ns = dict(DRIVE_ROOT=self.drive)
        exec(CODE[2].split("SRC = verify_bundle(")[0], ns)
        verify = ns["verify_bundle"]
        sha = hashlib.sha256(archive.read_bytes()).hexdigest()
        for wrong_digest, wrong_base in [(sha, "0"*40), ("a"*64, head)]:
            with self.assertRaises(ValueError):
                verify(archive, wrong_digest, wrong_base, self.drive / "local")
        source = verify(archive, sha, head, self.drive / "local")
        self.assertEqual((source / "clipdraw.ipynb").read_bytes(), (vt.SRC / "clipdraw.ipynb").read_bytes())
        (source / "unified_colab.py").write_text("# drift")
        with self.assertRaises(ValueError):
            verify(archive, sha, head, self.drive / "local")

    def test_manifest_persistence_and_shard_guard(self):
        inputs = self.drive / "existing"
        for i in range(2):
            folder = inputs / f"part{i}" / "001_cat"
            folder.mkdir(parents=True)
            (folder / "clipdraw_final.png").write_bytes(PNG)
            vt.create(folder / "experiment.json", {"word": "cat", "seed": i})
        flow = vt.Workflow(**self.kw, input_roots=[inputs], shards=2)
        flow.bind("MANIFEST")
        manifest = flow.manifest(publish=True)
        cfg, _ = flow.job("siglip2", 2, manifest)
        self.assertEqual(cfg["expected_image_ids"], [manifest["images"][1]["image_id"]])
        with patch.object(scaling, "build_judge", side_effect=AssertionError("model loaded")):
            with self.assertRaisesRegex(ValueError, "shard differs"):
                scaling.run_job({**cfg, "expected_image_ids": []})
            with self.assertRaisesRegex(ValueError, "vocabulary differs"):
                scaling.run_job({**cfg, "expected_vocabulary_sha256": "wrong"})
        self.assertEqual(flow.manifest(), manifest)
        with self.assertRaisesRegex(ValueError, "Different or unrecognized"):
            vt.Workflow(**self.kw, input_roots=[inputs], shards=1).bind("ANALYZE")
        (folder / "experiment.json").write_text('{"word":"cat","seed":42}')
        with self.assertRaisesRegex(ValueError, "inputs differ"):
            flow.manifest()
        self.assertEqual(vt.read(flow.manifest_path), manifest)

    def test_generation_assignment_isolation_and_resume(self):
        flow = vt.Workflow(**self.kw, mode="smoke_test", people=2)
        self.assertEqual(flow.assignment(2), ["dog"])
        with self.assertRaises(ValueError):
            flow.generate(2, resume=True)
        with self.assertRaises(ValueError):
            vt.check_job(self.drive, True, "job_manifest.json")
        with self.assertRaises(FileExistsError):
            vt.check_job(self.drive, False, "job_manifest.json")
        (self.drive / "results.json").touch()
        with self.assertRaises(FileExistsError):
            vt.check_job(self.drive, True, "run.json")
        with patch("unified_colab.subprocess.run") as call:
            env = vt.uv_launcher()
            self.assertIn("--target", call.call_args.args[0])
            self.assertIn(str(Path("/content/qmind-uv")), env["PATH"])
        # Judge environment setup runs outside the kernel. The second command uses its interpreter.
        manifest = {"images": [{"image_id": "abc"}], "csv_sha256": {"vocabulary_csv": "test"}}
        with patch.object(flow, "manifest", return_value=manifest), patch("unified_colab.uv_launcher"), patch("subprocess.run") as run:
            flow.judge("florence2", 1, False)
            setup, score = [args.args[0] for args in run.call_args_list]
            self.assertIn("judges.colab_setup", setup)
            self.assertIn("judges.scaling", score)
            self.assertEqual(Path(score[0]).as_posix(), "/content/judge_envs/florence2/bin/python")
        with patch("unified_colab.subprocess.run") as run:
            with self.assertRaisesRegex(ValueError, "verified full commit"):
                flow.text_score("", False)
            run.assert_not_called()


    def test_saved_exports_feed_analysis_without_inference(self):
        inputs = self.drive / "part1" / "001_cat"
        inputs.mkdir(parents=True)
        (inputs / "clipdraw_final.png").write_bytes(PNG)
        vt.create(inputs / "experiment.json", {"word": "cat"})
        flow = vt.Workflow(**self.kw, input_roots=[inputs.parent])
        flow.bind("MANIFEST")
        manifest = flow.manifest(publish=True)
        ranking = [{"word": "cat", "score": 1}, {"word": "dog", "score": 0}]
        outputs = {
            "siglip2": JudgeOutput({"configuration": {"candidate_vocabulary_size": 2,
                "candidate_vocabulary_sha256": hashlib.sha256(json.dumps(["cat", "dog"]).encode()).hexdigest()}, "top5": ranking, "candidate_scores": ranking},
                "cat", ["cat", "dog"], [1, 0]),
            "florence2": JudgeOutput({"configuration": {"task_prompt": "<CAPTION>"}, "caption": "a cat"}, "a cat"),
        }
        for model, output in outputs.items():
            cfg, directory = flow.job(model, 1, manifest)
            model_id = "google/siglip2-base-patch16-224" if model == "siglip2" else "microsoft/Florence-2-large-ft"
            adapter = Mock(model_name=model_id + "@" + cfg["revision"])
            output.raw_output["configuration"]["resolved_revision"] = cfg["revision"]
            adapter.checkpoint_configuration.return_value = {"revision": cfg["revision"]}
            adapter.predict.return_value = output
            adapter.predict_batch.return_value = [output]
            with patch.object(scaling, "build_judge", return_value=adapter), patch.object(scaling, "cuda_preflight"):
                scaling.run_job(cfg)
            self.assertTrue((directory / "complete.json").is_file())
        with self.assertRaisesRegex(ValueError, "text_score: 1 missing"):
            flow.analyze()
        dest = flow.analyze(partial=True)
        summary = vt.read(dest / "summary.json")
        combined = vt.read(dest / "combined_summary.json")
        self.assertEqual(combined["coverage"]["text_score"]["missing_ids"], [manifest["images"][0]["image_id"]])
        self.assertEqual(combined["overall"]["text_score"]["n"], 0)
        self.assertEqual(summary["matched_images"], 1)
        self.assertEqual(summary["imageability"]["groups"]["low"]["evaluated_images"], 1)
        (directory / "results.json").write_text("[]")
        with self.assertRaisesRegex(ValueError, "Export or job configuration differs"):
            flow.analyze()
