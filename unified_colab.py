"""Local orchestration for visibility_thresholds_colab.ipynb. Imports no ML models."""

import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from dataset.validate_dataset import validate_dataset
from judges.checkpoints import digest, file_digest
from judges.experiment_manifest import ExperimentConfig, build_manifest

SRC = Path(__file__).resolve().parent
PINS = {"siglip2": "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2",
        "florence2": "4a12a2b54b7016a48a22037fbd62da90cd566f2a"}
STAGES = ("PREFLIGHT", "GENERATE", "MANIFEST", "JUDGE_SIGLIP2", "JUDGE_FLORENCE2",
          "TEXT_SCORE", "ANALYZE", "ALL")


def select_stage(stage, confirmed):
    if stage not in STAGES:
        raise ValueError(f"Choose one of {STAGES}.")
    if stage in ("GENERATE", "JUDGE_SIGLIP2", "JUDGE_FLORENCE2", "TEXT_SCORE") and confirmed != stage:
        raise ValueError(f"To authorize setup, downloads and execution, type {stage} in CONFIRM.")
    return stage


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def create(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        json.dump(data, f, indent=2, allow_nan=False)


def consistent(path, data):
    if path.exists():
        if read(path) != data:
            raise ValueError(f"Configuration or inputs differ from {path}. Choose a new run.")
    else:
        create(path, data)


def dataset_check(processed):
    report = validate_dataset(processed)
    errors = [i for i in report["issues"] if i["severity"] == "error"]
    if any((i["code"], i["file"]) != ("missing_file", "cleaned_words.csv") for i in errors):
        raise ValueError(f"Dataset validation failed: {errors[:5]}")
    print("Dataset validation:", report["status"], [i["message"] for i in errors])
    return report


def checked_manifest(cfg):
    manifest = build_manifest(cfg)
    if manifest["issues"] or not manifest["images"]:
        raise ValueError(f"Image verification failed: {manifest['issues'][:5]}")
    return manifest


class Workflow:
    def __init__(self, *, drive_root, run_id, cohort_id, generation_csv, vocabulary_csv,
                 metadata_csv, source_hash, input_roots=(), people=1, mode="full", shards=1):
        if not re.fullmatch("[A-Za-z0-9_-]+", run_id):
            raise ValueError("Run ID must contain only letters, digits, underscores or hyphens.")
        if type(shards) is not int or shards < 1 or mode not in ("full", "smoke_test"):
            raise ValueError("Check shard count and generation mode.")
        self.run = Path(drive_root).resolve() / "runs" / run_id
        self.generated = self.run / "images"
        self.external = bool(input_roots)
        self.cfg = ExperimentConfig(cohort_id, generation_csv, vocabulary_csv,
                                    tuple(input_roots) or (self.generated,), self.run / "stages",
                                    metadata_csv)
        if self.external and any(self.run.is_relative_to(p) or p.is_relative_to(self.run)
                                 for p in self.cfg.input_roots):
            raise ValueError("Use external input roots separate from the run directory.")
        words, candidates, _ = self.cfg.validate(planned_inputs=not self.external)
        if type(people) is not int or not 1 <= people <= len(words):
            raise ValueError("Team size must allow a nonempty assignment for each person.")
        self.words, self.people, self.mode, self.shards = words, people, mode, shards
        self.artifacts = self.cfg.output_root / cohort_id
        self.manifest_path = self.artifacts / "manifest.json"
        self.settings = dict(schema=1, config=self.cfg.to_dict(), bundle_sha256=source_hash,
                             people=people, mode=mode, shards=shards,
                             csv_hashes=[file_digest(p) for p in
                                         (generation_csv, vocabulary_csv, metadata_csv)])
        print(f"{len(words)} generation words, {len(candidates)} candidates. Run directory: {self.run}")

    def bind(self, stage):
        marker = self.run / "run.json"
        if self.run.exists():
            if not marker.is_file() or read(marker) != self.settings:
                raise ValueError(f"Different or unrecognized run at {self.run}. Check paths and settings.")
        elif stage in ("GENERATE", "MANIFEST"):
            self.run.mkdir(parents=True, exist_ok=False)
            create(marker, self.settings)
        elif stage not in ("ALL", "PREFLIGHT"):
            raise ValueError("Missing run. Check the mounted Drive, run ID and completed stages.")

    def assignment(self, person):
        if type(person) is not int or not 1 <= person <= self.people:
            raise ValueError("Person must be a 1-based team number.")
        words = self.words[person - 1::self.people]
        return words[:1] if self.mode == "smoke_test" else words

    def generate(self, person, resume):
        if self.external or self.manifest_path.exists():
            raise ValueError("Generation requires this run's own image directory and an unfrozen cohort.")
        words = self.assignment(person)
        part = self.generated / f"part{person}"
        receipt = self.run / "generation" / f"part{person}.json"
        if receipt.exists():
            raise FileExistsError(f"Part already completed: {receipt}")
        if part.exists() != resume:
            raise ValueError("Fresh generation needs a new part. Resume needs an existing part.")
        plan_path = self.run / "generation" / f"assignment{person}.json"
        plan = {"run_configuration_sha256": digest(self.settings), "assigned_words": words}
        if resume and not plan_path.is_file():
            raise ValueError(f"No generation plan for existing part: {part}")
        consistent(plan_path, plan)
        part.mkdir(parents=True, exist_ok=resume)
        cfg = ExperimentConfig(**{**self.cfg.to_dict(), "input_roots": [part]})
        if any(part.iterdir()):
            prior = build_manifest(cfg)
            issues = [i for i in prior["issues"] if i["code"] != "no_experiments"]
            if issues:
                raise ValueError(f"Incomplete or damaged part; inspect before resuming: {issues[:3]}")
            expected_folders = {f"{i:03d}_{w}" for i, w in enumerate(words, 1)}
            for row in prior["images"]:
                info = read(Path(row["metadata_path"]))
                if (row["folder"] not in expected_folders
                        or info["run_mode"] != self.mode or row["folder"].split("_", 1)[1] != row["target_word"]):
                    raise ValueError(f"Unexpected existing generation record: {row['metadata_path']}")
        nb = read(SRC / "clipdraw.ipynb")
        ns = {"__name__": "__main__"}
        # Original Sections 1-4 supply the kernel setup, exact defaults and rendering globals.
        for i in (2, 4, 6, 8):
            source = "".join(nb["cells"][i]["source"])
            exec(compile(source, f"clipdraw.ipynb:cell-{i}", "exec"), ns)
        script = SRC / "batch/batch_runner.py"
        exec(compile(script.read_text(encoding="utf-8-sig"), str(script), "exec"), ns)
        result = ns["batch_run"](words, prompt_template="A drawing of {}.", run_mode=self.mode,
                                 output_root=part, skip_existing=True, enable_prediction=False,
                                 zip_results=False)
        if failures := [r for r in result if r["status"] == "error"]:
            raise RuntimeError(f"Batch failed: {failures[:3]}")
        verified = checked_manifest(cfg)["images"]
        if sorted(r["target_word"] for r in verified) != sorted(words):
            raise ValueError("Verified output differs from the assigned generation words.")
        create(receipt, [row["image_id"] for row in verified])

    def manifest(self, *, publish=False):
        current = checked_manifest(self.cfg)
        if not self.external:
            ids = []
            for i in range(1, self.people + 1):
                receipt = self.run / "generation" / f"part{i}.json"
                if not receipt.is_file():
                    raise ValueError(f"Generation incomplete; missing {receipt}")
                ids.extend(read(receipt))
            if sorted(ids) != [r["image_id"] for r in current["images"]]:
                raise ValueError("Generated cohort differs from verified completion records.")
        if not publish and not self.manifest_path.is_file():
            raise ValueError("Run MANIFEST first.")
        consistent(self.manifest_path, current)
        return current

    def job(self, model, shard, manifest):
        ids = [r["image_id"] for r in manifest["images"]]
        if type(shard) is not int or not 1 <= shard <= self.shards <= len(ids):
            raise ValueError("Choose nonempty shards using 1-based numbering.")
        cfg = dict(model=model, revision=PINS[model], input_roots=list(map(str, self.cfg.input_roots)),
                   output_root=str(self.artifacts / "judges"), job_name=self.cfg.cohort_id,
                   words_source=str(self.cfg.vocabulary_csv), device="cuda",
                   batch_size=2 if model == "siglip2" else 1, shard_index=shard, num_shards=self.shards,
                   cache_dir="/content/hf_judge_cache", expected_image_ids=ids[shard-1::self.shards],
                   expected_vocabulary_sha256=manifest["csv_sha256"]["vocabulary_csv"],
                   manifest_sha256=digest(manifest), source_bundle_sha256=self.settings["bundle_sha256"])
        job_dir = Path(cfg["output_root"]) / model / cfg["job_name"] / f"shard{shard}of{self.shards}" / "full"
        return cfg, job_dir

    def judge(self, model, shard, resume):
        manifest = self.manifest()
        cfg, job_dir = self.job(model, shard, manifest)
        check_job(job_dir, resume, "job_manifest.json", "complete.json")
        launcher = uv_launcher()
        env_root = Path("/content/judge_envs")
        subprocess.run([sys.executable, "-m", "judges.colab_setup", "--model", model, "--env-root",
                        env_root, "--lock-root", self.artifacts / "locks"], cwd=SRC,
                       env=launcher, check=True)
        path = self.artifacts / "configs" / f"{model}-{shard}.json"
        consistent(path, cfg)
        subprocess.run([env_root / model / "bin/python", "-m", "judges.scaling", "--config", path,
                        "--mode", "run", *(["--resume"] if resume else [])], cwd=SRC, check=True)

    def text_score(self, revision, resume):
        if not re.fullmatch("[0-9a-f]{40}", revision):
            raise ValueError("Supply the verified full commit for google/siglip-so400m-patch14-384.")
        self.manifest()
        out = self.artifacts / "text_score"
        check_job(out, resume, "run.json")
        if resume and read(out / "run.json")["configuration"]["revision"] != revision:
            raise ValueError("Text-score revision differs from the interrupted run.")
        env = uv_launcher()
        python = text_environment(self.artifacts / "locks", env)
        env.update(HF_HOME=str(self.artifacts / "cache/hub"), EASYOCR_MODULE_PATH=str(self.artifacts / "cache/ocr"),
                   CUDA_VISIBLE_DEVICES="-1")
        subprocess.run([python, "-m", "text_score.run_manifest", "--manifest", self.manifest_path,
                        "--output-dir", out, "--revision", revision, *(["--resume"] if resume else [])],
                       cwd=SRC, env=env, check=True)

    def analyze(self, *, partial=False):
        from judges.analysis.combined import report_run

        dest, summary = report_run(self, partial=partial)
        overall = summary["overall"]
        print(f"{summary['mode']} analysis ({summary['status']}): {overall['images']} images, "
              f"{overall['unique_targets']} unique targets.")
        for source, coverage in summary["coverage"].items():
            print(f"{source}: {len(coverage['included_ids'])} evaluated, {len(coverage['missing_ids'])} missing")
        print("SigLIP2 top-1 / top-5 hits:", overall["siglip2"]["top1_hits"], "/",
              overall["siglip2"]["top5_hits"], "among", overall["siglip2"]["denominator"], "ranked images")
        print("Median text score:", overall["text_score"]["median"], "among", overall["text_score"]["n"], "scored images")
        for name in ("combined.csv", "combined_summary.json", "combined_report.md"):
            print(dest / name)
        return dest


def check_job(directory, resume, marker, complete="results.json"):
    if (directory / complete).exists():
        raise FileExistsError(f"Completed export exists at {directory}; choose ANALYZE or a new run.")
    if resume and not (directory / marker).is_file():
        raise ValueError(f"No resumable job at {directory}")
    if not resume and directory.exists():
        raise FileExistsError(f"Partial or unrecognized job at {directory}; inspect before resuming.")


def uv_launcher():
    """Install only the launcher into a separate runtime-local directory."""
    from judges.colab_setup import UV_VERSION
    target = Path("/content/qmind-uv") / UV_VERSION
    if not (target / "bin/uv").is_file():
        subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "--target", target,
                        f"uv=={UV_VERSION}"], check=True)
    return {**os.environ, "PATH": str(target / "bin") + os.pathsep + os.environ["PATH"]}


def text_environment(locks, env):
    from judges.colab_setup import PYTHON_VERSION
    root = Path("/content/qmind-text-env")
    python = root / "bin/python"
    if not python.is_file():
        subprocess.run(["uv", "venv", "--python", PYTHON_VERSION, root], env=env, check=True)
    lock = locks / "text-score-linux.lock"
    index = ["--index", "https://download.pytorch.org/whl/cpu", "--index-strategy", "unsafe-best-match"]
    if not lock.exists():
        locks.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory() as temp:
            compiled = Path(temp) / "requirements.txt"
            subprocess.run(["uv", "pip", "compile", "--python", python, "--generate-hashes", *index,
                            SRC / "text_score/requirements-colab.in", "-o", compiled], env=env, check=True)
            with lock.open("xb") as f:
                f.write(compiled.read_bytes())
    subprocess.run(["uv", "pip", "sync", "--python", python, "--require-hashes", *index, lock],
                   env=env, check=True)
    return python
