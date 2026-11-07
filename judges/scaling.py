"""Drive-safe job orchestration and measured benchmarks, reusing judge APIs."""

import argparse
import importlib.metadata
import json
import platform
import re
import time
from dataclasses import asdict
from pathlib import Path

from . import discover_experiments, evaluate, export_results, load_candidates
from .checkpoints import digest, file_digest
from .revisions import is_commit_sha


def source_fingerprint():
    root = Path(__file__).resolve().parent
    return {str(p.relative_to(root)): file_digest(p) for p in sorted(root.rglob("*"))
            if p.is_file() and p.suffix in {".py", ".txt"} and "tests" not in p.parts}


def environment_profile():
    versions = {}
    for name in ("torch", "torchvision", "transformers", "huggingface_hub", "tokenizers",
                 "timm", "einops", "pillow", "numpy", "sentencepiece", "protobuf"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {"python": platform.python_version(), "packages": versions}


def select_experiments(experiments, shard_index=1, num_shards=1):
    if type(num_shards) is not int or num_shards < 1 or type(shard_index) is not int or not 1 <= shard_index <= num_shards:
        raise ValueError("Use 1 <= shard_index <= num_shards with integer values.")
    return sorted(experiments, key=lambda e: e.image_id)[shard_index - 1::num_shards]


def benchmark_summary(results, elapsed_seconds, model_load_seconds, warmup_seconds, estimate_sizes):
    inference_seconds = sum(row.inference_time_seconds for row in results)
    if not results or inference_seconds <= 0 or elapsed_seconds <= 0:
        raise ValueError("Benchmark needs successful measured inference and positive elapsed time.")
    inference_rate = len(results) / inference_seconds
    observed_rate = len(results) / elapsed_seconds
    if any(type(n) is not int or n < 1 for n in estimate_sizes):
        raise ValueError("Estimate sizes must be positive integers.")
    return {
        "sample_images": len(results), "inference_seconds": inference_seconds,
        "evaluation_seconds_including_checkpoint_io": elapsed_seconds,
        "model_load_and_text_cache_seconds": model_load_seconds, "warmup_seconds": warmup_seconds,
        "inference_images_per_second": inference_rate,
        "evaluation_images_per_second": observed_rate,
        "estimates": {str(n): {"inference_seconds": n / inference_rate,
                                "evaluation_seconds": n / observed_rate,
                                "evaluation_plus_model_load_seconds": n / observed_rate + model_load_seconds}
                      for n in estimate_sizes},
        "limitations": "Measured sample only. Estimates exclude dependency installation, downloads outside model loading, export and interruptions; caption lengths, Drive IO and GPU allocation can change throughput. No paid-Colab speed guarantee.",
    }


def build_judge(config, candidates):
    common = dict(revision=config["revision"], device=config.get("device", "cuda"),
                  cache_dir=config.get("cache_dir"))
    if config["model"] == "siglip2":
        from .siglip2 import SigLIP2Judge
        return SigLIP2Judge(candidates, prompt_template=config.get("prompt_template", "A drawing of a {}."),
                            text_batch_size=config.get("text_batch_size", 32), **common)
    if config["model"] == "florence2":
        from .florence2 import Florence2Judge
        return Florence2Judge(dtype=config.get("dtype", "auto"),
                              max_new_tokens=config.get("max_new_tokens", 128),
                              num_beams=config.get("num_beams", 3), **common)
    raise ValueError("model must be siglip2 or florence2.")


def cuda_preflight():
    """Check actual CUDA kernels before downloading/loading large model weights."""
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable in judge environment. Select a Colab GPU; verify the cu124 driver/wheel pairing.")
    try:
        with torch.inference_mode():
            value = torch.ones(4, 4, device="cuda")
            (value @ value).sum().item()
            torch.cuda.synchronize()
        print("GPU:", torch.cuda.get_device_name(), "torch:", torch.__version__, "CUDA wheel:", torch.version.cuda)
    except Exception as exc:
        raise RuntimeError("Pinned CUDA wheel cannot execute on this GPU/driver. Use a compatible T4/L4/A100-class allocation; do not replace kernel packages.") from exc


def _new_json(path, value):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def run_job(config, mode="run", *, resume=False):
    if mode not in {"run", "benchmark"}:
        raise ValueError("mode must be run or benchmark.")
    if mode == "benchmark" and resume:
        raise ValueError("Benchmarks cannot resume: cached rows would invalidate measured throughput. Use a fresh benchmark_tag.")
    if config.get("model") not in {"siglip2", "florence2"} or not is_commit_sha(config.get("revision")):
        raise ValueError("Select a judge and pin revision to its full immutable commit SHA.")
    for field in (("job_name", "benchmark_tag") if mode == "benchmark" else ("job_name",)):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", config.get(field, "")):
            raise ValueError(f"{field} must be a simple nonempty name (letters, numbers, _ or -).")
    batch_size = config.get("batch_size", 1)
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer.")
    roots = config["input_roots"]
    experiments = select_experiments(discover_experiments(roots).experiments,
                                     config.get("shard_index", 1), config.get("num_shards", 1))
    if not experiments:
        raise ValueError("No images assigned to this shard.")
    if mode == "benchmark":
        size = config.get("benchmark_images", 8)
        if type(size) is not int or size < 1:
            raise ValueError("benchmark_images must be positive.")
        experiments = experiments[:size]
    words_source = config["words_source"]
    candidates = load_candidates(words_source)
    if any(e.target_word.lower() not in {w.lower() for w in candidates} for e in experiments):
        raise ValueError("A selected target is absent from the configured vocabulary.")
    shard = f"shard{config.get('shard_index', 1)}of{config.get('num_shards', 1)}"
    job_dir = Path(config["output_root"]) / config["model"] / config["job_name"] / shard / (config["benchmark_tag"] if mode == "benchmark" else "full")
    recorded_config = config if mode == "benchmark" else {k: v for k, v in config.items()
        if k not in {"benchmark_tag", "benchmark_images", "warmup_images", "estimate_sizes"}}
    manifest = {"version": 1, "mode": mode, "configuration": recorded_config,
                "selected_image_ids": [e.image_id for e in experiments],
                "vocabulary_file_sha256": file_digest(words_source), "vocabulary_sha256": digest(candidates),
                "judge_source_sha256": source_fingerprint(), "environment": environment_profile()}
    # Canonicalize tuples/Path-free config to the same representation read from disk.
    manifest = json.loads(json.dumps(manifest, allow_nan=False))
    if resume:
        saved = json.loads((job_dir / "job_manifest.json").read_text(encoding="utf-8"))
        if saved != manifest:
            raise ValueError("Job/source/environment/cohort changed; use the original configuration/environment or a new job_name.")
    else:
        job_dir.mkdir(parents=True, exist_ok=False)
        _new_json(job_dir / "job_manifest.json", manifest)
    if config.get("device", "cuda") == "cuda":
        cuda_preflight()
    started = time.perf_counter()
    judge = build_judge(config, candidates)
    model_load_seconds = time.perf_counter() - started
    warmup_seconds = 0.0
    if mode == "benchmark":
        warmup_count = config.get("warmup_images", 1)
        if type(warmup_count) is not int or warmup_count < 0:
            raise ValueError("warmup_images must be nonnegative.")
        start = time.perf_counter()
        warmup = experiments[:warmup_count]
        for offset in range(0, len(warmup), batch_size):
            paths = [e.image_path for e in warmup[offset:offset + batch_size]]
            if batch_size == 1:
                judge.predict(paths[0], candidates)
            else:
                judge.predict_batch(paths, candidates)
        warmup_seconds = time.perf_counter() - start
    checkpoint = job_dir / "checkpoint"
    start = time.perf_counter()
    results = evaluate(experiments, candidates, judge, batch_size=batch_size,
                       checkpoint_dir=checkpoint, resume=resume and checkpoint.exists())
    elapsed = time.perf_counter() - start
    export = job_dir / "results.json"
    if export.exists():
        if json.loads(export.read_text(encoding="utf-8")) != [asdict(row) for row in results]:
            raise ValueError("Existing export disagrees with verified checkpoint results; choose a new export/job.")
    else:
        export_results(results, export)
    if mode == "benchmark":
        report = benchmark_summary(results, elapsed, model_load_seconds, warmup_seconds,
                                   config.get("estimate_sizes", [150, 2000, 2500]))
        report["job_manifest"] = str(job_dir / "job_manifest.json")
        _new_json(job_dir / "benchmark.json", report)
        print(json.dumps(report, indent=2))
    print(f"{len(results)} results: {export}")
    return job_dir


def main(argv=None):
    parser = argparse.ArgumentParser(description="Checkpointed judge jobs and measured scaling benchmarks.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mode", choices=["benchmark", "run"], required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    run_job(json.loads(args.config.read_text(encoding="utf-8")), args.mode, resume=args.resume)


if __name__ == "__main__":
    main()
