"""Score a verified Phase 7 manifest in a dedicated CPU process."""

import argparse
import hashlib
import io
import json
import platform
import re
from importlib import metadata
from pathlib import Path

from judges.checkpoints import atomic_write, digest, file_digest
from judges.discovery import canonical_image_id
from judges.experiment_manifest import ExperimentConfig

IDENTITY = ("image_id", "image_sha256", "target_word", "image_path", "metadata_path",
            "cohort_id", "part", "folder")


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _manifest(path):
    data = _read(path)
    if (not isinstance(data, dict) or data.get("schema_version") != 1
            or data.get("issues") != [] or not isinstance(data.get("images"), list)
            or not data["images"]):
        raise ValueError("Expected a nonempty version 1 manifest with no discovery issues.")
    config = ExperimentConfig(**data["config"])
    _, candidates, _ = config.validate()
    hashes = {name: file_digest(path) for name in ("generation_csv", "vocabulary_csv", "imageability_csv")
              if (path := getattr(config, name)) is not None}
    if hashes != data.get("csv_sha256"):
        raise ValueError("Manifest CSV fingerprints do not match current inputs.")
    words = {word.lower() for word in candidates}
    seen = set()
    targets = {}
    for row in data["images"]:
        if not isinstance(row, dict) or any(not isinstance(row.get(k), str) or not row[k].strip() for k in IDENTITY):
            raise ValueError("Missing required image identity fields.")
        image = Path(row["image_path"])
        word = row["target_word"].lower()
        if (image != image.resolve() or image.name != "clipdraw_final.png"
                or not any(image.is_relative_to(root) for root in config.input_roots)
                or Path(row["metadata_path"]) != image.with_name("experiment.json")
                or row["part"] != image.parent.parent.name or row["folder"] != image.parent.name
                or row["cohort_id"] != config.cohort_id or word not in words):
            raise ValueError(f"Inconsistent manifest entry for {image}")
        if not all(re.fullmatch("[0-9a-f]{64}", row[k]) for k in ("image_id", "image_sha256")):
            raise ValueError(f"Invalid hash for {image}")
        if row["image_id"] in seen or targets.setdefault(row["image_sha256"], word) != word:
            raise ValueError(f"Duplicate identity or conflicting target: {image}")
        seen.add(row["image_id"])
    return data, config


def _verified_bytes(row):
    content = Path(row["image_path"]).read_bytes()
    info = _read(Path(row["metadata_path"]))
    canonical = json.dumps(info, sort_keys=True, ensure_ascii=True, allow_nan=False)
    if (hashlib.sha256(content).hexdigest() != row["image_sha256"]
            or canonical_image_id(canonical, content) != row["image_id"]
            or not isinstance(info, dict) or not isinstance(info.get("word"), str)
            or info["word"].strip() != row["target_word"]):
        raise ValueError("Source PNG, metadata or target differs from manifest.")
    return io.BytesIO(content)


def _load_scorer(revision, upscale):
    from .text_score import Scorer, SIGLIP

    scorer = Scorer(upscale, revision=revision)
    weights = {p.name: file_digest(p) for p in Path(scorer.ocr.model_storage_directory).glob("*.pth")}
    if not weights:
        raise ValueError("Could not fingerprint EasyOCR weight files.")
    provenance = {"model": SIGLIP, "revision": revision, "upscale": upscale,
                  "easyocr_language": "en", "device": "cpu", "ocr_weights_sha256": weights,
                  "scorer_sha256": file_digest(Path(__file__).with_name("text_score.py")),
                  "python": platform.python_version(),
                  "packages": {d.metadata["Name"]: d.version for d in metadata.distributions()
                               if d.metadata["Name"]}}
    return scorer, provenance


def run(manifest, output_dir, *, revision, upscale=3, resume=False):
    """Return the complete export path. On partial failure raise after recording all errors."""
    from filelock import FileLock

    if not isinstance(revision, str) or not re.fullmatch("[0-9a-f]{40}", revision):
        raise ValueError("Supply the verified 40-character SigLIP Hub commit SHA.")
    if type(upscale) is not int or upscale < 1:
        raise ValueError("upscale must be a positive integer.")
    data, config = _manifest(Path(manifest))
    out = Path(output_dir).resolve()
    if (out == config.output_root or not out.is_relative_to(config.output_root)
            or any(out.is_relative_to(root) for root in config.input_roots)):
        raise ValueError("Use a dedicated run directory below output_root and outside image inputs.")
    settings = out / "run.json"
    final = out / "results.json"
    if resume:
        if not settings.is_file() or final.exists():
            raise ValueError("Resume requires an initialized, unfinished text-score run.")
    elif out.exists():
        raise FileExistsError(out)
    scorer, provenance = _load_scorer(revision, upscale)
    if not resume:
        out.mkdir(parents=True, exist_ok=False)
    with FileLock(str(out / ".writer.lock"), timeout=0):
        if final.exists():
            raise FileExistsError(final)
        job = {"stage": "text_score", "version": 1, "manifest_sha256": digest(data),
               "configuration": provenance}
        checkpoints = out / "checkpoints"
        if resume:
            if _read(settings) != job:
                raise ValueError("Manifest or scoring configuration changed. Use a new run directory.")
        else:
            checkpoints.mkdir()
            atomic_write(settings, job)

        rows, errors = [], []
        condition = digest(job)
        for row in data["images"]:
            identity = {k: row[k] for k in IDENTITY}
            checkpoint = checkpoints / (row["image_id"] + ".json")
            try:
                png = _verified_bytes(row)
                if checkpoint.exists():
                    saved = _read(checkpoint)
                    result = saved["result"]
                    if (saved["checksum"] != digest([condition, result])
                            or any(result[k] != value for k, value in identity.items())):
                        raise ValueError("Damaged or incompatible text-score checkpoint.")
                else:
                    scores = scorer(png, row["target_word"])
                    if scores["word"] != row["target_word"]:
                        raise ValueError("Scorer returned a different target.")
                    result = {**scores, **identity}
                    atomic_write(checkpoint, {"result": result, "checksum": digest([condition, result])})
            except Exception as exc:
                errors.append({**identity, "error": f"{type(exc).__name__}: {exc}"})
                continue
            rows.append(result)
        if errors:
            atomic_write(out / "attempt_errors.json", {"status": "partial", "completed": len(rows),
                                                      "errors": errors})
            raise ValueError(f"{len(errors)} image(s) failed. See {out / 'attempt_errors.json'}")
        atomic_write(final, {**job, "status": "complete", "results": rows})
        return final


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--upscale", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    print(run(**vars(parser.parse_args())))


if __name__ == "__main__":
    main()
