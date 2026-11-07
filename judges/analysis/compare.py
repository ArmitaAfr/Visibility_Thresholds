"""Validate, join and summarize saved judge outputs, without model imports."""

import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from statistics import mean

from ..checkpoints import digest, file_digest
from ..discovery import discover_experiments
from .patterns import CUES, HEURISTIC_VERSION, caption_patterns


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _load_exports(paths, family):
    if isinstance(paths, (str, Path)):
        paths = [paths]
    records, provenance, configurations = {}, [], set()
    for path in paths:
        path = Path(path).resolve()
        content = path.read_bytes()
        try:
            rows = json.loads(content)
        except (ValueError, UnicodeError) as exc:
            raise ValueError(f"Invalid JSON export {path}: {exc}") from exc
        if not isinstance(rows, list):
            raise ValueError(f"Export {path} must contain a JSON list.")
        provenance.append({"path": str(path), "sha256": hashlib.sha256(content).hexdigest(), "records": len(rows)})
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError(f"Non-object result in {path}.")
            for key in ("image_id", "target_word", "image_path", "metadata_path", "model_name", "normalized_prediction"):
                if not isinstance(row.get(key), str) or not row[key].strip():
                    raise ValueError(f"Invalid {key} in {path}.")
            image_id = row["image_id"]
            if image_id in records:
                raise ValueError(f"Duplicate {family} image_id {image_id}; supply each evaluation once.")
            if family not in row["model_name"].lower():
                raise ValueError(f"Wrong model family in {path}: {row['model_name']}")
            if not _finite(row.get("inference_time_seconds")) or row["inference_time_seconds"] < 0:
                raise ValueError(f"Invalid inference time for {image_id}.")
            raw = row.get("raw_output")
            config = raw.get("configuration") if isinstance(raw, dict) else None
            if not isinstance(config, dict):
                raise ValueError(f"Missing inference configuration for {image_id}.")
            # Final partial batches legitimately differ in actual image count.
            profile = {k: v for k, v in config.items() if k != "image_batch_size"}
            configurations.add(digest({"model_name": row["model_name"], "configuration": profile}))
            records[image_id] = row
    if not records:
        raise ValueError(f"No {family} results supplied.")
    if len(configurations) != 1:
        raise ValueError(f"Mixed {family} model/configuration profiles; analyze conditions separately.")
    return records, provenance, next(iter(configurations))


def _siglip_prediction(row):
    raw = row["raw_output"]
    config = raw["configuration"]
    ranking = raw.get("candidate_scores")
    size = config.get("candidate_vocabulary_size")
    vocabulary_hash = config.get("candidate_vocabulary_sha256")
    if type(size) is not int or size < 1 or not isinstance(vocabulary_hash, str) or len(vocabulary_hash) != 64:
        raise ValueError("Invalid SigLIP2 vocabulary size/hash.")
    if not isinstance(ranking, list) or len(ranking) != size:
        raise ValueError("SigLIP2 needs its complete candidate_scores ranking.")
    words, scores = [], []
    for prediction in ranking:
        if (not isinstance(prediction, dict) or not isinstance(prediction.get("word"), str)
                or not prediction["word"].strip() or not _finite(prediction.get("score"))):
            raise ValueError("Invalid SigLIP2 candidate prediction.")
        words.append(prediction["word"].strip().lower())
        scores.append(prediction["score"])
    if len(set(words)) != size or scores != sorted(scores, reverse=True):
        raise ValueError("SigLIP2 ranking is duplicate or not sorted by descending score.")
    if raw.get("top5") != ranking[:min(5, size)] or row["normalized_prediction"].strip().lower() != words[0]:
        raise ValueError("SigLIP2 top1/top5 disagrees with its complete ranking.")
    target = row["target_word"].strip().lower()
    if target not in words:
        raise ValueError(f"Target {target!r} absent from SigLIP2 vocabulary.")
    rank = words.index(target) + 1
    if (type(row.get("target_rank")) is not int or row["target_rank"] != rank
            or not _finite(row.get("target_score"))
            or not math.isclose(row["target_score"], scores[rank - 1], rel_tol=1e-6, abs_tol=1e-6)):
        raise ValueError("SigLIP2 target rank/score disagrees with its ranking.")
    return words, size, vocabulary_hash


def analyze_exports(siglip_paths, florence_paths, *, source_roots=None, allow_unmatched=False):
    """Return combined rows and a deterministic summary; never write source files.

    Source verification uses Phase 1 discovery to recompute the image ID from
    current metadata and PNG bytes. source_roots supports relocated artifacts.
    Unmatched records cause an error unless the caller explicitly selects the
    intersection; denominators and omitted IDs are then recorded.
    """
    siglip, sig_provenance, sig_config = _load_exports(siglip_paths, "siglip2")
    florence, flo_provenance, flo_config = _load_exports(florence_paths, "florence-2")
    only_siglip, only_florence = sorted(siglip.keys() - florence.keys()), sorted(florence.keys() - siglip.keys())
    if (only_siglip or only_florence) and not allow_unmatched:
        raise ValueError(f"Unmatched IDs: {len(only_siglip)} SigLIP2-only, {len(only_florence)} Florence-only. "
                         "Supply matching exports or explicitly use --allow-unmatched.")
    matched = sorted(siglip.keys() & florence.keys())
    if not matched:
        raise ValueError("No matching image IDs.")
    relocated = None
    if source_roots:
        relocated = {e.image_id: e for e in discover_experiments(source_roots).experiments}
    verified, rows = {}, []
    vocabularies = set()
    pattern_counts = Counter()
    cue_counts = {category: Counter() for category in CUES}

    def verify(row):
        image_id = row["image_id"]
        if relocated is not None:
            experiment = relocated.get(image_id)
            if experiment is None:
                raise ValueError(f"Image ID {image_id} absent from verified --source-root artifacts.")
        else:
            key = (row["metadata_path"], row["image_path"])
            if key not in verified:
                report = discover_experiments(Path(row["metadata_path"]).parent)
                experiment = next((e for e in report.experiments if e.image_id == image_id), None)
                if experiment is None:
                    raise ValueError(f"Source content/metadata no longer matches image ID {image_id}.")
                if file_digest(row["image_path"]) != file_digest(experiment.image_path):
                    raise ValueError(f"Export image path disagrees with metadata sibling for {image_id}.")
                verified[key] = experiment
            experiment = verified[key]
        if experiment.image_id != image_id or experiment.target_word.strip().lower() != row["target_word"].strip().lower():
            raise ValueError(f"Source target/identity mismatch for {image_id}.")
        return experiment

    for image_id in matched:
        sig, flo = siglip[image_id], florence[image_id]
        source = verify(sig)
        verify(flo)
        if sig["target_word"].strip().lower() != flo["target_word"].strip().lower():
            raise ValueError(f"Target disagreement for {image_id}.")
        words, size, vocabulary_hash = _siglip_prediction(sig)
        vocabularies.add((size, vocabulary_hash, tuple(sorted(words))))
        raw = flo["raw_output"]
        caption = raw.get("caption")
        if not isinstance(caption, str) or not caption.strip():
            raise ValueError(f"Missing Florence original caption for {image_id}.")
        if flo.get("target_rank") is not None or flo.get("target_score") is not None:
            raise ValueError("Florence caption results must not contain candidate ranks/scores.")
        patterns = caption_patterns(caption)
        for category, cues in patterns.items():
            pattern_counts[category] += bool(cues)
            cue_counts[category].update(cues)
        rows.append({
            "image_id": image_id, "target_word": sig["target_word"], "image_path": str(source.image_path),
            "image_sha256": file_digest(source.image_path),
            "siglip2_top1_prediction": words[0],
            "siglip2_top5_predictions": [p["word"] for p in sig["raw_output"]["top5"]],
            "siglip2_target_rank": sig["target_rank"], "siglip2_target_score": sig["target_score"],
            "florence2_original_caption": caption, "florence2_normalized_caption": flo["normalized_prediction"],
            "florence2_mentions_apparent_text": bool(patterns["apparent_text"]),
            "florence2_text_mention_cues": patterns["apparent_text"],
            "florence2_object_cues": patterns["objects"], "florence2_abstract_concept_cues": patterns["abstract_concepts"],
            "siglip2_inference_time_seconds": sig["inference_time_seconds"],
            "florence2_inference_time_seconds": flo["inference_time_seconds"],
        })
    if len(vocabularies) != 1:
        raise ValueError("SigLIP2 candidate vocabulary differs across compared images.")
    size, vocabulary_hash, _ = next(iter(vocabularies))
    count = len(rows)
    top1 = sum(row["siglip2_target_rank"] == 1 for row in rows)
    top5 = sum(row["siglip2_target_rank"] <= min(5, size) for row in rows)
    summary = {
        "analysis_version": 1, "matched_images": count,
        "input_records": {"siglip2": len(siglip), "florence2": len(florence)},
        "unmatched_ids": {"siglip2_only": only_siglip, "florence2_only": only_florence},
        "allow_unmatched": allow_unmatched, "source_verification": "recomputed Phase 1 IDs from current metadata and PNG bytes",
        "inputs": {"siglip2": sig_provenance, "florence2": flo_provenance},
        "models": {"siglip2": next(iter(siglip.values()))["model_name"], "florence2": next(iter(florence.values()))["model_name"]},
        "configuration_hashes": {"siglip2": sig_config, "florence2": flo_config},
        "siglip2": {"denominator": count, "vocabulary_size": size, "vocabulary_sha256": vocabulary_hash,
                    "top1_target_hits": top1, "top1_target_hit_rate": top1 / count,
                    "top5_target_hits": top5, "top5_target_hit_rate": top5 / count,
                    "mean_target_rank": mean(row["siglip2_target_rank"] for row in rows)},
        "florence2_patterns": {category: {"captions_with_cues": pattern_counts[category],
            "fraction": pattern_counts[category] / count, "cue_caption_counts": dict(sorted(cue_counts[category].items()))}
            for category in CUES},
        "heuristic": {"version": HEURISTIC_VERSION, "lexicons": CUES,
            "interpretation": "Descriptive whole-token caption mentions, including negation; not verified OCR, object detection or semantic classification."},
        "interpretation": "Judge model outputs and target-hit rates are not actual human recognition or calibrated probabilities.",
    }
    return rows, summary


def write_report(rows, summary, output_dir):
    """Create a fresh report directory. Existing reports/exports are never overwritten."""
    output_dir = Path(output_dir)
    if not rows:
        raise ValueError("Cannot export an empty comparison.")
    output_dir.mkdir(parents=True, exist_ok=False)
    with (output_dir / "comparison.csv").open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, list) else v for k, v in row.items()})
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    sig = summary["siglip2"]
    text = ["# Judge comparison report", "", summary["interpretation"], "",
            f"Matched images: {summary['matched_images']}; metric denominator: {sig['denominator']}.",
            f"SigLIP2 vocabulary: {sig['vocabulary_size']} candidates (SHA-256: {sig['vocabulary_sha256']}).", "",
            f"- Top-1 target hits: {sig['top1_target_hits']}/{sig['denominator']} ({sig['top1_target_hit_rate']:.2%}).",
            f"- Top-5 target hits: {sig['top5_target_hits']}/{sig['denominator']} ({sig['top5_target_hit_rate']:.2%}).",
            f"- Mean target rank: {sig['mean_target_rank']:.3f} (1-based).", "",
            "## Florence caption wording", "", summary["heuristic"]["interpretation"], ""]
    for category, pattern in summary["florence2_patterns"].items():
        text.append(f"- {category}: {pattern['captions_with_cues']}/{sig['denominator']} captions; cues: {pattern['cue_caption_counts']}.")
    text.extend(["", "Categories overlap; lexicons are illustrative, not exhaustive. Absence of a cue does not establish absence of an object or concept.", "",
                 "Unmatched image IDs (excluded only when explicitly requested): " + json.dumps(summary["unmatched_ids"]), "",
                 "See summary.json for input hashes, model IDs, configuration hashes and complete heuristic definitions.", ""])
    (output_dir / "report.md").write_text("\n".join(text), encoding="utf-8")
    return output_dir
