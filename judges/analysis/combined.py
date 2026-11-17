"""Descriptive reports for a frozen unified run. Analysis reads saved artifacts only."""

import csv
import hashlib
import json
import math
import re
import tempfile
from collections import Counter
from pathlib import Path
from statistics import mean, median

from ..checkpoints import digest, file_digest
from ..vocabulary import load_candidates
from .compare import _load_exports, analyze_exports, caption_fields, recognition_fields, write_report
from .imageability import GROUP_BINS, canonical_word

TEXT_FIELDS = ("ocr_strings", "ocr_max_conf", "ocr_n_boxes", "text_area_frac", "target_sim",
               "target_partial", "zs_p_text", "zs_typo_margin", "text_level_pred", "text_score")
SIG_FIELDS = ("siglip2_top1_prediction", "siglip2_top5_predictions", "siglip2_target_rank",
              "siglip2_target_score", "siglip2_inference_time_seconds")
FLO_FIELDS = ("florence2_original_caption", "florence2_normalized_caption", "florence2_mentions_apparent_text",
              "florence2_text_mention_cues", "florence2_object_cues", "florence2_abstract_concept_cues",
              "florence2_inference_time_seconds")
SOURCES = ("siglip2", "florence2", "text_score")
MODELS = {"siglip2": "google/siglip2-base-patch16-224", "florence2": "microsoft/Florence-2-large-ft",
          "text_score": "google/siglip-so400m-patch14-384"}
BANDS = ("[0, 0.25)", "[0.25, 0.5)", "[0.5, 0.75)", "[0.75, 1]")
LIMITATIONS = ("SigLIP2 hit rates describe retrieval within the selected candidate vocabulary. "
               "Florence text mentions are lexical cues in generated captions, not verified OCR. "
               "The text/no-text probe is a relative two-prompt softmax, not a calibrated probability. "
               "OCR and heuristic text levels can be wrong. The four equal-width text-score intervals "
               "are descriptive reporting bins, independent of the scorer's provisional levels. "
               "Differences between bins do not establish that lettering caused recognition.")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def verify_identity(row, images, *, text=False):
    image = images.get(row.get("image_id"))
    if image is None:
        raise ValueError(f"Unknown image ID: {row.get('image_id')}")
    required = ["target_word", "image_path", "metadata_path"]
    for key in ("image_sha256", "cohort_id", "part", "folder"):
        if text or key in row:
            required.append(key)
    for key in required:
        if row.get(key) != image[key]:
            raise ValueError(f"{key} mismatch for {image['image_id']}.")
    if text and row.get("word") != image["target_word"]:
        raise ValueError(f"Text scorer word mismatch for {image['image_id']}.")


def judge_source(flow, manifest, images, model, vocabulary):
    """Only completed, receipt-verified shards contribute observations."""
    rows, shards, profiles, paths = {}, [], set(), []
    for index in range(1, flow.shards + 1):
        cfg, directory = flow.job(model, index, manifest)
        export = directory / "results.json"
        job_file = directory / "job_manifest.json"
        receipt = directory / "complete.json"
        expected = cfg["expected_image_ids"]
        item = {"export": str(export), "shard": index, "expected_ids": expected}
        shards.append(item)
        if job_file.exists():
            job = read(job_file)
            if (job.get("mode") != "run" or job.get("configuration") != cfg
                    or job.get("selected_image_ids") != expected
                    or job.get("vocabulary_file_sha256") != manifest["csv_sha256"]["vocabulary_csv"]):
                raise ValueError(f"Export or job configuration differs from the planned shard: {directory}")
            item.update(job_manifest=str(job_file), job_manifest_sha256=digest(job))
        if not receipt.exists():
            item["status"] = "unfinished" if directory.exists() else "absent"
            continue
        if not job_file.is_file() or not export.is_file():
            raise ValueError(f"Completion receipt has missing inputs: {directory}")
        family = "florence-2" if model == "florence2" else model
        completion = read(receipt)
        if completion != {"job_manifest_sha256": digest(job), "export_sha256": file_digest(export)}:
            raise ValueError(f"Export or job configuration differs from completion receipt: {directory}")
        records, provenance, profile = _load_exports(export, family)
        if sorted(records) != expected or rows.keys() & records.keys():
            raise ValueError(f"Duplicate, unexpected or missing IDs in completed shard: {directory}")
        if provenance[0]["sha256"] != completion["export_sha256"]:
            raise ValueError(f"Export changed during analysis: {export}")
        profiles.add(profile)
        paths.append(export)
        for key, row in records.items():
            verify_identity(row, images)
            config = row["raw_output"]["configuration"]
            if (row["model_name"] != f"{MODELS[model]}@{cfg['revision']}"
                    or config.get("resolved_revision") != cfg["revision"]):
                raise ValueError(f"Model revision differs from the planned {model} job: {key}")
            if model == "siglip2":
                fields, signature = recognition_fields(row)
                if signature != vocabulary:
                    raise ValueError(f"Wrong recognition vocabulary for {key}.")
            else:
                fields = caption_fields(row)
            rows[key] = {**fields, f"{model}_inference_time_seconds": row["inference_time_seconds"]}
        item.update(status="complete", export_sha256=provenance[0]["sha256"], receipt=str(receipt),
                    configuration={k: v for k, v in config.items() if k != "image_batch_size"})
    if len(profiles) > 1:
        raise ValueError(f"Mixed {model} configurations across shards.")
    return rows, {"shards": shards, "model": MODELS[model], "revision": cfg["revision"],
                  "configuration_hash": next(iter(profiles), None)}, paths


def text_fields(row):
    for key in TEXT_FIELDS:
        value = row.get(key)
        if key == "ocr_strings":
            valid = isinstance(value, str)
        elif key in ("ocr_n_boxes", "text_level_pred"):
            valid = type(value) is int and value >= 0 and (key != "text_level_pred" or value <= 4)
        else:
            valid = (type(value) in (float, int) and math.isfinite(value)
                     and (key == "zs_typo_margin" or 0 <= value <= 1))
        if not valid:
            raise ValueError(f"Invalid {key} for {row['image_id']}: {value!r}")
    score = row["text_score"]
    return {**{key: row[key] for key in TEXT_FIELDS}, "text_score_interval": BANDS[min(3, int(score * 4))]}


def text_source(directory, manifest, images):
    """Reconcile a complete export or read only checksummed successes from an unfinished run."""
    run_file, export = directory / "run.json", directory / "results.json"
    info = {"export": str(export), "status": "absent"}
    if not run_file.is_file():
        if export.exists() or (directory / "checkpoints").exists() or (directory / "attempt_errors.json").exists():
            raise ValueError(f"Missing text-score job identity: {run_file}")
        return {}, info
    job = read(run_file)
    config = job.get("configuration", {})
    if (job.get("stage") != "text_score" or job.get("version") != 1
            or job.get("manifest_sha256") != digest(manifest) or config.get("model") != MODELS["text_score"]
            or not re.fullmatch("[a-f0-9]{40}", config.get("revision", ""))):
        raise ValueError(f"Text-score run has incompatible model, manifest or job identity: {run_file}")
    condition = digest(job)
    info.update(status="unfinished", job=job, run_file=str(run_file), configuration_hash=digest(config))
    final_rows = {}
    if export.is_file():
        content = export.read_bytes()
        final = json.loads(content)
        if (final.get("status") != "complete" or any(final.get(k) != v for k, v in job.items())
                or not isinstance(final.get("results"), list)):
            raise ValueError(f"Text export differs from its run metadata: {export}")
        for row in final["results"]:
            verify_identity(row, images, text=True)
            if row["image_id"] in final_rows:
                raise ValueError(f"Duplicate text-score ID: {row['image_id']}")
            final_rows[row["image_id"]] = row
        if final_rows.keys() != images.keys():
            raise ValueError(f"Missing IDs in completed text export: {sorted(images.keys()-final_rows.keys())}")
        info.update(status="complete", export_sha256=hashlib.sha256(content).hexdigest())
    rows = {}
    folder = directory / "checkpoints"
    for path in sorted(folder.glob("*.json")):
        saved = read(path)
        row = saved["result"]
        verify_identity(row, images, text=True)
        key = row["image_id"]
        if (path.stem != key or key in rows or saved["checksum"] != digest([condition, row])
                or info["status"] == "complete" and final_rows.get(key) != row):
            raise ValueError(f"Conflicting or damaged text checkpoint: {path}")
        rows[key] = text_fields(row)
    if final_rows.keys() - rows.keys():
        raise ValueError(f"Completed text export missing verifying checkpoints under {folder}")
    info["checkpoint_dir"] = str(folder)
    errors = directory / "attempt_errors.json"
    if errors.exists():
        history = read(errors)
        for error in history["errors"]:
            verify_identity(error, images)
        info.update(last_failure_file=str(errors), last_recorded_errors=history["errors"])
    return rows, info


def distribution(values):
    return {"n": len(values), **{key: fn(values) if values else None for key, fn in
                                (("min", min), ("median", median), ("mean", mean), ("max", max))}}


def measures(rows):
    ranks = [r["siglip2_target_rank"] for r in rows if r["has_siglip2"]]
    scores = [r["text_score"] for r in rows if r["has_text_score"]]
    recognition = {"denominator": len(ranks), "rank_distribution": distribution(ranks),
                   "rank_counts": dict(sorted(Counter(ranks).items()))}
    for k in (1, 5):
        hits = sum(rank <= k for rank in ranks)
        recognition.update({f"top{k}_hits": hits, f"top{k}_accuracy": hits/len(ranks) if ranks else None})
    return {"images": len(rows), "unique_targets": len({canonical_word(r["target_word"]) for r in rows}),
            "siglip2": recognition, "text_score": distribution(scores),
            "text_interval_counts": {b: sum(r["text_score_interval"] == b for r in rows) for b in BANDS}}


def markdown(summary):
    def hit_rate(hits, n):
        return f"{hits}/{n} ({hits/n:.1%})" if n else "N/A (0 ranked)"

    def fmt(number):
        return "N/A" if number is None else f"{number:.3f}"

    counts = summary["overall"]
    lines = [f"# Combined report for {summary['run_id']} / {summary['cohort_id']}", "",
             f"Mode: {summary['mode']}. Evaluation coverage: {summary['status']}.",
             f"{counts['images']} verified images represent {counts['unique_targets']} distinct targets.", "",
             "| Source | Included | Missing |", "|---|---:|---:|"]
    for source, coverage in summary["coverage"].items():
        lines.append(f"| {source} | {len(coverage['included_ids'])} | {len(coverage['missing_ids'])} |")
    lines += ["", "| Group | Images | Unique targets | Top-1 hits / ranked | Top-5 hits / ranked | "
              "Median rank | Median text score (scored images) |", "|---|---:|---:|---:|---:|---:|---:|"]
    groups = {"All images": counts, **summary["by_imageability"], **summary["by_text_score_interval"]}
    for name, group in groups.items():
        sig, txt = group["siglip2"], group["text_score"]
        n = sig["denominator"]
        lines.append(f"| {name} | {group['images']} | {group['unique_targets']} | {hit_rate(sig['top1_hits'], n)} | "
                     f"{hit_rate(sig['top5_hits'], n)} | {fmt(sig['rank_distribution']['median'])} | "
                     f"{fmt(txt['median'])} ({txt['n']}) |")
    words = summary["generation_word_coverage"]
    lines += ["", f"Generation list coverage: {words['represented']} of {words['generation_word_count']} words. "
              f"The judge vocabulary contains {words['candidate_word_count']} candidates.",
              "", "### Source records", "", f"Manifest digest: `{summary['manifest_sha256']}`."]
    for source in SOURCES[:2]:
        entry = summary["sources"][source]
        lines.append(f"- {source}: `{entry['model']}@{entry['revision']}`.")
        for shard in entry["shards"]:
            lines.append(f"  - Shard {shard['shard']} ({shard['status']}): `{shard['export']}`.")
    txt = summary["sources"]["text_score"]
    lines.append(f"- Text scoring ({txt['status']}): `{txt['export']}`.")
    if "job" in txt:
        conf = txt["job"]["configuration"]
        lines.append(f"  - Model: `{conf['model']}@{conf['revision']}`.")
    if "last_failure_file" in txt:
        lines.append(f"  - Last recorded failed attempt: `{txt['last_failure_file']}`. "
                     "Current coverage uses verified scores.")
    lines += ["", "Text-score interval rows include scored images only. A missing denominator gives an undefined rate.",
              "", LIMITATIONS, "", "The CSV retains individual images. See combined_summary.json for exact included "
              "and missing IDs, rank counts, score distributions, source files and model configurations."]
    return "\n".join(lines) + "\n"


def report_run(flow, *, partial=False):
    """Publish a fresh report for the exact shared manifest and planned judge shards."""
    manifest = flow.manifest()
    images = {row["image_id"]: row for row in manifest["images"]}
    if len(images) != len(manifest["images"]):
        raise ValueError("Duplicate manifest image IDs.")
    candidates = load_candidates(flow.cfg.vocabulary_csv)
    vocabulary = (len(candidates), hashlib.sha256(json.dumps(candidates, ensure_ascii=False).encode()).hexdigest(),
                  tuple(sorted(w.lower() for w in candidates)))
    data, provenance, paths = {}, {}, {}
    for model in SOURCES[:2]:
        data[model], provenance[model], paths[model] = judge_source(flow, manifest, images, model, vocabulary)
    data["text_score"], provenance["text_score"] = text_source(flow.artifacts / "text_score", manifest, images)
    coverage = {s: {"included_ids": sorted(data[s]), "missing_ids": sorted(images.keys()-data[s].keys())}
                for s in SOURCES}
    incomplete = any(v["missing_ids"] for v in coverage.values()) or provenance["text_score"]["status"] != "complete"
    if incomplete and not partial:
        reasons = [f"{s}: {len(v['missing_ids'])} missing of {len(images)}" for s, v in coverage.items()]
        for model in SOURCES[:2]:
            reasons.extend(f"{j['status']}: {j['export']}" for j in provenance[model]["shards"] if j["status"] != "complete")
        reasons.append(f"text export {provenance['text_score']['status']}: {provenance['text_score']['export']}")
        raise ValueError("Incomplete production analysis. " + "; ".join(reasons))

    rows = []
    blank = dict.fromkeys((*SIG_FIELDS, *FLO_FIELDS, *TEXT_FIELDS, "text_score_interval"))
    for key, image in sorted(images.items()):
        row = {**{k: v for k, v in image.items() if k != "imageability"}, "run_id": flow.run.name,
               **(image["imageability"] or {}), **blank}
        for source in SOURCES:
            present = key in data[source]
            row[f"has_{source}"] = present
            if present:
                row.update(data[source][key])
                model_info = provenance[source]
                if source == "text_score":
                    model_info = model_info["job"]["configuration"]
                row[f"{source}_model"] = model_info["model"] + "@" + model_info["revision"]
            else:
                row[f"{source}_model"] = None
        rows.append(row)
    targets = sorted({canonical_word(r["target_word"]) for r in rows})
    generation_words = {canonical_word(w) for w in flow.words}
    observed = set(targets)
    target_coverage = {"generation_word_count": len(generation_words), "represented": len(generation_words & observed),
                       "unrepresented_generation_words": sorted(generation_words - observed),
                       "observed_outside_generation_csv": sorted(observed - generation_words),
                       "candidate_word_count": len(candidates)}
    summary = {"schema_version": 1, "run_id": flow.run.name, "cohort_id": flow.cfg.cohort_id,
               "mode": "exploratory" if partial else "production", "status": "partial" if incomplete else "complete",
               "manifest": str(flow.manifest_path), "manifest_sha256": digest(manifest),
               "config": manifest["config"], "csv_sha256": manifest["csv_sha256"],
               "source_bundle_sha256": flow.settings["bundle_sha256"], "sources": provenance, "coverage": coverage,
               "generation_word_coverage": target_coverage, "target_words": targets, "overall": measures(rows), "limitations": LIMITATIONS,
               "by_imageability": {g: measures([r for r in rows if r.get("imageability_group") == g])
                                   for g in GROUP_BINS},
               "by_text_score_interval": {b: measures([r for r in rows if r["text_score_interval"] == b])
                                          for b in BANDS}}
    paired = data["siglip2"].keys() & data["florence2"].keys()
    summary["judge_comparison_ids"] = sorted(paired)
    dest = flow.artifacts / ("analysis_exploratory" if partial else "analysis")
    if paired:
        comparison, judge_summary = analyze_exports(paths["siglip2"], paths["florence2"],
            source_roots=flow.cfg.input_roots, allow_unmatched=partial, imageability_source=flow.cfg.imageability_csv)
        if judge_summary["imageability"]["metadata"]["sha256"] != manifest["csv_sha256"]["imageability_csv"]:
            raise ValueError("Imageability metadata changed during analysis.")
        for row in comparison:
            if row["image_sha256"] != images[row["image_id"]]["image_sha256"]:
                raise ValueError(f"Verified image changed during analysis: {row['image_id']}")
        for source in SOURCES[:2]:
            verified = {s["export"]: s["export_sha256"] for s in provenance[source]["shards"] if s["status"] == "complete"}
            if verified != {s["path"]: s["sha256"] for s in judge_summary["inputs"][source]}:
                raise ValueError(f"{source} inputs changed during analysis.")
    else:
        comparison, judge_summary = [], None
    publish(dest, rows, summary, comparison, judge_summary)
    return dest, summary


def publish(dest, rows, summary, comparison, judge_summary):
    """Serialize on the destination filesystem. Exclusively claim the final directory."""
    if dest.exists():
        raise FileExistsError(dest)
    with tempfile.TemporaryDirectory(dir=dest.parent) as scratch:
        staged = Path(scratch) / "reports"
        if comparison:
            write_report(comparison, judge_summary, staged)
        else:
            staged.mkdir()
        with (staged / "combined.csv").open("x", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            for row in rows:
                writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, list) else v
                                 for k, v in row.items()})
        with (staged / "combined_summary.json").open("x", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
        report = staged / "combined_report.md"
        report.write_text(markdown(summary), encoding="utf-8")
        dest.mkdir(exist_ok=False)
        for file in staged.iterdir():
            if file != report:
                file.rename(dest / file.name)
        report.rename(dest / report.name)  # Visible completion marker, published last.
