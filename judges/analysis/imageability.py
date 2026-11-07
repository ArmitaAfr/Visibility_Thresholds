"""Strict word-metadata joins and descriptive imageability strata; no inference."""

import csv
import hashlib
import io
import math
from collections import Counter
from pathlib import Path
from statistics import mean, median

# Existing dataset labels and boundaries, documented in dataset/README.md and
# emitted by build_dataset.py. Assign groups from labels, never from new cuts.
GROUP_BINS = {"low": "[1, 3]", "medium": "(3, 5]", "high": "(5, 7]"}
BIN_GROUPS = {label: group for group, label in GROUP_BINS.items()}
SCORE = "glasgow_imageability_mean"
OPTIONAL = ("glasgow_imageability_sd", "glasgow_imageability_n")


def canonical_word(word):
    """Match the dataset builder and batch vocabulary: strip and lowercase only."""
    return word.strip().lower()


class ImageabilityMetadataError(ValueError):
    def __init__(self, issues):
        self.issues = issues
        super().__init__("Imageability metadata validation failed: " + "; ".join(
            f"{issue['code']}: {issue['message']}" for issue in issues))


def load_imageability(source):
    source = Path(source).resolve()
    content = source.read_bytes()
    table = csv.DictReader(io.StringIO(content.decode("utf-8-sig"), newline=""), strict=True)
    columns = table.fieldnames or []
    missing = sorted({"word", "score_bin", SCORE} - set(columns))
    if missing or len(columns) != len(set(columns)):
        raise ImageabilityMetadataError([{"code": "invalid_columns", "message":
            f"Missing required columns: {missing}; duplicate headers: {len(columns) != len(set(columns))}."}])
    index, seen, issues = {}, {}, []
    counts = Counter()
    row_count = 0
    for line, row in enumerate(table, start=2):
        row_count += 1
        word = canonical_word(row.get("word") or "")
        if None in row or any(value is None for value in row.values()):
            issues.append({"code": "malformed_row", "message": f"CSV record {line} has missing or extra cells."})
        if not word:
            issues.append({"code": "missing_word", "message": f"CSV record {line} has no canonical word."})
            continue
        signature = tuple((row.get(k) or "").strip() for k in ("score_bin", SCORE, *OPTIONAL))
        if word in seen:
            prior_line, prior_signature = seen[word]
            code = "duplicate_word" if signature == prior_signature else "ambiguous_word"
            issues.append({"code": code, "word": word, "message":
                f"{word!r} occurs in CSV records {prior_line} and {line}; no first/last-match selection is allowed."})
        else:
            seen[word] = (line, signature)
        label = (row.get("score_bin") or "").strip()
        if label not in BIN_GROUPS:
            issues.append({"code": "missing_or_unknown_category", "word": word,
                           "message": f"{word!r}, CSV record {line}: invalid score_bin {label!r}."})
            continue
        try:
            score = float(row.get(SCORE) or "")
            if not math.isfinite(score):
                raise ValueError("nonfinite score")
            # Validate consistency with the recorded source interval. Do not
            # assign missing labels from scores, or override recorded labels.
            valid = {"[1, 3]": 1 <= score <= 3, "(3, 5]": 3 < score <= 5,
                     "(5, 7]": 5 < score <= 7}[label]
            if not valid:
                raise ValueError(f"score {score} is inconsistent with recorded bin {label}")
            optional = {}
            for column in OPTIONAL:
                value = None if column not in columns else float(row.get(column) or "")
                if value is not None and (not math.isfinite(value) or value < 0):
                    raise ValueError(f"invalid {column}")
                if column.endswith("_n") and value is not None:
                    if value < 1 or not value.is_integer():
                        raise ValueError(f"{column} must be a positive integer")
                    value = int(value)
                optional[column] = value
        except ValueError as exc:
            issues.append({"code": "incomplete_or_inconsistent_score", "word": word,
                           "message": f"{word!r}, CSV record {line}: {exc}."})
            continue
        index[word] = {"canonical_target_word": word, "imageability_metadata_word": row["word"].strip(),
                       "imageability_group": BIN_GROUPS[label], "score_bin": label,
                       SCORE: score, **optional}
        counts[BIN_GROUPS[label]] += 1
    if not row_count:
        issues.append({"code": "empty_metadata", "message": "The metadata CSV has no word records."})
    if issues:
        raise ImageabilityMetadataError(issues)
    return index, {
        "path": str(source), "sha256": hashlib.sha256(content).hexdigest(),
        "word_column": "word", "category_column": "score_bin", "score_column": SCORE,
        "canonicalization": "strip whitespace, lowercase; no fuzzy matching or stemming",
        "group_bin_mapping": GROUP_BINS, "metadata_word_rows": row_count,
        "metadata_unique_words": len(index), "category_word_counts": {g: counts[g] for g in GROUP_BINS},
        "categories_absent_from_metadata": [g for g in GROUP_BINS if not counts[g]],
        "optional_columns_absent": [column for column in OPTIONAL if column not in columns],
        "validation_issues": [],
    }


def stratify_imageability(rows, summary, source):
    """Enrich copies of comparison rows, retaining every matched image observation."""
    index, audit = load_imageability(source)
    missing = sorted({canonical_word(row["target_word"]) for row in rows} - index.keys())
    if missing:
        raise ImageabilityMetadataError([{"code": "missing_target_match", "word": word,
            "message": f"Evaluated target {word!r} has no metadata match."} for word in missing])
    enriched = [{**row, **index[canonical_word(row["target_word"]) ]} for row in rows]
    vocabulary_size = summary["siglip2"]["vocabulary_size"]
    groups = {}
    for group, label in GROUP_BINS.items():
        selected = [row for row in enriched if row["imageability_group"] == group]
        n = len(selected)
        ranks = [row["siglip2_target_rank"] for row in selected]
        top1 = sum(rank == 1 for rank in ranks)
        top5 = sum(rank <= min(5, vocabulary_size) for rank in ranks)
        patterns = {}
        # Reuse the existing Phase 4 outputs/cues without redefining heuristics.
        for category, column in (("apparent_text", "florence2_text_mention_cues"),
                                 ("objects", "florence2_object_cues"),
                                 ("abstract_concepts", "florence2_abstract_concept_cues")):
            count = sum(bool(row[column]) for row in selected)
            cues = Counter(cue for row in selected for cue in row[column])
            patterns[category] = {"captions_with_cues": count, "denominator": n,
                "fraction": count / n if n else None, "percentage": 100 * count / n if n else None,
                "cue_caption_counts": dict(sorted(cues.items()))}
        groups[group] = {
            "score_bin": label, "status": "evaluated" if n else "empty",
            "evaluated_images": n, "unique_target_words": len({row["canonical_target_word"] for row in selected}),
            "target_words": sorted({row["canonical_target_word"] for row in selected}),
            "siglip2": {"denominator": n, "vocabulary_size": vocabulary_size,
                "vocabulary_sha256": summary["siglip2"]["vocabulary_sha256"],
                "top1_target_hits": top1, "top1_target_hit_rate": top1 / n if n else None,
                "top5_target_hits": top5, "top5_target_hit_rate": top5 / n if n else None,
                "mean_target_rank": mean(ranks) if n else None,
                "median_target_rank": median(ranks) if n else None},
            "florence2_patterns": patterns,
        }
    extended = {**summary, "imageability": {
        "stratification_version": 1, "metadata": audit, "groups": groups,
        "coverage": {"matched_images": len(enriched),
            "matched_unique_target_words": len({row["canonical_target_word"] for row in enriched}),
            "excluded_images_due_to_metadata": 0, "missing_target_words": [],
            "empty_evaluated_groups": [g for g in GROUP_BINS if groups[g]["status"] == "empty"]},
        "interpretation": "Descriptive per-image summaries only. Repeated target words remain separate image observations; no causal, significance, internal-attention or human-recognition claims.",
    }}
    return enriched, extended


def imageability_markdown(stratification):
    def number(value):
        return "N/A" if value is None else f"{value:.3f}"

    def percentage(value):
        return "N/A" if value is None else f"{100 * value:.2f}%"

    audit = stratification["metadata"]
    lines = ["", "## Imageability-stratified analysis", "", stratification["interpretation"], "",
             f"Metadata: {audit['path']} (SHA-256: {audit['sha256']}).",
             "Categories reuse recorded score_bin labels; scores are glasgow_imageability_mean. No new thresholds were estimated.", "",
             "| Category | Source bin | Images | Unique words | Top-1 hits | Top-1 rate | Top-5 hits | Top-5 rate | Mean rank | Median rank |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for group, data in stratification["groups"].items():
        sig = data["siglip2"]
        lines.append(f"| {group} | {data['score_bin']} | {data['evaluated_images']} | {data['unique_target_words']} | "
            f"{sig['top1_target_hits']} | {percentage(sig['top1_target_hit_rate'])} | {sig['top5_target_hits']} | "
            f"{percentage(sig['top5_target_hit_rate'])} | {number(sig['mean_target_rank'])} | {number(sig['median_target_rank'])} |")
    lines += ["", "### Florence caption lexical heuristics", "",
              "These are caption word-list mentions, not verified OCR, semantic recognition, or evidence of what the model internally attends to. Categories overlap; negated mentions still count.", "",
              "| Category | Images | Apparent-writing mentions | Percentage | Object-related cues | Percentage | Abstract-concept cues | Percentage |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for group, data in stratification["groups"].items():
        p = data["florence2_patterns"]
        cells = [str(p[k]["captions_with_cues"]) + " | " + percentage(p[k]["fraction"])
                 for k in ("apparent_text", "objects", "abstract_concepts")]
        lines.append(f"| {group} | {data['evaluated_images']} | " + " | ".join(cells) + " |")
    coverage = stratification["coverage"]
    lines += ["", f"Metadata validation issues: {audit['validation_issues']} (strict validation; no unmatched metadata records were dropped).",
              f"Categories absent from metadata: {audit['categories_absent_from_metadata']}.",
              f"Evaluated groups with zero images: {coverage['empty_evaluated_groups']}; undefined rates/ranks are N/A (JSON null), not zero.",
              f"Optional metadata columns absent: {audit['optional_columns_absent']}.",
              f"Joined {coverage['matched_images']} generated images and {coverage['matched_unique_target_words']} unique target words; excluded for missing metadata: 0.", ""]
    return lines
