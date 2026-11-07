"""Read-only validation of locally available processed word datasets (stdlib only)."""

import argparse
import csv
import hashlib
import io
import json
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path

GROUPS = {"[1, 3]": "low", "(3, 5]": "medium", "(5, 7]": "high"}
MEAN = "glasgow_imageability_mean"
SD = "glasgow_imageability_sd"
RATERS = "glasgow_imageability_n"
REQUIRED = ("word", "length", "pos", "pos_share", "score_bin", MEAN, SD, RATERS)
FILES = ("cleaned_words.csv", "noun_words.csv", "sampled_words.csv",
         *(f"noun_words_split/{i}.csv" for i in range(1, 7)))
DEFAULT_PROCESSED = Path(__file__).resolve().parent / "processed"


def validate_dataset(processed_dir=DEFAULT_PROCESSED, *, expected_sample_size=150):
    """Return all diagnostics without rebuilding, repairing or fetching any data."""
    if type(expected_sample_size) is not int or expected_sample_size < 1:
        raise ValueError("expected_sample_size must be a positive integer.")
    root = Path(processed_dir).resolve()
    issues, files, tables = [], {}, {}

    def issue(code, message, *, file=None, word=None, record=None, severity="error", **details):
        issues.append({"severity": severity, "code": code, "file": file, "word": word,
                       "record": record, "message": message, **details})

    for name in FILES:
        path = root / name
        stats = {"path": str(path), "exists": path.is_file(), "read_complete": False,
                 "rows": None, "unique_words": None, "group_counts": {g: None for g in GROUPS.values()},
                 "missing_values": {}, "duplicate_words": [], "columns": [], "sha256": None}
        files[name] = stats
        table = {"words": set(), "noun_words": set(), "index": {}, "ready": False, "metadata_valid": False}
        tables[name] = table
        if not stats["exists"]:
            issue("missing_file", f"Required local file is unavailable: {path}. No rebuild or remote lookup was attempted.", file=name)
            stats["status"] = "missing"
            continue
        start_errors = len(issues)
        try:
            content = path.read_bytes()
            stats["sha256"] = hashlib.sha256(content).hexdigest()
            stats.update(rows=0, unique_words=0, group_counts={g: 0 for g in GROUPS.values()})
            reader = csv.DictReader(io.StringIO(content.decode("utf-8-sig"), newline=""), strict=True)
            columns = reader.fieldnames or []
            stats["columns"] = columns
            absent = sorted(set(REQUIRED) - set(columns))
            if absent or len(columns) != len(set(columns)):
                issue("invalid_schema", f"Missing columns: {absent}; duplicate headers: {len(columns) != len(set(columns))}.", file=name)
            missing, seen, group_counts = Counter(), {}, Counter()
            for number, row in enumerate(reader, start=2):
                stats["rows"] += 1
                if None in row or any(value is None for value in row.values()):
                    issue("malformed_row", "CSV record has missing or extra cells.", file=name, record=number)
                for column in columns:
                    if not (row.get(column) or "").strip():
                        missing[column] += 1
                        issue("missing_value", f"Blank/missing value in {column}.", file=name, record=number,
                              severity="error" if column in REQUIRED else "warning", column=column)
                word = (row.get("word") or "").strip().lower()
                if not word:
                    continue
                table["words"].add(word)
                if word in seen:
                    issue("duplicate_word", f"Canonical word {word!r} occurs in records {seen[word]} and {number}.",
                          file=name, word=word, record=number)
                else:
                    seen[word] = number
                if not word.isascii() or not word.isalpha():
                    issue("invalid_word", "Processed words must be single alphabetic ASCII words.", file=name, word=word, record=number)
                pos = (row.get("pos") or "").strip()
                if pos == "Noun":
                    table["noun_words"].add(word)
                if name != "cleaned_words.csv" and pos != "Noun":
                    issue("non_noun", f"Expected pos='Noun', got {pos!r}.", file=name, word=word, record=number)
                label = (row.get("score_bin") or "").strip()
                if label not in GROUPS:
                    issue("invalid_imageability_label", f"Unknown/missing source score_bin {label!r}.", file=name, word=word, record=number)
                    continue
                group_counts[GROUPS[label]] += 1
                if name.startswith("noun_words_split/"):
                    split_number = int(Path(name).stem)
                    expected_group = ("low", "medium", "high")[(split_number - 1) // 2]
                    if GROUPS[label] != expected_group:
                        issue("wrong_split_group", f"Split {split_number} must contain {expected_group} words, got {GROUPS[label]}.", file=name, word=word, record=number)
                try:
                    numeric = {}
                    for key in ("length", "pos_share", MEAN, SD, RATERS):
                        raw = (row.get(key) or "").strip()
                        try:
                            value = Decimal(raw)
                        except InvalidOperation:
                            raise ValueError(f"{key} must be numeric, got {raw!r}.") from None
                        if not value.is_finite():
                            raise ValueError(f"{key} must be finite, got {raw!r}.")
                        numeric[key] = value
                    score = numeric[MEAN]
                    consistent = {"[1, 3]": 1 <= score <= 3, "(3, 5]": 3 < score <= 5,
                                  "(5, 7]": 5 < score <= 7}[label]
                    if not consistent:
                        raise ValueError(f"Imageability mean {score} disagrees with recorded bin {label}.")
                    if numeric[SD] < 0:
                        raise ValueError("Imageability SD must be nonnegative.")
                    if numeric[RATERS] < 1 or numeric[RATERS] != numeric[RATERS].to_integral_value():
                        raise ValueError("Imageability rater count must be a positive integer.")
                    if numeric["length"] != len(word):
                        raise ValueError("Word length does not match the canonical word.")
                    if not 0 <= numeric["pos_share"] <= 1:
                        raise ValueError("pos_share must lie in [0, 1].")
                except (InvalidOperation, ValueError) as exc:
                    issue("invalid_metadata", str(exc), file=name, word=word, record=number)
                    continue
                if word not in table["index"]:
                    table["index"][word] = {"score_bin": label, "pos": pos, **numeric}
            stats["missing_values"] = dict(sorted(missing.items()))
            stats["group_counts"] = {g: group_counts[g] for g in GROUPS.values()}
            stats["unique_words"] = len(table["words"])
            stats["read_complete"] = True
            table["ready"] = not absent and len(columns) == len(set(columns))
            if not stats["rows"]:
                issue("empty_file", "Required dataset file has no word rows.", file=name)
        except (OSError, UnicodeError, csv.Error) as exc:
            issue("unreadable_file", f"Cannot fully read CSV: {exc}", file=name)
        new_errors = [i for i in issues[start_errors:] if i["severity"] == "error"]
        stats["duplicate_words"] = sorted({i["word"] for i in new_errors if i["code"] == "duplicate_word"})
        stats["status"] = "invalid" if new_errors else "valid"
        table["metadata_valid"] = stats["read_complete"] and not new_errors

    checks = {}

    def check(name, dependencies, differences, message):
        unavailable = [f for f in dependencies if not tables[f]["ready"] or not files[f]["read_complete"]]
        if unavailable:
            checks[name] = {"status": "blocked", "unavailable_files": unavailable}
            return
        checks[name] = {"status": "failed" if differences else "passed", **differences}
        if differences:
            issue(name, message, differences=differences)

    nouns = tables["noun_words.csv"]["words"]
    sampled = tables["sampled_words.csv"]["words"]
    splits = list(FILES[3:])
    owners = {}
    for name in splits:
        for word in tables[name]["words"]:
            owners.setdefault(word, []).append(name)
    overlaps = {word: sorted(names) for word, names in sorted(owners.items()) if len(names) > 1}
    check("split_disjointness", splits, {"overlapping_words": overlaps} if overlaps else {}, "Nouns occur in multiple splits.")
    union = set(owners)
    diff = {"missing_from_splits": sorted(nouns - union), "extra_in_splits": sorted(union - nouns)}
    check("split_union_matches_nouns", ["noun_words.csv", *splits], diff if any(diff.values()) else {},
          "The six-split union differs from the full noun dataset.")
    absent_sample = sorted(sampled - nouns)
    check("sample_subset_of_nouns", ["sampled_words.csv", "noun_words.csv"],
          {"missing_sampled_words": absent_sample} if absent_sample else {}, "Sampled words are absent from noun_words.csv.")
    check("sample_size", ["sampled_words.csv"],
          {"expected": expected_sample_size, "rows": files["sampled_words.csv"]["rows"], "unique_words": len(sampled)}
          if len(sampled) != expected_sample_size or files["sampled_words.csv"]["rows"] != expected_sample_size else {},
          "Sampled row/unique-word count does not match the expected sample size.")
    cleaned_nouns = tables["cleaned_words.csv"]["noun_words"]
    diff = {"missing_from_noun_dataset": sorted(cleaned_nouns - nouns), "not_in_cleaned_nouns": sorted(nouns - cleaned_nouns)}
    check("nouns_match_cleaned_noun_subset", ["cleaned_words.csv", "noun_words.csv"], diff if any(diff.values()) else {},
          "noun_words.csv differs from cleaned_words.csv's Noun subset.")

    references, mismatches, skipped = {}, [], []
    for name in FILES:
        if not tables[name]["metadata_valid"]:
            skipped.append(name)
            continue
        for word, entry in tables[name]["index"].items():
            if word not in references:
                references[word] = (name, entry)
                continue
            prior_name, prior_entry = references[word]
            changed = {key: {"reference": str(prior_entry[key]), "observed": str(value)}
                       for key, value in entry.items() if value != prior_entry[key]}
            if changed:
                mismatches.append({"word": word, "reference_file": prior_name, "file": name, "columns": changed})
                issue("cross_file_metadata_mismatch", f"Metadata for {word!r} differs between {prior_name} and {name}.",
                      file=name, word=word, columns=changed)
    checks["cross_file_metadata_consistency"] = {
        "status": "failed" if mismatches else "blocked" if skipped else "passed",
        "available_files_status": "failed" if mismatches else "passed",
        "compared_files": [f for f in FILES if f not in skipped], "skipped_files": skipped,
        "mismatches": mismatches, "compared_columns": [c for c in REQUIRED if c != "word"],
    }
    errors = [i for i in issues if i["severity"] == "error"]
    complete = all(stats["read_complete"] for stats in files.values())
    return {
        "validator_version": 1, "processed_dir": str(root), "valid": not errors and complete,
        "complete": complete, "status": "invalid" if any(i["code"] != "missing_file" for i in errors)
            else "incomplete" if not complete else "valid",
        "scope": "Only local files were read. No main/remote comparison, rebuild, data integration or freshness claim.",
        "required_columns": list(REQUIRED), "imageability_labels": GROUPS,
        "expected_sample_size": expected_sample_size, "files": files,
        "checks": checks, "error_count": len(errors),
        "warning_count": sum(i["severity"] == "warning" for i in issues), "issues": issues,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED)
    parser.add_argument("--expected-sample-size", type=int, default=150)
    parser.add_argument("--output", type=Path, help="Optional NEW JSON report outside the processed data directory; JSON is always printed.")
    args = parser.parse_args(argv)
    if args.output:
        output = args.output.resolve()
        if output.is_relative_to(args.processed_dir.resolve()) or output.suffix.lower() != ".json" or output.exists():
            parser.error("--output must be a new .json file outside the processed data directory.")
    summary = validate_dataset(args.processed_dir, expected_sample_size=args.expected_sample_size)
    serialized = json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(serialized)
    print(serialized, end="")
    return 0 if summary["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
