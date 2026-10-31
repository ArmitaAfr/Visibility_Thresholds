"""Merge word norms, clean the merged list, and draw a reproducible, stratified noun sample."""

import argparse
import json
import os
from pathlib import Path

import pandas as pd

DATASET_DIR = Path(__file__).resolve().parent
DATA_DIR = DATASET_DIR / "raw"
OUTPUT_DIR = DATASET_DIR / "processed"
SCORE = "glasgow_imageability_mean"
EXAMPLES_PER_RULE = 8


def display_path(path):
    # Relative to dataset/ so the report is identical on every machine.
    return Path(os.path.relpath(path.resolve(), DATASET_DIR)).as_posix()


def load_glasgow(path):
    # The dimension name appears only above the first statistic in each group.
    frame = pd.read_csv(path, header=[0, 1], encoding="utf-8-sig")
    columns = []
    dimension = None
    statistics = {"mean": "mean", "m": "mean", "sd": "sd",
                  "# of raters": "n", "n": "n"}
    for top, bottom in frame.columns:
        if not str(top).startswith("Unnamed:"):
            dimension = str(top).strip().lower().replace(" ", "_")
        statistic = str(bottom).strip().lower()
        if dimension in {"words", "word"}:
            columns.append("word")
        elif dimension == "length":
            columns.append("length")
        else:
            if statistic not in statistics:
                raise ValueError(f"Unknown Glasgow statistic: {bottom!r}")
            columns.append(f"glasgow_{dimension}_{statistics[statistic]}")
    frame.columns = columns
    return frame


def load_word_list(path, rename):
    # keep_default_na=False stops pandas from reading words such as "null" or "nan" as missing.
    frame = pd.read_csv(path, encoding="utf-8-sig", keep_default_na=False, dtype={"Word": str})
    return frame.rename(columns=rename)


def normalize_words(frame):
    frame = frame.copy()
    frame["word"] = frame["word"].astype("string").str.strip().str.lower()
    frame = frame.loc[frame["word"].notna() & frame["word"].ne("")]
    duplicates = frame.loc[frame["word"].duplicated(False), "word"].unique()
    if len(duplicates):
        raise ValueError(f"Duplicate normalized words: {duplicates[:10].tolist()}")
    return frame


def merge_sources(glasgow, ratings, pos):
    """Inner-join Glasgow with the ratings, then attach SUBTLEX-US part of speech."""
    merged = glasgow.merge(ratings, on="word", how="inner", validate="one_to_one")
    merged = merged.merge(pos, on="word", how="left", validate="one_to_one")
    for column in ["Bigram", "Percent_known", "SUBTLEX", SCORE]:
        merged[column] = pd.to_numeric(merged[column], errors="coerce")
    return merged.sort_values("word").reset_index(drop=True)


def load_word_set(path):
    """Read one word per line, ignoring blank lines and # comments."""
    lines = (line.split("#", 1)[0].strip().lower() for line in path.read_text(encoding="utf-8").splitlines())
    return {line for line in lines if line}


def bin_labels(edges):
    return [f"{'[' if index == 0 else '('}{left:g}, {right:g}]"
            for index, (left, right) in enumerate(zip(edges, edges[1:]))]


def clean(merged, glasgow, args):
    """Apply the cleaning rules in order. Returns the cleaned words and what each rule removed."""
    # Glasgow rates ambiguous words once per sense ("bank (money)", "bank (river)") and once unmarked.
    senses = glasgow["word"].str.extract(r"^(.*?)\s*\(", expand=False).dropna().str.strip()
    ambiguous = set(senses)
    vulgar = load_word_set(args.vulgar_words)
    score_bin = pd.cut(merged[SCORE], bins=args.bin_edges, include_lowest=True, right=True,
                       labels=bin_labels(args.bin_edges))
    rules = [
        ("not_single_word", "Not a single alphabetic word (space, hyphen, punctuation, or Bigram != 0)",
         ~(merged["word"].str.fullmatch(r"[a-z]+", na=False) & merged["Bigram"].eq(0))),
        ("ambiguous", "Glasgow lists separate senses for the word, e.g. bark (dog) / bark (tree)",
         merged["word"].isin(ambiguous)),
        ("not_well_known", f"Percent_known < {args.min_known:g}",
         ~merged["Percent_known"].ge(args.min_known)),
        ("too_rare", f"SUBTLEX < {args.min_subtlex:g}",
         ~merged["SUBTLEX"].ge(args.min_subtlex)),
        ("vulgar", f"Profanity, sexual or otherwise vulgar word listed in {display_path(args.vulgar_words)}",
         merged["word"].isin(vulgar)),
        ("outside_bins", f"Imageability outside {args.bin_edges[0]:g}-{args.bin_edges[-1]:g}",
         score_bin.isna()),
    ]
    keep = pd.Series(True, index=merged.index)
    removed = {}
    for name, description, drop in rules:
        newly_removed = keep & drop
        removed[name] = {
            "description": description,
            "count": int(newly_removed.sum()),
            "examples": merged.loc[newly_removed, "word"].head(EXAMPLES_PER_RULE).tolist(),
        }
        keep &= ~drop
    cleaned = merged.loc[keep].copy()
    cleaned["score_bin"] = score_bin[keep]
    return cleaned.reset_index(drop=True), removed


def sample(cleaned, args):
    candidates = cleaned.loc[cleaned["pos"].eq(args.sample_pos)]
    parts = []
    bin_counts = {}
    for interval in cleaned["score_bin"].cat.categories:
        group = candidates.loc[candidates["score_bin"] == interval]
        count = min(args.per_bin, len(group))
        bin_counts[str(interval)] = {
            "cleaned": int(cleaned["score_bin"].eq(interval).sum()),
            "candidates": len(group),
            "sampled": count,
        }
        if count < args.per_bin:
            print(f"Warning: {interval} has only {count} {args.sample_pos} words; using all available.")
        parts.append(group.sample(n=count, random_state=args.seed))
    sampled = pd.concat(parts).sort_values(["score_bin", "word"]).reset_index(drop=True)
    return sampled, bin_counts


def build_dataset(args):
    glasgow_raw = load_glasgow(args.glasgow)
    ratings_raw = load_word_list(args.ratings, {
        "Word": "word", "Conc.M": "ratings_concreteness_mean", "Conc.SD": "ratings_concreteness_sd"})
    pos_raw = load_word_list(args.pos, {
        "Word": "word", "Dom_PoS_SUBTLEX": "pos", "Percentage_dom_PoS": "pos_share"})
    glasgow = normalize_words(glasgow_raw)
    ratings = normalize_words(ratings_raw)
    pos = normalize_words(pos_raw)[["word", "pos", "pos_share"]]

    merged = merge_sources(glasgow, ratings, pos)
    cleaned, removed = clean(merged, glasgow, args)
    sampled, bin_counts = sample(cleaned, args)

    report = {
        "counts": {"glasgow_raw": len(glasgow_raw), "ratings_raw": len(ratings_raw),
                   "pos_raw": len(pos_raw), "merged": len(merged), "cleaned": len(cleaned),
                   "sampled": len(sampled)},
        "removed": removed,
        "cleaned_pos": {str(key): int(value) for key, value in
                        cleaned["pos"].fillna("missing").value_counts().items()},
        "bins": bin_counts,
        "parameters": {"glasgow": display_path(args.glasgow),
                       "ratings": display_path(args.ratings),
                       "pos": display_path(args.pos),
                       "vulgar_words": display_path(args.vulgar_words),
                       "score": SCORE,
                       "min_known": args.min_known, "min_subtlex": args.min_subtlex,
                       "bin_edges": args.bin_edges, "per_bin": args.per_bin,
                       "sample_pos": args.sample_pos,
                       "seed": args.seed, "pandas_version": pd.__version__},
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in [("cleaned", cleaned), ("sampled", sampled)]:
        frame.to_csv(args.output_dir / f"{name}_words.csv", index=False)
    (args.output_dir / "build_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return cleaned, sampled


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--glasgow", type=Path, default=DATA_DIR / "glasgow.csv")
    parser.add_argument("--ratings", type=Path, default=DATA_DIR / "ratings.csv")
    parser.add_argument("--pos", type=Path, default=DATA_DIR / "subtlex_pos.csv")
    parser.add_argument("--vulgar-words", type=Path, default=DATASET_DIR / "vulgar_words.txt",
                        help="Word list removed by the vulgar cleaning rule")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--min-known", type=float, default=0.9)
    parser.add_argument("--min-subtlex", type=float, default=10)
    parser.add_argument("--bin-edges", type=float, nargs="+", default=[1, 3, 5, 7])
    parser.add_argument("--per-bin", type=int, default=50)
    parser.add_argument("--sample-pos", default="Noun",
                        help="Dominant SUBTLEX-US part of speech to sample (default: Noun)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 0 <= args.min_known <= 1:
        parser.error("--min-known must be between 0 and 1")
    if not 0 <= args.min_subtlex < float("inf") or args.per_bin < 1:
        parser.error("--min-subtlex must be finite and nonnegative; --per-bin must be positive")
    if len(args.bin_edges) < 2 or not all(
        float("-inf") < left < right < float("inf")
        for left, right in zip(args.bin_edges, args.bin_edges[1:])
    ):
        parser.error("--bin-edges must contain at least two finite, increasing values")
    return args


if __name__ == "__main__":
    build_dataset(parse_args())
