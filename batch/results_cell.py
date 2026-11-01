# Results cell: run after batches finish.
#   %run -i batch/results_cell.py      (or paste this file into a cell)
#
# Uses WORDS_SOURCE and OUTPUT_ROOT:
#   - OUTPUT_ROOT = one run's folder                  -> results for that run
#   - OUTPUT_ROOT = shared folder with part1, part2… -> combines every part
# Saves batch_summary_all.csv (when combining) and results_with_scores.csv in OUTPUT_ROOT.

from pathlib import Path

import pandas as pd

MISSING = ["", "NA", "N/A", "NaN", "nan", "null", "NULL", "None"]


def _read_csv_keep_words(path):
    """Read a CSV keeping words like 'null' as text; also adds a lowercase '_key' for matching."""
    header = pd.read_csv(path, nrows=0, encoding="utf-8-sig").columns
    word_col = next((c for c in header if str(c).strip().lower() in ("word", "words")), None)
    if word_col is None:
        raise KeyError(f"No 'word' column in {path}. Columns are: {list(header)}")
    table = pd.read_csv(path, encoding="utf-8-sig", keep_default_na=False, dtype={word_col: str},
                        na_values={c: MISSING for c in header if c != word_col})
    table["_key"] = table[word_col].str.strip().str.lower()
    return table, word_col


root = Path(OUTPUT_ROOT)
part_files = sorted(root.glob("part*/batch_summary.csv"))
summary_files = part_files or [root / "batch_summary.csv"]
if not summary_files[0].exists():
    raise FileNotFoundError(f"No batch_summary.csv in {root} or its part folders.")

summary = pd.concat([_read_csv_keep_words(p)[0] for p in summary_files], ignore_index=True)
summary = summary.drop_duplicates("_key", keep="last")
if part_files:
    summary.drop(columns="_key").to_csv(root / "batch_summary_all.csv", index=False)
    print(f"Combined {len(part_files)} parts: {[p.parent.name for p in part_files]}")

scores, word_col = _read_csv_keep_words(WORDS_SOURCE)
results_table = summary.merge(scores.drop(columns=[word_col]), on="_key", how="left").drop(columns="_key")
results_table.to_csv(root / "results_with_scores.csv", index=False)

unmatched = len(results_table) - summary["_key"].isin(scores["_key"]).sum()
print(results_table["status"].value_counts().to_string())
if unmatched:
    print(f"Note: {unmatched} word(s) had no scores in {WORDS_SOURCE}.")
print(f"Saved {len(results_table)} rows to {root / 'results_with_scores.csv'}")
results_table.head(20)
