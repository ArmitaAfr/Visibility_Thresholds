from pathlib import Path

import pandas as pd

WORD_COLUMN_NAMES = {"word", "words"}


def load_words(source, column=None):
    """
    Load words from a CSV file, e.g. dataset/processed/sampled_words.csv.

    source : path to a .csv file, or an http(s) link to one (e.g. a raw GitHub URL).
    column : name of the column holding the words. If None, uses a column named
             "word" or "words" (any capitalization).
    Blank cells are skipped and duplicates removed ("Dog" and "dog" count as one).
    Words like "null" or "NA" are kept as words.
    """
    source = str(source).strip()
    is_url = source.startswith(("http://", "https://"))
    name = source.split("?")[0].rstrip("/").rsplit("/", 1)[-1] if is_url else Path(source).name

    if not name.lower().endswith(".csv"):
        raise ValueError(f"Expected a .csv file, got {name!r}.")
    if not is_url and not Path(source).exists():
        raise FileNotFoundError(f"No file at {source}. Check the path, or mount Drive / clone the repo first.")

    table = pd.read_csv(source, encoding="utf-8-sig", keep_default_na=False, dtype=str)

    if column is None:
        matches = [c for c in table.columns if str(c).strip().lower() in WORD_COLUMN_NAMES]
        if not matches:
            raise KeyError(f"No 'word' or 'words' column found. Columns are: {list(table.columns)}. "
                           "Pass column='...' to choose one.")
        column = matches[0]
    elif column not in table.columns:
        raise KeyError(f"Column {column!r} not found. Columns are: {list(table.columns)}")

    words, seen = [], set()
    for value in table[column]:
        word = value.strip()
        if word and word.lower() not in seen:
            seen.add(word.lower())
            words.append(word)

    if not words:
        raise ValueError(f"No words found in column {column!r} of {name}.")
    print(f"Loaded {len(words)} words from {name} (column {column!r}). First few: {words[:5]}")
    return words
