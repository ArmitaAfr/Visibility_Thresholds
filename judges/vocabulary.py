"""Local CSV vocabulary loading without a pandas or GPU dependency."""

import csv
from pathlib import Path

DEFAULT_WORDS_SOURCE = Path(__file__).resolve().parents[1] / "dataset/processed/sampled_words.csv"


def load_candidates(source: str | Path = DEFAULT_WORDS_SOURCE) -> list[str]:
    """Preserve dataset order and spelling; match batch loader's cleanup rules.

    Strip whitespace, discard blanks, deduplicate with lower(), retain literal
    NA/null. Deliberately accepts local CSVs only; download URL sources first.
    """
    source = Path(source)
    if source.suffix.lower() != ".csv":
        raise ValueError("Candidate vocabulary must be a CSV file.")
    with source.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        column = next((c for c in reader.fieldnames or [] if c.strip().lower() in {"word", "words"}), None)
        if column is None:
            raise ValueError("Candidate CSV needs a word/words column.")
        words, seen = [], set()
        for row in reader:
            word = (row.get(column) or "").strip()
            if word and word.lower() not in seen:
                words.append(word)
                seen.add(word.lower())
    if not words:
        raise ValueError("Candidate vocabulary is empty.")
    return words
