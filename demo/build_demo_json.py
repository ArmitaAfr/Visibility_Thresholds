"""
Join the team judges' results for the stroke-budget images with the stroke order,
producing the JSON the demo page loads ("Load results JSON").

Inputs:
  combined.csv   from the evaluation notebook's ANALYZE stage, run on the b### folders
                 written by demo/export_budgets.py
  strokes root   OUT_ROOT/strokes from demo/export_budgets.py

Usage (Colab or anywhere with Python 3, no ML libraries needed):

    python demo/build_demo_json.py \
        --combined ".../runs/<RUN_ID>/stages/<COHORT_ID>/analysis/combined.csv" \
        --strokes  ".../demo_inputs/prune_v1/strokes" \
        --out      ".../demo_inputs/prune_v1/demo_results.json"

Recognized (for the threshold) means SigLIP2 ranks the target word first among
the vocabulary (top-1, the main metric in combined_report.md). Florence's caption
and the text score are shown on the page but do not decide the threshold.
"""

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

BUDGET_PATH = re.compile(r"[/\\]b(\d{3})[/\\]([^/\\]+)[/\\]([^/\\]+)[/\\]clipdraw_final\.png$")


def mentions(caption, word):
    """Whole-word mention of the target in Florence's caption (strict: misses synonyms)."""
    if not caption:
        return False
    return re.search(r"(?<![a-z])" + re.escape(word.lower()) + r"(?![a-z])", caption.lower()) is not None


def _num(value, cast=float):
    return None if value in (None, "", "None", "nan") else cast(float(value))


def build(combined_csv, strokes_root, rank_cutoff=1):
    with open(combined_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    by_word = defaultdict(list)
    for row in rows:
        m = BUDGET_PATH.search(row["image_path"])
        if not m:
            raise ValueError(f"Not a stroke-budget image: {row['image_path']}")
        n, part, folder = int(m.group(1)), m.group(2), m.group(3)
        caption = row.get("florence2_normalized_caption") or row.get("florence2_original_caption") or ""
        by_word[(part, folder)].append({
            "n": n,
            "siglip_rank": _num(row.get("siglip2_target_rank"), int),
            "siglip_top1": row.get("siglip2_top1_prediction") or None,
            "florence_score": 1.0 if mentions(caption, row["target_word"]) else 0.0,
            "caption": caption,
            "caption_mentions_writing": str(row.get("florence2_mentions_apparent_text")).lower() == "true",
            "text_score": _num(row.get("text_score")),
        })

    out = []
    for (part, folder), budgets in sorted(by_word.items()):
        record = json.loads((Path(strokes_root) / part / f"{folder}.json").read_text())
        budgets.sort(key=lambda b: b["n"])
        if len({b["n"] for b in budgets}) != len(budgets):
            raise ValueError(f"Duplicate budgets for {part}/{folder}.")
        out.append({
            "word": record["word"],
            "prompt": record["prompt"],
            "canvas": record["canvas"],
            "part": part,
            "folder": folder,
            "rank_cutoff": rank_cutoff,
            "num_candidates": 150,
            "florence_cutoff": None,          # shown, but not part of the threshold rule
            "data_label": "Real run: strokes pruned from a 256-stroke CLIPDraw drawing",
            "strokes": [{k: s[k] for k in ("points", "width", "color", "index") if k in s} for s in record["strokes"]],
            "budgets": budgets,
        })
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--combined", required=True)
    p.add_argument("--strokes", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--rank-cutoff", type=int, default=1)
    p.add_argument("--num-candidates", type=int, default=150)
    a = p.parse_args()
    concepts = build(a.combined, a.strokes, a.rank_cutoff)
    for c in concepts:
        c["num_candidates"] = a.num_candidates
    Path(a.out).write_text(json.dumps(concepts))
    for c in concepts:
        ranks = ", ".join(f"{b['n']}->#{b['siglip_rank']}" for b in c["budgets"])
        print(f"{c['word']}: {ranks}")
    print(f"Wrote {len(concepts)} concept(s) to {a.out}")


if __name__ == "__main__":
    main()
