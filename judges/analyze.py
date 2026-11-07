"""python -m judges.analyze: compare saved exports without inference."""

import argparse
from pathlib import Path

from .analysis import analyze_exports, write_report


def main(argv=None):
    parser = argparse.ArgumentParser(description="Compare saved SigLIP2 and Florence-2 JSON exports.")
    parser.add_argument("--siglip2", nargs="+", type=Path, required=True)
    parser.add_argument("--florence2", nargs="+", type=Path, required=True)
    parser.add_argument("--source-root", nargs="+", type=Path,
                        help="Optional relocated batch roots, verified using stable image IDs.")
    parser.add_argument("--allow-unmatched", action="store_true",
                        help="Analyze the ID intersection and explicitly report omitted records.")
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory; existing reports are never overwritten.")
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        parser.error("--output-dir already exists; choose a new directory.")
    rows, summary = analyze_exports(args.siglip2, args.florence2,
                                   source_roots=args.source_root, allow_unmatched=args.allow_unmatched)
    write_report(rows, summary, args.output_dir)
    print(f"Compared {len(rows)} verified images; vocabulary={summary['siglip2']['vocabulary_size']}. Report: {args.output_dir}")


if __name__ == "__main__":
    main()
