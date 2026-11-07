"""Standalone captioning; no CLIPDraw notebook globals or SigLIP2 predictions."""

import argparse
from pathlib import Path

from . import discover_experiments, evaluate, export_results, load_candidates
from .florence2 import DEFAULT_MODEL, Florence2Judge
from .run_siglip2 import positive_int
from .vocabulary import DEFAULT_WORDS_SOURCE


def main(argv=None):
    parser = argparse.ArgumentParser(description="Independent Florence-2 <CAPTION> evaluation.")
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=positive_int)
    parser.add_argument("--batch-size", type=positive_int, default=1)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--dtype", choices=["auto", "float32", "float16"], default="auto")
    parser.add_argument("--max-new-tokens", type=positive_int, default=128)
    parser.add_argument("--num-beams", type=positive_int, default=3)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--words", type=Path, default=DEFAULT_WORDS_SOURCE,
                        help="For shared pipeline target validation only; never passed to the model.")
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.suffix.lower() not in {".csv", ".json"}:
        parser.error("--output must be a new .csv or .json file.")
    checkpoint = args.checkpoint or Path(str(args.output) + ".checkpoint")
    if checkpoint.exists() and not args.resume:
        parser.error("Checkpoint already exists; use --resume or a new directory.")
    if args.resume and not (checkpoint / "manifest.json").is_file():
        parser.error("--resume requires an existing checkpoint manifest.")
    experiments = discover_experiments(args.roots).experiments[:args.limit]
    candidates = load_candidates(args.words)
    if any(e.target_word.lower() not in {w.lower() for w in candidates} for e in experiments):
        parser.error("Experiment target absent from the validation vocabulary.")
    judge = Florence2Judge(revision=args.revision, device=args.device, dtype=args.dtype,
                          max_new_tokens=args.max_new_tokens, num_beams=args.num_beams,
                          cache_dir=args.cache_dir, local_files_only=args.local_files_only)
    results = evaluate(experiments, candidates, judge, batch_size=args.batch_size,
                       checkpoint_dir=checkpoint, resume=args.resume)
    export_results(results, args.output)
    print(f"Saved {len(results)} captions to {args.output} using {judge.model_name}")


if __name__ == "__main__":
    main()
