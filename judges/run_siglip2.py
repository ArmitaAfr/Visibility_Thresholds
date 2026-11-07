"""Run from the repository root: python -m judges.run_siglip2 --help."""

import argparse
from pathlib import Path

from . import discover_experiments, evaluate, export_results, load_candidates
from .siglip2 import DEFAULT_MODEL, DEFAULT_PROMPT, SigLIP2Judge
from .vocabulary import DEFAULT_WORDS_SOURCE


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Expected a positive integer.")
    return number


def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate completed CLIPDraw images using a fixed SigLIP2 vocabulary.")
    parser.add_argument("roots", nargs="+", type=Path, help="Part folders, or a parent containing only intended runs.")
    parser.add_argument("--words", type=Path, default=DEFAULT_WORDS_SOURCE)
    parser.add_argument("--output", type=Path, required=True, help="New .csv or .json file; existing files are never overwritten.")
    parser.add_argument("--limit", type=positive_int, default=None)
    parser.add_argument("--batch-size", type=positive_int, default=4)
    parser.add_argument("--text-batch-size", type=positive_int, default=32)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default="main", help="Use a Hugging Face commit hash for repeatable runs.")
    parser.add_argument("--cache-dir", type=Path, help="Optional Hugging Face Hub cache directory.")
    parser.add_argument("--local-files-only", action="store_true", help="Resolve and load exclusively from the local Hub cache.")
    parser.add_argument("--prompt-template", default=DEFAULT_PROMPT)
    parser.add_argument("--checkpoint", type=Path, help="Checkpoint directory (default: OUTPUT.checkpoint).")
    parser.add_argument("--resume", action="store_true", help="Reuse matching completed evaluations from an existing checkpoint.")
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.suffix.lower() not in {".csv", ".json"}:
        parser.error("--output must be a new .csv or .json file.")
    checkpoint = args.checkpoint or Path(str(args.output) + ".checkpoint")
    if not args.resume and checkpoint.exists():
        parser.error("Checkpoint already exists; use --resume or a new --checkpoint directory.")
    if args.resume and not (checkpoint / "manifest.json").is_file():
        parser.error("--resume requires an existing checkpoint manifest.")
    report = discover_experiments(args.roots)
    experiments = report.experiments[:args.limit]
    candidates = load_candidates(args.words)
    if any(e.target_word.lower() not in {w.lower() for w in candidates} for e in experiments):
        parser.error("An experiment target is absent from the candidate vocabulary.")
    judge = SigLIP2Judge(candidates, model_id=args.model, revision=args.revision,
                        device=args.device, prompt_template=args.prompt_template,
                        text_batch_size=args.text_batch_size, cache_dir=args.cache_dir,
                        local_files_only=args.local_files_only)
    results = evaluate(experiments, candidates, judge, batch_size=args.batch_size,
                       checkpoint_dir=checkpoint, resume=args.resume)
    export_results(results, args.output)
    print(f"Saved {len(results)} results to {args.output} using {judge.model_name}")


if __name__ == "__main__":
    main()
