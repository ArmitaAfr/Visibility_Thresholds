"""Package local uncommitted judge source for manual upload; no remote writes."""

import argparse
import json
import subprocess
import zipfile
from pathlib import Path

from .checkpoints import file_digest


def prepare_bundle(output, repo=None):
    repo = Path(repo) if repo else Path(__file__).resolve().parents[1]
    files = [p for p in (repo / "judges").rglob("*")
             if p.is_file() and p.suffix in {".py", ".txt", ".md"} and "__pycache__" not in p.parts]
    files += [repo / "judges_colab.ipynb", repo / "dataset/processed/sampled_words.csv"]
    manifest = {str(p.relative_to(repo)).replace("\\", "/"): file_digest(p) for p in sorted(files)}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in manifest:
            archive.write(repo / name, name)
        archive.writestr("source_manifest.json", json.dumps(manifest, sort_keys=True, indent=2))
    return output


def prepare_unified(output, base_commit):
    """Pin the actual working-copy bytes as well as their Git baseline."""
    repo = Path(__file__).resolve().parents[1]
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    if head != base_commit:
        raise ValueError(f"Expected full base commit {base_commit}, found {head}.")
    tracked = subprocess.check_output(["git", "ls-files"], cwd=repo, text=True).splitlines()
    names = {name for name in tracked if "tests" not in Path(name).parts and
             (name.startswith(("batch/", "judges/")) and Path(name).suffix in {".py", ".txt", ".md"}
              or name.startswith("dataset/processed/") and name.endswith(".csv"))}
    names.update({"clipdraw.ipynb", "judges_colab.ipynb", "visibility_thresholds_colab.ipynb",
                  "THIRD_PARTY_NOTICES.md", "dataset/validate_dataset.py", "dataset/README.md",
                  "dataset/VALIDATION.md", "judges/experiment_manifest.py", "judges/analysis/combined.py", "unified_colab.py",
                  "text_score/run_manifest.py", "text_score/text_score.py", "text_score/MANIFEST.md",
                  "text_score/FEATURE.md", "text_score/requirements-colab.in"})
    files = {name: (repo / name).read_bytes() for name in sorted(names)}
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in files.items():
            archive.writestr(name, content)
        archive.writestr("base_commit.txt", head)
    print(f"Base {head}; archive SHA-256 {file_digest(output)}")
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description="Bundle judge source/dataset for Colab without a commit or push.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--unified-base", help="Build a unified working-copy bundle at this full HEAD SHA.")
    args = parser.parse_args(argv)
    print(prepare_unified(args.output, args.unified_base) if args.unified_base else prepare_bundle(args.output))


if __name__ == "__main__":
    main()
