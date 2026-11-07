"""Package local uncommitted judge source for manual upload; no remote writes."""

import argparse
import json
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


def main(argv=None):
    parser = argparse.ArgumentParser(description="Bundle judge source/dataset for Colab without a commit or push.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    print(prepare_bundle(args.output))


if __name__ == "__main__":
    main()
