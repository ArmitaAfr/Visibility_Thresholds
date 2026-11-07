"""Read-only discovery of batch artifacts, with explicit validation reports."""

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from .schema import Experiment


@dataclass(frozen=True)
class DiscoveryIssue:
    code: str
    path: str
    message: str


@dataclass
class DiscoveryReport:
    experiments: list[Experiment] = field(default_factory=list)
    issues: list[DiscoveryIssue] = field(default_factory=list)


class DiscoveryError(ValueError):
    def __init__(self, report: DiscoveryReport):
        self.report = report
        super().__init__("\n".join(f"{i.code}: {i.path}: {i.message}" for i in report.issues))


def discover_experiments(
    roots: str | Path | Iterable[str | Path], *, strict: bool = True
) -> DiscoveryReport:
    """Recursively inspect one or more part folders or their common parent.

    Completion requires experiment.json with a nonblank word and a nonempty
    sibling clipdraw_final.png. No labels are inferred from folder names.
    IDs hash canonical metadata plus PNG bytes, remaining stable after moving
    a tree. Identical copies are duplicates; different metadata is a separate
    experiment even when its target word or pixels are the same.
    """
    if isinstance(roots, (str, Path)):
        roots = [roots]
    report = DiscoveryReport()
    seen_paths, seen_ids = set(), {}

    def issue(code, path, message):
        report.issues.append(DiscoveryIssue(code, str(path), message))

    for root in roots:
        root = Path(root).resolve()
        if not root.is_dir():
            issue("invalid_root", root, "Expected an existing output directory.")
            continue
        metadata_paths = set(root.rglob("experiment.json"))
        for image in sorted(root.rglob("clipdraw_final.png")):
            if image.with_name("experiment.json") not in metadata_paths:
                issue("missing_metadata", image, "No completion marker; image excluded.")
        for path in sorted(metadata_paths):
            path = path.resolve()
            if path in seen_paths:
                issue("duplicate_record", path, "Already encountered through another root.")
                continue
            seen_paths.add(path)
            try:
                metadata = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(metadata, dict):
                    raise ValueError("Metadata must be a JSON object.")
                word = metadata.get("word")
                if not isinstance(word, str) or not word.strip():
                    raise ValueError("Metadata must contain a nonblank string 'word'.")
                canonical = json.dumps(metadata, sort_keys=True, ensure_ascii=True, allow_nan=False)
            except (OSError, UnicodeError, ValueError) as exc:
                issue("malformed_metadata", path, str(exc))
                continue
            image = path.with_name("clipdraw_final.png")
            try:
                content = image.read_bytes()
                if not content:
                    raise ValueError("Image is empty.")
            except (OSError, ValueError) as exc:
                issue("missing_image", image, str(exc))
                continue
            image_id = hashlib.sha256(canonical.encode() + b"\x00" + content).hexdigest()
            if image_id in seen_ids:
                issue("duplicate_record", path, f"Identical metadata and image: {seen_ids[image_id]}")
                continue
            seen_ids[image_id] = path
            report.experiments.append(Experiment(image_id, word.strip(), image, path))
    if not report.experiments and not report.issues:
        issue("no_experiments", "", "No experiment.json records found in supplied roots.")
    if strict and report.issues:
        raise DiscoveryError(report)
    return report
