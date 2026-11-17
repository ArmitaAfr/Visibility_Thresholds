"""Dataset preflight and image manifests. All input paths use the local/mounted filesystem."""

import hashlib
import io
import json
import os
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from .analysis.imageability import canonical_word, load_imageability
from .checkpoints import file_digest
from .discovery import canonical_image_id, discover_experiments
from .vocabulary import load_candidates


@dataclass(frozen=True)
class ExperimentConfig:
    cohort_id: str
    generation_csv: Path
    vocabulary_csv: Path
    input_roots: tuple[Path, ...]
    output_root: Path
    imageability_csv: Path | None = None
    person: int = 1
    num_people: int = 1

    def __post_init__(self):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", self.cohort_id):
            raise ValueError("Use letters, digits, underscores or hyphens for cohort_id.")
        if (type(self.person) is not int or type(self.num_people) is not int
                or not 1 <= self.person <= self.num_people):
            raise ValueError("Expected 1 <= person <= num_people.")
        if isinstance(self.input_roots, (str, Path)) or not self.input_roots:
            raise ValueError("Supply a nonempty sequence of image input roots.")
        for name in ("generation_csv", "vocabulary_csv", "output_root", "imageability_csv"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, Path(value).expanduser().resolve())
        roots = tuple(sorted(Path(p).expanduser().resolve() for p in self.input_roots))
        object.__setattr__(self, "input_roots", roots)

    def to_dict(self):
        values = asdict(self)
        values["input_roots"] = list(map(str, self.input_roots))
        return {k: str(v) if isinstance(v, Path) else v for k, v in values.items()}

    def validate(self, *, planned_inputs=False):
        """Return assigned generation words, ordered candidates and indexed metadata."""
        for i, root in enumerate(self.input_roots):
            if not root.is_dir() and not (planned_inputs and not root.exists()):
                raise ValueError(f"Missing image input directory: {root}")
            if any(root.is_relative_to(other) or other.is_relative_to(root)
                   for other in self.input_roots[:i]):
                raise ValueError(f"Overlapping image input roots: {root}")
            if self.output_root.is_relative_to(root):
                raise ValueError(f"Output root lies inside image inputs: {root}")
        output_parent = self.output_root
        while not output_parent.exists():
            output_parent = output_parent.parent
        if not output_parent.is_dir() or not os.access(output_parent, os.W_OK):
            raise ValueError(f"Output directory unavailable: {output_parent}")
        assigned = load_candidates(self.generation_csv)[self.person - 1::self.num_people]
        if not assigned:
            raise ValueError("Empty generation assignment.")
        candidates = load_candidates(self.vocabulary_csv)
        targets = set(map(canonical_word, assigned))
        missing = targets.difference(map(canonical_word, candidates))
        if missing:
            raise ValueError(f"Generation targets missing from judge vocabulary: {sorted(missing)}")
        metadata = {}
        if self.imageability_csv is not None:
            metadata, _ = load_imageability(self.imageability_csv)
            if missing := targets.difference(metadata):
                raise ValueError(f"Generation targets missing from imageability CSV: {sorted(missing)}")
        return assigned, candidates, metadata


def build_manifest(config: ExperimentConfig):
    """Return a JSON-compatible snapshot. Inspect issues before evaluating a partial cohort."""
    from PIL import Image

    _, candidates, metadata = config.validate()
    vocabulary = set(map(canonical_word, candidates))
    discovery = discover_experiments(config.input_roots, strict=False)
    issues = [asdict(issue) for issue in discovery.issues]
    rows = []
    targets_by_hash = defaultdict(set)

    def issue(code, path, message):
        issues.append(dict(code=code, path=str(path), message=message))

    for root in config.input_roots:
        for folder in (root, *root.rglob("*")):
            if (folder.is_dir() and re.match(r"^\d{3,}_", folder.name)
                    and not (folder / "experiment.json").exists()
                    and not (folder / "clipdraw_final.png").exists()):
                issue("incomplete_folder", folder, "Missing both experiment.json and clipdraw_final.png.")

    for record in discovery.experiments:
        image = record.image_path
        try:
            content = image.read_bytes()
            info = json.loads(record.metadata_path.read_text(encoding="utf-8"))
            canonical = json.dumps(info, sort_keys=True, ensure_ascii=True, allow_nan=False)
            if canonical_image_id(canonical, content) != record.image_id:
                issue("source_changed", image, "Image or metadata changed during discovery.")
                continue
            with Image.open(io.BytesIO(content), formats=["PNG"]) as png:
                png.verify()
            with Image.open(io.BytesIO(content), formats=["PNG"]) as png:
                png.load()
        except (ValueError, OSError, SyntaxError, Image.DecompressionBombError) as exc:
            issue("invalid_source", image, str(exc))
            continue
        word = canonical_word(record.target_word)
        image_hash = hashlib.sha256(content).hexdigest()
        targets_by_hash[image_hash].add(word)
        if word not in vocabulary:
            issue("target_outside_vocabulary", image, record.target_word)
            continue
        if config.imageability_csv and word not in metadata:
            issue("missing_imageability", image, record.target_word)
            continue
        rows.append(dict(image_id=record.image_id, image_sha256=image_hash,
                         target_word=record.target_word, image_path=str(image),
                         metadata_path=str(record.metadata_path), cohort_id=config.cohort_id,
                         part=image.parent.parent.name, folder=image.parent.name,
                         imageability=metadata.get(word)))

    verified = []
    for row in rows:
        targets = targets_by_hash[row["image_sha256"]]
        if len(targets) > 1:
            issue("conflicting_targets", row["image_path"], f"Identical PNG has targets {sorted(targets)}.")
        else:
            verified.append(row)
    sources = {name: file_digest(path) for name, path in (
        ("generation_csv", config.generation_csv), ("vocabulary_csv", config.vocabulary_csv),
        ("imageability_csv", config.imageability_csv)) if path is not None}
    return dict(schema_version=1, config=config.to_dict(), csv_sha256=sources,
                images=sorted(verified, key=lambda row: row["image_id"]),
                issues=sorted(issues, key=lambda item: (item["path"], item["code"], item["message"])))


def save_manifest(config: ExperimentConfig) -> Path:
    """Build and exclusively create output_root/cohort_id/manifest.json."""
    manifest = build_manifest(config)
    path = config.output_root / config.cohort_id / "manifest.json"
    if (path.resolve() != path or any(path.is_relative_to(root) for root in config.input_roots)):
        raise ValueError(f"Unsafe manifest destination: {path}")
    payload = json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        try:
            handle.write(payload)
            handle.flush()
        except OSError:
            handle.close()
            path.unlink()
            raise
    return path
