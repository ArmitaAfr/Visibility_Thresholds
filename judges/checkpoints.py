"""Versioned, checksummed checkpoints. Source artifacts are read only."""

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, replace
from pathlib import Path

from .schema import EvaluationResult


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True,
                                     allow_nan=False, separators=(",", ":")).encode()).hexdigest()


def file_digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def atomic_write(path, value):
    """Flush a same-directory temporary file, then atomically replace the target."""
    payload = json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n"
    descriptor, temporary = tempfile.mkstemp(prefix=".tmp-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class CheckpointStore:
    def __init__(self, directory, configuration, *, resume):
        self.directory = Path(directory).resolve()
        configuration = json.loads(json.dumps(configuration, allow_nan=False))
        self.configuration_hash = digest(configuration)
        manifest = {"version": 1, "configuration": configuration,
                    "configuration_hash": self.configuration_hash}
        manifest_path = self.directory / "manifest.json"
        if resume:
            try:
                saved = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise ValueError(f"Cannot resume checkpoint manifest {manifest_path}: {exc}") from exc
            if saved != manifest:
                raise ValueError("Checkpoint configuration changed or manifest is damaged; use a new checkpoint directory.")
        else:
            # Existing directories require explicit resume; never reset old predictions.
            self.directory.mkdir(parents=True, exist_ok=False)
            atomic_write(manifest_path, manifest)

    def identity(self, experiment):
        identity = {"image_id": experiment.image_id, "image_sha256": file_digest(experiment.image_path),
                    "target_word": experiment.target_word, "configuration_hash": self.configuration_hash}
        if experiment.metadata_path.is_file():
            identity["metadata_sha256"] = file_digest(experiment.metadata_path)
        return identity

    def load(self, identity, experiment):
        path = self.directory / (digest(identity) + ".json")
        if not path.exists():
            return None
        try:
            envelope = json.loads(path.read_text(encoding="utf-8"))
            payload = envelope["payload"]
            if envelope["checksum"] != digest(payload) or payload["identity"] != identity:
                raise ValueError("Checksum or evaluation identity mismatch.")
            result = EvaluationResult(**payload["result"])
            if (result.image_id != experiment.image_id or result.target_word != experiment.target_word
                    or not isinstance(result.normalized_prediction, str)):
                raise ValueError("Invalid result identity/schema.")
            # Preserve original inference timing, but update paths after relocation.
            return replace(result, image_path=str(experiment.image_path), metadata_path=str(experiment.metadata_path))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise ValueError(f"Damaged checkpoint record {path}: {exc}") from exc

    def save(self, identity, experiment, result):
        if self.identity(experiment) != identity:
            raise ValueError(f"Source changed during inference: {experiment.image_path}")
        payload = {"identity": identity, "result": asdict(result)}
        atomic_write(self.directory / (digest(identity) + ".json"),
                     {"payload": payload, "checksum": digest(payload)})
